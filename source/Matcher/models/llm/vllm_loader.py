# Matcher/models/llm/vllm_loader.py
from __future__ import annotations

import os
from typing import Optional, Tuple

from Matcher.utils.logging_config import setup_logging

logger = setup_logging()


def _infer_max_ctx_len(model_path: str) -> Optional[int]:
    """
    Try to infer the model's max context length from its HF/transformers config.
    """
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        for k in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "sliding_window",
            "model_max_length",
        ):
            v = getattr(cfg, k, None)
            if isinstance(v, int) and v > 0:
                return v
    except Exception as e:
        logger.warning(f"[vLLM] Could not infer max context length: {e}")
    return None


def _as_str(x, name: str) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (str, os.PathLike)):
        return str(x)
    # Hard guard against floats accidentally being passed as paths (e.g., 1.0)
    if isinstance(x, float):
        raise TypeError(f"{name} must be a string path/repo id, not float: {x!r}")
    return str(x)


def _as_int(x, name: str) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"{name} must be int-compatible, got {type(x)}: {x!r}") from e


def _as_float(x, name: str) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"{name} must be float-compatible, got {type(x)}: {x!r}") from e


def load_vllm_engine(
    model_config: dict, vllm_cfg: dict | None = None
) -> Tuple[object, Optional[object], Optional[object]]:
    """
    Build a vLLM LLM engine safely with strict type guards to avoid:
      - Tokenizer/path errors like "expected str, got float (1.0)"
      - Passing unexpected kwargs to LLM(...)
    Returns: (engine, tokenizer, lora_request)
    """
    # Lazy imports to avoid import costs when not used
    from vllm import LLM  # type: ignore
    from vllm.lora.request import LoRARequest  # type: ignore

    vllm_cfg = dict(vllm_cfg or {})

    # Resolve model path / repo id
    model_path = (
        vllm_cfg.get("model_path")
        or model_config.get("base_model")
        or model_config.get("model_path")
        or model_config.get("name_or_path")
    )
    model_path = _as_str(model_path, "model_path")
    if not model_path:
        raise ValueError("vLLM model_path could not be resolved to a non-empty string.")

    # Sanitize core engine options
    dtype = _as_str(vllm_cfg.get("dtype", "bfloat16"), "dtype") or "bfloat16"
    tp = _as_int(vllm_cfg.get("tensor_parallel_size", 1), "tensor_parallel_size") or 1
    gmu = (
        _as_float(
            vllm_cfg.get("gpu_memory_utilization", 0.95), "gpu_memory_utilization"
        )
        or 0.95
    )
    max_lora_rank = _as_int(vllm_cfg.get("max_lora_rank", 64), "max_lora_rank") or 64

    # Optional max_model_len with safe cap based on HF config (unless overridden)
    requested_len = _as_int(vllm_cfg.get("max_model_len", None), "max_model_len")
    allow_long = os.getenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN") == "1"
    if requested_len is not None and not allow_long:
        derived = _infer_max_ctx_len(model_path)
        if derived and requested_len > derived:
            logger.warning(
                f"[vLLM] Requested max_model_len={requested_len} > model limit {derived}; "
                f"capping to {derived}. Set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 to override."
            )
            requested_len = derived

    # Build engine kwargs explicitly; never unpack untrusted dicts
    engine_kwargs = dict(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gmu,
        enable_lora=True,  # allows dynamic LoRA loading
        max_lora_rank=max_lora_rank,
        # NOTE: trust_remote_code is controlled inside transformers when vLLM loads; passing it
        # here may be ignored depending on vLLM version, so we avoid it to prevent confusion.
    )
    if requested_len is not None:
        engine_kwargs["max_model_len"] = requested_len

    logger.info(
        "[vLLM] Initializing engine: model=%s dtype=%s tp=%s max_len=%s gmu=%.2f",
        model_path,
        dtype,
        tp,
        engine_kwargs.get("max_model_len", "model_default"),
        gmu,
    )

    # Create engine
    engine = LLM(**engine_kwargs)  # type: ignore

    # Obtain tokenizer (for chat templates, stopwords, etc.)
    tokenizer = None
    try:
        get_tok = getattr(engine, "get_tokenizer", None)
        if callable(get_tok):
            tokenizer = get_tok()
        elif hasattr(engine, "tokenizer"):
            tokenizer = getattr(engine, "tokenizer")
    except Exception as e:
        logger.warning(f"[vLLM] Unable to fetch tokenizer from engine: {e}")

    # Optional LoRA (single adapter). We preload if API is available; we still return LoRARequest
    # so the caller can use it in generate() calls if needed.
    adapter_path = vllm_cfg.get("adapter_path") or model_config.get("cot_adapter_path")
    lora_request = None
    if adapter_path:
        adapter_path = _as_str(adapter_path, "adapter_path")
        if not adapter_path:
            logger.warning("[vLLM] adapter_path was provided but empty; skipping LoRA.")
        else:
            name = (
                _as_str(vllm_cfg.get("adapter_name", "cot_adapter"), "adapter_name")
                or "cot_adapter"
            )
            weight = (
                _as_float(vllm_cfg.get("adapter_weight", 1.0), "adapter_weight") or 1.0
            )
            try:
                lora_request = LoRARequest(name, adapter_path, weight)  # type: ignore
            except TypeError:
                # Fallback other vLLM signatures (some versions don't take weight)
                lora_request = LoRARequest(name, adapter_path)  # type: ignore
            logger.info(
                "[vLLM] Using LoRA adapter: name=%s path=%s weight=%.3f",
                name,
                adapter_path,
                float(weight),
            )
            try:
                add_lora = getattr(engine, "add_lora", None)
                if callable(add_lora):
                    # Some versions accept (name, path) only
                    try:
                        add_lora(name, adapter_path)
                    except TypeError:
                        # Others accept dict or kwargs; attempt a generic call
                        add_lora(name=name, path=adapter_path)
                    logger.info("[vLLM] Preloaded LoRA adapter.")
            except Exception as e:
                logger.warning(f"[vLLM] Preload adapter failed (safe to ignore): {e}")

    return engine, tokenizer, lora_request
