from typing import Tuple

import torch
from Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = setup_logging()


def load_model_and_tokenizer(
    model_config: dict, device: int
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer with safe device handling and optional 4-bit."""
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"
    quant_config = None
    attn_impl = None
    # Select best dtype
    compute_dtype = torch.float32
    if use_cuda and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif use_cuda:
        compute_dtype = torch.float16

    if use_cuda:

        cuda_count = torch.cuda.device_count()
        idx = int(device) if isinstance(device, int) else 0
        if idx < 0 or idx >= cuda_count:
            logger.warning(
                f"Requested CUDA device {device} invalid; using 0 (num_gpus={cuda_count})."
            )
            idx = 0
        try:
            torch.cuda.set_device(idx)
        except Exception as e:
            logger.warning(
                f"torch.cuda.set_device({idx}) failed: {e}. Falling back to 0."
            )
            idx = 0
            torch.cuda.set_device(idx)
        device_str = f"cuda:{idx}"

        # Prefer FlashAttention-2 if available, else SDPA
        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401

            major, minor = torch.cuda.get_device_capability(idx)
            if (major * 10 + minor) >= 75:
                attn_impl = "flash_attention_2"
                logger.info("Using FlashAttention-2.")
            else:
                logger.info("FlashAttention-2 unsupported on this GPU; using SDPA.")
        except Exception:
            logger.info("flash-attn not available; using SDPA.")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_config["quantization"]["load_in_4bit"]),
            bnb_4bit_use_double_quant=bool(
                model_config["quantization"]["bnb_4bit_use_double_quant"]
            ),
            bnb_4bit_quant_type=str(
                model_config["quantization"]["bnb_4bit_quant_type"]
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        logger.info(f"Loading model on {device_str} with 4-bit quantization.")
    else:
        logger.warning(
            "CUDA not available; loading model on CPU without 4-bit quantization."
        )
        device_str = "cpu"
        quant_config = BitsAndBytesConfig(load_in_4bit=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True
    )
    # Always left-pad decoder-only models; keep most recent tokens if truncation occurs.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if attn_impl == "flash_attention_2":
        logger.info(
            "Using FlashAttention-2; keeping padding_side='left' for decoder-only models."
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
        torch_dtype=compute_dtype if use_cuda else torch.float32,
        device_map=device_str,
        attn_implementation=attn_impl,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    # Ensure KV cache usage for faster generation
    try:
        model.config.use_cache = True
    except Exception:
        pass

    model = PeftModel.from_pretrained(
        model, model_config["cot_adapter_path"], device_map=device_str
    )

    # Optional: compile for extra speed when supported
    if bool(model_config.get("compile", False)):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed; continuing without it. Err: {e}")

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if tokenizer.encode(tokenizer.eos_token)[0] != model.config.eos_token_id:
        logger.warning(
            "Tokenizer and model EOS token IDs do not match! This may be the problem!!!."
        )


    # debugging tests
    logger.warning('######## DEBUGGING TESTS ########')
    logger.warning(f"TEMPLATE CHECK : {hasattr(tokenizer, 'apply_chat_template')}")
    logger.warning(f"Tokenizer vocab size: {len(tokenizer)}")
    logger.warning(f"Model vocab size: {model.config.vocab_size}")
    logger.warning(tokenizer.special_tokens_map)
    logger.warning(tokenizer.all_special_tokens)
    logger.warning(tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens))


    inputs = tokenizer("Hello, this is a test.", return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    # Grab hidden states of the last step
    last_hidden_states = outputs.hidden_states[-1][0]  # [batch, seq, hidden_dim]

    logger.warning(f"Hidden states shape:{last_hidden_states.shape}" )
    logger.warning(f"Norms of all tokens:{last_hidden_states.norm(dim=-1)} ")
    logger.warning(f"Sample generation: {tokenizer.decode(outputs.sequences[0])}")
    logger.warning('######## END DEBUGGING TESTS ########')

    #stop script here for debugging
    import sys; sys.exit(0)

    return model, tokenizer
