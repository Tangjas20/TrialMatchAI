from typing import Tuple

import torch
from Matcher.utils.logging_config import setup_logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = setup_logging()


def load_model_and_tokenizer(
    model_config: dict, device: int
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer with 4-bit quantization."""
    torch.cuda.set_device(device)
    logger.info(f"Loading model on GPU {device}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=model_config["quantization"]["load_in_4bit"],
        bnb_4bit_use_double_quant=model_config["quantization"][
            "bnb_4bit_use_double_quant"
        ],
        bnb_4bit_quant_type=model_config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model"],
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=f"cuda:{device}",
        attn_implementation="flash_attention_2",
        quantization_config=quant_config,
    )
    model = PeftModel.from_pretrained(model, model_config["fine_tuned_adapter_phi"])
    model.eval()
    logger.info(f"Model loaded on GPU {device}.")
    return model, tokenizer
