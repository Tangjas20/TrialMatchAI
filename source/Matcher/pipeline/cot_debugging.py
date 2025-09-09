import sys
import os
import json
import torch
from typing import List, Dict
from .cot_reasoning import BatchTrialProcessor
from Matcher.utils.file_utils import read_json_file, write_json_file, write_text_file, create_directory
from Matcher.utils.logging_config import setup_logging
from Matcher.models.llm.llm_loader import load_model_and_tokenizer
from Matcher.config.config_loader import load_config

logger = setup_logging()

def debug_cot_reasoning(
    config: dict,
    nct_ids: List[str],
    patient_profile: List[str],
):
    """Debug CoT reasoning with a small set of trials."""

    logger.warning(f"Model CACHE : {os.environ['TRANSFORMERS_CACHE']}")  # Check if the environment variable is set correctly

    # Construct patient profile
    patient_profile = (
        patient_info.get("split_raw_description")
        or patient_info.get("expanded_sentences")
        or (
            [patient_info.get("raw_description")]
            if patient_info.get("raw_description")
            else []
        )
    )
    if not patient_profile:
        logger.error("No valid patient profile found for debugging.")
        return
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config["model"], config["global"]["device"])
    logger.info("Model and tokenizer loaded successfully.")

    """ # if last two don't match, set config to tokenizer
    if model.config.eos_token_id != tokenizer.eos_token_id:
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    if model.config.pad_token_id != tokenizer.pad_token_id:
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Config EOS: {model.config.eos_token_id}")
    logger.info(f"Config PAD: {model.config.pad_token_id}") """

    model.eval()
    print(f"Model training mode: {model.training}")
    for name, module in model.named_modules():
        if hasattr(module, 'training') and module.training:
            print(f"Module {name} still in training mode!")

    print(f"Gradients enabled: {torch.is_grad_enabled()}")
    print(f"Model dtype: {model.dtype}") #should be bfloat16
    print(f"Model device: {model.device}")

    # Initialize BatchTrialProcessor
    cot_processor = BatchTrialProcessor(
        model=model,
        tokenizer=tokenizer,
        device=config["global"]["device"],
        batch_size=config["cot"]["batch_size"],
        use_cot=True,
    )

    # Ensure output directory exists
    output_folder = config["paths"]["output_dir"]
    create_directory(output_folder)

    # Process trials
    logger.info("Processing trials with CoT reasoning...")
    cot_processor.process_trials(
        nct_ids=nct_ids,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )
    logger.info("CoT reasoning completed.")

if __name__ == "__main__":
    # Debugging configuration
    config = load_config(config_path='Matcher/config/config.json')

    top_trials_path = "../results/TREC21/trec-20211/fl-hybrid_rerank_cot_ner/top_trials.txt"
    # Example patient information
    patient_info = {
        "split_raw_description": [
            "Patient is a 45-year-old male with a history of hypertension and diabetes."
        ],
        "expanded_sentences": [],
        "raw_description": "45-year-old male with hypertension and diabetes.",
    }

        # Read trial IDs from the file
    if os.path.exists(top_trials_path):
        with open(top_trials_path, "r") as f:
            nct_ids = [line.strip() for line in f if line.strip()]
    else:
        logger.error(f"Top trials file not found: {top_trials_path}")
        nct_ids = []

    logger.warning(f"Debugging CoT reasoning with {len(nct_ids)} trials.")

    # Run debugging script
    debug_cot_reasoning(
        config=config,
        nct_ids=nct_ids,
        patient_profile=patient_info,
    )