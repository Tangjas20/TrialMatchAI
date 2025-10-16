from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import torch
from Parser.biomedner_engine import BioMedNER

from elasticsearch import Elasticsearch

from .config.config_loader import load_config
from .models.embedding.query_embedder import QueryEmbedder
from .models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from .models.llm.llm_loader import load_model_and_tokenizer
from .models.llm.vllm_loader import load_vllm_engine
from .models.llm.llm_reranker import LLMReranker
from .pipeline.cot_reasoning import BatchTrialProcessor
from .pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM
from .pipeline.phenopacket_processor import process_phenopacket
from .pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from .pipeline.trial_search.first_level_search import ClinicalTrialSearch
from .pipeline.trial_search.second_level_search import SecondStageRetriever
from .services.biomedner_service import initialize_biomedner_services
from .utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from .utils.logging_config import setup_logging

# vLLM environment variables (only used if vLLM is enabled)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_DISABLE_ASYNC_OUTPUT_PROC"] = "1"

logger = setup_logging()


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: QueryEmbedder,
    config: Dict,
    es_client: Elasticsearch,
    restrict_nct_ids: Optional[List[str]] = None,
) -> Optional[tuple]:
    """Run first-level search with optional NCT ID restriction."""
    main_conditions = keywords.get("main_conditions", [])
    other_conditions = keywords.get("other_conditions", [])
    expanded_sentences = keywords.get("expanded_sentences", [])

    if not main_conditions:
        logger.error("No main_conditions found in keywords.")
        return None

    condition = main_conditions[0]
    age = patient_info.get("age", "all")
    sex = patient_info.get("gender", "all")
    overall_status = "All"

    index_name = config["elasticsearch"]["index_trials"]
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)

    # Get synonyms and expand main conditions
    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[:5])

    # Use max_trials_first_level from default_ablation_config if available
    search_size = (
        config.get("default_ablation_config", {}).get("max_trials_first_level") or
        config.get("search", {}).get("max_trials_first_level", 300)
    )
    
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=search_size,
        pre_selected_nct_ids=restrict_nct_ids,
        synonyms=main_conditions,
        other_conditions=other_conditions,
        vector_score_threshold=config["search"]["vector_score_threshold"],
    )

    nct_ids = [trial.get("nct_id") for trial in trials if trial.get("nct_id")]
    first_level_scores = {
        trial.get("nct_id"): score
        for trial, score in zip(trials, scores)
        if trial.get("nct_id")
    }

    write_text_file([str(nid) for nid in nct_ids], f"{output_folder}/nct_ids.txt")
    write_json_file(first_level_scores, f"{output_folder}/first_level_scores.json")

    logger.info(f"First-level search complete: {len(nct_ids)} trial IDs saved.")
    return (
        nct_ids,
        main_conditions,
        other_conditions,
        expanded_sentences,
        first_level_scores,
    )


def run_second_level_search(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    expanded_sentences: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
    config: Dict,
) -> tuple:
    """Run second-level retrieval and ranking."""
    queries = list(set(main_conditions + other_conditions + expanded_sentences))[:10]
    logger.info(f"Running second-level retrieval with {len(queries)} queries...")

    # Add synonyms for second level
    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    # Use max_trials_second_level from default_ablation_config if available
    max_trials = (
        config.get("default_ablation_config", {}).get("max_trials_second_level") or
        config.get("search", {}).get("max_trials_second_level", 100)
    )
    top_n = min(len(nct_ids), max_trials)
    
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=top_n
    )

    combined_scores = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = trial["score"]
        first_score = first_level_scores.get(trial_id, 0)
        combined_scores[trial_id] = first_score + second_score

    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, min(len(sorted_trials) // 3, top_n))
    semi_final_trials = sorted_trials[:num_top]

    top_trials_path = f"{output_folder}/top_trials.txt"
    write_text_file([trial_id for trial_id, _ in semi_final_trials], top_trials_path)

    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


def run_cot_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    model,
    tokenizer,
    config: Dict,
    use_vllm: bool = False,
    vllm_engine=None,
    vllm_lora_request=None,
):
    """Run CoT processing with either HuggingFace or vLLM backend."""
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for CoT processing.")
        return

    # Use max_trials_cot from default_ablation_config if available, else from cot section
    max_cot_trials = (
        config.get("default_ablation_config", {}).get("max_trials_cot") or
        config["cot"].get("max_trials_cot", 20)
    )
    top_trials = top_trials[:max_cot_trials]
    
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for CoT processing.")
        return

    patient_location_dict = patient_info.get("location", {})

    if use_vllm:
        if vllm_engine is None:
            raise RuntimeError("vLLM engine not initialized but use_vllm=True")
        
        logger.info("Using vLLM backend for CoT processing")
        vllm_config = config.get("vllm", {})
        
        cot_processor = BatchTrialProcessorVLLM(
            llm=vllm_engine,
            tokenizer=tokenizer,
            batch_size=vllm_config.get("batch_size", config["cot"].get("batch_size", 4)),  # Fallback to cot.batch_size
            use_cot=True,
            max_new_tokens=config["cot"].get("max_new_tokens", 5000),
            temperature=vllm_config.get("temperature", 0.0),
            top_p=vllm_config.get("top_p", 1.0),
            seed=vllm_config.get("seed", 1234),
            length_bucket=vllm_config.get("length_bucket", True),
            lora_request=vllm_lora_request,
            use_geographic_reasoning=config["cot"].get("use_geographic_reasoning", False),
            patient_location=patient_location_dict,
            trial_locations_cache=config["cot"].get("trial_locations_cache_dir"),
            max_json_retries=vllm_config.get("max_json_retries", 1),
        )
    else:
        logger.info("Using HuggingFace backend for CoT processing")
        batch_size = min(config["cot"]["batch_size"] * 2, 8)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cot_processor = BatchTrialProcessor(
            model,
            tokenizer,
            device=config["global"]["device"],
            batch_size=batch_size,
            use_cot=True,
        )

    cot_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
        patient_location_dict=patient_location_dict,
    )

    write_json_file({"status": "done"}, f"{output_folder}/cot_output.json")
    logger.info(f"CoT-based trial matching complete (backend={'vLLM' if use_vllm else 'HuggingFace'}).")


def main_pipeline():
    """Main TrialMatchAI pipeline with configurable vLLM/HuggingFace backend."""
    logger.info("Starting TrialMatchAI pipeline...")
    config = load_config()
    create_directory(config["paths"]["output_dir"])

    # Check if vLLM should be used (from config)
    use_vllm = config.get("global", {}).get("use_vllm", False)
    logger.info(f"CoT backend: {'vLLM' if use_vllm else 'HuggingFace'}")

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Initialize BioMedNER services
    initialize_biomedner_services(config)

    # Initialize model based on backend choice
    model = None
    tokenizer = None
    vllm_engine = None
    vllm_lora_request = None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        
        if use_vllm:
            logger.info("Loading vLLM engine for CoT...")
            vllm_config = config.get("vllm", {})
            
            # Reserve GPU memory for embedders and reranker
            if "gpu_memory_utilization" not in vllm_config:
                vllm_config["gpu_memory_utilization"] = 0.7
                logger.info(f"Setting vLLM gpu_memory_utilization to {vllm_config['gpu_memory_utilization']}")
            
            vllm_engine, tokenizer, vllm_lora_request = load_vllm_engine(
                config["model"], vllm_config
            )
        else:
            logger.info("Loading HuggingFace model for CoT...")
            model, tokenizer = load_model_and_tokenizer(
                config["model"], config["global"]["device"]
            )

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if model and hasattr(model, "config") and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # Apply half precision for HF model if not quantized
    if not use_vllm and model and config["global"]["device"] != "cpu" and torch.cuda.is_available():
        if not getattr(model, "is_loaded_in_4bit", False) and not getattr(model, "is_loaded_in_8bit", False):
            model = model.half()

    # Initialize embedders
    first_level_embedder = QueryEmbedder(
        model_name=config["embedder"]["model_name"],
        device_id=config["global"]["device"]
    )
    second_level_embedder = SecondLevelSentenceEmbedder(
        model_name=config["embedder"]["model_name"],
        device_id=config["global"]["device"]
    )
    
    # Initialize BioMedNER
    bio_med_ner = BioMedNER(**config["bio_med_ner"])

    # Initialize reranker
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        llm_reranker = LLMReranker(
            model_path=config["model"]["reranker_model_path"],
            adapter_path=config["model"]["reranker_adapter_path"],
            device=config["global"]["device"],
            batch_size=config["cot"]["batch_size"] * 2,
        )

    # Elasticsearch setup
    es_kwargs = {
        "hosts": [config["elasticsearch"]["host"]],
        "basic_auth": (
            config["elasticsearch"]["username"],
            config["elasticsearch"]["password"],
        ),
        "request_timeout": config["elasticsearch"]["request_timeout"],
        "retry_on_timeout": config["elasticsearch"]["retry_on_timeout"],
    }
    
    ca_certs = config["paths"].get("docker_certs")
    if ca_certs:
        es_kwargs["ca_certs"] = ca_certs
        
    es_client = Elasticsearch(**es_kwargs)

    # Initialize second-stage retriever
    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=second_level_embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
    )

    # Optional: restrict to specific NCT IDs
    restrict_nct_ids = None
    nct_ids_file = config.get("paths", {}).get("nct_ids_file")
    if nct_ids_file and os.path.exists(nct_ids_file):
        restrict_nct_ids = read_text_file(nct_ids_file)
        logger.info(f"Restricting to {len(restrict_nct_ids)} NCT IDs from {nct_ids_file}")

    # Process patients
    patient_folder = config["paths"]["patients_dir"]
    
    # Support both single JSON file and directory of phenopackets
    if os.path.isfile(patient_folder):
        # Single patients JSON file
        patients_data = read_json_file(patient_folder)
        patient_files = list(patients_data.keys())
        process_from_json = True
        logger.info(f"Processing {len(patient_files)} patients from {patient_folder}")
    else:
        # Directory with phenopacket files
        patient_files = [f for f in os.listdir(patient_folder) if f.endswith(".json")]
        process_from_json = False
        logger.info(f"Processing {len(patient_files)} phenopackets from {patient_folder}")

    for patient_file in patient_files:
        if process_from_json:
            # Patient data from JSON file
            patient_id = patient_file
            patient_info = patients_data[patient_id]
            output_folder = f"{config['paths']['output_dir']}/{patient_id}"
            create_directory(output_folder)
            
            # Create keywords from patient info
            keywords = {
                "main_conditions": patient_info.get("main_conditions", []),
                "other_conditions": patient_info.get("other_conditions", []),
                "expanded_sentences": (
                    patient_info.get("split_raw_description", [])
                    if "split_raw_description" in patient_info
                    else patient_info.get("expanded_sentences", [])
                ),
            }
            keywords_file = f"{output_folder}/keywords.json"
            write_json_file(keywords, keywords_file)
            
        else:
            # Process phenopacket file
            patient_id = patient_file.split(".")[0]
            output_folder = f"{config['paths']['output_dir']}/{patient_id}"
            create_directory(output_folder)

            input_file = f"{patient_folder}/{patient_file}"
            keywords_file = f"{output_folder}/keywords.json"

            # Process phenopacket (only with HF model, not needed for vLLM)
            if not use_vllm and model:
                with torch.no_grad():
                    process_phenopacket(
                        input_file, keywords_file, model=model, tokenizer=tokenizer
                    )
            elif use_vllm:
                logger.warning(
                    f"Phenopacket processing skipped for {patient_id} (vLLM mode). "
                    "Ensure keywords.json exists or process phenopackets separately."
                )

            keywords = read_json_file(keywords_file)
            patient_info = read_json_file(input_file)

        # Ensure patient profile is available
        patient_info["split_raw_description"] = keywords.get("expanded_sentences", [])

        logger.info(f"Processing patient {patient_id}...")

        # First-level search
        with torch.no_grad():
            result = run_first_level_search(
                keywords,
                output_folder,
                patient_info,
                bio_med_ner,
                first_level_embedder,
                config,
                es_client,
                restrict_nct_ids=restrict_nct_ids,
            )
        if not result:
            logger.error(f"First-level search failed for {patient_id}")
            continue

        (
            nct_ids,
            main_conditions,
            other_conditions,
            expanded_sentences,
            first_level_scores,
        ) = result

        # Second-level search
        with torch.no_grad():
            semi_final_trials, top_trials_path = run_second_level_search(
                output_folder,
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                gemma_retriever,
                first_level_scores,
                config,
            )

        # CoT processing
        with torch.no_grad():
            run_cot_processing(
                output_folder,
                top_trials_path,
                patient_info,
                model,
                tokenizer,
                config,
                use_vllm=use_vllm,
                vllm_engine=vllm_engine,
                vllm_lora_request=vllm_lora_request,
            )

        # Final ranking
        trial_data = load_trial_data(output_folder)
        ranked_trials = rank_trials(
            trial_data, 
            use_geographic_penalty=config["cot"].get("use_geographic_reasoning", False)
        )
        save_ranked_trials(ranked_trials, f"{output_folder}/ranked_trials.json")

        logger.info(f"✅ Pipeline completed for patient {patient_id}")

    logger.info("✅ All patients processed successfully!")


if __name__ == "__main__":
    main_pipeline()