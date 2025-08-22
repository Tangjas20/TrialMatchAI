from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from Parser.biomedner_engine import BioMedNER

from elasticsearch import Elasticsearch

from .config.config_loader import load_config
from .models.embedding.query_embedder import QueryEmbedder
from .models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from .models.llm.llm_loader import load_model_and_tokenizer
from .models.llm.llm_reranker import LLMReranker
from .pipeline.cot_reasoning import BatchTrialProcessor
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

logger = setup_logging()


def run_first_level_search(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: QueryEmbedder,
    config: Dict,
    es_client: Elasticsearch,
) -> Optional[tuple]:
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

    search_size = config["search"].get("max_trials_first_level", 300)
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=search_size,
        pre_selected_nct_ids=None,
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
    queries = list(set(main_conditions + other_conditions + expanded_sentences))[:10]
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")

    # Add synonyms for second level
    if queries:
        synonyms = gemma_retriever.get_synonyms(queries[0])
        queries.extend(synonyms[:3])

    top_n = min(len(nct_ids), config["search"].get("max_trials_second_level", 100))
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


def run_rag_processing(
    output_folder: str,
    top_trials_file: str,
    patient_info: Dict,
    model,
    tokenizer,
    config: Dict,
):
    top_trials = read_text_file(top_trials_file)
    if not top_trials:
        logger.error("No top trials available for RAG processing.")
        return

    top_trials = top_trials[: config["rag"].get("max_trials_rag", 20)]
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return

    batch_size = min(config["rag"]["batch_size"] * 2, 8)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rag_processor = BatchTrialProcessor(
        model,
        tokenizer,
        device=config["global"]["device"],
        batch_size=batch_size,
    )
    rag_processor.process_trials(
        nct_ids=top_trials,
        json_folder=config["paths"]["trials_json_folder"],
        output_folder=output_folder,
        patient_profile=patient_profile,
    )

    write_json_file({"status": "done"}, f"{output_folder}/rag_output.json")
    logger.info("RAG-based trial matching complete.")


def main_pipeline():
    logger.info("Starting TrialMatchAI pipeline...")
    config = load_config()
    create_directory(config["paths"]["output_dir"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    initialize_biomedner_services(config)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        model, tokenizer = load_model_and_tokenizer(
            config["model"], config["global"]["device"]
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    if config["global"]["device"] != "cpu" and torch.cuda.is_available():
        model = model.half()

    # Initialize components
    first_level_embedder = QueryEmbedder(model_name=config["embedder"]["model_name"])
    second_level_embedder = SecondLevelSentenceEmbedder(
        model_name=config["embedder"]["model_name"]
    )
    bio_med_ner = BioMedNER(**config["bio_med_ner"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*quantization_config.*", category=UserWarning
        )
        llm_reranker = LLMReranker(
            model_path=config["model"]["reranker_model_path"],
            adapter_path=config["model"]["reranker_adapter_path"],
            device=config["global"]["device"],
            batch_size=config["rag"]["batch_size"] * 2,
        )

    es_client = Elasticsearch(
        hosts=[config["elasticsearch"]["host"]],
        ca_certs=config["paths"]["docker_certs"],
        basic_auth=(
            config["elasticsearch"]["username"],
            config["elasticsearch"]["password"],
        ),
        request_timeout=config["elasticsearch"]["request_timeout"],
        retry_on_timeout=config["elasticsearch"]["retry_on_timeout"],
    )

    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=second_level_embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
    )

    # Process phenopackets
    patient_folder = config["paths"]["patients_dir"]
    phenopacket_files = [f for f in os.listdir(patient_folder) if f.endswith(".json")]

    for phenopacket_file in phenopacket_files:
        patient_id = phenopacket_file.split(".")[0]
        output_folder = f"{config['paths']['output_dir']}/{patient_id}"
        create_directory(output_folder)

        input_file = f"{patient_folder}/{phenopacket_file}"
        output_file = f"{output_folder}/keywords.json"

        with torch.no_grad():
            process_phenopacket(
                input_file, output_file, model=model, tokenizer=tokenizer
            )

        keywords = read_json_file(output_file)
        patient_info = read_json_file(input_file)
        patient_info["split_raw_description"] = keywords.get("expanded_sentences", [])

        # Run pipeline
        with torch.no_grad():
            result = run_first_level_search(
                keywords,
                output_folder,
                patient_info,
                bio_med_ner,
                first_level_embedder,
                config,
                es_client,
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

        with torch.no_grad():
            run_rag_processing(
                output_folder,
                top_trials_path,
                patient_info,
                model,
                tokenizer,
                config,
            )

        # Final ranking
        trial_data = load_trial_data(output_folder)
        ranked_trials = rank_trials(trial_data)
        save_ranked_trials(ranked_trials, f"{output_folder}/ranked_trials.json")

        logger.info(f"Pipeline completed for patient {patient_id}")


if __name__ == "__main__":
    main_pipeline()
