import os
from typing import Dict, List, Optional

from elasticsearch import Elasticsearch
from Matcher.config.config_loader import load_config
from Matcher.models.embedding.query_embedder import QueryEmbedder
from Matcher.models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from Matcher.models.llm.llm_loader import load_model_and_tokenizer
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.pipeline.cot_reasoning import BatchTrialProcessor
from Matcher.pipeline.phenopacket_processor import process_phenopacket
from Matcher.pipeline.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
)
from Matcher.pipeline.trial_search.first_level_search import ClinicalTrialSearch
from Matcher.pipeline.trial_search.second_level_search import SecondStageRetriever
from Matcher.services.biomedner_service import initialize_biomedner_services
from Matcher.utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from Matcher.utils.logging_config import setup_logging
from Parser.biomedner_engine import BioMedNER

logger = setup_logging()


def run_first_level(
    keywords: Dict,
    output_folder: str,
    patient_info: Dict,
    bio_med_ner,
    embedder: QueryEmbedder,
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
    config = load_config()
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
    index_name = config["elasticsearch"]["index_trials"]
    cts = ClinicalTrialSearch(es_client, embedder, index_name, bio_med_ner)
    synonyms = cts.get_synonyms(condition.lower().strip())
    main_conditions.extend(synonyms[:10])
    trials, scores = cts.search_trials(
        condition=condition,
        age_input=age,
        sex=sex,
        overall_status=overall_status,
        size=500,
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


def run_second_level(
    output_folder: str,
    nct_ids: List[str],
    main_conditions: List[str],
    other_conditions: List[str],
    expanded_sentences: List[str],
    gemma_retriever: SecondStageRetriever,
    first_level_scores: Dict,
) -> tuple:
    queries = list(set(main_conditions + other_conditions + expanded_sentences))
    logger.info(f"Running second-level retrieval with {len(queries)} queries ...")
    synonyms = gemma_retriever.get_synonyms(queries[0])
    queries.extend(synonyms)
    second_level_results = gemma_retriever.retrieve_and_rank(
        queries, nct_ids, top_n=len(nct_ids)
    )
    combined_scores = {}
    for trial in second_level_results:
        trial_id = trial["nct_id"]
        second_score = trial["score"]
        first_score = first_level_scores.get(trial_id, 0)
        combined_scores[trial_id] = first_score + second_score
    sorted_trials = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    num_top = max(1, len(sorted_trials) // 2)
    semi_final_trials = sorted_trials[:num_top]
    top_trials_path = f"{output_folder}/top_trials.txt"
    write_text_file([trial_id for trial_id, _ in semi_final_trials], top_trials_path)
    logger.info("Second-level retrieval and ranking complete. Top trials saved.")
    return semi_final_trials, top_trials_path


def run_rag_global(
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
    patient_profile = patient_info.get("split_raw_description", [])
    if not patient_profile:
        logger.error("No patient profile available for RAG processing.")
        return
    rag_processor = BatchTrialProcessor(
        model,
        tokenizer,
        device=config["global"]["device"],
        batch_size=config["rag"]["batch_size"],
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

    # Initialize BioMedNER services
    initialize_biomedner_services(config)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config["model"], config["global"]["device"]
    )

    # Initialize embedders
    first_level_embedder = QueryEmbedder(model_name=config["embedder"]["model_name"])
    second_level_embedder = SecondLevelSentenceEmbedder(
        model_name=config["embedder"]["model_name"]
    )

    # Initialize BioMedNER
    bio_med_ner = BioMedNER(**config["bio_med_ner"])

    # Initialize LLM reranker
    llm_reranker = LLMReranker(
        model_path=config["model"]["base_model"],
        adapter_path=config["model"]["fine_tuned_adapter_phi"],
        device=config["global"]["device"],
        batch_size=config["rag"]["batch_size"],
    )

    # Initialize Elasticsearch client for second-level search
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

    # Initialize second-level retriever
    gemma_retriever = SecondStageRetriever(
        es_client=es_client,
        llm_reranker=llm_reranker,
        embedder=second_level_embedder,
        index_name=config["elasticsearch"]["index_trials_eligibility"],
        bio_med_ner=bio_med_ner,
    )

    # Process phenopackets
    patient_folder = config["paths"]["patients_dir"]
    for phenopacket_file in os.listdir(patient_folder):
        if not phenopacket_file.endswith(".json"):
            continue
        patient_id = phenopacket_file.split(".")[0]
        output_folder = f"{config['paths']['output_dir']}/{patient_id}"
        create_directory(output_folder)

        # Process phenopacket
        input_file = f"{patient_folder}/{phenopacket_file}"
        output_file = f"{output_folder}/keywords.json"
        process_phenopacket(input_file, output_file, model=model, tokenizer=tokenizer)

        # Load keywords and patient info
        keywords = read_json_file(output_file)
        patient_info = read_json_file(input_file)
        patient_info["split_raw_description"] = keywords.get("expanded_sentences", [])

        # Run first-level search
        result = run_first_level(
            keywords, output_folder, patient_info, bio_med_ner, first_level_embedder
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

        # Run second-level search
        semi_final_trials, top_trials_path = run_second_level(
            output_folder,
            nct_ids,
            main_conditions,
            other_conditions,
            expanded_sentences,
            gemma_retriever,
            first_level_scores,
        )

        # Run RAG processing
        run_rag_global(
            output_folder, top_trials_path, patient_info, model, tokenizer, config
        )

        # Rank trials
        trial_data = load_trial_data(output_folder)
        ranked_trials = rank_trials(trial_data)
        save_ranked_trials(ranked_trials, f"{output_folder}/ranked_trials.json")

        logger.info(f"Pipeline completed for patient {patient_id}")


if __name__ == "__main__":
    main_pipeline()
