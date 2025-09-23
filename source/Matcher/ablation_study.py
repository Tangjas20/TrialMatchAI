# Example usage:
# cd TrialMatchAI/source/
# python -m Matcher.ablation_study \
#     --ablation-config /path/to/ablation_config.json \
#     --trec-ground-truth /path/to/trec_ground_truth.csv \
#     --patients-file /path/to/patients.json \
#     --output-dir /path/to/output_directory \
#     --nct-ids-file /path/to/nct_ids.txt \
#     --patient P001 P002,P003
from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from elasticsearch import Elasticsearch
from Parser.biomedner_engine import BioMedNER

from .config.config_loader import load_config
from .models.embedding.query_embedder import QueryEmbedder
from .models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from .models.llm.llm_loader import load_model_and_tokenizer
from .models.llm.llm_reranker import LLMReranker
from .pipeline.cot_reasoning import BatchTrialProcessor
from .pipeline.trial_ranker import load_trial_data, rank_trials, save_ranked_trials
from .pipeline.trial_search.first_level_search import ClinicalTrialSearch
from .pipeline.trial_search.second_level_search import SecondStageRetriever
from .services.biomedner_service import initialize_biomedner_services
from .utils.evaluation import evaluate_and_save_metrics, load_ground_truth
from .utils.file_utils import (
    create_directory,
    read_json_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from .utils.logging_config import setup_logging

logger = setup_logging()

# Bounded synonym cache (trim when large)
_SYNONYM_CACHE: Dict[str, List[str]] = {}
_MAX_CACHE_SIZE = 1000

# Default ground-truth CSV path (can be overridden by CLI/config)
TREC_GROUND_TRUTH_DEFAULT = (
    "/home/testgpu/TrialGPT/dataset/trec_2021/qrels/test.tsv"
)

DEFAULT_ABLATION_CONFIG: Dict[str, Any] = {
    "use_biomedner": True,
    "first_level_search_mode": "hybrid",  # "bm25", "vector", "hybrid"
    "use_second_level_search": True,
    "use_second_level_rerank": True,
    "use_cot_reasoning": True,
    "max_trials_first_level": 1000,
    "max_trials_second_level": 500,
    "max_trials_cot": 100,
}


def _read_nct_ids_file(path: str) -> List[str]:
    """Read a .txt file with one NCT ID per line; return cleaned, unique list."""
    ids: List[str] = []
    if not path or not os.path.exists(path):
        return ids
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            token = raw.split()[0].upper()
            if re.fullmatch(r"NCT\d{8}", token) and token not in seen:
                seen.add(token)
                ids.append(token)
    return ids


def _normalize_patient_ids(arg: Optional[Union[List[str], str]]) -> Optional[List[str]]:
    """Turn a single string or list (with possible commas/spaces) into a clean, unique list."""
    if arg is None:
        return None
    parts: List[str] = []
    if isinstance(arg, str):
        parts = [p.strip() for p in re.split(r"[,\s]+", arg) if p.strip()]
    else:
        for item in arg:
            parts.extend([p.strip() for p in re.split(r"[,\s]+", item) if p.strip()])
    seen: Set[str] = set()
    out: List[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out or None


class AblationMetrics:
    """Tracks performance metrics and timing for each component."""

    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {}
        self._timers: Dict[str, float] = {}

    def start_timer(self, component: str) -> None:
        self._timers[component] = time.time()

    def end_timer(self, component: str) -> None:
        start = self._timers.pop(component, None)
        if start is None:
            return
        elapsed = time.time() - start
        self.metrics[f"{component}_time"] = elapsed
        logger.info(f"{component} took {elapsed:.2f} seconds")

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)


class AblationStudyRunner:
    """Main class for running ablation studies."""

    def __init__(
        self, config: Dict[str, Any], ablation_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.config = config
        self.ablation_config = {**DEFAULT_ABLATION_CONFIG, **(ablation_config or {})}
        self.ground_truth_map: Dict[str, Dict[str, int]] = {}
        self.model = None
        self.tokenizer = None
        self.first_level_embedder = None
        self.second_level_embedder = None
        self.bio_med_ner: Optional[BioMedNER] = None
        self.llm_reranker: Optional[LLMReranker] = None
        self.es_client: Optional[Elasticsearch] = None
        self.gemma_retriever: Optional[SecondStageRetriever] = None
        self.restrict_nct_ids: Optional[Set[str]] = None
        self.setup_components()

    def _normalize_tokenizer_padding(self) -> None:
        if self.tokenizer is None:
            return
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)
        if (
            hasattr(self.model, "config")
            and getattr(self.model.config, "pad_token_id", None) is None
            and getattr(self.tokenizer, "pad_token_id", None) is not None
        ):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def setup_components(self) -> None:
        """Initialize all components needed for the pipeline."""
        if self.ablation_config.get("use_biomedner", True):
            initialize_biomedner_services(self.config)
        else:
            logger.info(
                "BioMedNER disabled by ablation; services will not be initialized."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*quantization_config.*", category=UserWarning
            )
            self.model, self.tokenizer = load_model_and_tokenizer(
                self.config["model"], self.config["global"]["device"]
            )

        self._normalize_tokenizer_padding()

        # Mixed precision only when applicable and not 4/8-bit
        if (
            self.config["global"]["device"] != "cpu"
            and torch.cuda.is_available()
            and not getattr(self.model, "is_loaded_in_4bit", False)
            and not getattr(self.model, "is_loaded_in_8bit", False)
            and hasattr(self.model, "half")
        ):
            self.model = self.model.half()

        # Embedders + NER
        self.first_level_embedder = QueryEmbedder(
            model_name=self.config["embedder"]["model_name"],
            device_id=self.config["global"]["device"],
        )
        self.second_level_embedder = SecondLevelSentenceEmbedder(
            model_name=self.config["embedder"]["model_name"],
            device_id=self.config["global"]["device"],
        )
        self.bio_med_ner = (
            BioMedNER(**self.config["bio_med_ner"])
            if self.ablation_config.get("use_biomedner", True)
            else None
        )
        if self.bio_med_ner is None:
            logger.info("BioMedNER client is disabled (None).")

        # Reranker
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*quantization_config.*", category=UserWarning
            )
            self.llm_reranker = LLMReranker(
                model_path=self.config["model"]["reranker_model_path"],
                adapter_path=self.config["model"]["reranker_adapter_path"],
                device=self.config["global"]["device"],
                batch_size=self.config["cot"]["batch_size"] * 2,
            )

        # Elasticsearch client
        es_kwargs = dict(
            hosts=[self.config["elasticsearch"]["host"]],
            basic_auth=(
                self.config["elasticsearch"]["username"],
                self.config["elasticsearch"]["password"],
            ),
            request_timeout=self.config["elasticsearch"]["request_timeout"],
            retry_on_timeout=self.config["elasticsearch"]["retry_on_timeout"],
        )
        ca_certs = self.config["paths"].get("docker_certs")
        if ca_certs:
            es_kwargs["ca_certs"] = ca_certs
        self.es_client = Elasticsearch(**es_kwargs)

        # Second-stage retriever (NER may be None)
        self.gemma_retriever = SecondStageRetriever(
            es_client=self.es_client,
            llm_reranker=self.llm_reranker,
            embedder=self.second_level_embedder,
            index_name=self.config["elasticsearch"]["index_trials_eligibility"],
            bio_med_ner=self.bio_med_ner,
        )

    # ---------- First Level ----------

    def run_first_level_ablation(
        self,
        keywords: Dict[str, Any],
        output_folder: str,
        patient_info: Dict[str, Any],
        ablation_metrics: AblationMetrics,
    ) -> Optional[Tuple[List[str], List[str], List[str], List[str], Dict[str, float]]]:
        """Run first-level search with ablation controls."""
        ablation_metrics.start_timer("first_level")

        # Defensive copies; avoid mutating 'keywords'
        main_conditions = list(keywords.get("main_conditions") or [])
        other_conditions = list(keywords.get("other_conditions") or [])
        expanded_sentences = list(keywords.get("expanded_sentences") or [])

        if not main_conditions:
            logger.error("No main_conditions found in keywords.")
            ablation_metrics.end_timer("first_level")
            return None

        condition = main_conditions[0]
        age = patient_info.get("age", "all")
        sex = patient_info.get("gender", "all")
        overall_status = "All"

        index_name = self.config["elasticsearch"]["index_trials"]
        search_mode = (
            self.ablation_config.get("first_level_search_mode", "hybrid") or "hybrid"
        ).lower()
        if search_mode not in {"bm25", "vector", "hybrid"}:
            logger.warning(
                f"Unknown first_level_search_mode '{search_mode}', defaulting to 'hybrid'"
            )
            search_mode = "hybrid"

        embedder = None if search_mode == "bm25" else self.first_level_embedder
        if embedder is None:
            logger.info("First-level search mode: BM25 only (no embeddings).")

        bio_med_ner = (
            self.bio_med_ner
            if self.ablation_config.get("use_biomedner", True)
            else None
        )
        if bio_med_ner is None:
            logger.info("Synonym expansion disabled (BioMedNER off)")

        cts = ClinicalTrialSearch(self.es_client, embedder, index_name, bio_med_ner)

        # Synonym expansion (bounded cache)
        if bio_med_ner is not None and condition:
            key = condition.lower().strip()
            synonyms = _SYNONYM_CACHE.get(key)
            if synonyms is None:
                synonyms = cts.get_synonyms(key) or []
                _SYNONYM_CACHE[key] = synonyms
                if len(_SYNONYM_CACHE) > _MAX_CACHE_SIZE:
                    _SYNONYM_CACHE.clear()
            ablation_metrics.add_metric("synonyms_count", len(synonyms))
            main_conditions = main_conditions + synonyms[:5]
        else:
            ablation_metrics.add_metric("synonyms_count", 0)

        # Restrict list (if provided)
        pre_selected = list(self.restrict_nct_ids) if self.restrict_nct_ids else None

        search_size = int(self.ablation_config["max_trials_first_level"])
        trials, scores = cts.search_trials(
            condition=condition,
            age_input=age,
            sex=sex,
            overall_status=overall_status,
            size=search_size,
            pre_selected_nct_ids=pre_selected,  # restrict here
            synonyms=main_conditions,
            other_conditions=other_conditions,
            vector_score_threshold=self.config["search"]["vector_score_threshold"],
            search_mode=search_mode,
        )

        nct_ids: List[str] = [t.get("nct_id") for t in trials if t.get("nct_id")]
        first_level_scores: Dict[str, float] = {
            t.get("nct_id"): float(s) for t, s in zip(trials, scores) if t.get("nct_id")
        }

        # Hard-enforce exclusivity if restriction set present
        if self.restrict_nct_ids is not None:
            nct_ids = [nid for nid in nct_ids if nid in self.restrict_nct_ids]
            first_level_scores = {
                k: v
                for k, v in first_level_scores.items()
                if k in self.restrict_nct_ids
            }
            ablation_metrics.add_metric(
                "restricted_nct_ids", len(self.restrict_nct_ids)
            )

        # Persist ablation snapshot
        ablation_results = {
            "use_biomedner": self.ablation_config.get("use_biomedner", True),
            "first_level_search_mode": search_mode,
            "trials_found": len(nct_ids),
            "search_size": search_size,
            "restricted": self.restrict_nct_ids is not None,
        }
        write_json_file(
            ablation_results, os.path.join(output_folder, "first_level_ablation.json")
        )
        write_text_file(
            [str(nid) for nid in nct_ids], os.path.join(output_folder, "nct_ids.txt")
        )
        write_json_file(
            first_level_scores, os.path.join(output_folder, "first_level_scores.json")
        )

        ablation_metrics.add_metric("first_level_trials", len(nct_ids))
        ablation_metrics.end_timer("first_level")
        logger.info(f"First-level search complete: {len(nct_ids)} trial IDs saved.")

        return (
            nct_ids,
            main_conditions,
            other_conditions,
            expanded_sentences,
            first_level_scores,
        )

    # ---------- Second Level ----------

    @staticmethod
    def _unique_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def run_second_level_ablation(
        self,
        output_folder: str,
        nct_ids: List[str],
        main_conditions: List[str],
        other_conditions: List[str],
        expanded_sentences: List[str],
        first_level_scores: Dict[str, float],
        ablation_metrics: AblationMetrics,
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Run second-level search with ablation controls."""
        ablation_metrics.start_timer("second_level")

        if not self.ablation_config.get("use_second_level_search", False):
            logger.info(
                "ABLATION: Second-level search disabled, using first-level results only"
            )
            sorted_trials = sorted(
                first_level_scores.items(), key=lambda x: x[1], reverse=True
            )
            num_top = min(
                len(sorted_trials), int(self.ablation_config["max_trials_second_level"])
            )
            semi_final_trials = sorted_trials[:num_top]
            write_json_file(
                {
                    "second_level_enabled": False,
                    "trials_processed": len(semi_final_trials),
                },
                os.path.join(output_folder, "second_level_ablation.json"),
            )
            top_trials_path = os.path.join(output_folder, "top_trials.txt")
            write_text_file(
                [trial_id for trial_id, _ in semi_final_trials], top_trials_path
            )
            ablation_metrics.add_metric("second_level_trials", len(semi_final_trials))
            ablation_metrics.end_timer("second_level")
            return semi_final_trials, top_trials_path

        # Build queries (preserve order + uniqueness)
        queries = self._unique_preserve_order(
            main_conditions + other_conditions + expanded_sentences
        )
        logger.info(f"Running second-level retrieval with {len(queries)} queries ...")
        top_n = min(len(nct_ids), int(self.ablation_config["max_trials_second_level"]))
        use_reranker = bool(self.ablation_config.get("use_second_level_rerank", True))

        second_level_results = self.gemma_retriever.retrieve_and_rank(  # type: ignore[union-attr]
            queries, nct_ids, top_n=top_n, use_reranker=use_reranker
        )

        combined_scores: Dict[str, float] = {}
        for trial in second_level_results:
            trial_id = trial["nct_id"]
            second_score = float(trial["score"])
            first_score = float(first_level_scores.get(trial_id, 0.0))
            combined_scores[trial_id] = first_score + second_score

        sorted_trials = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        semi_final_trials = sorted_trials[
            : int(self.ablation_config["max_trials_second_level"])
        ]

        write_json_file(
            {
                "second_level_enabled": True,
                "queries_used": len(queries),
                "trials_processed": len(semi_final_trials),
                "use_second_level_rerank": use_reranker,
            },
            os.path.join(output_folder, "second_level_ablation.json"),
        )

        top_trials_path = os.path.join(output_folder, "top_trials.txt")
        write_text_file(
            [trial_id for trial_id, _ in semi_final_trials], top_trials_path
        )

        ablation_metrics.add_metric("second_level_trials", len(semi_final_trials))
        ablation_metrics.end_timer("second_level")
        logger.info("Second-level retrieval and ranking complete. Top trials saved.")
        return semi_final_trials, top_trials_path

    # ---------- CoT ----------

    def run_cot_ablation(
        self,
        output_folder: str,
        top_trials_file: str,
        patient_info: Dict[str, Any],
        ablation_metrics: AblationMetrics,
    ) -> None:
        """Run CoT processing with ablation controls."""
        ablation_metrics.start_timer("cot")

        top_trials = read_text_file(top_trials_file)
        if not top_trials:
            logger.error("No top trials available for cot processing.")
            ablation_metrics.end_timer("cot")
            return

        top_trials = top_trials[: int(self.ablation_config["max_trials_cot"])]

        patient_profile = patient_info.get("split_raw_description", [])
        if not patient_profile:
            logger.error("No patient profile available for cot processing.")
            ablation_metrics.end_timer("cot")
            return

        self._normalize_tokenizer_padding()
        batch_size = min(int(self.config["cot"]["batch_size"]) * 2, 4)

        cot_processor = BatchTrialProcessor(
            self.model,
            self.tokenizer,
            device=self.config["global"]["device"],
            batch_size=batch_size,
            use_cot=bool(self.ablation_config.get("use_cot_reasoning", True)),
        )
        cot_processor.process_trials(
            nct_ids=top_trials,
            json_folder=self.config["paths"]["trials_json_folder"],
            output_folder=output_folder,
            patient_profile=patient_profile,
        )

        write_json_file(
            {
                "cot_enabled": True,
                "trials_processed": len(top_trials),
                "batch_size": batch_size,
                "use_cot_reasoning": bool(
                    self.ablation_config.get("use_cot_reasoning", True)
                ),
            },
            os.path.join(output_folder, "cot_ablation.json"),
        )
        write_json_file(
            {"status": "done"}, os.path.join(output_folder, "cot_output.json")
        )

        ablation_metrics.add_metric("cot_trials", len(top_trials))
        ablation_metrics.end_timer("cot")
        logger.info("CoT-based trial matching complete.")

    # ---------- Scenario + Runner ----------

    def _scenario_name(self, cfg: Dict[str, Any]) -> str:
        parts = [
            f"fl-{cfg.get('first_level_search_mode', 'hybrid')}",
            "rerank" if cfg.get("use_second_level_rerank", True) else "no-rerank",
            "cot" if cfg.get("use_cot_reasoning", True) else "no-cot",
            "ner" if cfg.get("use_biomedner", True) else "no-ner",
        ]
        return "_".join(parts)

    @staticmethod
    def _is_scenario_complete(output_folder: str) -> bool:
        # Consider scenario complete if evaluation metrics exist
        eval_metrics = os.path.exists(
            os.path.join(output_folder, "evaluation_metrics.json")
        )
        return eval_metrics

    def _run_patient_scenarios(
        self,
        patient_id: str,
        patient_info: Dict[str, Any],
        keywords: Dict[str, Any],
        results_patient_dir: str,
    ) -> None:
        """Run the ablation once for a single patient."""
        profile_lines = (
            patient_info.get("split_raw_description")
            or patient_info.get("expanded_sentences")
            or (
                [patient_info.get("raw_description")]
                if patient_info.get("raw_description")
                else []
            )
        )
        patient_info_local = dict(patient_info)  # shallow copy
        patient_info_local["split_raw_description"] = profile_lines

        scenario_name = self._scenario_name(self.ablation_config)
        output_folder = os.path.join(results_patient_dir, scenario_name)

        if self._is_scenario_complete(output_folder):
            logger.info(
                f"Skipping patient {patient_id} for scenario '{scenario_name}' (already complete)."
            )
            return

        create_directory(output_folder)
        write_json_file(
            self.ablation_config.copy(),
            os.path.join(output_folder, "ablation_config.json"),
        )

        ablation_metrics = AblationMetrics()
        ablation_metrics.add_metric("patient_id", patient_id)
        ablation_metrics.add_metric("ablation_config", self.ablation_config.copy())

        # First-level (reuse if present)
        nct_ids_path = os.path.join(output_folder, "nct_ids.txt")
        first_scores_path = os.path.join(output_folder, "first_level_scores.json")
        if os.path.exists(nct_ids_path) and os.path.exists(first_scores_path):
            logger.info("Skipping first-level: existing outputs found.")
            nct_ids = read_text_file(nct_ids_path)
            first_level_scores = read_json_file(first_scores_path)
            main_conditions = list(keywords.get("main_conditions") or [])
            other_conditions = list(keywords.get("other_conditions") or [])
            expanded_sentences = list(keywords.get("expanded_sentences") or [])
        else:
            with torch.no_grad():
                result = self.run_first_level_ablation(
                    keywords, output_folder, patient_info_local, ablation_metrics
                )
            if not result:
                logger.error(
                    f"First-level search failed for {patient_id} in scenario {scenario_name}"
                )
                return
            (
                nct_ids,
                main_conditions,
                other_conditions,
                expanded_sentences,
                first_level_scores,
            ) = result

        # Second-level (reuse if present)
        top_trials_path = os.path.join(output_folder, "top_trials.txt")
        if not os.path.exists(top_trials_path):
            with torch.no_grad():
                _, top_trials_path = self.run_second_level_ablation(
                    output_folder,
                    nct_ids,
                    main_conditions,
                    other_conditions,
                    expanded_sentences,
                    first_level_scores,
                    ablation_metrics,
                )
        else:
            logger.info("Skipping second-level: top_trials.txt already exists.")

        # CoT (reuse if done)
        cot_done_path = os.path.join(output_folder, "cot_output.json")
        skip_cot = False
        if os.path.exists(cot_done_path):
            try:
                status_obj = read_json_file(cot_done_path)
                skip_cot = (
                    isinstance(status_obj, dict) and status_obj.get("status") == "done"
                )
            except Exception:
                skip_cot = False

        if not skip_cot:
            with torch.no_grad():
                self.run_cot_ablation(
                    output_folder, top_trials_path, patient_info_local, ablation_metrics
                )
        else:
            logger.info("Skipping CoT: cot_output.json indicates completion.")

        # Final ranking (reuse if present)
        ranked_path = os.path.join(output_folder, "ranked_trials.json")
        if os.path.exists(ranked_path):
            logger.info("Skipping ranking: ranked_trials.json already exists.")
            ranked_trials = read_json_file(ranked_path)
        else:
            trial_data = load_trial_data(output_folder)
            ranked_trials = rank_trials(trial_data)
            save_ranked_trials(ranked_trials, ranked_path)

        # Evaluation (if ground truth available) and persist metrics
        try:
            evaluate_path = os.path.join(output_folder, "evaluation_metrics.json")
            if not os.path.exists(evaluate_path):
                logger.info("Running evaluation...")
                metrics = evaluate_and_save_metrics(
                    ranked_trials,
                    patient_id,
                    getattr(self, "ground_truth_map", {}),
                    output_folder,
                )
                ablation_metrics.add_metric("evaluation", metrics)
            else:
                logger.info(
                    "Skipping evaluation: evaluation_metrics.json already exists."
                )
        except Exception as e:
            logger.warning(f"Evaluation failed for {patient_id}: {e}")

        final_metrics = ablation_metrics.get_metrics()
        final_metrics["total_ranked_trials"] = len(ranked_trials)
        write_json_file(
            final_metrics, os.path.join(output_folder, "ablation_metrics.json")
        )
        logger.info(
            f"Ablation scenario '{scenario_name}' completed for patient {patient_id}"
        )

        # Trim cache if it grows too large
        if len(_SYNONYM_CACHE) > _MAX_CACHE_SIZE:
            _SYNONYM_CACHE.clear()

    # ---------- Entry Point ----------

    def run_ablation_study(self, patient_ids: Optional[List[str]] = None) -> None:
        """Run the complete ablation study from a patients JSON file."""
        logger.info("Starting TrialMatchAI ablation study...")

        # Results root
        results_root = self.config["paths"]["output_dir"]
        create_directory(results_root)

        # Patients file (configurable; fallback to default path)
        patients_file = self.config.get("paths", {}).get(
            "patients_file",
            "../data/processed_patients21.json",
        )
        if not os.path.exists(patients_file):
            raise FileNotFoundError(f"Patients file not found: {patients_file}")
        patients: Dict[str, Dict[str, Any]] = read_json_file(patients_file)
        logger.info(f"Loaded {len(patients)} patients from {patients_file}")

        # Load TREC ground truth once
        gt_path = self.config.get("paths", {}).get(
            "trec_ground_truth", TREC_GROUND_TRUTH_DEFAULT
        )
        if os.path.exists(gt_path):
            try:
                self.ground_truth_map = load_ground_truth(gt_path)
                logger.info(
                    f"Loaded ground truth for {len(self.ground_truth_map)} queries from {gt_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load ground truth from {gt_path}: {e}")
                self.ground_truth_map = {}
        else:
            logger.warning(
                f"Ground truth file not found at {gt_path}; evaluation disabled."
            )
            self.ground_truth_map = {}

        # Load optional NCT restriction list
        nct_ids_file = self.config.get("paths", {}).get("nct_ids_file")
        if nct_ids_file:
            restrict_list = _read_nct_ids_file(nct_ids_file)
            if restrict_list:
                self.restrict_nct_ids = set(restrict_list)
                logger.info(
                    f"Restricting retrieval to {len(self.restrict_nct_ids)} NCT IDs from {nct_ids_file}"
                )
            else:
                logger.warning(
                    f"--nct-ids-file provided but no valid NCT IDs found in {nct_ids_file}; ignoring."
                )

        # Restrict to specific patients if requested
        if patient_ids:
            missing = [pid for pid in patient_ids if pid not in patients]
            if missing:
                logger.warning(
                    f"{len(missing)} requested patient ids not found: {missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            present = {pid: patients[pid] for pid in patient_ids if pid in patients}
            if not present:
                logger.error("None of the requested patient ids were found; aborting.")
                return
            patients_to_iterate: Dict[str, Dict[str, Any]] = present
        else:
            patients_to_iterate = patients

        # Iterate patients
        for patient_id, patient_info in patients_to_iterate.items():
            #translate patient_id to trec-{trec_year}{id}. Ex. P001 -> trec-20211 if Trec 2021
            if re.fullmatch(r"P\d{3}", patient_id):
                # get trec year by reading the processed patient file name, if it contains 21 or 22
                trec_year = "21"  # default
                if "22" in os.path.basename(patients_file):
                    trec_year = "22"
                # Remove leading zeroes from the numeric part of the patient ID
                patient_number = int(patient_id[1:])  # Convert "001" to 1, "075" to 75, etc.
                patient_id = f"trec-20{trec_year}{patient_number}"

            results_patient_dir = os.path.join(results_root, patient_id)
            create_directory(results_patient_dir)

            scenario_name = self._scenario_name(self.ablation_config)
            output_folder = os.path.join(results_patient_dir, scenario_name)
            if self._is_scenario_complete(output_folder):
                logger.info(
                    f"Skipping patient {patient_id} (scenario '{scenario_name}' already complete)."
                )
                continue

            # Precomputed keywords.json required
            keywords_path = os.path.join(results_patient_dir, "keywords.json")
            if not os.path.exists(keywords_path):
                logger.warning(
                    f"Missing keywords.json for {patient_id} at {keywords_path}; skipping patient."
                )
                continue
            keywords = read_json_file(keywords_path)

            self._run_patient_scenarios(
                patient_id, patient_info, keywords, results_patient_dir
            )


def run_ablation_study_from_config(ablation_config_path: Optional[str] = None) -> None:
    """Run ablation study with configuration from file."""
    config = load_config()
    ablation_config = DEFAULT_ABLATION_CONFIG.copy()
    if ablation_config_path and os.path.exists(ablation_config_path):
        ablation_override = read_json_file(ablation_config_path)
        ablation_config.update(ablation_override)
        logger.info(
            f"Loaded ablation config from {ablation_config_path}: {ablation_config}"
        )
    runner = AblationStudyRunner(config, ablation_config)
    runner.run_ablation_study()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TrialMatchAI ablation study")
    parser.add_argument(
        "--ablation-config",
        "-a",
        help="Path to JSON file with ablation configuration overrides",
        default=None,
    )
    parser.add_argument(
        "--trec-ground-truth",
        "-t",
        help="Path to TREC ground truth CSV (overrides default)",
        default=None,
    )
    parser.add_argument(
        "--patients-file",
        "-p",
        help="Path to patients JSON file (overrides config.paths.patients_file)",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Results output directory (overrides config.paths.output_dir)",
        default=None,
    )
    parser.add_argument(
        "--nct-ids-file",
        "-n",
        help="Path to a .txt file with one NCT ID per line; restrict search/retrieval exclusively to these trials",
        default=None,
    )
    parser.add_argument(
        "--patient",
        nargs="+",
        help="One or more patient ids (space- or comma-separated)",
        default=None,
    )
    args = parser.parse_args()

    # Load base config + ablation config
    config = load_config()
    ablation_config = DEFAULT_ABLATION_CONFIG.copy()
    if args.ablation_config and os.path.exists(args.ablation_config):
        ablation_override = read_json_file(args.ablation_config)
        ablation_config.update(ablation_override)
        logger.info(
            f"Loaded ablation config from {args.ablation_config}: {ablation_config}"
        )

    # Build runner and apply CLI path overrides
    runner = AblationStudyRunner(config, ablation_config)
    runner.config.setdefault("paths", {})
    if args.trec_ground_truth:
        runner.config["paths"]["trec_ground_truth"] = args.trec_ground_truth
    if args.patients_file:
        runner.config["paths"]["patients_file"] = args.patients_file
    if args.output_dir:
        runner.config["paths"]["output_dir"] = args.output_dir
    if args.nct_ids_file:
        runner.config["paths"]["nct_ids_file"] = args.nct_ids_file

    # Log GPU name
    logger.warning(
        f"Using device: {runner.config['global']['device']} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Normalize patient ids and run
    patient_ids = _normalize_patient_ids(args.patient)
    runner.run_ablation_study(patient_ids=patient_ids)
