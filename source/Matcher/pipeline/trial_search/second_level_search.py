import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from Matcher.models.embedding.sentence_embedder import SecondLevelSentenceEmbedder
from Matcher.models.llm.llm_reranker import LLMReranker
from Matcher.utils.file_utils import write_text_file
from Matcher.utils.logging_config import setup_logging

from elasticsearch import Elasticsearch

logger = setup_logging()


class SecondStageRetriever:
    def __init__(
        self,
        es_client: Elasticsearch,
        llm_reranker: LLMReranker,
        embedder: SecondLevelSentenceEmbedder,
        index_name: str,
        size: int = 250,
        inclusion_weight: float = 1.0,
        exclusion_weight: float = 0.25,
        bio_med_ner=None,
    ):
        self.es_client = es_client
        self.llm_reranker = llm_reranker
        self.embedder = embedder
        self.index_name = index_name
        self.size = size
        self.inclusion_weight = inclusion_weight
        self.exclusion_weight = exclusion_weight
        self.bio_med_ner = bio_med_ner

    def get_synonyms(self, condition: str) -> List[str]:
        if self.bio_med_ner is None:
            logger.warning("BioMedNER not initialized; cannot extract synonyms.")
            return []
        raw_result = self.bio_med_ner.annotate_texts_in_parallel(
            [condition], max_workers=1
        )
        ner_results = raw_result
        if ner_results and ner_results[0]:
            synonyms = set()
            for entity in ner_results[0]:
                if entity.get("entity_group", "").lower() == "disease":
                    synonyms.update(entity.get("synonyms", []))
            return list(synonyms)
        logger.warning(f"No annotations found for condition: {condition}")
        return []

    def retrieve_criteria(
        self, nct_ids: List[str], queries: List[str]
    ) -> Dict[str, List[Dict]]:
        query_to_hits = {}

        def execute_query(query):
            query_vector = self.embedder.get_embeddings(query)

            # Use entities.synonyms only if BioMedNER is enabled
            fields_to_search = (
                ["criterion", "entities.synonyms"]
                if self.bio_med_ner is not None
                else ["criterion"]
            )

            es_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": fields_to_search,
                                        "type": "best_fields",
                                        "operator": "and",
                                    }
                                },
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": fields_to_search,
                                        "type": "phrase",
                                    }
                                },
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": fields_to_search,
                                        "type": "best_fields",
                                        "operator": "or",
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                            "filter": {"terms": {"nct_id": nct_ids}},
                        }
                    },
                    "script": {
                        "source": """
                            double alpha = 0.5;
                            double beta = 0.5;
                            double textScore = _score;
                            double vectorScore = (cosineSimilarity(params.query_vector, 'criterion_vector') + 1.0) / 2.0;
                            if (vectorScore < params.vector_score_threshold) {
                                return 0;
                            }
                            return alpha * textScore + beta * vectorScore;
                        """,
                        "params": {
                            "query_vector": query_vector,
                            "vector_score_threshold": 0.5,
                        },
                    },
                }
            }
            response = self.es_client.search(
                index=self.index_name, body={"size": self.size, "query": es_query}
            )
            hits = response["hits"]["hits"]
            logger.info(f"Retrieved {len(hits)} documents for query: '{query}'")
            return query, hits

        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as executor:
            future_to_query = {
                executor.submit(execute_query, query): query for query in queries
            }
            for future in as_completed(future_to_query):
                query, hits = future.result()
                query_to_hits[query] = hits
        return query_to_hits

    def rerank_criteria(self, queries: List[str], criteria: List[Dict]) -> List[Dict]:
        pairs = [
            (criterion["query"], criterion["_source"]["criterion"])
            for criterion in criteria
        ]
        llm_scores = self.llm_reranker.rank_pairs(pairs)
        llm_scores = [
            score.get("llm_score", 0.0) if isinstance(score, dict) else float(score)
            for score in llm_scores
        ]
        if len(llm_scores) != len(pairs):
            logger.error("Mismatch between LLM scores and pairs!")
            raise ValueError("Mismatch between LLM scores and pairs!")
        for i, criterion in enumerate(criteria):
            llm_score = llm_scores[i]
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion criteria":
                llm_score *= self.inclusion_weight
            elif eligibility_type == "exclusion criteria":
                llm_score *= self.exclusion_weight
            criterion["llm_score"] = llm_score
        return criteria

    def score_criteria_without_llm(self, criteria: List[Dict]) -> List[Dict]:
        if not criteria:
            return criteria
        max_es = max((c.get("_score", 0.0) for c in criteria), default=1.0) or 1.0
        for criterion in criteria:
            base = float(criterion.get("_score", 0.0)) / max_es
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion criteria":
                base *= self.inclusion_weight
            elif eligibility_type == "exclusion criteria":
                base *= self.exclusion_weight
            criterion["llm_score"] = base
        return criteria

    def aggregate_to_trials(
        self, criteria: List[Dict], threshold: float = 0.5, method: str = "weighted"
    ) -> List[Dict]:
        trial_scores = defaultdict(list)
        for criterion in criteria:
            nct_id = criterion["_source"]["nct_id"]
            score = criterion["llm_score"]
            if score >= threshold:
                trial_scores[nct_id].append(score)
        aggregated_scores = {}
        for nct_id, scores in trial_scores.items():
            count = len(scores)
            total = sum(scores)
            if count == 0:
                continue
            if method == "avg":
                agg_score = total / count
            elif method == "sqrt":
                agg_score = total / math.sqrt(count)
            elif method == "log":
                agg_score = total / math.log(count + 1)
            elif method == "weighted":
                agg_score = 0.7 * (total / math.sqrt(count)) + 0.3 * max(scores)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            aggregated_scores[nct_id] = agg_score
        sorted_trials = [
            {"nct_id": nct_id, "score": score}
            for nct_id, score in sorted(
                aggregated_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return sorted_trials

    def retrieve_and_rank(
        self,
        queries: List[str],
        nct_ids: List[str],
        top_n: int,
        use_reranker: bool = True,
        save_path: Optional[str] = None,
    ) -> List[Dict]:
        query_to_hits = self.retrieve_criteria(nct_ids, queries)
        all_criteria = []
        for query, hits in query_to_hits.items():
            for hit in hits:
                hit["query"] = query
                all_criteria.append(hit)

        if use_reranker:
            ranked_criteria = self.rerank_criteria(queries, all_criteria)
        else:
            logger.info(
                "Second-level reranking disabled; using ES scores for aggregation."
            )
            ranked_criteria = self.score_criteria_without_llm(all_criteria)

        sorted_trials = self.aggregate_to_trials(ranked_criteria)
        top_trials = sorted_trials[:top_n]
        logger.info(f"Top {top_n} trials retrieved: {top_trials}")
        if save_path:
            write_text_file([trial["nct_id"] for trial in top_trials], save_path)
            logger.info(f"Top trials saved to {save_path}")
        return top_trials
