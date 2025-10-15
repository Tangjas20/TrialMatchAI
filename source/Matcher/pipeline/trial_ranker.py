import os
from typing import Dict, List

from Matcher.utils.file_utils import read_json_file, write_json_file
from Matcher.utils.logging_config import setup_logging

logger = setup_logging()


def load_trial_data(json_folder: str) -> List[Dict]:
    trial_data = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json") and file_name.startswith("NCT"):
            file_path = os.path.join(json_folder, file_name)
            trial_id = os.path.splitext(file_name)[0]
            try:
                trial = read_json_file(file_path)
                trial["TrialID"] = trial_id
                trial_data.append(trial)
            except Exception as e:
                logger.error(f"Failed to load {file_name}: {e}")
    return trial_data


def score_trial(trial: Dict, use_geographic_penalty: bool = False) -> float:
    def calculate_ratio(
        criteria_list, positive_classifications, negative_classifications
    ):
        criteria_to_exclude = ["Irrelevant", "Unclear"]
        criteria_list = [
            c
            for c in criteria_list
            if c.get("Classification") not in criteria_to_exclude
        ]
        total_criteria = len(criteria_list)
        if total_criteria == 0:
            return 0.0
        positive_count = sum(
            1
            for c in criteria_list
            if c.get("Classification") in positive_classifications
        )
        negative_count = sum(
            1
            for c in criteria_list
            if c.get("Classification") in negative_classifications
        )
        penalty_factor_negative = 1.0
        reward_factor_positive = 1.0
        score = (
            reward_factor_positive * positive_count
            - penalty_factor_negative * negative_count
        ) / total_criteria
        return score

    inclusion_criteria = trial.get("Inclusion_Criteria_Evaluation", [])
    exclusion_criteria = trial.get("Exclusion_Criteria_Evaluation", [])
    inclusion_ratio = calculate_ratio(
        inclusion_criteria, ["Met", "Not Violated"], ["Violated", "Not Met"]
    )
    exclusion_ratio = calculate_ratio(
        exclusion_criteria, ["Not Violated", "Met"], ["Violated"]
    )

    base_score = (inclusion_ratio + exclusion_ratio) / 2
    if use_geographic_penalty and "Geographic_Assessment" in trial:
        geo_assessment = trial.get("Geographic_Assessment", {})
        geo_match = geo_assessment.get("Geographic_Match", "Unknown Geography")

        geo_mult = {
            "Strong Geographic Match": 1.00,      # No penalty
            "Moderate Geographic Match": 0.85,    # 10% penalty
            "Weak Geographic Match": 0.75,        # 25% penalty
            "Geographic Mismatch": 0.60,          # 40% penalty
            "Unknown Geography": 0.92,            # 8% penalty
        }

        mult = geo_mult.get(geo_match, 0.92)
        return base_score * mult
        
    return base_score


def rank_trials(trial_data: List[Dict], use_geographic_penalty: bool = False) -> List[Dict]:
    ranked_trials = []
    for trial in trial_data:
        trial_id = trial.get("TrialID", "Unknown")
        score = score_trial(trial, use_geographic_penalty)
        ranked_trials.append({"TrialID": trial_id, "Score": score})
    ranked_trials.sort(key=lambda x: x["Score"], reverse=True)
    return ranked_trials


def save_ranked_trials(ranked_trials: List[Dict], output_file: str):
    try:
        write_json_file({"RankedTrials": ranked_trials}, output_file)
        logger.info(f"Ranked trials saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save ranked trials: {e}")
