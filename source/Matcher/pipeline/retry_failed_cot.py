# Matcher/pipeline/retry_failed_cot.py
# Example usage: python retry_failed_cot.py --results-dir ../data/lung_results --trials-json ../data/lung_processed_trials --patients-file ../data/final_lung_dataset_pipeline.json --scenario fl-hybrid_rerank_cot-vllm_ner
import os
import json
from typing import List, Dict, Set, Tuple
from pathlib import Path
import re

from Matcher.utils.file_utils import read_text_file, read_json_file, write_json_file
from Matcher.utils.logging_config import setup_logging
from Matcher.pipeline.cot_reasoning_vllm import BatchTrialProcessorVLLM
from Matcher.pipeline.trial_ranker import score_trial
from Matcher.config.config_loader import load_config
from Matcher.models.llm.vllm_loader import load_vllm_engine

logger = setup_logging()


class FailedCoTRetrier:
    """Retry CoT reasoning for trials that failed to produce valid JSON. Runnable post-analysis."""
    
    def __init__(self, cot_processor: BatchTrialProcessorVLLM):
        self.cot_processor = cot_processor
    
    def find_failed_trials(self, cot_output_folder: str) -> List[Tuple[str, str]]:
        """
        Find trials with .txt but no valid .json output.
        
        Returns:
            List of (nct_id, txt_file_path) tuples
        """
        failed_trials = []
        
        txt_files = list(Path(cot_output_folder).glob("NCT*.txt"))
        
        for txt_file in txt_files:
            nct_id = txt_file.stem  # NCT12345678
            json_file = txt_file.with_suffix('.json')
            
            # Check if JSON exists and is valid
            needs_retry = False
            
            if not json_file.exists():
                logger.warning(f"{nct_id}: No JSON file found")
                needs_retry = True
            else:
                try:
                    json_data = read_json_file(str(json_file))
                    
                    # Check if JSON has required fields
                    if not isinstance(json_data, dict):
                        logger.warning(f"{nct_id}: JSON is not a dict")
                        needs_retry = True
                    elif 'Final Decision' not in json_data and 'Final_Decision' not in json_data:
                        logger.warning(f"{nct_id}: JSON missing Final Decision")
                        needs_retry = True
                    elif 'Recap' not in json_data and 'recap' not in json_data:
                        logger.warning(f"{nct_id}: JSON missing Recap")
                        needs_retry = True
                        
                except Exception as e:
                    logger.warning(f"{nct_id}: Invalid JSON - {e}")
                    needs_retry = True
            
            if needs_retry:
                # Check if txt file shows repetition/hallucination
                txt_content = read_text_file(str(txt_file))
                if txt_content:
                    full_text = '\n'.join(txt_content)
                    
                    # Detect repetition - same sentence 3+ times
                    if self._is_repetitive(full_text):
                        logger.error(f"{nct_id}: Detected repetitive hallucination")
                        failed_trials.append((nct_id, str(txt_file)))
                    # Detect truncation - no closing brace
                    elif '{' in full_text and full_text.count('{') > full_text.count('}'):
                        logger.error(f"{nct_id}: Detected truncated output")
                        failed_trials.append((nct_id, str(txt_file)))
                    else:
                        logger.warning(f"{nct_id}: Failed but unclear reason")
                        failed_trials.append((nct_id, str(txt_file)))
        
        return failed_trials
    
    def _is_repetitive(self, text: str) -> bool:
        """Detect if text has repetitive patterns (hallucination)."""
        lines = text.split('\n')
        
        # Check for exact line repetition
        if len(lines) > 10:
            line_counts = {}
            for line in lines:
                line = line.strip()
                if len(line) > 20:  # Only check substantial lines
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            # If any line appears 3+ times, it's likely repetition
            max_count = max(line_counts.values()) if line_counts else 0
            if max_count >= 3:
                return True
        
        # Check for phrase repetition
        phrases = re.findall(r"The patient hasn't \w+", text)
        if len(phrases) >= 5:
            phrase_counts = {}
            for phrase in phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            if max(phrase_counts.values()) >= 3:
                return True
        
        return False
    
    def retry_failed_trial(
        self, 
        nct_id: str, 
        trials_json_folder: str,
        patient_profile: List[str],
        output_folder: str,
        max_attempts: int = 3
    ) -> bool:
        """
        Retry CoT reasoning for a single failed trial.
        
        Returns:
            True if retry succeeded, False otherwise
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Retrying CoT for {nct_id} (max {max_attempts} attempts)")
        logger.info(f"{'='*70}")
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Attempt {attempt}/{max_attempts} for {nct_id}")
            
            try:
                # Process single trial
                self.cot_processor.process_trials(
                    nct_ids=[nct_id],
                    json_folder=trials_json_folder,
                    output_folder=output_folder,
                    patient_profile=patient_profile
                )
                
                # Check if it worked
                json_file = os.path.join(output_folder, f"{nct_id}.json")
                if os.path.exists(json_file):
                    json_data = read_json_file(json_file)
                    
                    # Validate
                    has_decision = ('Final Decision' in json_data or 
                                   'Final_Decision' in json_data)
                    has_recap = 'Recap' in json_data or 'recap' in json_data
                    
                    if has_decision and has_recap:
                        logger.info(f"✓ Retry succeeded for {nct_id} on attempt {attempt}")
                        return True
                    else:
                        logger.warning(f"✗ Retry produced incomplete JSON for {nct_id}")
                
            except Exception as e:
                logger.error(f"✗ Retry failed for {nct_id} on attempt {attempt}: {e}")
        
        logger.error(f"✗ All {max_attempts} retry attempts failed for {nct_id}")
        return False
    
    def update_ranked_trials(
        self,
        nct_id: str,
        cot_output_folder: str,
        ranked_trials_file: str
    ) -> None:
        """
        Insert the newly processed trial into ranked_trials.json.
        """
        # Load existing ranked trials
        ranked_data = read_json_file(ranked_trials_file)
        
        if isinstance(ranked_data, dict) and 'RankedTrials' in ranked_data:
            ranked_trials = ranked_data['RankedTrials']
        else:
            ranked_trials = ranked_data
        
        # Check if trial already exists
        existing_nct_ids = {t['TrialID'] for t in ranked_trials}
        if nct_id in existing_nct_ids:
            logger.info(f"{nct_id} already in ranked_trials.json, skipping insert")
            return
        
        # Load the trial's CoT output to get classification
        cot_file = os.path.join(cot_output_folder, f"{nct_id}.json")
        cot_data = read_json_file(cot_file)
        
        # Calculate score based on CoT decision
        final_decision = (cot_data.get('Final Decision') or 
                         cot_data.get('Final_Decision', '')).lower()
        
        # Assign score based on decision, following same logic as original ranking
        score = score_trial(cot_data)
        
        logger.info(f"Calculated score for {nct_id}: {score} (decision: {final_decision})")
        
        # Insert in sorted order
        new_trial = {'TrialID': nct_id, 'Score': score}
        
        # Find insertion point
        insert_idx = len(ranked_trials)
        for i, trial in enumerate(ranked_trials):
            if trial['Score'] < score:
                insert_idx = i
                break
        
        ranked_trials.insert(insert_idx, new_trial)
        
        logger.info(f"Inserted {nct_id} at position {insert_idx + 1}/{len(ranked_trials)}")
        
        # Save updated ranked trials
        if isinstance(ranked_data, dict) and 'RankedTrials' in ranked_data:
            ranked_data['RankedTrials'] = ranked_trials
            write_json_file(ranked_data, ranked_trials_file)
        else:
            write_json_file(ranked_trials, ranked_trials_file)

def retry_all_failed_patients(
    results_dir: str,
    trials_json_folder: str,
    patients_file: str,
    config: Dict,
    scenario_name: str = "fl-hybrid_rerank_cot-vllm_ner"
) -> Dict[str, List[str]]:
    """
    Scan all patients and retry failed CoT trials.
    
    Returns:
        Dict mapping patient_id -> list of failed trial NCT IDs
    """
    # Initialize CoT processor
    vllm_cfg = config.get('vllm', {})
    vllm_engine, tokenizer, lora_request = load_vllm_engine(config['model'], vllm_cfg)
    
    cot_processor = BatchTrialProcessorVLLM(
        llm=vllm_engine,
        tokenizer=tokenizer,
        batch_size=vllm_cfg.get('batch_size', 32),
        use_cot=True,
        max_new_tokens=config['cot'].get('max_new_tokens', 6000),
        temperature=vllm_cfg.get('temperature', 0.0),
        top_p=vllm_cfg.get('top_p', 1.0),
        seed=vllm_cfg.get('seed', 1234),
        length_bucket=vllm_cfg.get('length_bucket', True),
        lora_request=lora_request
    )
    
    retrier = FailedCoTRetrier(cot_processor)
    
    # Load patients
    patients_data = read_json_file(patients_file)
    
    results = {}
    
    for patient_id, patient_info in patients_data.items():
        patient_folder = os.path.join(results_dir, patient_id, scenario_name)
        
        if not os.path.exists(patient_folder):
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Checking patient {patient_id}")
        logger.info(f"{'='*80}")
        
        # Find failed trials
        failed_trials = retrier.find_failed_trials(patient_folder)
        
        if not failed_trials:
            logger.info(f"No failed trials for {patient_id}")
            continue
        
        logger.warning(f"Found {len(failed_trials)} failed trials for {patient_id}")
        
        # Get patient profile
        patient_profile = (patient_info.get('split_raw_description') or 
                          patient_info.get('expanded_sentences') or
                          [patient_info.get('raw_description', '')])
        
        # Retry each failed trial
        successfully_retried = []
        for nct_id, txt_file in failed_trials:
            success = retrier.retry_failed_trial(
                nct_id=nct_id,
                trials_json_folder=trials_json_folder,
                patient_profile=patient_profile,
                output_folder=patient_folder,
                max_attempts=3
            )
            
            if success:
                successfully_retried.append(nct_id)
                
                # Update ranked_trials.json
                ranked_file = os.path.join(patient_folder, "ranked_trials.json")
                if os.path.exists(ranked_file):
                    retrier.update_ranked_trials(nct_id, patient_folder, ranked_file)
        
        if successfully_retried:
            logger.info(f"✓ Successfully retried {len(successfully_retried)}/{len(failed_trials)} "
                       f"trials for {patient_id}")
            results[patient_id] = successfully_retried
        else:
            logger.warning(f"✗ Failed to retry any trials for {patient_id}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retry failed CoT reasoning")
    parser.add_argument("--results-dir", default="../data/lung_results")
    parser.add_argument("--trials-json", default="../data/lung_processed_trials")
    parser.add_argument("--patients-file", default="../data/final_lung_dataset_pipeline.json")
    parser.add_argument("--scenario", default="fl-hybrid_rerank_cot-vllm_ner")
    parser.add_argument("--patient-id", help="Retry specific patient only")
    
    args = parser.parse_args()
    
    config = load_config()
    
    if args.patient_id:
        # Retry single patient
        patients_data = {args.patient_id: read_json_file(args.patients_file)[args.patient_id]}
    else:
        # Retry all patients
        patients_data = read_json_file(args.patients_file)
    
    results = retry_all_failed_patients(
        results_dir=args.results_dir,
        trials_json_folder=args.trials_json,
        patients_file=args.patients_file,
        config=config,
        scenario_name=args.scenario
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("RETRY SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Patients with retried trials: {len(results)}")
    for patient_id, nct_ids in results.items():
        logger.info(f"  {patient_id}: {len(nct_ids)} trials - {nct_ids}")