# Matcher/targeted_evaluator.py
"""
Evaluate targeted patient-trial matching experiments using the same comprehensive
metrics as cluster evaluation.
"""

import json
import os
from typing import Dict, List, Optional, Set
from collections import defaultdict
import numpy as np

from Matcher.utils.logging_config import setup_logging

logger = setup_logging()


class TargetedExperimentEvaluator:
    """
    Evaluates targeted experiments where a patient generated for specific trials
    is matched against a large corpus (e.g., 10k trials).
    
    Uses the same comprehensive metrics as cluster evaluation:
    1. Exhaustiveness (recall@K over ground truth trials)
    2. Ranking quality (precision@K, nDCG@K)
    3. Missed opportunity analysis
    """
    
    def __init__(self, trials_folder: str):
        """
        Args:
            trials_folder: Path to folder containing trial JSON files
        """
        self.trials_folder = trials_folder
        self.trial_cache = {}
    
    def get_trial_data(self, trial_nct_id: str) -> Optional[Dict]:
        """Load and cache trial data."""
        if trial_nct_id in self.trial_cache:
            return self.trial_cache[trial_nct_id]
        
        trial_path = os.path.join(self.trials_folder, f"{trial_nct_id}.json")
        if not os.path.exists(trial_path):
            return None
        
        with open(trial_path, 'r') as f:
            trial_data = json.load(f)
        
        self.trial_cache[trial_nct_id] = trial_data
        return trial_data
    
    def extract_trial_countries(self, trial_data: Dict) -> List[str]:
        """Extract all countries from a trial."""
        countries = set()
        
        if 'protocolSection' in trial_data:
            contacts_locations = trial_data['protocolSection'].get('contactsLocationsModule', {})
            locations = contacts_locations.get('locations', [])
            
            for location in locations:
                country = location.get('country', '')
                if country:
                    countries.add(country)
        elif 'location' in trial_data:
            legacy_locations = trial_data['location']
            if isinstance(legacy_locations, list):
                for loc in legacy_locations:
                    address = loc.get('location_address', '')
                    country = address.split(',')[-1].strip() if address else ''
                    if country:
                        countries.add(country)
            else:
                country = legacy_locations.get('country', '')
                if country:
                    countries.add(country)
        
        return list(countries) if countries else []
    
    def is_trial_geographically_appropriate(self, trial_nct_id: str, 
                                           patient_country: str) -> Optional[bool]:
        """Check if trial operates in patient's country."""
        trial_data = self.get_trial_data(trial_nct_id)
        if not trial_data:
            return None
        
        trial_countries = self.extract_trial_countries(trial_data)
        if not trial_countries:
            return None
        
        return patient_country in trial_countries
    
    def calculate_dcg(self, relevances: List[float], k: Optional[int] = None) -> float:
        """Calculate Discounted Cumulative Gain."""
        if k:
            relevances = relevances[:k]
        
        dcg = 0.0
        for i, rel in enumerate(relevances, start=1):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    def calculate_ndcg(self, relevances: List[float], k: Optional[int] = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        dcg = self.calculate_dcg(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.calculate_dcg(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def analyze_missed_opportunities(self, patient_id: str, ranked_trials: List[Dict],
                                    ground_truth: Set[str], k: int = 10) -> Dict:
        """
        Analyze why ground truth trials were missed in top-K.
        
        Returns dict with:
        - missed_trials: List of ground truth trials not in top-K
        - reasons: Analysis of why each was missed
        """
        top_k_nct_ids = {t['nct_id'] for t in ranked_trials[:k] if 'nct_id' in t}
        missed_gt = ground_truth - top_k_nct_ids
        
        missed_analysis = []
        for nct_id in missed_gt:
            # Find where this trial ranked
            trial_rank = None
            trial_score = None
            for i, trial in enumerate(ranked_trials, start=1):
                if trial.get('nct_id') == nct_id:
                    trial_rank = i
                    trial_score = trial.get('score', trial.get('final_score'))
                    break
            
            reason = self._diagnose_miss_reason(nct_id, trial_rank, trial_score, len(ranked_trials), k)
            
            missed_analysis.append({
                'nct_id': nct_id,
                'actual_rank': trial_rank,
                'score': trial_score,
                'reason': reason
            })
        
        return {
            'missed_count': len(missed_gt),
            'missed_trials': list(missed_gt),
            'missed_analysis': missed_analysis
        }
    
    def _diagnose_miss_reason(self, nct_id: str, rank: Optional[int], 
                             score: Optional[float], total_trials: int, k: int) -> str:
        """Diagnose why a trial was missed."""
        if rank is None:
            return "NOT_RETRIEVED: Trial not found in any results (first-level search failure)"
        
        # Calculate relative position
        if total_trials > 0:
            relative_position = rank / total_trials
            if relative_position > 0.67:
                return f"LOW_RANK: Trial ranked in bottom third (rank {rank}/{total_trials})"
            elif relative_position > 0.33:
                return f"MID_RANK: Trial ranked in middle third (rank {rank}/{total_trials})"
            elif rank <= k + 10:
                return f"MARGINAL_MISS: Ranked {rank}, just outside top-{k}"
            else:
                return f"SECOND_LEVEL_FAILURE: Ranked {rank}, in top third but missed reasoning"
        
        return f"UNKNOWN: Ranked {rank}/{total_trials}"
    
    def evaluate_patient(self, patient_id: str, patient_country: str,
                        patient_biomarker: str, ground_truth_trials: List[str],
                        ranked_trials: List[Dict],
                        k_values: List[int] = [5, 10, 20, 50, 100]) -> Dict:
        """
        Comprehensive evaluation for a single patient.
        
        Args:
            patient_id: Patient identifier
            patient_country: Patient's country for geographic filtering
            patient_biomarker: Patient's biomarker
            ground_truth_trials: List of NCT IDs that are ground truth (the 3 target trials)
            ranked_trials: Ranked list of trial dicts from pipeline
            k_values: K values for metrics
        
        Returns:
            Dict with exhaustiveness, appropriateness, and ranking quality metrics
        """
        ground_truth = set(ground_truth_trials)
        
        if not ground_truth:
            return {
                'patient_id': patient_id,
                'error': 'No ground truth trials provided'
            }
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating patient {patient_id}")
        logger.info(f"Biomarker: {patient_biomarker}")
        logger.info(f"Ground truth trials: {ground_truth_trials}")
        logger.info(f"Total ranked trials: {len(ranked_trials)}")
        logger.info(f"{'='*70}\n")
        
        # Normalize trial format
        normalized_trials = []
        for i, trial in enumerate(ranked_trials, start=1):
            nct_id = trial.get('TrialID') or trial.get('nct_id')
            score = trial.get('Score') or trial.get('score') or trial.get('final_score')
            normalized_trials.append({
                'nct_id': nct_id,
                'rank': i,
                'score': score
            })
        
        ranked_nct_ids = [t['nct_id'] for t in normalized_trials if t['nct_id']]
        
        # 1. EXHAUSTIVENESS METRICS (over ground truth)
        gt_metrics = {}
        
        # Find ranks of all ground truth trials
        gt_ranks = []
        gt_found = set()
        for trial in normalized_trials:
            nct_id = trial['nct_id']
            if nct_id in ground_truth:
                gt_ranks.append(trial['rank'])
                gt_found.add(nct_id)
                logger.info(f"✓ Found ground truth trial {nct_id} at rank {trial['rank']}")
        
        gt_not_found = ground_truth - gt_found
        if gt_not_found:
            for nct_id in gt_not_found:
                logger.warning(f"✗ Ground truth trial {nct_id} NOT in ranked results")
        
        gt_metrics['gt_size'] = len(ground_truth)
        gt_metrics['gt_found_count'] = len(gt_found)
        gt_metrics['gt_missing_count'] = len(gt_not_found)
        gt_metrics['gt_missing_trials'] = list(gt_not_found)
        
        # Rank statistics
        if gt_ranks:
            gt_metrics['gt_median_rank'] = float(np.median(gt_ranks))
            gt_metrics['gt_mean_rank'] = float(np.mean(gt_ranks))
            gt_metrics['gt_best_rank'] = int(min(gt_ranks))
            gt_metrics['gt_worst_rank'] = int(max(gt_ranks))
            gt_metrics['gt_all_ranks'] = gt_ranks
        else:
            gt_metrics['gt_median_rank'] = None
            gt_metrics['gt_mean_rank'] = None
            gt_metrics['gt_best_rank'] = None
            gt_metrics['gt_worst_rank'] = None
            gt_metrics['gt_all_ranks'] = []
        
        # Recall@K over ground truth
        for k in k_values:
            top_k_nct_ids = set(ranked_nct_ids[:k])
            gt_in_top_k = len(ground_truth & top_k_nct_ids)
            gt_metrics[f'recall_at_{k}'] = gt_in_top_k / len(ground_truth) if ground_truth else 0.0
            gt_metrics[f'gt_count_at_{k}'] = gt_in_top_k
        
        # 2. Missed Opportunity Analysis
        missed_analysis = self.analyze_missed_opportunities(
            patient_id, normalized_trials, ground_truth, k=10
        )
        
        # 3. APPROPRIATENESS FILTERING (Geographic)
        geo_appropriate_trials = []
        geo_appropriate_in_gt = []
        
        for trial in normalized_trials:
            nct_id = trial['nct_id']
            if not nct_id:
                continue
            
            is_appropriate = self.is_trial_geographically_appropriate(nct_id, patient_country)
            
            if is_appropriate:
                geo_appropriate_trials.append(trial)
                if nct_id in ground_truth:
                    geo_appropriate_in_gt.append(trial)
        
        geo_metrics = {
            'patient_country': patient_country,
            'total_geo_appropriate': len(geo_appropriate_trials),
            'gt_geo_appropriate': len(geo_appropriate_in_gt),
        }
        
        # Recall@K for geographically appropriate ground truth trials
        if geo_appropriate_in_gt:
            for k in k_values:
                top_k_geo = set(t['nct_id'] for t in geo_appropriate_trials[:k])
                gt_geo_in_top_k = sum(1 for t in geo_appropriate_in_gt 
                                     if t['nct_id'] in top_k_geo)
                geo_metrics[f'geo_gt_recall_at_{k}'] = gt_geo_in_top_k / len(geo_appropriate_in_gt)
        
        # 4. RANKING QUALITY METRICS (p@K, nDCG@K)
        ranking_metrics = {}
        
        # Binary relevance: 1 if in ground truth, 0 otherwise
        relevances = [1.0 if t['nct_id'] in ground_truth else 0.0 for t in normalized_trials]
        
        # Precision@K
        for k in k_values:
            if k <= len(normalized_trials):
                top_k_nct_ids = set(ranked_nct_ids[:k])
                relevant_in_top_k = len(ground_truth & top_k_nct_ids)
                ranking_metrics[f'precision_at_{k}'] = relevant_in_top_k / k
            else:
                ranking_metrics[f'precision_at_{k}'] = 0.0
        
        # nDCG@K
        for k in k_values:
            ranking_metrics[f'ndcg_at_{k}'] = self.calculate_ndcg(relevances, k)
        
        # Overall nDCG
        ranking_metrics['ndcg'] = self.calculate_ndcg(relevances)
        
        # Geographic-filtered ranking quality
        geo_relevances = [1.0 if t['nct_id'] in ground_truth else 0.0 
                         for t in geo_appropriate_trials]
        
        for k in k_values:
            if k <= len(geo_appropriate_trials):
                geo_top_k = set(t['nct_id'] for t in geo_appropriate_trials[:k])
                geo_relevant = len(ground_truth & geo_top_k)
                ranking_metrics[f'geo_precision_at_{k}'] = geo_relevant / k if k > 0 else 0.0
            
            ranking_metrics[f'geo_ndcg_at_{k}'] = self.calculate_ndcg(geo_relevances, k)
        
        # 5. Top-K Analysis (what else ranked highly?)
        top_k_analysis = self._analyze_top_k(normalized_trials[:20], ground_truth)
        
        # Combine all metrics
        return {
            'patient_id': patient_id,
            'patient_biomarker': patient_biomarker,
            'patient_country': patient_country,
            'total_ranked_trials': len(normalized_trials),
            **gt_metrics,
            **geo_metrics,
            **ranking_metrics,
            'missed_opportunities': missed_analysis,
            'top_k_analysis': top_k_analysis
        }
    
    def _analyze_top_k(self, top_trials: List[Dict], ground_truth: Set[str]) -> Dict:
        """Analyze characteristics of top-K trials."""
        biomarkers = []
        statuses = []
        
        for trial in top_trials:
            nct_id = trial['nct_id']
            trial_data = self.get_trial_data(nct_id)
            
            if not trial_data:
                continue
            
            # Extract biomarker from conditions or title
            if 'protocolSection' in trial_data:
                conditions = trial_data['protocolSection'].get('conditionsModule', {}).get('conditions', [])
                title = trial_data['protocolSection'].get('identificationModule', {}).get('briefTitle', '')
                status = trial_data['protocolSection'].get('statusModule', {}).get('overallStatus', '')
            else:
                conditions = trial_data.get('conditions', [])
                title = trial_data.get('brief_title', '')
                status = ''
            
            # Check for biomarkers
            text = f"{title} {' '.join(conditions) if isinstance(conditions, list) else conditions}".upper()
            if 'KRAS' in text:
                biomarkers.append('KRAS')
            elif 'EGFR' in text:
                biomarkers.append('EGFR')
            elif 'ROS1' in text or 'ROS-1' in text:
                biomarkers.append('ROS1')
            elif 'ALK' in text:
                biomarkers.append('ALK')
            else:
                biomarkers.append('Other')
            
            if status:
                statuses.append(status)
        
        from collections import Counter
        
        return {
            'biomarker_distribution': dict(Counter(biomarkers)),
            'status_distribution': dict(Counter(statuses)),
            'num_ground_truth_in_top_20': sum(1 for t in top_trials if t['nct_id'] in ground_truth)
        }
    
    def run_experiment(
        self,
        patients_file: str,
        output_dir: str,
        trials_folder: str,
        scenario: str = 'fl-hybrid_no-rerank_cot-vllm_ner',
        k_values: List[int] = [5, 10, 20, 50, 100]
    ) -> Dict:
        """
        Run the full targeted experiment evaluation.
        
        Args:
            patients_file: JSON file with generated patients (containing ground_truth_trials)
            output_dir: Base directory containing per-patient results folders
            trials_folder: Folder containing trial JSON files for metadata lookup
            k_values: K values for evaluation metrics
        
        Returns:
            Dict with evaluation results
        """
        logger.info(f"\n{'='*70}")
        logger.info("TARGETED TRIAL MATCHING EXPERIMENT EVALUATION")
        logger.info(f"{'='*70}\n")
        
        # Load patients
        logger.info(f"Loading patients from {patients_file}...")
        with open(patients_file, 'r') as f:
            patients_data = json.load(f)
        
        # Convert to list
        if isinstance(patients_data, dict):
            patients = list(patients_data.values())
        else:
            patients = patients_data
        
        logger.info(f"Loaded {len(patients)} patient(s)")
        
        # Create results directory
        results_dir = os.path.join(output_dir, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluate each patient
        all_results = []
        for patient in patients:
            patient_id = patient.get('patient_id', 'P001')
            logger.info(f"\nProcessing patient {patient_id}...")
            
            # Load patient's ranked trials
            ranked_trials_path = os.path.join(output_dir, patient_id, scenario, "ranked_trials.json")
            if not os.path.exists(ranked_trials_path):
                logger.warning(f"No ranked trials file found for patient {patient_id} at {ranked_trials_path}")
                continue
                
            try:
                with open(ranked_trials_path, 'r') as f:
                    ranked_data = json.load(f)
                
                # Handle both formats
                if isinstance(ranked_data, dict) and 'RankedTrials' in ranked_data:
                    ranked_trials = ranked_data['RankedTrials']
                else:
                    ranked_trials = ranked_data
                    
                logger.info(f"Loaded {len(ranked_trials)} ranked trials for patient {patient_id}")
                
            except Exception as e:
                logger.error(f"Error loading ranked trials for patient {patient_id}: {e}")
                continue
            
            # Get patient details
            ground_truth = patient.get('ground_truth_trials', patient.get('target_trials', []))
            patient_country = patient.get('location', {}).get('country', 'United States')
            patient_biomarker = patient.get('biomarker', 'KRAS')
            
            # Evaluate this patient
            result = self.evaluate_patient(
                patient_id=patient_id,
                patient_country=patient_country,
                patient_biomarker=patient_biomarker, 
                ground_truth_trials=ground_truth,
                ranked_trials=ranked_trials,
                k_values=k_values
            )
            
            all_results.append(result)
            
            # Print summary for this patient
            self._print_patient_summary(result)
            
            # Save individual patient results
            patient_results_file = os.path.join(results_dir, f"{patient_id}_evaluation.json")
            with open(patient_results_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save overall results
        results_file = os.path.join(results_dir, 'all_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✓ Saved detailed results to {results_file}")
        
        # Create summary report
        summary = self._create_summary(all_results, k_values)
        summary_file = os.path.join(results_dir, 'summary_report.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved summary report to {summary_file}")
                
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*70}\n")
        
        return {
            'patient_results': all_results,
            'summary': summary
        }
    
    def _print_patient_summary(self, result: Dict):
        """Print summary for one patient."""
        logger.info(f"\n{'-'*70}")
        logger.info(f"Results for {result['patient_id']} ({result['patient_biomarker']})")
        logger.info(f"{'-'*70}")
        
        logger.info(f"\nGround Truth Rankings:")
        gt_size = result['gt_size']
        gt_found = result['gt_found_count']
        logger.info(f"  Found {gt_found}/{gt_size} ground truth trials in results")
        
        if result['gt_all_ranks']:
            for i, rank in enumerate(result['gt_all_ranks'], 1):
                logger.info(f"  Ground truth trial #{i}: Rank {rank}")
        
        if result['gt_missing_trials']:
            logger.info(f"  Missing: {result['gt_missing_trials']}")
        
        logger.info(f"\nExhaustiveness Metrics:")
        logger.info(f"  Best rank: {result['gt_best_rank']}")
        logger.info(f"  Median rank: {result['gt_median_rank']:.1f}" if result['gt_median_rank'] else "  Median rank: N/A")
        logger.info(f"  Mean rank: {result['gt_mean_rank']:.1f}" if result['gt_mean_rank'] else "  Mean rank: N/A")
        
        logger.info(f"\nRecall@K:")
        for k in [5, 10, 20, 50, 100]:
            if f'recall_at_{k}' in result:
                logger.info(f"  Recall@{k}: {result[f'recall_at_{k}']:.1%} ({result[f'gt_count_at_{k}']}/{gt_size})")
        
        logger.info(f"\nRanking Quality:")
        for k in [10, 20]:
            if f'precision_at_{k}' in result:
                logger.info(f"  P@{k}: {result[f'precision_at_{k}']:.3f}")
            if f'ndcg_at_{k}' in result:
                logger.info(f"  nDCG@{k}: {result[f'ndcg_at_{k}']:.3f}")
    
    def _create_summary(self, all_results: List[Dict], k_values: List[int]) -> Dict:
        """Create summary statistics across all patients."""
        summary = {
            'total_patients': len(all_results),
            'metrics': {}
        }
        
        # Aggregate metrics
        for k in k_values:
            recalls = [r[f'recall_at_{k}'] for r in all_results if f'recall_at_{k}' in r]
            precisions = [r[f'precision_at_{k}'] for r in all_results if f'precision_at_{k}' in r]
            ndcgs = [r[f'ndcg_at_{k}'] for r in all_results if f'ndcg_at_{k}' in r]
            
            if recalls:
                summary['metrics'][f'mean_recall_at_{k}'] = float(np.mean(recalls))
            if precisions:
                summary['metrics'][f'mean_precision_at_{k}'] = float(np.mean(precisions))
            if ndcgs:
                summary['metrics'][f'mean_ndcg_at_{k}'] = float(np.mean(ndcgs))
        
        # Rank statistics
        all_median_ranks = [r['gt_median_rank'] for r in all_results if r.get('gt_median_rank')]
        if all_median_ranks:
            summary['metrics']['mean_median_rank'] = float(np.mean(all_median_ranks))
            summary['metrics']['median_median_rank'] = float(np.median(all_median_ranks))
        
        return summary
    
    def _print_patient_summary(self, result: Dict):
        """Print summary for one patient."""
        logger.info(f"\n{'-'*70}")
        logger.info(f"Results for {result['patient_id']} ({result['patient_biomarker']})")
        logger.info(f"{'-'*70}")
        
        logger.info(f"\nGround Truth Rankings:")
        gt_size = result['gt_size']
        gt_found = result['gt_found_count']
        logger.info(f"  Found {gt_found}/{gt_size} ground truth trials in results")
        
        if result['gt_all_ranks']:
            for i, rank in enumerate(result['gt_all_ranks'], 1):
                logger.info(f"  Ground truth trial #{i}: Rank {rank}")
        
        if result['gt_missing_trials']:
            logger.info(f"  Missing: {result['gt_missing_trials']}")
        
        logger.info(f"\nExhaustiveness Metrics:")
        logger.info(f"  Best rank: {result['gt_best_rank']}")
        logger.info(f"  Median rank: {result['gt_median_rank']:.1f}" if result['gt_median_rank'] else "  Median rank: N/A")
        logger.info(f"  Mean rank: {result['gt_mean_rank']:.1f}" if result['gt_mean_rank'] else "  Mean rank: N/A")
        
        logger.info(f"\nRecall@K:")
        for k in [5, 10, 20, 50, 100]:
            if f'recall_at_{k}' in result:
                logger.info(f"  Recall@{k}: {result[f'recall_at_{k}']:.1%} ({result[f'gt_count_at_{k}']}/{gt_size})")
        
        logger.info(f"\nGeographic Appropriateness:")
        logger.info(f"  Patient country: {result.get('patient_country', 'Unknown')}")
        logger.info(f"  Total geo-appropriate trials: {result.get('total_geo_appropriate', 0)}")
        logger.info(f"  GT trials geo-appropriate: {result.get('gt_geo_appropriate', 0)}/{gt_size}")
        
        # Show geo-filtered recall if available
        has_geo_recall = any(f'geo_gt_recall_at_{k}' in result for k in [5, 10, 20])
        if has_geo_recall:
            logger.info(f"\n  Geo-filtered Recall@K (GT trials in patient's country):")
            for k in [5, 10, 20, 50]:
                key = f'geo_gt_recall_at_{k}'
                if key in result:
                    logger.info(f"    Geo-Recall@{k}: {result[key]:.1%}")
        else:
            logger.warning(f"  ⚠️ No geo-appropriate GT trials found in patient's country")
        
        logger.info(f"\nRanking Quality:")
        for k in [10, 20]:
            if f'precision_at_{k}' in result:
                logger.info(f"  P@{k}: {result[f'precision_at_{k}']:.3f}")
            if f'ndcg_at_{k}' in result:
                logger.info(f"  nDCG@{k}: {result[f'ndcg_at_{k}']:.3f}")
        
        has_geo_precision = any(f'geo_precision_at_{k}' in result for k in [10, 20])
        if has_geo_precision:
            logger.info(f"\n  Geo-filtered Ranking Quality:")
            for k in [10, 20]:
                geo_prec_key = f'geo_precision_at_{k}'
                geo_ndcg_key = f'geo_ndcg_at_{k}'
                if geo_prec_key in result:
                    logger.info(f"    Geo-P@{k}: {result[geo_prec_key]:.3f}")
                if geo_ndcg_key in result:
                    logger.info(f"    Geo-nDCG@{k}: {result[geo_ndcg_key]:.3f}")



# Modify main() to match new interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate targeted patient-trial matching experiment"
    )
    parser.add_argument(
        '--patients-file',
        type=str,
        required=True,
        help='JSON file with generated patients (from generate_targeted_patients.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Base directory containing per-patient results folders'
    )
    parser.add_argument(
        '--trials-folder',
        type=str,
        required=True,
        help='Folder containing trial JSON files for metadata lookup'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='fl-hybrid_no-rerank_cot-vllm_ner',
        help='Experiment scenario name (for logging purposes)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    evaluator = TargetedExperimentEvaluator(args.trials_folder)
    evaluator.run_experiment(
        patients_file=args.patients_file,
        output_dir=args.output_dir,
        trials_folder=args.trials_folder,
        scenario=args.scenario
    )