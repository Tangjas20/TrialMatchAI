import json
import os
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
from pathlib import Path


class ComprehensiveMatchingEvaluator:
    """
    Evaluates trial matching on:
    1. Ground Truth ETS (Eligible Trial Set) definition
    2. Exhaustiveness (recall@K, median rank, missed opportunities)
    3. Appropriateness filters (geographic availability)
    4. Ranking quality (p@K, nDCG@K)
    """
    
    def __init__(self, trials_folder: str, cluster_metadata_file: str):
        """
        Args:
            trials_folder: Path to folder containing trial JSON files
            cluster_metadata_file: CSV with Trial NCT ID, Cluster ID, Biomarker columns
        """
        self.trials_folder = trials_folder
        self.trial_cache = {}
        
        # Load cluster metadata to define ground truth ETS
        import pandas as pd
        self.cluster_df = pd.read_csv(cluster_metadata_file)
        self.biomarker_to_trials = self._build_ets_mapping()
    
    def _build_ets_mapping(self) -> Dict[str, Set[str]]:
        """Build ground truth ETS: biomarker -> set of NCT IDs."""
        mapping = defaultdict(set)
        for _, row in self.cluster_df.iterrows():
            biomarker = row['Biomarker']
            nct_id = row['Trial NCT ID']
            mapping[biomarker].add(nct_id)
        return dict(mapping)
    
    def get_ground_truth_ets(self, patient_biomarker: str) -> Set[str]:
        """Get Ground Truth Eligible Trial Set for a patient's biomarker."""
        return self.biomarker_to_trials.get(patient_biomarker, set())
    
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
            if isinstance(legacy_locations, list): #list of dicts containing two keys each : 'location_name' and 'location_address'. Country is the last delimited string in address
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
        """Check if trial operates in patient's country (appropriateness filter)."""
        trial_data = self.get_trial_data(trial_nct_id)
        if not trial_data:
            return None
        
        trial_countries = self.extract_trial_countries(trial_data)
        if not trial_countries:
            return None
        
        #print(f"Checking appropriateness for trial {trial_nct_id} in country {patient_country}. Trial countries: {trial_countries}")
        
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
                                    ets: Set[str], k: int = 10) -> Dict:
        """
        Analyze why ETS trials were missed in top-K.
        
        Returns dict with:
        - missed_trials: List of ETS trials not in top-K
        - reasons: Analysis of why each was missed
        """
        top_k_nct_ids = {t['nct_id'] for t in ranked_trials[:k] if 'nct_id' in t}
        missed_ets = ets - top_k_nct_ids
        
        missed_analysis = []
        for nct_id in missed_ets:
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
            'missed_count': len(missed_ets),
            'missed_trials': list(missed_ets),
            'missed_analysis': missed_analysis
        }
    
    def _diagnose_miss_reason(self, nct_id: str, rank: Optional[int], 
                             score: Optional[float], total_trials: int, k: int) -> str:
        """Diagnose why a trial was missed."""
        if rank is None:
            return "NOT_RETRIEVED: Trial not found in any results (likely first-level search failure)"
        
        # Calculate relative position
        relative_position = rank / total_trials # Assuming 33 trials total
        if relative_position > 0.67:
            return "LOW_RANK: Trial ranked in bottom third"
        elif relative_position > 0.33:
            return "COT_REASONING_FAILURE"
        elif rank <= k + 5: #missed by small margin
            return f"MARGINAL_MISS: Ranked {rank}, just outside top-{k}"
        
        else:
            return "SECOND_LEVEL_FAILURE"
    
    def evaluate_patient(self, patient_id: str, patient_country: str,
                        patient_biomarker: str, ranked_trials: List[Dict],
                        k_values: List[int] = [5, 10, 20, 33]) -> Dict:
        """
        Comprehensive evaluation for a single patient.
        
        Returns metrics for:
        1. Exhaustiveness (recall@K over ETS)
        2. Appropriateness (geographic filtering)
        3. Ranking quality (p@K, nDCG@K)
        """
        # 1. Get Ground Truth ETS
        ets = self.get_ground_truth_ets(patient_biomarker)
        
        if not ets:
            return {
                'patient_id': patient_id,
                'error': f'No ETS found for biomarker {patient_biomarker}'
            }
        
        # Normalize trial format: handle both {nct_id, rank} and {TrialID, Score}
        # Add explicit rank based on position
        normalized_trials = []
        for i, trial in enumerate(ranked_trials, start=1):
            nct_id = trial.get('TrialID') or trial.get('nct_id')
            score = trial.get('Score') or trial.get('score')
            normalized_trials.append({
                'nct_id': nct_id,
                'rank': i,
                'score': score
            })
        
        # Extract NCT IDs from ranked trials
        ranked_nct_ids = [t['nct_id'] for t in normalized_trials if t['nct_id']]
        
        # 2. EXHAUSTIVENESS METRICS (over ETS)
        ets_metrics = {}
        
        # Find ranks of all ETS trials
        ets_ranks = []
        ets_found = set()
        for trial in normalized_trials:
            nct_id = trial['nct_id']
            if nct_id in ets:
                ets_ranks.append(trial['rank'])
                ets_found.add(nct_id)
        
        ets_metrics['ets_size'] = len(ets)
        ets_metrics['ets_found_count'] = len(ets_found)
        ets_metrics['ets_missing_count'] = len(ets - ets_found)
        #print(f"Patient {patient_id}: ETS size {len(ets)}, found {len(ets_found)}, missing {len(ets - ets_found)}")
        
        # Median rank of ETS trials
        if ets_ranks:
            ets_metrics['ets_median_rank'] = float(np.median(ets_ranks))
            ets_metrics['ets_mean_rank'] = float(np.mean(ets_ranks))
            ets_metrics['ets_best_rank'] = int(min(ets_ranks))
            ets_metrics['ets_worst_rank'] = int(max(ets_ranks))
        else:
            ets_metrics['ets_median_rank'] = None
            ets_metrics['ets_mean_rank'] = None
        
        # Recall@K over ETS
        for k in k_values:
            top_k_nct_ids = set(ranked_nct_ids[:k])
            ets_in_top_k = len(ets & top_k_nct_ids)
            ets_metrics[f'recall_at_{k}'] = ets_in_top_k / len(ets) if ets else 0.0
            ets_metrics[f'ets_count_at_{k}'] = ets_in_top_k
        
        # 3. Missed Opportunity Analysis
        missed_analysis = self.analyze_missed_opportunities(
            patient_id, normalized_trials, ets, k=10
        )
        
        # 4. APPROPRIATENESS FILTERING (Geographic)
        geo_appropriate_trials = []
        geo_appropriate_in_ets = []
        
        for trial in normalized_trials:
            nct_id = trial['nct_id']
            if not nct_id:
                continue
            
            is_appropriate = self.is_trial_geographically_appropriate(nct_id, patient_country)
            #print(f"Trial {nct_id} appropriateness for patient in {patient_country}: {is_appropriate}")
            
            if is_appropriate:
                geo_appropriate_trials.append(trial)
                if nct_id in ets:
                    geo_appropriate_in_ets.append(trial)
        
        geo_metrics = {
            'patient_country': patient_country,
            'total_geo_appropriate': len(geo_appropriate_trials),
            'ets_geo_appropriate': len(geo_appropriate_in_ets),
        }
        
        # Recall@K for geographically appropriate ETS trials
        if geo_appropriate_in_ets:
            for k in k_values:
                top_k_geo = set(t['nct_id'] for t in geo_appropriate_trials[:k])
                ets_geo_in_top_k = sum(1 for t in geo_appropriate_in_ets 
                                      if t['nct_id'] in top_k_geo)
                geo_metrics[f'geo_ets_recall_at_{k}'] = ets_geo_in_top_k / len(geo_appropriate_in_ets)
        
        # 5. RANKING QUALITY METRICS (p@K, nDCG@K)
        ranking_metrics = {}
        
        # Binary relevance: 1 if in ETS, 0 otherwise
        relevances = [1.0 if t['nct_id'] in ets else 0.0 for t in normalized_trials]
        
        # Precision@K
        for k in k_values:
            top_k_nct_ids = set(ranked_nct_ids[:k])
            relevant_in_top_k = len(ets & top_k_nct_ids)
            ranking_metrics[f'precision_at_{k}'] = relevant_in_top_k / k if k <= len(normalized_trials) else 0.0
        
        # nDCG@K
        for k in k_values:
            ranking_metrics[f'ndcg_at_{k}'] = self.calculate_ndcg(relevances, k)
        
        # Overall nDCG (no cutoff)
        ranking_metrics['ndcg'] = self.calculate_ndcg(relevances)
        
        # Geographic-filtered ranking quality (appropriateness-aware metrics)
        geo_relevances = [1.0 if t['nct_id'] in ets else 0.0 
                         for t in geo_appropriate_trials]
        
        for k in k_values:
            # Precision@K for geo-appropriate trials
            if k <= len(geo_appropriate_trials):
                geo_top_k = set(t['nct_id'] for t in geo_appropriate_trials[:k])
                geo_relevant = len(ets & geo_top_k)
                ranking_metrics[f'geo_precision_at_{k}'] = geo_relevant / k
            
            # nDCG@K for geo-appropriate trials
            ranking_metrics[f'geo_ndcg_at_{k}'] = self.calculate_ndcg(geo_relevances, k)
        
        # Combine all metrics
        return {
            'patient_id': patient_id,
            'patient_biomarker': patient_biomarker,
            'patient_country': patient_country,
            'total_ranked_trials': len(normalized_trials),
            **ets_metrics,
            **geo_metrics,
            **ranking_metrics,
            'missed_opportunities': missed_analysis
        }
    
    def evaluate_dataset(self, results_dir: str, patients_file: str) -> Dict:
        """
        Evaluate entire dataset with comprehensive metrics.
        """
        # Load patients data
        with open(patients_file, 'r') as f:
            patients_data = json.load(f)
        
        # Handle both cluster format and flat format
        all_patients = {}
        if isinstance(patients_data, dict):
            for key, value in patients_data.items():
                if key.startswith('cluster_') and 'patients' in value:
                    # Cluster format - extract biomarker from cluster
                    cluster_biomarker = value.get('biomarker')
                    for patient in value['patients']:
                        patient_id = patient.get('patient_id')
                        if patient_id:
                            # Add cluster-level biomarker to patient if not present
                            if 'biomarker' not in patient and cluster_biomarker is not None:
                                patient['biomarker'] = cluster_biomarker
                            all_patients[patient_id] = patient
                else:
                    all_patients = patients_data
                    break
        
        all_metrics = []
        cluster_metrics = defaultdict(list)
        
        for patient_id, patient_info in all_patients.items():
            # Map cluster-based IDs to sequential folder IDs
            # KRAS-P001 to KRAS-P100 -> P001 to P100
            # EGFR-P001 to EGFR-P100 -> P101 to P200
            # ROS1-P001 to ROS1-P100 -> P201 to P300
            
            if '-P' in patient_id:
                # Format is "BIOMARKER-P###"
                parts = patient_id.split('-P')
                biomarker_prefix = parts[0]
                patient_num = int(parts[1])
                
                # Map to sequential ID based on biomarker cluster offset
                biomarker_to_offset = {'KRAS': 0, 'EGFR': 100, 'ROS1': 200}
                offset = biomarker_to_offset.get(biomarker_prefix, 0)
                sequential_id = offset + patient_num
                normalized_id = f"P{sequential_id:03d}"
            else:
                # Already in sequential P### format
                normalized_id = patient_id
            location = patient_info.get('location', {})
            patient_country = location.get('country')
            patient_biomarker = patient_info.get('biomarker')
            
            if not patient_country or not patient_biomarker:
                print(f"Warning: Missing data for {patient_id}, skipping")
                continue
            
            # Load ranked trials - handle nested folder structure
            # Try: results_dir/normalized_id/fl-hybrid_rerank_cot-vllm_ner/ranked_trials.json
            ranked_file = None
            possible_paths = [
                os.path.join(results_dir, normalized_id, "fl-hybrid_rerank_cot-vllm_ner", "ranked_trials.json"),
                os.path.join(results_dir, normalized_id, "ranked_trials.json"),
                os.path.join(results_dir, normalized_id, "fl-hybrid_rerank_cot-vllm", "ranked_trials.json"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    ranked_file = path
                    break
            
            if not ranked_file:
                print(f"Warning: No ranked trials found for {patient_id} in any expected location, skipping")
                continue
            
            with open(ranked_file, 'r') as f:
                ranked_data = json.load(f)
            
            # Handle both direct list and {"RankedTrials": [...]} format
            if isinstance(ranked_data, dict) and 'RankedTrials' in ranked_data:
                ranked_trials = ranked_data['RankedTrials']
            else:
                ranked_trials = ranked_data
            
            # Evaluate this patient
            patient_metrics = self.evaluate_patient(
                patient_id, patient_country, patient_biomarker, ranked_trials
            )
            
            if 'error' not in patient_metrics:
                all_metrics.append(patient_metrics)
                cluster_metrics[patient_biomarker].append(patient_metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics, cluster_metrics)
        
        return aggregated
    
    def _aggregate_metrics(self, all_metrics: List[Dict], 
                          cluster_metrics: Dict[str, List[Dict]]) -> Dict:
        """Aggregate metrics across patients."""
        
        def safe_mean(values):
            valid = [v for v in values if v is not None]
            return float(np.mean(valid)) if valid else None
        
        def safe_median(values):
            valid = [v for v in values if v is not None]
            return float(np.median(valid)) if valid else None
        
        def safe_std(values):
            valid = [v for v in values if v is not None]
            return float(np.std(valid)) if valid else None
        
        # Aggregate all metric types
        aggregated = {
            'total_patients': len(all_metrics),
            'overall_metrics': {},
            'cluster_metrics': {},
            'patient_level_metrics': all_metrics,
            'individual_values': {}  # For boxplot generation
        }
        
        # 1. Exhaustiveness metrics
        aggregated['overall_metrics']['exhaustiveness'] = {
            'mean_ets_size': safe_mean([m['ets_size'] for m in all_metrics]),
            'mean_ets_found_count': safe_mean([m['ets_found_count'] for m in all_metrics]),
            'mean_ets_median_rank': safe_mean([m['ets_median_rank'] for m in all_metrics]),
            'median_ets_median_rank': safe_median([m['ets_median_rank'] for m in all_metrics]),
            'mean_ets_mean_rank': safe_mean([m['ets_mean_rank'] for m in all_metrics]),
        }
        
        # Store individual values for boxplots
        aggregated['individual_values']['ets_median_ranks'] = [m['ets_median_rank'] for m in all_metrics if m.get('ets_median_rank') is not None]
        
        # Recall@K with both mean and median
        for k in [5, 10, 20, 33]:
            recall_key = f'recall_at_{k}'
            recalls = [m[recall_key] for m in all_metrics if recall_key in m]
            aggregated['overall_metrics']['exhaustiveness'][f'mean_{recall_key}'] = safe_mean(recalls)
            aggregated['overall_metrics']['exhaustiveness'][f'median_{recall_key}'] = safe_median(recalls)
            aggregated['overall_metrics']['exhaustiveness'][f'std_{recall_key}'] = safe_std(recalls)
            
            # Store for boxplot
            aggregated['individual_values'][recall_key] = recalls
        
        # 2. Appropriateness metrics
        aggregated['overall_metrics']['appropriateness'] = {
            'mean_total_geo_appropriate': safe_mean([m['total_geo_appropriate'] for m in all_metrics]),
            'mean_ets_geo_appropriate': safe_mean([m['ets_geo_appropriate'] for m in all_metrics]),
        }
        
        for k in [5, 10, 20]:
            geo_recall_key = f'geo_ets_recall_at_{k}'
            geo_recalls = [m.get(geo_recall_key) for m in all_metrics if m.get(geo_recall_key) is not None]
            if geo_recalls:
                aggregated['overall_metrics']['appropriateness'][f'mean_{geo_recall_key}'] = safe_mean(geo_recalls)
                aggregated['overall_metrics']['appropriateness'][f'median_{geo_recall_key}'] = safe_median(geo_recalls)
        
        # 3. Ranking quality metrics
        aggregated['overall_metrics']['ranking_quality'] = {}
        
        # Precision@K and nDCG@K with mean and median
        for k in [5, 10, 20, 33]:
            prec_key = f'precision_at_{k}'
            ndcg_key = f'ndcg_at_{k}'
            
            precs = [m[prec_key] for m in all_metrics if prec_key in m]
            ndcgs = [m[ndcg_key] for m in all_metrics if ndcg_key in m]
            
            aggregated['overall_metrics']['ranking_quality'][f'mean_{prec_key}'] = safe_mean(precs)
            aggregated['overall_metrics']['ranking_quality'][f'median_{prec_key}'] = safe_median(precs)
            aggregated['overall_metrics']['ranking_quality'][f'mean_{ndcg_key}'] = safe_mean(ndcgs)
            aggregated['overall_metrics']['ranking_quality'][f'median_{ndcg_key}'] = safe_median(ndcgs)
            
            # Store for boxplots
            aggregated['individual_values'][prec_key] = precs
            aggregated['individual_values'][ndcg_key] = ndcgs
            
            # Geographic-filtered metrics
            geo_prec_key = f'geo_precision_at_{k}'
            geo_ndcg_key = f'geo_ndcg_at_{k}'
            
            geo_precs = [m.get(geo_prec_key) for m in all_metrics if m.get(geo_prec_key) is not None]
            geo_ndcgs = [m.get(geo_ndcg_key) for m in all_metrics if m.get(geo_ndcg_key) is not None]
            
            if geo_precs:
                aggregated['overall_metrics']['ranking_quality'][f'mean_{geo_prec_key}'] = safe_mean(geo_precs)
                aggregated['overall_metrics']['ranking_quality'][f'median_{geo_prec_key}'] = safe_median(geo_precs)
                aggregated['individual_values'][geo_prec_key] = geo_precs
            if geo_ndcgs:
                aggregated['overall_metrics']['ranking_quality'][f'mean_{geo_ndcg_key}'] = safe_mean(geo_ndcgs)
                aggregated['overall_metrics']['ranking_quality'][f'median_{geo_ndcg_key}'] = safe_median(geo_ndcgs)
                aggregated['individual_values'][geo_ndcg_key] = geo_ndcgs
        
        # 4. Missed opportunity analysis
        all_missed_reasons = defaultdict(int)
        for m in all_metrics:
            for missed in m['missed_opportunities']['missed_analysis']:
                reason = missed['reason'].split(':')[0]  # Get category
                all_missed_reasons[reason] += 1
        
        total_missed = sum(all_missed_reasons.values())
        aggregated['overall_metrics']['missed_opportunities'] = {
            'total_missed_trials': total_missed,
            'missed_by_reason': dict(all_missed_reasons),
            'missed_reasons_percentage': {
                reason: (count / total_missed * 100) if total_missed > 0 else 0
                for reason, count in all_missed_reasons.items()
            }
        }
        
        # Per-cluster aggregation
        for biomarker, metrics_list in cluster_metrics.items():
            aggregated['cluster_metrics'][biomarker] = {
                'patient_count': len(metrics_list),
                'mean_recall_at_10': safe_mean([m.get('recall_at_10') for m in metrics_list]),
                'mean_ndcg_at_10': safe_mean([m.get('ndcg_at_10') for m in metrics_list]),
                'mean_ets_median_rank': safe_mean([m['ets_median_rank'] for m in metrics_list]),
                'mean_geo_precision_at_10': safe_mean([m.get('geo_precision_at_10') for m in metrics_list 
                                                       if m.get('geo_precision_at_10') is not None]),
            }
        
        return aggregated
    
    def save_evaluation(self, results_dir: str, patients_file: str, 
                       output_file: str = None):
        """Run evaluation and save results with comprehensive reporting."""
        if output_file is None:
            output_file = os.path.join(results_dir, "comprehensive_evaluation.json")
        
        evaluation_results = self.evaluate_dataset(results_dir, patients_file)
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Print comprehensive report
        self._print_report(evaluation_results)
        
        print(f"\nResults saved to: {output_file}")
        return evaluation_results
    
    def _print_report(self, results: Dict):
        """Print comprehensive evaluation report."""
        print(f"\n{'='*80}")
        print(f"{'COMPREHENSIVE TRIAL MATCHING EVALUATION':^80}")
        print(f"{'='*80}")
        print(f"\nTotal patients evaluated: {results['total_patients']}")
        
        overall = results['overall_metrics']
        
        # 1. EXHAUSTIVENESS
        print(f"\n{'--- 1. EXHAUSTIVENESS (Coverage of Ground Truth ETS) ---':^80}")
        exh = overall['exhaustiveness']
        print(f"Mean ETS size: {exh['mean_ets_size']:.1f} trials")
        print(f"Mean ETS trials found: {exh['mean_ets_found_count']:.1f}")
        print(f"Mean ETS median rank: {exh.get('mean_ets_median_rank', 0):.1f}")
        print(f"Median ETS median rank: {exh.get('median_ets_median_rank', 0):.1f}")
        print(f"\nRecall@K over ETS (Mean / Median):")
        for k in [5, 10, 20, 33]:
            mean_recall = exh.get(f'mean_recall_at_{k}', 0)
            median_recall = exh.get(f'median_recall_at_{k}', 0)
            std = exh.get(f'std_recall_at_{k}', 0)
            print(f"  Recall@{k:2d}: {mean_recall:.1%} / {median_recall:.1%} (±{std:.1%})")
        
        # 2. MISSED OPPORTUNITIES
        print(f"\n{'--- 2. MISSED OPPORTUNITY ANALYSIS ---':^80}")
        missed = overall['missed_opportunities']
        print(f"Total missed ETS trials in top-10: {missed['total_missed_trials']}")
        print(f"\nBreakdown by reason:")
        for reason, pct in missed['missed_reasons_percentage'].items():
            count = missed['missed_by_reason'][reason]
            print(f"  {reason:30s}: {count:4d} ({pct:5.1f}%)")
        
        # 3. APPROPRIATENESS
        print(f"\n{'--- 3. APPROPRIATENESS (Geographic Filtering) ---':^80}")
        app = overall['appropriateness']
        print(f"Mean geographically appropriate trials: {app['mean_total_geo_appropriate']:.1f}")
        print(f"Mean geo-appropriate ETS trials: {app['mean_ets_geo_appropriate']:.1f}")
        print(f"\nRecall@K for geo-appropriate ETS trials (Mean / Median):")
        for k in [5, 10, 20]:
            mean_geo_recall = app.get(f'mean_geo_ets_recall_at_{k}')
            median_geo_recall = app.get(f'median_geo_ets_recall_at_{k}')
            if mean_geo_recall is not None:
                median_str = f" / {median_geo_recall:.1%}" if median_geo_recall else ""
                print(f"  Geo-ETS-Recall@{k:2d}: {mean_geo_recall:.1%}{median_str}")
        
        # 4. RANKING QUALITY
        print(f"\n{'--- 4. RANKING QUALITY (Precision & nDCG) ---':^80}")
        rank = overall['ranking_quality']
        print(f"\nStandard metrics (all trials) - Mean / Median:")
        for k in [5, 10, 20]:
            mean_prec = rank.get(f'mean_precision_at_{k}', 0)
            median_prec = rank.get(f'median_precision_at_{k}', 0)
            mean_ndcg = rank.get(f'mean_ndcg_at_{k}', 0)
            median_ndcg = rank.get(f'median_ndcg_at_{k}', 0)
            print(f"  P@{k:2d}: {mean_prec:.3f} / {median_prec:.3f}  |  nDCG@{k:2d}: {mean_ndcg:.3f} / {median_ndcg:.3f}")
        
        print(f"\nAppropriateness-filtered metrics (geo-appropriate only) - Mean / Median:")
        for k in [5, 10, 20]:
            mean_geo_prec = rank.get(f'mean_geo_precision_at_{k}')
            median_geo_prec = rank.get(f'median_geo_precision_at_{k}')
            mean_geo_ndcg = rank.get(f'mean_geo_ndcg_at_{k}')
            median_geo_ndcg = rank.get(f'median_geo_ndcg_at_{k}')
            if mean_geo_prec is not None and mean_geo_ndcg is not None:
                median_prec_str = f" / {median_geo_prec:.3f}" if median_geo_prec else ""
                median_ndcg_str = f" / {median_geo_ndcg:.3f}" if median_geo_ndcg else ""
                print(f"  Geo-P@{k:2d}: {mean_geo_prec:.3f}{median_prec_str}  |  Geo-nDCG@{k:2d}: {mean_geo_ndcg:.3f}{median_ndcg_str}")
        
        # Per-cluster breakdown
        if 'cluster_metrics' in results:
            print(f"\n{'--- PER-BIOMARKER BREAKDOWN ---':^80}")
            for biomarker, metrics in results['cluster_metrics'].items():
                print(f"\n{biomarker} (n={metrics['patient_count']}):")
                print(f"  Recall@10:        {metrics.get('mean_recall_at_10', 0):.1%}")
                print(f"  nDCG@10:          {metrics.get('mean_ndcg_at_10', 0):.3f}")
                print(f"  ETS median rank:  {metrics.get('mean_ets_median_rank', 0):.1f}")
                if metrics.get('mean_geo_precision_at_10'):
                    print(f"  Geo-P@10:         {metrics['mean_geo_precision_at_10']:.3f}")
        
        # Boxplot data info
        print(f"\n{'--- INDIVIDUAL VALUES (for boxplot generation) ---':^80}")
        individual = results.get('individual_values', {})
        print(f"Available metrics with individual patient values:")
        for metric_name in sorted(individual.keys()):
            count = len(individual[metric_name])
            print(f"  - {metric_name}: {count} values")
        
        print(f"\n{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation: Exhaustiveness, Appropriateness, Ranking Quality"
    )
    parser.add_argument('--results-dir', '-r', required=True,
                       help='Directory containing patient result folders')
    parser.add_argument('--patients-file', '-p', required=True,
                       help='Path to patients JSON file')
    parser.add_argument('--trials-folder', '-t', required=True,
                       help='Path to folder containing trial JSON files')
    parser.add_argument('--cluster-metadata', '-c', required=True,
                       help='Path to cluster metadata CSV (Trial NCT ID, Cluster ID, Biomarker)')
    parser.add_argument('--output-file', '-o', default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveMatchingEvaluator(args.trials_folder, args.cluster_metadata)
    evaluator.save_evaluation(args.results_dir, args.patients_file, args.output_file)