# Matcher/geographic_experiment.py
"""
Geographic appropriateness experiment for TrialMatchAI.

Tests whether the current CoT reasoning considers patient-trial geographic 
proximity when ranking trials, without any prompt modifications.
"""

import json
import os
from typing import Dict, List, Set
import random
import numpy as np
import re

from generate_targeted_patients import TargetedPatientGenerator, load_trials_from_nct_ids
from generate_cluster_patients import ImprovedPatientGenerator, llm
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.file_utils import read_json_file, write_json_file

logger = setup_logging()


class GeographicExperiment:
    """
    Experiment to measure geographic awareness in trial matching.
    
    Design:
    1. Load KRAS G12C trials
    2. Categorize by location (US-available, China-available, etc.)
    3. Generate identical patients in different locations
    4. Run standard matching pipeline (no prompt changes)
    5. Measure if geographically appropriate trials rank higher
    """
    
    def __init__(self, trials_folder: str, locations_cache_file: str = None):
        self.trials_folder = trials_folder
        self.trials = []
        self.trials_by_location = {}
        self.locations_cache_file = locations_cache_file or 'trial_locations_cache.json'
        self.locations_cache = {}
        
    def load_and_categorize_trials(self, nct_ids_file: str):
        """Load trials and categorize by geographic availability."""
        logger.info(f"Loading trials from {nct_ids_file}...")
        
        # Read NCT IDs
        with open(nct_ids_file, 'r') as f:
            nct_ids = [line.strip() for line in f if line.strip().startswith('NCT')]
        
        logger.info(f"Found {len(nct_ids)} NCT IDs")
        
        # Load location cache (or fetch from API)
        self._load_or_fetch_locations(nct_ids)
        
        # Categorize by location
        self.trials_by_location = self._categorize_by_location_from_cache(nct_ids)
        
        # Log distribution
        logger.info(f"\n{'='*70}")
        logger.info("GEOGRAPHIC DISTRIBUTION OF TRIALS")
        logger.info(f"{'='*70}")
        for category, trial_list in self.trials_by_location.items():
            logger.info(f"  {category}: {len(trial_list)} trials")
        
        return self.trials_by_location

    def _load_or_fetch_locations(self, nct_ids: List[str]):
        """Load locations from cache or fetch from API."""
        from clinicaltrials_api import ClinicalTrialsAPI
        
        if os.path.exists(self.locations_cache_file):
            logger.info(f"Loading locations from cache: {self.locations_cache_file}")
            with open(self.locations_cache_file, 'r') as f:
                self.locations_cache = json.load(f)
            logger.info(f"Loaded {len(self.locations_cache)} trials from cache")
        
        # Check if we need to fetch any
        missing_ids = [nct_id for nct_id in nct_ids if nct_id not in self.locations_cache]
        
        if missing_ids:
            logger.info(f"Fetching {len(missing_ids)} trials from ClinicalTrials.gov...")
            api = ClinicalTrialsAPI(rate_limit_delay=0.5)
            
            new_locations = api.fetch_batch_locations(
                missing_ids, 
                cache_file=self.locations_cache_file
            )
            
            self.locations_cache.update(new_locations)


    def _categorize_by_location_from_cache(self, nct_ids: List[str]) -> Dict[str, List[str]]:
        """Categorize trials using cached location data."""
        categories = {
            'US_available': [],
            'China_available': [],
            'US_only': [],
            'China_only': [],
            'Both_US_China': [],
            'Other_only': []
        }
        
        for nct_id in nct_ids:
            location_data = self.locations_cache.get(nct_id)
            
            if not location_data:
                logger.warning(f"{nct_id}: No location data available")
                continue
            
            countries = location_data.get('countries', [])
            
            if not countries:
                logger.warning(f"{nct_id}: No countries found")
                continue
            
            has_us = 'United States' in countries
            has_china = 'China' in countries
            
            # Track availability
            if has_us:
                categories['US_available'].append(nct_id)
            if has_china:
                categories['China_available'].append(nct_id)
            
            # Track exclusivity
            if has_us and not has_china and len(countries) == 1:
                categories['US_only'].append(nct_id)
            elif has_china and not has_us and len(countries) == 1:
                categories['China_only'].append(nct_id)
            elif has_us and has_china:
                categories['Both_US_China'].append(nct_id)
            elif not has_us and not has_china:
                categories['Other_only'].append(nct_id)
        
        return categories
    
    def _categorize_by_location(self) -> Dict[str, List[str]]:
        """Categorize trials by where they operate."""
        categories = {
            'US_available': [],
            'China_available': [],
            'US_only': [],
            'China_only': [],
            'Both_US_China': [],
            'Other_only': []
        }
        
        for trial in self.trials:
            nct_id = self._extract_nct_id(trial)
            countries = self._extract_countries(trial)
            
            if not countries:
                logger.warning(f"{nct_id}: No location data found")
                continue
            
            has_us = any('united states' in c.lower() for c in countries)
            has_china = any('china' in c.lower() for c in countries)
            
            # Track availability
            if has_us:
                categories['US_available'].append(nct_id)
            if has_china:
                categories['China_available'].append(nct_id)
            
            # Track exclusivity
            if has_us and not has_china and len(countries) == 1:
                categories['US_only'].append(nct_id)
            elif has_china and not has_us and len(countries) == 1:
                categories['China_only'].append(nct_id)
            elif has_us and has_china:
                categories['Both_US_China'].append(nct_id)
            elif not has_us and not has_china:
                categories['Other_only'].append(nct_id)
        
        return categories
    
    def _extract_nct_id(self, trial: Dict) -> str:
        """Extract NCT ID from trial dict."""
        if 'nct_id' in trial:
            return trial['nct_id']
        elif 'protocolSection' in trial:
            return trial['protocolSection'].get('identificationModule', {}).get('nctId', 'Unknown')
        return 'Unknown'

    def get_trial_countries(self, nct_id: str) -> List[str]:
        """Get countries for a trial from cache."""
        location_data = self.locations_cache.get(nct_id, {})
        return location_data.get('countries', [])
    
    def _extract_countries(self, trial: Dict) -> List[str]:
        """Extract all countries from trial."""
        countries = set()
        
        if 'protocolSection' in trial:
            contacts_locations = trial['protocolSection'].get('contactsLocationsModule', {})
            locations = contacts_locations.get('locations', [])
            
            for location in locations:
                country = location.get('country', '')
                if country:
                    countries.add(country)
        elif 'location' in trial:
            # Legacy format
            legacy_locations = trial.get('location', [])
            if isinstance(legacy_locations, list):
                for loc in legacy_locations:
                    country = loc.get('country', '')
                    if country:
                        countries.add(country)
        
        return list(countries)
    
    def generate_location_matched_patients(
        self, 
        num_patients_per_location: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Generate clinically identical patients in different locations.
        
        This isolates geographic signal from clinical eligibility.
        """
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING LOCATION-MATCHED PATIENTS")
        logger.info(f"{'='*70}")
        
        generator = ImprovedPatientGenerator(llm)
        
        # Define locations
        us_locations = [
            {'city': 'Houston', 'state': 'Texas', 'country': 'United States'},
            {'city': 'Boston', 'state': 'Massachusetts', 'country': 'United States'},
            {'city': 'New York', 'state': 'New York', 'country': 'United States'},
            {'city': 'Los Angeles', 'state': 'California', 'country': 'United States'},
            {'city': 'Seattle', 'state': 'Washington', 'country': 'United States'},
        ]
        
        china_locations = [
            {'city': 'Beijing', 'province': 'Beijing', 'country': 'China'},
            {'city': 'Shanghai', 'province': 'Shanghai', 'country': 'China'},
            {'city': 'Guangzhou', 'province': 'Guangdong', 'country': 'China'},
            {'city': 'Shenzhen', 'province': 'Guangdong', 'country': 'China'},
            {'city': 'Hangzhou', 'province': 'Zhejiang', 'country': 'China'},
        ]
        
        patients = {
            'US': [],
            'China': []
        }
        
        # Generate identical clinical profiles, varying only location
        for i in range(num_patients_per_location):
            # Same clinical parameters for both locations
            age = random.randint(58, 72)
            gender = random.choice(['Male', 'Female'])
            
            # US patient
            us_location = us_locations[i]
            us_patient = self._generate_single_patient(
                generator=generator,
                patient_id=f"US_P{i+1:02d}",
                location=us_location,
                age=age,
                gender=gender,
                seed=i  # Same seed = similar clinical profile
            )
            patients['US'].append(us_patient)
            
            # Chinese patient (clinically identical, different location)
            china_location = china_locations[i]
            china_patient = self._generate_single_patient(
                generator=generator,
                patient_id=f"CN_P{i+1:02d}",
                location=china_location,
                age=age,
                gender=gender,
                seed=i  # Same seed = similar clinical profile
            )
            patients['China'].append(china_patient)
        
        logger.info(f"\nGenerated {len(patients['US'])} US patients")
        logger.info(f"Generated {len(patients['China'])} Chinese patients")
        
        return patients
    
    def _generate_single_patient(
        self,
        generator: ImprovedPatientGenerator,
        patient_id: str,
        location: Dict[str, str],
        age: int,
        gender: str,
        seed: int
    ) -> Dict:
        """
        Generate a single patient with specified parameters.
        
        Uses a simplified generation approach to ensure clinical consistency
        across locations.
        """
        random.seed(seed)  # For reproducibility
        
        # Fixed clinical parameters for all patients
        histology = 'Adenocarcinoma'
        mutation = 'KRAS G12C'
        stage = 'Stage IV'
        biomarker = 'KRAS'
        
        # Format location string
        if location['country'] == 'United States':
            location_str = f"{location['city']}, {location['state']}, United States"
        else:
            location_str = f"{location['city']}, {location['country']}"
        
        # Generate secondary conditions (age/gender appropriate)
        secondary_conditions, lab_values = generator.validator.generate_secondary_conditions(
            age, gender, histology
        )
        
        # Generate a consistent clinical profile using LLM
        # This is similar to generate_comprehensive_profile but more controlled
        patient_description = self._generate_consistent_profile(
            generator=generator,
            age=age,
            gender=gender,
            location_str=location_str,
            histology=histology,
            mutation=mutation,
            stage=stage,
            secondary_conditions=secondary_conditions,
            lab_values=lab_values
        )
        
        # Build patient dict
        patient_profile = {
            "patient_id": patient_id,
            "biomarker": biomarker,
            "location": location,
            "main_conditions": [histology, "NSCLC", "Lung Adenocarcinoma"],
            "other_conditions": secondary_conditions,
            "expanded_sentences": patient_description,
            "experiment_type": "geographic_matching"
        }
        
        logger.info(f"  Generated {patient_id} in {location_str}")
        
        return patient_profile


    def _generate_consistent_profile(
        self,
        generator: ImprovedPatientGenerator,
        age: int,
        gender: str,
        location_str: str,
        histology: str,
        mutation: str,
        stage: str,
        secondary_conditions: List[str],
        lab_values: Dict[str, str]
    ) -> List[str]:
        """
        Generate a consistent clinical profile using LLM.
        
        This creates standardized profiles that vary only in location details.
        """
        conditions_text = ", ".join(secondary_conditions) if secondary_conditions else "none"
        lab_text = ", ".join([f"{k}: {v}" for k, v in lab_values.items()]) if lab_values else "normal"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are creating a standardized patient profile.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Create a patient profile with these EXACT specifications:

        PATIENT: {age}-year-old {gender} from {location_str}

        DIAGNOSIS:
        - Histology: {histology}, moderately differentiated
        - Stage: {stage} with liver and lymph node metastases
        - Biomarker: {mutation} confirmed by next-generation sequencing
        - Variant allele frequency: 35-40%
        - PD-L1 tumor proportion score: 60-70%
        - ECOG performance status: 1

        DISEASE MEASUREMENTS:
        - Primary tumor: right upper lobe, 4.5-5.0 cm
        - Liver metastasis: segment 6, 3.2-3.5 cm
        - Mediastinal lymph nodes: 2.5-2.8 cm
        - Sum of target lesions: 11-12 cm per RECIST 1.1

        PRIOR TREATMENT:
        - First-line: carboplatin AUC 5 + pemetrexed 500 mg/m² 
        - Duration: 6 cycles (4-5 months ago)
        - Best response: partial response (40-50% reduction)
        - Progression: new liver lesions (2-3 months ago)

        COMORBIDITIES:
        {conditions_text}

        LABORATORY VALUES:
        - CBC: {lab_values.get('hemoglobin', 'Hgb 12.0-12.5 g/dL')}, ANC 4000-4500/μL, platelets 240,000-260,000/μL
        - Renal: creatinine 0.8-0.9 mg/dL, eGFR 88-92 mL/min
        - Hepatic: AST 22-28 U/L, ALT 20-26 U/L, bilirubin 0.5-0.7 mg/dL
        - No active infections, HIV-negative, no hepatitis B/C

        CRITICAL INSTRUCTIONS:
        - Write ONLY direct factual sentences
        - NO section headers
        - NO introductory phrases
        - Start immediately with patient facts
        - One sentence per line
        - Each sentence should be complete and standalone
        - Use SPECIFIC numbers within the ranges given above
        - Include diagnosis date, molecular testing date, treatment dates
        - State clearly: "Patient has no brain metastases" and "No prior KRAS inhibitor therapy"

        Example format (FOLLOW THIS):
        The patient is a 65-year-old male from Houston, Texas, United States.
        He was diagnosed with lung adenocarcinoma, acinar-predominant pattern, moderately differentiated, in January 2024.
        Molecular testing by FoundationOne CDx on February 2024 revealed KRAS G12C mutation at 38% variant allele frequency.

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        try:
            from langchain.schema import HumanMessage
            import re
            response = generator.llm.invoke([HumanMessage(content=prompt)])
            raw_content = response.content if hasattr(response, 'content') else str(response)
            raw_description = raw_content.strip()
            
            # Clean up any intro/outro
            raw_description = re.sub(r"^Here (?:is|are) (?:the )?.*?:\s*", "", raw_description, flags=re.IGNORECASE)
            raw_description = re.sub(r"^(?:Patient )?Profile:\s*", "", raw_description, flags=re.IGNORECASE)
            
            # Split into sentences
            lines = [s.strip() for s in raw_description.split('\n') if s.strip()]
            
            sentences = []
            for line in lines:
                # Skip headers
                if line.endswith(':') or re.match(r'^[\*\-\+\#\s]+', line):
                    continue
                
                # Skip intro phrases
                if re.match(r'^(?:Here|Patient|Primary|Critical|Molecular|Prior|Current|Laboratory|Comorbidities|Treatment)\s*:', line, re.IGNORECASE):
                    continue
                
                # Split on '. ' but keep the period
                parts = line.split('. ')
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part or len(part) < 15:
                        continue
                        
                    # Add period back if removed
                    if i < len(parts) - 1 and not part.endswith('.'):
                        part += '.'
                    
                    sentences.append(part)
            
            # Validate sentences
            validated = generator.validator.validate_and_fix_expanded_sentences(
                sentences, 
                [histology] + secondary_conditions
            )
            
            logger.debug(f"Generated {len(validated)} validated sentences")
            return validated
            
        except Exception as e:
            logger.warning(f"Profile generation failed: {e}, using template")
            
            # Fallback template
            return [
                f"The patient is a {age}-year-old {gender} from {location_str}.",
                f"Diagnosed with lung adenocarcinoma, {stage}, in January 2024.",
                f"Molecular testing revealed {mutation} at 37% variant allele frequency.",
                f"PD-L1 tumor proportion score was 65%.",
                f"Primary tumor in right upper lobe measures 4.7 cm.",
                f"Liver metastasis in segment 6 measures 3.3 cm.",
                f"Mediastinal lymph nodes measure 2.6 cm.",
                f"Sum of target lesions is 11.4 cm per RECIST 1.1.",
                f"ECOG performance status is 1.",
                f"Patient received first-line carboplatin and pemetrexed for 6 cycles.",
                f"Best response was partial response with 45% reduction.",
                f"Disease progressed 3 months ago with new liver lesions.",
                f"Patient has no brain metastases.",
                f"No prior KRAS G12C inhibitor therapy.",
                f"Laboratory values show adequate organ function.",
            ]
    
    def evaluate_geographic_ranking(
        self,
        patient_id: str,
        patient_country: str,
        ranked_trials: List[Dict]
    ) -> Dict:
        """
        Evaluate how well geographically appropriate AND ELIGIBLE trials are ranked.
        """
        # Determine which trials are geographically appropriate
        if patient_country == 'US':
            geo_available_trials = set(self.trials_by_location['US_available'])
        elif patient_country == 'China':
            geo_available_trials = set(self.trials_by_location['China_available'])
        else:
            geo_available_trials = set()
        
        if not geo_available_trials:
            logger.warning(f"No geo-appropriate trials found for {patient_country}")
            return {}
        
        # NEW: Filter for trials that are BOTH geo-appropriate AND eligible
        # Extract NCT IDs and scores from ranked trials
        ranked_nct_ids = []
        trial_scores = {}
        for trial in ranked_trials:
            nct_id = trial.get('TrialID') or trial.get('nct_id')
            score = trial.get('Score') or trial.get('score') or trial.get('final_score', 0)
            if nct_id:
                ranked_nct_ids.append(nct_id)
                trial_scores[nct_id] = float(score)
        
        # Define eligibility threshold (trials with score >= 0.5 are considered eligible)
        ELIGIBILITY_THRESHOLD = 0.7
        
        # Appropriate trials = Geo-appropriate AND Eligible
        appropriate_trials = set()
        for nct_id in geo_available_trials:
            if nct_id in trial_scores and trial_scores[nct_id] >= ELIGIBILITY_THRESHOLD:
                appropriate_trials.add(nct_id)
        
        # Also track geo-appropriate but ineligible (for analysis)
        geo_but_ineligible = set()
        for nct_id in geo_available_trials:
            if nct_id in trial_scores and trial_scores[nct_id] < ELIGIBILITY_THRESHOLD:
                geo_but_ineligible.add(nct_id)
        
        if not appropriate_trials:
            logger.warning(
                f"{patient_id}: No trials are BOTH geo-appropriate AND eligible "
                f"(geo-appropriate: {len(geo_available_trials)}, eligible: 0)"
            )
            return {
                'patient_id': patient_id,
                'patient_country': patient_country,
                'total_geo_available': len(geo_available_trials),
                'total_geo_and_eligible': 0,
                'total_geo_but_ineligible': len(geo_but_ineligible),
                'note': 'No trials meet both geographic and eligibility criteria'
            }
        
        logger.info(
            f"{patient_id}: {len(appropriate_trials)} trials are geo-appropriate AND eligible "
            f"(out of {len(geo_available_trials)} geo-appropriate total)"
        )
        
        metrics = {
            'patient_id': patient_id,
            'patient_country': patient_country,
            'total_geo_available': len(geo_available_trials),
            'total_geo_and_eligible': len(appropriate_trials),
            'total_geo_but_ineligible': len(geo_but_ineligible),
            'total_ranked_trials': len(ranked_trials)
        }
        
        # Recall@K for trials that are BOTH geo-appropriate AND eligible
        for k in [5, 10, 20, 30]:
            if k > len(ranked_nct_ids):
                continue
                
            top_k_ids = set(ranked_nct_ids[:k])
            appropriate_in_top_k = len(top_k_ids & appropriate_trials)
            
            metrics[f'geo_eligible_recall_at_{k}'] = appropriate_in_top_k / len(appropriate_trials)
            metrics[f'geo_eligible_count_at_{k}'] = appropriate_in_top_k
        
        # Precision@K (what % of top-K are geo-appropriate AND eligible)
        for k in [5, 10, 20, 30]:
            if k > len(ranked_nct_ids):
                continue
                
            top_k_ids = set(ranked_nct_ids[:k])
            appropriate_in_top_k = len(top_k_ids & appropriate_trials)
            
            metrics[f'geo_eligible_precision_at_{k}'] = appropriate_in_top_k / k
        
        # Rank statistics for appropriate trials
        appropriate_ranks = [
            i + 1 for i, nct_id in enumerate(ranked_nct_ids)
            if nct_id in appropriate_trials
        ]
        
        if appropriate_ranks:
            metrics['mean_appropriate_rank'] = float(np.mean(appropriate_ranks))
            metrics['median_appropriate_rank'] = float(np.median(appropriate_ranks))
            metrics['best_appropriate_rank'] = int(min(appropriate_ranks))
            metrics['worst_appropriate_rank'] = int(max(appropriate_ranks))
        
        # BONUS: Also track geographic-only metrics (for comparison)
        for k in [5, 10, 20]:
            top_k_ids = set(ranked_nct_ids[:k])
            geo_only_in_top_k = len(top_k_ids & geo_available_trials)
            metrics[f'geo_only_recall_at_{k}'] = geo_only_in_top_k / len(geo_available_trials)
        
        return metrics
    
    def run_experiment(
        self,
        patients_file: str,
        results_folder: str,
        scenario: str = 'fl-hybrid_no-rerank_cot-vllm_ner'
    ) -> Dict:
        """
        Run the geographic experiment.
        
        Args:
            patients_file: Path to generated patients JSON
            results_folder: Base folder with matching results (e.g., '../data/geo_experiment')
            scenario: Ablation scenario name
        
        Returns:
            Dict with aggregated results
        """
        logger.info(f"\n{'='*70}")
        logger.info("RUNNING GEOGRAPHIC EXPERIMENT")
        logger.info(f"{'='*70}")
        
        # Load patients
        with open(patients_file, 'r') as f:
            patients_data = json.load(f)
        
        if isinstance(patients_data, dict):
            all_patients = list(patients_data.values())
        else:
            all_patients = patients_data
        
        # Separate by location
        us_patients = [p for p in all_patients if 'US' in p['patient_id']]
        china_patients = [p for p in all_patients if 'CN' in p['patient_id']]
        
        logger.info(f"Loaded {len(us_patients)} US patients, {len(china_patients)} Chinese patients")
        
        results = {
            'US_patients': [],
            'China_patients': []
        }
        
        # Evaluate US patients
        logger.info("\n--- Evaluating US Patients ---")
        for patient in us_patients:
            patient_id = patient['patient_id']
            
            # Load ranked trials
            ranked_path = os.path.join(
                results_folder, patient_id, scenario, 'ranked_trials.json'
            )
            
            if not os.path.exists(ranked_path):
                logger.warning(f"No results found for {patient_id}")
                continue
            
            with open(ranked_path, 'r') as f:
                ranked_data = json.load(f)
                
            ranked_trials = ranked_data.get('RankedTrials', ranked_data)
            
            # Evaluate
            metrics = self.evaluate_geographic_ranking(
                patient_id=patient_id,
                patient_country='US',
                ranked_trials=ranked_trials
            )
            
            results['US_patients'].append(metrics)
            logger.info(f"  {patient_id}: Geo-Recall@10 = {metrics.get('geo_recall_at_10', 0):.1%}")
        
        # Evaluate Chinese patients
        logger.info("\n--- Evaluating Chinese Patients ---")
        for patient in china_patients:
            patient_id = patient['patient_id']
            
            ranked_path = os.path.join(
                results_folder, patient_id, scenario, 'ranked_trials.json'
            )
            
            if not os.path.exists(ranked_path):
                logger.warning(f"No results found for {patient_id}")
                continue
            
            with open(ranked_path, 'r') as f:
                ranked_data = json.load(f)
                
            ranked_trials = ranked_data.get('RankedTrials', ranked_data)
            
            metrics = self.evaluate_geographic_ranking(
                patient_id=patient_id,
                patient_country='China',
                ranked_trials=ranked_trials
            )
            
            results['China_patients'].append(metrics)
            logger.info(f"  {patient_id}: Geo-Recall@10 = {metrics.get('geo_recall_at_10', 0):.1%}")
        
        # Aggregate statistics
        summary = self._create_summary(results)
        
        # Save results
        os.makedirs(os.path.join(results_folder, 'geographic_evaluation'), exist_ok=True)
        
        write_json_file(
            results,
            os.path.join(results_folder, 'geographic_evaluation', 'detailed_results.json')
        )
        
        write_json_file(
            summary,
            os.path.join(results_folder, 'geographic_evaluation', 'summary.json')
        )
        
        self._print_summary(summary)
        
        return {'detailed': results, 'summary': summary}
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create summary statistics."""
        summary = {}
        
        for group in ['US_patients', 'China_patients']:
            group_results = results[group]
            
            if not group_results:
                continue
            
            summary[group] = {
                'num_patients': len(group_results),
                'mean_metrics': {}
            }
            
            # Average across patients
            metrics_to_average = [
                'geo_recall_at_5', 'geo_recall_at_10', 'geo_recall_at_20',
                'geo_precision_at_5', 'geo_precision_at_10', 'geo_precision_at_20',
                'mean_appropriate_rank', 'median_appropriate_rank'
            ]
            
            for metric in metrics_to_average:
                values = [r.get(metric) for r in group_results if r.get(metric) is not None]
                if values:
                    summary[group]['mean_metrics'][metric] = float(np.mean(values))
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print summary to console."""
        logger.info(f"\n{'='*70}")
        logger.info("GEOGRAPHIC EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        
        for group, data in summary.items():
            country = group.split('_')[0]
            logger.info(f"\n{country} Patients (n={data['num_patients']}):")
            
            metrics = data['mean_metrics']
            
            # Show breakdown
            logger.info(f"  Geo-available trials (avg): {metrics.get('total_geo_available', 0):.0f}")
            logger.info(f"  Geo + Eligible trials (avg): {metrics.get('total_geo_and_eligible', 0):.0f}")
            logger.info(f"  Geo but Ineligible (avg): {metrics.get('total_geo_but_ineligible', 0):.0f}")
            
            # Metrics on geo + eligible set
            logger.info(f"\n  Geo+Eligible Recall@10: {metrics.get('geo_eligible_recall_at_10', 0):.1%}")
            logger.info(f"  Geo+Eligible Recall@20: {metrics.get('geo_eligible_recall_at_20', 0):.1%}")
            logger.info(f"  Geo+Eligible Precision@10: {metrics.get('geo_eligible_precision_at_10', 0):.1%}")
            logger.info(f"  Mean rank (Geo+Eligible): {metrics.get('mean_appropriate_rank', 0):.1f}")
            
            # Show comparison with geo-only (for reference)
            logger.info(f"\n  For comparison - Geo-only Recall@10: {metrics.get('geo_only_recall_at_10', 0):.1%}")
        
        logger.info(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run geographic appropriateness experiment"
    )
    parser.add_argument(
        '--mode',
        choices=['generate', 'evaluate'],
        required=True,
        help='generate: Create location-matched patients | evaluate: Analyze results'
    )
    parser.add_argument(
        '--nct-ids-file',
        default='../data/lung_cluster_trials_raw/nct_ids.txt',
        help='File with NCT IDs (88 KRAS trials)'
    )
    parser.add_argument(
        '--trials-folder',
        default='../data/lung_processed_trials',
        help='Folder with trial JSON files'
    )
    parser.add_argument(
        '--output-file',
        default='../data/geographic_experiment_patients.json',
        help='Output file for generated patients'
    )
    parser.add_argument(
        '--results-folder',
        default='../data/geo_experiment_results',
        help='Folder with matching results'
    )
    parser.add_argument(
        '--num-patients',
        type=int,
        default=5,
        help='Number of patients per location'
    )
    
    args = parser.parse_args()
    
    experiment = GeographicExperiment(args.trials_folder)
    
    if args.mode == 'generate':
        # Step 1: Categorize trials by location
        experiment.load_and_categorize_trials(args.nct_ids_file)
        
        # Step 2: Generate location-matched patients
        patients = experiment.generate_location_matched_patients(
            num_patients_per_location=args.num_patients
        )
        
        # Step 3: Save patients
        all_patients = {}
        for location, patient_list in patients.items():
            for patient in patient_list:
                all_patients[patient['patient_id']] = patient
        
        with open(args.output_file, 'w') as f:
            json.dump(all_patients, f, indent=2)
        
        logger.info(f"\n✓ Saved {len(all_patients)} patients to {args.output_file}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Run ablation_study.py on these patients")
        logger.info(f"2. Run: python geographic_experiment.py --mode evaluate")
    
    elif args.mode == 'evaluate':
        # Load trial categories
        experiment.load_and_categorize_trials(args.nct_ids_file)
        
        # Run evaluation
        results = experiment.run_experiment(
            patients_file=args.output_file,
            results_folder=args.results_folder
        )
        
        logger.info("\n✓ Geographic experiment complete!")