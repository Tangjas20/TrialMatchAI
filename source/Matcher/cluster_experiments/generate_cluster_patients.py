import os
import json
import random
from typing import List, Dict, Tuple, Any, Set
import re
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from Matcher.utils.generation_utils import (
    extract_age_gender_from_summary,
    extract_main_condition_from_summary,
    split_into_sentences,
)
from cluster_utils import convert_cluster_file_to_pipeline_file

from langchain_huggingface import HuggingFacePipeline
from langchain.schema import HumanMessage
from transformers import pipeline, logging
from Matcher.utils.logging_config import setup_logging

logger = setup_logging()
logging.set_verbosity_error()

# Find Device and use GPU if possible
import torch
device = 0 if torch.cuda.is_available() else -1
logger.warning(f"Using device: {'GPU' if device==0 else 'CPU'}")

# Initialize the language model
generator = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="cuda" if device == 0 else "cpu",
    torch_dtype="auto",
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
)

generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

llm = HuggingFacePipeline(pipeline=generator)

@dataclass
class ConditionPrevalence:
    """Define condition prevalences and constraints"""
    name: str
    prevalence: float  # 0.0 to 1.0
    mutually_exclusive: List[str] = None
    lab_constraints: Dict[str, str] = None
    treatment_required: bool = False

class ClinicalValidator:
    """Validates clinical accuracy and manages condition distributions"""
    
    def __init__(self):
        # Define lung cancer type hierarchy
        self.lung_cancer_hierarchy = {
            'primary_types': ['Non-Small Cell Lung Cancer', 'Small Cell Lung Cancer'],
            'nsclc_subtypes': ['Pulmonary Adenocarcinoma', 'Squamous Cell Carcinoma', 'Large Cell Carcinoma'],
            'general_terms': ['Lung Cancer', 'Bronchogenic Carcinoma', 'Primary Lung Tumor']
        }

        self.nsclc_histologies = {
            'adenocarcinoma': {
                'primary': 'Lung Adenocarcinoma',
                'synonyms': [
                    'Pulmonary Adenocarcinoma',
                    'Adenocarcinoma of the Lung',
                    'Lung Adenocarcinoma NSCLC'
                ]
            },
            'squamous': {
                'primary': 'Lung Squamous Cell Carcinoma',
                'synonyms': [
                    'Squamous Cell Lung Cancer',
                    'Pulmonary Squamous Cell Carcinoma',
                    'Lung SCC'
                ]
            },
            'large_cell': {
                'primary': 'Large Cell Lung Carcinoma',
                'synonyms': [
                    'Large Cell Lung Cancer',
                    'Undifferentiated Large Cell Carcinoma'
                ]
            },
            'nos': {
                'primary': 'Non-Small Cell Lung Cancer',
                'synonyms': [
                    'NSCLC',
                    'Non-Small Cell Lung Carcinoma',
                    'NSCLC NOS'
                ]
            }
        }

        # Generic NSCLC synonyms (always included)
        self.generic_nsclc_synonyms = [
            'Lung Cancer',
            'Primary Lung Tumor',
            'Bronchogenic Carcinoma'
        ]
        
        # Define realistic condition prevalences
        self.condition_prevalences = {
            'pleural_effusion': ConditionPrevalence('Pleural Effusion', 0.30),
            'pneumothorax': ConditionPrevalence('Pneumothorax', 0.05),
            'copd': ConditionPrevalence('Chronic Obstructive Pulmonary Disease', 0.67),
            'diabetes': ConditionPrevalence('Diabetes Mellitus', 0.25),
            'hypertension': ConditionPrevalence('Hypertension', 0.45),
            'coronary_artery_disease': ConditionPrevalence('Coronary Artery Disease', 0.20),
            'osteoporosis': ConditionPrevalence('Osteoporosis', 0.15),
            'anemia': ConditionPrevalence(
                'Anemia', 0.40,
                lab_constraints={'hemoglobin': '10.0-11.5 g/dL'},
                treatment_required=False
            ),
            'anemia_treated': ConditionPrevalence(
                'Anemia', 0.20,
                lab_constraints={'hemoglobin': '12.0-12.8 g/dL'},
                treatment_required=True
            )
        }
        
        # Track condition assignments across all patients
        self.condition_tracker = defaultdict(int)
        self.total_patients_generated = 0
    
    def get_valid_lung_cancer_synonyms(self, histology: str) -> List[str]:
        """
        Generate simple synonym list based on histology.
        Returns a flat list - no nested structure.
        """
        synonyms = []
        
        histology_lower = histology.lower()
        
        # Start with the specific histology
        if 'adenocarcinoma' in histology_lower:
            synonyms = [
                "Lung Adenocarcinoma",
                "Pulmonary Adenocarcinoma",
                "Adenocarcinoma of the Lung",
                "Non-Small Cell Lung Cancer",
                "NSCLC",
                "Lung Cancer",
                "Primary Lung Tumor",
                "Bronchogenic Carcinoma"
            ]
        elif 'squamous' in histology_lower:
            synonyms = [
                "Lung Squamous Cell Carcinoma",
                "Squamous Cell Lung Cancer",
                "Pulmonary Squamous Cell Carcinoma",
                "Non-Small Cell Lung Cancer",
                "NSCLC",
                "Lung Cancer",
                "Primary Lung Tumor",
                "Bronchogenic Carcinoma"
            ]
        elif 'large cell' in histology_lower or 'large-cell' in histology_lower:
            synonyms = [
                "Large Cell Lung Carcinoma",
                "Large Cell Lung Cancer",
                "Non-Small Cell Lung Cancer",
                "NSCLC",
                "Lung Cancer",
                "Primary Lung Tumor",
                "Bronchogenic Carcinoma"
            ]
        else:
            # Generic NSCLC (no specific histology)
            synonyms = [
                "Non-Small Cell Lung Cancer",
                "NSCLC",
                "Lung Cancer",
                "Primary Lung Tumor",
                "Bronchogenic Carcinoma",
                "Non-Small Cell Lung Carcinoma"
            ]
        
        return synonyms
    
    def should_assign_condition(self, condition_key: str) -> bool:
        """Determine if a condition should be assigned based on prevalence"""
        if condition_key not in self.condition_prevalences:
            return False
            
        prevalence = self.condition_prevalences[condition_key]
        current_rate = (self.condition_tracker[condition_key] / 
                       max(1, self.total_patients_generated))
        
        target_rate = prevalence.prevalence
        
        if current_rate < target_rate:
            probability = min(1.0, target_rate * 1.5)
        else:
            probability = max(0.0, target_rate * 0.5)
            
        return random.random() < probability
    
    def generate_secondary_conditions(self, age: int, gender: str, 
                                    main_condition: str) -> Tuple[List[str], Dict[str, str]]:
        """Generate realistic secondary conditions with proper prevalences"""
        secondary_conditions = []
        lab_values = {}
        
        # Check each condition based on prevalence
        condition_checks = {
            'pleural_effusion': 'Pleural Effusion',
            'pneumothorax': 'Pneumothorax', 
            'copd': 'Chronic Obstructive Pulmonary Disease',
            'diabetes': 'Diabetes Mellitus',
            'hypertension': 'Hypertension',
            'coronary_artery_disease': 'Coronary Artery Disease'
        }
        
        # Age-dependent conditions
        if age >= 65:
            condition_checks['osteoporosis'] = 'Osteoporosis'
        
        for key, condition_name in condition_checks.items():
            if self.should_assign_condition(key):
                secondary_conditions.append(condition_name)
                self.condition_tracker[key] += 1
        
        # Handle anemia specially (with lab constraints)
        anemia_assigned = False
        if self.should_assign_condition('anemia'):
            secondary_conditions.append('Anemia')
            lab_values['hemoglobin'] = f"{random.uniform(10.0, 11.5):.1f} g/dL"
            self.condition_tracker['anemia'] += 1
            anemia_assigned = True
        elif self.should_assign_condition('anemia_treated'):
            secondary_conditions.append('Anemia')
            lab_values['hemoglobin'] = f"{random.uniform(12.0, 12.8):.1f} g/dL"
            self.condition_tracker['anemia_treated'] += 1
            anemia_assigned = True
        
        # Add some normal lab values for realism
        if not anemia_assigned and random.random() < 0.3:
            if gender.lower() == 'male':
                lab_values['hemoglobin'] = f"{random.uniform(13.5, 16.5):.1f} g/dL"
            else:
                lab_values['hemoglobin'] = f"{random.uniform(12.0, 15.5):.1f} g/dL"
        
        return secondary_conditions, lab_values
    
    def validate_and_fix_expanded_sentences(self, expanded_sentences: List[str], 
                                          conditions: List[str]) -> List[str]:
        """Ensure expanded sentences don't contain conditions not in the conditions list"""
        
        condition_terms = set()
        condition_mappings = {
            'copd': 'chronic obstructive pulmonary disease',
            'chronic obstructive pulmonary disease': 'copd',
            'cad': 'coronary artery disease',
            'coronary artery disease': 'cad',
            'diabetes': 'diabetes mellitus',
            'diabetes mellitus': 'diabetes',
            'htn': 'hypertension',
            'hypertension': 'htn'
        }
        
        for condition in conditions:
            condition_lower = condition.lower()
            condition_terms.add(condition_lower)
            words = condition_lower.split()
            condition_terms.update(words)
            if condition_lower in condition_mappings:
                condition_terms.add(condition_mappings[condition_lower])
        
        medical_conditions = {
            'diabetes': ['diabetes', 'diabetes mellitus', 'dm', 'diabetic'],
            'hypertension': ['hypertension', 'htn', 'high blood pressure'],
            'coronary artery disease': ['coronary artery disease', 'cad', 'coronary disease'],
            'copd': ['copd', 'chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
            'osteoporosis': ['osteoporosis'],
            'anemia': ['anemia', 'anaemia', 'anemic'],
            'pleural effusion': ['pleural effusion'],
            'pneumothorax': ['pneumothorax']
        }
        
        fixed_sentences = []
        for sentence in expanded_sentences:
            sentence_lower = sentence.lower()
            has_uncaptured_condition = False
            
            for condition_group, variations in medical_conditions.items():
                appears_in_sentence = any(var in sentence_lower for var in variations)
                
                if appears_in_sentence:
                    condition_captured = any(
                        any(var in condition.lower() for var in variations) or
                        condition.lower() in variations
                        for condition in conditions
                    )
                    
                    if not condition_captured:
                        has_uncaptured_condition = True
                        logger.warning(f"Found uncaptured condition '{condition_group}' in sentence (will be removed)")
                        break
            
            if not has_uncaptured_condition:
                fixed_sentences.append(sentence)
        
        return fixed_sentences

class ImprovedPatientGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.validator = ClinicalValidator()
        
        # Cache for trial locations
        self.trial_locations_cache = {}
        
        self.biomarker_params = {
        'KRAS': {
            'mutations': ['KRAS G12C', 'KRAS G12D', 'KRAS G12V', 'KRAS G13D'],
            'common_drugs': ['sotorasib', 'adagrasib', 'garsorasib', 'divarasib'],
            'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic'],
            'histology_preference': ['Adenocarcinoma', 'Non-Small Cell Lung Cancer'],
            'prior_therapy_appropriate': [
                'carboplatin AUC 5 + pemetrexed 500 mg/m²',
                'cisplatin 75 mg/m² + pemetrexed 500 mg/m²',
                'carboplatin AUC 5 + pemetrexed 500 mg/m² + pembrolizumab 200 mg',
                'pembrolizumab 200 mg monotherapy',
                'nivolumab 240 mg',
                'atezolizumab 1200 mg + carboplatin + pemetrexed',
                'no prior systemic therapy'
            ],
            'prior_therapy_inappropriate': [
                'gefitinib', 'erlotinib', 'osimertinib', 'afatinib', 'dacomitinib',  # EGFR inhibitors
                'crizotinib', 'alectinib', 'brigatinib', 'ceritinib', 'lorlatinib'   # ALK inhibitors
            ]
        },
        'EGFR': {
            'mutations': ['EGFR exon 19 deletion', 'EGFR L858R', 'EGFR T790M'],
            'common_drugs': ['osimertinib', 'erlotinib', 'gefitinib'],
            'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic'],
            'histology_preference': ['Adenocarcinoma', 'Non-Small Cell Lung Cancer'],
            'prior_therapy_appropriate': [
                'osimertinib 80 mg daily',
                'erlotinib 150 mg daily',
                'gefitinib 250 mg daily',
                'afatinib 40 mg daily',
                'carboplatin + pemetrexed (before EGFR testing)',
                'no prior systemic therapy'
            ],
            'prior_therapy_inappropriate': [
                'sotorasib', 'adagrasib', 'garsorasib'  # KRAS inhibitors
            ]
        },
        'ROS1': {
            'mutations': ['ROS1 fusion', 'ROS1 rearrangement'],
            'common_drugs': ['repotrectinib', 'crizotinib', 'entrectinib', 'lorlatinib'],
            'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic'],
            'histology_preference': ['Adenocarcinoma', 'Non-Small Cell Lung Cancer'],
            'prior_therapy_appropriate': [
                'crizotinib 250 mg twice daily',
                'entrectinib 600 mg once daily',
                'carboplatin + pemetrexed (before ROS1 testing)',
                'no prior systemic therapy'
            ],
            'prior_therapy_inappropriate': [
                'gefitinib', 'erlotinib', 'osimertinib',  # EGFR inhibitors
                'sotorasib', 'adagrasib'  # KRAS inhibitors
            ]
        }
    }
        
        # Major cities by country for realistic patient locations
        self.major_cities_by_country = {
            'United States': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                             'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                             'Boston', 'Seattle', 'Denver', 'Atlanta', 'Miami'],
            'Canada': ['Toronto', 'Vancouver', 'Montreal', 'Calgary', 'Edmonton', 'Ottawa'],
            'United Kingdom': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Edinburgh'],
            'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne', 'Stuttgart'],
            'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes'],
            'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'],
            'Japan': ['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka', 'Sapporo'],
            'China': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Chengdu'],
        }
    
    def extract_trial_locations(self, trial: Dict) -> List[str]:
        """Extract country locations from a trial."""
        countries = set()
        
        if 'protocolSection' in trial:
            contacts_locations = trial['protocolSection'].get('contactsLocationsModule', {})
            locations = contacts_locations.get('locations', [])
            
            for location in locations:
                country = location.get('country', '')
                if country:
                    countries.add(country)
        
        elif 'location' in trial:
            legacy_locations = trial['location']
            if isinstance(legacy_locations, list):
                for loc in legacy_locations:
                    if 'location_address' in loc:
                        address = loc['location_address']
                        country = address.strip().split(',')[-1].strip()
                        if country in self.major_cities_by_country.keys():
                            countries.add(country)
        
        return list(countries) if countries else ['United States']
    
    def get_cluster_countries(self, cluster_trials: List[Dict]) -> List[str]:
        """Get all unique countries from trials in a cluster."""
        all_countries = set()
        
        for trial in cluster_trials:
            trial_id = id(trial)
            
            if trial_id in self.trial_locations_cache:
                countries = self.trial_locations_cache[trial_id]
            else:
                countries = self.extract_trial_locations(trial)
                self.trial_locations_cache[trial_id] = countries
            
            all_countries.update(countries)
        
        return list(all_countries) if all_countries else ['United States']
    
    def assign_patient_location(self, cluster_countries: List[str]) -> Dict[str, str]:
        """Assign a realistic location to a patient based on cluster trial locations."""
        country = random.choice(cluster_countries)
        cities = self.major_cities_by_country.get(country, [country])
        city = random.choice(cities)
        
        return {
            'country': country,
            'city': city
        }
    
    def extract_structured_criteria(self, trials: List[Dict], biomarker: str = None) -> Dict:
        """Extract structured criteria from trials using LLM analysis."""
        
        logger.info(f"Extracting criteria from {len(trials)} trials for biomarker: {biomarker}")
        
        all_criteria = []
        all_conditions = []
        
        for trial in trials:
            criteria = ""
            if 'protocolSection' in trial:
                criteria = trial['protocolSection'].get('eligibilityModule', {}).get('eligibilityCriteria', '')
                conditions = trial['protocolSection'].get('conditionsModule', {}).get('conditions', [])
            else:
                criteria = trial.get('eligibility_criteria', '')
                conditions = trial.get('conditions', [])
            
            if criteria:
                all_criteria.append(criteria)
            
            if isinstance(conditions, list):
                all_conditions.extend(conditions)
            elif isinstance(conditions, str):
                all_conditions.append(conditions)
        
        # Extract biomarker mentions and split inclusion/exclusion
        biomarker_mentions = []
        inclusion_criteria = []
        exclusion_criteria = []
        
        for criteria_text in all_criteria:
            # Split into inclusion/exclusion
            parts = re.split(r'exclusion criteria', criteria_text, flags=re.IGNORECASE)
            inclusion = parts[0]
            exclusion = parts[1] if len(parts) > 1 else ""
            
            inclusion_criteria.append(inclusion)
            exclusion_criteria.append(exclusion)
            
            # Extract biomarker-specific mentions
            if biomarker:
                biomarker_pattern = rf'[^\n.]*\b{re.escape(biomarker)}\b[^\n.]*'
                matches = re.findall(biomarker_pattern, criteria_text, re.IGNORECASE)
                biomarker_mentions.extend(matches)
        
        # Combine but prioritize biomarker mentions
        biomarker_text = " ".join(set(biomarker_mentions))  # Remove duplicates
        combined_inclusion = " ".join(inclusion_criteria)
        combined_exclusion = " ".join(exclusion_criteria)
        
        # Create VERY explicit prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a clinical trial expert.<|eot_id|><|start_header_id|>user<|end_header_id|>

        I am giving you eligibility criteria from {len(trials)} clinical trials. EVERY SINGLE ONE of these trials is EXCLUSIVELY for patients with {biomarker}-positive non-small cell lung cancer.

        BIOMARKER-SPECIFIC MENTIONS:
        {biomarker_text[:1000]}

        INCLUSION CRITERIA (from {biomarker}-only trials):
        {combined_inclusion[:1500]}

        EXCLUSION CRITERIA (from {biomarker}-only trials):
        {combined_exclusion[:1500]}

        Here are additional guidelines for each biomarker:
        - ROS1 : gene rearrangements by either tissue or plasma genotyping + advanced/metastatic disease status + absence of other targetable driver mutations (such as ALK, EGFR, or BRAF) + adequate organ function and ECOG performance status
        - KRAS: KRAS mutations demonstrated in tumor tissue or circulating tumor DNA + Measurable disease per RECIST 1.1 + ECOG performance status ≤1 + histologically confirmed advanced solid tumors + additional markers such as positive HLA alleles (for immunotherapeutic studies)
        - EGFR : activating mutations + diagnosis by a CLIA-certified assay + histopathological confirmation of NSCLC + absence of prior therapy for advanced disease + presence resistance mutations (e.g., T790M) + tissue availability for biomarker and resistance pathway profiling

        YOUR TASK:
        Extract the common eligibility requirements across these {biomarker} trials.

        Format your response EXACTLY as follows:

        **Age Requirements:**
        [Common age requirements]

        **Cancer Stage Requirements:**
        [Common stage requirements]

        **Performance Status Requirements:**
        [ECOG or Karnofsky requirements]

        **Required Biomarker Status:**
        Must have {biomarker}-positive NSCLC confirmed by [testing method]
        [Additional {biomarker}-specific requirements]

        **Prior Treatment Requirements:**
        [Treatment history requirements]

        **Inclusion Criteria (Major Requirements):**
        - [Requirement 1]
        - [Requirement 2]

        **Exclusion Criteria (Major Contraindications):**
        - [Exclusion 1]
        - [Exclusion 2]

        REMEMBER: Write as if ALL trials are for {biomarker} patients.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            if hasattr(response, 'content'):
                raw_content = response.content
            elif isinstance(response, str):
                raw_content = response
            else:
                raw_content = str(response)
            structured_criteria = raw_content.strip()
            structured_criteria = re.sub(r"^Human[:\-]?\s*", "", structured_criteria)
            
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            structured_criteria = f"**Required Biomarker Status:**\nMust have {biomarker}-positive NSCLC"
        
        return {
            'structured_criteria': structured_criteria,
            'all_conditions': list(set(all_conditions)),
            'raw_criteria': combined_inclusion + "\n\nExclusion:\n" + combined_exclusion,
            'biomarker_specific': biomarker_text
        }
    
    def generate_biomarker_aware_patients(self, cluster_trials: List[Dict], 
                                        biomarker: int, cluster_id: str, 
                                        num_patients: int = 100) -> List[Dict]:
        """Generate patients with clinical accuracy and proper condition distributions."""
        
        # Reset validator for this cluster
        self.validator.condition_tracker.clear()
        self.validator.total_patients_generated = 0
        
        # MAP BIOMARKER INDEX TO NAME
        biomarker_names = {0: 'KRAS', 1: 'EGFR', 2: 'ROS1'}
        biomarker_name = biomarker_names.get(biomarker, f'Biomarker_{biomarker}')
        
        # Pass the NAME, not the index!
        criteria_info = self.extract_structured_criteria(cluster_trials, biomarker=biomarker_name)
        
        # Extract cluster countries for patient location assignment
        cluster_countries = self.get_cluster_countries(cluster_trials)
        logger.warning(f"\nCluster trial locations: {cluster_countries}")
        
        biomarker_info = self.biomarker_params.get(biomarker_name, {})
        
        logger.warning(f"\nCluster {cluster_id} ({biomarker_name}) - Structured Criteria:")
        logger.warning(criteria_info['structured_criteria'])
        
        patients = []
        for i in range(num_patients):
            logger.warning(f"\nGenerating patient {i+1}/{num_patients} for biomarker {biomarker_name}...")
            
            # Generate realistic demographics
            age = self._generate_realistic_age()
            gender = self._generate_realistic_gender()
            
            # Select appropriate histology for biomarker
            histology = random.choice(biomarker_info.get('histology_preference', ['Non-Small Cell Lung Cancer']))
            
            # Select mutation
            mutation = random.choice(biomarker_info.get('mutations', [biomarker_name])) if biomarker_info else biomarker_name
            
            # Select stage
            stage = random.choice(biomarker_info.get('stage_preference', ['Stage IV', 'Advanced']))
            
            # Generate secondary conditions with proper prevalence
            secondary_conditions, lab_values = self.validator.generate_secondary_conditions(
                age, gender, histology
            )
            
            # Assign patient location from cluster trial countries
            patient_location = self.assign_patient_location(cluster_countries)
            
            # Generate comprehensive patient profile
            patient_profile = self._generate_comprehensive_profile(
                criteria_info['structured_criteria'],
                age, gender, histology, mutation, stage, biomarker_name,
                secondary_conditions, lab_values, patient_location
            )
            
            # Increment total patients for prevalence tracking
            self.validator.total_patients_generated += 1
            
            logger.warning('='*50)
            logger.warning(f"Generated Patient {i+1} of biomarker {biomarker_name}.")
            logger.warning('='*50)
            
            # Add metadata for evaluation
            patient_profile.update({
                "cluster_id": cluster_id,
                "biomarker": biomarker_name,
                #"ground_truth_trials": [i+int(len(cluster_trials)*biomarker) for i in range(len(cluster_trials))],
                "patient_id": f"{biomarker_name}-P{i+1:03d}"
            })
            
            patients.append(patient_profile)
        
        # Print final condition prevalences for this cluster
        self._print_condition_summary(num_patients)
        
        return patients
    
    def _print_condition_summary(self, total_patients: int):
        """Print summary of condition distributions"""
        logger.warning("\nCondition Distribution Summary:")
        for condition, count in self.validator.condition_tracker.items():
            actual_rate = count / total_patients
            target_rate = self.validator.condition_prevalences.get(condition, 
                         type('obj', (object,), {'prevalence': 0})).prevalence
            logger.warning(f"{condition}: {count}/{total_patients} ({actual_rate:.1%}) - Target: {target_rate:.1%}")
    
    def _generate_realistic_age(self) -> int:
        """Generate realistic age distribution for lung cancer patients."""
        return max(18, int(np.random.normal(67, 12)))
    
    def _generate_realistic_gender(self) -> str:
        """Generate realistic gender distribution."""
        return random.choices(['male', 'female'], weights=[0.55, 0.45])[0]
    
    def _generate_comprehensive_profile(self, structured_criteria: str, age: int, 
                                  gender: str, histology: str, mutation: str, 
                                  stage: str, biomarker: str,
                                  secondary_conditions: List[str],
                                  lab_values: Dict[str, str],
                                  patient_location: Dict[str, str]) -> Dict:
        """Generate a comprehensive patient profile with clinical accuracy."""
        
        # Format for LLM prompt
        conditions_text = ", ".join(secondary_conditions) if secondary_conditions else "none"
        lab_text = ", ".join([f"{k}: {v}" for k, v in lab_values.items()]) if lab_values else "none"
        location_text = f"{patient_location['city']}, {patient_location['country']}"
        
        # Get biomarker-specific treatment info
        biomarker_info = self.biomarker_params.get(biomarker, {})
        common_drugs = biomarker_info.get('common_drugs', [])
        drugs_text = ", ".join(common_drugs) if common_drugs else "targeted therapy"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a medical expert creating detailed patient profiles.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Create a comprehensive clinical profile for a lung cancer patient with these characteristics:

        PATIENT DEMOGRAPHICS:
        - Age: {age} years
        - Gender: {gender}
        - Location: {location_text}

        PRIMARY DIAGNOSIS:
        - Cancer type: {histology}
        - Stage: {stage}
        - Biomarker: {mutation}
        - Performance status: ECOG {random.choice([0, 1])}

        COMORBIDITIES (Medical History):
        {conditions_text}

        LABORATORY VALUES:
        {lab_text}

        TREATMENT CONTEXT:
        - Patient has {stage} disease requiring systemic therapy
        - Biomarker status ({mutation}) may indicate eligibility for {drugs_text}

        INSTRUCTIONS:
        Write a detailed clinical profile that includes:
        1. Patient demographics (age, gender, location)
        2. Primary diagnosis with specific histology, stage, and biomarker mutation
        3. ECOG performance status
        4. Medical history with specific comorbidities (for dates, pick random but realistic years)
        5. Current medications for comorbidities (with specific drug names and dosages where appropriate.)
        6. Laboratory findings (hemoglobin, platelet counts, etc.)
        7. Prior cancer treatments if stage IV (mention specific chemotherapy regimens, cycles completed, response)
        8. Current disease status or recent imaging findings
        9. Treatment-related symptoms or side effects if applicable

        Include specific medication names, dosages, and treatment details.
        Write ONLY complete, factual sentences with factual information.
        DO NOT use placeholders like [current_year] or [specific medication name] [dosage] [frequency], and instead pick actual and realistic values.
        DO NOT include recommendations, future plans, or signature blocks.

        Example style:
        The patient is a 65-year-old male from Paris, France.
        He was diagnosed with lung adenocarcinoma in March 2023.
        Molecular testing revealed a KRAS G12C mutation.
        The disease is classified as stage IV with metastases to the liver and bone.
        His ECOG performance status is 1.
        He has a history of chronic obstructive pulmonary disease, managed with albuterol 90 mcg inhaler twice daily.
        He also has type 2 diabetes mellitus, controlled with metformin 1000 mg twice daily.
        For hypertension, he takes lisinopril 10 mg daily.
        For his lung cancer, he previously received four cycles of carboplatin (AUC 5) and pemetrexed (500 mg/m²).
        He completed chemotherapy in July 2023 with partial response.
        Recent CT imaging showed disease progression with new liver lesions.
        His hemoglobin level is 11.2 g/dL.
        He reports moderate fatigue but maintains good functional status.

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            if hasattr(response, 'content'):
                raw_content = response.content
            elif isinstance(response, str):
                raw_content = response
            else:
                raw_content = str(response)

            raw_description = raw_content.strip()
            raw_description = re.sub(r"^Human[:\-]?\s*", "", raw_description)
            
        except Exception as e:
            logger.warning(f"Patient generation failed: {e}")
            raw_description = f"The patient is a {age}-year-old {gender} from {location_text}. Diagnosed with {histology}, {stage}. Molecular testing revealed {mutation}."
        
        # Extract structured info
        try:
            extracted_age, extracted_gender = extract_age_gender_from_summary(raw_description)
            main_condition = extract_main_condition_from_summary(raw_description)
            
            # Get synonyms for main condition
            synonyms = self.validator.get_valid_lung_cancer_synonyms(histology)

            if not main_condition or 'lung' not in main_condition.lower():
                main_condition = histology
            
            # Split by newlines for natural sentences
            sentences = [s.strip() for s in raw_description.split('\n') if s.strip() and len(s.strip()) > 10]
            
            # Filter out formatting artifacts
            sentences = [s for s in sentences if not re.match(r'^[\*\-\+\#\s]+', s)]
            
            # Validate sentences don't mention uncaptured conditions
            validated_sentences = self.validator.validate_and_fix_expanded_sentences(
                sentences, [main_condition] + secondary_conditions
            )
            
        except Exception as e:
            logger.warning(f"Extraction failed for patient: {e}")
            extracted_age, extracted_gender = age, gender
            main_condition = histology
            synonyms = self.validator.get_valid_lung_cancer_synonyms(histology)
            validated_sentences = [raw_description]
        
        # Build final structure with CLEAN other_conditions
        return {
            "main_conditions": [main_condition] + synonyms,
            "other_conditions": secondary_conditions,  # ONLY comorbidities - NO age/gender/stage/biomarker
            "expanded_sentences": validated_sentences,
            # Metadata (not used in search, but useful for tracking)
            "age": extracted_age or age,
            "gender": extracted_gender or gender,
            "biomarker_mutation": mutation,
            "cancer_stage": stage,
            "location": patient_location,
            "histology": histology
        }
    
    def validate_patient_quality(self, patients: List[Dict], cluster_trials: List[Dict]) -> Dict:
        """Validate the quality of generated patients with clinical accuracy checks."""
        validation_results = {
            'total_patients': len(patients),
            'age_distribution': {},
            'gender_distribution': {},
            'condition_distribution': {},
            'clinical_accuracy_issues': [],
            'condition_prevalence_check': {}
        }
        
        # Basic distributions
        ages = [p.get('age', 0) for p in patients if p.get('age')]
        genders = [p.get('gender', '') for p in patients if p.get('gender')]
        
        validation_results['age_distribution'] = {
            'mean': np.mean(ages) if ages else 0,
            'std': np.std(ages) if ages else 0,
            'min': min(ages) if ages else 0,
            'max': max(ages) if ages else 0
        }
        
        validation_results['gender_distribution'] = dict(Counter(genders))
        
        # Check condition prevalences
        secondary_conditions = defaultdict(int)
        for patient in patients:
            for condition in patient.get('other_conditions', []):
                secondary_conditions[condition] += 1
        
        total = len(patients)
        for condition, count in secondary_conditions.items():
            actual_rate = count / total
            validation_results['condition_prevalence_check'][condition] = {
                'count': count,
                'rate': actual_rate
            }
        
        # Clinical accuracy checks
        for i, patient in enumerate(patients):
            main_conditions = patient.get('main_conditions', [])
            
            # Check for mutually exclusive lung cancer types
            has_nsclc = any('non-small cell' in s.lower() or s.lower() == 'nsclc' for s in main_conditions)
            has_sclc = any('small cell' in s.lower() and 'non-small cell' not in s.lower() for s in main_conditions)
            
            if has_nsclc and has_sclc:
                validation_results['clinical_accuracy_issues'].append(
                    f"Patient {i}: Has both NSCLC and SCLC (mutually exclusive)"
                )
            
            # Check for conflicting histologies
            histology_types = []
            for cond in main_conditions:
                cond_lower = cond.lower()
                if 'adenocarcinoma' in cond_lower:
                    histology_types.append('adenocarcinoma')
                elif 'squamous' in cond_lower:
                    histology_types.append('squamous')
                elif 'large cell' in cond_lower:
                    histology_types.append('large cell')
            
            unique_histologies = set(histology_types)
            if len(unique_histologies) > 1:
                validation_results['clinical_accuracy_issues'].append(
                    f"Patient {i}: Has multiple conflicting histologies: {unique_histologies}"
                )
        
        return validation_results

def process_hierarchical_clusters(clusters: List[int], metadata: List[Dict], 
                                trials_data: List[Dict], output_file: str, 
                                patients_per_cluster: int = 100):
    """Process hierarchical clustering results to generate synthetic patients."""
    
    generator = ImprovedPatientGenerator(llm)
    all_results = {}
    biomarker_to_indices = defaultdict(list)
    
    for idx, biomarker in enumerate(clusters):
        biomarker_to_indices[biomarker].append(idx)
    
    for biomarker, trial_indices in biomarker_to_indices.items():
        cluster_id = f"cluster_{biomarker}"
        
        logger.warning(f"\n{'='*50}")
        logger.warning(f"Processing cluster n°{biomarker}")
        logger.warning(f"{'='*50}")
        
        cluster_trials = [trials_data[idx] for idx in trial_indices]

        patients = generator.generate_biomarker_aware_patients(
            cluster_trials, biomarker, cluster_id, patients_per_cluster
        )
        
        validation = generator.validate_patient_quality(patients, cluster_trials)
        
        logger.warning(f"\nGenerated {len(patients)} patients for biomarker {biomarker}")
        logger.warning(f"Age distribution: {validation['age_distribution']}")
        logger.warning(f"Gender distribution: {validation['gender_distribution']}")
        
        if validation['clinical_accuracy_issues']:
            logger.warning(f"Clinical accuracy issues: {len(validation['clinical_accuracy_issues'])}")
            for issue in validation['clinical_accuracy_issues'][:5]:
                logger.warning(f"  {issue}")
        
        all_results[cluster_id] = {
            'biomarker': biomarker,
            'cluster_size': len(cluster_trials),
            'cluster_indices': trial_indices,
            'patients': patients,
            'validation': validation
        }
    
    # Save all results to output file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.warning(f"\nResults saved to {output_file}")
    
    total_patients = sum(len(result['patients']) for result in all_results.values())
    logger.warning(f"\nSUMMARY:")
    logger.warning(f"Total clusters processed: {len(all_results)}")
    logger.warning(f"Total patients generated: {total_patients}")
    logger.warning(f"Average patients per cluster: {total_patients / len(all_results):.1f}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Generate synthetic patients from clustered trials.")
    parser.add_argument('--clusters_file', type=str, default="../../data/clusters_metadata.csv", help='Path to clusters+metadata csv file')
    parser.add_argument('--trials_file', type=str, default="../../data/cluster_trials", help='Path to trials JSON folder')
    parser.add_argument('--output_file', type=str, default='../data/generated_patient_dataset.json', help='Output JSON file for patients')
    parser.add_argument('--patients_per_cluster', type=int, default=100, help='Number of patients to generate per cluster')
    
    args = parser.parse_args()
    
    # Load clustering results
    clustering_df = pd.read_csv(args.clusters_file)
    logger.warning(clustering_df.head())
    clustering_data = {
        'clusters': clustering_df['Cluster ID'].tolist(),
        'metadata': clustering_df['Biomarker'].to_dict()
    }
    logger.warning(f"Loaded {len(clustering_data['clusters']) // 3} clusters from {args.clusters_file}")
    
    clusters = clustering_data['clusters']
    metadata = clustering_data['metadata']
    
    # Load trials data from the folder
    trials_data = []
    trials_folder = os.path.abspath(args.trials_file)
    for filename in os.listdir(trials_folder):
        if filename.startswith('NCT') and filename.endswith('.json'):
            with open(os.path.join(trials_folder, filename), 'r') as f:
                trial = json.load(f)
                trials_data.append(trial)
    
    logger.warning(f"Loaded {len(trials_data)} trials from {trials_folder}")
    
    # Process clusters and generate patients
    process_hierarchical_clusters(
        clusters=clusters,
        metadata=metadata,
        trials_data=trials_data,
        output_file=args.output_file,
        patients_per_cluster=args.patients_per_cluster
    )

    # Convert to pipeline format and save
    pipeline_output_file = args.output_file.replace('.json', '_pipeline.json')
    convert_cluster_file_to_pipeline_file(args.output_file, pipeline_output_file)