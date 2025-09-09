import os
import json
import random
from typing import List, Dict, Tuple, Any
import re
import numpy as np
from collections import Counter, defaultdict
from .utils.generation_utils import (
    extract_age_gender_from_summary,
    extract_main_condition_from_summary,
    generate_synonyms_for_condition,
    generate_related_conditions,
    split_into_sentences,
)
from .utils.cluster_utils import convert_cluster_file_to_pipeline_file
# unload previous import and reload for debugging
# import importlib

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

class PatientGenerator:
    def __init__(self, llm):
        self.llm = llm
        
        # Biomarker-specific parameters
        self.biomarker_params = {
            'KRAS': {
                'mutations': ['KRAS G12C', 'KRAS G12D', 'KRAS G12V', 'KRAS G13D'],
                'common_drugs': ['sotorasib', 'adagrasib'],
                'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic']
            },
            'EGFR': {
                'mutations': ['EGFR exon 19 deletion', 'EGFR L858R', 'EGFR T790M'],
                'common_drugs': ['osimertinib'],
                'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic']
            },
            'ROS1': {
                'mutations': ['ROS1 fusion', 'ROS1 rearrangement'],
                'common_drugs': ['repotrectinib', 'crizotinib', 'entrectinib', 'lorlatinib'],
                'stage_preference': ['Stage IV', 'Stage IIIB', 'Advanced', 'Metastatic']
            }
        }
    
    def extract_structured_criteria(self, trials: List[Dict]) -> Dict:
        """
        Extract structured criteria from trials using LLM analysis.
        """
        all_criteria = []
        all_conditions = []
        
        for trial in trials:
            # Extract from different possible fields
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
        
        # Use LLM to extract structured criteria
        combined_criteria = " ".join(all_criteria)
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a clinical trial expert.<|eot_id|><|start_header_id|>user<|end_header_id|>
          Analyze the following eligibility criteria and extract key requirements in a structured format.

            Eligibility Criteria:
            {combined_criteria[:2000]}...

            Please extract and format the following information:
            1. Age requirements (e.g., "18 years or older")
            2. Cancer stage requirements (e.g., "Stage IV", "Advanced", "Metastatic")
            3. Performance status requirements (e.g., "ECOG 0-1")
            4. Required biomarker status
            5. Prior treatment requirements
            6. Exclusion criteria (major contraindications)

            Format your response as a structured summary with clear categories. 
            The response should only include the requested information and no additional text like 'thank you' or 'sincerely'. Additionally, avoid using sentences like 'Based on the patient's profile' that don't contribute to a compact profile.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Extract actual content from LangChain response
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
            structured_criteria = combined_criteria[:500] + "..."
        
        return {
            'structured_criteria': structured_criteria,
            'all_conditions': list(set(all_conditions)),
            'raw_criteria': combined_criteria
        }
    
    def generate_biomarker_aware_patients(self, cluster_trials: List[Dict], 
                                        biomarker: str, cluster_id: str, 
                                        num_patients: int = 100) -> List[Dict]:
        """
        Generate patients with biomarker-specific awareness.
        """
        # Extract structured criteria
        criteria_info = self.extract_structured_criteria(cluster_trials)

        # Get biomarker specific info
        biomarker_names = {0: 'KRAS', 1: 'EGFR', 2: 'ROS1'}
        biomarker_name = biomarker_names.get(biomarker, f'Biomarker_{biomarker}')
        biomarker_info = self.biomarker_params.get(biomarker_name, {})
        mutation = random.choice(biomarker_info.get('mutations', [biomarker_name])) if biomarker_info else biomarker_name

        logger.warning(f"\nCluster {cluster_id} ({biomarker_name}) - Structured Criteria:")
        logger.warning(criteria_info['structured_criteria'])
        logger.warning(f"Conditions: {criteria_info['all_conditions'][:5]}...")  # Show first 5
        
        # Get biomarker-specific parameters
        biomarker_info = self.biomarker_params.get(biomarker, {})
        
        patients = []
        for i in range(num_patients):
            logger.warning(f"\nGenerating patient {i+1}/{num_patients} for biomarker {biomarker_name}...\n")

            # Generate realistic age distribution (lung cancer typically older patients)
            age = self._generate_realistic_age()
            
            # Gender distribution (lung cancer slight male predominance)
            gender = self._generate_realistic_gender()
            
            # Select condition
            condition = self._select_condition(criteria_info['all_conditions'])
            
            # Select biomarker mutation if available
            mutation = random.choice(biomarker_info.get('mutations', [biomarker])) if biomarker_info else biomarker
            
            # Select cancer stage
            stage = random.choice(biomarker_info.get('stage_preference', ['Stage IV', 'Advanced']))
            
            # Generate comprehensive patient profile
            patient_profile = self._generate_comprehensive_profile(
                criteria_info['structured_criteria'],
                age, gender, condition, mutation, stage, biomarker_name
            )

            logger.warning('='*50)
            logger.warning(f"Generated Patient {i+1} of biomarker {biomarker_name}.")
            logger.warning('='*50)
            
            # Add metadata for evaluation
            patient_profile.update({
                "cluster_id": cluster_id,
                "biomarker": biomarker_name,
                "ground_truth_trials": [i+int(11*biomarker) for i in range(len(cluster_trials))],  # All trials in cluster are ground truth
                "patient_id": f"{biomarker_name}-P{i+1:03d}"
            })
            
            patients.append(patient_profile)
        
        return patients
    
    def _generate_realistic_age(self) -> int:
        """Generate realistic age distribution for lung cancer patients."""
        # Lung cancer age distribution peaks around 65-70
        return max(18, int(np.random.normal(67, 12)))
    
    def _generate_realistic_gender(self) -> str:
        """Generate realistic gender distribution."""
        return random.choices(['male', 'female'], weights=[0.55, 0.45])[0]
    
    def _select_condition(self, available_conditions: List[str]) -> str:
        """Select most appropriate condition."""
        if not available_conditions:
            return "Non-small cell lung cancer"
        
        # Prioritize lung cancer conditions
        lung_cancer_terms = ['lung cancer', 'nsclc', 'non-small cell', 'adenocarcinoma']
        for condition in available_conditions:
            condition_lower = condition.lower()
            if any(term in condition_lower for term in lung_cancer_terms):
                return condition
        
        return available_conditions[0]
    
    def _generate_comprehensive_profile(self, structured_criteria: str, age: int, 
                                      gender: str, condition: str, mutation: str, 
                                      stage: str, biomarker: str) -> Dict:
        """Generate a comprehensive patient profile using LLM."""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a medical expert creating realistic patient profiles for clinical trials.<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        Generate a realistic lung cancer patient profile that would be eligible for clinical trials based on these criteria:

            Structured Criteria:
            {structured_criteria[:800]}

            Patient Parameters:
            - Age: {age} years
            - Gender: {gender}
            - Primary condition: {condition}
            - Cancer stage: {stage}
            - Biomarker: {mutation}
            - Performance status: ECOG 0-1 (randomly assign)

            CURRENT TREATMENT GUIDELINES TO FOLLOW FOR COMMON CONDITIONS (2024-2025):
            - EGFR+ NSCLC: Osimertinib is first-line standard, erlotinib is outdated
            - KRAS G12C+: Sotorasib or adagrasib are targeted options
            - ROS1+: Repotrectinib is the new generation standard for ROS1 mutations, but if not available Crizotinib, entrectinib, or lorlatinib
            - Hypertension: ACE inhibitors (lisinopril) or ARBs preferred over diuretics
            - COPD: Long-acting bronchodilators (tiotropium) + ICS/LABA combinations


            Generate a detailed patient summary including:
            1. Demographic information
            2. Primary and secondary diagnoses
            3. Current disease stage and biomarker status
            4. Performance status
            5. Relevant medical history
            6. Current medications (if any)
            7. Laboratory values (if relevant)

            Make the patient profile realistic and clinically coherent. The patient should clearly meet the eligibility criteria for {biomarker}-targeted lung cancer trials.
            The response should only include the patient profile and no additional text.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Extract actual content from LangChain response
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
            raw_description = f"{age}-year-old {gender} with {condition}, {stage}, {mutation} positive."
        
        # Use existing utility functions to extract structured data
        try:
            extracted_age, extracted_gender = extract_age_gender_from_summary(raw_description)
            main_condition = extract_main_condition_from_summary(raw_description)
            logger.warning(f"Extracted main condition: {main_condition}")

            try:
                synonyms = generate_synonyms_for_condition(main_condition, llm=self.llm)
                # Check for placeholder responses
                if not synonyms or any('synonym' in str(s).lower() for s in synonyms):
                    synonyms = ["Non-small cell lung cancer", "NSCLC", "Lung adenocarcinoma"]
            except:
                synonyms = ["Non-small cell lung cancer", "NSCLC", "Lung adenocarcinoma"]

            try:
                conditions_from_note = generate_related_conditions(main_condition, llm=self.llm)
                if not conditions_from_note or any('condition' in str(c).lower() for c in conditions_from_note):
                    conditions_from_note = ["Advanced non-small cell lung cancer", "Metastatic lung cancer"]
            except:
                conditions_from_note = ["Advanced non-small cell lung cancer", "Metastatic lung cancer"]
                        
            # Ensure main condition is in the conditions list
            if main_condition not in conditions_from_note:
                conditions = [main_condition] + conditions_from_note
            else:
                conditions = conditions_from_note
                
        except Exception as e:
            logger.warning(f"Extraction failed for patient: {e}")
            extracted_age, extracted_gender = age, gender
            main_condition = condition
            synonyms = [condition]
            conditions = [condition]
        
        return {
            "raw_description": raw_description,
            "age": extracted_age or age,
            "gender": extracted_gender or gender,
            "main_condition": main_condition,
            "synonyms": synonyms,
            "conditions": conditions,
            "split_raw_description": split_into_sentences(raw_description),
            "biomarker_mutation": mutation,
            "cancer_stage": stage
        }
    
    def validate_patient_quality(self, patients: List[Dict], cluster_trials: List[Dict]) -> Dict:
        """Validate the quality of generated patients."""
        validation_results = {
            'total_patients': len(patients),
            'age_distribution': {},
            'gender_distribution': {},
            'condition_distribution': {},
            'validation_errors': []
        }
        
        # Analyze distributions
        ages = [p.get('age', 0) for p in patients if p.get('age')]
        genders = [p.get('gender', '') for p in patients if p.get('gender')]
        conditions = [p.get('main_condition', '') for p in patients if p.get('main_condition')]
        biomarkers = [p.get('biomarker_mutation', '') for p in patients if p.get('biomarker_mutation')]
        
        validation_results['age_distribution'] = {
            'mean': np.mean(ages) if ages else 0,
            'std': np.std(ages) if ages else 0,
            'min': min(ages) if ages else 0,
            'max': max(ages) if ages else 0
        }
        
        validation_results['gender_distribution'] = dict(Counter(genders))
        validation_results['condition_distribution'] = dict(Counter(conditions))
        
        # Check for common issues
        for i, patient in enumerate(patients):
            if not patient.get('raw_description'):
                validation_results['validation_errors'].append(f"Patient {i}: Missing description")
            if not patient.get('age') or patient.get('age') < 18:
                validation_results['validation_errors'].append(f"Patient {i}: Invalid age")
        
        return validation_results

def process_hierarchical_clusters(clusters: List[int], metadata: List[Dict], 
                                trials_data: List[Dict], output_file: str, 
                                patients_per_cluster: int = 100):
    """
    Process hierarchical clustering results to generate synthetic patients.
    
    Args:
        clusters: List of cluster indices from hierarchical clustering
        metadata: List of biomarker values (one per trial)
        trials_data: Path to the folder of JSON trials
        output_file: Path to save results
        patients_per_cluster: Number of patients per cluster
    """
    
    generator = PatientGenerator(llm)
    all_results = {}
    biomarker_to_indices = defaultdict(list)
    for idx, biomarker in enumerate(clusters):
        biomarker_to_indices[biomarker].append(idx)
    
    for biomarker, trial_indices in biomarker_to_indices.items():
        cluster_id = f"cluster_{biomarker}"
        
        logger.warning(f"\n{'='*50}")
        logger.warning(f"Processing cluster n°{biomarker}")
        logger.warning(f"{'='*50}")
        
        # Extract trials for this cluster (where biomarker matches the current cluster_idx)
        cluster_trials = [
            trials_data[idx] for idx in trial_indices
        ]

        # Generate patients
        patients = generator.generate_biomarker_aware_patients(
            cluster_trials, biomarker, cluster_id, patients_per_cluster
        )
        
        # Validate quality
        validation = generator.validate_patient_quality(patients, cluster_trials)
        
        logger.warning(f"\nGenerated {len(patients)} patients for {biomarker}")
        logger.warning(f"Age distribution: {validation['age_distribution']}")
        logger.warning(f"Gender distribution: {validation['gender_distribution']}")
        
        if validation['validation_errors']:
            logger.warning(f"Validation errors: {len(validation['validation_errors'])}")
        
        all_results[cluster_id] = {
            'biomarker': biomarker,
            'cluster_size': len(cluster_trials),
            'cluster_indices': trial_indices,
            'patients': patients,
            'validation': validation
        }
    
    # Create output file and save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.warning(f"\nResults saved to {output_file}")
    
    # Summary statistics
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
    parser.add_argument('--output_file', type=str, default='../../data/generated_patient_dataset.json', help='Output JSON file for patients')
    parser.add_argument('--patients_per_cluster', type=int, default=100, help='Number of patients to generate per cluster')
    
    args = parser.parse_args()
    
    # Load clustering results (which is .csv)

    clustering_df = pd.read_csv(args.clusters_file)
    logger.warning(clustering_df.head())
    clustering_data = {
        'clusters': clustering_df['Cluster ID'].tolist(),
        'metadata': clustering_df['Biomarker'].to_dict()
    }
    logger.warning(f"Loaded {len(clustering_data['clusters']) // 11} clusters from {args.clusters_file}")
    
    clusters = clustering_data['clusters']  # List of trial indices
    metadata = clustering_data['metadata']  # List of dicts with biomarker info
    
    # Load trials data from the folder and create a list of trial dicts
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

# Example command to run:
# python generate_cluster_patients.py --clusters_file path/cluster.csv --trials_file path/trials_folder --output_file cluster_patients_dataset.json --patients_per_cluster 100