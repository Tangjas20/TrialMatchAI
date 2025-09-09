import os
import json
from typing import List, Dict, Tuple, Any
import re
from collections import Counter, defaultdict

def convert_cluster_to_pipeline_format(cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert cluster-based JSON structure to pipeline input format.
    
    Args:
        cluster_data: Dictionary with cluster_0, cluster_1, etc. as keys
        
    Returns:
        Dictionary with patient IDs as keys and pipeline-format patient data as values
    """
    pipeline_format = {}
    
    # Iterate through all clusters
    for cluster_key, cluster_info in cluster_data.items():
        if not isinstance(cluster_info, dict) or 'patients' not in cluster_info:
            continue
            
        # Process each patient in the cluster
        for patient in cluster_info['patients']:
            # Generate patient ID (use existing patient_id if available, otherwise generate one)
            patient_id = f"P{len(pipeline_format) + 1:03d}" # Default TrialMatchAI ID format, without biomarker differentiation
            
            # Convert patient data to pipeline format
            pipeline_patient = convert_patient_to_pipeline_format(patient)
            pipeline_format[patient_id] = pipeline_patient
    
    return pipeline_format

def convert_patient_to_pipeline_format(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single patient from cluster format to pipeline format.
    """
    # Extract main conditions - start with main_condition and add synonyms
    main_conditions = []
    
    # Add the primary condition
    if 'main_condition' in patient:
        main_conditions.append(patient['main_condition'])
    
    # Add synonyms (these are typically variations of the main condition)
    if 'synonyms' in patient and isinstance(patient['synonyms'], list):
        for synonym in patient['synonyms']:
            if synonym not in main_conditions:
                main_conditions.append(synonym)
    
    # Extract other conditions - use the conditions list, excluding the main condition
    other_conditions = []
    
    if 'conditions' in patient and isinstance(patient['conditions'], list):
        for condition in patient['conditions']:
            # Skip the main condition to avoid duplication
            if condition != patient.get('main_condition'):
                other_conditions.append(condition)
    
    # Add biomarker information to other conditions if available
    if 'biomarker_mutation' in patient or 'biomarker' in patient:
        biomarker = patient.get('biomarker', str(patient.get('biomarker_mutation', '')))
        if biomarker and biomarker != '0':
            other_conditions.append(f"{biomarker} mutation")
    
    # Add cancer stage if available
    if 'cancer_stage' in patient:
        other_conditions.append(f"Cancer stage: {patient['cancer_stage']}")
    
    # Extract demographic info and add to other conditions
    if 'age' in patient:
        other_conditions.append(f"Age: {patient['age']} years")
    
    if 'gender' in patient:
        other_conditions.append(f"Gender: {patient['gender']}")
    
    # Generate expanded sentences from split_raw_description or raw_description
    expanded_sentences = []
    
    if 'split_raw_description' in patient and isinstance(patient['split_raw_description'], list):
        # Use the pre-split sentences
        expanded_sentences = patient['split_raw_description']
    elif 'raw_description' in patient:
        # Split the raw description into sentences
        expanded_sentences = split_into_sentences(patient['raw_description'])
    
    # If no sentences were extracted, create some basic ones from available data
    if not expanded_sentences:
        expanded_sentences = generate_fallback_sentences(patient)
    
    return {
        "main_conditions": main_conditions,
        "other_conditions": other_conditions,
        "expanded_sentences": expanded_sentences
    }

def generate_fallback_sentences(patient: Dict[str, Any]) -> List[str]:
    """
    Generate basic sentences from patient data when no description is available.
    """
    sentences = []
    
    # Basic demographic sentence
    age = patient.get('age', 'unknown age')
    gender = patient.get('gender', 'unknown gender')
    main_condition = patient.get('main_condition', 'cancer')
    sentences.append(f"Patient is a {age}-year-old {gender} with {main_condition}.")
    
    # Biomarker sentence
    biomarker = patient.get('biomarker', patient.get('biomarker_mutation'))
    if biomarker and biomarker != '0':
        sentences.append(f"The patient has {biomarker} mutation.")
    
    # Stage sentence
    if 'cancer_stage' in patient:
        sentences.append(f"The cancer is classified as {patient['cancer_stage']} stage.")
    
    # Conditions sentence
    if 'conditions' in patient and len(patient['conditions']) > 1:
        other_conditions = [c for c in patient['conditions'] if c != main_condition][:3]
        if other_conditions:
            sentences.append(f"Associated conditions include {', '.join(other_conditions)}.")
    
    return sentences

def save_pipeline_format(pipeline_data: Dict[str, Any], output_file: str):
    """
    Save the converted data to a JSON file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Overwrite the file if it exists
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_data, f, indent=2, ensure_ascii=False)

def load_cluster_data(input_file: str) -> Dict[str, Any]:
    """
    Load cluster data from a JSON file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_cluster_file_to_pipeline_file(input_file: str, output_file: str):
    """
    Convert a cluster JSON file to pipeline format and save it.
    
    Args:
        input_file: Path to the cluster JSON file
        output_file: Path where the pipeline JSON should be saved
    """
    # Load cluster data
    cluster_data = load_cluster_data(input_file)
    
    # Convert to pipeline format
    pipeline_data = convert_cluster_to_pipeline_format(cluster_data)
    
    # Save converted data
    save_pipeline_format(pipeline_data, output_file)
    
    print(f"Converted {len(pipeline_data)} patients from {input_file} to {output_file}")
    return pipeline_data