import os
import json
from typing import List, Dict, Tuple, Any
import re
from collections import Counter, defaultdict

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, handling medical abbreviations properly."""
    abbreviations = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'vs.', 'etc.', 'i.e.', 'e.g.',
        'ECOG', 'IV', 'III', 'II', 'CT', 'MRI', 'PET', 'mg.', 'kg.',
        'cm.', 'mm.', 'ml.', 'mcg.', 'pts.', 'pt.'
    ]
    
    temp_text = text
    for i, abbrev in enumerate(abbreviations):
        temp_text = temp_text.replace(abbrev, f"ABBREV{i}")
    
    sentences = re.split(r'(?<=[.!?])\s+', temp_text)
    
    final_sentences = []
    for sentence in sentences:
        for i, abbrev in enumerate(abbreviations):
            sentence = sentence.replace(f"ABBREV{i}", abbrev)
        
        sentence = sentence.strip()
        if sentence:
            final_sentences.append(sentence)
    
    return final_sentences

def convert_cluster_to_pipeline_format(cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert cluster-based JSON structure to pipeline input format.
    
    Args:
        cluster_data: Dictionary with cluster_0, cluster_1, etc. as keys
        
    Returns:
        Dictionary with patient IDs as keys and pipeline-format patient data as values
    """
    pipeline_format = {}
    
    for cluster_key, cluster_info in cluster_data.items():
        if not isinstance(cluster_info, dict) or 'patients' not in cluster_info:
            continue
            
        for patient in cluster_info['patients']:
            patient_id = f"P{len(pipeline_format) + 1:03d}"
            pipeline_patient = convert_patient_to_pipeline_format(patient)
            pipeline_format[patient_id] = pipeline_patient
    
    return pipeline_format

def convert_patient_to_pipeline_format(patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single patient from cluster format to pipeline format.
    FIXED: Only includes actual medical conditions in other_conditions.
    """
    # Extract main conditions - start with main_condition and add synonyms
    main_conditions = []
    
    if 'main_conditions' in patient and isinstance(patient['main_conditions'], list):
        main_conditions = patient['main_conditions']

    elif 'main_condition' in patient:
        main_conditions.append(patient['main_condition'])
    
    if 'synonyms' in patient and isinstance(patient['synonyms'], list):
        for synonym in patient['synonyms']:
            if synonym not in main_conditions:
                main_conditions.append(synonym)

    # If still empty, try to build from available data
    if not main_conditions:
        if 'histology' in patient:
            main_conditions.append(patient['histology'])
        if 'main_condition' in patient:
            main_conditions.append(patient['main_condition'])
        # Add generic fallback
        if not main_conditions:
            main_conditions = ["Non-Small Cell Lung Cancer", "NSCLC", "Lung Cancer"]
    
    
    # FIXED: Extract ONLY medical conditions (no demographics, stage, or biomarker)
    other_conditions = []
    
    if 'other_conditions' in patient and isinstance(patient['other_conditions'], list):
        # Use these directly - they should already be clean medical conditions
        other_conditions = [
            cond for cond in patient['other_conditions']
            if is_medical_condition(cond)
        ]
    
    elif 'conditions' in patient and isinstance(patient['conditions'], list):
        for condition in patient['conditions']:
            # Skip the main condition and non-medical items
            if condition != patient.get('main_condition') and is_medical_condition(condition):
                other_conditions.append(condition)
    
    # DO NOT ADD: biomarker, stage, age, gender to other_conditions
    
    # Generate expanded sentences from split_raw_description or raw_description
    expanded_sentences = []
    
    if 'split_raw_description' in patient and isinstance(patient['split_raw_description'], list):
        expanded_sentences = patient['split_raw_description']
    elif 'raw_description' in patient:
        expanded_sentences = split_into_sentences(patient['raw_description'])
    elif 'expanded_sentences' in patient and isinstance(patient['expanded_sentences'], list):
        expanded_sentences = patient['expanded_sentences']
    
    if not expanded_sentences:
        expanded_sentences = generate_fallback_sentences(patient)
    
    return {
        "main_conditions": main_conditions,
        "other_conditions": other_conditions,  # Clean medical conditions only
        "expanded_sentences": expanded_sentences
    }

def is_medical_condition(condition: str) -> bool:
    """
    Check if a string represents an actual medical condition.
    Returns False for demographics, staging, biomarkers, etc.
    """
    condition_lower = condition.lower().strip()
    
    # Patterns that indicate non-medical metadata
    non_medical_patterns = [
        r'^age:?\s*\d+',
        r'^gender:?\s*(male|female)',
        r'^cancer stage:?',
        r'^stage\s+(i|ii|iii|iv|advanced|metastatic)',
        r'\d+\s*years\s*(old)?$',
        r'^ecog\s*\d+',
        r'^performance status',
        r'(kras|egfr|ros1|alk|braf|met|ret|her2)\s*(mutation|fusion|positive|negative)',
        r'^location:',
        r'^biomarker:',
    ]
    
    # Check if it matches any non-medical pattern
    for pattern in non_medical_patterns:
        if re.search(pattern, condition_lower):
            return False
    
    # Additional checks for very short strings or pure demographic info
    if len(condition_lower) < 3:
        return False
    
    if condition_lower in ['male', 'female', 'age', 'gender', 'stage']:
        return False
    
    return True

def generate_fallback_sentences(patient: Dict[str, Any]) -> List[str]:
    """Generate basic sentences from patient data when no description is available."""
    sentences = []
    
    age = patient.get('age', 'unknown age')
    gender = patient.get('gender', 'unknown gender')
    main_condition = patient.get('main_condition', 'cancer')
    
    # Demographics
    sentences.append(f"Patient is a {age}-year-old {gender} with {main_condition}.")
    
    # Biomarker
    biomarker = patient.get('biomarker_mutation', patient.get('biomarker'))
    if biomarker and str(biomarker) != '0':
        sentences.append(f"The patient has {biomarker} mutation.")
    
    # Stage
    if 'cancer_stage' in patient:
        sentences.append(f"The cancer is classified as {patient['cancer_stage']}.")
    
    # Location
    if 'location' in patient:
        location = patient['location']
        if isinstance(location, dict):
            city = location.get('city', '')
            country = location.get('country', '')
            if city and country:
                sentences.append(f"Patient is located in {city}, {country}.")
    
    # Medical conditions
    if 'other_conditions' in patient and isinstance(patient['other_conditions'], list):
        medical_conds = [c for c in patient['other_conditions'] if is_medical_condition(c)]
        if medical_conds:
            sentences.append(f"Comorbidities include {', '.join(medical_conds[:3])}.")
    
    return sentences

def save_pipeline_format(pipeline_data: Dict[str, Any], output_file: str):
    """Save the converted data to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_data, f, indent=2, ensure_ascii=False)

def load_cluster_data(input_file: str) -> Dict[str, Any]:
    """Load cluster data from a JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_cluster_file_to_pipeline_file(input_file: str, output_file: str):
    """
    Convert a cluster JSON file to pipeline format and save it.
    
    Args:
        input_file: Path to the cluster JSON file
        output_file: Path where the pipeline JSON should be saved
    """
    cluster_data = load_cluster_data(input_file)
    pipeline_data = convert_cluster_to_pipeline_format(cluster_data)
    save_pipeline_format(pipeline_data, output_file)
    
    print(f"Converted {len(pipeline_data)} patients from {input_file} to {output_file}")
    
    # Validation check
    contaminated_count = 0
    for patient_id, patient_data in pipeline_data.items():
        for condition in patient_data.get('other_conditions', []):
            if not is_medical_condition(condition):
                contaminated_count += 1
                print(f"⚠️  {patient_id} has contaminated condition: {condition}")
    
    if contaminated_count == 0:
        print("✅ All other_conditions are clean medical conditions")
    else:
        print(f"❌ Found {contaminated_count} contaminated conditions across all patients")
    
    return pipeline_data