import re
from typing import List, Dict, Tuple, Optional
from langchain.schema import HumanMessage
import ast

def safe_parse_list(response_content: str) -> List[str]:
    """
    Safely parses a string that should contain a Python list.
    """
    try:
        # Try to find a list-like pattern in the response
        list_pattern = r'\[.*?\]'
        match = re.search(list_pattern, response_content, re.DOTALL)
        if match:
            list_str = match.group(0)
            parsed = ast.literal_eval(list_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
    except:
        pass
    
    # Fallback: try to extract items from numbered or bulleted list
    lines = response_content.split('\n')
    items = []
    for line in lines:
        line = line.strip()
        # Match patterns like "1. item", "- item", "• item"
        if re.match(r'^[\d\-\•]\s*\.?\s+(.+)$', line):
            item = re.sub(r'^[\d\-\•]\s*\.?\s+', '', line).strip()
            if item and item not in items:
                items.append(item)
    
    return items if items else []

def extract_conditions_from_trial(trial_data: Dict) -> List[str]:
    """
    Extract all conditions/diseases from trial data structure.
    """
    conditions = []
    
    # Try multiple possible locations for conditions
    if 'protocolSection' in trial_data:
        protocol = trial_data['protocolSection']
        
        # Check conditionsModule
        conditions_module = protocol.get('conditionsModule', {})
        trial_conditions = conditions_module.get('conditions', [])
        
        if isinstance(trial_conditions, list):
            conditions.extend(trial_conditions)
        elif isinstance(trial_conditions, str):
            conditions.append(trial_conditions)
        
        # Also check keywords
        keywords_module = protocol.get('conditionsModule', {}).get('keywords', [])
        if keywords_module:
            if isinstance(keywords_module, list):
                conditions.extend(keywords_module)
            elif isinstance(keywords_module, str):
                conditions.append(keywords_module)
        
        # Check brief title and summary for additional conditions
        identification = protocol.get('identificationModule', {})
        brief_title = identification.get('briefTitle', '')
        
        description = protocol.get('descriptionModule', {})
        brief_summary = description.get('briefSummary', '')
        
        # Extract condition mentions from text
        condition_patterns = [
            r'lung cancer', r'NSCLC', r'non-small cell lung cancer',
            r'adenocarcinoma', r'squamous cell carcinoma', r'SCLC',
            r'small cell lung cancer', r'mesothelioma', r'carcinoma'
        ]
        
        for text in [brief_title, brief_summary]:
            for pattern in condition_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    conditions.append(pattern.replace(r'\\', ''))
    
    # Legacy format fallback
    elif 'conditions' in trial_data:
        legacy_conditions = trial_data['conditions']
        if isinstance(legacy_conditions, list):
            conditions.extend(legacy_conditions)
        elif isinstance(legacy_conditions, str):
            conditions.append(legacy_conditions)
    
    # Clean and deduplicate
    cleaned_conditions = []
    for condition in conditions:
        if condition and isinstance(condition, str):
            condition = condition.strip()
            if condition and condition not in cleaned_conditions:
                cleaned_conditions.append(condition)
    
    return cleaned_conditions if cleaned_conditions else ["Lung Cancer"]

def extract_eligibility_criteria_from_trial(trial_data: Dict) -> str:
    """
    Extract eligibility criteria from trial data structure.
    """
    eligibility_text = ""
    
    if 'protocolSection' in trial_data:
        protocol = trial_data['protocolSection']
        eligibility_module = protocol.get('eligibilityModule', {})
        eligibility_text = eligibility_module.get('eligibilityCriteria', '')
        
        # Also include other relevant eligibility info
        age_info = eligibility_module.get('stdAges', [])
        sex_info = eligibility_module.get('sex', '')
        healthy_volunteers = eligibility_module.get('healthyVolunteers', '')
        
        additional_info = []
        if age_info:
            if isinstance(age_info, list):
                additional_info.append(f"Age groups: {', '.join(age_info)}")
            else:
                additional_info.append(f"Age: {age_info}")
        
        if sex_info:
            additional_info.append(f"Sex: {sex_info}")
        
        if healthy_volunteers:
            additional_info.append(f"Healthy volunteers: {healthy_volunteers}")
        
        if additional_info:
            eligibility_text += " " + " ".join(additional_info)
    
    # Legacy format fallback
    elif 'eligibility_criteria' in trial_data:
        eligibility_text = trial_data['eligibility_criteria']
    
    return eligibility_text.strip() if eligibility_text else "No specific eligibility criteria provided."

def extract_trial_summary_info(trial_data: Dict) -> Dict[str, str]:
    """
    Extract key summary information from trial data.
    """
    summary_info = {
        'brief_title': '',
        'brief_summary': '',
        'detailed_description': '',
        'study_type': '',
        'phase': '',
        'primary_purpose': ''
    }
    
    if 'protocolSection' in trial_data:
        protocol = trial_data['protocolSection']
        
        # Identification info
        identification = protocol.get('identificationModule', {})
        summary_info['brief_title'] = identification.get('briefTitle', '')
        
        # Description info
        description = protocol.get('descriptionModule', {})
        summary_info['brief_summary'] = description.get('briefSummary', '')
        summary_info['detailed_description'] = description.get('detailedDescription', '')
        
        # Design info
        design = protocol.get('designModule', {})
        summary_info['study_type'] = design.get('studyType', '')
        
        phases = design.get('phases', [])
        if phases:
            summary_info['phase'] = ', '.join(phases) if isinstance(phases, list) else str(phases)
        
        design_info = design.get('designInfo', {})
        summary_info['primary_purpose'] = design_info.get('primaryPurpose', '')
    
    # Legacy format fallback
    else:
        summary_info['brief_title'] = trial_data.get('brief_title', '')
        summary_info['brief_summary'] = trial_data.get('brief_summary', '')
        summary_info['detailed_description'] = trial_data.get('detailed_description', '')
        summary_info['study_type'] = trial_data.get('study_type', '')
        summary_info['phase'] = trial_data.get('phase', '')
    
    return summary_info

def generate_synonyms_for_condition(condition: str, llm) -> List[str]:
    """
    Generates synonyms for a medical condition using LLM.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a medical expert generating synonyms for medical conditions.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Generate 8-10 well-known synonyms or alternative names for: {condition}

        Provide output as a Python list:
        ["synonym1", "synonym2", "synonym3", ...]

        Focus on:
        - Medical synonyms
        - Common abbreviations  
        - Alternative clinical terms
        - Related diagnostic terms
        
        Additionally, avoid using sentences like 'Based on the patient's profile' that don't contribute to a compact profile.
        Do not include explanatory text, just the list.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response = re.sub('^Human[:\-]?\s*', '', response).strip()
        print(f"Synonyms response for '{condition}':", response)
        synonyms = safe_parse_list(response)
        
        if isinstance(synonyms, list) and len(synonyms) > 0:
            # Clean and validate synonyms
            cleaned_synonyms = []
            for s in synonyms[:10]:  # Limit to 10
                if isinstance(s, str) and s.strip() and s.strip() != condition:
                    cleaned_synonyms.append(s.strip())
            return cleaned_synonyms
        else:
            print(f"Could not parse synonyms for condition '{condition}'.")
            return [condition]
    except Exception as e:
        print(f"Error generating synonyms for condition '{condition}': {e}")
        return [condition]

def generate_patient_summary_from_trial(trial_data: Dict, biomarker: str, 
                                       age: int, gender: str, llm) -> str:
    """
    Generate a comprehensive patient summary that matches trial eligibility criteria.
    """
    # Extract trial information
    conditions = extract_conditions_from_trial(trial_data)
    eligibility_criteria = extract_eligibility_criteria_from_trial(trial_data)
    summary_info = extract_trial_summary_info(trial_data)
    
    # Select primary condition
    primary_condition = conditions[0] if conditions else "Non-small cell lung cancer"
    
    # Create detailed prompt
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are an expert oncologist creating patient case summaries.<|eot_id|><|start_header_id|>user<|end_header_id|>

        Generate a realistic patient note for someone eligible for this trial:

        TRIAL INFO:
        - Title: {summary_info['brief_title']}
        - Primary Condition: {primary_condition}
        - Biomarker: {biomarker}
        - Phase: {summary_info['phase']}

        ELIGIBILITY CRITERIA:
        {eligibility_criteria[:1000]}

        PATIENT REQUIREMENTS:
        - Age: {age} years
        - Gender: {gender}
        - Biomarker: {biomarker}

        Include:
        1. Demographics and presentation (Diversify the names, as 'John Smith' and 'Emily Wilson' are overused in your outputs)
        2. Medical history
        3. Current disease status and staging
        4. Biomarker results ({biomarker})
        5. Previous treatments
        6. Performance status
        7. Lab values and imaging
        8. Social history

        Write as a clinical note demonstrating eligibility without explicitly referencing the trial.

        End with: Age: {age}, Gender: {gender}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"Patient summary generated for {biomarker} patient")
        response = response.strip()
        return re.sub(r"^Human[:\-]?\s*", "", response)
    
    except Exception as e:
        print(f"Error generating patient summary: {e}")
        # Fallback simple summary
        return f"""A {age}-year-old {gender} patient diagnosed with {primary_condition}. The patient has confirmed {biomarker} mutation and meets the eligibility criteria for targeted therapy trials. Current disease status shows advanced stage with good performance status. Previous standard treatments have been completed.

Age: {age}, Gender: {gender}"""

def extract_age_gender_from_summary(summary: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract age and gender from patient summary."""
    
    # Try birth date pattern first
    birth_pattern = r"Date of Birth:\s*\w+\s+\d+,\s*(\d{4})"
    birth_match = re.search(birth_pattern, summary)
    if birth_match:
        birth_year = int(birth_match.group(1))
        current_year = 2024  # Adjust as needed
        age = current_year - birth_year
    else:
        # Existing age patterns...
        age_patterns = [
            r"Age:\s*(\d+)\s*,\s*Gender:\s*([A-Za-z]+)",  # Your target format
            r"(\d+)[-\s]*year[-\s]*old",
            r"age[:\s]*(\d+)",
        ]
        age = None
        for pattern in age_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                age = int(match.group(1))
                break
    
    # Gender extraction
    gender_patterns = [
        r"Gender:\s*([A-Za-z]+)",  # Direct format
        r"\b(male|female|man|woman)\b"
    ]
    
    gender = None
    for pattern in gender_patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            found_gender = match.group(1).lower()
            if found_gender in ['male', 'man']:
                gender = 'male'
            elif found_gender in ['female', 'woman']:
                gender = 'female'
            break
    
    return age, gender

def extract_main_condition_from_summary(summary: str) -> str:
    """
    Extract the main medical condition from patient summary using NLP patterns.
    """
    # Common lung cancer terms to look for
    lung_cancer_patterns = [
        r"non[-\s]*small[-\s]*cell lung cancer",
        r"NSCLC",
        r"lung adenocarcinoma",
        r"pulmonary adenocarcinoma",
        r"lung cancer",
        r"lung carcinoma",
        r"bronchogenic carcinoma",
        r"squamous cell carcinoma of lung",
        r"small cell lung cancer",
        r"SCLC"
    ]
    
    # Look for diagnostic patterns
    diagnostic_patterns = [
        r"diagnosed with ([^.]+)",
        r"history of ([^.]+)",
        r"presenting with ([^.]+)",
        r"confirmed ([^.]+)",
        r"stage \w+ ([^.]+)"
    ]
    
    summary_lower = summary.lower()
    
    # First, try specific lung cancer patterns
    for pattern in lung_cancer_patterns:
        if re.search(pattern, summary_lower):
            match = re.search(pattern, summary_lower)
            if match:
                return match.group(0).title()
    
    # Then try general diagnostic patterns
    for pattern in diagnostic_patterns:
        match = re.search(pattern, summary_lower)
        if match:
            condition = match.group(1).strip()
            # Clean up the condition text
            condition = re.sub(r'[,;].*$', '', condition)  # Remove everything after comma/semicolon
            if len(condition) > 5 and 'lung' in condition:
                return condition.title()
    
    # Default fallback
    return "Non-small cell lung cancer"

def generate_related_conditions(main_condition: str, llm) -> List[str]:
    """
    Generate related conditions and comorbidities that might be relevant.
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a medical expert generating related conditions.<|eot_id|><|start_header_id|>user<|end_header_id|>

        For a patient with {main_condition}, list 5-8 related conditions, comorbidities, or secondary diagnoses.

        Provide ONLY as a Python list:
        ["condition1", "condition2", "condition3", ...]

        Focus on:
        - Common comorbidities
        - Related symptoms/syndromes
        - Secondary conditions
        - Staging-related terms
        
        Additionally, avoid using sentences like 'Based on the patient's profile' that don't contribute to a compact profile. Do not include explanatory text, just the list.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>

        """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"Related conditions response for '{main_condition}': {response}")
        conditions = safe_parse_list(response)
        
        if isinstance(conditions, list) and len(conditions) > 0:
            cleaned_conditions = []
            for condition in conditions[:8]:  # Limit to 8
                if isinstance(condition, str) and condition.strip():
                    cleaned_conditions.append(condition.strip())
            return cleaned_conditions
        else:
            print(f"Could not parse related conditions for '{main_condition}'.")
            return [main_condition]
    except Exception as e:
        print(f"Error generating related conditions for '{main_condition}': {e}")
        return [main_condition, "Metastatic disease", "Advanced cancer"]

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling medical abbreviations properly.
    """
    # Common medical abbreviations that shouldn't trigger sentence splits
    abbreviations = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'vs.', 'etc.', 'i.e.', 'e.g.',
        'ECOG', 'IV', 'III', 'II', 'CT', 'MRI', 'PET', 'mg.', 'kg.',
        'cm.', 'mm.', 'ml.', 'mcg.', 'pts.', 'pt.'
    ]
    
    # Temporarily replace abbreviations
    temp_text = text
    for i, abbrev in enumerate(abbreviations):
        temp_text = temp_text.replace(abbrev, f"ABBREV{i}")
    
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', temp_text)
    
    # Restore abbreviations and clean up
    final_sentences = []
    for sentence in sentences:
        for i, abbrev in enumerate(abbreviations):
            sentence = sentence.replace(f"ABBREV{i}", abbrev)
        
        sentence = sentence.strip()
        if sentence:
            final_sentences.append(sentence)
    
    return final_sentences

# Example usage function that integrates with your trial data
def create_patient_from_trial_data(trial_data: Dict, biomarker: str, 
                                  age: int, gender: str, llm) -> Dict:
    """
    Create a complete patient profile from trial data structure.
    """
    # Generate patient summary
    raw_description = generate_patient_summary_from_trial(
        trial_data, biomarker, age, gender, llm
    )
    
    # Extract structured information
    extracted_age, extracted_gender = extract_age_gender_from_summary(raw_description)
    main_condition = extract_main_condition_from_summary(raw_description)
    
    # Generate synonyms and related conditions
    synonyms = generate_synonyms_for_condition(main_condition, llm)
    related_conditions = generate_related_conditions(main_condition, llm)
    
    # Combine main condition with related conditions
    all_conditions = [main_condition]
    for condition in related_conditions:
        if condition not in all_conditions:
            all_conditions.append(condition)
    
    # Split description into sentences
    sentences = split_into_sentences(raw_description)
    
    return {
        "raw_description": raw_description,
        "age": extracted_age or age,
        "gender": extracted_gender or gender,
        "main_condition": main_condition,
        "synonyms": synonyms,
        "conditions": all_conditions,
        "split_raw_description": sentences,
        "biomarker_mutation": biomarker,
        "source_trial_conditions": extract_conditions_from_trial(trial_data),
        "source_trial_title": extract_trial_summary_info(trial_data)['brief_title']
    }