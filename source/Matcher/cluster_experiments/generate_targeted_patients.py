# Matcher/generate_targeted_patients.py
"""
Generate highly specific patients designed to match a small set of target trials.

This is used for precision experiments where you want to test if the pipeline
can correctly rank specific trials highly when competing against a larger dataset.
"""

import os
import json
import random
import re
from typing import List, Dict, Tuple, Any
import numpy as np

from langchain_huggingface import HuggingFacePipeline
from langchain.schema import HumanMessage
from transformers import pipeline, logging as transformers_logging

# Import from existing modules
from generate_cluster_patients import (
    ImprovedPatientGenerator,
    ClinicalValidator,
    llm  # Reuse the existing LLM instance
)
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.generation_utils import (
    extract_age_gender_from_summary,
    extract_main_condition_from_summary,
)

logger = setup_logging()
transformers_logging.set_verbosity_error()


class TargetedPatientGenerator(ImprovedPatientGenerator):
    """
    Extended patient generator for creating patients targeted at specific trials.
    
    Inherits from ImprovedPatientGenerator to reuse all existing functionality
    while adding specialized methods for precise trial matching.
    """
    
    def __init__(self, llm):
        super().__init__(llm)
    
    def generate_patient_matching_specific_trials(
        self,
        target_trials: List[Dict],
        biomarker: str = "KRAS",
        num_patients: int = 1,
        output_file: str = None
    ) -> List[Dict]:
        """
        Generate patients specifically designed to match a small set of target trials.
        
        Args:
            target_trials: List of trial dicts (the handpicked trials)
            biomarker: Biomarker name (e.g., "KRAS", "EGFR", "ROS1")
            num_patients: Number of patients to generate
            output_file: Optional path to save results
        
        Returns:
            List of patient dicts with ground_truth_trials set to target trial NCT IDs
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Generating {num_patients} patient(s) for {len(target_trials)} target trials")
        logger.info(f"{'='*70}")
        
        # Step 1: Extract common requirements (what ALL trials need)
        logger.info("Step 1: Extracting common requirements...")
        common_requirements = self._extract_common_requirements(target_trials, biomarker)
        logger.info(f"Common requirements extracted: {json.dumps(common_requirements, indent=2)}")
        
        # Step 2: Extract distinguishing features
        logger.info("\nStep 2: Extracting distinguishing features...")
        distinguishing_features = self._extract_distinguishing_features(target_trials, biomarker)
        logger.info(f"Distinguishing features: {distinguishing_features.get('distinguishing_features', [])}")
        
        # Step 3: Get trial locations
        cluster_countries = self.get_cluster_countries(target_trials)
        logger.info(f"\nStep 3: Trial locations: {cluster_countries}")
        
        # Step 4: Generate patients
        logger.info(f"\nStep 4: Generating {num_patients} patient(s)...")
        patients = []
        trial_ids = [self._extract_nct_id(t) for t in target_trials]
        
        for i in range(num_patients):
            logger.info(f"\nGenerating patient {i+1}/{num_patients}...")
            
            # Demographics
            age = self._generate_realistic_age()
            gender = self._generate_realistic_gender()
            
            # Determine optimal mutation and additional requirements
            mutation, additional_reqs = self._determine_optimal_mutation(
                common_requirements, biomarker, trial_ids
            )

            # Determine treatment status
            is_treatment_naive = self._determine_treatment_status(common_requirements)

            # Use requirements from trials
            stage = common_requirements.get('required_stage', 'Stage IV')
            histology = common_requirements.get('required_histology', 'Adenocarcinoma')

            # Secondary conditions
            secondary_conditions, lab_values = self.validator.generate_secondary_conditions(
                age, gender, histology
            )

            # Location from trial countries
            patient_location = self.assign_patient_location(cluster_countries)

            # Generate highly specific profile
            patient_profile = self._generate_targeted_profile(
                common_requirements=common_requirements,
                distinguishing_features=distinguishing_features,
                age=age,
                gender=gender,
                histology=histology,
                mutation=mutation,
                stage=stage,
                biomarker=biomarker,
                secondary_conditions=secondary_conditions,
                lab_values=lab_values,
                patient_location=patient_location,
                target_trial_ids=trial_ids,
                is_treatment_naive=is_treatment_naive,
                additional_requirements=additional_reqs
            )
            
            # Add metadata
            patient_profile.update({
                "patient_id": f"P{i+1:03d}",
                "biomarker": biomarker,
                "ground_truth_trials": trial_ids,
                "target_trials": trial_ids,
                "experiment_type": "targeted_matching",
                "num_target_trials": len(trial_ids)
            })
            
            patients.append(patient_profile)
            
            logger.info(f"✓ Patient {i+1} generated with {len(patient_profile['expanded_sentences'])} sentences")
        
        # Step 5: Save if requested
        if output_file:
            output_data = {
                f"P{i+1:03d}": patient 
                for i, patient in enumerate(patients)
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"\n✓ Saved {num_patients} patient(s) to {output_file}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Patient generation complete!")
        logger.info(f"Ground truth trials: {trial_ids}")
        logger.info(f"{'='*70}\n")
        
        return patients
    
    def _extract_nct_id(self, trial: Dict) -> str:
        """Extract NCT ID from trial dict."""
        if 'nct_id' in trial:
            return trial['nct_id']
        elif 'protocolSection' in trial:
            return trial['protocolSection'].get('identificationModule', {}).get('nctId', 'Unknown')
        return 'Unknown'
    
    def _extract_common_requirements(self, trials: List[Dict], biomarker: str) -> Dict:
        """
        Extract requirements that ALL trials share (intersection).
        This ensures the patient is eligible for all target trials.
        """
        # Collect all criteria
        all_criteria = []
        for trial in trials:
            nct_id = self._extract_nct_id(trial)
            
            if 'protocolSection' in trial:
                criteria = trial['protocolSection'].get('eligibilityModule', {}).get('eligibilityCriteria', '')
            else:
                criteria = trial.get('eligibility_criteria', '')
            
            all_criteria.append(f"TRIAL {nct_id}:\n{criteria}")
        
        combined = "\n\n" + "="*50 + "\n\n".join(all_criteria)
        
        # Use LLM to extract common requirements
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are extracting common eligibility requirements from clinical trials.<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are eligibility criteria from {len(trials)} {biomarker}-positive NSCLC trials:

{combined[:5000]}

Extract ONLY the requirements that ALL {len(trials)} trials share (the intersection).

You MUST respond with VALID JSON only. No extra text before or after.

{{
    "required_mutation": "specific mutation variant ALL trials accept (e.g., 'KRAS G12C' or 'any KRAS mutation')",
    "required_stage": "stage requirement ALL trials have (e.g., 'Stage IV' or 'Stage IIIB/IV')",
    "required_histology": "histology ALL trials require (e.g., 'NSCLC' or 'Adenocarcinoma')",
    "ecog_requirement": "ECOG ALL trials require (e.g., '0-1' or '0-2')",
    "prior_therapy": "prior therapy requirements common to ALL (e.g., 'treatment-naive' or '1-2 prior lines')",
    "brain_mets_policy": "brain mets policy ALL trials share (e.g., 'allowed if treated' or 'excluded')",
    "measurable_disease": "measurable disease requirement (e.g., 'required per RECIST 1.1')",
    "key_inclusions": [
        "Include 4-7 specific inclusion criteria that ALL trials share",
        "Example: Histologically confirmed NSCLC",
        "Example: Adequate organ function (specify: ANC >1500, platelets >100k, etc.)"
    ],
    "key_exclusions": [
        "Include 4-7 specific exclusion criteria that ALL trials share",
        "Example: Active brain metastases",
        "Example: Prior KRAS G12C inhibitor therapy"
    ]
}}

IMPORTANT: 
- key_inclusions and key_exclusions must have at least 3 items each
- Be SPECIFIC with actual criteria text from the trials
- Output ONLY valid JSON, nothing else

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON - be more aggressive in finding it
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate that key_inclusions and key_exclusions are not empty
                if not result.get('key_inclusions'):
                    result['key_inclusions'] = [
                        f"Confirmed {biomarker}-positive NSCLC by molecular testing",
                        "Measurable disease per RECIST 1.1",
                        "Adequate organ function (ANC ≥1500/μL, platelets ≥100,000/μL, Hgb ≥9 g/dL)",
                        "Life expectancy ≥12 weeks"
                    ]
                
                if not result.get('key_exclusions'):
                    result['key_exclusions'] = [
                        "Active or untreated brain metastases",
                        f"Prior treatment with {biomarker} G12C inhibitor",
                        "Active infection requiring systemic therapy",
                        "Concurrent malignancy requiring active treatment"
                    ]

                for trial in trials:
                    nct_id = self._extract_nct_id(trial)
                    if 'protocolSection' in trial:
                        title = trial['protocolSection'].get('identificationModule', {}).get('briefTitle', '').lower()
                    else:
                        title = trial.get('brief_title', '').lower()
                    
                    # Check for first-line indicators
                    if 'first-line' in title or 'first line' in title or '1l' in title:
                        logger.warning(f"⚠️ Trial {nct_id} is FIRST-LINE - forcing treatment-naive requirement")
                        result['prior_therapy'] = 'treatment-naive (no prior systemic therapy)'
                        break
                
                return result
                
        except Exception as e:
            logger.warning(f"Failed to extract common requirements: {e}")
        
        # Fallback with populated lists
        return {
            "required_mutation": f"{biomarker} mutation",
            "required_stage": "Stage IV",
            "required_histology": "NSCLC",
            "ecog_requirement": "ECOG 0-1",
            "prior_therapy": "1-2 prior lines allowed",
            "brain_mets_policy": "allowed if treated and stable",
            "measurable_disease": "required per RECIST 1.1",
            "key_inclusions": [
                f"Histologically confirmed {biomarker}-positive NSCLC",
                "Measurable disease per RECIST 1.1",
                "Adequate organ function (ANC ≥1500/μL, platelets ≥100,000/μL)",
                "ECOG performance status 0-1"
            ],
            "key_exclusions": [
                "Active brain metastases requiring immediate treatment",
                f"Prior therapy with {biomarker}-targeted agents",
                "Active uncontrolled infection",
                "Concurrent malignancy"
            ]
        }
    
    def _extract_distinguishing_features(self, trials: List[Dict], biomarker: str) -> Dict:
        """
        Extract what makes these trials SPECIAL vs generic trials.
        
        This helps create a patient profile that will make these trials
        rank highly when competing in a larger dataset.
        """
        # Get trial summaries
        trial_summaries = []
        for trial in trials:
            nct_id = self._extract_nct_id(trial)
            
            if 'protocolSection' in trial:
                title = trial['protocolSection'].get('identificationModule', {}).get('briefTitle', '')
                summary = trial['protocolSection'].get('descriptionModule', {}).get('briefSummary', '')
                interventions = trial['protocolSection'].get('armsInterventionsModule', {}).get('interventions', [])
            else:
                title = trial.get('brief_title', '')
                summary = trial.get('brief_summary', '')
                interventions = trial.get('intervention', [])
            
            # Extract intervention names
            intervention_names = []
            if isinstance(interventions, list):
                for interv in interventions:
                    if isinstance(interv, dict):
                        name = interv.get('name', '')
                        if name:
                            intervention_names.append(name)
            
            trial_summaries.append({
                'nct_id': nct_id,
                'title': title[:300],
                'summary': summary[:600],
                'interventions': ', '.join(intervention_names[:5])
            })
        
        summaries_text = "\n\n".join([
            f"NCT ID: {s['nct_id']}\n"
            f"Title: {s['title']}\n"
            f"Summary: {s['summary']}\n"
            f"Interventions: {s['interventions']}"
            for s in trial_summaries
        ])
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are analyzing clinical trial characteristics.<|eot_id|><|start_header_id|>user<|end_header_id|>

I have {len(trials)} {biomarker}-positive NSCLC trials. I need to understand what makes them DISTINCT from other generic {biomarker} trials.

{summaries_text}

What are the KEY DISTINGUISHING FEATURES that would make a patient specifically match THESE trials vs hundreds of other {biomarker} trials?

Consider:
1. Specific drug mechanisms or novel agents
2. Combination therapy approaches
3. Specific biomarker subtype requirements
4. Unique eligibility criteria
5. Specific patient populations targeted
6. Novel endpoints or study design features

List 4-7 SPECIFIC, CONCRETE features that distinguish these trials. Be precise.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, 'content') else str(response)
            
            # Extract bullet points or numbered items
            features = re.findall(r'(?:[-•\d]+\.?\s*)(.+)', raw)
            features = [f.strip() for f in features if len(f.strip()) > 20][:5]
            
            return {
                'distinguishing_features': features,
                'raw_analysis': raw[:600]
            }
        except Exception as e:
            logger.warning(f"Failed to extract distinguishing features: {e}")
            return {
                'distinguishing_features': [
                    f"{biomarker}-positive NSCLC requiring systemic therapy"
                ],
                'raw_analysis': ''
            }

    def _determine_optimal_mutation(self, common_requirements: Dict, biomarker: str, target_trial_ids: List[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Determine the optimal mutation variant that matches ALL trials.
        Also determines additional requirements like PD-L1, HLA type.
        
        Returns:
            Tuple of (mutation_string, additional_requirements_dict)
        """
        required_mutation = common_requirements.get('required_mutation', '').lower()
        additional_reqs = {}
        
        logger.info(f"Determining optimal mutation from requirement: {required_mutation}")
        
        # Check if all trials require G12C specifically
        if biomarker == 'KRAS':
            # If requirement mentions ONLY G12C (not G12D, G12V, G13D)
            if 'g12c' in required_mutation and not any(x in required_mutation for x in ['g12d', 'g12v', 'g13d', '4mut']):
                mutation = 'KRAS G12C'
                logger.info(f"✓ Using KRAS G12C (specific requirement)")
            # If requirement mentions 4MUT+ or multiple variants
            elif '4mut' in required_mutation or all(x in required_mutation for x in ['g12c', 'g12d']):
                # Use G12C as it's most common and matches more trials
                mutation = 'KRAS G12C'
                logger.info(f"✓ Using KRAS G12C (chosen from 4MUT+ options for broader trial match)")
            else:
                # Default to G12C (most common in NSCLC)
                mutation = 'KRAS G12C'
                logger.info(f"✓ Using KRAS G12C (default)")
            
            # Check for specific trial requirements
            trial_ids_str = ' '.join(target_trial_ids)
            
            # NCT06345729 requires PD-L1 ≥50%
            if 'NCT06345729' in trial_ids_str:
                additional_reqs['pdl1_tps'] = random.randint(50, 90)
                logger.info(f"✓ Added PD-L1 TPS requirement: {additional_reqs['pdl1_tps']}%")
            
            # NCT03948763 requires HLA types
            if 'NCT03948763' in trial_ids_str:
                additional_reqs['hla_type'] = random.choice(['HLA-A*11:01', 'HLA-C*08:02'])
                logger.info(f"✓ Added HLA type requirement: {additional_reqs['hla_type']}")
        
        elif biomarker == 'EGFR':
            mutation = random.choice(['EGFR exon 19 deletion', 'EGFR L858R'])
        elif biomarker == 'ROS1':
            mutation = 'ROS1 fusion'
        else:
            mutation = f"{biomarker} mutation"
        
        return mutation, additional_reqs


    def _determine_treatment_status(self, common_requirements: Dict) -> bool:
        """
        Determine if patient should be treatment-naive based on trial requirements.
        
        Returns:
            True if patient should be treatment-naive, False otherwise
        """
        prior_therapy_req = common_requirements.get('prior_therapy', '').lower()
        
        # Check for treatment-naive indicators
        is_treatment_naive = any(keyword in prior_therapy_req for keyword in [
            'treatment-naive',
            'treatment naive', 
            'no prior systemic',
            'first-line',
            'first line',
            '0 prior lines'
        ])
        
        if is_treatment_naive:
            logger.info("✓ Trials require TREATMENT-NAIVE patient (no prior systemic therapy)")
        else:
            logger.info(f"✓ Prior therapy allowed: {prior_therapy_req}")
        
        return is_treatment_naive
    
    def _generate_targeted_profile(
        self,
        common_requirements: Dict,
        distinguishing_features: Dict,
        age: int,
        gender: str,
        histology: str,
        mutation: str,
        stage: str,
        biomarker: str,
        secondary_conditions: List[str],
        lab_values: Dict[str, str],
        patient_location: Dict[str, str],
        target_trial_ids: List[str],
        is_treatment_naive: bool = False,
        additional_requirements: Dict[str, Any] = None
    ) -> Dict:
        """
        Generate a highly detailed patient profile targeted at specific trials.
        """
        if additional_requirements is None:
            additional_requirements = {}

        conditions_text = ", ".join(secondary_conditions) if secondary_conditions else "none"
        lab_text = ", ".join([f"{k}: {v}" for k, v in lab_values.items()]) if lab_values else "none"
        location_text = f"{patient_location['city']}, {patient_location['country']}"

        # Extract PD-L1 and HLA requirements
        pdl1_tps = additional_requirements.get('pdl1_tps')
        hla_type = additional_requirements.get('hla_type')

        # Format additional requirements text
        additional_reqs_text = []
        if pdl1_tps:
            additional_reqs_text.append(f"PD-L1 TPS must be ≥50% (use {pdl1_tps}%)")
        if hla_type:
            additional_reqs_text.append(f"HLA type must be {hla_type}")

        additional_reqs_formatted = "\n".join([f"  - {req}" for req in additional_reqs_text]) if additional_reqs_text else "  - None"
        
        # Format distinguishing features - clean them up
        distinguishing_list = distinguishing_features.get('distinguishing_features', [])
        # Remove header-like entries (lines ending with colon)
        distinguishing_list = [f for f in distinguishing_list if not f.strip().endswith(':')]
        # Remove leading markers
        distinguishing_list = [re.sub(r'^[\*\-\+\d]+\.?\s*:?\s*', '', f).strip() for f in distinguishing_list]
        
        distinguishing_text = "\n".join([
            f"  - {feat}"
            for feat in distinguishing_list[:5] if len(feat) > 10
        ])

        # Check brain mets policy
        brain_mets_policy = common_requirements.get('brain_mets_policy', '').lower()
        has_brain_mets_restriction = 'excluded' in brain_mets_policy or 'no brain' in brain_mets_policy

        if has_brain_mets_restriction:
            brain_mets_instruction = "Patient MUST NOT have brain metastases (state clearly: 'Patient has no brain metastases')"
        else:
            brain_mets_instruction = f"Brain metastases status: {brain_mets_policy}"
        
        # Format key inclusions/exclusions
        inclusions_text = "\n".join([f"  - {inc}" for inc in common_requirements.get('key_inclusions', [])])
        exclusions_text = "\n".join([f"  - {exc}" for exc in common_requirements.get('key_exclusions', [])])
        
        # Get target drugs
        biomarker_info = self.biomarker_params.get(biomarker, {})
        drugs_text = ", ".join(biomarker_info.get('common_drugs', ['targeted therapy']))

        unique_combinations = self._identify_unique_combinations(
            common_requirements, 
            distinguishing_features,
            target_trial_ids
        )
        unique_text = "\n".join([f"  - {feat}" for feat in unique_combinations])
        
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are creating a detailed patient profile.<|eot_id|><|start_header_id|>user<|end_header_id|>

Create a patient profile for a {biomarker}-positive NSCLC patient matching trials: {', '.join(target_trial_ids)}. It is ideal that the patient matches those targets, and matches few other trials.
The purpose of the distinguishing features and additional requirements is to make the patient uniquely suited to these trials.

UNIQUE COMBINATIONS that distinguish target trials:
{unique_text}

ADD STRATEGIC NUANCES to reduce false positives:
1. If targets allow specific HLA types (e.g., HLA-A*11:01), include it
   → Most generic trials don't check HLA, so this won't disqualify from targets
2. If targets accept specific mutation subtypes (e.g., G12C), use the exact subtype
   → Generic "KRAS+" trials may score lower with precise mutation details
3. If targets are first-line, make patient treatment-naive
   → Second-line trials will score lower
4. If targets allow treated brain mets, mention "prior SRS to brain met, now stable"
   → Trials excluding brain mets entirely will score lower
5. If targets have specific PD-L1 thresholds, match them precisely
   → Trials with different thresholds may score lower
6. Add very recent diagnosis dates (e.g., "diagnosed February 2025")
   → Trials requiring longer disease history may score lower
7. Use specific trial-phase language if targets are Phase 1/2
   → Add phrases like "willing to accept investigational therapy risks"

EXAMPLES OF STRATEGIC NUANCES:
- "Patient specifically interested in novel targeted therapies and immune-oncology combinations"
  → Generic chemo trials may score this lower
  
- "Previously evaluated for but declined standard second-line docetaxel due to preference for targeted options"
  → Generic second-line trials may be less relevant
  
- "Seeking trials with oral medications and outpatient administration when possible"
  → IV chemotherapy trials may score lower

DO NOT make these disqualifiers obvious or absolute - they should be subtle preferences
and details that naturally favor the target trials.

PATIENT: {age}-year-old {gender} from {location_text}

DIAGNOSIS:
- Histology: {histology} (specify subtype and differentiation)
- Stage: {stage} with specific metastatic sites and measurements
- Biomarker: {mutation} (testing method, VAF %, date)
- ECOG: {common_requirements.get('ecog_requirement', '0-1').split('-')[0]}

CRITICAL ADDITIONAL REQUIREMENTS FOR THESE SPECIFIC TRIALS:
{additional_reqs_formatted}

DISTINGUISHING FEATURES:
{distinguishing_text}

KEY TRIAL REQUIREMENTS (patient MUST meet):
Inclusions:
{inclusions_text}

Exclusions patient MUST NOT have:
{exclusions_text}

Other requirements:
- Prior therapy: {common_requirements.get('prior_therapy', '1-2 prior lines')}
- Prior therapy MUST be appropriate for KRAS-mutant NSCLC
- NEVER use EGFR inhibitors (gefitinib, erlotinib, osimertinib) for KRAS patients
- Appropriate prior therapy: carboplatin + pemetrexed ± pembrolizumab
- If treatment-naive per trial requirements, state clearly: "Patient has not received prior systemic therapy"
- Brain mets: {common_requirements.get('brain_mets_policy', 'none or treated')}
- Measurable disease: {common_requirements.get('measurable_disease', 'Yes per RECIST 1.1')}
- Mention the absence of HIV, active hepatitis B/C, or other immunosuppressive conditions


DETAILED SPECIFICATIONS:

1. Molecular Testing:
   - Method and {mutation} with VAF and date
   - TMB, PD-L1 if relevant

2. Prior Treatment:
   - Specific regimen (drug names, doses, schedule)
   - Number of cycles and dates
   - Best response (% reduction)
   - Progression details
   - Weeks since last treatment

3. Disease Measurements (BE SPECIFIC):
   - Primary tumor: location and size (e.g., "right upper lobe mass 4.8 cm")
   - Metastatic sites: (e.g., "liver segment 6 lesion 3.4 cm, mediastinal lymph nodes 2.8 cm")
   - ⚠️ CRITICAL BRAIN METASTASES: {brain_mets_instruction}
   - HIV/hepatitis status: "Patient is HIV-negative and has no active hepatitis B or C"
   - RECIST 1.1: (e.g., "4 target lesions, sum of diameters 12.3 cm")
   - Symptoms: consistent with ECOG {common_requirements.get('ecog_requirement', '0-1').split('-')[0]}

4. Laboratory (specific numbers):
   - CBC: Hgb {lab_values.get('hemoglobin', '12.2 g/dL')}, ANC, platelets
   - Hepatic: AST, ALT, bilirubin, albumin  
   - Renal: creatinine, eGFR
   - Other relevant values

5. Comorbidities:
   {conditions_text}
   (For each: year, medications with doses, control status)

CRITICAL INSTRUCTIONS:
- Write ONLY direct factual sentences
- NO section headers like "Patient Demographics:" or "Primary Diagnosis:"
- NO introductory phrases like "Here is the profile" or "The patient profile is as follows"
- NO conclusions or summaries like "Note: Patient is eligible for trials" or "Note: I have included..."
- Start immediately with patient facts
- Put each sentence ON A NEW LINE
- Each line should be ONE complete sentence
- Use specific numbers, dates, and measurements throughout
- NO placeholders

Example format (FOLLOW THIS EXACTLY - one sentence per line):
The patient is a 65-year-old male from Boston, United States.
He was diagnosed with lung adenocarcinoma, acinar-predominant pattern, moderately differentiated, in March 2023.
Molecular testing by FoundationOne CDx on April 2023 revealed KRAS G12C mutation at 38% variant allele frequency.
Tumor mutational burden was 9 mutations per megabase and PD-L1 tumor proportion score was 42%.
The disease is classified as stage IV with metastases to the liver and bone.
A 4.2 cm liver lesion is present in segment 6 and there is a lytic lesion in the T7 vertebra measuring 2.8 cm.
The primary tumor in the right upper lobe measures 5.1 cm.
Per RECIST 1.1, there are 5 target lesions with a sum of diameters of 13.6 cm.
His ECOG performance status is 1 with mild dyspnea on exertion but good functional capacity overall.
He received first-line carboplatin AUC 5 and pemetrexed 500 mg/m² for 6 cycles from March to August 2023.
Best response was partial response with 45% reduction in target lesions.
Disease progression occurred in September 2023 with new liver metastases.
Time since last chemotherapy is 8 weeks.
He has chronic obstructive pulmonary disease managed with tiotropium 18 mcg inhaler once daily since 2018.
His hypertension is controlled with amlodipine 10 mg daily since 2020.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_content = response.content if hasattr(response, 'content') else str(response)
            raw_description = raw_content.strip()
            
            # Remove any intro phrases
            raw_description = re.sub(r"^Human[:\-]?\s*", "", raw_description)
            raw_description = re.sub(r"^Here (?:is|are) (?:the )?.*?:\s*", "", raw_description, flags=re.IGNORECASE)
            raw_description = re.sub(r"^(?:Patient )?Profile:\s*", "", raw_description, flags=re.IGNORECASE)
            
        except Exception as e:
            logger.warning(f"Targeted profile generation failed: {e}")
            raw_description = (
                f"The patient is a {age}-year-old {gender} from {location_text}.\n"
                f"Diagnosed with {histology}, {stage}.\n"
                f"Molecular testing by next-generation sequencing revealed {mutation}.\n"
                f"ECOG performance status is {common_requirements.get('ecog_requirement', '0-1').split('-')[0]}.\n"
                f"Patient has received prior therapy and has measurable disease per RECIST 1.1."
            )
        
        # Extract and structure
        try:
            extracted_age, extracted_gender = extract_age_gender_from_summary(raw_description)
            main_condition = extract_main_condition_from_summary(raw_description)
            synonyms = self.validator.get_valid_lung_cancer_synonyms(histology)
            
            if not main_condition or 'lung' not in main_condition.lower():
                main_condition = histology
            
            # BETTER SENTENCE SPLITTING LOGIC
            # First split on newlines
            lines = [s.strip() for s in raw_description.split('\n') if s.strip()]
            
            # Then split each line on '. ' (period followed by space) to handle paragraphs
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
                    if not part:
                        continue
                        
                    # Add period back if it was removed and this isn't the last part
                    if i < len(parts) - 1 and not part.endswith('.'):
                        part += '.'
                    
                    # Only keep substantial sentences
                    if len(part) > 15:
                        sentences.append(part)
            
            logger.info(f"Split into {len(sentences)} sentences before validation")
            
            # Validate sentences
            validated_sentences = self.validator.validate_and_fix_expanded_sentences(
                sentences, [main_condition] + secondary_conditions
            )
            
            logger.info(f"Generated {len(validated_sentences)} validated sentences")
            
            # Log first few sentences for debugging
            for i, sent in enumerate(validated_sentences[:3]):
                logger.info(f"  Sentence {i+1}: {sent[:80]}...")
            
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            extracted_age, extracted_gender = age, gender
            main_condition = histology
            synonyms = self.validator.get_valid_lung_cancer_synonyms(histology)
            validated_sentences = [raw_description]
        
        return {
            "main_conditions": [main_condition] + synonyms,
            "other_conditions": secondary_conditions,
            "expanded_sentences": validated_sentences
        }
    
    def _identify_unique_combinations(
        self,
        common_requirements: Dict,
        distinguishing_features: Dict,
        target_trial_ids: List[str]
    ) -> List[str]:
        """
        Identify unique feature combinations that distinguish targets from generic trials.
        """
        unique_combos = []
        
        # Check for HLA requirements (very specific to certain trials)
        if any(trial_id == "NCT03948763" for trial_id in target_trial_ids):
            unique_combos.append("Requires specific HLA typing (HLA-A*11:01 or HLA-C*08:02) - rare in generic trials")
        
        # Check for specific biomarker requirements
        required_mut = common_requirements.get('required_mutation', '')
        if 'G12C' in required_mut and 'G12D' not in required_mut:
            unique_combos.append("Requires exact KRAS G12C subtype (not just 'KRAS+') - more specific than generic trials")
        
        # Check for treatment-line specificity
        prior_therapy = common_requirements.get('prior_therapy', '').lower()
        if 'treatment-naive' in prior_therapy or 'first-line' in prior_therapy:
            unique_combos.append("First-line treatment setting - excludes second-line trials")
        
        # Check for PD-L1 requirements
        # (You'd extract this from trial details)
        
        # Check for novel mechanisms from distinguishing features
        dist_features = distinguishing_features.get('distinguishing_features', [])
        for feature in dist_features:
            if 'vaccine' in feature.lower() or 'mrna' in feature.lower():
                unique_combos.append("Novel mRNA vaccine approach - distinct from standard targeted therapy trials")
            if 'combination' in feature.lower() and 'pembrolizumab' in feature.lower():
                unique_combos.append("Specific combination with pembrolizumab - not in monotherapy trials")
        
        return unique_combos[:5]


def load_trials_from_nct_ids(nct_ids: List[str], trials_folder: str) -> List[Dict]:
    """
    Load trial JSON files given a list of NCT IDs.
    
    Args:
        nct_ids: List of NCT IDs (e.g., ['NCT03948764', 'NCT06345729'])
        trials_folder: Path to folder containing trial JSON files
    
    Returns:
        List of trial dicts
    """
    trials = []
    
    for nct_id in nct_ids:
        trial_file = os.path.join(trials_folder, f"{nct_id}.json")
        
        if not os.path.exists(trial_file):
            logger.error(f"Trial file not found: {trial_file}")
            continue
        
        try:
            with open(trial_file, 'r') as f:
                trial = json.load(f)
                trials.append(trial)
                logger.info(f"✓ Loaded {nct_id}")
        except Exception as e:
            logger.error(f"Failed to load {nct_id}: {e}")
    
    return trials


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate patients targeted at specific trials for precision matching experiments"
    )
    parser.add_argument(
        '--nct-ids-file',
        type=str,
        required=True,
        help='Path to text file with one NCT ID per line (e.g., nct_ids.txt)'
    )
    parser.add_argument(
        '--trials-folder',
        type=str,
        default='../data/lung_processed_trials',
        help='Folder containing trial JSON files'
    )
    parser.add_argument(
        '--biomarker',
        type=str,
        default='KRAS',
        help='Biomarker name (KRAS, EGFR, ROS1, etc.)'
    )
    parser.add_argument(
        '--num-patients',
        type=int,
        default=1,
        help='Number of patients to generate'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='../data/targeted_patient_experiment.json',
        help='Output JSON file for generated patients'
    )
    
    args = parser.parse_args()
    
    # Read NCT IDs from file
    if not os.path.exists(args.nct_ids_file):
        logger.error(f"NCT IDs file not found: {args.nct_ids_file}")
        exit(1)
    
    with open(args.nct_ids_file, 'r') as f:
        nct_ids = [line.strip() for line in f if line.strip() and line.strip().startswith('NCT')]
    
    if not nct_ids:
        logger.error(f"No valid NCT IDs found in {args.nct_ids_file}")
        exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info("TARGETED PATIENT GENERATION")
    logger.info(f"{'='*70}")
    logger.info(f"NCT IDs file: {args.nct_ids_file}")
    logger.info(f"Target NCT IDs: {nct_ids}")
    logger.info(f"Biomarker: {args.biomarker}")
    logger.info(f"Num patients: {args.num_patients}")
    logger.info(f"{'='*70}\n")
    
    # Load target trials
    target_trials = load_trials_from_nct_ids(nct_ids, args.trials_folder)
    
    if not target_trials:
        logger.error("No trials loaded! Check NCT IDs and trials folder path.")
        exit(1)
    
    if len(target_trials) != len(nct_ids):
        logger.warning(f"Only loaded {len(target_trials)}/{len(nct_ids)} trials")
    
    # Generate patients
    generator = TargetedPatientGenerator(llm)
    
    patients = generator.generate_patient_matching_specific_trials(
        target_trials=target_trials,
        biomarker=args.biomarker,
        num_patients=args.num_patients,
        output_file=args.output_file
    )
    
    logger.info(f"\n✓ SUCCESS! Generated {len(patients)} patient(s)")
    logger.info(f"✓ Saved to: {args.output_file}")
    logger.info(f"\nNext steps:")