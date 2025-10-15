#!/usr/bin/env python3
""" Script to transform raw trial data into a structured format suitable for indexing.
The input JSON files are expected to be in a nested format as provided by ClinicalTrials.gov.
Example usage:
python transform_trials.py --input-folder raw_trials/ --output-folder transformed_trials/ --ids-file nct_ids.txt
"""
import argparse
import json
import os
from pathlib import Path


def extract_nested_value(data, path):
    """Extract value from nested dictionary using dot notation path"""
    current = data
    for key in path.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def transform_trial_json(input_data):
    """Transform nested protocolSection format to flat format expected by embedder"""
    
    # Helper function to safely get nested values
    def get_value(path, default=None):
        return extract_nested_value(input_data, path) or default
    
    # Start with the basic structure
    transformed = {}
    
    # NCT ID
    transformed["nct_id"] = get_value("protocolSection.identificationModule.nctId")
    
    # Titles
    transformed["brief_title"] = get_value("protocolSection.identificationModule.briefTitle")
    transformed["official_title"] = get_value("protocolSection.identificationModule.officialTitle")
    
    # Descriptions
    transformed["brief_summary"] = get_value("protocolSection.descriptionModule.briefSummary")
    transformed["detailed_description"] = get_value("protocolSection.descriptionModule.detailedDescription")
    
    # Status information
    transformed["overall_status"] = get_value("protocolSection.statusModule.overallStatus")
    
    # Dates - extract just the date part from the structured format
    start_date_struct = get_value("protocolSection.statusModule.startDateStruct")
    if start_date_struct and isinstance(start_date_struct, dict):
        transformed["start_date"] = start_date_struct.get("date")
    
    completion_date_struct = get_value("protocolSection.statusModule.completionDateStruct")
    if completion_date_struct and isinstance(completion_date_struct, dict):
        transformed["completion_date"] = completion_date_struct.get("date")
    
    # Study type
    transformed["study_type"] = get_value("protocolSection.designModule.studyType")
    
    # Conditions
    conditions = get_value("protocolSection.conditionsModule.conditions")
    if conditions:
        transformed["condition"] = conditions
    
    # Interventions
    interventions = get_value("protocolSection.armsInterventionsModule.interventions")
    if interventions and isinstance(interventions, list):
        transformed_interventions = []
        for intervention in interventions:
            if isinstance(intervention, dict):
                transformed_intervention = {
                    "intervention_type": intervention.get("type"),
                    "intervention_name": intervention.get("name")
                }
                transformed_interventions.append(transformed_intervention)
        transformed["intervention"] = transformed_interventions
    
    # Eligibility criteria
    eligibility = get_value("protocolSection.eligibilityModule")
    if eligibility:
        # Gender
        transformed["gender"] = eligibility.get("sex")
        
        # Age limits
        transformed["minimum_age"] = eligibility.get("minimumAge")
        transformed["maximum_age"] = eligibility.get("maximumAge")
        
        # Eligibility criteria text
        criteria = eligibility.get("eligibilityCriteria")
        if criteria:
            transformed["eligibility_criteria"] = criteria
    
    # Locations
    locations = get_value("protocolSection.contactsLocationsModule.locations")
    if locations and isinstance(locations, list):
        transformed_locations = []
        for location in locations:
            if isinstance(location, dict):
                if isinstance(location.get("facility"), str):
                        transformed_location = {
                            "location_name": location.get("facility"),
                            "location_address": f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}".strip(', ')
                        }
                    # Handle case where facility is a dictionary
                else:
                    facility = location.get("facility", {})
                    transformed_location = {
                        "location_name": facility.get("name"),
                        "location_address": f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}".strip(', ')
                    }
                transformed_locations.append(transformed_location)
        transformed["location"] = transformed_locations
    
    # References
    references = get_value("protocolSection.referencesModule.references")
    if references and isinstance(references, list):
        transformed["reference"] = references
    else:
        transformed["reference"] = []
    
    # Clean up None values
    cleaned = {k: v for k, v in transformed.items() if v is not None}
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Transform nested clinical trial JSONs to flat format")
    parser.add_argument("--input-folder", required=True, help="Folder containing nested JSON files")
    parser.add_argument("--output-folder", required=True, help="Folder to write transformed JSON files")
    parser.add_argument("--ids-file", help="Optional: file with NCT IDs to process (one per line)")
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Determine which files to process
    if args.ids_file:
        with open(args.ids_file, 'r') as f:
            nct_ids = [line.strip() for line in f if line.strip()]
        input_files = [f"{nct_id}.json" for nct_id in nct_ids]
    else:
        # Process all JSON files in input folder
        input_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]
    
    processed = 0
    errors = 0
    
    for filename in input_files:
        input_path = os.path.join(args.input_folder, filename)
        output_path = os.path.join(args.output_folder, filename)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Missing file: {filename}")
            continue
            
        try:
            with open(input_path, 'r') as f:
                input_data = json.load(f)
            
            transformed_data = transform_trial_json(input_data)
            
            with open(output_path, 'w') as f:
                json.dump(transformed_data, f, indent=2)
            
            processed += 1
            print(f"✅ Transformed {filename}")
            
        except Exception as e:
            errors += 1
            print(f"❌ Error processing {filename}: {str(e)}")
            # get traceback for debugging
            import traceback
            traceback.print_exc()

    
    print(f"\nSummary: {processed} files transformed, {errors} errors")


if __name__ == "__main__":
    main()