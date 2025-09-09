import os
import json
import argparse

# Arguments
parser = argparse.ArgumentParser(description="Process TREC patient data to form a single json.")
parser.add_argument(
    "--trec_year",
    type=str,
    help="Year of TREC data to find the folder, between 21 and 22.",
    default="21",
)
args = parser.parse_args()

# Define the base directory containing the patient folders
trec_year = args.trec_year
patients_folder = f"/scratch/mgeorges/TrialMatchAI/matching_results/TREC{trec_year}"
output_file = f"processed_patients{trec_year}.json"

# Initialize the dictionary to store processed patients
processed_patients = {}

# Iterate over the folders named "trec-{year}{number}"
for i in range(1, len(os.listdir(patients_folder))):
    folder_name = f"trec-20{trec_year}{i:01d}"  # Format as "trec-21X"
    folder_path = os.path.join(patients_folder, folder_name)
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    # Path to the keywords.json file
    keywords_file = os.path.join(folder_path, "keywords.json")
    
    # Check if the keywords.json file exists
    if not os.path.exists(keywords_file):
        print(f"keywords.json not found in {folder_path}")
        continue
    
    # Read the contents of the keywords.json file
    with open(keywords_file, "r", encoding="utf-8") as f:
        try:
            keywords_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading {keywords_file}: {e}")
            continue
    
    # Add the data to the processed_patients dictionary with the key "PXXX"
    patient_id = f"P{i:03d}"  # Format as "PXXX"
    processed_patients[patient_id] = keywords_data

# Write the processed patients data to the output JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_patients, f, indent=4)

print(f"Processed patients saved to {output_file}")