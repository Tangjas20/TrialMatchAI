#!/usr/bin/env python3
"""
Extract and structure eligibility criteria from transformed trial JSONs
Usage: python transform_criteria.py --input-folder transformed_trials/ --output-folder criteria_trials/ [--ids-file ids.txt]
"""
import argparse
import json
import os
import re
from pathlib import Path


def clean_criterion_text(text):
    """
    Clean up criterion text by removing bullets, extra whitespace, etc.
    Preserves nested sub-points within a criterion.
    """
    if not text:
        return ""
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[\s\t]+', ' ', text)
    
    # Replace multiple newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading bullets/numbers but keep the text structure
    text = text.strip()
    
    # Remove common leading markers only at the very start
    text = re.sub(r'^[-•*]+\s*', '', text)
    
    # Clean up but preserve internal structure
    text = text.strip()
    
    # Remove trailing colons that might be section headers
    if text.endswith(':'):
        return ""
    
    return text


def extract_individual_criteria(text, criteria_type):
    """
    Extract individual criteria from a block of text.
    Handles numbered lists, bullet points, and nested structures.
    """
    criteria = []
    
    # Look for top-level numbered patterns: digit(s) followed by period at start or after space
    # Match patterns like " 1. ", " 2. " but NOT " 1.5 " or nested numbering like "i. ", "a. "
    # Some texts separate by *, so also consider that in pattern
    pattern = r'(?:^|\s)(?:(\d{1,2})\.|[*])\s+'

    matches = list(re.finditer(pattern, text))
    
    print(f"  Found {len(matches)} numbered items in {criteria_type}")
    
    if matches:
        # Extract text between numbered items
        for i, match in enumerate(matches):
            item_num = match.group(1)
            start_pos = match.end()
            # End position is either the start of next match or end of text
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            criterion_text = text[start_pos:end_pos].strip()
            criterion = clean_criterion_text(criterion_text)
            
            if criterion and len(criterion) > 15:
                criteria.append({
                    "criterion": criterion,
                    "type": criteria_type,
                    "entities": []
                })
                print(f"    Item {item_num}: {criterion[:80]}...")
        
        # Also check if there's text before the first numbered item
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            preamble = clean_criterion_text(preamble)
            if preamble and len(preamble) > 15:
                criteria.insert(0, {
                    "criterion": preamble,
                    "type": criteria_type,
                    "entities": []
                })
                print(f"    Preamble: {preamble[:80]}...")
    else:
        print(f"  No numbered items found, treating as single criterion")
        # No numbered items found
        cleaned = clean_criterion_text(text)
        if cleaned and len(cleaned) > 20:
            criteria.append({
                "criterion": cleaned,
                "type": criteria_type,
                "entities": []
            })
    
    return criteria


def parse_eligibility_criteria(text):
    """
    Parse eligibility criteria text into individual criteria with inclusion/exclusion labels.
    Handles complex nested numbering and bullet points.
    """
    if not text or not isinstance(text, str):
        return []
    
    criteria_list = []
    
    # Split by "Inclusion Criteria:" and "Exclusion Criteria:" first
    inclusion_match = re.search(r'Inclusion Criteria?:?\s*', text, re.IGNORECASE)
    exclusion_match = re.search(r'Exclusion Criteria?:?\s*', text, re.IGNORECASE)
    
    inclusion_text = ""
    exclusion_text = ""
    
    if inclusion_match and exclusion_match:
        inclusion_start = inclusion_match.end()
        exclusion_start = exclusion_match.start()
        inclusion_text = text[inclusion_start:exclusion_start]
        exclusion_text = text[exclusion_match.end():]
    elif inclusion_match:
        inclusion_text = text[inclusion_match.end():]
    elif exclusion_match:
        exclusion_text = text[exclusion_match.end():]
    else:
        # No clear sections, treat all as inclusion
        inclusion_text = text
    
    # Process inclusion criteria - look for top-level numbered items only
    if inclusion_text:
        criteria_list.extend(extract_individual_criteria(inclusion_text, "Inclusion Criteria"))
    
    # Process exclusion criteria
    if exclusion_text:
        criteria_list.extend(extract_individual_criteria(exclusion_text, "Exclusion Criteria"))
    
    return criteria_list


def transform_trial_for_criteria_processing(trial_json_path):
    """
    Transform a trial JSON to the format expected by prepare_criteria.py
    """
    with open(trial_json_path, 'r') as f:
        trial_data = json.load(f)
    
    nct_id = trial_data.get('nct_id')
    if not nct_id:
        return None
    
    # Get eligibility criteria text
    eligibility_text = trial_data.get('eligibility_criteria', '')
    
    if not eligibility_text:
        return None
    
    # Parse into individual criteria
    criteria = parse_eligibility_criteria(eligibility_text)
    
    if not criteria:
        return None
    
    return {
        "nct_id": nct_id,
        "criteria": criteria
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract and structure eligibility criteria from transformed trial JSONs"
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing transformed trial JSONs"
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder to write criteria-only JSONs"
    )
    parser.add_argument(
        "--ids-file",
        help="Optional: file with NCT IDs to process (one per line)"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine which files to process
    if args.ids_file:
        with open(args.ids_file, 'r') as f:
            nct_ids = [line.strip() for line in f if line.strip()]
        input_files = [f"{nct_id}.json" for nct_id in nct_ids]
    else:
        input_files = [f.name for f in input_folder.glob("*.json")]
    
    processed = 0
    skipped = 0
    no_criteria = 0
    
    for filename in input_files:
        input_path = input_folder / filename
        output_path = output_folder / filename
        
        if not input_path.exists():
            print(f"Missing file: {filename}")
            continue
        
        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {filename}: already processed")
            skipped += 1
            continue
        
        try:
            result = transform_trial_for_criteria_processing(input_path)
            
            if not result:
                print(f"No criteria found in {filename}")
                no_criteria += 1
                continue
            
            # Write output
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            processed += 1
            criteria_count = len(result['criteria'])
            print(f"Processed {filename} ({criteria_count} criteria)")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already done): {skipped}")
    print(f"  No criteria found: {no_criteria}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()