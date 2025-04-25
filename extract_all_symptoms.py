import json
import os
import sys
from typing import Dict, Any, List, Set
import argparse

# Symptoms to always exclude (hardcoded)
EXCLUDED_SYMPTOMS = {"general", "surgery", "recent hospitalizaiton", "past experience"}

def standardize_symptom(symptom: str) -> str:
    """Standardize symptom spelling - just handle vomiting vs vomitting."""
    # Convert to lowercase for comparison
    symptom_lower = symptom.lower().strip()
    
    # Handle the vomitting/vomiting case
    if symptom_lower == "vomitting":
        return "vomiting"
    
    return symptom

def extract_symptoms_from_annotations(annotations: Dict[str, Any], start_id: int, end_id: int) -> Set[str]:
    """Extract unique symptoms from annotations for a range of conversations."""
    unique_symptoms = set()
    processed_count = 0
    
    # Filter conversation IDs within the specified range
    conversation_ids = [str(i) for i in range(start_id, end_id + 1) if str(i) in annotations]
    
    # Process each conversation
    for conversation_id in conversation_ids:
        conversation_data = annotations.get(conversation_id, [])
        
        for utterance_list in conversation_data:
            for utterance in utterance_list:
                if utterance.get("slot") == "symptom" and "value" in utterance:
                    # Split combined symptoms by comma and add each individually
                    symptom_values = utterance["value"].split(",")
                    for symptom in symptom_values:
                        symptom = symptom.strip()
                        # Skip empty symptoms, "all", and excluded symptoms
                        if (symptom and 
                            symptom.lower() != "all" and 
                            symptom.lower() not in EXCLUDED_SYMPTOMS):
                            # Standardize symptom spelling/formatting
                            standardized = standardize_symptom(symptom)
                            unique_symptoms.add(standardized)
        
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"Processed {processed_count} conversations, found {len(unique_symptoms)} unique symptoms so far...")
    
    return unique_symptoms

def main():
    """Extract all symptoms from a range of conversations and save to a file."""
    parser = argparse.ArgumentParser(description="Extract symptoms from a range of conversations")
    parser.add_argument("annotations_file", help="Path to the main annotations JSON file")
    parser.add_argument("--output", default="all_symptoms.json", help="Output file path (default: all_symptoms.json)")
    parser.add_argument("--start", type=int, default=101, help="Starting conversation ID")
    parser.add_argument("--end", type=int, default=200, help="Ending conversation ID")
    parser.add_argument("--append", action="store_true", help="Append to existing all_symptoms.json instead of overwriting")
    
    args = parser.parse_args()
    
    # Read existing symptoms if append flag is set
    existing_symptoms = set()
    if args.append and os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
                if isinstance(existing_data, list):
                    existing_symptoms = set(existing_data)
                    print(f"Loaded {len(existing_symptoms)} existing symptoms from {args.output}")
        except Exception as e:
            print(f"Warning: Could not read existing symptoms file: {e}")
    
    # Read annotations
    print(f"Reading annotations from {args.annotations_file}...")
    try:
        with open(args.annotations_file, 'r', encoding='utf-8') as file:
            annotations = json.load(file)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        sys.exit(1)
    
    # Extract symptoms
    print(f"Extracting symptoms from conversations {args.start} to {args.end}...")
    extracted_symptoms = extract_symptoms_from_annotations(annotations, args.start, args.end)
    
    # Combine with existing symptoms if append flag is set
    if args.append:
        combined_symptoms = existing_symptoms.union(extracted_symptoms)
        print(f"Added {len(extracted_symptoms)} new symptoms to {len(existing_symptoms)} existing symptoms")
        print(f"Total unique symptoms: {len(combined_symptoms)}")
    else:
        combined_symptoms = extracted_symptoms
        print(f"Extracted {len(combined_symptoms)} unique symptoms")
    
    # Sort and save
    sorted_symptoms = sorted(combined_symptoms)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(sorted_symptoms, file, indent=2)
    
    print(f"Saved {len(sorted_symptoms)} symptoms to {args.output}")

if __name__ == "__main__":
    main()