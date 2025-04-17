import json
import os
import sys
from typing import List, Set

def read_criteria_file(file_path: str) -> List[str]:
    """Read symptoms from a criteria file, handling both flat list and categorized formats."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Handle list format
            if isinstance(data, list):
                return data
            
            # Handle categorized dictionary format
            elif isinstance(data, dict):
                flattened = []
                for category_symptoms in data.values():
                    flattened.extend(category_symptoms)
                return flattened
            
            else:
                print(f"Warning: Unexpected format in {file_path}")
                return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def combine_all_criteria(criteria_files: List[str], output_file: str) -> Set[str]:
    """Combine symptoms from multiple criteria files into a single set."""
    all_symptoms = set()
    
    # Read and combine all criteria files
    for file_path in criteria_files:
        if os.path.exists(file_path):
            symptoms = read_criteria_file(file_path)
            print(f"Read {len(symptoms)} symptoms from {file_path}")
            
            # Add to master set, excluding "all" if present
            for symptom in symptoms:
                if symptom.lower() != "all":
                    all_symptoms.add(symptom)
        else:
            print(f"Warning: File {file_path} not found, skipping")
    
    # Sort the final set of symptoms
    sorted_symptoms = sorted(all_symptoms)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(sorted_symptoms, file, indent=2)
    
    print(f"\nCombined {len(sorted_symptoms)} unique symptoms into {output_file}")
    return all_symptoms

def main():
    """Main function to combine multiple criteria files."""
    if len(sys.argv) < 3:
        print("Usage: python combine_all_criteria.py <output_file> <criteria_file1> [criteria_file2] ...")
        print("Alternative: python combine_all_criteria.py <output_file> --range <start> <end>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    
    # Check if we're using the --range option
    if len(sys.argv) > 3 and sys.argv[2].lower() == '--range':
        if len(sys.argv) < 5:
            print("Error: When using --range, you must provide start and end numbers")
            print("Usage: python combine_all_criteria.py <output_file> --range <start> <end>")
            sys.exit(1)
            
        try:
            start_num = int(sys.argv[3])
            end_num = int(sys.argv[4])
            
            criteria_files = [f"criteria{i}.json" for i in range(start_num, end_num + 1)]
            print(f"Using range: criteria{start_num}.json to criteria{end_num}.json")
        except ValueError:
            print("Error: Start and end must be integers")
            sys.exit(1)
    else:
        # Use the explicitly provided file list
        criteria_files = sys.argv[2:]
    
    # Combine all criteria
    combine_all_criteria(criteria_files, output_file)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()