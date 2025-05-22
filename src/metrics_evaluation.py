import json
import sys
import os
import re
from typing import Dict, Any, Tuple, List, Set
import argparse

# OpenAI initialization that avoids the proxies issue
def create_openai_client(api_key=None):
    """Create an OpenAI client that works around the proxies issue"""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("No OpenAI API key found. OpenAI functionality will be limited.")
            return None
    
    try:
        # Try the standard initialization
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except TypeError as e:
        if "unexpected keyword argument 'proxies'" in str(e):
            print("Detected proxies issue in OpenAI initialization. Using workaround...")
            
            # Method 2: Manual client creation
            try:
                from openai._client import OpenAI as OpenAIBase
                
                class CleanOpenAI(OpenAIBase):
                    def __init__(self, api_key=None):
                        super().__init__(api_key=api_key)
                
                return CleanOpenAI(api_key=api_key)
            except Exception as e2:
                print(f"Error with OpenAI manual client creation: {e2}")
                return None
        else:
            # Unexpected error
            print(f"Error initializing OpenAI client: {e}")
            return None
    except Exception as e:
        print(f"Unexpected error initializing OpenAI client: {e}")
        return None

# Initialize OpenAI client if API key is provided
api_key = os.environ.get("OPENAI_API_KEY")
client_openai = create_openai_client(api_key) if api_key else None

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON file.")
        sys.exit(1)

def extract_symptoms_from_annotations(annotations: Dict[str, List[List[Dict[str, Any]]]]) -> Set[str]:
    """Extract unique symptoms from annotations."""
    unique_symptoms = set()
    symptom_quotes = {}
    
    for conversation_id, utterances in annotations.items():
        for utterance_list in utterances:
            for utterance in utterance_list:
                if utterance.get("slot") == "symptom" and "value" in utterance:
                    # Split combined symptoms by comma and add each individually
                    symptom_values = utterance["value"].split(",")
                    for symptom in symptom_values:
                        symptom = symptom.strip()
                        # Skip empty symptoms and "all"
                        if symptom and symptom.lower() != "all":
                            unique_symptoms.add(symptom)
                            
                            # Store the utterance text as a quote for this symptom
                            symptom_key = symptom.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
                            if symptom_key not in symptom_quotes:
                                symptom_quotes[symptom_key] = []
                            
                            speaker_text = utterance.get("speaker_text", "")
                            if speaker_text and speaker_text not in symptom_quotes[symptom_key]:
                                symptom_quotes[symptom_key].append(speaker_text)
    
    return unique_symptoms, symptom_quotes

def generate_ground_truth_from_annotations(annotations: Dict[str, Any], criteria: List[str]) -> Dict[str, Any]:
    """Generate ground truth assessment data from annotations file."""
    # Extract symptoms from annotations
    discussed_symptoms, symptom_quotes = extract_symptoms_from_annotations(annotations)
    
    # Create ground truth structure
    ground_truth = {"medical_history_assessed": {}}
    
    # Process each item in criteria
    for item in criteria:
        item_key = item.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        
        # Check if this item was discussed according to annotations
        is_assessed = any(symptom.lower() == item.lower() for symptom in discussed_symptoms)
        
        # Get example quotes for this item if available
        quotes = symptom_quotes.get(item_key, [])
        
        # Add to ground truth
        ground_truth["medical_history_assessed"][item_key] = {
            "assessed": "Yes" if is_assessed else "No",
            "example_quotes": quotes
        }
    
    return ground_truth

def calculate_metrics(results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for assessment results."""
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Get the assessment data
    result_assessments = results.get("medical_history_assessed", {})
    truth_assessments = ground_truth.get("medical_history_assessed", {})
    
    # Collect all unique keys from both dictionaries
    all_keys = set(result_assessments.keys()).union(set(truth_assessments.keys()))
    
    # Compare assessments
    for key in all_keys:
        # Get assessment values from results and ground truth
        result_assessed = result_assessments.get(key, {}).get("assessed", "No") == "Yes"
        truth_assessed = truth_assessments.get(key, {}).get("assessed", "No") == "Yes"
        
        # Count true positives, false positives, and false negatives
        if result_assessed and truth_assessed:
            true_positives += 1
        elif result_assessed and not truth_assessed:
            false_positives += 1
        elif not result_assessed and truth_assessed:
            false_negatives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_items": len(all_keys)
    }

def get_mismatches(results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get detailed information about false positives and false negatives."""
    false_positives = []
    false_negatives = []
    
    # Get the assessment data
    result_assessments = results.get("medical_history_assessed", {})
    truth_assessments = ground_truth.get("medical_history_assessed", {})
    
    # Collect all unique keys from both dictionaries
    all_keys = set(result_assessments.keys()).union(set(truth_assessments.keys()))
    
    # Compare assessments and collect mismatches with details
    for key in all_keys:
        # Format the key for readability
        readable_key = key.replace("_", " ").title()
        
        # Get assessment values
        result_assessed = result_assessments.get(key, {}).get("assessed", "No") == "Yes"
        truth_assessed = truth_assessments.get(key, {}).get("assessed", "No") == "Yes"
        
        # Identify mismatches and include example quotes
        if result_assessed and not truth_assessed:
            false_positives.append({
                "item": readable_key,
                "result_quotes": result_assessments.get(key, {}).get("example_quotes", [])
            })
        elif not result_assessed and truth_assessed:
            false_negatives.append({
                "item": readable_key,
                "truth_quotes": truth_assessments.get(key, {}).get("example_quotes", [])
            })
    
    return false_positives, false_negatives

def save_metrics(metrics: Dict[str, Any], output_file: str) -> None:
    """Save metrics to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    print(f"Metrics saved to {output_file}")

def main():
    """Main function to run the metrics evaluation."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate medical assessment accuracy using annotations")
    parser.add_argument("conversation_number", help="Conversation number (e.g., 101, 102)")
    parser.add_argument("--results", help="Results file path (default: results{conversation_number}.json)")
    parser.add_argument("--annotations", help="Annotations file path (default: annotations{conversation_number}.json)")
    parser.add_argument("--criteria", help="Criteria file path (default: criteria{conversation_number}.json)")
    parser.add_argument("--output", help="Output metrics file path (default: metrics{conversation_number}.json)")
    parser.add_argument("--save-ground-truth", action="store_true", help="Save generated ground truth to file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate file paths based on conversation number
    conversation_number = args.conversation_number
    results_file = args.results or f"results/results{conversation_number}.json"
    annotations_file = args.annotations or f"data/annotations/annotations{conversation_number}.json"
    criteria_file = args.criteria or f"output/criteria/criteria{conversation_number}.json"
    output_file = args.output or f"output/metrics/metrics{conversation_number}.json"
    
    # Load results file
    print(f"Loading results from {results_file}...")
    results = load_json_file(results_file)
    
    # Load criteria file
    print(f"Loading criteria from {criteria_file}...")
    criteria_data = load_json_file(criteria_file)
    
    # Handle both formats (list or dict with categories)
    if isinstance(criteria_data, dict):
        # Extract symptoms from categorized format
        criteria_list = []
        for category_symptoms in criteria_data.values():
            criteria_list.extend(category_symptoms)
    else:
        criteria_list = criteria_data
    
    # Load annotations file
    print(f"Loading annotations from {annotations_file}...")
    annotations = load_json_file(annotations_file)
    
    # Generate ground truth from annotations
    print("Analyzing annotations to determine ground truth...")
    ground_truth = generate_ground_truth_from_annotations(annotations, criteria_list)
    
    # Save ground truth if requested
    if args.save_ground_truth:
        ground_truth_file = f"ground_truth{conversation_number}.json"
        with open(ground_truth_file, 'w', encoding='utf-8') as file:
            json.dump(ground_truth, file, indent=2)
        print(f"Ground truth saved to {ground_truth_file}")
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results, ground_truth)
    
    # Get detailed mismatches
    false_positives, false_negatives = get_mismatches(results, ground_truth)
    
    # Add mismatches to metrics
    metrics["false_positive_details"] = false_positives
    metrics["false_negative_details"] = false_negatives
    
    # Save metrics
    save_metrics(metrics, output_file)
    
    # Print summary
    print("\nMetrics Summary:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total Items: {metrics['total_items']}")
    
    # Print mismatches if any
    if false_positives:
        print("\nFalse Positives (incorrectly marked as assessed):")
        for item in false_positives:
            print(f"- {item['item']}")
            if item['result_quotes']:
                print("  Quotes from results:")
                for quote in item['result_quotes'][:2]:  # Limit to 2 quotes for readability
                    print(f"  • \"{quote}\"")
    
    if false_negatives:
        print("\nFalse Negatives (incorrectly marked as not assessed):")
        for item in false_negatives:
            print(f"- {item['item']}")
            if item['truth_quotes']:
                print("  Evidence from annotations:")
                for quote in item['truth_quotes'][:2]:  # Limit to 2 quotes for readability
                    print(f"  • \"{quote}\"")

if __name__ == "__main__":
    main()