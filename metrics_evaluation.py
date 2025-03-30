import json
import sys
import os
import re
from typing import Dict, Any, Tuple, List, Set
import argparse
import openai
from openai import OpenAI

# Initialize OpenAI client (if API key is provided)
api_key = os.environ.get("OPENAI_API_KEY")
client_openai = OpenAI(api_key=api_key) if api_key else None

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

def read_transcript(file_path: str) -> str:
    """Read transcript from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: Transcript file {file_path} not found.")
        sys.exit(1)

def generate_ground_truth_from_transcript(transcript: str, criteria: Dict[str, List[str]], use_ai: bool = False) -> Dict[str, Any]:
    """Generate ground truth assessment data from transcript."""
    if use_ai and client_openai:
        # Use AI to assess the transcript (more accurate but requires API key)
        return ai_assess_transcript(transcript, criteria)
    else:
        # Use rule-based approach (less accurate but doesn't require API key)
        return rule_based_assess_transcript(transcript, criteria)

def rule_based_assess_transcript(transcript: str, criteria: Dict[str, List[str]]) -> Dict[str, Any]:
    """Assess transcript using rule-based approach to identify discussed items."""
    ground_truth = {"medical_history_assessed": {}}
    
    # Preprocess transcript to lowercase for better matching
    transcript_lower = transcript.lower()
    
    # Create lookup dictionary for various phrasings of the same concept
    concept_variations = {}
    # Map common variations
    variation_mapping = {
        "shortness of breath": ["shortness of breath", "short of breath", "sob", "difficulty breathing", "trouble breathing"],
        "high blood pressure": ["high blood pressure", "hypertension", "elevated blood pressure"],
        "colored/excess sputum": ["sputum", "phlegm", "mucus"],
        "hemoptysis": ["blood in sputum", "bloody sputum", "coughing up blood", "blood tinged sputum", "hemoptysis"],
    }
    
    # Process each category and its items
    for category, items in criteria.items():
        for item in items:
            item_key = item.lower().replace(' ', '_').replace('-', '_')
            
            # Set up variations for matching
            variations = [item.lower()]
            
            # Add mapped variations if they exist
            for concept, concept_vars in variation_mapping.items():
                if item.lower() == concept:
                    variations.extend(concept_vars)
            
            # Check if any variation is mentioned in the transcript
            is_assessed = False
            example_quotes = []
            
            # Check for direct mentions
            for variation in variations:
                # Look for the term surrounded by word boundaries or punctuation
                pattern = r'(?<![a-z])' + re.escape(variation) + r'(?![a-z])'
                matches = re.finditer(pattern, transcript_lower)
                
                for match in matches:
                    # Get the context (sentence or question) containing the match
                    start = max(0, match.start() - 100)
                    end = min(len(transcript_lower), match.end() + 100)
                    context = transcript[start:end]
                    
                    # Extract the complete sentence or question
                    sentence_match = re.search(r'([^.!?]+[.!?])', context)
                    if sentence_match:
                        quote = sentence_match.group(0).strip()
                        if quote not in example_quotes:
                            example_quotes.append(quote)
                    else:
                        # If we can't extract a clean sentence, use the context
                        if context.strip() not in example_quotes:
                            example_quotes.append(context.strip())
                    
                    is_assessed = True
            
            # Look for clear indications of assessment through questions
            # For example, a medical student asking "Do you have any allergies?"
            assessment_patterns = [
                rf"(?:any|have|experiencing|noticed) (?:.*?){re.escape(item.lower())}",
                rf"(?:check|assess|evaluate|asked about) (?:.*?){re.escape(item.lower())}"
            ]
            
            for pattern in assessment_patterns:
                matches = re.finditer(pattern, transcript_lower)
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(transcript_lower), match.end() + 50)
                    context = transcript[start:end]
                    
                    sentence_match = re.search(r'([^.!?]+[.!?])', context)
                    if sentence_match:
                        quote = sentence_match.group(0).strip()
                        if quote not in example_quotes:
                            example_quotes.append(quote)
                    else:
                        if context.strip() not in example_quotes:
                            example_quotes.append(context.strip())
                    
                    is_assessed = True
            
            # Add to ground truth
            ground_truth["medical_history_assessed"][item_key] = {
                "assessed": "Yes" if is_assessed else "No",
                "example_quotes": example_quotes
            }
    
    return ground_truth

def ai_assess_transcript(transcript: str, criteria: Dict[str, List[str]]) -> Dict[str, Any]:
    """Use OpenAI to assess the transcript for medical history elements."""
    # Flatten criteria into a single list
    all_items = []
    for category, items in criteria.items():
        all_items.extend(items)
    
    # Construct prompt
    prompt = "# Medical History Assessment Evaluation\n"
    prompt += "Analyze this conversation between a medical student and a patient. "
    prompt += "Determine whether each of the following medical history elements was assessed or discussed.\n\n"
    
    # Add list of items to check
    prompt += "## Items to evaluate:\n"
    for item in all_items:
        prompt += f"- {item}\n"
    
    # Add output format instructions
    prompt += "\n## Required Output Format\n"
    prompt += "Provide your evaluation in the following JSON format:\n```json\n{\n"
    prompt += "  \"medical_history_assessed\": {\n"
    
    # Generate JSON structure
    for item in all_items:
        item_key = item.lower().replace(' ', '_').replace('-', '_')
        prompt += f"    \"{item_key}\": {{\n"
        prompt += f"      \"assessed\": \"Yes/No\",\n"
        prompt += f"      \"example_quotes\": []\n"
        prompt += f"    }}"
        if item != all_items[-1]:
            prompt += ","
        prompt += "\n"
    
    prompt += "  }\n}\n```\n\n"
    
    # Add instructions
    prompt += "### Instructions\n"
    prompt += "1. For each item, determine if it was assessed in the conversation.\n"
    prompt += "2. Include example quotes from the transcript that demonstrate the assessment.\n"
    prompt += "3. Mark an item as 'Yes' if the medical student directly asked about it OR if the patient volunteered information about it.\n"
    prompt += "4. Use an empty array for example_quotes when an item was not assessed.\n"
    
    # Make API call
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript}
            ],
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        return json.loads(content)
    
    except Exception as e:
        print(f"Error using AI to assess transcript: {e}")
        print("Falling back to rule-based assessment...")
        return rule_based_assess_transcript(transcript, criteria)

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
    parser = argparse.ArgumentParser(description="Evaluate medical assessment accuracy using transcript")
    parser.add_argument("conversation_number", help="Conversation number (e.g., 101, 102)")
    parser.add_argument("--results", help="Results file path (default: results{conversation_number}.json)")
    parser.add_argument("--transcript", help="Transcript file path (default: transcript{conversation_number}.txt)")
    parser.add_argument("--criteria", help="Criteria file path (default: criteria{conversation_number}.json)")
    parser.add_argument("--output", help="Output metrics file path (default: metrics{conversation_number}.json)")
    parser.add_argument("--use-ai", action="store_true", help="Use AI to assess transcript (requires OpenAI API key)")
    parser.add_argument("--save-ground-truth", action="store_true", help="Save generated ground truth to file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate file paths based on conversation number
    conversation_number = args.conversation_number
    results_file = args.results or f"results{conversation_number}.json"
    transcript_file = args.transcript or f"transcript{conversation_number}.txt"
    criteria_file = args.criteria or f"criteria{conversation_number}.json"
    output_file = args.output or f"metrics{conversation_number}.json"
    
    # Load results file
    print(f"Loading results from {results_file}...")
    results = load_json_file(results_file)
    
    # Load criteria file
    print(f"Loading criteria from {criteria_file}...")
    criteria = load_json_file(criteria_file)
    
    # Load transcript file
    print(f"Loading transcript from {transcript_file}...")
    transcript = read_transcript(transcript_file)
    
    # Generate ground truth from transcript
    print("Analyzing transcript to determine ground truth...")
    ground_truth = generate_ground_truth_from_transcript(transcript, criteria, args.use_ai)
    
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
                print("  Evidence from transcript:")
                for quote in item['truth_quotes'][:2]:  # Limit to 2 quotes for readability
                    print(f"  • \"{quote}\"")

if __name__ == "__main__":
    main()