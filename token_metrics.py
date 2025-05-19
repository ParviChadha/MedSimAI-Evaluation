import json
import os
import sys
from typing import Dict, Any, List
from model_interface import ModelInterface, select_model 
from tiktoken import encoding_for_model

# Directory to store token metrics
TOKEN_METRICS_DIR = "token_metrics"

def read_transcript(file_path: str) -> str:
    """Read transcript from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_assessment_criteria(file_path: str) -> List[str]:
    """Read assessment criteria from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        # Check if it's a flat list or categorized
        data = json.load(file)
        # If it's a dictionary with categories, flatten it
        if isinstance(data, dict):
            flattened = []
            for category_symptoms in data.values():
                flattened.extend(category_symptoms)
            return sorted(flattened)
        # If it's already a list, return it directly
        return data

def construct_prompt(symptoms: List[str]) -> str:
    """Construct the evaluation prompt based on a flat list of symptoms."""
    prompt = "# Medical History Assessment Evaluation\n"
    prompt += "You are tasked with evaluating a conversation between a medical student and a patient "
    prompt += "to determine whether the student appropriately explored relevant medical history risk factors.\n\n"
    
    # Add list of symptoms to evaluate
    prompt += "## Symptoms and Medical History Factors to Evaluate\n"
    for symptom in symptoms:
        prompt += f"- **{symptom}**\n"
    
    # Add output format instructions
    prompt += "\n## Required Output Format\n"
    prompt += "Provide your evaluation in the following JSON format:\n```json\n{\n"
    prompt += "  \"medical_history_assessed\": {\n"
    
    # Generate the JSON structure for all symptoms
    for i, symptom in enumerate(symptoms):
        # Convert to snake_case for JSON keys
        key = symptom.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        prompt += f"    \"{key}\": {{\n"
        prompt += f"      \"assessed\": \"Yes/No\",\n"
        prompt += f"      \"example_quotes\": []\n"
        prompt += f"    }}"
        if i < len(symptoms) - 1:
            prompt += ","
        prompt += "\n"
    
    prompt += "  }\n}\n```\n\n"
    
    # Add instructions
    prompt += "### Instructions\n"
    prompt += "1. Provide **only** the JSON output, without additional explanation or comments.\n"
    prompt += "2. Every history element must be evaluated, even if not assessed in the transcript.\n"
    prompt += "3. Include all relevant quotes that demonstrate how each element was assessed.\n"
    prompt += "4. Use an empty array `[]` for `example_quotes` when a history element was **not** assessed in the conversation.\n"
    prompt += "5. Include either the student's questions or the patient's relevant responses in quotes.\n"
    
    return prompt

def evaluate_medical_history(transcript: str, symptoms: List[str], model_interface: ModelInterface, conversation_id: str) -> Dict[str, Any]:
    """Evaluate whether the medical history topics were assessed in the conversation."""
    # Construct prompt from symptoms list
    prompt = construct_prompt(symptoms)
    
    # Count tokens in the input
    enc = encoding_for_model("gpt-4")
    num_tokens_transcript = len(enc.encode(transcript))
    num_tokens_prompt = len(enc.encode(prompt))
    total_input_tokens = num_tokens_transcript + num_tokens_prompt
    
    # Make API call
    response = model_interface.call_model(
        system_prompt=prompt,
        user_message=transcript,
        response_type="json_object"
    )
    
    # Count tokens in the output
    response_text = json.dumps(response)
    num_tokens_response = len(enc.encode(response_text))
    
    # Save token metrics
    save_token_metrics(conversation_id, model_interface.model_name, {
        "input_tokens": {
            "transcript": num_tokens_transcript,
            "prompt": num_tokens_prompt,
            "total": total_input_tokens
        },
        "output_tokens": num_tokens_response,
        "total_tokens": total_input_tokens + num_tokens_response
    })
    
    return response

def save_token_metrics(conversation_id: str, model_name: str, metrics: Dict[str, Any]) -> None:
    """Save token metrics to a JSON file."""
    # Create token metrics directory if it doesn't exist
    os.makedirs(TOKEN_METRICS_DIR, exist_ok=True)
    
    # File path for token metrics
    metrics_file = os.path.join(TOKEN_METRICS_DIR, f"token_metrics_{conversation_id}.json")
    
    # Add timestamp and model info
    from datetime import datetime
    metrics_data = {
        "conversation_id": conversation_id,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    # Save metrics to file
    with open(metrics_file, 'w', encoding='utf-8') as file:
        json.dump(metrics_data, file, indent=2)
    
    print(f"Token metrics saved to {metrics_file}")
    
    # Also update the aggregated metrics file
    update_aggregated_metrics()

def update_aggregated_metrics() -> None:
    """Update the aggregated token metrics file with data from all individual metrics files."""
    aggregated_file = os.path.join(TOKEN_METRICS_DIR, "aggregated_token_metrics.json")
    
    # Read all individual metrics files
    all_metrics = []
    total_input_tokens = 0
    total_output_tokens = 0
    model_stats = {}
    
    # Process all metrics files
    for filename in os.listdir(TOKEN_METRICS_DIR):
        if filename.startswith("token_metrics_") and filename.endswith(".json"):
            file_path = os.path.join(TOKEN_METRICS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    metrics_data = json.load(file)
                    all_metrics.append(metrics_data)
                    
                    # Update totals
                    metrics = metrics_data.get("metrics", {})
                    input_tokens = metrics.get("input_tokens", {}).get("total", 0)
                    output_tokens = metrics.get("output_tokens", 0)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    # Update model-specific stats
                    model_name = metrics_data.get("model", "unknown")
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            "conversations": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0
                        }
                    model_stats[model_name]["conversations"] += 1
                    model_stats[model_name]["input_tokens"] += input_tokens
                    model_stats[model_name]["output_tokens"] += output_tokens
                    model_stats[model_name]["total_tokens"] += input_tokens + output_tokens
            except Exception as e:
                print(f"Error reading metrics file {filename}: {e}")
    
    # Create aggregated metrics
    aggregated_metrics = {
        "updated_at": datetime.now().isoformat(),
        "total_conversations": len(all_metrics),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "average_input_tokens_per_conversation": total_input_tokens / len(all_metrics) if all_metrics else 0,
        "average_output_tokens_per_conversation": total_output_tokens / len(all_metrics) if all_metrics else 0,
        "model_statistics": model_stats,
        "conversation_metrics": all_metrics
    }
    
    # Save aggregated metrics
    with open(aggregated_file, 'w', encoding='utf-8') as file:
        json.dump(aggregated_metrics, file, indent=2)
    
    print(f"Aggregated token metrics updated in {aggregated_file}")

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    print(f"Results saved to {output_file}")

def extract_conversation_id(file_path: str) -> str:
    """Extract conversation ID from a file path."""
    # Extract filename without extension
    filename = os.path.basename(file_path)
    # Remove non-numeric characters and get the conversation ID
    # This assumes files follow naming convention like "results101.json"
    conv_id = ''.join(c for c in filename if c.isdigit())
    return conv_id

def main():
    """Main function to evaluate a transcript."""
    if len(sys.argv) < 4:
        print("Usage: python create_results.py <transcript_file> <criteria_file> <output_results_file> [--api-key KEY]")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    criteria_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Extract conversation ID from output file
    conversation_id = extract_conversation_id(output_file)
    
    # Check for API key argument
    api_key = None
    if "--api-key" in sys.argv:
        api_key_index = sys.argv.index("--api-key")
        if api_key_index + 1 < len(sys.argv):
            api_key = sys.argv[api_key_index + 1]
    
    # Check for model selection in environment variable (set by analysis_automation.py)
    model_from_env = os.environ.get('SELECTED_MODEL')
    if model_from_env:
        selected_model = model_from_env
        print(f"Using model specified in environment: {selected_model}")
    else:
        # Select model interactively
        selected_model = select_model()
    
    # Initialize model interface
    try:
        model_interface = ModelInterface(selected_model, api_key=api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Read transcript and criteria
    transcript = read_transcript(transcript_file)
    symptoms = read_assessment_criteria(criteria_file)
    
    print(f"Loaded {len(symptoms)} symptoms from criteria file")
    print(f"Using {selected_model} model for evaluation...")
    
    # Evaluate medical history
    print("Evaluating medical history in transcript...")
    results = evaluate_medical_history(transcript, symptoms, model_interface, conversation_id)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Summary:")
    for item, details in results["medical_history_assessed"].items():
        print(f"{item.replace('_', ' ').title()}: {details['assessed']}")
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Output token usage summary
    from datetime import datetime
    token_metrics_file = os.path.join(TOKEN_METRICS_DIR, f"token_metrics_{conversation_id}.json")
    
    with open(token_metrics_file, 'r', encoding='utf-8') as file:
        metrics_data = json.load(file)
        metrics = metrics_data["metrics"]
        print("\nToken Usage Summary:")
        print(f"Input Tokens:  {metrics['input_tokens']['total']:,}")
        print(f"Output Tokens: {metrics['output_tokens']:,}")
        print(f"Total Tokens:  {metrics['total_tokens']:,}")

if __name__ == "__main__":
    main()