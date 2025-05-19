import json
import os
import sys
from typing import Dict, Any, List, Set
import random
from model_interface import ModelInterface, select_model
from tiktoken import encoding_for_model  # Import for token counting

# Configuration - Change this to set the conversation number
CONVERSATION_NUMBER = "102"  # Edit this line to change conversation number

# Path to the comprehensive symptoms file
ALL_SYMPTOMS_FILE = "all_symptoms.json"  # Edit if your file is named differently

# Token metrics directory
TOKEN_METRICS_DIR = "token_metrics"  # Directory to store token metrics

# Symptoms to always exclude (hardcoded)
EXCLUDED_SYMPTOMS = {"general", "surgery", "recent hospitalizaiton", "past experience"} #made a big choice

# Maximum ratio of additional symptoms to real symptoms
# Set to 1.0 to add the same number of fake symptoms as real symptoms
ADDITIONAL_SYMPTOMS_RATIO = 1.0

# Absolute maximum number of additional symptoms (as a safety cap)
MAX_ABSOLUTE_ADDITIONAL_SYMPTOMS = 40

def read_annotations(file_path: str) -> Dict[str, Any]:
    """Read annotations from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_all_symptoms(file_path: str) -> List[str]:
    """Read the comprehensive list of all symptoms."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Warning: Could not read all symptoms file ({file_path}): {e}")
        return []

def standardize_symptom(symptom: str) -> str: #made a big choice here
    """Standardize symptom spelling - just handle vomiting vs vomitting."""
    # Convert to lowercase for comparison
    symptom_lower = symptom.lower().strip()
    
    # Handle the vomitting/vomiting case
    if symptom_lower == "vomitting":
        return "vomiting"
    
    return symptom

def extract_symptoms_from_annotations(annotations: Dict[str, List[List[Dict[str, Any]]]]) -> Set[str]:
    """Extract unique symptoms from annotations."""
    unique_symptoms = set()
    
    for conversation_id, utterances in annotations.items():
        for utterance_list in utterances:
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
    
    return unique_symptoms

def filter_similar_symptoms(existing_symptoms: Set[str], additional_symptoms: List[str], model_interface: ModelInterface) -> List[str]:
    """Use LLM to filter out additional symptoms that are too similar to existing ones."""
    # Convert sets to lists for the API
    existing_list = list(existing_symptoms)
    
    # If there are no additional symptoms to filter, return empty list
    if not additional_symptoms:
        return []
    
    # Construct prompt with more detailed examples and stricter criteria
    prompt = """
You are a medical terminology expert tasked with identifying which symptoms are truly distinct versus those that are essentially synonyms or closely related.

Existing symptoms in the conversation:
""" + str(existing_list) + """

Potential additional symptoms to include:
""" + str(additional_symptoms) + """

For each potential symptom in the second list, determine if it should be EXCLUDED from the final criteria list based on these rules:

1. EXCLUDE if it's a synonym or alternate phrasing of any existing symptom
   - Examples: "trouble breathing" is a synonym for "dyspnea"
   - Examples: "sinus congestion" overlaps significantly with "nasal congestion"

2. EXCLUDE if it's merely a more specific or general version of an existing symptom
   - Example: If "cough" exists, exclude "dry cough" or "productive cough"
   - Example: If "headache" exists, exclude "migraine" as it's a type of headache

3. EXCLUDE if it would be assessed together with an existing symptom during a medical examination
   - Example: "runny nose" and "nasal congestion" would be assessed together
   - Example: "fever" and "chills" are often assessed together

4. EXCLUDE if it affects the same body system and presents with similar manifestations
   - Example: "hoarse tone" and "sore throat" both affect the throat area

Be strict about applying these criteria - when in doubt, EXCLUDE the additional symptom.
Make sure all the symptoms you keep are TRULY DISTINCT from the existing symptoms and from each other.

Return a JSON object with TWO arrays:
1. "keep" - symptoms to include in the criteria list
2. "exclude" - symptoms being filtered out and why they were excluded

Example response:
{
  "keep": ["truly distinct symptom 1", "truly distinct symptom 2"],
  "exclude": [
    {"symptom": "similar symptom 1", "reason": "Synonym for existing symptom X"},
    {"symptom": "similar symptom 2", "reason": "More specific version of existing symptom Y"}
  ]
}
"""

    # Count tokens in the input
    enc = encoding_for_model("gpt-4")
    tokens_existing_symptoms = len(enc.encode(str(existing_list)))
    tokens_additional_symptoms = len(enc.encode(str(additional_symptoms)))
    tokens_prompt = len(enc.encode(prompt))
    total_input_tokens = tokens_prompt + tokens_existing_symptoms + tokens_additional_symptoms
    
    try:
        # Make API call
        response = model_interface.call_model(
            system_prompt="You are a medical terminology expert assistant.",
            user_message=prompt,
            response_type="json_object"
        )
        
        # Count tokens in the output
        response_text = json.dumps(response)
        output_tokens = len(enc.encode(response_text))
        
        # Save token metrics
        save_token_metrics(CONVERSATION_NUMBER, model_interface.model_name, "filter_similar_symptoms", {
            "input_tokens": {
                "existing_symptoms": tokens_existing_symptoms,
                "additional_symptoms": tokens_additional_symptoms,
                "prompt": tokens_prompt,
                "total": total_input_tokens
            },
            "output_tokens": output_tokens,
            "total_tokens": total_input_tokens + output_tokens
        })
        
        # Extract the filtered symptoms
        kept_symptoms = response.get("keep", [])
        excluded_symptoms = response.get("exclude", [])
        
        # Log what was excluded for transparency
        if excluded_symptoms:
            print("\nExcluded symptoms:")
            for item in excluded_symptoms:
                if isinstance(item, dict):
                    print(f"- {item.get('symptom', 'unknown')}: {item.get('reason', 'No reason given')}")
                else:
                    print(f"- {item}")
        
        return kept_symptoms
    
    except Exception as e:
        print(f"Warning: Error using AI to filter symptoms: {e}")
        print("Continuing with unfiltered additional symptoms...")
        # As a fallback, just return the additional symptoms
        return additional_symptoms

def save_token_metrics(conversation_id: str, model_name: str, operation: str, metrics: Dict[str, Any]) -> None:
    """Save token metrics to a JSON file."""
    # Create token metrics directory if it doesn't exist
    os.makedirs(TOKEN_METRICS_DIR, exist_ok=True)
    
    # File path for token metrics
    metrics_file = os.path.join(TOKEN_METRICS_DIR, f"token_metrics_criteria_{conversation_id}.json")
    
    # Add timestamp and model info
    from datetime import datetime
    metrics_data = {
        "conversation_id": conversation_id,
        "model": model_name,
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    # Save metrics to file
    with open(metrics_file, 'w', encoding='utf-8') as file:
        json.dump(metrics_data, file, indent=2)
    
    print(f"Token metrics for criteria generation saved to {metrics_file}")
    
    # Update aggregated metrics file
    update_aggregated_metrics()

def update_aggregated_metrics() -> None:
    """Update the aggregated token metrics file with data from all individual metrics files."""
    aggregated_file = os.path.join(TOKEN_METRICS_DIR, "aggregated_token_metrics.json")
    
    # Check if aggregated file exists
    if os.path.exists(aggregated_file):
        try:
            with open(aggregated_file, 'r', encoding='utf-8') as file:
                aggregated_metrics = json.load(file)
        except Exception as e:
            print(f"Error reading aggregated metrics: {e}")
            aggregated_metrics = {"conversation_metrics": []}
    else:
        aggregated_metrics = {"conversation_metrics": []}
    
    # Read all individual metrics files
    all_metrics = aggregated_metrics.get("conversation_metrics", [])
    
    # Initialize counters for overall totals
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Initialize counters for operation-specific totals
    operation_totals = {
        "results": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "count": 0
        },
        "filter_similar_symptoms": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "count": 0
        }
    }
    
    # Initialize model statistics
    model_stats = {}
    
    # Get existing metrics from aggregated file
    existing_ids = set()
    for metric in all_metrics:
        if "conversation_id" in metric and "operation" in metric:
            existing_ids.add(f"{metric['conversation_id']}_{metric['operation']}")
    
    # Process all metrics files
    new_metrics = []
    for filename in os.listdir(TOKEN_METRICS_DIR):
        # Check both results and criteria metrics files
        if (filename.startswith("token_metrics_") and filename.endswith(".json") and 
            not filename == "aggregated_token_metrics.json"):
            
            file_path = os.path.join(TOKEN_METRICS_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    metrics_data = json.load(file)
                    
                    # Only add metrics if they aren't already in the aggregated file
                    metric_id = f"{metrics_data.get('conversation_id', '')}_{metrics_data.get('operation', '')}"
                    if metric_id not in existing_ids:
                        new_metrics.append(metrics_data)
                        existing_ids.add(metric_id)
            except Exception as e:
                print(f"Error reading metrics file {filename}: {e}")
    
    # Add new metrics to existing ones
    all_metrics.extend(new_metrics)
    
    # Recalculate totals
    for metrics_data in all_metrics:
        metrics = metrics_data.get("metrics", {})
        operation = metrics_data.get("operation", "results")  # Default to results if not specified
        
        # Normalize operation names for consistency
        if "criteria" in operation.lower():
            operation = "filter_similar_symptoms"
        elif operation == "":
            operation = "results"
        
        # Extract input tokens - handle different formats
        input_tokens = 0
        if isinstance(metrics.get("input_tokens"), dict):
            input_tokens = metrics.get("input_tokens", {}).get("total", 0)
        elif isinstance(metrics.get("input_tokens"), (int, float)):
            input_tokens = metrics.get("input_tokens", 0)
        
        output_tokens = metrics.get("output_tokens", 0)
        
        # Update total counters
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Update operation-specific counters
        if operation in operation_totals:
            operation_totals[operation]["input_tokens"] += input_tokens
            operation_totals[operation]["output_tokens"] += output_tokens
            operation_totals[operation]["total_tokens"] += (input_tokens + output_tokens)
            operation_totals[operation]["count"] += 1
        
        # Update model-specific stats
        model_name = metrics_data.get("model", "unknown")
        if model_name not in model_stats:
            model_stats[model_name] = {
                "conversations": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "operations": {
                    "results": {"input_tokens": 0, "output_tokens": 0, "count": 0},
                    "filter_similar_symptoms": {"input_tokens": 0, "output_tokens": 0, "count": 0}
                }
            }
        
        # Update model totals
        model_stats[model_name]["conversations"] += 1
        model_stats[model_name]["input_tokens"] += input_tokens
        model_stats[model_name]["output_tokens"] += output_tokens
        model_stats[model_name]["total_tokens"] += input_tokens + output_tokens
        
        # Update model operation-specific counters
        if operation in model_stats[model_name]["operations"]:
            model_stats[model_name]["operations"][operation]["input_tokens"] += input_tokens
            model_stats[model_name]["operations"][operation]["output_tokens"] += output_tokens
            model_stats[model_name]["operations"][operation]["count"] += 1
    
    # Calculate averages for operations
    for op, totals in operation_totals.items():
        if totals["count"] > 0:
            totals["avg_input_tokens"] = totals["input_tokens"] / totals["count"]
            totals["avg_output_tokens"] = totals["output_tokens"] / totals["count"]
            totals["avg_total_tokens"] = totals["total_tokens"] / totals["count"]
    
    # Create aggregated metrics
    from datetime import datetime
    aggregated_metrics = {
        "updated_at": datetime.now().isoformat(),
        "total_conversations": len(set(m.get("conversation_id", "") for m in all_metrics)),
        "total_operations": len(all_metrics),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "average_input_tokens_per_operation": total_input_tokens / len(all_metrics) if all_metrics else 0,
        "average_output_tokens_per_operation": total_output_tokens / len(all_metrics) if all_metrics else 0,
        
        # Add operation-specific totals
        "operation_totals": operation_totals,
        
        # Model statistics with operation breakdowns
        "model_statistics": model_stats,
        
        # Individual metrics
        "conversation_metrics": all_metrics
    }
    
    # Save aggregated metrics
    with open(aggregated_file, 'w', encoding='utf-8') as file:
        json.dump(aggregated_metrics, file, indent=2)
    
    print(f"Updated aggregated token metrics in {aggregated_file}")

def generate_criteria_from_annotations(annotations_file: str, output_file: str, model_interface: ModelInterface, all_symptoms_file: str = ALL_SYMPTOMS_FILE) -> List[str]:
    """Generate criteria list from annotations file and add symptoms from all_symptoms."""
    # Read annotations
    annotations = read_annotations(annotations_file)
    
    # Extract unique symptoms from the current conversation
    conversation_symptoms = extract_symptoms_from_annotations(annotations)
    print(f"Found {len(conversation_symptoms)} real symptoms in the conversation")
    
    # Read all possible symptoms from the comprehensive file
    all_symptoms = read_all_symptoms(all_symptoms_file)
    
    # Filter all_symptoms to remove excluded and standardize spelling
    filtered_all_symptoms = []
    for symptom in all_symptoms:
        if symptom.lower() not in EXCLUDED_SYMPTOMS and symptom.lower() != "all":
            standardized = standardize_symptom(symptom)
            if standardized not in filtered_all_symptoms:  # Avoid duplicates
                filtered_all_symptoms.append(standardized)
    
    # Identify potential additional symptoms (those not in the conversation)
    potential_additional_symptoms = [
        symptom for symptom in filtered_all_symptoms 
        if symptom not in conversation_symptoms
    ]
    
    # Filter additional symptoms to only include those that are conceptually distinct
    if potential_additional_symptoms:
        print(f"Filtering {len(potential_additional_symptoms)} potential additional symptoms...")
        filtered_additional_symptoms = filter_similar_symptoms( #made a big choice to filter using LLM
            conversation_symptoms, 
            potential_additional_symptoms,
            model_interface
        )
    else:
        filtered_additional_symptoms = []
    
    # Calculate target number of additional symptoms (same as real symptoms)
    target_additional_count = min(
        int(len(conversation_symptoms) * ADDITIONAL_SYMPTOMS_RATIO),
        MAX_ABSOLUTE_ADDITIONAL_SYMPTOMS #TODO: check if this is necessary now that symptoms cannot be more than target
    )
    
    # If we have more filtered symptoms than our target, randomly select a subset
    if len(filtered_additional_symptoms) > target_additional_count:
        print(f"Limiting additional symptoms from {len(filtered_additional_symptoms)} to {target_additional_count} " +
              f"(matching real symptom count of {len(conversation_symptoms)})")
        random.seed(int(CONVERSATION_NUMBER))  # Use conversation number as seed for consistent results
        additional_symptoms = random.sample(filtered_additional_symptoms, target_additional_count)
    else:
        print(f"Using all {len(filtered_additional_symptoms)} filtered additional symptoms " +
              f"(fewer than real symptom count of {len(conversation_symptoms)})")
        additional_symptoms = filtered_additional_symptoms #check how many times there are less fake symptoms than real symptoms
    
    # Combine symptoms: those from the current conversation, plus filtered additions
    combined_symptoms = set(conversation_symptoms)
    for symptom in additional_symptoms:
        combined_symptoms.add(symptom)
    
    # Sort the symptoms alphabetically
    sorted_symptoms = sorted(combined_symptoms) #TODO: check if this is necessary
    
    # Save to output file - flat list of symptoms
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(sorted_symptoms, file, indent=2)
    
    print(f"\nCriteria file successfully generated at {output_file}")
    print(f"- {len(conversation_symptoms)} real symptoms from this conversation")
    print(f"- {len(additional_symptoms)} additional symptoms from comprehensive list")
    print(f"- {len(sorted_symptoms)} total symptoms in criteria file")
    
    return sorted_symptoms

def main():
    """Main function to create criteria from annotations."""
    if len(sys.argv) < 3:
        print("Usage: python create_criteria.py <annotations_file> <output_criteria_file> [conversation_number] [all_symptoms_file] [--api-key KEY]")
        sys.exit(1)
    
    annotations_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check for API key argument
    api_key = None
    if "--api-key" in sys.argv:
        api_key_index = sys.argv.index("--api-key")
        if api_key_index + 1 < len(sys.argv):
            api_key = sys.argv[api_key_index + 1]
    
    # Select model interactively or from environment
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
    
    # If conversation number is provided, update the global variable
    if len(sys.argv) >= 4 and sys.argv[3] != "--api-key":
        global CONVERSATION_NUMBER
        CONVERSATION_NUMBER = sys.argv[3]
    
    # If all_symptoms file is provided, use it
    all_symptoms_file = ALL_SYMPTOMS_FILE
    if len(sys.argv) >= 5 and sys.argv[4] != "--api-key":
        all_symptoms_file = sys.argv[4]
    
    # Generate criteria from annotations
    generate_criteria_from_annotations(annotations_file, output_file, model_interface, all_symptoms_file)
    
    print(f"Completed creating criteria for conversation {CONVERSATION_NUMBER}")

if __name__ == "__main__":
    main()