# measure_criteria_tokens.py
import os
import sys
import json
import re
import argparse
from tiktoken import encoding_for_model
from model_interface import ModelInterface, select_model
import time

def measure_criteria_tokens(conv_id: str, model_name: str = "claude"):
    """Directly measure token usage for criteria generation."""
    # Paths
    annotations_file = f"annotations/annotations{conv_id}.json"
    all_symptoms_file = "all_symptoms.json"
    token_metrics_dir = "token_metrics"
    
    # Check if annotations file exists
    if not os.path.exists(annotations_file):
        print(f"Annotations file {annotations_file} not found, skipping conversation {conv_id}")
        return False
    
    # Read annotations
    try:
        with open(annotations_file, 'r', encoding='utf-8') as file:
            annotations = json.load(file)
    except Exception as e:
        print(f"Error reading annotations file for conversation {conv_id}: {e}")
        return False
    
    # Extract symptoms
    unique_symptoms = set()
    for _, utterances in annotations.items():
        for utterance_list in utterances:
            for utterance in utterance_list:
                if utterance.get("slot") == "symptom" and "value" in utterance:
                    symptom_values = utterance["value"].split(",")
                    for symptom in symptom_values:
                        symptom = symptom.strip()
                        if symptom and symptom.lower() != "all":
                            unique_symptoms.add(symptom)
    
    print(f"Found {len(unique_symptoms)} unique symptoms in conversation {conv_id}")
    
    # Read all symptoms
    try:
        with open(all_symptoms_file, 'r', encoding='utf-8') as file:
            all_symptoms = json.load(file)
    except Exception as e:
        print(f"Error reading all symptoms file: {e}")
        return False
    
    # Identify additional symptoms (limit to 50 for API stability)
    additional_symptoms = [symptom for symptom in all_symptoms if symptom not in unique_symptoms][:50]
    
    # Construct a simplified prompt for Claude Haiku
    prompt = """
I need to filter a list of symptoms based on distinctness criteria.

Here are the existing symptoms: 
""" + str(list(unique_symptoms)) + """

Here are potential additional symptoms:
""" + str(additional_symptoms) + """

Please categorize the additional symptoms into two lists:
1. Keep - distinct symptoms that should be included
2. Exclude - symptoms that are too similar or redundant

Return ONLY a JSON object with these two arrays, like:
{"keep": ["symptom1", "symptom2"], "exclude": ["symptom3", "symptom4"]}
"""
    
    # Count input tokens
    enc = encoding_for_model("gpt-4")
    tokens_existing_symptoms = len(enc.encode(str(list(unique_symptoms))))
    tokens_additional_symptoms = len(enc.encode(str(additional_symptoms)))
    tokens_prompt = len(enc.encode(prompt))
    total_input_tokens = tokens_prompt + tokens_existing_symptoms + tokens_additional_symptoms
    
    print(f"Input token counts for conversation {conv_id}:")
    print(f"- Existing symptoms: {tokens_existing_symptoms}")
    print(f"- Additional symptoms: {tokens_additional_symptoms}")
    print(f"- Prompt: {tokens_prompt}")
    print(f"- Total input: {total_input_tokens}")
    
    # Initialize the model
    try:
        model_interface = ModelInterface(model_name)
    except Exception as e:
        print(f"Error initializing model for conversation {conv_id}: {e}")
        return False
    
    try:
        # Make a simplified API call
        response = model_interface.call_model(
            system_prompt="You are a medical expert assistant. Respond only with valid JSON in the exact format requested.",
            user_message=prompt,
            response_type="json_object"
        )
        
        # Count output tokens
        response_text = json.dumps(response)
        output_tokens = len(enc.encode(response_text))
        print(f"Output tokens: {output_tokens}")
        
        # Create metrics file
        os.makedirs(token_metrics_dir, exist_ok=True)
        metrics_file = os.path.join(token_metrics_dir, f"token_metrics_criteria_{conv_id}.json")
        
        from datetime import datetime
        metrics_data = {
            "conversation_id": conv_id,
            "model": model_name,
            "operation": "filter_similar_symptoms",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "input_tokens": {
                    "existing_symptoms": tokens_existing_symptoms,
                    "additional_symptoms": tokens_additional_symptoms,
                    "prompt": tokens_prompt,
                    "total": total_input_tokens
                },
                "output_tokens": output_tokens,
                "total_tokens": total_input_tokens + output_tokens,
                "manually_measured": True
            }
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as file:
            json.dump(metrics_data, file, indent=2)
        
        print(f"Created token metrics file for conversation {conv_id}")
        return True
        
    except Exception as e:
        print(f"Error processing conversation {conv_id}: {e}")
        return False

def process_range(start_id: int, end_id: int, model_name: str = "claude"):
    """Process a range of conversation IDs."""
    print(f"Processing conversations {start_id} to {end_id} using model {model_name}...")
    
    successful = 0
    failed = 0
    
    for conv_id in range(start_id, end_id + 1):
        print(f"\nProcessing conversation {conv_id}...")
        
        # Process the conversation
        result = measure_criteria_tokens(str(conv_id), model_name)
        
        if result:
            successful += 1
        else:
            failed += 1
        
        # Add a small delay between requests to avoid rate limits
        time.sleep(1)
    
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {successful}")
    print(f"- Failed: {failed}")
    print(f"- Total: {successful + failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure token usage for criteria generation")
    parser.add_argument("--start", type=int, required=True, help="Starting conversation ID")
    parser.add_argument("--end", type=int, required=True, help="Ending conversation ID")
    parser.add_argument("--model", default="claude", help="Model to use (default: claude)")
    
    args = parser.parse_args()
    
    if args.start > args.end:
        print("Error: Start ID must be less than or equal to end ID")
        sys.exit(1)
    
    process_range(args.start, args.end, args.model)