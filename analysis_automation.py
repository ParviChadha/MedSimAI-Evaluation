import os
import sys
import json
import importlib
import time
from typing import Dict, Any, List, Optional
import multiprocessing
import tqdm
from functools import partial
import argparse

# Import our ModelInterface 
from model_interface import ModelInterface, process_conversation_worker

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    directories = [
        "transcripts", 
        "dialogs", 
        "annotations", 
        "criteria", 
        "results", 
        "metrics"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return directories

def extract_conversation(data: Dict[str, Any], conversation_id: str, output_dir: str, file_prefix: str):
    """Extract a single conversation from the main JSON file and save it to a separate file."""
    if conversation_id not in data:
        print(f"Warning: Conversation {conversation_id} not found in data")
        return False
    
    # Create a new dictionary with just this conversation
    conversation_data = {conversation_id: data[conversation_id]}
    
    # Save to a separate file
    output_file = os.path.join(output_dir, f"{file_prefix}{conversation_id}.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(conversation_data, file, indent=2)
    
    return True

def extract_all_conversations(input_file: str, output_dir: str, file_prefix: str, start_id: int, end_id: int):
    """Extract all specified conversations from the main JSON file."""
    # Read the JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {input_file} is not a valid JSON file.")
        return []
    
    # Track successfully extracted conversations
    successful_ids = []
    
    # Extract each conversation
    for i in tqdm.tqdm(range(start_id, end_id + 1), desc=f"Extracting {file_prefix} files"):
        conversation_id = str(i)
        if extract_conversation(data, conversation_id, output_dir, file_prefix):
            successful_ids.append(conversation_id)
    
    print(f"Successfully extracted {len(successful_ids)} {file_prefix} files")
    return successful_ids

def calculate_aggregated_metrics(metrics_dir: str, conversation_ids: List[str]):
    """Calculate aggregated metrics across all processed conversations."""
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Metrics for each conversation
    individual_metrics = []
    
    # Collect metrics from all conversations
    for conversation_id in conversation_ids:
        metrics_file = os.path.join(metrics_dir, f"metrics{conversation_id}.json")
        try:
            with open(metrics_file, 'r', encoding='utf-8') as file:
                metrics = json.load(file)
                
                # Add to counters
                total_tp += metrics.get("true_positives", 0)
                total_fp += metrics.get("false_positives", 0)
                total_fn += metrics.get("false_negatives", 0)
                
                # Store individual metrics
                individual_metrics.append({
                    "conversation_id": conversation_id,
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "true_positives": metrics.get("true_positives", 0),
                    "false_positives": metrics.get("false_positives", 0),
                    "false_negatives": metrics.get("false_negatives", 0)
                })
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading metrics file for conversation {conversation_id}: {e}")
    
    # Calculate aggregated metrics
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
    else:
        precision = 0
    
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
    else:
        recall = 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    # Prepare the aggregated metrics
    aggregated_metrics = {
        "aggregated": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "total_conversations": len(individual_metrics)
        },
        "individual_metrics": individual_metrics
    }
    
    # Save aggregated metrics
    output_file = os.path.join(metrics_dir, "aggregated_metrics.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(aggregated_metrics, file, indent=2)
    
    print(f"\nAggregated Metrics Summary:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print(f"Total Conversations: {len(individual_metrics)}")
    print(f"Saved aggregated metrics to {output_file}")
    
    # Create a visual representation of the metrics distribution
    if len(individual_metrics) > 0:
        try:
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame(individual_metrics)
            
            # Calculate statistics
            stats = {
                "precision": {
                    "mean": df["precision"].mean(),
                    "median": df["precision"].median(),
                    "std": df["precision"].std(),
                    "min": df["precision"].min(),
                    "max": df["precision"].max()
                },
                "recall": {
                    "mean": df["recall"].mean(),
                    "median": df["recall"].median(),
                    "std": df["recall"].std(),
                    "min": df["recall"].min(),
                    "max": df["recall"].max()
                },
                "f1_score": {
                    "mean": df["f1_score"].mean(),
                    "median": df["f1_score"].median(),
                    "std": df["f1_score"].std(),
                    "min": df["f1_score"].min(),
                    "max": df["f1_score"].max()
                }
            }
            
            print("\nMetrics Distribution Statistics:")
            for metric, values in stats.items():
                print(f"\n{metric.title()}:")
                print(f"  Mean: {values['mean']:.4f}")
                print(f"  Median: {values['median']:.4f}")
                print(f"  Std Dev: {values['std']:.4f}")
                print(f"  Min: {values['min']:.4f}")
                print(f"  Max: {values['max']:.4f}")
            
            # Save statistics to file
            stats_file = os.path.join(metrics_dir, "metrics_statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as file:
                json.dump(stats, file, indent=2)
            print(f"\nSaved metrics statistics to {stats_file}")
        except ImportError:
            print("Pandas or NumPy not installed. Skipping detailed statistics.")
    
    return aggregated_metrics

def main():
    """Main function to run the entire pipeline with proper multiprocessing."""
    parser = argparse.ArgumentParser(description="Automate medical dialogue analysis pipeline")
    
    parser.add_argument("--dialogs", default="dialogs.json", help="Main dialogs JSON file")
    parser.add_argument("--annotations", default="annotations.json", help="Main annotations JSON file")
    parser.add_argument("--start", type=int, default=101, help="Starting conversation ID")
    parser.add_argument("--end", type=int, default=201, help="Ending conversation ID")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all conversations")
    parser.add_argument("--api-key", help="API key for the selected model")
    parser.add_argument("--model", help="Pre-select model (openai/claude/gemini/fireworks) to skip interactive selection")
    
    args = parser.parse_args()
    
    # Select model (interactively or from command line)
    if args.model:
        if args.model.lower() in ["openai", "claude", "gemini", "fireworks"]:
            selected_model = args.model.lower()
            print(f"Using pre-selected model: {selected_model}")
        else:
            print(f"Invalid model: {args.model}. Please choose from: openai, claude, gemini, fireworks")
            sys.exit(1)
    else:
        # Import here to avoid circular imports
        from model_interface import select_model
        selected_model = select_model()
    
    # Create directory structure
    directories = create_directory_structure()
    
    # If force flag is set, clear existing files
    if args.force:
        for directory in directories:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared all files in {directory}")
    
    # Extract individual conversations from main files
    extracted_dialogs = extract_all_conversations(
        args.dialogs, directories[1], "dialogs", args.start, args.end
    )
    
    extracted_annotations = extract_all_conversations(
        args.annotations, directories[2], "annotations", args.start, args.end
    )
    
    # Find common conversation IDs that were successfully extracted from both files
    conversation_ids = list(set(extracted_dialogs).intersection(set(extracted_annotations)))
    conversation_ids.sort()
    
    if not conversation_ids:
        print("Error: No conversations could be extracted from both files")
        return
    
    print(f"Processing {len(conversation_ids)} conversations with {selected_model} model...")
    
    # Process each conversation using our worker function
    successful_conversations = []

    if args.parallel > 1:
        # Parallel processing with the new worker function
        print(f"Note: Using parallel processing with {args.parallel} workers")
        
        # Create a partial function with the fixed arguments
        worker_func = partial(process_conversation_worker, 
                             model_name=selected_model, 
                             api_key=args.api_key)
        
        # Use a context manager to handle process cleanup
        with multiprocessing.Pool(processes=args.parallel) as pool:
            # Map the worker function to conversation IDs with a progress bar
            results = list(tqdm.tqdm(
                pool.imap(worker_func, conversation_ids),
                total=len(conversation_ids),
                desc="Processing conversations"
            ))
            
            # Collect successful conversations
            successful_conversations = [
                conv_id for conv_id, success in zip(conversation_ids, results) if success
            ]
    else:
        # Sequential processing 
        for conv_id in tqdm.tqdm(conversation_ids, desc="Processing conversations"):
            if process_conversation_worker(conv_id, selected_model, args.api_key):
                successful_conversations.append(conv_id)
    
    print(f"Successfully processed {len(successful_conversations)} out of {len(conversation_ids)} conversations")
    
    # Calculate aggregated metrics
    if successful_conversations:
        calculate_aggregated_metrics(directories[5], successful_conversations)
    else:
        print("No conversations were successfully processed. Cannot calculate aggregated metrics.")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for better compatibility on Windows
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()