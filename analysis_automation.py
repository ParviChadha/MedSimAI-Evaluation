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
        "metrics",
        "token_metrics"  # New directory for token metrics
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

def summarize_token_metrics(token_metrics_dir: str, conversation_ids: List[str]):
    """Analyze and summarize token metrics across all processed conversations."""
    print("\nSummarizing token usage metrics...")
    
    # Check if token metrics directory exists
    if not os.path.exists(token_metrics_dir):
        print(f"Token metrics directory {token_metrics_dir} not found. Skipping token analysis.")
        return
    
    # Check for aggregated token metrics file
    aggregated_file = os.path.join(token_metrics_dir, "aggregated_token_metrics.json")
    if os.path.exists(aggregated_file):
        try:
            with open(aggregated_file, 'r', encoding='utf-8') as file:
                metrics = json.load(file)
            
            # Display token usage summary
            print("\nToken Usage Summary:")
            print(f"Total Conversations Analyzed: {metrics['total_conversations']}")
            print(f"Total Input Tokens:  {metrics['total_input_tokens']:,}")
            print(f"Total Output Tokens: {metrics['total_output_tokens']:,}")
            print(f"Total Tokens Used:   {metrics['total_tokens']:,}")
            
            # Check which key is available for average tokens
            if 'average_input_tokens_per_operation' in metrics:
                avg_input_key = 'average_input_tokens_per_operation'
                avg_output_key = 'average_output_tokens_per_operation'
                print(f"Average Input Tokens Per Operation:  {metrics[avg_input_key]:.1f}")
                print(f"Average Output Tokens Per Operation: {metrics[avg_output_key]:.1f}")
            elif 'average_input_tokens_per_conversation' in metrics:
                avg_input_key = 'average_input_tokens_per_conversation' 
                avg_output_key = 'average_output_tokens_per_conversation'
                print(f"Average Input Tokens Per Conversation:  {metrics[avg_input_key]:.1f}")
                print(f"Average Output Tokens Per Conversation: {metrics[avg_output_key]:.1f}")
            
            # Display operation-specific totals if available
            if "operation_totals" in metrics:
                print("\nToken Usage by Operation:")
                
                # Results metrics
                results = metrics["operation_totals"].get("results", {})
                if results:
                    print(f"\nMedical History Assessment (Results):")
                    print(f"  Input Tokens:  {results.get('input_tokens', 0):,}")
                    print(f"  Output Tokens: {results.get('output_tokens', 0):,}")
                    print(f"  Total Tokens:  {results.get('total_tokens', 0):,}")
                    print(f"  Operations:    {results.get('count', 0):,}")
                    if "avg_total_tokens" in results:
                        print(f"  Avg. Tokens per Operation: {results.get('avg_total_tokens', 0):,.1f}")
                
                # Criteria metrics
                criteria = metrics["operation_totals"].get("filter_similar_symptoms", {})
                if criteria:
                    print(f"\nSymptom Filtering (Criteria):")
                    print(f"  Input Tokens:  {criteria.get('input_tokens', 0):,}")
                    print(f"  Output Tokens: {criteria.get('output_tokens', 0):,}")
                    print(f"  Total Tokens:  {criteria.get('total_tokens', 0):,}")
                    print(f"  Operations:    {criteria.get('count', 0):,}")
                    if "avg_total_tokens" in criteria:
                        print(f"  Avg. Tokens per Operation: {criteria.get('avg_total_tokens', 0):,.1f}")
            
            # Display model-specific statistics
            if metrics.get("model_statistics"):
                print("\nToken Usage by Model:")
                for model, stats in metrics["model_statistics"].items():
                    print(f"\n{model.upper()} ({stats['conversations']} operations):")
                    print(f"  Input Tokens:  {stats['input_tokens']:,}")
                    print(f"  Output Tokens: {stats['output_tokens']:,}")
                    print(f"  Total Tokens:  {stats['total_tokens']:,}")
                    
                    # Show operation breakdowns if available
                    if "operations" in stats:
                        for op_name, op_stats in stats["operations"].items():
                            if op_stats.get("count", 0) > 0:
                                op_display_name = "Medical History Assessment" if op_name == "results" else "Symptom Filtering"
                                print(f"    {op_display_name}: {op_stats.get('input_tokens', 0):,} input + {op_stats.get('output_tokens', 0):,} output = {op_stats.get('input_tokens', 0) + op_stats.get('output_tokens', 0):,} tokens")
            
            return metrics
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading aggregated token metrics: {e}")
    
    # If we couldn't read the aggregated file, try to gather metrics from individual files
    print("No aggregated token metrics found. Analyzing individual metrics files...")
    
    # Read individual metrics files
    all_tokens = {
        "input": 0,
        "output": 0,
        "total": 0,
        "files_processed": 0
    }
    
    for conv_id in conversation_ids:
        # Check both regular and criteria metrics files
        metrics_files = [
            os.path.join(token_metrics_dir, f"token_metrics_{conv_id}.json"),
            os.path.join(token_metrics_dir, f"token_metrics_criteria_{conv_id}.json")
        ]
        
        for metrics_file in metrics_files:
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as file:
                        metrics = json.load(file)
                    
                    # Add to totals
                    metrics_data = metrics.get("metrics", {})
                    
                    # Handle different input token formats
                    input_tokens = 0
                    if isinstance(metrics_data.get("input_tokens"), dict):
                        input_tokens = metrics_data.get("input_tokens", {}).get("total", 0)
                    else:
                        input_tokens = metrics_data.get("input_tokens", 0)
                    
                    output_tokens = metrics_data.get("output_tokens", 0)
                    
                    all_tokens["input"] += input_tokens
                    all_tokens["output"] += output_tokens
                    all_tokens["total"] += (input_tokens + output_tokens)
                    all_tokens["files_processed"] += 1
                    
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading token metrics file {metrics_file}: {e}")
    
    # Display summary
    if all_tokens["files_processed"] > 0:
        print(f"\nToken Usage from {all_tokens['files_processed']} operations:")
        print(f"Total Input Tokens:  {all_tokens['input']:,}")
        print(f"Total Output Tokens: {all_tokens['output']:,}")
        print(f"Total Tokens Used:   {all_tokens['total']:,}")
        print(f"Average Tokens Per Operation: {all_tokens['total'] / all_tokens['files_processed']:.1f}")
    else:
        print("No token metrics files found.")
    
    return all_tokens

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
    parser.add_argument("--skip-token-summary", action="store_true", help="Skip token usage summary")
    
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
        
        # Summarize token metrics unless explicitly skipped
        if not args.skip_token_summary:
            summarize_token_metrics("token_metrics", successful_conversations)
    else:
        print("No conversations were successfully processed. Cannot calculate aggregated metrics.")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' for better compatibility on Windows
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    main()