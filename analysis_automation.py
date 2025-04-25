import os
import json
import sys
import subprocess
import argparse
from typing import Dict, Any, List, Set
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    for i in tqdm(range(start_id, end_id + 1), desc=f"Extracting {file_prefix} files"):
        conversation_id = str(i)
        if extract_conversation(data, conversation_id, output_dir, file_prefix):
            successful_ids.append(conversation_id)
    
    print(f"Successfully extracted {len(successful_ids)} {file_prefix} files")
    return successful_ids

def create_transcript(dialog_file: str, output_file: str):
    """Create a transcript from a dialog file using the Node.js script."""
    # Create a temporary JavaScript file that uses the conversion function
    # Use a unique filename based on the conversation ID to avoid conflicts
    conversation_id = os.path.basename(dialog_file).replace("dialogs", "").replace(".json", "")
    temp_js = f"temp_convert_{conversation_id}.js"
    
    with open(temp_js, 'w', encoding='utf-8') as file:
        file.write('''const fs = require('fs');

function convertDialogsToTranscript(inputFile, outputFile) {
  // Read the JSON file
  fs.readFile(inputFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading file: ${err}`);
      process.exit(1);
      return;
    }

    try {
      // Parse the JSON data
      const dialogsData = JSON.parse(data);
      let transcript = '';

      // Process each dialog
      Object.values(dialogsData).forEach(dialog => {
        // Process each utterance in the dialog
        dialog.utterances.forEach(utterance => {
          // Format the speaker role properly
          let speakerRole;
          if (utterance.speaker === "doctor") {
            speakerRole = "Medical Student";
          } else {
            // Capitalize first letter for other roles
            speakerRole = utterance.speaker.charAt(0).toUpperCase() + utterance.speaker.slice(1);
          }
          
          // Format the line according to the specified format
          transcript += `**${speakerRole}: **${utterance.text}\\n`;
        });
        
        // Add a newline between different dialogs
        transcript += '\\n';
      });

      // Write the transcript to the output file
      fs.writeFile(outputFile, transcript, err => {
        if (err) {
          console.error(`Error writing transcript file: ${err}`);
          process.exit(1);
          return;
        }
        console.log(`Transcript successfully written to ${outputFile}`);
        process.exit(0);
      });
    } catch (error) {
      console.error(`Error parsing JSON: ${error}`);
      process.exit(1);
    }
  });
}

// Call the function with the command line arguments
const inputFile = process.argv[2];
const outputFile = process.argv[3];
convertDialogsToTranscript(inputFile, outputFile);
''')
    
    try:
        # Run the Node.js script with a timeout
        result = subprocess.run(['node', temp_js, dialog_file, output_file], 
                               capture_output=True, text=True, timeout=30)
        
        # Check the result
        if result.returncode != 0:
            print(f"Error creating transcript: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout while creating transcript for {dialog_file}")
        return False
    except Exception as e:
        print(f"Exception while creating transcript: {e}")
        return False
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_js):
                os.remove(temp_js)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_js}: {e}")
    
    # Verify the transcript file was created
    if not os.path.exists(output_file):
        print(f"Transcript file {output_file} was not created")
        return False
    
    return True

def process_conversation(conversation_id: str, directories: List[str]):
    """Process a single conversation through the entire pipeline."""
    conversation_id_str = str(conversation_id)
    
    # File paths
    dialog_file = os.path.join(directories[1], f"dialogs{conversation_id_str}.json")
    annotation_file = os.path.join(directories[2], f"annotations{conversation_id_str}.json")
    transcript_file = os.path.join(directories[0], f"transcript{conversation_id_str}.txt")
    criteria_file = os.path.join(directories[3], f"criteria{conversation_id_str}.json")
    results_file = os.path.join(directories[4], f"results{conversation_id_str}.json")
    metrics_file = os.path.join(directories[5], f"metrics{conversation_id_str}.json")
    
    # Step 1: Create transcript from dialog
    if not os.path.exists(transcript_file):
        print(f"Creating transcript for conversation {conversation_id_str}")
        if not create_transcript(dialog_file, transcript_file):
            print(f"Failed to create transcript for conversation {conversation_id_str}")
            return False
    
    # Step 2: Create criteria from annotations
    if not os.path.exists(criteria_file):
        print(f"Creating criteria for conversation {conversation_id_str}")
        create_criteria_cmd = [
            'python', 'create_criteria.py', 
            annotation_file, criteria_file, conversation_id_str, 'all_symptoms.json'
        ]
        result = subprocess.run(create_criteria_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating criteria for conversation {conversation_id_str}: {result.stderr}")
            return False
    
    # Step 3: Create results from transcript and criteria
    if not os.path.exists(results_file):
        print(f"Creating results for conversation {conversation_id_str}")
        create_results_cmd = [
            'python', 'create_results.py',
            transcript_file, criteria_file, results_file
        ]
        result = subprocess.run(create_results_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating results for conversation {conversation_id_str}: {result.stderr}")
            return False
    
    # Step 4: Calculate metrics from results and annotations
    if not os.path.exists(metrics_file):
        print(f"Calculating metrics for conversation {conversation_id_str}")
        metrics_cmd = [
            'python', 'metrics_evaluation.py',
            conversation_id_str,
            f"--results={results_file}",
            f"--annotations={annotation_file}",
            f"--criteria={criteria_file}",
            f"--output={metrics_file}"
        ]
        result = subprocess.run(metrics_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error calculating metrics for conversation {conversation_id_str}: {result.stderr}")
            return False
    
    return True

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
    
    return aggregated_metrics

def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Automate medical dialogue analysis pipeline")
    
    parser.add_argument("--dialogs", default="dialogs.json", help="Main dialogs JSON file")
    parser.add_argument("--annotations", default="annotations.json", help="Main annotations JSON file")
    parser.add_argument("--start", type=int, default=101, help="Starting conversation ID")
    parser.add_argument("--end", type=int, default=201, help="Ending conversation ID")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all conversations")
    
    args = parser.parse_args()
    
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
    
    print(f"Processing {len(conversation_ids)} conversations...")
    
    # Process each conversation
    successful_conversations = []
    
    if args.parallel > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_id = {
                executor.submit(process_conversation, conv_id, directories): conv_id 
                for conv_id in conversation_ids
            }
            
            for future in tqdm(as_completed(future_to_id), total=len(conversation_ids), desc="Processing conversations"):
                conv_id = future_to_id[future]
                try:
                    if future.result():
                        successful_conversations.append(conv_id)
                except Exception as e:
                    print(f"Error processing conversation {conv_id}: {e}")
    else:
        # Sequential processing
        for conv_id in tqdm(conversation_ids, desc="Processing conversations"):
            if process_conversation(conv_id, directories):
                successful_conversations.append(conv_id)
    
    print(f"Successfully processed {len(successful_conversations)} out of {len(conversation_ids)} conversations")
    
    # Calculate aggregated metrics
    if successful_conversations:
        calculate_aggregated_metrics(directories[5], successful_conversations)
    else:
        print("No conversations were successfully processed. Cannot calculate aggregated metrics.")

if __name__ == "__main__":
    main()