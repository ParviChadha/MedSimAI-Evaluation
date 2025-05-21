# debug_criteria.py
import os
import sys
import json
import subprocess

def debug_criteria_metrics():
    """Debug script to check criteria metrics generation."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python debug_criteria.py <conversation_id>")
        return
    
    conv_id = sys.argv[1]
    model_name = "claude"  # Or get from environment
    
    # Paths
    annotations_file = f"annotations/annotations{conv_id}.json"
    criteria_file = f"criteria/criteria{conv_id}.json"
    token_metrics_dir = "token_metrics"
    criteria_metrics_file = f"{token_metrics_dir}/token_metrics_criteria_{conv_id}.json"
    
    # Check if files exist
    print(f"Checking if annotations file exists: {os.path.exists(annotations_file)}")
    print(f"Checking if criteria file exists: {os.path.exists(criteria_file)}")
    print(f"Checking if token metrics directory exists: {os.path.exists(token_metrics_dir)}")
    if os.path.exists(token_metrics_dir):
        metrics_files = [f for f in os.listdir(token_metrics_dir) if f.startswith("token_metrics_")]
        print(f"Found {len(metrics_files)} token metrics files:")
        for f in metrics_files:
            print(f"  - {f}")
    
    # Check if criteria metrics file exists
    if os.path.exists(criteria_metrics_file):
        print(f"Criteria metrics file exists at {criteria_metrics_file}")
        try:
            with open(criteria_metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                print(f"Criteria metrics content: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            print(f"Error reading criteria metrics file: {e}")
    else:
        print(f"Criteria metrics file does not exist at {criteria_metrics_file}")
    
    # Try running create_criteria.py directly
    print("\nTrying to run create_criteria.py directly...")
    env = os.environ.copy()
    env['SELECTED_MODEL'] = model_name
    
    # Create token metrics directory if it doesn't exist
    os.makedirs(token_metrics_dir, exist_ok=True)
    
    # Run create_criteria.py directly
    create_criteria_cmd = [
        'python', 'create_criteria.py', 
        annotations_file, criteria_file, conv_id, 'all_symptoms.json'
    ]
    
    # Use subprocess.run
    try:
        process = subprocess.run(
            create_criteria_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"Process return code: {process.returncode}")
        print(f"Process stdout: {process.stdout}")
        print(f"Process stderr: {process.stderr}")
    except Exception as e:
        print(f"Error running create_criteria.py: {e}")
    
    # Check if criteria metrics file was created after direct run
    if os.path.exists(criteria_metrics_file):
        print(f"Criteria metrics file was created after direct run at {criteria_metrics_file}")
        try:
            with open(criteria_metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                print(f"Criteria metrics content after direct run: {json.dumps(metrics, indent=2)}")
        except Exception as e:
            print(f"Error reading criteria metrics file after direct run: {e}")
    else:
        print(f"Criteria metrics file was NOT created after direct run at {criteria_metrics_file}")

if __name__ == "__main__":
    debug_criteria_metrics()