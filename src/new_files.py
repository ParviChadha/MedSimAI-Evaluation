#!/usr/bin/env python3
"""
Script to reorganize the project structure according to the new layout:

New structure:
main.py (renamed from analysis_automation.py)
data/
  ├── dialogs.json
  ├── annotations.json
  ├── dialogs/ (auto-created)
  ├── annotations/ (auto-created)
  └── transcripts/ (auto-created)
results/ (auto-created, stays in main directory)
output/
  ├── all_symptoms.json
  ├── criteria/ (auto-created)
  ├── metrics/ (auto-created)
  └── token_metrics/ (auto-created)
src/
  ├── model_interface.py
  ├── metrics_evaluation.py
  ├── create_results.py
  ├── create_criteria.py
  ├── extract_all_symptoms.py
  ├── create_transcript.py
  ├── combine_criteria.py
  └── token_metrics.py
"""

import os
import shutil
import sys

def create_new_structure():
    """Create the new directory structure"""
    directories = [
        "data",
        "data/dialogs",
        "data/annotations", 
        "data/transcripts",
        "output",
        "output/criteria",
        "output/metrics",
        "output/token_metrics",
        "src",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files():
    """Move files to their new locations"""
    # Files to move to src/
    src_files = [
        "model_interface.py",
        "metrics_evaluation.py", 
        "create_results.py",
        "create_criteria.py",
        "extract_all_symptoms.py",
        "create_transcript.py",
        "combine_criteria.py",
        "token_metrics.py"
    ]
    
    # Move Python files to src/
    for file in src_files:
        if os.path.exists(file):
            shutil.move(file, f"src/{file}")
            print(f"Moved {file} to src/")
    
    # Move data files to data/
    data_files = ["dialogs.json", "annotations.json"]
    for file in data_files:
        if os.path.exists(file):
            shutil.move(file, f"data/{file}")
            print(f"Moved {file} to data/")
    
    # Move all_symptoms.json to output/
    if os.path.exists("all_symptoms.json"):
        shutil.move("all_symptoms.json", "output/all_symptoms.json")
        print("Moved all_symptoms.json to output/")
    
    # Rename analysis_automation.py to main.py
    if os.path.exists("analysis_automation.py"):
        shutil.move("analysis_automation.py", "main.py")
        print("Renamed analysis_automation.py to main.py")
    
    # Move existing auto-created directories if they exist
    existing_dirs = {
        "dialogs": "data/dialogs",
        "annotations": "data/annotations", 
        "transcripts": "data/transcripts",
        "criteria": "output/criteria",
        "metrics": "output/metrics",
        "token_metrics": "output/token_metrics"
    }
    
    for old_dir, new_dir in existing_dirs.items():
        if os.path.exists(old_dir) and os.path.isdir(old_dir):
            # Move contents instead of the directory itself
            if os.listdir(old_dir):  # If directory is not empty
                for item in os.listdir(old_dir):
                    old_path = os.path.join(old_dir, item)
                    new_path = os.path.join(new_dir, item)
                    shutil.move(old_path, new_path)
                print(f"Moved contents of {old_dir}/ to {new_dir}/")
            # Remove the old empty directory
            os.rmdir(old_dir)

def update_imports_in_file(file_path, old_import, new_import):
    """Update import statements in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace the import statement
        updated_content = content.replace(old_import, new_import)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        if old_import in content:
            print(f"Updated imports in {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def main():
    """Main reorganization function"""
    print("Starting project reorganization...")
    
    # Create new directory structure
    create_new_structure()
    
    # Move files to new locations
    move_files()
    
    print("\nReorganization complete!")
    print("\nNew project structure:")
    print("main.py")
    print("data/")
    print("  ├── dialogs.json")
    print("  ├── annotations.json")
    print("  ├── dialogs/")
    print("  ├── annotations/")
    print("  └── transcripts/")
    print("results/")
    print("output/")
    print("  ├── all_symptoms.json")
    print("  ├── criteria/")
    print("  ├── metrics/")
    print("  └── token_metrics/")
    print("src/")
    print("  ├── model_interface.py")
    print("  ├── metrics_evaluation.py")
    print("  ├── create_results.py")
    print("  ├── create_criteria.py")
    print("  ├── extract_all_symptoms.py")
    print("  ├── create_transcript.py")
    print("  ├── combine_criteria.py")
    print("  └── token_metrics.py")

if __name__ == "__main__":
    main()