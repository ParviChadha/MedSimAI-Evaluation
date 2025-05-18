import os
import sys
import json
from typing import Dict, Any

def create_transcript(dialog_file: str, output_file: str) -> bool:
    """Create a transcript from a dialog file."""
    # Load dialog data
    try:
        with open(dialog_file, 'r', encoding='utf-8') as file:
            dialog_data = json.load(file)
    except Exception as e:
        print(f"Error reading dialog file {dialog_file}: {e}")
        return False
    
    # Generate transcript
    transcript = ""
    
    # Process each dialog
    for dialog_id, dialog in dialog_data.items():
        # Process utterances
        if "utterances" not in dialog:
            print(f"Warning: No utterances found in dialog {dialog_id}")
            continue
            
        for utterance in dialog["utterances"]:
            # Format speaker role
            speaker = utterance.get("speaker", "")
            speaker_role = "Medical Student" if speaker.lower() == "doctor" else speaker.capitalize()
            
            # Get utterance text
            text = utterance.get("text", "")
            
            # Format according to specified format
            transcript += f"**{speaker_role}: **{text}\n"
    
    # Write transcript to file
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(transcript)
        return True
    except Exception as e:
        print(f"Error writing transcript to {output_file}: {e}")
        return False

if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python create_transcript.py <dialog_file> <output_file>")
        sys.exit(1)
    
    dialog_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if create_transcript(dialog_file, output_file):
        print(f"Transcript successfully written to {output_file}")
    else:
        print(f"Failed to create transcript from {dialog_file}")
        sys.exit(1)