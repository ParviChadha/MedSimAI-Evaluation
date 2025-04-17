import json
import os
import sys
from typing import Dict, Any, List
import openai
from openai import OpenAI

# Initialize OpenAI client
client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

def openai_api_call(transcript: str, prompt: str, response_type: str = "json_object") -> Any:
    """Make a call to OpenAI API with the given transcript and prompt."""
    response = client_openai.chat.completions.create(
        model="gpt-4o",
        response_format={"type": response_type},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript}
        ],
    )
    return response

def evaluate_medical_history(transcript: str, symptoms: List[str]) -> Dict[str, Any]:
    """Evaluate whether the medical history topics were assessed in the conversation."""
    # Construct prompt from symptoms list
    prompt = construct_prompt(symptoms)
    
    # Make API call
    response = openai_api_call(transcript, prompt)
    
    # Extract and return the evaluation
    if hasattr(response, 'choices') and len(response.choices) > 0:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return json.loads(content)
        return content
    else:
        return response.choices[0].message.content

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    print(f"Results saved to {output_file}")

def main():
    """Main function to evaluate a transcript."""
    if len(sys.argv) < 4:
        print("Usage: python evaluate_transcript.py <transcript_file> <criteria_file> <output_results_file>")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    criteria_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Read transcript and criteria
    transcript = read_transcript(transcript_file)
    symptoms = read_assessment_criteria(criteria_file)
    
    print(f"Loaded {len(symptoms)} symptoms from criteria file")
    
    # Evaluate medical history
    print("Evaluating medical history in transcript...")
    results = evaluate_medical_history(transcript, symptoms)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Summary:")
    for item, details in results["medical_history_assessed"].items():
        print(f"{item.replace('_', ' ').title()}: {details['assessed']}")
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()