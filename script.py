import json
import os
import sys
from typing import Dict, Any, List, Set
import openai
from openai import OpenAI

# Initialize OpenAI client
client_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration - Change this to set the conversation number
CONVERSATION_NUMBER = "110"  # Edit this line to change conversation number

def read_transcript(file_path: str) -> str:
    """Read transcript from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_assessment_criteria(file_path: str) -> Dict[str, Any]:
    """Read assessment criteria from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_annotations(file_path: str) -> Dict[str, Any]:
    """Read annotations from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_symptoms_from_annotations(annotations: Dict[str, List[List[Dict[str, Any]]]]) -> Set[str]:
    """Extract unique symptoms from annotations."""
    unique_symptoms = set()
    
    for conversation_id, utterances in annotations.items():
        for utterance_list in utterances:
            for utterance in utterance_list:
                if utterance.get("slot") == "symptom" and "value" in utterance:
                    # Add the symptom value to our set of unique symptoms
                    unique_symptoms.add(utterance["value"])
    
    return unique_symptoms

def categorize_symptoms(symptoms: Set[str]) -> Dict[str, List[str]]:
    """Categorize symptoms into appropriate categories."""
    # Default categories
    categories = {
        "Respiratory Symptoms": [],
        "ENT Symptoms": [],
        "Systemic Symptoms": [],
        "Pain Assessment": [],
        "Gastrointestinal Symptoms": [],
        "Urinary Symptoms": [],
        "Medical History": [],
        "Family History": [],
        "Medication Use": [],
        "Social History": [],
        "Lifestyle Factors": []
    }
    
    # Basic categorization rules - expand these based on your specific needs
    respiratory_keywords = ["cough", "wheeze", "breath", "sputum", "hemoptysis", "dyspnea", "chest congestion"]
    ent_keywords = ["throat", "nose", "smell", "taste", "nasal", "ear", "sinus"]
    systemic_keywords = ["fever", "chills", "fatigue", "weight", "sleep", "headache"]
    pain_keywords = ["pain", "ache", "sore", "hurt"]
    gi_keywords = ["nausea", "vomit", "bowel", "diarrhea", "constipation", "appetite"]
    urinary_keywords = ["urine", "urination", "bladder"]
    
    # Categorize each symptom
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        
        # Respiratory symptoms
        if any(keyword in symptom_lower for keyword in respiratory_keywords):
            categories["Respiratory Symptoms"].append(symptom)
        
        # ENT symptoms
        elif any(keyword in symptom_lower for keyword in ent_keywords):
            categories["ENT Symptoms"].append(symptom)
        
        # Systemic symptoms
        elif any(keyword in symptom_lower for keyword in systemic_keywords):
            categories["Systemic Symptoms"].append(symptom)
        
        # Pain assessment
        elif any(keyword in symptom_lower for keyword in pain_keywords):
            if "chest" in symptom_lower:
                categories["Pain Assessment"].append(symptom)
            else:
                categories["Systemic Symptoms"].append(symptom)
        
        # Gastrointestinal symptoms
        elif any(keyword in symptom_lower for keyword in gi_keywords):
            categories["Gastrointestinal Symptoms"].append(symptom)
        
        # Urinary symptoms
        elif any(keyword in symptom_lower for keyword in urinary_keywords):
            categories["Urinary Symptoms"].append(symptom)
        
        # Default category if no match
        else:
            categories["Systemic Symptoms"].append(symptom)
    
    # Remove empty categories
    return {k: sorted(v) for k, v in categories.items() if v}

def generate_criteria_from_annotations(annotations_file: str, output_file: str) -> None:
    """Generate criteria JSON from annotations file."""
    # Read annotations
    annotations = read_annotations(annotations_file)
    
    # Extract unique symptoms
    unique_symptoms = extract_symptoms_from_annotations(annotations)
    
    # Categorize symptoms
    categorized_symptoms = categorize_symptoms(unique_symptoms)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(categorized_symptoms, file, indent=2)
    
    print(f"Criteria file successfully generated at {output_file}")
    return categorized_symptoms

def construct_prompt(criteria: Dict[str, Any]) -> str:
    """Construct the evaluation prompt based on criteria."""
    prompt = "# Medical History Assessment Evaluation\n"
    prompt += "You are tasked with evaluating a conversation between a medical student and a patient "
    prompt += "to determine whether the student appropriately explored relevant medical history risk factors.\n\n"
    
    # Add sections for each category
    prompt += "## Medical History Categories to Evaluate\n"
    for category, items in criteria.items():
        prompt += f"### {category}\n"
        for item in items:
            prompt += f"- **{item}**\n"
    
    # Add output format instructions
    prompt += "\n## Required Output Format\n"
    prompt += "Provide your evaluation in the following JSON format:\n```json\n{\n"
    prompt += "  \"medical_history_assessed\": {\n"
    
    # Generate the JSON structure for all items
    flattened_items = []
    for category, items in criteria.items():
        for item in items:
            # Convert to snake_case for JSON keys
            key = item.lower().replace(' ', '_').replace('-', '_')
            flattened_items.append(key)
            prompt += f"    \"{key}\": {{\n"
            prompt += f"      \"assessed\": \"Yes/No\",\n"
            prompt += f"      \"example_quotes\": []\n"
            prompt += f"    }}"
            if item != items[-1] or category != list(criteria.keys())[-1]:
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

def evaluate_medical_history(transcript: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate whether the medical history topics were assessed in the conversation."""
    # Construct prompt from criteria
    prompt = construct_prompt(criteria)
    
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
    """Main function to run the evaluation."""
    if len(sys.argv) < 4:
        print("Usage: python script.py <transcript_file> <annotations_file> <output_file> [criteria_file]")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    annotations_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Generate default file names based on CONVERSATION_NUMBER
    default_criteria_file = f"criteria{CONVERSATION_NUMBER}.json"
    
    # If criteria file is provided, use it, otherwise generate from annotations
    if len(sys.argv) >= 5:
        criteria_file = sys.argv[4]
        criteria = read_assessment_criteria(criteria_file)
    else:
        # Use the default criteria file name
        criteria_file = default_criteria_file
        criteria = generate_criteria_from_annotations(annotations_file, criteria_file)
    
    # Read transcript
    transcript = read_transcript(transcript_file)
    
    # Evaluate medical history
    results = evaluate_medical_history(transcript, criteria)
    
    # Save results
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Summary:")
    for item, details in results["medical_history_assessed"].items():
        print(f"{item.replace('_', ' ').title()}: {details['assessed']}")

if __name__ == "__main__":
    main()