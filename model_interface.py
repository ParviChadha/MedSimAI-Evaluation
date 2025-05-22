import os
import sys
import json
import importlib
import time
import re
from typing import Dict, Any, List, Optional

# Add these imports for proper multiprocessing support
import multiprocessing
from functools import partial

def parse_json(response):
    """
    Parse JSON from a response that might include markdown formatting.
    
    Parameters:
        response (str): The response text, potentially containing markdown code blocks
        
    Returns:
        dict: The parsed JSON object, or an error object if parsing fails
    """
    try:
        # First, check if the response contains a markdown JSON code block
        if "```json" in response:
            # Extract the JSON from the markdown code block
            match = re.search(r'```json\n([\s\S]*?)\n```', response)
            if match:
                json_content = match.group(1).strip()
                return json.loads(json_content)
        
        # If no markdown block or couldn't extract, try to find JSON object directly
        json_match = re.search(r'(\{[\s\S]*\})', response)
        if json_match:
            json_content = json_match.group(1)
            return json.loads(json_content)
        
        # If all else fails, try to parse the entire response
        return json.loads(response)
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing error: {e}")
        print(f"Response excerpt: {response[:200]}...")
        # Return a valid but empty structure to avoid null reference errors
        return {
            "medical_history_assessed": {},
            "error": "Failed to parse JSON response",
            "raw_content": response[:1000]  # Include truncated content for debugging
        }

# Dynamically import libraries if available
def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

class ModelInterface:
    """Unified interface for multiple LLM models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name.lower()
        
        # Use provided API key or get from environment
        if api_key is None:
            api_key = self._get_api_key_from_env()
        
        if api_key is None:
            raise ValueError(f"No API key found for {model_name}. Please provide via argument or environment variable.")
        
        # Store API key for potential recreation in worker processes
        self.api_key = api_key
        
        # Track initialization state
        self.initialized = False
        
        # Initialize the appropriate client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the client based on model name"""
        if self.initialized:
            return
            
        if self.model_name == "openai":
            self._init_openai_client(self.api_key)
        elif self.model_name == "claude":
            anthropic = safe_import("anthropic")
            if anthropic is None:
                raise ValueError("Anthropic library not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.model_name == "gemini":
            genai = safe_import("google.genai")
            if genai is None:
                raise ValueError("Google AI library not installed. Run: pip install google-generativeai")
            self.client = genai.Client(api_key=self.api_key)
        elif self.model_name == "fireworks":
            fireworks_client = safe_import("fireworks.client")
            if fireworks_client is None:
                raise ValueError("Fireworks library not installed. Run: pip install fireworks-ai")
            self.client = fireworks_client.Fireworks(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        self.initialized = True
    
    def _init_openai_client(self, api_key: str):
        """Initialize OpenAI client with improved error handling"""
        print("Initializing OpenAI client...")
        
        # Try the standard initialization without any proxy configuration
        try:
            # Import at function level to avoid any global monkey-patching
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.openai_client_type = "new"
            print("Successfully initialized OpenAI client using standard method")
            return
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
        
        # If standard method fails, try creating a client with explicit timeout
        try:
            print("Trying with explicit timeout...")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=120.0)
            self.openai_client_type = "new_with_timeout"
            print("Successfully initialized OpenAI client with explicit timeout")
            return
        except Exception as e:
            print(f"Error initializing with timeout: {e}")
        
        # If both standard methods fail, try our minimal custom implementation
        try:
            print("Trying minimal custom client implementation...")
            import httpx
            
            class MinimalOpenAI:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.base_url = "https://api.openai.com/v1"
                    self.http_client = httpx.Client(
                        base_url=self.base_url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=120.0  # 3 minute timeout
                    )
                    # Create a chat namespace with completions method
                    self.chat = type('ChatNamespace', (), {})()
                    self.chat.completions = type('CompletionsNamespace', (), {
                        'create': self.create_completion
                    })()
                
                def create_completion(self, model, messages, response_format=None, **kwargs):
                    """Direct API call to OpenAI's chat completion endpoint with retry logic"""
                    data = {
                        "model": model,
                        "messages": messages,
                        **kwargs
                    }
                    
                    # Add response_format if provided
                    if response_format:
                        data["response_format"] = response_format
                    
                    # Retry mechanism
                    max_retries = 3
                    backoff_factor = 2
                    
                    for attempt in range(max_retries):
                        try:
                            print(f"API Request attempt {attempt+1}/{max_retries}...")
                            response = self.http_client.post(
                                "/chat/completions",
                                json=data,
                                timeout=120.0  # Ensure timeout is set here too
                            )
                            
                            if response.status_code != 200:
                                error_content = response.text
                                raise ValueError(f"API Error: {response.status_code} - {error_content}")
                            
                            # Return a response-like object that matches the OpenAI client's structure
                            resp_data = response.json()
                            
                            # Create a choices object that has the expected structure
                            class Choice:
                                def __init__(self, choice_data):
                                    self.message = type('Message', (), {
                                        'content': choice_data['message']['content']
                                    })
                            
                            # Create a response object with the choices
                            class Response:
                                def __init__(self, data):
                                    self.choices = [Choice(choice) for choice in data['choices']]
                            
                            return Response(resp_data)
                            
                        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                            # If we've used all retries, re-raise the exception
                            if attempt == max_retries - 1:
                                print(f"All retry attempts failed. Last error: {e}")
                                raise
                            
                            # Otherwise, wait and retry
                            wait_time = backoff_factor ** attempt
                            print(f"Request timed out. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        
                        except Exception as e:
                            # For other errors, don't retry
                            print(f"Unexpected error in API request: {e}")
                            raise
            
            self.client = MinimalOpenAI(api_key=api_key)
            self.openai_client_type = "minimal"
            print("Successfully initialized OpenAI client using minimal custom implementation")
            return
        except Exception as e:
            print(f"Error with minimal client: {e}")
        
        # If all methods fail, raise an error
        raise ValueError("All methods to initialize OpenAI client failed. Please try a different model.")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables"""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY", 
            "gemini": "GOOGLE_API_KEY",
            "fireworks": "FIREWORKS_API_KEY"
        }
        
        env_var = env_map.get(self.model_name)
        if env_var:
            return os.environ.get(env_var)
        return None
    
    def call_model(self, system_prompt: str, user_message: str, response_type: str = "json_object") -> Any:
        """Call the model with a unified interface"""
        # Ensure client is initialized (important for multiprocessing)
        if not self.initialized:
            self._initialize_client()
            
        if self.model_name == "openai":
            return self._call_openai(system_prompt, user_message, response_type)
        elif self.model_name == "claude":
            return self._call_claude(system_prompt, user_message, response_type)
        elif self.model_name == "gemini":
            return self._call_gemini(system_prompt, user_message, response_type)
        elif self.model_name == "fireworks":
            return self._call_fireworks(system_prompt, user_message, response_type)
    
    def _call_openai(self, system_prompt: str, user_message: str, response_type: str) -> Any:
        """Call OpenAI API with retry logic"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if hasattr(self, 'openai_client_type'):
            client_type = self.openai_client_type
        else:
            client_type = "unknown"
        
        print(f"Using OpenAI client type: {client_type}")
        print(f"System prompt length: {len(system_prompt)} characters")
        print(f"User message length: {len(user_message)} characters")
        print(f"Using model: gpt-4o-mini-2024-07-18")
        
        # Retry mechanism (only for non-minimal client types)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if client_type == "new" or client_type == "custom_clean":
                    # Standard new client (OpenAI >= 1.0.0)
                    print(f"API Request attempt {attempt+1}/{max_retries}...")
                    
                    # Try with a smaller, faster model first to avoid timeouts
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                        # No JSON mode for more compatibility
                        timeout=120  # 3 minute timeout
                    )
                    content = response.choices[0].message.content
                    
                elif client_type == "minimal":
                    # Our minimal custom implementation (already has retries)
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",
                        messages=messages,
                        # No JSON mode for more compatibility
                    )
                    content = response.choices[0].message.content
                
                else:
                    raise ValueError(f"Unknown OpenAI client type: {client_type}")
                
                # If we get here, the request was successful
                break
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    # This was our last attempt
                    raise
                
                # Wait before retrying
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
        # Parse JSON if needed
        if response_type == "json_object" and isinstance(content, str):
            # Add special handling for JSON
            try:
                # First try to parse directly
                return json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, look for JSON in markdown blocks
                if "```json" in content:
                    try:
                        json_content = content.split("```json")[1].split("```")[0].strip()
                        return json.loads(json_content)
                    except (IndexError, json.JSONDecodeError):
                        # If that fails too, try one more approach
                        try:
                            # Try to find anything that looks like JSON
                            import re
                            json_match = re.search(r'\{[\s\S]*\}', content)
                            if json_match:
                                return json.loads(json_match.group(0))
                        except:
                            pass
                
                # If all parsing attempts fail
                print(f"WARNING: Could not parse JSON response. Returning raw content.")
                print(f"First 500 chars of content: {content[:500]}...")
                
                # Fallback - try to construct something semi-valid as JSON
                try:
                    # Convert the LLM response to a simple "raw_content" JSON object
                    return {"error": "Failed to parse JSON", "raw_content": content}
                except:
                    # If all else fails, return a minimal error object
                    return {"error": "Failed to create JSON"}
        
        return content
    
    def _call_claude(self, system_prompt: str, user_message: str, response_type: str) -> Any:
        """Call Claude API"""
        # Claude uses a different message format
        messages = [
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        # Claude expects JSON mode to be handled differently
        if response_type == "json_object":
            system_prompt += "\n\nYou must respond with valid JSON only. No additional text or formatting."
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Using a verified stable Claude model
                max_tokens=4096,
                temperature=0,
                system=system_prompt,
                messages=messages
            )
            
            # Handle Claude's response format
            if response.content and len(response.content) > 0:
                content = response.content[0].text
                if response_type == "json_object":
                    # Clean up any potential formatting issues
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    return json.loads(content)
                return content
            else:
                raise ValueError("Empty response from Claude")
                
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise
    
    def _call_gemini(self, system_prompt: str, user_message: str, response_type: str) -> Any:
        """Call Gemini API"""
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_message}"
        
        if response_type == "json_object":
            full_prompt += "\n\nRespond with valid JSON only."
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=full_prompt,
        )
        
        content = response.text
        if response_type == "json_object":
            # Clean up any markdown formatting that Gemini might add
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        return content
    
    def _call_fireworks(self, system_prompt: str, user_message: str, response_type: str) -> Any:
        """Call Fireworks API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if response_type == "json_object":
            messages[0]["content"] += "\n\nYou must respond with valid JSON only."
        
        try:
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/gemma-3-27b-it",
                messages=messages,
                max_tokens=4096
            )
        
            content = response.choices[0].message.content
        
            # Log the content for debugging
            with open("fireworks_content.log", "a") as log_file:
                log_file.write(f"Raw Fireworks content: {content}\n\n")
            
            if response_type == "json_object":
                # Parse the JSON content
                parsed_content = parse_json(content)
                
                # Log the parsed content for debugging
                with open("fireworks_parsed.log", "a") as log_file:
                    log_file.write(f"Parsed content: {parsed_content}\n\n")
                
                return parsed_content
            
            return content
    
        except Exception as e:
            print(f"Error in Fireworks API call: {e}")
            with open("fireworks_error.log", "a") as log_file:
                log_file.write(f"Error: {e}\n\n")
            
            # Return an empty structure to avoid downstream errors
            if response_type == "json_object":
                return {
                    "medical_history_assessed": {},
                    "error": f"API error: {str(e)}"
                }
            
            return f"Error: {str(e)}"

    
        # create log file if it doesn't exist
        # if not os.path.exists("fireworks_response.log"):
        #     with open("fireworks_response.log", "w") as log_file:
        #         log_file.write("Fireworks response log\n")
        # # send response to a log file
        # with open("fireworks_response.log", "a") as log_file:
        #     log_file.write(f"Fireworks response: {response}\n")

        
        #content = response.choices[0].message.content
        #content = parse_json(response.choices[0].message.content)
        # #create log file if it doesn't exist
        # if not os.path.exists("fireworks_content.log"):
        #     with open("fireworks_content.log", "w") as log_file:
        #         log_file.write("Fireworks content log\n")
        # # send content to a log file
        # with open("fireworks_content.log", "a") as log_file:
        #     log_file.write(f"Fireworks content: {content}\n")

        # if response_type == "json_object":
        #     return json.loads(content)
        # return content

# Helper function for process_conversation that creates its own ModelInterface
def process_conversation_worker(conv_id, model_name, api_key=None):
    """Process a single conversation with a new ModelInterface instance"""
    try:
        # Import necessary modules within the worker
        from create_transcript import create_transcript
        import subprocess
        import os
        
        print(f"Worker processing conversation {conv_id}")
        
        # Create a fresh ModelInterface for this worker
        model_interface = ModelInterface(model_name, api_key)
        
        # File paths
        transcript_file = f"transcripts/transcript{conv_id}.txt"
        dialog_file = f"dialogs/dialogs{conv_id}.json"
        annotation_file = f"annotations/annotations{conv_id}.json"
        criteria_file = f"criteria/criteria{conv_id}.json"
        results_file = f"results/results{conv_id}.json"
        metrics_file = f"metrics/metrics{conv_id}.json"
        
        # Step 1: Create transcript from dialog
        if not os.path.exists(transcript_file):
            print(f"Creating transcript for conversation {conv_id}")
            if not create_transcript(dialog_file, transcript_file):
                print(f"Failed to create transcript for conversation {conv_id}")
                return False
        
        # Step 2: Create criteria from annotations with the model
        if not os.path.exists(criteria_file):
            print(f"Creating criteria for conversation {conv_id} using {model_name}")
            create_criteria_cmd = [
                'python', 'create_criteria.py', 
                annotation_file, criteria_file, conv_id, 'all_symptoms.json'
            ]
            
            if api_key:
                create_criteria_cmd.extend(['--api-key', api_key])
            
            # Set environment variable to pass model selection
            env = os.environ.copy()
            env['SELECTED_MODEL'] = model_name
            
            # Use subprocess.run for simplicity in worker processes
            process = subprocess.run(
                create_criteria_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                input={"openai": "1", "claude": "2", "gemini": "3", "fireworks": "4"}.get(model_name, "1")
            )
            
            if process.returncode != 0:
                print(f"Error creating criteria for conversation {conv_id}: {process.stderr}")
                return False
        
        # Step 3: Create results from transcript and criteria
        if not os.path.exists(results_file):
            print(f"Creating results for conversation {conv_id} using {model_name}")
            create_results_cmd = [
                'python', 'create_results.py',
                transcript_file, criteria_file, results_file
            ]
            
            if api_key:
                create_results_cmd.extend(['--api-key', api_key])
            
            # Set environment variable to pass model selection
            env = os.environ.copy()
            env['SELECTED_MODEL'] = model_name
            
            # Use subprocess.run for simplicity in worker processes
            process = subprocess.run(
                create_results_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                input={"openai": "1", "claude": "2", "gemini": "3", "fireworks": "4"}.get(model_name, "1")
            )
            
            if process.returncode != 0:
                print(f"Error creating results for conversation {conv_id}: {process.stderr}")
                return False
        
        # Step 4: Calculate metrics from results and annotations
        if not os.path.exists(metrics_file):
            print(f"Calculating metrics for conversation {conv_id}")
            metrics_cmd = [
                'python', 'metrics_evaluation.py',
                conv_id,
                f"--results={results_file}",
                f"--annotations={annotation_file}",
                f"--criteria={criteria_file}",
                f"--output={metrics_file}"
            ]
            
            process = subprocess.run(
                metrics_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                print(f"Error calculating metrics for conversation {conv_id}: {process.stderr}")
                return False
        
        return True
    except Exception as e:
        print(f"Error in worker processing conversation {conv_id}: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def select_model() -> str:
    """Interactive model selection in terminal"""
    print("\nSelect the model to use:")
    print("1. OpenAI (GPT-4)")
    print("2. Claude (Anthropic)")
    print("3. Gemini (Google)")
    print("4. Fireworks (Llama)")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        model_map = {
            "1": "openai",
            "2": "claude", 
            "3": "gemini",
            "4": "fireworks"
        }
        
        if choice in model_map:
            selected_model = model_map[choice]
            print(f"\nSelected model: {selected_model}")
            return selected_model
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")