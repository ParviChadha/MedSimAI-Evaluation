import os
import json
import re
from typing import Dict, Any, List, Optional
import sys

# Check OpenAI Python version
def check_openai_version() -> Optional[str]:
    try:
        import openai
        return openai.__version__
    except ImportError:
        return None

# Print diagnostic info
print(f"Python version: {sys.version}")
openai_version = check_openai_version()
print(f"OpenAI version: {openai_version if openai_version else 'Not installed'}")
print(f"In this script, we'll check for known issues with your OpenAI configuration...")

# Step 1: Check for proxy settings in environment
print("\nChecking for proxy settings in environment variables...")
proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"]
found_proxies = []
for var in proxy_vars:
    if var in os.environ:
        found_proxies.append(f"{var}={os.environ[var]}")

if found_proxies:
    print(f"Found proxy environment variables: {', '.join(found_proxies)}")
    print("These might be causing issues with the OpenAI client.")
else:
    print("No proxy environment variables found.")

# Step 2: Check if the openai_proxy.py module exists (which might be causing issues)
print("\nChecking for proxy configuration in OpenAI package...")

# Function to find openai package path
def find_openai_path() -> str:
    try:
        import openai
        return os.path.dirname(openai.__file__)
    except ImportError:
        return "OpenAI package not found"

openai_path = find_openai_path()
print(f"OpenAI package path: {openai_path}")

# Check for openai_proxy.py or similar files
if os.path.isdir(openai_path):
    proxy_related_files = [f for f in os.listdir(openai_path) if 'proxy' in f.lower()]
    if proxy_related_files:
        print(f"Found proxy-related files in OpenAI package: {', '.join(proxy_related_files)}")
    else:
        print("No proxy-related files found in OpenAI package.")
else:
    print("Could not check for proxy files (OpenAI package not found or path is not a directory)")

# Step 3: Test OpenAI initialization
print("\nTesting OpenAI client initialization...")

try:
    from openai import OpenAI
    
    # Try to initialize without any extra parameters
    client = OpenAI(api_key="test_key")
    print("SUCCESS: Basic OpenAI client initialization works!")
    
    # Try to access attributes to verify it's working
    print(f"Client has chat attribute: {hasattr(client, 'chat')}")
    print(f"Client has chat.completions attribute: {hasattr(client.chat, 'completions') if hasattr(client, 'chat') else False}")
    
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {str(e)}")
    
    # Check if it's the proxies error
    if "unexpected keyword argument 'proxies'" in str(e):
        print("\nDetected the 'proxies' error. This usually happens when:")
        print("1. Your OpenAI package has custom proxy settings")
        print("2. There's a conflict with environment variables")
        print("3. Another package or file is altering the OpenAI client initialization")

# Step 4: Print recommended fixes
print("\nRECOMMENDED FIXES:")
print("1. Update your OpenAI package to the latest version:")
print("   pip install --upgrade openai")
print("2. If proxy settings are needed for your environment, configure them properly for httpx")
print("   (the library OpenAI uses internally) rather than for the OpenAI client directly.")
print("3. Check if you have a custom version of the OpenAI library or any monkey-patching code")
print("   that might be modifying the OpenAI client initialization.")
print("4. Try using a different model (Claude, Gemini, etc.) if OpenAI continues to have issues.")

# Step 5: Generate a modified ModelInterface that should work
print("\nGENERATING PATCHED MODEL_INTERFACE.PY:")
print("I'll create a version of model_interface.py that should work despite the proxies issue.")