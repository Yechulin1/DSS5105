"""
Test script to debug API configuration issues
Run this file directly: python test_config.py
"""

import sys
import os

print("=" * 50)
print("🔍 CONFIGURATION DEBUG TEST")
print("=" * 50)

# 1. Check if config.py exists
print("\n1. Checking config.py file...")
if os.path.exists("config.py"):
    print("✅ config.py exists")
else:
    print("❌ config.py NOT found")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in directory: {os.listdir('.')}")

# 2. Try to import config
print("\n2. Trying to import config...")
try:
    from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE, MAX_TOKENS
    print("✅ Successfully imported config")
    print(f"   API Key: {'*' * 10}{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 4 else 'TOO SHORT'}")
    print(f"   Model: {OPENAI_MODEL}")
    print(f"   Temperature: {TEMPERATURE}")
    print(f"   Max Tokens: {MAX_TOKENS}")
    
    # Check if API key is placeholder
    if OPENAI_API_KEY == "your-api-key-here":
        print("⚠️  WARNING: API key is still the placeholder value!")
    elif not OPENAI_API_KEY.startswith("sk-"):
        print("⚠️  WARNING: API key doesn't start with 'sk-'. Is it valid?")
    else:
        print("✅ API key format looks correct")
        
except ImportError as e:
    print(f"❌ Failed to import config: {e}")
    print("   Make sure config.py has the correct format")
except Exception as e:
    print(f"❌ Error: {e}")

# 3. Check OpenAI library
print("\n3. Checking OpenAI library...")
try:
    import openai
    from openai import OpenAI
    print(f"✅ OpenAI library installed (version: {openai.__version__})")
except ImportError:
    print("❌ OpenAI library NOT installed")
    print("   Run: pip install openai")

# 4. Test API connection
print("\n4. Testing API connection...")
try:
    from config import OPENAI_API_KEY
    from openai import OpenAI
    
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-api-key-here":
        print(f"   Creating client with API key...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        print(f"   Sending test request...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test ok'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ API Connection successful! Response: {result}")
        print(f"   Model used: {response.model}")
    else:
        print("❌ Invalid or missing API key")
        
except Exception as e:
    print(f"❌ API connection failed: {e}")
    if "401" in str(e):
        print("   → Invalid API key (authentication failed)")
    elif "429" in str(e):
        print("   → Rate limit exceeded")
    elif "Connection" in str(e):
        print("   → Network connection issue")
    else:
        print(f"   → Error details: {type(e).__name__}")

# 5. Check Python path
print("\n5. Python environment info...")
print(f"   Python version: {sys.version}")
print(f"   Python executable: {sys.executable}")
print(f"   Current working directory: {os.getcwd()}")

print("\n" + "=" * 50)
print("Test complete! Check the output above for issues.")
print("=" * 50)