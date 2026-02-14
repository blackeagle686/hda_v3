import sys
import os
# Add project root to path
sys.path.append(os.getcwd())

from ml.llm_pipeline import LLMResponder

def test_llm():
    print("Initializing LLMResponder...")
    try:
        llm = LLMResponder(mock=False)
    except Exception as e:
        print(f"Failed to init LLM: {e}")
        return

    print("Generating response...")
    response = llm.generate_response("Hello, I have a cough.", context="Patient has history of smoking.")
    print(f"Response: {response}")

if __name__ == "__main__":
    test_llm()
