import sys

try:
    import google.generativeai
    print("google-generativeai installed")
except ImportError:
    print("google-generativeai NOT installed")

try:
    import transformers
    print("transformers installed")
except ImportError:
    print("transformers NOT installed")

try:
    import torch
    print(f"torch installed: {torch.__version__}, cuda: {torch.cuda.is_available()}")
except ImportError:
    print("torch NOT installed")
