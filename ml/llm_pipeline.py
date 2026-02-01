from transformers import pipeline
import torch
import os

class LLMResponder:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mock: bool = False):
        """
        Initialize the LLM Responder.
        Args:
            model_name: HuggingFace model repo id.
            mock: If True, uses dummy responses instead of loading the model.
        """
        self.mock = mock
        self.model_name = model_name
        self.pipe = None
        
        if not self.mock:
            try:
                print(f"[INFO] Loading LLM: {model_name}...")
                # Check for GPU
                device = 0 if torch.cuda.is_available() else -1
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                self.pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=dtype,
                    device=device,
                    max_new_tokens=256
                )
                print("[INFO] LLM Loaded successfully.")
            except Exception as e:
                print(f"[WARN] Failed to load LLM ({e}). Switching to MOCK mode.")
                self.mock = True
        else:
            print("[INFO] LLM initialized in MOCK mode.")

    def generate_response(self, user_text: str, context: str = "") -> str:
        """
        Generate a response given user text and optional medical context.
        """
        if self.mock:
            return self._mock_response(user_text, context)
            
        # Prompt formatting for TinyLlama Chat
        # <|system|>
        # You are a helpful medical assistant.</s>
        # <|user|>
        # ...</s>
        # <|assistant|>
        
        system_prompt = "You are a helpful and empathetic AI medical assistant. You provide advice based on the analysis of medical images and user symptoms."
        
        full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\nContext: {context}\n\nUser Question: {user_text}</s>\n<|assistant|>\n"
        
        try:
            outputs = self.pipe(full_prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            generated_text = outputs[0]['generated_text']
            # Extract only the assistant part
            response = generated_text.split("<|assistant|>\n")[-1].strip()
            return response
        except Exception as e:
            print(f"[ERROR] LLM generation failed: {e}")
            return "I apologize, but I am having trouble generating a detailed response right now. Please consult a doctor."

    def _mock_response(self, user_text: str, context: str) -> str:
        """
        Return a dummy response for testing.
        """
        import time
        time.sleep(1) # Simulate delay
        
        if "Lung" in context:
            return (
                "Based on the analysis, there are signs that could indicate a lung condition. "
                "I recommend consulting a specialist for a biopsy. "
                "In the meantime, avoid smoking and monitor for shortness of breath."
            )
        elif "Colon" in context:
             return (
                "The image analysis has highlighted potential irregularities in the colon tissue. "
                "It is important to follow up with a colonoscopy for confirmation. "
                "Please maintain a balanced diet and stay hydrated."
            )
        else:
            return (
                "I am ready to help you. Please upload a medical image or ask a specific question. "
                f"(You said: {user_text})"
            )
