import torch
import os
from typing import Dict, List
import google.generativeai as genai

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
                    max_new_tokens=1024 # TinyLlama max context is ~2048
                )
                print("[INFO] LLM Loaded successfully.")
            except Exception as e:
                print(f"[WARN] Failed to load LLM ({e}). Switching to MOCK mode.")
                self.mock = True
        else:
            print("[INFO] LLM initialized in MOCK mode.")

    def generate_response(self, user_text: str, context: str = "", history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response given user text and optional medical context.
        """
        if self.mock:
            return self._mock_response(user_text, context)
            
        # Prompt formatting for TinyLlama Chat
        # <|system|>...</s>
        # <|user|>...</s>
        # <|assistant|>...</s>
        
        system_prompt = "You are a helpful and empathetic AI medical assistant. You provide advice based on the analysis of medical images and user symptoms."
        
        # Build history text
        history_text = ""
        if history:
            for turn in history:
                role = turn['role']
                content = turn['content']
                if role == 'user':
                    history_text += f"<|user|>\n{content}</s>\n"
                elif role == 'assistant':
                    history_text += f"<|assistant|>\n{content}</s>\n"

        full_prompt = f"<|system|>\n{system_prompt}</s>\n"
        
        # Add context if provided (as first user message or separate block)
        if context:
             full_prompt += f"<|user|>\nContext: {context}</s>\n"
             
        full_prompt += history_text
        full_prompt += f"<|user|>\nUser Question: {user_text}</s>\n<|assistant|>\n"
        
        try:
            # Shift to more deterministic output to prevent "my my my" loops on low-resource hardware
            outputs = self.pipe(
                full_prompt, 
                do_sample=False, # Use greedy decoding for stability
                num_beams=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            generated_text = outputs[0]['generated_text']
            # Extract only the NEW assistant part
            response = generated_text.split("<|assistant|>\n")[-1].strip()
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
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

class GeminiResponder:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini Responder (Free Tier).
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            print("[WARN] GEMINI_API_KEY NOT FOUND in environment. Falling back to mock or local LLM.")
            self.model = None
        else:
            print(f"[INFO] Initializing Gemini API: {model_name}...")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction="You are a professional medical AI assistant. Provide concise, accurate, and empathetic medical advice. Always suggest clinical follow-up for serious findings."
            )

    def generate_response(self, user_text: str, context: str = "", history: List[Dict[str, str]] = None) -> str:
        if not self.model:
            return "Gemini API is not configured. Please add GEMINI_API_KEY to your .env file."

        # Convert history format for Gemini
        chat_session = self.model.start_chat(history=[])
        
        # We'll build the prompt with context
        prompt_with_context = f"Medical Context: {context}\n\nUser Question: {user_text}"
        
        try:
            # If history exists, we should ideally populate the chat_session
            # But for simplicity and to avoid token bloat in free tier, we'll pass recent history in prompt or as turns
            if history:
                 # Minimalist history integration
                 history_str = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
                 prompt_with_context = f"Previous Conversation:\n{history_str}\n\n{prompt_with_context}"

            response = self.model.generate_content(prompt_with_context)
            return response.text.strip()
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {e}")
            return "I am currently having trouble connecting to the cloud medical brain. Please try again or consult a professional."

import torch
from typing import Dict
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLMReporter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        mock: bool = False
    ):
        self.mock = mock
        self.model = None
        self.processor = None

        if self.mock:
            print("[INFO] VLM initialized in MOCK mode.")
            return

        try:
            print(f"[INFO] Loading Qwen2-VL: {model_id}")
            
            # Qwen2-VL benefits from flash_attention_2 if available, but we'll stick to auto/default for stability
            # We use float16 for GPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # If device_map not used (cpu), move manually
            if self.device == "cpu":
                self.model.to("cpu")

            self.processor = AutoProcessor.from_pretrained(model_id)
            
            print("[INFO] Qwen2-VL loaded successfully")

        except Exception as e:
            print(f"[WARN] VLM failed to load ({e}) → MOCK mode")
            self.mock = True

    def generate_report(self, image_path: str, classification_result: Dict) -> str:
        if self.mock:
            return self._mock_report(classification_result)

        try:
            # Prepare Image
            # Qwen2-VL handles images via the processor/messages format
            cls = classification_result.get("class", "Unknown")
            conf = classification_result.get("confidence", 0.0) * 100

            prompt_text = (
                f"The image detected as '{cls}' with {conf:.1f}% confidence. "
                "Describe the visual features in the image that support this diagnosis "
                "and explain what they might indicate physically. "
                "Then suggest immediate next clinical steps."
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # Process Inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(self.model.device)

            # Generate
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
            
            # Trim inputs from output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            print(f"[ERROR] VLM Generation Error: {e}")
            return f"Error generating report: {str(e)}"

    def _mock_report(self, classification_result: Dict) -> str:
        cls = classification_result.get("class", "Unknown")
        return (
            f"[MOCK REPORT]\n"
            f"Image findings are consistent with {cls}. "
            "Further clinical correlation is advised."
        )
