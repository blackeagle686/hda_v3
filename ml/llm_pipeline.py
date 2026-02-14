import torch
from typing import Dict, List
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

class UnifiedQwen:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        mock: bool = False
    ):
        """
        Unified Model for HDA v2: Handles both Vision (X-ray analysis) and Text Chat (RAG).
        """
        self.mock = mock
        self.model = None
        self.processor = None

        if self.mock:
            print("[INFO] UnifiedQwen initialized in MOCK mode.")
            return

        try:
            print(f"[INFO] Loading Unified Qwen Model: {model_id}...")
            
            # Auto-detect device configuration
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                attn_implementation="eager" if self.device == "cuda" else "eager"
            )
            
            if self.device == "cpu":
                self.model.to("cpu")

            self.processor = AutoProcessor.from_pretrained(model_id)
            print("[INFO] Unified Qwen loaded successfully.")

        except Exception as e:
            print(f"[WARN] Failed to load Qwen ({e}). Switching to MOCK mode.")
            self.mock = True

    def generate_report(self, image_path: str, classification_result: Dict) -> Dict[str, str]:
        """
        Generates a medical report + summary for a given image.
        """
        if self.mock:
            return self._mock_report(classification_result)

        cls = classification_result.get("class", "Unknown")
        conf = classification_result.get("confidence", 0.0) * 100

        prompt_text = (
            f"The image was classified as '{cls}' with {conf:.1f}% confidence. "
            "Analyze the image features visible that support this diagnosis. "
            "Provide clinical advice and next steps."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        response_text = self._run_inference(messages)
        summary = self._generate_summary(response_text)

        return {
            "response": response_text,
            "summary": summary
        }

    def chat_with_rag(self, user_text: str, rag_context: str, context_summaries: List[str]) -> Dict[str, str]:
        """
        Chat functionality with RAG and Memory Summaries.
        
        Args:
            user_text: The user's current question.
            rag_context: Retrieved documents from Vector DB.
            context_summaries: List of short summaries from previous turns.
        """
        if self.mock:
            return self._mock_chat(user_text)

        # 1. Build System Context from Summaries
        memory_block = ""
        if context_summaries:
            memory_block = "Previous Context:\n" + "\n".join([f"- {s}" for s in context_summaries]) + "\n\n"

        # 2. Build Prompt
        rag_block = f"Reference Medical Knowledge:\n{rag_context}\n\n" if rag_context else ""
        
        full_prompt = (
            f"You are HDA, an expert medical AI. Use the context below to answer.\n\n"
            f"{memory_block}"
            f"{rag_block}"
            f"User Question: {user_text}"
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": full_prompt}]
            }
        ]

        # 3. Generate Main Response
        response_text = self._run_inference(messages)

        # 4. Generate Short Summary for Memory
        summary = self._generate_summary(response_text)

        return {
            "response": response_text,
            "summary": summary
        }

    def _run_inference(self, messages, max_tokens=1024):
        """Helper to run model inference."""
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

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def _generate_summary(self, text_to_summarize: str) -> str:
        """
        Generates a <10 token summary of the provided text.
        """
        prompt = (
            f"Summarize the following medical advice in less than 10 words for memory context:\n"
            f"'{text_to_summarize}'\n"
            f"Summary:"
        )
        
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        
        try:
            summary = self._run_inference(messages, max_tokens=20) # Keep max tokens low
            return summary.strip()
        except:
            return "Medical advice given."

    def _mock_report(self, classification_result):
        return {
            "response": f"[MOCK] Qwen 7B Report for {classification_result.get('class')}. Image shows specific features...",
            "summary": f"Diagnosed {classification_result.get('class')}."
        }

    def _mock_chat(self, user_text):
        return {
            "response": f"[MOCK] Qwen 7B Response to '{user_text}'. RAG context was used.",
            "summary": "Answered user query."
        }
