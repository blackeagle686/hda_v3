import torch
from typing import Dict, List
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from transformers import BitsAndBytesConfig

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
            
            if self.device == "cuda":
                # 4-bit Quantization Config to save VRAM (15GB -> ~6GB)
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                print("[INFO] Using 4-bit quantization for VRAM optimization.")
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    attn_implementation="eager"
                )
            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    device_map=None
                )
            
            # Set image resolution limits to prevent VRAM spikes
            # min_pixels = 256*256, max_pixels = 1280*28*28 (standard for medicine usually)
            self.processor = AutoProcessor.from_pretrained(
                model_id, 
                min_pixels=256*28*28, 
                max_pixels=512*28*28 
            )
            print("[INFO] Unified Qwen loaded successfully (Optimized).")

        except Exception as e:
            print(f"[WARN] Failed to load Qwen ({e}). Switching to MOCK mode.")
            self.mock = True

    def generate_report(self, image_path: str, classification_result: Dict, user_question: str = "") -> Dict[str, str]:
        """
        Generates a medical report + summary for a given image, taking an optional user question into account.
        """
        if self.mock:
            return self._mock_report(classification_result)

        cls = classification_result.get("class", "Unknown")
        conf = classification_result.get("confidence", 0.0) * 100

        if user_question:
            prompt_text = (
                f"The image was classified as '{cls}' with {conf:.1f}% confidence. "
                f"The user has a specific question: '{user_question}'.\n\n"
                "Please provide a **lengthy and extremely detailed clinical analysis** of this image. "
                "1. **Visual Findings**: Describe specific medical features visible in the image that support the classification.\n"
                "2. **Detailed Answer**: Provide a comprehensive and professional answer to the user's question based on clinical knowledge.\n"
                "3. **Clinical Context**: Explain the implications of these findings for the patient.\n"
                "4. **Recommendations**: List detailed next steps, tests, and professional medical advice.\n\n"
                "Use well-structured markdown for the report."
            )
        else:
            prompt_text = (
                f"The image was classified as '{cls}' with {conf:.1f}% confidence.\n\n"
                "Please generate a **comprehensive, structured medical report**. "
                "Include a detailed description of the radiological features, characteristic patterns of the identified disease, "
                "and a thorough clinical recommendation section. The report should be professional and informative for healthcare providers."
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
            "You are HDA, an expert medical AI assistant. Using the provided context and history, "
            "provide accurate, professional, and helpful medical guidance.\n\n"
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
        response_text = self._run_inference(messages, max_tokens=4096)

        # 4. Generate Short Summary for Memory (Safety wrapped to prevent response failure)
        try:
            summary = self._generate_summary(response_text)
        except Exception as e:
            print(f"[WARN] Summary generation failed: {e}")
            summary = "Context updated."

        return {
            "response": response_text,
            "summary": summary
        }

    def _run_inference(self, messages, max_tokens=4096):
        """Helper to run model inference with stable decoding."""
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

        # Stable generation settings
        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            do_sample=False, # Use greedy decoding for stability (prevents probability tensor errors)
            num_beams=1,
            repetition_penalty=1.1, # Prevent loops without extreme changes to distribution
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
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
