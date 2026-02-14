from typing import List, Optional
from .image_model import HDAImgClassifier
from .llm_pipeline import UnifiedQwen
import os

class HDAPipeline:
    def __init__(self, model_path: str = None, mock_llm: bool = True, use_gemini: bool = False):
        # use_gemini is deprecated in v2 but kept for signature compatibility
        self.classifier = HDAImgClassifier(model_path=model_path)
        
        # Unified Model (Vision + Chat)
        self.qwen = UnifiedQwen(model_id="Qwen/Qwen2-VL-7B-Instruct", mock=mock_llm)
        
    def analyze_image(self, image_path: str):
        # 1. Classification (CNN)
        classification = self.classifier.predict(image_path)
        
        # 2. VLM Report (Qwen)
        # Returns { "response": text, "summary": text }
        report_data = self.qwen.generate_report(image_path, classification)

        return {
            "classification": classification,
            "advice": report_data["response"],
            "summary": report_data["summary"]
        }

    def chat(self, message: str, context_summaries: Optional[List[str]] = None, rag_context: str = ""):
        """
        RAG Chat with Qwen.
        """
        # Returns { "response": text, "summary": text }
        return self.qwen.chat_with_rag(message, rag_context, context_summaries or [])
