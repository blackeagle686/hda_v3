from typing import List, Optional
from .image_model import HDAImgClassifier
from .llm_pipeline import UnifiedQwen
import os

class HDAPipeline:
    def __init__(self, model_path: str = None, mock_llm: bool = False, use_gemini: bool = False, rag_system=None):
        # use_gemini is deprecated in v2 but kept for signature compatibility
        self.classifier = HDAImgClassifier(model_path=model_path)
        
        # Unified Model (Vision + Chat)
        self.qwen = UnifiedQwen(model_id="Qwen/Qwen2-VL-7B-Instruct", mock=mock_llm)
        self.rag = rag_system
        
    def analyze_image(self, image_path: str, user_question: str = ""):
        # 1. Classification (CNN)
        classification = self.classifier.predict(image_path)
        
        # 2. RAG Context (Search based on classification + user question)
        rag_context = ""
        if self.rag:
            try:
                search_query = f"{classification.get('class', '')} {user_question}".strip()
                docs = self.rag.search(search_query)
                rag_context = self.rag.format_context(docs)
                print(f"[INFO] Image Analysis RAG Context found: {len(docs)} docs")
            except Exception as e:
                print(f"[WARN] RAG search during image analysis failed: {e}")

        # 3. VLM Report (Qwen)
        # Returns { "response": text, "summary": text }
        report_data = self.qwen.generate_report(image_path, classification, user_question, rag_context)

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
