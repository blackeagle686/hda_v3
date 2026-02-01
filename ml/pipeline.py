from .image_model import HDAImgClassifier
from .llm_pipeline import LLMResponder
import os

class HDAPipeline:
    def __init__(self, model_path: str = None, mock_llm: bool = True):
        # Initialize sub-components
        # Note: We default mock_llm to True to prevent heavy downloads on local dev env
        self.classifier = HDAImgClassifier(model_path=model_path)
        self.llm = LLMResponder(mock=mock_llm)
        
    def analyze_image(self, image_path: str, heatmap_output_path: str):
        """
        Run full analysis on an image: Prediction + Heatmap + Initial Advice
        """
        # 1. Prediction
        result = self.classifier.predict(image_path)
        
        # 2. Heatmap
        # Only generate heatmap if confidence is high or specific mock logic needed. 
        # But we act always for demo.
        heatmap_path = self.classifier.generate_heatmap(image_path, heatmap_output_path)
        
        # 3. Initial LLM Context
        diagnosis_context = f"The patient's scan shows signs of {result['class']} with {result['confidence']*100:.1f}% confidence."
        
        # 4. Generate initial advice
        initial_advice = self.llm.generate_response(
            user_text="Please explain this diagnosis and what I should do next.",
            context=diagnosis_context
        )
        
        return {
            "classification": result,
            "heatmap_path": heatmap_path,
            "advice": initial_advice
        }

    def chat(self, message: str, context: str = ""):
        """
        Pure text chat.
        """
        return self.llm.generate_response(message, context)
