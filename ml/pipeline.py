from .image_model import HDAImgClassifier
from .llm_pipeline import LLMResponder, VLMReporter
import os

class HDAPipeline:
    def __init__(self, model_path: str = None, mock_llm: bool = True):
        # Initialize sub-components
        # Note: We default mock_llm to True to prevent heavy downloads on local dev env
        self.classifier = HDAImgClassifier(model_path=model_path)
        self.llm = LLMResponder(mock=mock_llm)
        self.vlm = VLMReporter(mock=mock_llm)
        
    def analyze_image(self, image_path: str):
        classification = self.classifier.predict(image_path)
        # expected: {"class": "Pneumonia", "confidence": 0.92}

        # Generate advice using VLM which sees the image + classification context
        advice = self.vlm.generate_report(image_path, classification)

        return {
            "classification": classification,
            "advice": advice
        }

    def chat(self, message: str, context: str = ""):
        """
        Pure text chat.
        """
        return self.llm.generate_response(message, context)
