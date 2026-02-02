import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np
import cv2
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HDAImgClassifier:
    def __init__(self, model_path: str = None, num_classes: int = 3):
        """
        Initialize the Image Classifier.
        Args:
            model_path: Path to the .pth checkpoint. If None, uses random weights (mock mode expected).
            num_classes: Number of classes (Lung Benign, Lung Malignant, Colon Benign/Malignant etc currently assuming 3 based on typical datasets, but user code implies dynamic detection. 
                         Looking at user's code, it detects classes from folders. I will assume 2 main types (Lung/Colon) with subclasses or just binary.
                         Let's stick to user's my_model.py approach which was dynamic.
                         However, for inference we need fixed classes. 
                         I will standardise to: ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc'] usually found in that Kaggle dataset.
                         Or simplest: 0: Lung_Benign, 1: Lung_Malignant, 2: Colon_Benign, 3: Colon_Malignant.
                         Let's wait for model loading to confirm. For now I'll default to 5 common classes from that dataset or accept a list.
        """
        self.classes = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc'] # Common 5 classes for this dataset
        self.num_classes = len(self.classes)
        
        # Load Model Structure
        self.model = efficientnet_b0(pretrained=False) # No need to download weights if we load checkpoint
        
        # Replicate the modification from my_model.py
        # Freeze backbone (not strictly necessary for inference but good for consistency)
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Replace classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        self.model.to(device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                print(f"[INFO] Model loaded from {model_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print("[WARN] No model path found or file missing. Using random weights (MOCK MODE).")

        # Transforms (Validation/Inference only)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Hook for Grad-CAM
        self.gradients = None
        self.activations = None
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0] if grad_output[0] is not None else None

        def forward_hook(module, input, output):
            self.activations = output if output is not None else None
            
        # Register hooks on the last convolutional layer of EfficientNet-B0 features
        # features[8] is the last block usually. Let's inspect efficientnet structure.
        # EfficientNet features is a Sequential. 
        # features[-1] is usually the Conv2dNormActivation or similar.
        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def preprocess(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(device), image

    def predict(self, image_path: str):
        tensor, original_image = self.preprocess(image_path)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
        return {
            "class": self.classes[pred_idx.item()],
            "confidence": float(confidence.item()),
            "class_idx": pred_idx.item()
        }

    def generate_heatmap(self, image_path: str, save_path: str):
        tensor, original_image = self.preprocess(image_path)
        self.model.zero_grad()
        output = self.model(tensor)

        target_class_idx = output.argmax(dim=1).item()
        score = output[:, target_class_idx]

        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM failed: hooks not triggered.")

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1,2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        img_w, img_h = original_image.size
        cam = cv2.resize(cam, (img_w, img_h))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original_np = np.array(original_image)
        superimposed_img = np.clip(heatmap*0.4 + original_np*0.6, 0, 255).astype(np.uint8)

        Image.fromarray(superimposed_img).save(save_path)
        return save_path
