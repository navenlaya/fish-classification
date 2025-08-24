import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any
import os

class FishClassifier:
    def __init__(self, model_path: str = "model.pt"):
        """
        Initialize the fish classifier with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.class_names = ["rainbow_trout", "largemouth_bass", "chinook_salmon"]
        
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load the trained model from file.
        
        Args:
            model_path: Path to the trained model file
            
        Returns:
            Loaded PyTorch model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the model architecture
        model = models.resnet18(pretrained=False)
        num_classes = len(self.class_names)
        
        # Modify the final layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """
        Get the image transformation pipeline for preprocessing.
        
        Returns:
            Composition of image transformations
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict the fish species from an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with predicted species and confidence
        """
        with torch.no_grad():
            # Preprocess the image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # Get model prediction
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get the predicted class and confidence
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_species = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            return {
                "species": predicted_species,
                "confidence": confidence_score
            }
    
    def get_class_names(self) -> list:
        """
        Get the list of class names.
        
        Returns:
            List of class names
        """
        return self.class_names.copy()

# Global model instance
_model_instance = None

def get_model(model_path: str = "model.pt") -> FishClassifier:
    """
    Get or create a global model instance.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        FishClassifier instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = FishClassifier(model_path)
    return _model_instance
