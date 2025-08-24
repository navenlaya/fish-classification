#!/usr/bin/env python3
"""
Demo Model Script for Fish Species Classification

This script creates a simple untrained model for testing the application
without requiring a full training dataset.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def create_demo_model(save_path: str = "model.pt"):
    """
    Create a demo model with random weights for testing.
    
    Args:
        save_path: Path to save the demo model
    """
    print("Creating demo model for testing...")
    
    # Create ResNet18 model
    model = models.resnet18(pretrained=False)
    
    # Modify the final layer for 3 classes
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    # Initialize weights randomly
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    
    print(f"Demo model saved to: {save_path}")
    print("Note: This model has random weights and will give random predictions.")
    print("Use it only for testing the application structure.")

if __name__ == "__main__":
    create_demo_model()
