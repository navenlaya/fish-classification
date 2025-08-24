#!/usr/bin/env python3
"""
Fish Species Classification Training Script

This script fine-tunes a ResNet18 model on a dataset of fish images.
The dataset should be organized in subdirectories, one per species.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import time

def get_data_transforms():
    """
    Get data transformations for training and validation.
    
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_model(num_classes):
    """
    Create a ResNet18 model with modified final layer.
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        Modified ResNet18 model
    """
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the final few layers
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Modify the final layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (epoch_loss, epoch_acc)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (epoch_loss, epoch_acc)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plot training history.
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Fish Species Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_save_path', type=str, default='model.pt', help='Path to save trained model')
    parser.add_argument('--plot_save_path', type=str, default='training_history.png', help='Path to save training plot')
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' does not exist.")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 50)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_save_path)
            print(f"New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs, args.plot_save_path)
    print(f"Training history plot saved to: {args.plot_save_path}")
    
    # Final evaluation on validation set
    print("\nFinal evaluation on validation set...")
    model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final metrics
    final_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Final validation accuracy: {final_accuracy*100:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=val_dataset.classes))
    
    print(f"\nModel saved to: {args.model_save_path}")

if __name__ == "__main__":
    main()
