#!/usr/bin/env python3
"""
Dataset Preparation Script for Fish Species Classification

This script helps organize fish images into the correct directory structure
for training the model.
"""

import os
import shutil
import argparse
from pathlib import Path
import random

def create_dataset_structure(base_dir: str):
    """
    Create the required dataset directory structure.
    
    Args:
        base_dir: Base directory for the dataset
    """
    # Define the species classes
    species = ['rainbow_trout', 'largemouth_bass', 'chinook_salmon']
    
    # Create main directories
    for split in ['train', 'val']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create species subdirectories
        for species_name in species:
            species_dir = os.path.join(split_dir, species_name)
            os.makedirs(species_dir, exist_ok=True)
    
    print(f"Created dataset structure in {base_dir}")
    print("Directory structure:")
    print(f"  {base_dir}/")
    print("  ├── train/")
    for species_name in species:
        print(f"  │   └── {species_name}/")
    print("  └── val/")
    for species_name in species:
        print(f"  │   └── {species_name}/")

def organize_images(source_dir: str, dataset_dir: str, train_split: float = 0.8):
    """
    Organize images from source directory into train/val splits.
    
    Args:
        source_dir: Directory containing species subdirectories
        dataset_dir: Target dataset directory
        train_split: Fraction of images to use for training (default: 0.8)
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Get species directories
    species_dirs = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d))]
    
    if not species_dirs:
        print(f"Error: No species subdirectories found in '{source_dir}'")
        print("Expected structure:")
        print("  source_dir/")
        print("  ├── rainbow_trout/")
        print("  ├── largemouth_bass/")
        print("  └── chinook_salmon/")
        return
    
    print(f"Found species directories: {species_dirs}")
    
    total_images = 0
    for species in species_dirs:
        source_species_dir = os.path.join(source_dir, species)
        train_species_dir = os.path.join(dataset_dir, 'train', species)
        val_species_dir = os.path.join(dataset_dir, 'val', species)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in os.listdir(source_species_dir) 
                      if Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"Warning: No images found in {species}")
            continue
        
        # Shuffle and split
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_split)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Copy training images
        for img_file in train_files:
            src = os.path.join(source_species_dir, img_file)
            dst = os.path.join(train_species_dir, img_file)
            shutil.copy2(src, dst)
        
        # Copy validation images
        for img_file in val_files:
            src = os.path.join(source_species_dir, img_file)
            dst = os.path.join(val_species_dir, img_file)
            shutil.copy2(src, dst)
        
        print(f"{species}: {len(train_files)} train, {len(val_files)} val")
        total_images += len(image_files)
    
    print(f"\nTotal images organized: {total_images}")
    print(f"Dataset ready in: {dataset_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare Fish Species Dataset')
    parser.add_argument('--action', choices=['create', 'organize'], required=True,
                       help='Action to perform: create (structure) or organize (images)')
    parser.add_argument('--dataset_dir', type=str, default='../dataset',
                       help='Path to dataset directory (default: ../dataset)')
    parser.add_argument('--source_dir', type=str,
                       help='Source directory containing species subdirectories (for organize action)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of images for training (default: 0.8)')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        create_dataset_structure(args.dataset_dir)
    elif args.action == 'organize':
        if not args.source_dir:
            print("Error: --source_dir is required for organize action")
            return
        organize_images(args.source_dir, args.dataset_dir, args.train_split)

if __name__ == "__main__":
    main()
