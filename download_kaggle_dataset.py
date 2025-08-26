#!/usr/bin/env python3
"""
Download Fish Species Dataset from Kaggle
Using kagglehub for easy dataset access
"""

import kagglehub
import os
import shutil
from pathlib import Path

def download_fish_dataset():
    """Download the large scale fish dataset from Kaggle"""
    
    print("🐟 Downloading Large Scale Fish Dataset from Kaggle...")
    print("=" * 60)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset: crowww/a-large-scale-fish-dataset")
        print("⏳ This may take a few minutes depending on your internet speed...")
        
        path = kagglehub.dataset_download("crowww/a-large-scale-fish-dataset")
        
        print(f"✅ Download complete!")
        print(f"📁 Dataset downloaded to: {path}")
        
        # Show what we got
        print(f"\n📋 Dataset contents:")
        if os.path.exists(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Count files in subdirectories
                    try:
                        files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        print(f"    📸 {len(files)} image files")
                    except:
                        print(f"    📸 (counting files...)")
                else:
                    print(f"  📄 {item}")
        
        # Ask if user wants to organize it
        print(f"\n🎯 Next steps:")
        print(f"  1. Check the downloaded dataset structure")
        print(f"  2. Organize images into train/val folders")
        print(f"  3. Train your model")
        
        return path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"💡 Make sure you have proper Kaggle access")
        return None

if __name__ == "__main__":
    download_fish_dataset()
