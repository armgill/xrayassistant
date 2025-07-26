#!/usr/bin/env python3
"""
Organize Roboflow Dataset for Classification
This script converts the object detection dataset into classification format.
"""

import pandas as pd
import shutil
from pathlib import Path
import os

def organize_dataset():
    """Organize the dataset into classification folders"""
    
    print("🦷 Organizing Dental X-Ray Dataset for Classification")
    print("=" * 60)
    
    # Create output directories
    output_dir = Path("data")
    classes = ["cavity", "filling", "implant", "impacted"]
    
    for class_name in classes:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {class_dir}")
    
    # Process each split (train, valid, test)
    splits = ["train", "valid", "test"]
    
    for split in splits:
        print(f"\n📊 Processing {split} split...")
        
        split_dir = Path(f"dataset/{split}")
        if not split_dir.exists():
            print(f"⚠️  {split} directory not found, skipping...")
            continue
        
        # Read annotations
        annotations_file = split_dir / "_annotations.csv"
        if not annotations_file.exists():
            print(f"⚠️  No annotations file found in {split}, skipping...")
            continue
        
        # Read CSV
        df = pd.read_csv(annotations_file)
        print(f"📋 Found {len(df)} annotations in {split}")
        
        # Process each annotation
        processed_images = set()  # Track processed images to avoid duplicates
        
        for _, row in df.iterrows():
            filename = row['filename']
            class_name = row['class'].lower()
            
            # Map class names
            if class_name == "fillings":
                class_name = "filling"
            elif class_name == "impacted tooth":
                class_name = "impacted"
            
            # Skip if class not in our list
            if class_name not in classes:
                print(f"⚠️  Unknown class: {row['class']}, skipping...")
                continue
            
            # Source and destination paths
            src_path = split_dir / filename
            dst_path = output_dir / class_name / f"{split}_{filename}"
            
            # Copy image if it exists and hasn't been processed
            if src_path.exists() and filename not in processed_images:
                try:
                    shutil.copy2(src_path, dst_path)
                    processed_images.add(filename)
                    print(f"✅ Copied {filename} to {class_name}/")
                except Exception as e:
                    print(f"❌ Error copying {filename}: {e}")
            elif filename in processed_images:
                print(f"⏭️  Skipped {filename} (already processed)")
            else:
                print(f"❌ Image not found: {filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Dataset Organization Complete!")
    print("\n📁 Organized images by class:")
    
    for class_name in classes:
        class_dir = output_dir / class_name
        if class_dir.exists():
            image_count = len(list(class_dir.glob("*.jpg")))
            print(f"  {class_name:10s}: {image_count:4d} images")
    
    print(f"\n💡 Your dataset is now ready for training!")
    print(f"📁 Check the 'data/' folder to see the organized images")
    print(f"🚀 Run 'python train_model.py' to start training")

def create_sample_dataset():
    """Create a smaller sample dataset for testing"""
    
    print("\n🧪 Creating Sample Dataset (for testing)")
    print("=" * 40)
    
    # Create sample directories
    sample_dir = Path("data_sample")
    classes = ["cavity", "filling", "implant", "impacted"]
    
    for class_name in classes:
        class_dir = sample_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy a few images from each class
    data_dir = Path("data")
    max_samples = 50  # Max images per class for sample
    
    for class_name in classes:
        src_class_dir = data_dir / class_name
        dst_class_dir = sample_dir / class_name
        
        if src_class_dir.exists():
            images = list(src_class_dir.glob("*.jpg"))[:max_samples]
            
            for i, img_path in enumerate(images):
                dst_path = dst_class_dir / f"sample_{i:03d}.jpg"
                shutil.copy2(img_path, dst_path)
            
            print(f"📸 Copied {len(images)} images to {class_name}/")
    
    print(f"\n✅ Sample dataset created in 'data_sample/'")
    print(f"💡 Use this for quick testing before training on full dataset")

def main():
    """Main function"""
    
    # Check if dataset exists
    if not Path("dataset").exists():
        print("❌ Dataset folder not found!")
        print("Please make sure you have extracted the dataset to the 'dataset/' folder")
        return
    
    # Organize the full dataset
    organize_dataset()
    
    # Ask if user wants a sample dataset
    response = input("\n🤔 Would you like to create a sample dataset for testing? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        create_sample_dataset()
    
    print("\n🎉 Dataset organization complete!")
    print("\nNext steps:")
    print("1. Check the 'data/' folder to verify organization")
    print("2. Run 'python train_model.py' to train your model")
    print("3. Run 'streamlit run xray_app.py' to test the web interface")

if __name__ == "__main__":
    main() 