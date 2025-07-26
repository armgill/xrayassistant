#!/usr/bin/env python3
"""
Demo script for the Dental X-Ray Assistant
This script demonstrates the main features of the project.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.image_utils import ImageProcessor, VisualizationUtils, DatasetUtils
import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo_image_processing():
    """Demonstrate image processing capabilities"""
    print("🔬 Image Processing Demo")
    print("=" * 40)
    
    # Check if we have sample images
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ No data directory found. Please add some X-ray images to the data/ folder.")
        return
    
    # Find a sample image
    sample_image = None
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if images:
                sample_image = str(images[0])
                break
    
    if not sample_image:
        print("❌ No sample images found in data/ directory.")
        return
    
    print(f"📸 Processing sample image: {sample_image}")
    
    # Load and process image
    img = ImageProcessor.load_image(sample_image)
    if img is None:
        print("❌ Failed to load image")
        return
    
    # Apply different processing techniques
    clahe_enhanced = ImageProcessor.apply_clahe(img)
    blurred = ImageProcessor.apply_gaussian_blur(clahe_enhanced)
    edges = ImageProcessor.detect_edges(blurred)
    
    # Display results
    images = [img, clahe_enhanced, blurred, edges]
    titles = ['Original', 'CLAHE Enhanced', 'Gaussian Blur', 'Edge Detection']
    
    VisualizationUtils.plot_image_comparison(images, titles, figsize=(16, 4))
    
    # Show histogram
    VisualizationUtils.plot_histogram(img, "Original Image Histogram")
    
    print("✅ Image processing demo completed!")

def demo_dataset_analysis():
    """Demonstrate dataset analysis capabilities"""
    print("\n📊 Dataset Analysis Demo")
    print("=" * 40)
    
    data_dir = "data"
    
    # Get dataset statistics
    stats = DatasetUtils.get_dataset_stats(data_dir)
    
    if not stats:
        print("❌ No dataset found or dataset is empty.")
        return
    
    print("📈 Dataset Statistics:")
    total_images = sum(stats.values())
    for class_name, count in stats.items():
        percentage = (count / total_images) * 100
        print(f"  {class_name:10s}: {count:3d} images ({percentage:5.1f}%)")
    
    print(f"\n📊 Total images: {total_images}")
    
    # Validate dataset
    expected_classes = ["cavity", "crown", "filling", "normal"]
    issues = DatasetUtils.validate_dataset(data_dir, expected_classes)
    
    if issues:
        print("\n⚠️  Dataset Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Dataset validation passed!")
    
    # Load sample images
    samples = DatasetUtils.load_sample_images(data_dir, max_per_class=2)
    
    if samples:
        print(f"\n🖼️  Sample images loaded from {len(samples)} classes")
        
        # Display sample images
        all_images = []
        all_titles = []
        
        for class_name, images in samples.items():
            for i, img in enumerate(images[:2]):  # Show max 2 per class
                all_images.append(img)
                all_titles.append(f"{class_name.title()} {i+1}")
        
        if all_images:
            VisualizationUtils.plot_image_comparison(all_images, all_titles, figsize=(16, 8))
    
    print("✅ Dataset analysis demo completed!")

def demo_ml_prediction():
    """Demonstrate ML prediction capabilities"""
    print("\n🤖 Machine Learning Demo")
    print("=" * 40)
    
    model_path = Path("models/dental_model.h5")
    
    if not model_path.exists():
        print("❌ No trained model found. Please run train_model.py first.")
        print("   This will train a model and save it to models/dental_model.h5")
        return
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path))
        print("✅ Model loaded successfully!")
        
        # Load model info if available
        info_path = Path("models/model_info.json")
        if info_path.exists():
            import json
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"📊 Model trained on {model_info['total_samples']} images")
            print(f"🎯 Final accuracy: {model_info['final_accuracy']:.3f}")
        
        # Find a test image
        data_dir = Path("data")
        test_image = None
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if images:
                    test_image = str(images[0])
                    break
        
        if test_image:
            print(f"\n🔍 Testing prediction on: {test_image}")
            
            # Preprocess image for prediction
            img = ImageProcessor.load_image(test_image)
            if img is not None:
                img_resized = ImageProcessor.resize_image(img, (256, 256))
                img_normalized = ImageProcessor.normalize_image(img_resized)
                img_input = np.expand_dims(img_normalized, axis=[0, -1])
                
                # Make prediction
                prediction = model.predict(img_input)
                class_names = ["cavity", "crown", "filling", "normal"]
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                print(f"🎯 Prediction: {predicted_class.title()}")
                print(f"📊 Confidence: {confidence:.1f}%")
                
                # Show all class probabilities
                print("\n📈 All Class Probabilities:")
                for i, (class_name, prob) in enumerate(zip(class_names, prediction[0])):
                    print(f"  {class_name:10s}: {prob*100:5.1f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    print("✅ Machine learning demo completed!")

def demo_web_interface():
    """Provide instructions for running the web interface"""
    print("\n🌐 Web Interface Demo")
    print("=" * 40)
    
    print("To run the web interface:")
    print("1. Open a terminal in the project directory")
    print("2. Activate your virtual environment:")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("3. Run the Streamlit app:")
    print("   streamlit run xray_app.py")
    print("4. Open your browser to http://localhost:8501")
    print("\n✨ The web interface provides:")
    print("  - Interactive image upload")
    print("  - Real-time image processing")
    print("  - ML predictions with confidence scores")
    print("  - Comprehensive visualizations")

def main():
    """Main demo function"""
    print("🦷 Dental X-Ray Assistant - Demo")
    print("=" * 50)
    print("This demo showcases the main features of the project.\n")
    
    # Run demos
    demo_image_processing()
    demo_dataset_analysis()
    demo_ml_prediction()
    demo_web_interface()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Add your X-ray images to the data/ directory")
    print("2. Run 'python train_model.py' to train the ML model")
    print("3. Run 'streamlit run xray_app.py' to start the web interface")
    print("4. Explore the code and customize for your needs")

if __name__ == "__main__":
    main() 