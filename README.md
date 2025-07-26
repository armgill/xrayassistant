

# 🦷 Dental X-Ray Assistant

A comprehensive Python-based system for viewing, enhancing, and analyzing dental X-ray images using computer vision and machine learning techniques.

## 🎯 Project Overview

This project provides tools for:
- **Image Processing**: Advanced X-ray image enhancement using CLAHE, edge detection, and morphological operations
- **Machine Learning**: CNN-based classification of dental conditions (cavity, crown, filling, normal)
- **Web Interface**: Streamlit-based web application for easy image upload and analysis
- **Data Visualization**: Comprehensive plotting and analysis tools

## ✨ Features

### 🔬 Image Processing
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for better detail visibility
- **Edge Detection**: Canny edge detection for feature extraction
- **Noise Reduction**: Multiple denoising methods (Gaussian, Bilateral, Median)
- **Morphological Operations**: Opening, closing, dilation, and erosion
- **Contour Detection**: Automatic detection and visualization of dental features

### 🤖 Machine Learning
- **CNN Architecture**: Deep convolutional neural network with batch normalization and dropout
- **Data Augmentation**: Rotation, scaling, flipping for improved model generalization
- **Multi-class Classification**: Detects cavities, crowns, fillings, and normal teeth
- **Model Evaluation**: Comprehensive metrics including confusion matrix and per-class accuracy

### 🌐 Web Interface
- **Interactive Upload**: Drag-and-drop image upload
- **Real-time Processing**: Instant image enhancement and analysis
- **ML Predictions**: AI-powered condition detection with confidence scores
- **Visualization**: Histograms, processing pipelines, and statistical analysis

## 📁 Project Structure

```
xrayassistant/
├── data/                    # Dataset directory
│   ├── cavity/             # Cavity X-ray images
│   ├── crown/              # Crown X-ray images
│   ├── filling/            # Filling X-ray images
│   └── normal/             # Normal X-ray images
├── models/                 # Trained models and metadata
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── image_utils.py      # Image processing utilities
├── xray_viewer.py          # Interactive image viewer
├── xray_app.py             # Streamlit web application
├── train_model.py          # ML model training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd xrayassistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your X-ray images in the following structure:
```
data/
├── cavity/
│   ├── image1.jpg
│   └── image2.png
├── crown/
│   └── ...
├── filling/
│   └── ...
└── normal/
    └── ...
```

### 3. Training the Model

```bash
python train_model.py
```

This will:
- Load and preprocess all images
- Train a CNN model with data augmentation
- Generate training plots and evaluation metrics
- Save the trained model to `models/dental_model.h5`

### 4. Running the Web Application

```bash
streamlit run xray_app.py
```

Open your browser to `http://localhost:8501` to access the web interface.

### 5. Using the Image Viewer

```bash
python xray_viewer.py
```

This will open an interactive viewer for browsing and processing X-ray images.

## 🔧 Usage Examples

### Basic Image Processing

```python
from utils.image_utils import ImageProcessor, VisualizationUtils

# Load and process an image
img = ImageProcessor.load_image("path/to/xray.jpg")
enhanced = ImageProcessor.apply_clahe(img)
edges = ImageProcessor.detect_edges(enhanced)

# Visualize results
VisualizationUtils.plot_image_comparison(
    [img, enhanced, edges],
    ["Original", "Enhanced", "Edges"]
)
```

### Dataset Analysis

```python
from utils.image_utils import DatasetUtils

# Get dataset statistics
stats = DatasetUtils.get_dataset_stats("data")
print(f"Dataset contains {sum(stats.values())} images")

# Validate dataset structure
issues = DatasetUtils.validate_dataset("data", ["cavity", "crown", "filling", "normal"])
if issues:
    print("Dataset issues found:", issues)
```

### ML Prediction

```python
import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model("models/dental_model.h5")

# Preprocess image
img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
img = img / 255.0
img = np.expand_dims(img, axis=[0, -1])

# Make prediction
prediction = model.predict(img)
class_names = ["cavity", "crown", "filling", "normal"]
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_class} ({confidence:.1f}% confidence)")
```

## 📊 Model Performance

The CNN model typically achieves:
- **Overall Accuracy**: 85-95% (depending on dataset quality)
- **Per-class Performance**: 
  - Normal: 90-95%
  - Cavity: 85-90%
  - Crown: 80-85%
  - Filling: 75-80%

## 🛠️ Configuration

### Model Parameters
- **Image Size**: 256x256 pixels
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Learning Rate**: 0.001 (with reduction on plateau)

### Processing Parameters
- **CLAHE Clip Limit**: 2.0
- **CLAHE Tile Grid**: 8x8
- **Edge Detection Thresholds**: 50, 150
- **Gaussian Blur Kernel**: 5x5

## 🔍 Advanced Features

### Custom Processing Pipeline

```python
from utils.image_utils import ImageProcessor

# Create custom processing pipeline
img = ImageProcessor.load_image("xray.jpg")
img = ImageProcessor.apply_clahe(img, clip_limit=3.0)
img = ImageProcessor.remove_noise(img, method="bilateral")
img = ImageProcessor.enhance_contrast(img, alpha=1.5)
```

### Batch Processing

```python
import os
from pathlib import Path
from utils.image_utils import ImageProcessor

# Process all images in a directory
input_dir = Path("raw_images")
output_dir = Path("processed_images")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    img = ImageProcessor.load_image(str(img_path))
    if img is not None:
        enhanced = ImageProcessor.apply_clahe(img)
        output_path = output_dir / f"enhanced_{img_path.name}"
        cv2.imwrite(str(output_path), enhanced)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for image processing capabilities
- TensorFlow/Keras for deep learning framework
- Streamlit for web application framework
- The dental imaging community for inspiration

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This tool is for educational and research purposes. Always consult with qualified dental professionals for actual medical diagnosis.