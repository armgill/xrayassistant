import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🦷 Dental X-Ray Assistant",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class DentalXRayAnalyzer:
    def __init__(self):
        self.classes = ["cavity", "crown", "filling", "normal"]
        self.model_path = "models/dental_model.h5"
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return False
    
    def preprocess_image(self, img, target_size=(256, 256)):
        """Preprocess image for model prediction"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalize
        img = img / 255.0
        
        # Add channel dimension
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def predict(self, img):
        """Make prediction on the image"""
        if self.model is None:
            return None, None
        
        # Preprocess
        processed_img = self.preprocess_image(img)
        
        # Predict
        prediction = self.model.predict(np.expand_dims(processed_img, axis=0))
        
        # Get class and confidence
        predicted_class = self.classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return predicted_class, confidence
    
    def apply_clahe(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply CLAHE for contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    
    def detect_edges(self, img, low_threshold=50, high_threshold=150):
        """Detect edges using Canny algorithm"""
        return cv2.Canny(img, low_threshold, high_threshold)
    
    def find_contours(self, img):
        """Find contours in the image"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

def main():
    # Header
    st.markdown('<h1 class="main-header">🦷 Dental X-Ray Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = DentalXRayAnalyzer()
    model_loaded = analyzer.load_model()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model status
    if model_loaded:
        st.sidebar.success("✅ ML Model Loaded")
    else:
        st.sidebar.warning("⚠️ ML Model Not Available")
        st.sidebar.info("Train the model first using train_model.py")
    
    # Processing options
    st.sidebar.subheader("Image Processing")
    apply_clahe = st.sidebar.checkbox("Apply CLAHE Enhancement", value=True)
    clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
    
    detect_edges = st.sidebar.checkbox("Detect Edges", value=False)
    edge_low = st.sidebar.slider("Edge Low Threshold", 10, 100, 50)
    edge_high = st.sidebar.slider("Edge High Threshold", 100, 300, 150)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a dental X-ray image for analysis"
        )
    
    with col2:
        st.subheader("📊 Dataset Info")
        data_dir = Path("data")
        if data_dir.exists():
            for class_name in analyzer.classes:
                class_dir = data_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
                    st.metric(f"{class_name.title()}", count)
        else:
            st.info("No data directory found")
    
    if uploaded_file is not None:
        # Convert uploaded image to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Display original image
        st.subheader("🖼️ Original Image")
        st.image(img, caption="Uploaded X-Ray", use_column_width=True)
        
        # Image processing
        processed_img = img.copy()
        
        if apply_clahe:
            processed_img = analyzer.apply_clahe(processed_img, clahe_clip_limit)
        
        # Create columns for processed images
        col1, col2 = st.columns(2)
        
        with col1:
            if apply_clahe:
                st.subheader("✨ Enhanced Image (CLAHE)")
                st.image(processed_img, caption="CLAHE Enhanced", use_column_width=True)
        
        with col2:
            if detect_edges:
                edges = analyzer.detect_edges(processed_img, edge_low, edge_high)
                st.subheader("🔍 Edge Detection")
                st.image(edges, caption="Canny Edge Detection", use_column_width=True)
        
        # ML Prediction
        if model_loaded:
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("Analyzing image..."):
                predicted_class, confidence = analyzer.predict(img)
            
            if predicted_class:
                # Create prediction display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.metric("Predicted Condition", predicted_class.title())
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Color-coded confidence bar
                    if confidence > 80:
                        st.success("High Confidence")
                    elif confidence > 60:
                        st.warning("Medium Confidence")
                    else:
                        st.error("Low Confidence")
                
                # Detailed predictions
                st.subheader("📈 Detailed Predictions")
                processed_img_for_pred = analyzer.preprocess_image(img)
                predictions = analyzer.model.predict(np.expand_dims(processed_img_for_pred, axis=0))[0]
                
                # Create a bar chart
                import plotly.express as px
                import pandas as pd
                
                df = pd.DataFrame({
                    'Condition': [c.title() for c in analyzer.classes],
                    'Probability': predictions * 100
                })
                
                fig = px.bar(df, x='Condition', y='Probability', 
                           title="Prediction Probabilities",
                           color='Probability',
                           color_continuous_scale='Blues')
                fig.update_layout(yaxis_title="Probability (%)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Image statistics
        st.subheader("📊 Image Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Width", f"{img.shape[1]} px")
        with col2:
            st.metric("Height", f"{img.shape[0]} px")
        with col3:
            st.metric("Mean Intensity", f"{np.mean(img):.1f}")
        with col4:
            st.metric("Std Deviation", f"{np.std(img):.1f}")
        
        # Histogram
        st.subheader("📈 Intensity Histogram")
        hist_values = np.histogram(img, bins=256, range=(0, 256))[0]
        st.bar_chart(hist_values)
    
    else:
        # Welcome message
        st.info("👆 Please upload an X-ray image to begin analysis")
        
        # Show sample images if available
        st.subheader("📁 Sample Images")
        data_dir = Path("data")
        if data_dir.exists():
            sample_images = []
            for class_name in analyzer.classes:
                class_dir = data_dir / class_name
                if class_dir.exists():
                    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                    if images:
                        sample_images.append((class_name, str(images[0])))
            
            if sample_images:
                cols = st.columns(len(sample_images))
                for i, (class_name, img_path) in enumerate(sample_images):
                    with cols[i]:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img_resized = cv2.resize(img, (200, 200))
                            st.image(img_resized, caption=f"{class_name.title()}", use_column_width=True)

if __name__ == "__main__":
    main()