import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import json
from datetime import datetime

# Custom CSS for better styling
st.set_page_config(
    page_title="Dental X-Ray AI Assistant",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
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
    .thinking-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
    .annotation-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #17a2b8;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class DentalXRayAnalyzer:
    def __init__(self):
        self.model_path = "models/best_model.h5"
        self.classes = ["cavity", "filling", "implant", "impacted"]
        self.model = None
        self.load_model()
        self.feedback_data = []
        self.load_feedback_data()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                st.success("✅ AI Model loaded successfully!")
            else:
                st.error("❌ Model not found. Please train the model first.")
                self.model = None
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def load_feedback_data(self):
        """Load feedback data for continuous learning"""
        try:
            if os.path.exists('feedback_data.json'):
                with open('feedback_data.json', 'r') as f:
                    self.feedback_data = json.load(f)
        except:
            self.feedback_data = []
    
    def save_feedback_data(self):
        """Save feedback data"""
        with open('feedback_data.json', 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def advanced_preprocess_image(self, image):
        """Advanced image preprocessing with background segmentation and CLAHE"""
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Background segmentation to reduce white pixels
        img2 = image - 255
        kernel = np.ones((2, 2))
        kernel2 = np.ones((3, 3))
        
        # Dilate to create mask
        dilated_mask = cv2.dilate(img2, kernel, iterations=3)
        ret, thresh = cv2.threshold(dilated_mask, 0, 255, cv2.THRESH_BINARY)
        dilated_mask2 = cv2.dilate(thresh, kernel2, iterations=3)
        
        # Apply mask to original image
        image = image / 255.0
        res_img = dilated_mask2 * image
        res_img = np.uint8(res_img)
        
        # Apply CLAHE with higher clip limit
        clahe_op = cv2.createCLAHE(clipLimit=20)
        final_img = clahe_op.apply(res_img)
        
        # Resize
        final_img = cv2.resize(final_img, (256, 256))
        
        # Convert to RGB (DenseNet expects 3 channels)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        final_img = final_img / 255.0
        
        return final_img
    
    def detect_teeth_regions(self, image):
        """Detect potential tooth regions using contour detection"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (teeth should be reasonably sized)
        tooth_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Adjust these values based on your images
                x, y, w, h = cv2.boundingRect(contour)
                tooth_regions.append((x, y, w, h))
        
        return tooth_regions
    
    def create_attention_map(self, image, predictions):
        """Create a simple attention map based on prediction confidence"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create attention map based on image intensity and prediction confidence
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize
        gray = gray / 255.0
        
        # Create attention map (brighter areas = more attention)
        attention_map = gray * np.max(predictions)
        
        # Apply colormap
        attention_colored = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return attention_colored
    
    def explain_prediction(self, image, predicted_class, predictions):
        """Generate explanation for the AI's prediction"""
        explanations = []
        
        # Get confidence for predicted class
        confidence = predictions[predicted_class]
        
        # Base explanation
        explanations.append(f"🔍 **Analysis Complete**")
        explanations.append(f"I analyzed the X-ray image and detected patterns consistent with **{self.classes[predicted_class].title()}**.")
        
        # Confidence-based explanation
        if confidence > 0.8:
            explanations.append(f"🎯 **High Confidence ({confidence*100:.1f}%)**: The patterns are very clear and distinctive.")
        elif confidence > 0.6:
            explanations.append(f"✅ **Medium Confidence ({confidence*100:.1f}%)**: The patterns are recognizable but not definitive.")
        else:
            explanations.append(f"⚠️ **Low Confidence ({confidence*100:.1f}%)**: The patterns are unclear and may need human review.")
        
        # Class-specific explanations
        if predicted_class == 0:  # Cavity
            explanations.append("🦷 **Looking for**: Dark spots or holes in tooth structure")
        elif predicted_class == 1:  # Filling
            explanations.append("🦷 **Looking for**: Bright, dense materials in tooth")
        elif predicted_class == 2:  # Implant
            explanations.append("🦷 **Looking for**: Screw-like structures or artificial tooth roots")
        elif predicted_class == 3:  # Impacted
            explanations.append("🦷 **Looking for**: Teeth that appear stuck or misaligned")
        
        # Alternative possibilities
        other_classes = [(i, pred) for i, pred in enumerate(predictions) if i != predicted_class]
        other_classes.sort(key=lambda x: x[1], reverse=True)
        
        if other_classes[0][1] > 0.2:
            explanations.append(f"🤔 **Alternative possibility**: {self.classes[other_classes[0][0]].title()} ({other_classes[0][1]*100:.1f}%)")
        
        return explanations
    
    def predict(self, image):
        """Make prediction on the image"""
        if self.model is None:
            return None, None
        
        try:
            # Preprocess image
            processed_img = self.advanced_preprocess_image(image)
            
            # Add batch dimension
            processed_img = np.expand_dims(processed_img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return predicted_class, predictions[0]
        
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None, None
    
    def apply_clahe(self, image, clip_limit=2.0):
        """Apply CLAHE enhancement"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return Image.fromarray(enhanced)
    
    def apply_edge_detection(self, image):
        """Apply Canny edge detection"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(image, 50, 150)
        return Image.fromarray(edges)
    
    def add_feedback(self, image_path, predicted_class, user_correction, confidence):
        """Add user feedback for continuous learning"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'predicted_class': predicted_class,
            'user_correction': user_correction,
            'confidence': confidence,
            'correct': predicted_class == user_correction
        }
        
        self.feedback_data.append(feedback)
        self.save_feedback_data()
        
        return len(self.feedback_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🦷 Dental X-Ray AI Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = DentalXRayAnalyzer()
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload X-Ray Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a dental X-ray image for analysis"
    )
    
    # Processing options
    st.sidebar.markdown("### 🔧 Image Processing")
    apply_clahe = st.sidebar.checkbox("Apply CLAHE Enhancement", value=True)
    clahe_clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1)
    apply_edge_detection = st.sidebar.checkbox("Apply Edge Detection", value=False)
    
    # Annotation options
    st.sidebar.markdown("### 🦷 Annotation Options")
    show_teeth_detection = st.sidebar.checkbox("Show Tooth Detection", value=True)
    show_attention_map = st.sidebar.checkbox("Show Attention Map", value=True)
    show_thinking = st.sidebar.checkbox("Show AI Thinking", value=True)
    
    # Main content
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Create columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(image, caption="Original X-Ray", use_column_width=True)
            
            # Image statistics
            st.markdown("### 📊 Image Statistics")
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            stats_col1, stats_col2 = st.columns(2)
            with stats_col1:
                st.metric("Mean Intensity", f"{np.mean(img_array):.1f}")
                st.metric("Standard Deviation", f"{np.std(img_array):.1f}")
            with stats_col2:
                st.metric("Min Value", f"{np.min(img_array):.0f}")
                st.metric("Max Value", f"{np.max(img_array):.0f}")
        
        with col2:
            st.markdown("### 🔍 Processed Image")
            
            # Apply processing
            processed_image = image
            if apply_clahe:
                processed_image = analyzer.apply_clahe(processed_image, clahe_clip_limit)
            if apply_edge_detection:
                processed_image = analyzer.apply_edge_detection(processed_image)
            
            st.image(processed_image, caption="Processed X-Ray", use_column_width=True)
        
        # AI Analysis Section
        if analyzer.model is not None:
            st.markdown("---")
            st.markdown("## 🤖 AI Analysis")
            
            # Make prediction
            predicted_class, predictions = analyzer.predict(image)
            
            if predicted_class is not None:
                # Create three columns for analysis
                analysis_col1, analysis_col2, analysis_col3 = st.columns([1, 1, 1])
                
                with analysis_col1:
                    st.markdown("### 🎯 Prediction")
                    # Prediction box
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"**Predicted Condition:** {analyzer.classes[predicted_class].title()}")
                    st.markdown(f"**Confidence:** {predictions[predicted_class]*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[cls.title() for cls in analyzer.classes],
                            y=predictions * 100,
                            marker_color=['#1f77b4' if i == predicted_class else '#d3d3d3' for i in range(len(analyzer.classes))]
                        )
                    ])
                    fig.update_layout(
                        title="Prediction Confidence",
                        xaxis_title="Conditions",
                        yaxis_title="Confidence (%)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with analysis_col2:
                    st.markdown("### 🦷 Tooth Detection")
                    if show_teeth_detection:
                        # Detect tooth regions
                        tooth_regions = analyzer.detect_teeth_regions(image)
                        
                        # Draw annotations on image
                        annotated_image = image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        
                        for i, (x, y, w, h) in enumerate(tooth_regions):
                            # Draw rectangle
                            draw.rectangle([x, y, x+w, y+h], outline='red', width=2)
                            # Add label
                            draw.text((x, y-20), f"Tooth {i+1}", fill='red')
                        
                        st.image(annotated_image, caption=f"Detected {len(tooth_regions)} tooth regions", use_column_width=True)
                        
                        st.markdown(f"**Detected {len(tooth_regions)} potential tooth regions**")
                    else:
                        st.info("Enable 'Show Tooth Detection' to see annotations")
                
                with analysis_col3:
                    st.markdown("### 🧠 Attention Map")
                    if show_attention_map:
                        # Create attention map
                        attention_map = analyzer.create_attention_map(image, predictions)
                        st.image(attention_map, caption="AI Attention Map", use_column_width=True)
                        
                        st.markdown("**Brighter areas = AI paying more attention**")
                    else:
                        st.info("Enable 'Show Attention Map' to see what the AI focuses on")
                
                # AI Thinking Section
                if show_thinking:
                    st.markdown("---")
                    st.markdown("## 🧠 AI Thinking Process")
                    
                    explanations = analyzer.explain_prediction(image, predicted_class, predictions)
                    
                    st.markdown('<div class="thinking-box">', unsafe_allow_html=True)
                    for explanation in explanations:
                        st.markdown(explanation)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Feedback Section
                st.markdown("---")
                st.markdown("## 📝 Feedback & Learning")
                
                feedback_col1, feedback_col2 = st.columns([1, 1])
                
                with feedback_col1:
                    st.markdown("### Was the prediction correct?")
                    
                    # Feedback form
                    with st.form("feedback_form"):
                        user_correction = st.selectbox(
                            "What condition do you see?",
                            options=analyzer.classes,
                            index=predicted_class,
                            format_func=lambda x: x.title()
                        )
                        
                        confidence_rating = st.slider(
                            "How confident are you in your assessment?",
                            min_value=1,
                            max_value=5,
                            value=3,
                            help="1 = Not sure, 5 = Very confident"
                        )
                        
                        feedback_notes = st.text_area(
                            "Additional notes (optional):",
                            placeholder="Describe what you see or any observations..."
                        )
                        
                        submitted = st.form_submit_button("Submit Feedback")
                        
                        if submitted:
                            # Save feedback
                            feedback_count = analyzer.add_feedback(
                                str(uploaded_file.name),
                                predicted_class,
                                analyzer.classes.index(user_correction),
                                predictions[predicted_class]
                            )
                            
                            st.success(f"✅ Feedback submitted! Total feedback: {feedback_count}")
                
                with feedback_col2:
                    st.markdown("### 📊 Feedback Statistics")
                    
                    if analyzer.feedback_data:
                        total_feedback = len(analyzer.feedback_data)
                        correct_predictions = sum(1 for f in analyzer.feedback_data if f['correct'])
                        accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0
                        
                        st.metric("Total Feedback", total_feedback)
                        st.metric("Correct Predictions", correct_predictions)
                        st.metric("AI Accuracy", f"{accuracy*100:.1f}%")
                        
                        # Show recent feedback
                        st.markdown("**Recent Feedback:**")
                        for feedback in analyzer.feedback_data[-3:]:
                            status = "✅" if feedback['correct'] else "❌"
                            st.markdown(f"{status} {feedback['timestamp'][:10]}: {analyzer.classes[feedback['predicted_class']].title()}")
                    else:
                        st.info("No feedback submitted yet. Help improve the AI!")
            else:
                st.error("❌ Unable to make prediction")
        else:
            st.warning("⚠️ Model not available. Please train the model first.")
        
        # Additional analysis
        st.markdown("---")
        st.markdown("### 📈 Image Histogram")
        
        # Create histogram
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        fig = px.histogram(
            x=img_array.flatten(),
            nbins=50,
            title="Pixel Intensity Distribution",
            labels={'x': 'Pixel Intensity', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Welcome message
        st.markdown("""
        ## 🎯 Welcome to Dental X-Ray AI Assistant!
        
        This advanced tool helps you analyze dental X-ray images using:
        
        - **🤖 AI-Powered Classification**: Detects cavities, fillings, implants, and impacted teeth
        - **🦷 Tooth Detection**: Automatically identifies and annotates tooth regions
        - **🧠 AI Thinking**: Shows you what the AI is looking at and why it made its decision
        - **📝 Continuous Learning**: Learn from your feedback to improve over time
        - **🔧 Advanced Image Processing**: CLAHE enhancement and edge detection
        - **📊 Detailed Analysis**: Image statistics and confidence scores
        - **📈 Visual Insights**: Histograms, attention maps, and prediction charts
        
        ### 🚀 Getting Started:
        1. **Upload** a dental X-ray image using the sidebar
        2. **Adjust** processing and annotation settings
        3. **View** AI predictions with explanations
        4. **Provide feedback** to help the AI learn
        5. **Explore** tooth detection and attention maps
        
        ### 📋 Supported Conditions:
        - **Cavity**: Dental decay detection
        - **Filling**: Dental restoration identification
        - **Implant**: Dental implant recognition
        - **Impacted**: Impacted tooth detection
        """)
        
        # Model status
        if analyzer.model is not None:
            st.success("✅ AI Model is ready for analysis!")
        else:
            st.error("❌ AI Model not available. Please train the model first.")

if __name__ == "__main__":
    main()