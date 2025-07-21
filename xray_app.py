import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🦷 Dental X-Ray Viewer")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Original Image")
    st.image(img, channels="GRAY", clamp=True)

    # Resize for consistent processing
    resized = cv2.resize(img, (1000, 400))  # Resize to a consistent, viewable size

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    edges = cv2.Canny(enhanced, 30, 100)

    # Show results
    st.subheader("Processed Images")
    st.image(enhanced, caption="CLAHE Enhanced", width=300)