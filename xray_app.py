import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🦷 Dental X-Ray Viewer")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("Original Image")
    st.image(img, channels="GRAY", clamp=True, width=800)

    # resize for consistent processing (800px wide, maintain aspect ratio)
    resized = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    edges = cv2.Canny(enhanced, 30, 100)

    # show
    st.subheader("Processed Images")
    st.image(enhanced, caption="CLAHE Enhanced", width=800)