

# 🦷 Dental X-Ray Assistant

A simple and interactive web app for viewing and enhancing dental X-ray images. Built with Python and Streamlit, this tool allows users to upload grayscale X-rays, apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for improved visibility, and optionally download the processed image.

---

## 🚀 Features

- Upload `.jpg`, `.jpeg`, or `.png` dental X-ray images
- Apply CLAHE enhancement for better contrast
- Resize images for consistent processing
- Download enhanced images for clinical or research use

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) – for UI
- [OpenCV](https://opencv.org/) – for image processing
- [NumPy](https://numpy.org/) – for numerical operations
- [Pillow (PIL)](https://python-pillow.org/) – for image conversion and downloads

---

## 📂 Project Structure

```
xrayassistant/
├── xray_app.py               # Main Streamlit app
├── xray_viewer.py            # (Optional) Matplotlib viewer for debugging
├── requirements.txt          # Installed Python packages
├── .python-version           # Python version tracking
├── env/                      # Virtual environment
├── README.md                 # Project overview and setup
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/xrayassistant.git
   cd xrayassistant
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate   # or `env\Scripts\activate` on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run xray_app.py
   ```

---

## 📸 Screenshot

![example](assets/example_output.png)

---

## 🔬 Future Additions (Planned)
- Zoom/pan functionality
- Annotation or region selection
- ML-based cavity/crown detection
- Upload multiple images for batch viewing

---

## 📄 License

MIT License – feel free to use and adapt for your own dental or medical imaging tools.