import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

class XRayViewer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.classes = ["cavity", "crown", "filling", "normal"]
        self.current_image = None
        self.current_path = None
        
    def load_image(self, image_path):
        """Load and preprocess an X-ray image"""
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            print("Error: Image failed to load")
            return None
            
        self.current_image = img
        self.current_path = image_path
        return img
    
    def apply_clahe(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply CLAHE for contrast enhancement"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    
    def apply_gaussian_blur(self, img, kernel_size=(5, 5)):
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(img, kernel_size, 0)
    
    def detect_edges(self, img, low_threshold=50, high_threshold=150):
        """Detect edges using Canny algorithm"""
        return cv2.Canny(img, low_threshold, high_threshold)
    
    def apply_morphological_operations(self, img, operation='open', kernel_size=3):
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if operation == 'open':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(img, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(img, kernel, iterations=1)
        return img
    
    def find_contours(self, img):
        """Find contours in the image"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def display_processing_pipeline(self, img, title="X-Ray Processing Pipeline"):
        """Display a comprehensive processing pipeline"""
        # Resize for consistent display
        display_size = (800, 600)
        img_resized = cv2.resize(img, display_size)
        
        # Apply different processing techniques
        clahe_enhanced = self.apply_clahe(img_resized)
        blurred = self.apply_gaussian_blur(clahe_enhanced)
        edges = self.detect_edges(blurred)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)
        
        # Original
        axes[0, 0].imshow(img_resized, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # CLAHE Enhanced
        axes[0, 1].imshow(clahe_enhanced, cmap='gray')
        axes[0, 1].set_title('CLAHE Enhanced')
        axes[0, 1].axis('off')
        
        # Blurred
        axes[0, 2].imshow(blurred, cmap='gray')
        axes[0, 2].set_title('Gaussian Blur')
        axes[0, 2].axis('off')
        
        # Edge Detection
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')
        
        # Histogram
        axes[1, 1].hist(img_resized.ravel(), bins=256, range=[0, 256], alpha=0.7)
        axes[1, 1].set_title('Histogram')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        
        # Contours
        contours = self.find_contours(edges)
        contour_img = np.zeros_like(edges)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
        axes[1, 2].imshow(contour_img, cmap='gray')
        axes[1, 2].set_title(f'Contours ({len(contours)} found)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def browse_dataset(self, max_images_per_class=3):
        """Browse through the dataset showing samples from each class"""
        fig, axes = plt.subplots(len(self.classes), max_images_per_class, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16)
        
        for i, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            image_files = image_files[:max_images_per_class]
            
            for j, img_path in enumerate(image_files):
                img = self.load_image(str(img_path))
                if img is not None:
                    img_resized = cv2.resize(img, (200, 200))
                    axes[i, j].imshow(img_resized, cmap='gray')
                    axes[i, j].set_title(f'{class_name.title()}')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].text(0.5, 0.5, 'No Image', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    viewer = XRayViewer()
    
    # Browse the dataset
    print("Browsing dataset...")
    viewer.browse_dataset()
    
    # Process a specific image (replace with actual path)
    # img_path = "data/normal/sample.jpg"
    # if os.path.exists(img_path):
    #     img = viewer.load_image(img_path)
    #     viewer.display_processing_pipeline(img, "Dental X-Ray Analysis")
    # else:
    #     print(f"Image not found: {img_path}")
    #     print("Please provide a valid image path to test the processing pipeline.")
