import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class ImageProcessor:
    """Utility class for X-ray image processing"""
    
    @staticmethod
    def load_image(image_path: str, grayscale: bool = True) -> Optional[np.ndarray]:
        """Load an image from path"""
        if grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        return img
    
    @staticmethod
    def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(img, target_size)
    
    @staticmethod
    def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    
    @staticmethod
    def apply_gaussian_blur(img: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(img, kernel_size, 0)
    
    @staticmethod
    def detect_edges(img: np.ndarray, low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """Detect edges using Canny algorithm"""
        return cv2.Canny(img, low_threshold, high_threshold)
    
    @staticmethod
    def apply_morphological_operation(img: np.ndarray, operation: str = 'open', 
                                    kernel_size: int = 3) -> np.ndarray:
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
        else:
            return img
    
    @staticmethod
    def find_contours(img: np.ndarray) -> List[np.ndarray]:
        """Find contours in binary image"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return img.astype(np.float32) / 255.0
    
    @staticmethod
    def enhance_contrast(img: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """Enhance contrast using linear transformation"""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    @staticmethod
    def remove_noise(img: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Remove noise using different methods"""
        if method == 'bilateral':
            return cv2.bilateralFilter(img, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(img, 5)
        elif method == 'gaussian':
            return cv2.GaussianBlur(img, (5, 5), 0)
        else:
            return img

class VisualizationUtils:
    """Utility class for image visualization"""
    
    @staticmethod
    def plot_image_comparison(images: List[np.ndarray], titles: List[str], 
                            figsize: Tuple[int, int] = (15, 5)) -> None:
        """Plot multiple images side by side"""
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
        
        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_histogram(img: np.ndarray, title: str = "Image Histogram") -> None:
        """Plot image histogram"""
        plt.figure(figsize=(10, 6))
        plt.hist(img.ravel(), bins=256, range=[0, 256], alpha=0.7)
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_processing_pipeline(img: np.ndarray, title: str = "Processing Pipeline") -> None:
        """Display a comprehensive processing pipeline"""
        # Create processed versions
        clahe_enhanced = ImageProcessor.apply_clahe(img)
        blurred = ImageProcessor.apply_gaussian_blur(clahe_enhanced)
        edges = ImageProcessor.detect_edges(blurred)
        
        # Create visualization
        images = [img, clahe_enhanced, blurred, edges]
        titles = ['Original', 'CLAHE Enhanced', 'Gaussian Blur', 'Edge Detection']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        for i, (img_processed, title_text) in enumerate(zip(images, titles)):
            row, col = i // 2, i % 2
            axes[row, col].imshow(img_processed, cmap='gray')
            axes[row, col].set_title(title_text)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_contours(img: np.ndarray, contours: List[np.ndarray], 
                     title: str = "Detected Contours") -> None:
        """Plot image with detected contours"""
        contour_img = np.zeros_like(img)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(contour_img, cmap='gray')
        plt.title(f"{title} ({len(contours)} contours found)")
        plt.axis('off')
        plt.show()

class DatasetUtils:
    """Utility class for dataset operations"""
    
    @staticmethod
    def get_dataset_stats(data_dir: str) -> dict:
        """Get statistics about the dataset"""
        data_path = Path(data_dir)
        stats = {}
        
        if not data_path.exists():
            return stats
        
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                stats[class_name] = len(image_files)
        
        return stats
    
    @staticmethod
    def validate_dataset(data_dir: str, expected_classes: List[str]) -> dict:
        """Validate dataset structure and return issues"""
        data_path = Path(data_dir)
        issues = []
        
        if not data_path.exists():
            issues.append(f"Data directory {data_dir} does not exist")
            return issues
        
        for class_name in expected_classes:
            class_path = data_path / class_name
            if not class_path.exists():
                issues.append(f"Class directory {class_name} does not exist")
            else:
                image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
                if len(image_files) == 0:
                    issues.append(f"No images found in {class_name} directory")
        
        return issues
    
    @staticmethod
    def load_sample_images(data_dir: str, max_per_class: int = 3) -> dict:
        """Load sample images from each class"""
        data_path = Path(data_dir)
        samples = {}
        
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                image_files = image_files[:max_per_class]
                
                samples[class_name] = []
                for img_path in image_files:
                    img = ImageProcessor.load_image(str(img_path))
                    if img is not None:
                        samples[class_name].append(img)
        
        return samples 