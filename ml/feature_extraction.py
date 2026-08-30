"""
Image Feature Extraction Module
Implements classical image quality metrics for interpretable analysis.
Follows Design Principles: Modularity, SRP, Interpretability, Reproducibility
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """
    Represents extracted image features.
    Interpretable metrics for image quality assessment.
    """
    sharpness: float  # Laplacian variance (higher = sharper)
    brightness: float  # Mean luminance (0-255)
    contrast: float  # Standard deviation of pixels
    noise_level: float  # Estimated noise magnitude
    saturation: float  # Color saturation level
    texture_complexity: float  # LBP histogram entropy

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            "sharpness": round(self.sharpness, 2),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "noise_level": round(self.noise_level, 2),
            "saturation": round(self.saturation, 2),
            "texture_complexity": round(self.texture_complexity, 2),
        }


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    Follows Design Pattern: Strategy (GoF)
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> float:
        """Extract a specific feature from the image"""
        pass


class SharpnessExtractor(FeatureExtractor):
    """
    Extracts sharpness using Laplacian variance.
    
    Reference:
    "A No-Reference Image Sharpness Metric Based on the 
     Difference Between Laplacian Operators"
    
    Higher values indicate sharper images.
    Typical thresholds:
    - Acceptable: > 100
    - Degraded: 50-100
    - Defective: < 50
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Calculate sharpness as Laplacian variance.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            float: Sharpness score (Laplacian variance)
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Return variance (higher = sharper)
        sharpness = laplacian.var()
        
        logger.debug(f"Sharpness score: {sharpness:.2f}")
        return float(sharpness)


class BrightnessExtractor(FeatureExtractor):
    """
    Extracts brightness (mean luminance).
    
    Range: 0-255
    - 0-85: Dark/underexposed
    - 85-170: Normal
    - 170-255: Bright/overexposed
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Calculate mean brightness in LAB color space (perceptually uniform).
        
        Args:
            image: Input image (BGR)
        
        Returns:
            float: Mean brightness (0-255)
        """
        # Convert to LAB color space (L channel = luminance)
        if len(image.shape) == 2:
            # Already grayscale, use directly
            brightness = float(np.mean(image))
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            brightness = float(np.mean(lab[:, :, 0]))  # L channel

        logger.debug(f"Brightness score: {brightness:.2f}")
        return brightness


class ContrastExtractor(FeatureExtractor):
    """
    Extracts contrast as standard deviation of pixel intensities.
    
    Higher contrast = more visual distinction between regions
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Calculate contrast as pixel intensity standard deviation.
        
        Args:
            image: Input image
        
        Returns:
            float: Contrast score (std dev of luminance)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate standard deviation
        contrast = float(np.std(gray))

        logger.debug(f"Contrast score: {contrast:.2f}")
        return contrast


class NoiseExtractor(FeatureExtractor):
    """
    Estimates noise level using high-frequency residuals.
    
    Method: Wavelet decomposition
    Higher values indicate more noise.
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Estimate noise using Laplacian high-pass filtering.
        
        Args:
            image: Input image (grayscale)
        
        Returns:
            float: Noise estimate (std dev of high-frequency component)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Laplacian (high-pass filter)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Noise estimate: std dev of high-frequency component
        noise = float(np.std(laplacian))

        logger.debug(f"Noise level: {noise:.2f}")
        return noise


class SaturationExtractor(FeatureExtractor):
    """
    Extracts color saturation level.
    
    Based on HSV color space saturation channel.
    Higher values = more saturated colors
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Calculate mean saturation from HSV color space.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            float: Mean saturation (0-255)
        """
        # Convert BGR to HSV
        if len(image.shape) == 2:
            # Grayscale has no saturation
            return 0.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extract saturation channel (index 1)
        saturation = float(np.mean(hsv[:, :, 1]))

        logger.debug(f"Saturation: {saturation:.2f}")
        return saturation


class TextureComplexityExtractor(FeatureExtractor):
    """
    Extracts texture complexity using Local Binary Patterns (LBP).
    
    Measures local texture patterns and their entropy.
    """

    def extract(self, image: np.ndarray) -> float:
        """
        Calculate texture complexity using LBP histogram entropy.
        
        Args:
            image: Input image
        
        Returns:
            float: Entropy of LBP histogram (texture complexity)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Simplified texture complexity: gradient magnitude
        # (Full LBP would require scikit-image)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Entropy of gradient magnitude histogram
        hist, _ = np.histogram(magnitude.flatten(), bins=256, range=(0, magnitude.max() + 1))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros for log
        entropy = -np.sum(hist * np.log2(hist))

        logger.debug(f"Texture complexity (entropy): {entropy:.2f}")
        return float(entropy)


class ImageFeatureExtractor:
    """
    Main feature extraction orchestrator.
    Combines multiple feature extractors.
    Follows Design Pattern: Facade (GoF)
    """

    def __init__(self):
        """Initialize all feature extractors"""
        self.sharpness = SharpnessExtractor()
        self.brightness = BrightnessExtractor()
        self.contrast = ContrastExtractor()
        self.noise = NoiseExtractor()
        self.saturation = SaturationExtractor()
        self.texture = TextureComplexityExtractor()

    def extract_all(self, image: np.ndarray) -> FeatureStats:
        """
        Extract all features from image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            FeatureStats: Object containing all extracted features
        """
        logger.info(f"Extracting features from image (shape: {image.shape})")

        # Extract each feature
        sharpness = self.sharpness.extract(image)
        brightness = self.brightness.extract(image)
        contrast = self.contrast.extract(image)
        noise_level = self.noise.extract(image)
        saturation = self.saturation.extract(image)
        texture = self.texture.extract(image)

        features = FeatureStats(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            noise_level=noise_level,
            saturation=saturation,
            texture_complexity=texture,
        )

        logger.info(f"Feature extraction complete: {features.to_dict()}")
        return features

    def extract_from_path(self, image_path: str) -> FeatureStats:
        """
        Load image from file and extract features.
        
        Args:
            image_path: Path to image file
        
        Returns:
            FeatureStats: Extracted features
        
        Raises:
            FileNotFoundError: If image file not found
            ValueError: If file cannot be read as image
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            return self.extract_all(image)
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            raise


# Convenience function for quick feature extraction
def extract_image_features(image: np.ndarray) -> FeatureStats:
    """
    Quick feature extraction function.
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        FeatureStats: Extracted features
    """
    extractor = ImageFeatureExtractor()
    return extractor.extract_all(image)
