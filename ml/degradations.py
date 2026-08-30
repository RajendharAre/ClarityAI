"""
Image Degradation Module
Generates synthetic degradations for dataset creation.
Follows Design Principles: Modularity, Reproducibility, Interpretability
"""

import numpy as np
import cv2
from typing import Tuple, List
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Degradation(ABC):
    """
    Abstract base class for image degradations.
    Follows Design Pattern: Strategy (GoF)
    """

    @abstractmethod
    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply degradation to image.
        
        Args:
            image: Input image
            severity: Degradation severity (0.0-1.0)
        
        Returns:
            Degraded image
        """
        pass

    @abstractmethod
    def get_issue_type(self) -> str:
        """Get issue type name"""
        pass


class BlurDegradation(Degradation):
    """
    Gaussian blur degradation.
    Simulates out-of-focus or motion blur.
    
    Severity 0.0: No blur (kernel size = 3)
    Severity 1.0: Heavy blur (kernel size = 21)
    """

    def get_issue_type(self) -> str:
        return "blur"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply Gaussian blur.
        
        Args:
            image: Input image
            severity: Blur severity (0.0-1.0)
        
        Returns:
            Blurred image
        """
        # Map severity to kernel size
        # severity 0.0 -> kernel 3 (minimal blur)
        # severity 1.0 -> kernel 21 (heavy blur)
        min_kernel = 3
        max_kernel = 21
        kernel_size = int(min_kernel + (max_kernel - min_kernel) * severity)
        
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        logger.debug(f"Applied blur with kernel size {kernel_size}, severity {severity:.2f}")
        return blurred


class ExposureDegradation(Degradation):
    """
    Brightness/exposure degradation.
    Simulates underexposure (too dark) or overexposure (too bright).
    
    Severity < 0.5: Underexposure (darkens image)
    Severity > 0.5: Overexposure (brightens image)
    """

    def get_issue_type(self) -> str:
        return "exposure"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply brightness adjustment.
        
        Args:
            image: Input image
            severity: Exposure severity (0.0-1.0)
                - 0.0: Very dark (multiply by 0.3)
                - 0.5: Normal
                - 1.0: Very bright (multiply by 1.7)
        
        Returns:
            Brightness-adjusted image
        """
        # Map severity to brightness factor
        # 0.0 -> 0.3 (30% brightness)
        # 0.5 -> 1.0 (normal)
        # 1.0 -> 1.7 (170% brightness)
        
        if severity < 0.5:
            # Underexposure: darken
            brightness_factor = 0.3 + (severity * 2) * (1.0 - 0.3)  # 0.3 to 1.0
        else:
            # Overexposure: brighten
            brightness_factor = 1.0 + ((severity - 0.5) * 2) * (1.7 - 1.0)  # 1.0 to 1.7

        # Apply brightness adjustment
        adjusted = cv2.convertScaleAbs(image.astype(float) * brightness_factor)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

        logger.debug(f"Applied exposure with factor {brightness_factor:.2f}, severity {severity:.2f}")
        return adjusted


class NoiseDegradation(Degradation):
    """
    Noise degradation.
    Simulates sensor noise or compression artifacts.
    
    Severity 0.0: No noise (sigma = 5)
    Severity 1.0: Heavy noise (sigma = 50)
    """

    def get_issue_type(self) -> str:
        return "noise"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add Gaussian noise.
        
        Args:
            image: Input image
            severity: Noise severity (0.0-1.0)
        
        Returns:
            Noisy image
        """
        # Map severity to noise sigma
        min_sigma = 5
        max_sigma = 50
        sigma = min_sigma + (max_sigma - min_sigma) * severity

        # Add Gaussian noise
        noise = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(float) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        logger.debug(f"Applied noise with sigma {sigma:.2f}, severity {severity:.2f}")
        return noisy


class SaltPepperNoiseDegradation(Degradation):
    """
    Salt-and-pepper noise degradation.
    Simulates random pixel errors or dead pixels.
    """

    def get_issue_type(self) -> str:
        return "salt_pepper_noise"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add salt-and-pepper noise.
        
        Args:
            image: Input image
            severity: Noise severity (0.0-1.0)
        
        Returns:
            Noisy image
        """
        # Map severity to noise probability
        noise_prob = severity * 0.1  # 0.0 to 10% of pixels affected

        noisy = image.copy().astype(float)
        
        # Add salt (white) noise
        salt_mask = np.random.random(image.shape) < noise_prob / 2
        noisy[salt_mask] = 255

        # Add pepper (black) noise
        pepper_mask = np.random.random(image.shape) < noise_prob / 2
        noisy[pepper_mask] = 0

        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        logger.debug(f"Applied salt-pepper noise with prob {noise_prob:.4f}, severity {severity:.2f}")
        return noisy


class JPEGCompressionDegradation(Degradation):
    """
    JPEG compression artifacts.
    Simulates lossy JPEG compression.
    """

    def get_issue_type(self) -> str:
        return "jpeg_compression"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Apply JPEG compression artifacts.
        
        Args:
            image: Input image
            severity: Compression severity (0.0-1.0)
        
        Returns:
            Compressed image
        """
        # Map severity to JPEG quality
        # 0.0 -> quality 95 (minimal compression)
        # 1.0 -> quality 10 (heavy compression)
        min_quality = 10
        max_quality = 95
        quality = int(max_quality - (max_quality - min_quality) * severity)

        # Encode and decode with JPEG
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            logger.warning("JPEG encoding failed, returning original")
            return image

        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        logger.debug(f"Applied JPEG compression with quality {quality}, severity {severity:.2f}")
        return decoded


class ScratchDegradation(Degradation):
    """
    Scratch/line defects.
    Simulates physical scratches or sensor defects.
    """

    def get_issue_type(self) -> str:
        return "scratch"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add scratch-like defects.
        
        Args:
            image: Input image
            severity: Defect severity (0.0-1.0)
        
        Returns:
            Scratched image
        """
        defected = image.copy()
        height, width = image.shape[:2]

        # Number of scratches based on severity
        num_scratches = int(severity * 5)

        for _ in range(num_scratches):
            # Random line position and angle
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)

            # Random line color and thickness
            color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            thickness = np.random.randint(1, 3)

            # Draw line (scratch)
            cv2.line(defected, (x1, y1), (x2, y2), color, thickness)

        logger.debug(f"Applied scratches, severity {severity:.2f}")
        return defected


class SpotDegradation(Degradation):
    """
    Spot/speckle defects.
    Simulates dust, spots, or blemishes on lens/sensor.
    """

    def get_issue_type(self) -> str:
        return "spot"

    def apply(self, image: np.ndarray, severity: float) -> np.ndarray:
        """
        Add spot-like defects.
        
        Args:
            image: Input image
            severity: Defect severity (0.0-1.0)
        
        Returns:
            Spotted image
        """
        defected = image.copy()
        height, width = image.shape[:2]

        # Number of spots based on severity
        num_spots = int(severity * 20)

        for _ in range(num_spots):
            # Random spot position and size
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            radius = np.random.randint(2, 15)

            # Random spot color (often dark)
            color = tuple(np.random.randint(0, 100, 3).tolist())

            # Draw circle (spot)
            cv2.circle(defected, (x, y), radius, color, -1)

        logger.debug(f"Applied spots, severity {severity:.2f}")
        return defected


class DegradationFactory:
    """
    Factory for creating degradation objects.
    Follows Design Pattern: Factory (GoF)
    """

    _degradations = {
        "blur": BlurDegradation,
        "exposure": ExposureDegradation,
        "noise": NoiseDegradation,
        "salt_pepper_noise": SaltPepperNoiseDegradation,
        "jpeg_compression": JPEGCompressionDegradation,
        "scratch": ScratchDegradation,
        "spot": SpotDegradation,
    }

    @classmethod
    def create(cls, degradation_type: str) -> Degradation:
        """
        Create degradation object.
        
        Args:
            degradation_type: Type of degradation
        
        Returns:
            Degradation object
        
        Raises:
            ValueError: If degradation type not supported
        """
        if degradation_type not in cls._degradations:
            raise ValueError(
                f"Unknown degradation type: {degradation_type}. "
                f"Supported types: {list(cls._degradations.keys())}"
            )
        return cls._degradations[degradation_type]()

    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all supported degradation types"""
        return list(cls._degradations.keys())


class DegradationPipeline:
    """
    Applies multiple degradations to an image.
    Follows Design Pattern: Decorator (GoF)
    """

    def __init__(self):
        """Initialize degradation pipeline"""
        self.degradations: List[Tuple[str, float]] = []

    def add_degradation(self, degradation_type: str, severity: float) -> "DegradationPipeline":
        """
        Add degradation to pipeline.
        
        Args:
            degradation_type: Type of degradation
            severity: Severity (0.0-1.0)
        
        Returns:
            Self for chaining
        """
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"Severity must be between 0.0 and 1.0, got {severity}")
        
        self.degradations.append((degradation_type, severity))
        return self

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all degradations in sequence.
        
        Args:
            image: Input image
        
        Returns:
            Degraded image
        """
        result = image.copy()
        
        for degradation_type, severity in self.degradations:
            degradation = DegradationFactory.create(degradation_type)
            result = degradation.apply(result, severity)

        return result

    def clear(self) -> None:
        """Clear all degradations"""
        self.degradations.clear()

    def __repr__(self) -> str:
        return f"DegradationPipeline({self.degradations})"
