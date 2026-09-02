"""
Image Validation Module
Validates image files for corruption, integrity, and compatibility.
Follows Design Principles: Security, Error Handling, SRP
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Custom exception for image validation errors"""
    pass


class ImageValidator:
    """
    Validates image files before processing.
    Detects corruption, checks dimensions, validates format.
    """

    # Allowed MIME types
    ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    # Min/max dimensions (in pixels)
    MIN_DIMENSION = 50
    MAX_DIMENSION = 10000

    # Min/max file size (in bytes)
    MIN_FILE_SIZE = 1024  # 1 KB
    MAX_FILE_SIZE = 52428800  # 50 MB

    @staticmethod
    def validate_file_extension(file_path: str) -> bool:
        """
        Check if file has allowed extension.
        
        Args:
            file_path: Path to file
        
        Returns:
            bool: True if extension is allowed
        
        Raises:
            ImageValidationError: If extension not allowed
        """
        ext = Path(file_path).suffix.lower()
        if ext not in ImageValidator.ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Invalid file extension: {ext}. "
                f"Allowed: {ImageValidator.ALLOWED_EXTENSIONS}"
            )
        return True

    @staticmethod
    def validate_file_size(file_path: str) -> bool:
        """
        Check if file size is within limits.
        
        Args:
            file_path: Path to file
        
        Returns:
            bool: True if size is valid
        
        Raises:
            ImageValidationError: If size exceeds limits
        """
        file_size = Path(file_path).stat().st_size

        if file_size < ImageValidator.MIN_FILE_SIZE:
            raise ImageValidationError(
                f"File too small: {file_size} bytes "
                f"(minimum: {ImageValidator.MIN_FILE_SIZE})"
            )

        if file_size > ImageValidator.MAX_FILE_SIZE:
            raise ImageValidationError(
                f"File too large: {file_size} bytes "
                f"(maximum: {ImageValidator.MAX_FILE_SIZE})"
            )

        return True

    @staticmethod
    def validate_image_dimensions(image: np.ndarray) -> bool:
        """
        Check if image dimensions are valid.
        
        Args:
            image: Image array
        
        Returns:
            bool: True if dimensions are valid
        
        Raises:
            ImageValidationError: If dimensions invalid
        """
        height, width = image.shape[:2]

        if height < ImageValidator.MIN_DIMENSION or width < ImageValidator.MIN_DIMENSION:
            raise ImageValidationError(
                f"Image too small: {width}x{height} "
                f"(minimum: {ImageValidator.MIN_DIMENSION}x{ImageValidator.MIN_DIMENSION})"
            )

        if height > ImageValidator.MAX_DIMENSION or width > ImageValidator.MAX_DIMENSION:
            raise ImageValidationError(
                f"Image too large: {width}x{height} "
                f"(maximum: {ImageValidator.MAX_DIMENSION}x{ImageValidator.MAX_DIMENSION})"
            )

        return True

    @staticmethod
    def validate_image_readability(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if file can be read as valid image.
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Try PIL verification (strict check for corruption)
            with Image.open(file_path) as img:
                img.verify()
            
            # Also try OpenCV loading
            image = cv2.imread(file_path)
            if image is None:
                return False, "OpenCV failed to read image"
            
            return True, None
        except Exception as e:
            return False, f"Image verification failed: {str(e)}"

    @staticmethod
    def detect_corruption(image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Detect potential image corruption/artifacts.
        
        Args:
            image: Image array
        
        Returns:
            Tuple[bool, Optional[str]]: (has_corruption, description)
        """
        # Check for all-zero or all-white frames (total corruption)
        if image.size == 0:
            return True, "Empty image array"

        mean_val = np.mean(image)
        std_val = np.std(image)

        # No variance = corrupted
        if std_val < 1:
            return True, "No pixel variance (possible corruption)"

        # Check for excessive NaN or Inf values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            return True, "Contains NaN or Inf values"

        # Check for extreme compression artifacts
        if image.dtype != np.uint8:
            return True, f"Invalid data type: {image.dtype}"

        return False, None

    @staticmethod
    def validate_complete(file_path: str, image: Optional[np.ndarray] = None) -> Tuple[bool, Optional[str]]:
        """
        Perform complete validation of an image, given a file path or array.

        Args:
            file_path: Path to image file (used when image is None)
            image: Optional pre-loaded image array (avoids re-reading from disk)

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            if image is None:
                # Check file exists
                if not Path(file_path).exists():
                    return False, f"File not found: {file_path}"

                # Validate extension
                ImageValidator.validate_file_extension(file_path)
                logger.debug(f"Extension valid: {file_path}")

                # Validate file size
                ImageValidator.validate_file_size(file_path)
                logger.debug(f"File size valid: {file_path}")

                # Validate readability
                is_readable, error = ImageValidator.validate_image_readability(file_path)
                if not is_readable:
                    return False, error
                logger.debug(f"Image readable: {file_path}")

                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    return False, "Failed to load image with OpenCV"
            else:
                # Array input — check content type
                if not isinstance(image, np.ndarray):
                    return False, "Image must be a numpy array or file path"

            # Validate dimensions
            ImageValidator.validate_image_dimensions(image)
            logger.debug(f"Dimensions valid: {image.shape}")

            # Detect corruption
            has_corruption, corruption_desc = ImageValidator.detect_corruption(image)
            if has_corruption:
                return False, f"Corruption detected: {corruption_desc}"

            logger.info(f"Image validation passed: {file_path}")
            return True, None

        except ImageValidationError as e:
            logger.warning(f"Validation error: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return False, f"Unexpected error: {str(e)}"


def validate_image(file_path: str, image: Optional[np.ndarray] = None) -> bool:
    """
    Quick validation function. Accepts either a file path or a loaded array.

    Args:
        file_path: Path to image file (or arbitrary label when image is given)
        image: Optional pre-loaded image array

    Returns:
        bool: True if image is valid

    Raises:
        ImageValidationError: If validation fails
    """
    is_valid, error = ImageValidator.validate_complete(file_path, image=image)
    if not is_valid:
        raise ImageValidationError(error or "Image validation failed")
    return True
