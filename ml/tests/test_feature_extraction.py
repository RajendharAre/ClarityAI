"""
Unit Tests for Feature Extraction Module
Validates feature extraction correctness on sample images.
Follows Design Principle: Testing & Quality Assurance
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from ml.feature_extraction import (
    ImageFeatureExtractor,
    SharpnessExtractor,
    BrightnessExtractor,
    ContrastExtractor,
    NoiseExtractor,
    extract_image_features,
)
from ml.image_validation import ImageValidator, validate_image


class TestSharpnessExtraction:
    """Tests for sharpness feature extraction"""

    @pytest.fixture
    def sharp_image(self):
        """Create a sharp test image with clear edges"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create sharp black and white regions
        img[:50, :] = 255  # Top white
        img[50:, :] = 0    # Bottom black
        return img

    @pytest.fixture
    def blurry_image(self, sharp_image):
        """Create a blurry version of the sharp image"""
        blurred = cv2.GaussianBlur(sharp_image, (21, 21), 0)
        return blurred

    def test_sharp_image_has_high_sharpness(self, sharp_image):
        """Sharp images should have high sharpness score"""
        extractor = SharpnessExtractor()
        sharpness = extractor.extract(sharp_image)
        assert sharpness > 50  # Should be reasonably high

    def test_blurry_image_has_low_sharpness(self, blurry_image, sharp_image):
        """Blurry images should have lower sharpness than sharp images"""
        extractor = SharpnessExtractor()
        sharp_score = extractor.extract(sharp_image)
        blurry_score = extractor.extract(blurry_image)
        assert sharp_score > blurry_score

    def test_grayscale_and_color_consistency(self, sharp_image):
        """Grayscale and color images should produce similar results"""
        gray = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
        extractor = SharpnessExtractor()
        
        sharp_from_color = extractor.extract(sharp_image)
        sharp_from_gray = extractor.extract(gray)
        
        # Should be similar (difference < 10%)
        assert abs(sharp_from_color - sharp_from_gray) / sharp_from_color < 0.1


class TestBrightnessExtraction:
    """Tests for brightness feature extraction"""

    def test_dark_image_has_low_brightness(self):
        """Dark images should have low brightness score"""
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 50  # Dark gray
        extractor = BrightnessExtractor()
        brightness = extractor.extract(dark_img)
        assert brightness < 100

    def test_bright_image_has_high_brightness(self):
        """Bright images should have high brightness score"""
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Light gray
        extractor = BrightnessExtractor()
        brightness = extractor.extract(bright_img)
        assert brightness > 100

    def test_brightness_range(self):
        """Brightness should be in range 0-255"""
        for value in [0, 50, 128, 200, 255]:
            img = np.ones((100, 100, 3), dtype=np.uint8) * value
            extractor = BrightnessExtractor()
            brightness = extractor.extract(img)
            assert 0 <= brightness <= 255


class TestContrastExtraction:
    """Tests for contrast feature extraction"""

    def test_high_contrast_image(self):
        """High contrast images should have high contrast score"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = 255  # Top white
        img[50:, :] = 0    # Bottom black
        
        extractor = ContrastExtractor()
        contrast = extractor.extract(img)
        assert contrast > 100

    def test_low_contrast_image(self):
        """Low contrast images (uniform color) should have low contrast"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Uniform gray
        extractor = ContrastExtractor()
        contrast = extractor.extract(img)
        assert contrast < 10

    def test_contrast_increases_with_variation(self):
        """Contrast should increase with pixel value variation"""
        extractor = ContrastExtractor()
        
        # Create images with increasing contrast
        img1 = np.ones((100, 100), dtype=np.uint8) * 128  # No variation
        img2 = np.concatenate([
            np.ones((50, 100), dtype=np.uint8) * 100,
            np.ones((50, 100), dtype=np.uint8) * 156,
        ])
        img3 = np.concatenate([
            np.ones((50, 100), dtype=np.uint8) * 50,
            np.ones((50, 100), dtype=np.uint8) * 200,
        ])
        
        c1 = extractor.extract(img1)
        c2 = extractor.extract(img2)
        c3 = extractor.extract(img3)
        
        assert c1 < c2 < c3


class TestNoiseExtraction:
    """Tests for noise feature extraction"""

    def test_clean_image_has_low_noise(self):
        """Clean images should have low noise scores"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Uniform
        extractor = NoiseExtractor()
        noise = extractor.extract(img)
        assert noise < 50

    def test_noisy_image_has_high_noise(self):
        """Noisy images should have high noise scores"""
        # Create image with Gaussian noise
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        noise_arr = np.random.normal(0, 30, img.shape)
        noisy_img = np.clip(img.astype(float) + noise_arr, 0, 255).astype(np.uint8)
        
        extractor = NoiseExtractor()
        clean_noise = extractor.extract(img)
        noisy_noise = extractor.extract(noisy_img)
        
        assert noisy_noise > clean_noise


class TestImageFeatureExtractor:
    """Tests for complete feature extraction"""

    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = 200  # Bright square in center
        return img

    def test_extract_all_returns_all_features(self, test_image):
        """extract_all should return all feature values"""
        extractor = ImageFeatureExtractor()
        features = extractor.extract_all(test_image)
        
        assert features.sharpness > 0
        assert features.brightness > 0
        assert features.contrast > 0
        assert features.noise_level >= 0
        assert features.saturation >= 0
        assert features.texture_complexity >= 0

    def test_to_dict_conversion(self, test_image):
        """Features should convert to dict properly"""
        extractor = ImageFeatureExtractor()
        features = extractor.extract_all(test_image)
        features_dict = features.to_dict()
        
        assert isinstance(features_dict, dict)
        assert "sharpness" in features_dict
        assert "brightness" in features_dict
        assert "contrast" in features_dict

    def test_convenience_function(self, test_image):
        """Convenience function should work"""
        features = extract_image_features(test_image)
        assert features.sharpness > 0
        assert features.brightness > 0


class TestImageValidation:
    """Tests for image validation"""

    @pytest.fixture
    def valid_image_file(self):
        """Create a temporary valid image file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        Path(tmp.name).unlink()

    def test_valid_image_passes_validation(self, valid_image_file):
        """Valid image should pass validation"""
        is_valid, error = ImageValidator.validate_complete(valid_image_file)
        assert is_valid
        assert error is None

    def test_invalid_extension_fails(self, tmp_path):
        """File with invalid extension should fail"""
        invalid_file = tmp_path / "image.txt"
        invalid_file.write_text("not an image")
        
        is_valid, error = ImageValidator.validate_complete(str(invalid_file))
        assert not is_valid
        assert error is not None

    def test_nonexistent_file_fails(self):
        """Nonexistent file should fail validation"""
        is_valid, error = ImageValidator.validate_complete("/nonexistent/file.png")
        assert not is_valid

    def test_validate_image_function(self, valid_image_file):
        """Quick validation function should work"""
        assert validate_image(valid_image_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
