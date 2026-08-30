# Phase 1 Documentation — Classical Image Feature Extraction

## Overview

Phase 1 implements deterministic, interpretable image quality signals using classical computer vision techniques. These features serve two purposes:

1. **Direct Quality Signals** — Some features (sharpness, exposure) directly indicate quality issues
2. **Input to ML Models** — Features feed into learned models in Phase 3

## Design Principles Applied

✅ **Separation of Concerns** — Each feature extractor is independent
✅ **Single Responsibility** — Each class does one thing well
✅ **Modularity** — Reusable components via Strategy pattern (GoF)
✅ **Interpretability** — Every feature is explainable to users
✅ **Reproducibility** — Methods documented with references
✅ **Testing** — Comprehensive unit tests for validation
✅ **Security** — Input validation in ImageValidator

## Implemented Features

### 1. Sharpness (Laplacian Variance)

**What it measures:** How sharp/in-focus the image is

**Method:** Variance of Laplacian operator
- Laplacian detects edges (high-frequency changes)
- Sharp images have high variance of these edges
- Blurry images have low variance

**Reference:** "A No-Reference Image Sharpness Metric Based on the Difference Between Laplacian Operators"

**Typical Thresholds:**
- Acceptable: > 100
- Degraded: 50-100
- Defective: < 50

**Implementation:** `SharpnessExtractor.extract()`

```python
# High sharpness (sharp image)
sharpness = 150.5

# Low sharpness (blurry image)
sharpness = 25.3
```

### 2. Brightness (Mean Luminance)

**What it measures:** Average brightness of the image

**Method:** Mean of L channel in LAB color space
- LAB L channel is perceptually uniform
- Range: 0-255
- Unaffected by color information

**Typical Ranges:**
- Underexposed (too dark): 0-85
- Normal: 85-170
- Overexposed (too bright): 170-255

**Implementation:** `BrightnessExtractor.extract()`

```python
# Dark image (underexposed)
brightness = 40.2

# Well-exposed image
brightness = 128.5

# Bright image (overexposed)
brightness = 215.8
```

### 3. Contrast (Pixel Intensity Std Dev)

**What it measures:** How much pixel values vary (visual distinction)

**Method:** Standard deviation of pixel intensities in grayscale
- Higher std dev = more contrast
- Lower std dev = flat, low-contrast image

**Typical Values:**
- Low contrast (flat image): < 20
- Normal contrast: 20-100
- High contrast (dramatic lighting): > 100

**Implementation:** `ContrastExtractor.extract()`

```python
# Low contrast (flat)
contrast = 5.2

# Normal contrast
contrast = 45.8

# High contrast
contrast = 125.3
```

### 4. Noise (High-Frequency Residuals)

**What it measures:** Amount of visible sensor/compression noise

**Method:** Standard deviation of Laplacian (high-frequency component)
- Laplacian captures rapid intensity changes
- Noise appears as high-frequency variation
- Clean images have low values

**Typical Values:**
- Clean image: 10-30
- Slightly noisy: 30-60
- Very noisy: > 60

**Implementation:** `NoiseExtractor.extract()`

```python
# Clean image
noise_level = 15.3

# Noisy image
noise_level = 52.7
```

### 5. Saturation (Color Intensity)

**What it measures:** How vibrant/colorful the image is

**Method:** Mean of S channel in HSV color space
- HSV S channel = saturation
- Range: 0-255
- 0 = grayscale, 255 = pure colors

**Typical Values:**
- Desaturated: 0-50
- Normal: 50-150
- Over-saturated: 150-255

**Implementation:** `SaturationExtractor.extract()`

```python
# Grayscale/desaturated
saturation = 10.5

# Normal colors
saturation = 100.2

# Vibrant colors
saturation = 180.7
```

### 6. Texture Complexity (Entropy)

**What it measures:** Local pattern complexity in the image

**Method:** Entropy of gradient magnitude histogram
- Computes Sobel gradients (edge detection)
- Calculates histogram of gradient magnitudes
- Entropy measures randomness/complexity

**Typical Values:**
- Simple texture: 2-4 bits
- Complex texture: 4-7 bits

**Implementation:** `TextureComplexityExtractor.extract()`

```python
# Simple, uniform texture
texture_complexity = 1.2

# Complex, detailed texture
texture_complexity = 5.8
```

## File Corruption Detection

Beyond features, the system detects common corruption issues:

**Implemented Checks:**
- Empty image array
- Zero variance (all pixels identical)
- NaN/Inf values
- Dimension sanity checks
- File integrity verification

**Implementation:** `ImageValidator` class

## Code Organization

```
ml/
├── feature_extraction.py      # Feature extraction classes
├── image_validation.py         # Image validation & corruption detection
├── tests/
│   ├── __init__.py
│   └── test_feature_extraction.py  # Unit tests
└── __init__.py
```

## Architecture

### Design Patterns Used

**Strategy Pattern (GoF):**
```
FeatureExtractor (abstract)
    ├── SharpnessExtractor
    ├── BrightnessExtractor
    ├── ContrastExtractor
    ├── NoiseExtractor
    ├── SaturationExtractor
    └── TextureComplexityExtractor
```

**Facade Pattern (GoF):**
```
ImageFeatureExtractor (orchestrator)
    └── Combines all extractors
    └── Single interface to extract all features
```

## Testing Strategy

**Unit Tests:** `test_feature_extraction.py`

Test classes:
- `TestSharpnessExtraction` — Verify sharpness detection works
- `TestBrightnessExtraction` — Verify brightness ranges
- `TestContrastExtraction` — Verify contrast calculations
- `TestNoiseExtraction` — Verify noise detection
- `TestImageFeatureExtractor` — Integration tests
- `TestImageValidation` — Validation tests

**Test Approach:**
- Create synthetic test images (sharp, blurry, dark, bright, etc.)
- Verify extracted features make intuitive sense
- Ensure comparisons work correctly (sharp > blurry, bright > dark)
- Test edge cases (invalid files, corrupted data)

## How to Use

### Quick Feature Extraction

```python
import cv2
from ml.feature_extraction import extract_image_features

# Load image
image = cv2.imread("photo.jpg")

# Extract all features
features = extract_image_features(image)

# Access individual features
print(f"Sharpness: {features.sharpness}")
print(f"Brightness: {features.brightness}")
print(f"Contrast: {features.contrast}")

# Convert to dict (for JSON serialization)
features_dict = features.to_dict()
```

### Validate Image Before Processing

```python
from ml.image_validation import validate_image, ImageValidator

# Quick validation
try:
    validate_image("photo.jpg")
    print("Image is valid!")
except ImageValidationError as e:
    print(f"Validation failed: {e}")

# Detailed validation
is_valid, error = ImageValidator.validate_complete("photo.jpg")
if is_valid:
    print("Image passed all checks")
else:
    print(f"Validation error: {error}")
```

### Running Tests

```bash
# Run all tests
cd ml
pytest tests/test_feature_extraction.py -v

# Run specific test class
pytest tests/test_feature_extraction.py::TestSharpnessExtraction -v

# Run with coverage
pytest tests/test_feature_extraction.py --cov=ml
```

## Dependencies

See `backend/requirements.txt`:
- opencv-python — Image processing
- numpy — Array operations
- Pillow — Image validation
- scikit-image — Advanced image processing

## Known Limitations

1. **Feature Interpretation is Linear** — Simple features like sharpness don't capture all quality issues
2. **No Context Awareness** — Features don't know if low brightness is intentional (e.g., night photography)
3. **Threshold Dependency** — Quality thresholds are hardcoded for typical images
4. **Grayscale Limitation** — Some features (saturation) return 0 for grayscale images

## Future Enhancements (Phase 3+)

- Combine features with machine learning for better accuracy
- Add anomaly detection for defects
- Learn optimal thresholds from data
- Add more specialized features (focus quality, motion blur)

## References

1. "A No-Reference Image Sharpness Metric Based on the Difference Between Laplacian Operators" — Pech-Pérez et al.
2. "BRISQUE: A Fast Non-Reference Image Quality Assessment Algorithm" — Mittal et al.
3. "NIQE: Natural Image Quality Evaluator" — Mittal et al.
4. OpenCV Documentation: Image Processing

---

**Next Phase:** Phase 2 — Dataset Preparation (synthetic degradations, data splits)
