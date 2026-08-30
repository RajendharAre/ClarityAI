# Phase 2 Documentation — Dataset Preparation

## Overview

Phase 2 creates a comprehensive synthetic dataset for training and evaluating the ClarityAI model. This phase generates labeled image examples with various quality degradations to simulate real-world quality issues.

## Design Principles Applied

✅ **Reproducibility** — Seeded random generation for exact replication
✅ **Modularity** — Degradation strategies are independent and composable
✅ **No Data Leakage** — Source images are strictly separated across splits
✅ **Interpretability** — All degradations have documented parameters
✅ **Testing** — Comprehensive test coverage for generation logic
✅ **GoF Design Patterns** — Factory, Strategy, Decorator patterns used

## Dataset Structure

### Output Directory Layout

```
ml/data/
├── train/                      # Training set
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── metadata.jsonl         # Training metadata
├── val/                        # Validation set
│   ├── image_1.jpg
│   └── metadata.jsonl
├── test/                       # Test set
│   ├── image_1.jpg
│   └── metadata.jsonl
├── dataset_report.json         # Generation report
└── source_images/             # (User-provided) Source images
    ├── photo_1.jpg
    └── photo_2.jpg
```

### Data Splits (No Leakage)

- **Training Set:** 70% of source images (and all their degradations)
- **Validation Set:** 15% of source images (and all their degradations)
- **Test Set:** 15% of source images (and all their degradations)

**Key Principle:** Source images are split first, then degraded. No source image appears in multiple splits, preventing data leakage.

## Implemented Degradations

### 1. Blur (Sharpness Issue)

**What it simulates:** Out-of-focus or motion blur

**Parameters:**
- Kernel size: 3 to 21 pixels
- Severity range: 0.0 (minimal) to 1.0 (heavy)

**Implementation:** Gaussian blur via OpenCV

```python
# Severity 0.3 → kernel 8 (slight blur)
# Severity 0.7 → kernel 17 (heavy blur)
```

**Code:** `BlurDegradation` class

---

### 2. Exposure (Brightness Issue)

**What it simulates:** Underexposure (too dark) or overexposure (too bright)

**Parameters:**
- Brightness factor: 0.3 to 1.7
- Severity < 0.5: Underexposure
- Severity > 0.5: Overexposure

**Implementation:** Brightness scaling with clipping

```python
# Severity 0.2 → brightness 0.5 (50% = very dark)
# Severity 0.5 → brightness 1.0 (normal)
# Severity 0.8 → brightness 1.54 (154% = very bright)
```

**Code:** `ExposureDegradation` class

---

### 3. Gaussian Noise (Noise Issue)

**What it simulates:** Sensor noise or thermal noise

**Parameters:**
- Noise sigma: 5 to 50
- Severity range: 0.0 to 1.0

**Implementation:** Gaussian noise addition

```python
# Severity 0.3 → sigma 18.5 (slight noise)
# Severity 0.8 → sigma 41 (heavy noise)
```

**Code:** `NoiseDegradation` class

---

### 4. Salt-and-Pepper Noise (Compression Artifact)

**What it simulates:** Random pixel errors or compression glitches

**Parameters:**
- Noise probability: 0% to 10%
- Affects individual pixels randomly

**Implementation:** Random pixel corruption

```python
# Severity 0.5 → 5% of pixels affected
```

**Code:** `SaltPepperNoiseDegradation` class

---

### 5. JPEG Compression (Corruption)

**What it simulates:** Lossy JPEG compression artifacts

**Parameters:**
- Quality: 95 (minimal) to 10 (heavy)
- Severity range: 0.0 to 1.0

**Implementation:** Re-encoding with lossy JPEG

```python
# Severity 0.2 → quality 80 (minor artifacts)
# Severity 0.9 → quality 12 (severe blocking)
```

**Code:** `JPEGCompressionDegradation` class

---

### 6. Scratch Defects (Visual Defect)

**What it simulates:** Scratches or sensor defects

**Parameters:**
- Number of scratches: 0 to 5
- Random lines with varying angles

**Implementation:** Drawing lines on image

**Code:** `ScratchDegradation` class

---

### 7. Spot Defects (Visual Defect)

**What it simulates:** Dust spots or lens blemishes

**Parameters:**
- Number of spots: 0 to 20
- Random circles with varying sizes

**Implementation:** Drawing circles on image

**Code:** `SpotDegradation` class

---

## Quality Labels

### ACCEPTABLE
- **Description:** No significant quality issues
- **Degradations:** None
- **Severities:** 0.0 (no degradation)
- **Count per source:** 1 image

Example: Clean, well-focused, properly exposed image

### DEGRADED
- **Description:** One or more quality issues present but still usable
- **Degradations:** Blur, exposure, noise, salt-pepper, JPEG compression
- **Severities:** 0.2-0.5 (moderate)
- **Count per source:** 3 images per source

Examples:
- Slightly blurry but readable
- Slightly underexposed but details visible
- Some noise but acceptable
- Minor JPEG artifacts

### DEFECTIVE
- **Description:** Severe quality problems or visible defects
- **Degradations:** All types including scratches and spots
- **Severities:** 0.5-1.0 (heavy)
- **Count per source:** 2 images per source

Examples:
- Very blurry, unusable
- Severely over/underexposed, details lost
- Heavy noise, graininess
- Visible scratches or spots
- Severe compression artifacts

## Generation Configuration

**Hardcoded Parameters (for reproducibility):**

```python
class DatasetConfig:
    SEED = 42                    # Random seed
    TRAIN_RATIO = 0.7           # 70% training
    VAL_RATIO = 0.15            # 15% validation
    TEST_RATIO = 0.15           # 15% test
    
    DEGRADATIONS_PER_LABEL = {
        "ACCEPTABLE": 1,        # 1 clean copy per source
        "DEGRADED": 3,          # 3 degraded copies per source
        "DEFECTIVE": 2,         # 2 heavily degraded copies per source
    }
```

## Design Patterns Used

### 1. Strategy Pattern

Each degradation type is an independent `Degradation` class implementing the `apply()` method. This allows:
- Easy addition of new degradation types
- Mixing and matching degradations
- Testing each degradation independently

```python
class Degradation(ABC):
    @abstractmethod
    def apply(self, image, severity):
        pass

class BlurDegradation(Degradation):
    def apply(self, image, severity):
        # Blur implementation
```

### 2. Factory Pattern

`DegradationFactory` creates degradation objects by type name:

```python
degradation = DegradationFactory.create("blur")  # Returns BlurDegradation()
```

Benefits:
- Centralized degradation creation
- Easy to extend with new types
- Type validation

### 3. Decorator Pattern (Pipeline)

`DegradationPipeline` chains multiple degradations:

```python
pipeline = DegradationPipeline()
pipeline.add_degradation("blur", 0.3)
pipeline.add_degradation("noise", 0.5)
result = pipeline.apply(image)
```

Benefits:
- Composable degradations
- Realistic multi-issue combinations
- Chainable API

## How to Use

### Generate Sample Dataset (No source images needed)

```bash
cd ml
python generate_dataset.py
```

Creates `ml/data/` with sample images for testing.

### Generate from Your Own Images

1. **Prepare source images:**
```bash
mkdir -p ml/data/source_images
# Copy your images to ml/data/source_images/
cp /path/to/images/*.jpg ml/data/source_images/
```

2. **Generate dataset:**
```bash
cd ml
python generate_dataset.py
```

### Programmatic Usage

```python
from ml.dataset_generator import DatasetGenerator
from ml.degradations import DegradationPipeline

# Generate dataset
generator = DatasetGenerator()
report = generator.generate_dataset("./ml/data/source_images", "./ml/data")

# Or manually create degradations
pipeline = DegradationPipeline()
pipeline.add_degradation("blur", 0.4)
pipeline.add_degradation("noise", 0.3)
degraded_image = pipeline.apply(image)
```

## Metadata Format

Each image includes metadata in JSON format:

```json
{
  "filename": "photo_degraded_0.jpg",
  "source_image": "photo.jpg",
  "quality_label": "DEGRADED",
  "degradations": [
    {"type": "blur", "severity": 0.35},
    {"type": "noise", "severity": 0.42}
  ],
  "generation_timestamp": "2026-08-30T12:00:00.000000",
  "degradation_seed": 1234567890
}
```

**Stored as:** `metadata.jsonl` (one JSON per line for efficient loading)

## Dataset Statistics

**Example report (5 source images):**

```json
{
  "timestamp": "2026-08-30T12:00:00.000000",
  "seed": 42,
  "total_images": 40,
  "label_distribution": {
    "ACCEPTABLE": 5,
    "DEGRADED": 15,
    "DEFECTIVE": 10
  },
  "train_size": 28,
  "val_size": 6,
  "test_size": 6,
  "output_directory": "./ml/data"
}
```

## Reproducibility Guarantees

**The same output is guaranteed if:**
1. Same source images
2. Same `SEED` value (42 by default)
3. Same `DatasetConfig` parameters

**To reproduce:**
```python
generator = DatasetGenerator(config=DatasetConfig())
report = generator.generate_dataset(source_dir)
```

Same report every time.

## Testing

Run dataset generation tests:

```bash
cd ml
pytest tests/test_dataset_generation.py -v
```

**Test coverage:**
- Degradation factory creation
- All degradation types
- Pipeline chaining
- Data split validation (no leakage)
- Metadata generation
- Report structure

## Known Limitations

1. **Synthetic Degradations** — Real-world defects are more complex
2. **Distribution Mismatch** — Generated data may not match real image distributions
3. **Correlation Ignored** — Multiple degradations are applied independently
4. **No Hard Negatives** — Dataset doesn't include tricky edge cases

## Future Enhancements

1. **Real Defect Dataset** — Augment with real images from MVTec AD or similar
2. **Domain Randomization** — Vary degradation parameters more broadly
3. **Intensity Tuning** — Learn optimal degradation parameters from real data
4. **Dynamic Generation** — Generate degradations on-the-fly during training
5. **Class Balancing** — Adjust counts to match real-world distributions

## References

1. "No-Reference Image Quality Assessment" (BRISQUE, NIQE)
2. "Anomaly Detection via Self-Supervised Learning" (MVTec AD)
3. OpenCV Image Processing Documentation
4. "Degradation Synthesis for Robust Image Quality Assessment"

---

**Next Phase:** Phase 3 — Model Development (train baseline and deep models)
