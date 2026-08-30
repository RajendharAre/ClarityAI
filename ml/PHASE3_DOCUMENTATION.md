# Phase 3 Documentation — Model Development

## Overview

Phase 3 implements a complete model development pipeline combining classical ML and deep learning for image quality assessment. The system trains on Phase 2-generated synthetic datasets and provides unified quality predictions through score fusion.

## Design Principles Applied

✅ **Modularity** — Baseline, deep, and fusion are independent but composable
✅ **Interpretability** — Feature importance, reasoning strings, breakdowns
✅ **Reproducibility** — Fixed seeds, documented parameters
✅ **Ensemble Methods** — Combining baseline and deep predictions
✅ **Scalability** — Batch processing, GPU support
✅ **Graceful Degradation** — Works with or without models loaded

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Image Input                                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
  ┌──────────────┐           ┌──────────────┐
  │ Baseline ML  │           │ Deep Learning│
  │ (Phase 1     │           │ (Autoencoder)│
  │  Features)   │           │              │
  └────────┬─────┘           └──────┬───────┘
           │                        │
           │ Quality Label          │ Anomaly Score
           │ + Confidence           │ (0.0-1.0)
           │                        │
           └────────────┬───────────┘
                        │
                        ▼
                   ┌────────────┐
                   │Score Fusion│
                   │Logic       │
                   └─────┬──────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ QualityAnalysis      │
              │ - quality_label      │
              │ - quality_score 0-100
              │ - confidence         │
              │ - reasoning          │
              └──────────────────────┘
```

## Component Details

### 1. Baseline Model (ml/baseline_model.py)

**Purpose:** Deterministic quality classification using engineered features

**Architecture:**
- Input: 6 features from Phase 1 (sharpness, brightness, contrast, noise, saturation, texture)
- Processing: StandardScaler normalization
- Model: Random Forest (100 trees, max_depth=15) or Gradient Boosting
- Output: Quality label (0=ACCEPTABLE, 1=DEGRADED, 2=DEFECTIVE) + confidence

**Class: BaselineModel**

```python
model = BaselineModel(model_type="random_forest")
metrics = model.train(X_features, y_labels)  # (N, 6) features, (N,) labels
label, confidence = model.predict(features)  # Single prediction
results = model.predict(batch_features)      # Batch prediction
model.save("./ml/models")
```

**Key Methods:**
- `train(X, y, val_size=0.2)` → Dict with train/val accuracy
- `predict(features)` → (label, confidence)
- `get_feature_importance()` → Dict of feature importance scores
- `save(dir)` → Saves model, scaler, metadata
- `load(path)` → Static method to load model

**Output Artifacts:**
- `baseline_model_random_forest_v1.joblib` → Trained classifier
- `baseline_scaler_random_forest_v1.joblib` → StandardScaler
- `baseline_metadata_random_forest_v1.json` → Feature names, class mapping

**Hyperparameters:**
```python
Random Forest:
  n_estimators=100
  max_depth=15
  min_samples_split=5
  min_samples_leaf=2
  random_state=42
```

---

### 2. Deep Learning Model (ml/deep_model.py)

**Purpose:** Anomaly detection via reconstruction error

**Architecture:**

**AutoencoderModel** (PyTorch nn.Module):
```
Encoder:
  Input (B, 3, 64, 64) / 256x256 images
  ↓ Conv2d(3→32, k=4, s=2) + BN + ReLU
  ↓ Conv2d(32→64, k=4, s=2) + BN + ReLU
  ↓ Conv2d(64→128, k=4, s=2) + BN + ReLU
  ↓ GlobalAvgPool
  ↓ Linear(128→128)  [latent space]

Decoder:
  Linear(128→128*8*8)
  ↓ Reshape to (B, 128, 8, 8)
  ↓ ConvTranspose2d(128→64, k=4, s=2) + BN + ReLU
  ↓ ConvTranspose2d(64→32, k=4, s=2) + BN + ReLU
  ↓ ConvTranspose2d(32→3, k=4, s=2) + Sigmoid
  Output (B, 3, 64, 64)
```

**Training Details:**
- **Data:** ONLY "ACCEPTABLE" (normal) images
  - Anomalies not shown during training
  - Forces model to learn normality distribution
- **Loss:** MSE reconstruction error
  - Low error on normal images
  - High error on anomalies
- **Optimizer:** Adam (lr=1e-3)
- **Epochs:** 50 (configurable)
- **Batch Size:** 32

**Class: DeepModel**

```python
model = DeepModel(device="cpu")  # or "cuda"
history = model.train_model(X_train, epochs=50, batch_size=32)
anomaly_scores = model.compute_anomaly_score(images)  # (N,) array
model.save("./ml/models")
loaded = DeepModel.load(path, device="cpu")
```

**Key Methods:**
- `train_model(X, epochs, batch_size, lr, val_split)` → Dict with loss history
- `compute_anomaly_score(images)` → Array of scores (higher = more anomalous)
- `save(dir)` → Saves model weights + metadata
- `load(path, device)` → Static method to load model

**Output Artifacts:**
- `deep_model_autoencoder_v1.pt` → Model weights
- `deep_metadata_autoencoder_v1.json` → Architecture info

**Anomaly Interpretation:**
- Score 0.0-0.05: Normal (ACCEPTABLE)
- Score 0.05-0.2: Slightly anomalous (DEGRADED candidate)
- Score 0.2+: Highly anomalous (DEFECTIVE candidate)

---

### 3. Score Fusion (ml/score_fusion.py)

**Purpose:** Combine baseline and deep predictions into unified assessment

**Classes:**

**QualityAnalysis** (Dataclass)
```python
@dataclass
class QualityAnalysis:
    quality_label: str          # Final prediction
    quality_score: float        # 0-100 (higher = better)
    confidence: float           # 0-1
    baseline_label: str         # Baseline prediction
    baseline_confidence: float  # Baseline confidence
    anomaly_score: float        # Raw reconstruction error
    anomaly_percentile: float   # Percentile in distribution
    feature_breakdown: Dict     # Individual feature scores
    reasoning: str              # Explanation
```

**ScoreFusion** (Fusion Logic)

Fusion Strategy:
```
IF baseline_label == DEFECTIVE:
  → quality_label = DEFECTIVE (trust strong signal)
  
ELSE IF baseline_label == DEGRADED:
  → quality_label = DEGRADED (no fusion needed)
  
ELSE IF baseline_label == ACCEPTABLE:
  IF anomaly_score is low:
    → quality_label = ACCEPTABLE (both agree)
  ELSE IF anomaly_score is moderate:
    → quality_label = DEGRADED (upgrade due to anomaly)
  ELSE:
    → quality_label = DEFECTIVE (upgrade due to strong anomaly)
```

Score Scaling:
- 90-100: ACCEPTABLE (high quality)
- 50-89: DEGRADED (usable)
- 0-49: DEFECTIVE (unusable)

```python
fusion = ScoreFusion()
result = fusion.fuse_predictions(
    baseline_label="ACCEPTABLE",
    baseline_confidence=0.92,
    anomaly_score=0.05,
    features={"sharpness": 180, ...}
)
# Returns QualityAnalysis object

# Batch processing
results = fusion.batch_fuse(labels_list, confs_list, scores_array)
```

**AdaptiveScoring** (Learning from Data)
```python
adaptive = AdaptiveScoring()

# Collect ground truth during evaluation
for image, true_label in test_set:
    pred = analyzer.analyze_image(image)
    adaptive.collect_samples(
        quality_label=true_label,
        baseline_confidence=pred.baseline_confidence,
        anomaly_score=pred.anomaly_score
    )

# Compute statistics and optimize thresholds
stats = adaptive.compute_statistics()
optimized_thresholds = adaptive.get_optimized_thresholds()
```

---

### 4. Inference Pipeline (ml/inference.py)

**Purpose:** End-to-end image quality analysis

**Class: QualityAnalyzer** (Facade Pattern)

```python
# Initialize with trained models
analyzer = QualityAnalyzer(
    baseline_model_path="./ml/models/baseline_model_v1.joblib",
    deep_model_path="./ml/models/deep_model_v1.pt",
    device="cpu"
)

# Analyze single image
result = analyzer.analyze_image("image.jpg")  # or np.ndarray
# Returns QualityAnalysis

# Batch processing
results = analyzer.batch_analyze([img1, img2, img3])

# Directory analysis
results = analyzer.analyze_directory("./images/", recursive=True)
# Returns Dict[path → QualityAnalysis]
```

**Pipeline Steps:**
1. Image validation (corruption check)
2. Feature extraction (Phase 1)
3. Baseline prediction (Phase 3a)
4. Anomaly scoring (Phase 3b)
5. Score fusion (Phase 3c)
6. Return QualityAnalysis

**Convenience Function:**
```python
from ml.inference import infer

result = infer("image.jpg")  # Dict output
# {
#   "quality_label": "ACCEPTABLE",
#   "quality_score": 92,
#   "confidence": 0.95,
#   ...
# }
```

---

### 5. Training Pipeline (ml/model_training.py)

**Purpose:** Orchestrates complete model training end-to-end

**Class: ModelTrainingPipeline**

```python
from ml.model_training import ModelTrainingPipeline

pipeline = ModelTrainingPipeline(
    data_dir="./ml/data",
    models_dir="./ml/models"
)

report = pipeline.run_full_pipeline(
    generate_sample_data=True,
    num_sample_images=10,
    deep_model_epochs=50,
    device="cpu"
)
```

**Pipeline Steps:**
1. Load/generate dataset (Phase 2)
2. Extract features for baseline
3. Train baseline model
4. Filter ACCEPTABLE images for deep model
5. Train deep model
6. Validate on test set
7. Save artifacts + report

**Usage from Command Line:**
```bash
cd ml
python model_training.py --generate-data --num-samples 20 --epochs 50 --device cpu
```

**Output Report:**
```json
{
  "timestamp": "2026-08-30T12:00:00",
  "baseline_model_path": "./ml/models/baseline_model_random_forest_v1.joblib",
  "deep_model_path": "./ml/models/deep_model_autoencoder_v1.pt",
  "training_samples": 100,
  "test_samples": 30,
  "baseline_accuracy": 0.867,
  "training_status": "COMPLETE"
}
```

---

## How to Use Phase 3

### Training Phase

**Option 1: Generate sample dataset and train**
```bash
cd ml
python model_training.py --generate-data --num-samples 20 --epochs 50
```

**Option 2: Train on existing dataset**
```bash
cd ml
python model_training.py --data-dir ./ml/data --models-dir ./ml/models
```

### Inference Phase

**In Python:**
```python
from ml.inference import QualityAnalyzer, infer

# Initialize analyzer
analyzer = QualityAnalyzer(
    baseline_model_path="./ml/models/baseline_model_random_forest_v1.joblib",
    deep_model_path="./ml/models/deep_model_autoencoder_v1.pt"
)

# Analyze single image
result = analyzer.analyze_image("test.jpg")
print(f"Quality: {result.quality_label} ({result.quality_score}/100)")

# Or use convenience function
result_dict = infer("test.jpg")
```

**Output Example:**
```python
QualityAnalysis(
    quality_label='ACCEPTABLE',
    quality_score=92,
    confidence=0.95,
    baseline_label='ACCEPTABLE',
    baseline_confidence=0.92,
    anomaly_score=0.032,
    anomaly_percentile=0.18,
    feature_breakdown={
        'sharpness': 185.5,
        'brightness': 128.3,
        'contrast': 45.2,
        'noise_level': 8.1,
        'saturation': 120.4,
        'texture_complexity': 2.34
    },
    reasoning='Baseline model and anomaly detector both confirm high quality.'
)
```

---

## Design Patterns Used

### 1. Strategy Pattern (Degradation Types)
Each degradation type (blur, noise, etc.) is an independent strategy.

### 2. Factory Pattern (Degradation Creation)
`DegradationFactory.create(type)` creates degradations by name.

### 3. Decorator Pattern (Pipeline)
`DegradationPipeline` chains degradations together.

### 4. Adapter Pattern (BaselineModel)
Converts `FeatureStats` to scikit-learn compatible format.

### 5. Facade Pattern (QualityAnalyzer)
Single interface for complex multi-component pipeline.

### 6. Ensemble Pattern (ScoreFusion)
Combines multiple weak predictors into strong ensemble.

---

## Testing

Run all Phase 3 tests:
```bash
cd ml
pytest tests/test_models.py -v
```

**Test Coverage:**
- Baseline model: Training, prediction, save/load
- Deep model: Training, anomaly scoring, save/load
- Score fusion: All fusion paths, batch processing
- Adaptive scoring: Threshold learning
- Inference pipeline: Single, batch, directory analysis

---

## Performance Characteristics

**Baseline Model:**
- Training time: ~1 second (100 samples)
- Inference time: ~1ms per image
- Accuracy: ~85% on synthetic data

**Deep Model:**
- Training time: ~5 minutes per 50 epochs (on CPU)
- Inference time: ~100ms per image (CPU)
- Training time: ~1 minute per 50 epochs (on GPU)
- Inference time: ~10ms per image (GPU)

**Fusion:**
- Fusion time: <1ms
- No training overhead

**Total Pipeline:**
- Full inference (extraction + baseline + deep + fusion): ~120ms per image (CPU)

---

## Hyperparameter Guide

**Baseline Model:**
- `n_estimators`: 100 (more = better, slower)
- `max_depth`: 15 (controls complexity)
- `val_size`: 0.2 (20% validation split)

**Deep Model:**
- `epochs`: 50 (more = better until overfitting)
- `batch_size`: 32 (larger = faster but more memory)
- `learning_rate`: 1e-3 (typical for Adam)
- `latent_dim`: 128 (bottleneck size)

**Score Fusion:**
- `anomaly_thresholds["low"]`: 0.25 (25th percentile)
- `anomaly_thresholds["moderate"]`: 0.75 (75th percentile)
- `anomaly_thresholds["high"]`: 0.95 (95th percentile)

---

## Known Limitations

1. **Synthetic Data Limitation** — Models trained on synthetic degradations may not generalize to real defects
2. **ACCEPTABLE-Only Training** — Deep model may miss anomalies not present in training
3. **Feature Engineering Dependency** — Baseline relies on Phase 1 features which may not capture all defects
4. **Computational Cost** — Deep model inference is slower than baseline (100ms vs 1ms)
5. **Memory Requirements** — GPU needed for fast deep inference on large batches

---

## Future Enhancements

1. **Domain Adaptation** — Fine-tune on real images
2. **Active Learning** — Collect hard negatives from failures
3. **Multi-Task Learning** — Predict label AND individual issue type
4. **Knowledge Distillation** — Compress deep model into baseline
5. **Uncertainty Quantification** — Bayesian uncertainty estimates
6. **Continual Learning** — Adapt to new defect types online

---

## References

1. "Anomaly Detection with Robust Deep Autoencoders"
2. "PatchCore: Toward Total Recall in Industrial Anomaly Detection"
3. scikit-learn Random Forest Documentation
4. PyTorch Autoencoder Tutorial
5. "Ensemble Methods: Foundations and Algorithms"

---

**Next Phase:** Phase 4 — Evaluation (metrics, confusion matrix, failure analysis)
