# Design Principles — ClarityAI

This document outlines the core design principles that guide ClarityAI development across all layers (Backend, Frontend, ML Engine, Database). These principles ensure consistency, maintainability, scalability, and quality throughout the project lifecycle.

---

## 1. Separation of Concerns (SoC)

**Definition:** Each component/module should have a single, well-defined responsibility. Different concerns should be isolated from each other.

### Application in ClarityAI:

**Backend:**
- API routes (controllers) only handle HTTP protocol concerns
- Services handle business logic (feature extraction, model inference)
- Models/ORM handle database concerns
- Utils handle cross-cutting concerns (logging, validation)

**Example:**
```python
# ✅ GOOD: Separated concerns
# api/analyze.py
@router.post("/analyze")
async def analyze_image(file: UploadFile):
    result = await inference_service.analyze(file)
    return result

# services/inference.py
class InferenceService:
    async def analyze(self, file):
        features = feature_extraction.extract(file)
        prediction = model.predict(features)
        return prediction
```

**Frontend:**
- Components render UI only
- Services handle API communication
- State management handles app state
- Utils handle common functions

**ML Engine:**
- Feature extraction module (independent of model)
- Model inference module (independent of features)
- Scoring/fusion module (combines both)

---

## 2. Modularity & Reusability (DRY — Don't Repeat Yourself)

**Definition:** Code should be organized into reusable modules. Common functionality should not be duplicated.

### Application in ClarityAI:

**Backend:**
```
backend/
├── app/
│   ├── api/              # All route handlers
│   ├── services/         # Business logic (reusable)
│   ├── models/           # Database models
│   ├── schemas/          # Request/Response validation
│   ├── core/             # Config, logging, constants
│   └── utils/            # Shared utilities
```

**Frontend:**
```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   ├── services/         # API client (single source)
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Helper functions
│   └── pages/            # Page-level components
```

**ML Engine:**
```
ml/
├── feature_extraction.py # Feature extraction (reusable)
├── model.py              # Model wrapper (reusable)
├── scoring.py            # Score fusion (reusable)
└── inference.py          # Entry point (orchestrates above)
```

---

## 3. Single Responsibility Principle (SRP)

**Definition:** A class/function should have one reason to change. It should do one thing and do it well.

### Application in ClarityAI:

**Backend — Bad:**
```python
# ❌ BAD: Multiple responsibilities
class ImageAnalyzer:
    def validate_file(self): pass
    def extract_features(self): pass
    def predict_quality(self): pass
    def save_to_db(self): pass
    def send_email_notification(self): pass
```

**Backend — Good:**
```python
# ✅ GOOD: Each class has one responsibility
class FileValidator:
    def validate(self): pass

class FeatureExtractor:
    def extract(self): pass

class QualityPredictor:
    def predict(self): pass

class AnalysisRepository:
    def save(self): pass

class NotificationService:
    def notify(self): pass
```

---

## 4. Interpretability & Explainability (ClarityAI-Specific)

**Definition:** Every quality decision must be explainable. The "why" is as important as the "what."

### Application in ClarityAI:

**Backend:**
```json
{
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    {
      "type": "noise",
      "severity": "low",
      "confidence": 0.71,
      "explanation": "Wavelet-based noise score: 23.5 (threshold: 30)"
    }
  ],
  "feature_stats": {
    "sharpness": 145.3,
    "brightness": 128,
    "contrast": 45.2
  }
}
```

**ML Engine:**
- Every feature extraction function returns not just the score, but the "why"
- Model predictions include confidence scores
- Grad-CAM/saliency maps explain defect localization

---

## 5. Reproducibility & Transparency

**Definition:** Every process (data generation, model training, deployment) should be documented and repeatable. No magic numbers or hidden assumptions.

### Application in ClarityAI:

**Data Generation:**
- Document exact degradation parameters (blur kernel size, noise sigma, JPEG quality, etc.)
- Version dataset generation scripts
- Log random seeds for reproducibility

**Model Training:**
- Document hyperparameters, learning rates, epochs
- Save training logs and metrics
- Version model checkpoints with metadata

**Deployment:**
- Use version tags for Docker images (`v1.0.0` not `latest`)
- Environment variables documented in `.env.example`
- Database migrations tracked in Git

**Example:**
```python
# ml/dataset_generation.py
"""
Dataset generation with documented parameters.
Ensures reproducibility across runs.
"""
BLUR_KERNEL_SIZES = [3, 5, 7, 9, 11]      # Documented
NOISE_SIGMA_VALUES = [10, 20, 30, 40]     # Documented
RANDOM_SEED = 42                           # Fixed for reproducibility
JPEG_QUALITY_VALUES = [30, 50, 70, 90]    # Documented

# Generate dataset with exact parameters
def generate_degraded_images(source_dir, output_dir, seed=RANDOM_SEED):
    random.seed(seed)
    # ... generation code ...
```

---

## 6. Independency from External APIs (ClarityAI-Specific)

**Definition:** The system must not depend on third-party AI/vision APIs. All core functionality must run self-hosted.

### Application in ClarityAI:

**Backend:**
- ❌ DO NOT use: OpenAI Vision, Google Vision, AWS Rekognition
- ✅ DO use: Local PyTorch/TensorFlow models, OpenCV, scikit-learn

**Frontend:**
- All API calls go to our backend (`/api/v1/analyze`)
- Backend handles all ML inference (no direct calls to external services)

**Deployment:**
- Dockerfile includes all model weights
- No API keys or external service credentials needed
- Fully self-contained Docker Compose setup

---

## 7. Open/Closed Principle (OCP)

**Definition:** Software should be open for extension but closed for modification. Add new features without changing existing code.

### Application in ClarityAI:

**Backend — Extensibility:**
```python
# ✅ GOOD: Adding new issue detectors without modifying existing code
class IssueDetector(ABC):
    @abstractmethod
    def detect(self, image) -> Issue: pass

class BlurDetector(IssueDetector):
    def detect(self, image): pass

class NoiseDetector(IssueDetector):
    def detect(self, image): pass

class CustomDefectDetector(IssueDetector):  # New detector, no changes to existing
    def detect(self, image): pass

# Factory pattern for extensibility
detectors = [BlurDetector(), NoiseDetector(), CustomDefectDetector()]
```

**ML Engine:**
```python
# New feature extractors can be added without modifying existing ones
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image): pass

class SharpnessExtractor(FeatureExtractor): pass
class ExposureExtractor(FeatureExtractor): pass
class TextureExtractor(FeatureExtractor): pass  # New extractor
```

---

## 8. Dependency Inversion Principle (DIP)

**Definition:** Depend on abstractions, not concretions. High-level modules should not depend on low-level modules.

### Application in ClarityAI:

**Backend:**
```python
# ❌ BAD: Tight coupling to specific model class
class AnalysisService:
    def __init__(self):
        self.model = PyTorchModel()  # Direct dependency

# ✅ GOOD: Depend on abstract interface
class AnalysisService:
    def __init__(self, model: ModelInterface):
        self.model = model  # Can be any model implementation

# Injected at runtime (Flask/FastAPI patterns)
service = AnalysisService(model=PyTorchModel())
# Or swap for different implementation
service = AnalysisService(model=TensorFlowModel())
```

---

## 9. Consistency & Conventions

**Definition:** Code style, naming, structure should be consistent across the project.

### Application in ClarityAI:

**Backend:**
- Use `snake_case` for variables/functions
- API endpoints follow RESTful conventions
- Error responses have consistent format
- All database timestamps in UTC

**Frontend:**
- Use `camelCase` for variables/functions
- Component names use `PascalCase`
- Consistent file structure (component + styles + tests together)
- Consistent error handling patterns

**ML Engine:**
- Consistent parameter naming across scripts
- Consistent return value formats
- Docstrings follow NumPy/Google style

---

## 10. Scalability & Performance

**Definition:** System should handle growth (more images, more users) without architectural changes.

### Application in ClarityAI:

**Backend:**
- Stateless API design (can run multiple instances)
- Database queries optimized (indexing on frequent searches)
- Model loading at startup (not per-request)
- Async/await for I/O operations

**Frontend:**
- Lazy load components (code splitting)
- Efficient state management (prevent unnecessary re-renders)
- Pagination for history view

**Database:**
- Proper indexing on `uploaded_at`, `quality_label`
- Archival strategy for old analyses
- Connection pooling for concurrent requests

---

## 11. Security & Input Validation

**Definition:** Never trust user input. Validate and sanitize everything.

### Application in ClarityAI:

**Backend:**
```python
# ✅ GOOD: Comprehensive validation
@router.post("/analyze")
async def analyze_image(file: UploadFile):
    # File type validation
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Invalid file type")
    
    # File size validation
    content = await file.read()
    if len(content) > 50_000_000:  # 50MB limit
        raise HTTPException(413, "File too large")
    
    # Try to open image (detect corruption)
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except Exception:
        raise HTTPException(400, "Corrupted image")
```

**Frontend:**
- Validate file type before upload
- Show clear error messages for invalid inputs
- XSS protection (sanitize API responses)

**Database:**
- Use parameterized queries (ORM handles this)
- Least-privilege database user

---

## 12. Error Handling & Graceful Degradation

**Definition:** Handle errors explicitly. Fail gracefully with meaningful messages.

### Application in ClarityAI:

**Backend:**
```python
# ✅ GOOD: Explicit error handling
try:
    result = model.predict(features)
except ModelInferenceError as e:
    logger.error(f"Model inference failed: {e}")
    raise HTTPException(500, "Analysis failed")
except FileCorruptionError as e:
    logger.warning(f"Corrupted image: {e}")
    raise HTTPException(400, "Image is corrupted")
```

**Response Format:**
```json
{
  "error": "Image is corrupted",
  "error_code": "CORRUPTED_IMAGE",
  "details": "PNG file header validation failed"
}
```

---

## 13. GoF Design Patterns in ClarityAI

Recommended Gang of Four patterns for this project:

### **Creational Patterns:**

**Singleton** — Model Loader
```python
class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = torch.load("model.pt")
        return cls._instance

# Single instance used throughout app
model_loader = ModelLoader()
```

**Factory** — Issue Detector Factory
```python
class IssueDetectorFactory:
    @staticmethod
    def create_detector(issue_type: str) -> IssueDetector:
        detectors = {
            "blur": BlurDetector(),
            "noise": NoiseDetector(),
            "exposure": ExposureDetector(),
        }
        return detectors.get(issue_type)
```

### **Structural Patterns:**

**Decorator** — Feature Enhancement
```python
def caching_decorator(func):
    cache = {}
    def wrapper(image):
        key = hash(image)
        if key not in cache:
            cache[key] = func(image)
        return cache[key]
    return wrapper

@caching_decorator
def extract_features(image):
    # extraction logic
    pass
```

**Facade** — Unified Analysis Interface
```python
class AnalysisFacade:
    """Simplifies complex ML pipeline for API consumers"""
    def analyze(self, image):
        features = self._extract_features(image)
        prediction = self._predict(features)
        result = self._fuse_scores(prediction, features)
        return result
```

### **Behavioral Patterns:**

**Strategy** — Multiple Analysis Strategies
```python
class AnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, image): pass

class ClassicalMLStrategy(AnalysisStrategy):
    def analyze(self, image): pass

class DeepLearningStrategy(AnalysisStrategy):
    def analyze(self, image): pass

class HybridStrategy(AnalysisStrategy):
    def analyze(self, image):
        # Combines both strategies
        pass
```

**Observer** — Frontend Event Listening
```javascript
// Frontend observes API events
class AnalysisObserver {
    update(result) {
        // Frontend updates UI with result
    }
}

// API notifies frontend
api.analyze(image).then(result => {
    observer.update(result);
});
```

---

## 14. Testing & Quality Assurance

**Definition:** Code should be testable. Automated tests ensure quality.

### Application in ClarityAI:

**Backend:**
```python
# Unit tests for feature extraction
def test_sharpness_extraction():
    sharp_image = load_test_image("sharp.jpg")
    blur_image = load_test_image("blur.jpg")
    
    sharp_score = extract_sharpness(sharp_image)
    blur_score = extract_sharpness(blur_image)
    
    assert sharp_score > blur_score

# Integration tests for API
def test_analyze_endpoint():
    response = client.post("/api/v1/analyze", files={"file": test_image})
    assert response.status_code == 200
    assert "quality_score" in response.json()
```

**Frontend:**
- Component unit tests
- Integration tests for API flow
- End-to-end tests for critical paths

---

## 15. Documentation & Communication

**Definition:** Code should be self-documenting. Complex logic needs comments. Architecture should be explained.

### Application in ClarityAI:

**Python Docstrings:**
```python
def extract_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    
    Args:
        image: Input image as numpy array (H×W×C)
    
    Returns:
        float: Sharpness score (higher = sharper)
    
    Note:
        Based on "A No-Reference Image Sharpness Metric..."
        Threshold for acceptable sharpness: 100.0
    """
    pass
```

**README & Architecture:**
- High-level system overview
- Setup instructions
- Contributing guidelines
- Deployment steps

---

## Summary Table

| Principle | Purpose | Key Benefit |
|---|---|---|
| Separation of Concerns | Isolate different responsibilities | Easy to understand, test, modify |
| Modularity & DRY | Reusable code components | Reduce duplication, faster development |
| SRP | One responsibility per class | Easier to debug and maintain |
| Interpretability | Explain quality decisions | Build user trust |
| Reproducibility | Document all processes | Enable collaboration and auditing |
| Independence | No external API dependency | Cost savings, privacy, control |
| OCP | Extend without modifying | Add features safely |
| DIP | Depend on abstractions | Flexible, testable code |
| Consistency | Uniform conventions | Team alignment, readability |
| Scalability | Handle growth | Future-proof architecture |
| Security | Validate all inputs | Prevent attacks |
| Error Handling | Fail gracefully | Better user experience |
| GoF Patterns | Proven design solutions | Industry-standard approaches |
| Testing | Automated quality checks | Catch bugs early |
| Documentation | Explain intent | Onboard new developers |

---

## How to Apply These Principles

1. **Before coding:** Review relevant principles for the task
2. **During coding:** Follow the patterns and conventions
3. **Code review:** Check if code adheres to principles
4. **Refactoring:** Fix violations when discovered
5. **Documentation:** Explain principle-based decisions in comments

---

## References

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Design Patterns by Gang of Four](https://en.wikipedia.org/wiki/Design_Patterns)
- [Clean Code by Robert C. Martin](https://en.wikipedia.org/wiki/Robert_C._Martin)
- [Software Architecture Guide](https://martinfowler.com/)
