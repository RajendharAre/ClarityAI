# Implementation Plan — ClarityAI

This document breaks the project into concrete, sequential phases. Each phase produces a working, testable increment.

---

## Phase 0 — Setup & Environment ✅ COMPLETE

**Goal:** Get repo scaffolding and tooling ready.

- [x] Initialize monorepo structure: `backend/`, `frontend/`, `ml/`, `docs/`
- [x] Set up Python virtual environment; pin dependencies (`fastapi`, `uvicorn`, `opencv-python`, `torch`, `torchvision`, `scikit-learn`, `sqlalchemy`, `alembic`, `pydantic-settings`)
- [x] Initialize React app (Vite recommended for faster dev loop) in `frontend/`
- [x] Set up `.env.example` with all configurable variables
- [x] Initialize Git with sensible `.gitignore` (models, datasets, `__pycache__`, `node_modules`, `.env`)
- [x] Set up pre-commit hooks (black/ruff for Python, eslint/prettier for JS) — optional but recommended

**Status:** ✅ All Phase 0 tasks completed. Project structure initialized with comprehensive configuration files.

---

## Phase 1 — Classical Image Feature Pipeline ✅ COMPLETE

**Goal:** Get deterministic, explainable quality signals working end-to-end before adding ML complexity.

- [x] Implement sharpness detection (Variance of Laplacian via OpenCV)
- [x] Implement exposure analysis (brightness measurement via LAB color space)
- [x] Implement contrast measurement (pixel intensity std deviation)
- [x] Implement noise estimation (high-frequency residual method with Laplacian)
- [x] Implement basic corruption/truncation detection (file integrity check, PIL `verify()`, dimension sanity checks)
- [x] Write unit tests for each feature function using synthetic test images (sharp/blurry, bright/dark, noisy/clean)
- [x] Wrap all of the above into complete `ImageFeatureExtractor` with `extract_all()` method
- [x] Implement saturation and texture complexity extractors (bonus features)
- [x] Create comprehensive image validation module
- [x] Write 30+ unit tests with edge case coverage
- [x] Add GoF design patterns (Strategy, Facade)
- [x] Document with Phase 1 documentation and API references

**Deliverable:** `ml/feature_extraction.py` + `ml/image_validation.py` + `ml/tests/test_feature_extraction.py` + `ml/PHASE1_DOCUMENTATION.md`

**Status:** ✅ Complete with 6 feature extractors, validation module, comprehensive tests, and documentation.

---

## Phase 2 — Dataset Preparation ✅ COMPLETE

**Goal:** Get labeled/structured data ready for training and evaluation.

- [x] Choose a base clean-image dataset (e.g., a subset of an open dataset such as DIV2K, COCO validation images, or similar permissively licensed image collection)
- [x] Generate synthetic degradations programmatically:
  - [x] Gaussian blur (varying kernel sizes) → blur examples
  - [x] Brightness/gamma adjustment → over/under-exposure examples
  - [x] Gaussian/salt-pepper noise injection → noise examples
  - [x] Byte-level truncation / JPEG corruption → corruption examples
  - [x] Scratch and spot defects (physical defect simulation)
- [x] For "defect" examples, use or reference an established defect-detection dataset (e.g., MVTec AD) as either supplementary training/evaluation data or as inspiration for synthetic defect generation (e.g., overlaying scratches/spots)
- [x] Split into train/validation/test sets, ensuring **no leakage** (same source image should not appear in both train and test after degradation)
- [x] Document the exact generation process and parameters in `ml/PHASE2_DOCUMENTATION.md` for reproducibility

**Deliverable:** A structured `data/` directory with `train/`, `val/`, `test/` splits and a data-generation script.

**Status:** ✅ Complete with 7 degradation types, dataset generator, 26 unit tests, and comprehensive documentation.

**Implementation Details:**
- **ml/degradations.py:** 7 degradation types (Blur, Exposure, Noise, SaltPepperNoise, JPEGCompression, Scratch, Spot)
- **ml/dataset_generator.py:** Complete pipeline with no data leakage, JSONL metadata, reproducible seed
- **ml/generate_dataset.py:** Standalone script for generating sample or real datasets
- **ml/tests/test_dataset_generation.py:** 26 comprehensive unit tests (all passing)
- **ml/PHASE2_DOCUMENTATION.md:** Full documentation of degradations, parameters, usage
- **Validation:** ✅ 26/26 tests passing, ✅ sample dataset generated (30 images), ✅ no leakage confirmed

---

## Phase 3 — Model Development ⏳ PENDING

**Goal:** Build and train the learned component(s) of the pipeline.

- [ ] **Baseline model:** Train a classical ML model (e.g., Gradient Boosted Trees / Random Forest / small MLP) on the engineered features from Phase 1 to classify: acceptable / degraded / defective
- [ ] **Deep model (defect-focused):** Fine-tune a lightweight pretrained CNN (ResNet18 / MobileNetV3) as a feature extractor; implement an anomaly-scoring head (e.g., simple autoencoder reconstruction error, or a memory-bank/k-NN approach inspired by PatchCore) trained only on "normal/acceptable" images
- [ ] Implement the **score fusion** logic combining classical-feature-based predictions with the anomaly score into the final `quality_score` (0–100) and `quality_label`
- [ ] Save trained model artifacts (`.pt` / `.joblib`) with a clear versioning scheme (`model_v1.pt`)
- [ ] Write an `infer(image) -> AnalysisResult` function that wraps the entire pipeline (features → model → fusion)

**Deliverable:** Saved model weights + a working inference function callable from Python.

---

## Phase 4 — Evaluation ⏳ PENDING

**Goal:** Rigorously measure model performance (see `Evaluation.md` for full detail).

- [ ] Run inference on the held-out test set
- [ ] Compute accuracy, precision, recall, F1-score per class (acceptable/degraded/defective)
- [ ] Compute ROC-AUC for the anomaly-detection component
- [ ] Generate and save a confusion matrix
- [ ] Collect and manually review 10–20 misclassified examples; write up failure-case analysis
- [ ] Generate Grad-CAM visualizations for a sample of defect predictions

**Deliverable:** `ml/evaluation_report.md` + saved plots (`confusion_matrix.png`, `roc_curve.png`, sample Grad-CAMs)

---

## Phase 5 — Backend API ⏳ PENDING

**Goal:** Expose the ML pipeline via a robust REST API.

- [ ] Set up FastAPI app skeleton with routers (`analyze`, `history`, `health`)
- [ ] Implement `POST /api/v1/analyze`:
  - File type/size validation
  - Graceful error handling for corrupt/unreadable files
  - Calls `infer()` from Phase 3
  - Persists result to DB
  - Returns structured JSON response
- [ ] Implement `GET /api/v1/history` and `GET /api/v1/history/{id}`
- [ ] Set up SQLAlchemy models + Alembic migration for the `analyses` table
- [ ] Implement `GET /health` (checks DB connection + model loaded flag)
- [ ] Add structured logging and consistent error responses (4xx/5xx with clear messages)
- [ ] Write API integration tests (pytest + httpx/TestClient)

**Deliverable:** Running FastAPI server, testable via Swagger UI (`/docs`) and Postman/curl.

---

## Phase 6 — Frontend ⏳ PENDING

**Goal:** Build a usable interface for the whole flow.

- [ ] Build Upload page: drag-and-drop + file picker, image preview
- [ ] Wire upload to `POST /analyze`, handle loading/error/success states
- [ ] Build Result view: quality score gauge/badge, label, per-issue breakdown cards (type/severity/confidence)
- [ ] If Grad-CAM/heatmap available, overlay it on the image
- [ ] Build History page: paginated table/list of past analyses, click to view detail
- [ ] Responsive layout (mobile/tablet/desktop breakpoints)
- [ ] Basic styling pass (Tailwind CSS or CSS Modules recommended for speed)

**Deliverable:** Working React app that talks to the live backend.

---

## Phase 7 — Containerization & Deployment ⏳ PENDING

**Goal:** Make the whole system runnable anywhere with one command (see `DeploymentPlan.md`).

- [ ] Write `Dockerfile` for backend (multi-stage build: install deps → copy app → copy model weights)
- [ ] Write `Dockerfile` for frontend (build React app → serve via nginx)
- [ ] Write `docker-compose.yml` wiring frontend, backend, and Postgres together
- [ ] Externalize all config via environment variables
- [ ] Verify `/health` endpoint works inside container
- [ ] Test full teardown/rebuild (`docker compose down -v && docker compose up --build`) on a clean checkout
- [ ] (Optional) Deploy to a free-tier cloud host (Render/Railway/Fly.io) for a live demo URL

**Deliverable:** `docker-compose up` brings up the entire working application from a fresh clone.

---

## Phase 8 — Documentation & Polish ⏳ PENDING

- [ ] Finalize `README.md` with setup, run, and usage instructions
- [ ] Write API documentation (or rely on FastAPI's auto-generated OpenAPI docs, linked in README)
- [ ] Include sample images demonstrating each quality condition in `samples/`
- [ ] Write final evaluation summary and known limitations
- [ ] Record a short demo (screen recording or GIF) — highly recommended for portfolio value

---

## Project Progress Summary

### Completion Status

| Phase | Status | Completion % |
|-------|--------|-------------|
| Phase 0 | ✅ **COMPLETE** | 100% |
| Phase 1 | ✅ **COMPLETE** | 100% |
| Phase 2 | ⏳ **IN PROGRESS** | 0% |
| Phase 3 | ⏳ **PENDING** | 0% |
| Phase 4 | ⏳ **PENDING** | 0% |
| Phase 5 | ⏳ **PENDING** | 0% |
| Phase 6 | ⏳ **PENDING** | 0% |
| Phase 7 | ⏳ **PENDING** | 0% |
| Phase 8 | ⏳ **PENDING** | 0% |
| **TOTAL** | **⏳ 22% Complete** | **22%** |

### Suggested Timeline (if compressed)

| Phase | Focus | Priority | Status |
|-------|-------|----------|--------|
| 0–1 | Setup + classical features | Must-have | ✅ Complete |
| 2–3 | Data + model | Must-have | ⏳ In Progress |
| 4 | Evaluation | Must-have | ⏳ Pending |
| 5–6 | Backend + Frontend | Must-have | ⏳ Pending |
| 7 | Docker deployment | Must-have | ⏳ Pending |
| 8 | Docs/polish | Must-have (even if brief) | ⏳ Pending |

### What's Been Completed ✅

**Phase 0 — Project Setup:**
- Full directory structure (backend, frontend, ML, docs)
- Python requirements.txt with all dependencies
- Environment configuration (.env.example)
- Git initialization with .gitignore
- Pre-commit hooks configuration
- README.md with comprehensive documentation
- Design Principles document (15 core principles)

**Phase 1 — Classical Features:**
- 6 feature extractors (sharpness, brightness, contrast, noise, saturation, texture)
- Image validation module with corruption detection
- 30+ unit tests covering all features
- GoF design patterns (Strategy, Facade)
- Comprehensive documentation with references and thresholds

### What's Next ⏳

**Phase 2 — Dataset Preparation:**
- Dataset generation script with synthetic degradations
- Train/val/test split creation
- Documentation of generation process

Optional/bonus items (batch analysis, heatmap localization, CI/CD, calibration, automated tests beyond the basics) should only be tackled after all "must-have" phases are functionally complete.