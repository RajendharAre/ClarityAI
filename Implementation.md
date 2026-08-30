# Implementation Plan — ClarityAI

This document breaks the project into concrete, sequential phases. Each phase produces a working, testable increment.

---

## Phase 0 — Setup & Environment

**Goal:** Get repo scaffolding and tooling ready.

- [ ] Initialize monorepo structure: `backend/`, `frontend/`, `ml/`, `docs/`
- [ ] Set up Python virtual environment; pin dependencies (`fastapi`, `uvicorn`, `opencv-python`, `torch`, `torchvision`, `scikit-learn`, `sqlalchemy`, `alembic`, `pydantic-settings`)
- [ ] Initialize React app (Vite recommended for faster dev loop) in `frontend/`
- [ ] Set up `.env.example` with all configurable variables
- [ ] Initialize Git with sensible `.gitignore` (models, datasets, `__pycache__`, `node_modules`, `.env`)
- [ ] Set up pre-commit hooks (black/ruff for Python, eslint/prettier for JS) — optional but recommended

---

## Phase 1 — Classical Image Feature Pipeline

**Goal:** Get deterministic, explainable quality signals working end-to-end before adding ML complexity.

- [ ] Implement sharpness detection (Variance of Laplacian via OpenCV)
- [ ] Implement exposure analysis (histogram-based over/under-exposure ratio)
- [ ] Implement contrast measurement (pixel intensity std deviation)
- [ ] Implement noise estimation (wavelet-based or high-frequency residual method)
- [ ] Implement basic corruption/truncation detection (file integrity check, PIL `verify()`, dimension sanity checks)
- [ ] Write unit tests for each feature function using a small set of hand-picked sample images (sharp/blurry, bright/dark, noisy/clean)
- [ ] Wrap all of the above into a single `extract_features(image) -> dict` function

**Deliverable:** A script that takes an image path and prints a feature dictionary.

---

## Phase 2 — Dataset Preparation

**Goal:** Get labeled/structured data ready for training and evaluation.

- [ ] Choose a base clean-image dataset (e.g., a subset of an open dataset such as DIV2K, COCO validation images, or similar permissively licensed image collection)
- [ ] Generate synthetic degradations programmatically:
  - Gaussian blur (varying kernel sizes) → blur examples
  - Brightness/gamma adjustment → over/under-exposure examples
  - Gaussian/salt-pepper noise injection → noise examples
  - Byte-level truncation / JPEG corruption → corruption examples
- [ ] For "defect" examples, use or reference an established defect-detection dataset (e.g., MVTec AD) as either supplementary training/evaluation data or as inspiration for synthetic defect generation (e.g., overlaying scratches/spots)
- [ ] Split into train/validation/test sets, ensuring **no leakage** (same source image should not appear in both train and test after degradation)
- [ ] Document the exact generation process and parameters in `ml/DATASET.md` for reproducibility

**Deliverable:** A structured `data/` directory with `train/`, `val/`, `test/` splits and a data-generation script.

---

## Phase 3 — Model Development

**Goal:** Build and train the learned component(s) of the pipeline.

- [ ] **Baseline model:** Train a classical ML model (e.g., Gradient Boosted Trees / Random Forest / small MLP) on the engineered features from Phase 1 to classify: acceptable / degraded / defective
- [ ] **Deep model (defect-focused):** Fine-tune a lightweight pretrained CNN (ResNet18 / MobileNetV3) as a feature extractor; implement an anomaly-scoring head (e.g., simple autoencoder reconstruction error, or a memory-bank/k-NN approach inspired by PatchCore) trained only on "normal/acceptable" images
- [ ] Implement the **score fusion** logic combining classical-feature-based predictions with the anomaly score into the final `quality_score` (0–100) and `quality_label`
- [ ] Save trained model artifacts (`.pt` / `.joblib`) with a clear versioning scheme (`model_v1.pt`)
- [ ] Write an `infer(image) -> AnalysisResult` function that wraps the entire pipeline (features → model → fusion)

**Deliverable:** Saved model weights + a working inference function callable from Python.

---

## Phase 4 — Evaluation

**Goal:** Rigorously measure model performance (see `Evaluation.md` for full detail).

- [ ] Run inference on the held-out test set
- [ ] Compute accuracy, precision, recall, F1-score per class (acceptable/degraded/defective)
- [ ] Compute ROC-AUC for the anomaly-detection component
- [ ] Generate and save a confusion matrix
- [ ] Collect and manually review 10–20 misclassified examples; write up failure-case analysis
- [ ] Generate Grad-CAM visualizations for a sample of defect predictions

**Deliverable:** `ml/evaluation_report.md` + saved plots (`confusion_matrix.png`, `roc_curve.png`, sample Grad-CAMs)

---

## Phase 5 — Backend API

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

## Phase 6 — Frontend

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

## Phase 7 — Containerization & Deployment

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

## Phase 8 — Documentation & Polish

- [ ] Finalize `README.md` with setup, run, and usage instructions
- [ ] Write API documentation (or rely on FastAPI's auto-generated OpenAPI docs, linked in README)
- [ ] Include sample images demonstrating each quality condition in `samples/`
- [ ] Write final evaluation summary and known limitations
- [ ] Record a short demo (screen recording or GIF) — highly recommended for portfolio value

---

## Suggested Timeline (if compressed)

| Phase | Focus | Priority |
|---|---|---|
| 0–1 | Setup + classical features | Must-have |
| 2–3 | Data + model | Must-have |
| 4 | Evaluation | Must-have |
| 5–6 | Backend + Frontend | Must-have |
| 7 | Docker deployment | Must-have |
| 8 | Docs/polish | Must-have (even if brief) |

Optional/bonus items (batch analysis, heatmap localization, CI/CD, calibration, automated tests beyond the basics) should only be tackled after all "must-have" phases are functionally complete.