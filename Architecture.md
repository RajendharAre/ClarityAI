# Architecture — ClarityAI

## 1. High-Level System Overview

```
┌─────────────────┐        HTTPS/REST        ┌──────────────────────┐
│                  │ ───────────────────────▶ │                      │
│   Frontend       │                           │   Backend (FastAPI)  │
│   (React SPA)    │ ◀─────────────────────── │                      │
│                  │        JSON responses     └──────────┬───────────┘
└─────────────────┘                                        │
                                                            │
                                     ┌──────────────────────┼───────────────────────┐
                                     │                      │                       │
                              ┌──────▼──────┐      ┌────────▼────────┐     ┌────────▼────────┐
                              │  Feature     │      │  ML/DL Model     │     │   Database       │
                              │  Extraction  │      │  Inference       │     │  (PostgreSQL /   │
                              │  (OpenCV)    │      │  (PyTorch)       │     │   SQLite)         │
                              └─────────────┘      └──────────────────┘     └──────────────────┘
```

The system is composed of four logical layers, containerized independently but orchestrated together via Docker Compose:

1. **Frontend** — user-facing web application
2. **Backend / API** — orchestration, validation, persistence
3. **ML/CV Engine** — feature extraction + model inference
4. **Data Layer** — relational database for analysis history

## 2. Component Breakdown

### 2.1 Frontend (React)

- **Upload View** — drag-and-drop or file-picker image upload, image preview
- **Result View** — displays quality score, label (ACCEPTABLE / DEGRADED / DEFECTIVE), per-issue cards (type, severity, confidence), and any visual explanation (heatmap overlay)
- **History View** — paginated list of past analyses, filterable by label/date
- **State Management** — React state/hooks (or Redux/Zustand if complexity grows) for upload/loading/error states
- **API Client** — thin `services/api.js` wrapping `fetch`/`axios` calls to the backend

### 2.2 Backend (FastAPI)

Chosen for its native async support, automatic OpenAPI docs, and Pydantic-based validation — all useful for an image-upload + ML-inference service.

**Key modules:**

```
backend/
├── app/
│   ├── main.py              # FastAPI app init, routers, middleware
│   ├── api/
│   │   ├── analyze.py       # POST /analyze
│   │   ├── history.py       # GET /history, GET /history/{id}
│   │   └── health.py        # GET /health
│   ├── core/
│   │   ├── config.py        # env-based settings (Pydantic BaseSettings)
│   │   └── logging.py
│   ├── services/
│   │   ├── feature_extraction.py   # OpenCV-based feature engineering
│   │   ├── model_inference.py      # loads model, runs prediction
│   │   └── scoring.py               # combines features + model output → final response
│   ├── models/               # SQLAlchemy ORM models
│   │   └── analysis.py
│   ├── schemas/              # Pydantic request/response schemas
│   │   └── analysis.py
│   └── db/
│       ├── session.py
│       └── migrations/       # Alembic migrations
└── requirements.txt
```

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/analyze` | Upload an image, run analysis, persist + return result |
| `GET` | `/api/v1/history` | List previous analyses (paginated) |
| `GET` | `/api/v1/history/{id}` | Retrieve a specific past analysis |
| `GET` | `/health` | Liveness/readiness probe |

### 2.3 ML / Computer Vision Engine

Two cooperating sub-components:

**a) Classical Feature Extraction (OpenCV / scikit-image)**
Used both as direct signals and as input features to the learned model:

- **Sharpness** → Variance of Laplacian (fast, well-established no-reference blur metric)
- **Exposure** → mean/histogram analysis of luminance channel (over/under-exposure ratio)
- **Contrast** → standard deviation of pixel intensities
- **Noise** → wavelet-based or high-frequency residual estimation
- **Texture** → Local Binary Patterns / GLCM statistics

**b) Learned Model (PyTorch)**
Recommended approach for this project: a **hybrid pipeline**:

1. Engineered features (above) → feed a lightweight classical ML model (e.g., Gradient Boosted Trees / small MLP) for the "quality issue" classification (blur/exposure/noise), since these are well-defined, interpretable signals.
2. A **transfer-learning CNN backbone** (e.g., ResNet18/MobileNetV3, fine-tuned or used as frozen feature extractor) feeding an anomaly-detection head (e.g., a PatchCore-style memory bank or a small autoencoder) for the **"potential visual defect"** category — since defects are open-ended and not well captured by hand-engineered features alone.
3. A final **score fusion layer** combines both outputs into the unified `quality_score` and `quality_label`.

This hybrid design is deliberately chosen because:
- Classical features are fast, interpretable, and require no training data for blur/exposure/noise.
- Defect/anomaly detection benefits from a learned visual representation, since "defects" are too varied to hand-engineer.
- It keeps the system explainable — each issue type can point to a concrete signal.

### 2.4 Data Layer

- **Development:** SQLite (zero-config, file-based)
- **Production:** PostgreSQL (via Docker Compose service)
- **ORM:** SQLAlchemy + Alembic for migrations

**Core schema:**

```sql
Table: analyses
- id (PK, UUID)
- filename
- uploaded_at (timestamp)
- quality_score (float)
- quality_label (enum: ACCEPTABLE | DEGRADED | DEFECTIVE)
- issues (JSON array: [{type, severity, confidence}])
- image_stats (JSON: sharpness, brightness, contrast, noise, ...)
- explanation_ref (path/URI to saliency map or heatmap, nullable)
```

## 3. Request Flow (Sequence)

1. User uploads an image via the frontend.
2. Frontend sends `multipart/form-data POST /api/v1/analyze`.
3. Backend validates file type/size → rejects with `400` if invalid.
4. Backend passes the image to the **feature extraction** service.
5. Extracted features + raw image tensor go to the **model inference** service.
6. Scores from classical + learned components are fused into a final result.
7. Result is persisted to the database.
8. JSON response returned to frontend, matching the documented schema.
9. Frontend renders the result; entry becomes available in **History**.

## 4. Explainability Layer

- For CNN-based anomaly scoring: **Grad-CAM** overlay highlighting the region contributing most to an anomaly score.
- For classical features: the raw statistic itself (e.g., "Laplacian variance = 12.4, below acceptable threshold of 100") serves as a human-readable explanation.
- Confidence per issue is reported so consumers of the API can apply their own thresholds.

## 5. Deployment Architecture

```
docker-compose.yml
├── frontend   (nginx serving built React app)
├── backend    (uvicorn + FastAPI, model loaded at startup)
└── db         (postgres:16-alpine)
```

- Backend loads the trained model weights **once at startup** (not per-request) for performance.
- Environment variables (`.env`) control DB connection string, model path, CORS origins, and log level.
- `/health` endpoint checks DB connectivity and model-loaded status, used for container health checks.

## 6. Design Decisions & Trade-offs

| Decision | Reasoning |
|---|---|
| FastAPI over Flask | Native async, automatic validation & docs, better fit for ML-serving workloads |
| Hybrid classical + DL approach | Balances interpretability (classical) with flexibility for open-ended defects (DL) |
| PatchCore/anomaly-style detection for defects | Defect examples are rare/diverse — mirrors industry-standard approaches (e.g., MVTec AD benchmark methods) that train only on "normal" images |
| PostgreSQL for production | Concurrency and reliability beyond SQLite's single-writer limitation |
| Docker Compose (not Kubernetes) | Right-sized for a single-service internship-scale deployment; documented as extensible to K8s later |