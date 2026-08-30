# ClarityAI — AI-Powered Image Quality & Defect Detection

**A full-stack AI application that evaluates image visual quality and detects defects without external APIs.**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (for deployment)

### Development Setup

#### 1. Clone & Setup Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Setup Frontend
```bash
cd frontend
npm install
```

#### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

#### 4. Run Backend
```bash
cd backend
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

#### 5. Run Frontend
```bash
cd frontend
npm run dev
# Frontend available at http://localhost:3000
```

---

## 📂 Project Structure

```
ClarityAI/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/               # REST API endpoints
│   │   ├── services/          # Business logic
│   │   ├── models/            # Database models
│   │   ├── schemas/           # Request/response schemas
│   │   ├── core/              # Configuration & setup
│   │   ├── db/                # Database utilities
│   │   └── utils/             # Helper functions
│   ├── tests/                 # Unit & integration tests
│   └── requirements.txt
│
├── frontend/                   # React SPA frontend
│   ├── src/
│   │   ├── components/        # Reusable React components
│   │   ├── pages/            # Page-level components
│   │   ├── services/         # API client
│   │   ├── hooks/            # Custom React hooks
│   │   ├── utils/            # Utilities
│   │   └── styles/           # CSS/styling
│   ├── public/
│   └── package.json
│
├── ml/                         # ML/CV pipeline
│   ├── data/                  # Training/validation/test data
│   ├── training/              # Training scripts
│   ├── evaluation/            # Evaluation & metrics
│   └── notebooks/             # Jupyter notebooks
│
├── models/                     # Trained model weights
├── docs/                       # Documentation
├── samples/                    # Sample images
│
├── DesignPrinciples.md        # Design principles & patterns
├── ProblemStatement.md         # Problem definition
├── Architecture.md            # System architecture
├── Implementation.md          # Implementation plan
├── Evaluation.md              # Evaluation methodology
├── DeploymentPlan.md          # Deployment guide
├── UnderstandProject.md       # Project overview
│
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── docker-compose.yml         # Docker Compose configuration
```

---

## 🏗️ Architecture Overview

```
Frontend (React)
       ↕ HTTPS/REST
Backend (FastAPI)
   ├── Feature Extraction (OpenCV)
   ├── Model Inference (PyTorch)
   └── Database (PostgreSQL/SQLite)
```

**Key Components:**
- **Frontend:** Image upload, results display, history
- **Backend:** REST API, validation, orchestration
- **ML Engine:** Feature extraction + learned models
- **Database:** Analysis history & metadata

---

## 🎯 Detection Capabilities

The system identifies:
- ✓ Blur / insufficient sharpness
- ✓ Underexposure (too dark)
- ✓ Overexposure (too bright)
- ✓ Image noise
- ✓ Image corruption
- ✓ Potential visual defects

**Output:** Quality score (0-100), label (ACCEPTABLE/DEGRADED/DEFECTIVE), per-issue breakdown with confidence.

---

## 📊 Implementation Phases

| Phase | Task | Status |
|-------|------|--------|
| 0 | Project setup & environment | ✅ In Progress |
| 1 | Classical feature extraction | ⏳ Pending |
| 2 | Dataset preparation | ⏳ Pending |
| 3 | Model development | ⏳ Pending |
| 4 | Evaluation & metrics | ⏳ Pending |
| 5 | Backend API | ⏳ Pending |
| 6 | Frontend UI | ⏳ Pending |
| 7 | Docker & deployment | ⏳ Pending |
| 8 | Documentation & polish | ⏳ Pending |

---

## 🧠 Design Principles

This project follows **15 core design principles**:
1. Separation of Concerns
2. Modularity & Reusability (DRY)
3. Single Responsibility (SRP)
4. Interpretability & Explainability
5. Reproducibility & Transparency
6. Independence from External APIs
7. Open/Closed Principle (OCP)
8. Dependency Inversion (DIP)
9. Consistency & Conventions
10. Scalability & Performance
11. Security & Input Validation
12. Error Handling & Graceful Degradation
13. GoF Design Patterns
14. Testing & Quality Assurance
15. Documentation & Communication

See [DesignPrinciples.md](DesignPrinciples.md) for detailed guidance.

---

## 🐳 Docker Deployment

### Local Development
```bash
docker compose up --build
# Services available:
# - Frontend: http://localhost
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Teardown
```bash
docker compose down -v
```

---

## 📚 Documentation

- [Problem Statement](ProblemStatement.md) — Why this project exists
- [Architecture](Architecture.md) — System design & components
- [Implementation Plan](Implementation.md) — Phase-by-phase roadmap
- [Design Principles](DesignPrinciples.md) — Code quality guidelines
- [Evaluation Methodology](Evaluation.md) — Metrics & testing approach
- [Deployment Guide](DeploymentPlan.md) — Production deployment

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

---

## 🔐 Security

- ✓ No hardcoded secrets (environment variables only)
- ✓ Input validation & file size limits
- ✓ CORS properly configured
- ✓ Database user has minimal privileges
- ✓ File upload type/size validation

---

## 📈 Performance

- Model loaded at startup (not per-request)
- Async API for concurrent requests
- Database indexing on frequent queries
- Stateless backend (scales horizontally)
- Frontend lazy loading & code splitting

---

## 🤝 Contributing

1. Read [DesignPrinciples.md](DesignPrinciples.md)
2. Follow code style (Black for Python, Prettier for JS)
3. Write tests for new features
4. Document your changes
5. Submit a pull request

---

## 📄 License

MIT License — See LICENSE file for details

---

## 📞 Support

For questions or issues:
- Check the [docs/](docs/) folder
- Review [DesignPrinciples.md](DesignPrinciples.md) for best practices
- Check API documentation at `/docs` endpoint

---

**Built with ❤️ for quality image analysis**
