# ClarityAI — AI-Powered Image Quality & Defect Detection

A full-stack AI application that accepts an image and automatically evaluates its visual quality — identifying common image-quality issues and classifying the image as **acceptable**, **degraded**, or **potentially defective**.

---

## 🎯 Project Overview

**Domain:** Computer Vision, Machine Learning / Deep Learning, Full-Stack Engineering
**Primary Focus:** Building a deployable AI system that performs meaningful image-quality analysis without relying on external AI/vision APIs.

---

## 🧠 Detection Capabilities

The system detects the following image-quality issues:

- Blur / insufficient sharpness
- Underexposure
- Overexposure
- Image noise
- Image corruption or severe degradation
- Potential visual defects

Additional quality issues may be added where technically justified.

---

## 🔬 AI / Computer Vision Approach

The core decision-making component is AI-based rather than purely rule-based computer vision. Possible approaches include:

- Classical machine learning using engineered image features
- A lightweight deep-learning model or transfer-learning approach (PyTorch / TensorFlow)
- A hybrid approach combining image-quality features with a learned model
- An anomaly-detection formulation

The chosen approach, along with data preparation, training/model acquisition, and evaluation methodology, is documented in the `/docs` folder.

---

## 🖼️ Image Analysis

The system extracts meaningful visual features such as:

- Sharpness
- Brightness / Exposure
- Contrast
- Noise levels
- Texture
- Saturation

These features feed into the quality-scoring model.

---

## ⚙️ Backend

- REST API for image upload and analysis
- Input validation and graceful handling of invalid/unreadable images
- Structured JSON analysis responses
- Persistent storage of analysis results (SQLite / PostgreSQL)
- Endpoint to retrieve historical analysis results
- Proper error handling and HTTP status codes

**Example Response:**
```json
{
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "issues": [
    { "type": "noise", "severity": "low", "confidence": 0.71 }
  ]
}
```

---

## 💻 Frontend

- Web interface for uploading images and viewing results
- Displays uploaded image alongside quality assessment
- Shows overall quality score, detected issues, severity, and confidence
- History view for previous analyses
- Handles loading, success, and error states
- Responsive, functional-first UI (React / Vue / plain HTML-CSS-JS)

---

## 📊 Dataset & Training

- Uses a public dataset, custom dataset, or synthetically generated image degradations from clean source images
- Evaluation performed on unseen data to demonstrate generalization
- Data generation/collection methodology documented for reproducibility

---

## 📈 Evaluation

Model performance is assessed using metrics appropriate to the chosen approach, such as:

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix
- Anomaly-detection metrics
- Regression error metrics (if applicable)

Includes discussion of failure cases, limitations, and uncertain predictions.

---

## 🔍 Explainability

Quality decisions are made interpretable through techniques such as:

- Interpretable image statistics
- Feature importance
- Confidence scores
- Saliency maps / Grad-CAM

---

## 🚀 Deployment

- Fully containerized using Docker / Docker Compose
- Runs outside the development environment with clear setup instructions
- Environment-variable based configuration
- Health/status endpoint for service monitoring
- Documented model loading and inference pipeline

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| AI/ML | PyTorch / TensorFlow, OpenCV, scikit-learn |
| Backend | Python (FastAPI / Flask) |
| Database | SQLite / PostgreSQL |
| Frontend | React.js |
| Deployment | Docker, Docker Compose |

---

## 📂 Project Structure

```
ClarityAI/
├── backend/
│   ├── app/
│   ├── models/
│   ├── api/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
├── ml/
│   ├── training/
│   ├── evaluation/
│   └── notebooks/
├── docker-compose.yml
├── docs/
└── README.md
```

---

## 📌 Skills Demonstrated

**AI/ML & Computer Vision:** Image Processing, Feature Engineering, Transfer Learning, Anomaly Detection, Model Evaluation, Explainable AI

**Backend Engineering:** REST API Design, Database Modeling, Error Handling, File Validation

**Frontend Engineering:** Responsive UI, API Integration, State Management

**DevOps:** Docker, Environment Configuration, Health Checks, CI/CD (optional)

---

## ✨ Optional Enhancements

- Batch image analysis
- Quality heatmaps / defect localization
- Confidence calibration & uncertainty estimation
- Model versioning
- Automated tests
- Performance optimization for concurrent requests
- CI/CD pipeline
- Logging & monitoring

---
MIT Licence 

-> Need to be include here.