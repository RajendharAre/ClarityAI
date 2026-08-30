# Problem Statement — ClarityAI

## 1. Background

Images captured in real-world conditions — by phone cameras, industrial inspection cameras, drones, or web uploads — frequently suffer from quality issues that are not immediately obvious to an untrained eye but significantly affect downstream usability. A blurry product photo, an overexposed inspection image, or a corrupted upload can silently degrade the quality of any pipeline that depends on that image (e-commerce catalogs, quality control lines, medical imaging intake, user-generated content platforms, etc.).

Manually reviewing every image for quality issues does not scale. There is a need for an **automated, AI-driven system** that can look at an image, judge its visual quality, and clearly explain *why* it passed or failed — without depending on a third-party AI/vision API (for cost, privacy, and reproducibility reasons).

## 2. Problem Definition

**Build a full-stack application that accepts an image and automatically evaluates its visual quality**, classifying it as:

- **ACCEPTABLE** — no significant quality issues
- **DEGRADED** — one or more quality issues present but the image is still usable
- **DEFECTIVE** — the image has severe quality problems or visible defects that make it unsuitable for use

The system must go beyond simple rule-based image processing and include a genuine **machine learning / deep learning decision component** — meaning the final quality judgment (or a meaningful part of it) is learned from data, not just hard-coded thresholds.

## 3. Issues to Detect

| Issue | Description |
|---|---|
| **Blur / low sharpness** | Image lacks fine detail/edge definition (e.g., out-of-focus, motion blur) |
| **Underexposure** | Image is too dark, losing detail in shadows |
| **Overexposure** | Image is too bright, losing detail in highlights |
| **Noise** | Visible sensor noise / graininess degrading clarity |
| **Corruption / severe degradation** | File-level corruption, artifacts, truncated data, compression damage |
| **Potential visual defect** | Localized anomalies not explained by the above (scratches, spots, structural damage) |

## 4. Why This Is a Non-Trivial ML Problem

- **No single visual cue determines quality** — sharpness, exposure, noise, and defects interact (e.g., a slightly noisy but sharp image may be acceptable, while a sharp image with a structural defect is not).
- **Labeled defect data is scarce** — in most real-world settings, "bad" examples are rare and diverse, which naturally pushes toward **anomaly-detection style formulations** (learn what "good" looks like, flag deviations) rather than plain supervised classification.
- **Interpretability matters** — a binary pass/fail is not enough; the system must explain *what* is wrong, *how confident* it is, and ideally *where* in the image the problem is.

## 5. Goals

1. Accept an arbitrary uploaded image via a web interface and a REST API.
2. Extract meaningful image-quality features (sharpness, brightness, contrast, noise, texture).
3. Feed these features (or the raw image) into a trained ML/DL model that outputs a quality score and issue breakdown.
4. Persist every analysis so results can be reviewed later (audit trail / history).
5. Present results in a clear, understandable UI, including confidence and severity per detected issue.
6. Package the whole system so it can be deployed and run independently of the original development machine.

## 6. Non-Goals (Out of Scope for v1)

- Real-time video quality analysis (only static images for v1).
- Multi-language OCR / content-based analysis (this project is about *visual quality*, not *content*).
- Training a foundation-scale model from scratch — transfer learning / lightweight architectures are preferred.
- Reliance on any external paid AI/vision API (OpenAI Vision, Google Vision API, AWS Rekognition, etc.) — everything must run locally/self-hosted.

## 7. Success Criteria

- The system correctly separates clearly-good and clearly-bad images with strong evaluation metrics (see `Evaluation.md`).
- The application runs end-to-end (upload → analyze → store → retrieve) with no manual intervention.
- The full stack can be started with a single `docker-compose up` on a fresh machine.
- Documentation is sufficient for a new developer to understand, run, and extend the system.

## 8. Reference Problem Category

This problem is closely related to established computer vision research areas:

- **No-Reference Image Quality Assessment (NR-IQA)** — metrics like BRISQUE and NIQE assess quality without needing a "clean" reference image, which fits this use case since uploaded images have no ground-truth clean version.
- **Unsupervised / Semi-Supervised Anomaly Detection** — datasets and methods such as MVTec AD and PatchCore-style approaches are designed exactly for the "learn what normal looks like, flag what deviates" pattern relevant to defect detection.

These bodies of work directly inform the modeling approach documented in `Architecture.md` and `ImplementationPlan.md`.