# Evaluation Plan — ClarityAI

## 1. Purpose

This document defines how model performance will be measured, what data will be used for evaluation, and how results will be reported and interpreted. Evaluation must demonstrate **generalization to unseen images**, not just performance on training data.

## 2. Evaluation Data

- A **held-out test set**, generated using the same synthetic-degradation pipeline as training data (Phase 2 of `ImplementationPlan.md`), but built from **source images not used in training**.
- Where possible, include a small set of **real-world images** (not synthetically degraded) to sanity-check that the model generalizes beyond synthetic artifacts — e.g., genuinely blurry photos taken on a phone, genuinely under/overexposed shots.
- If using a defect-focused dataset (e.g., MVTec-AD-style data) for the anomaly-detection component, use its standard train/test split (defect-free images for training, defect + defect-free images for testing) to remain consistent with established benchmarking practice in this field.

## 3. Metrics

### 3.1 Classification Metrics (Quality Label: Acceptable / Degraded / Defective)

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness — reported but not relied on alone (can be misleading under class imbalance) |
| **Precision** (per class) | Of images predicted as a given label, how many actually are |
| **Recall** (per class) | Of images that truly are a given label, how many were correctly identified |
| **F1-score** (per class + macro-average) | Harmonic mean of precision/recall — primary metric for imbalanced classes |
| **Confusion Matrix** | Visualizes exactly where misclassifications occur (e.g., "degraded" mistaken for "defective") |

### 3.2 Anomaly / Defect Detection Metrics

Since defective examples are expected to be rarer than normal/acceptable ones (mirroring real-world class imbalance), accuracy alone is insufficient — a model predicting "acceptable" for everything could still score high accuracy while being useless.

| Metric | Purpose |
|---|---|
| **ROC-AUC** | Measures the model's ability to separate normal vs. anomalous images independent of a fixed threshold — the standard metric used in anomaly-detection benchmarks such as MVTec AD |
| **Precision-Recall AUC** | More informative than ROC-AUC under strong class imbalance |
| **Per-category anomaly score distribution** | Visualize score separation between normal and defective samples (histogram/violin plot) |

### 3.3 Per-Issue Metrics (Blur / Exposure / Noise)

Since these are derived largely from classical, well-defined features, they are evaluated as **binary detectors** against labeled synthetic ground truth:

- Precision/Recall/F1 for "blur present" vs. "not present" (and similarly for exposure and noise), at the chosen severity threshold.
- Threshold sensitivity analysis: how metrics change as the decision threshold varies (helps choose the operating point).

## 4. Evaluation Methodology

1. Run the full inference pipeline (`infer()`) on every test-set image.
2. Collect predictions alongside ground-truth labels into a results table.
3. Compute all metrics in Section 3 using `scikit-learn.metrics` (or `torchmetrics` for the DL component).
4. Generate visualizations:
   - Confusion matrix heatmap
   - ROC curve (with AUC annotated)
   - Precision-Recall curve
   - Score-distribution histogram (normal vs. defective)
5. Save all outputs under `ml/evaluation/results/`.

## 5. Failure Case Analysis

A model is only as trustworthy as its documented weaknesses. For every evaluation run:

- Manually inspect a sample of **false positives** (flagged as degraded/defective but actually fine) and **false negatives** (missed real issues).
- Categorize failure patterns, e.g.:
  - Does the model struggle with a specific type of blur (motion vs. focus)?
  - Does it confuse low-light scenes (intentionally dark) with underexposure defects?
  - Are borderline/ambiguous severity cases driving most errors?
- Document these patterns explicitly — this analysis is as valuable as the raw metrics for demonstrating technical judgment.

## 6. Explainability Validation

- For a sample of anomaly predictions, generate Grad-CAM (or equivalent) visualizations and manually verify the highlighted region corresponds to the actual defect location where ground truth is available (e.g., using MVTec AD's pixel-level annotations, if that dataset is used).
- Report qualitative examples (a handful of side-by-side original image / heatmap pairs) in the final evaluation report.

## 7. Confidence Calibration (Optional/Bonus)

- Assess whether reported confidence scores are well-calibrated (e.g., a "0.9 confidence" prediction should be correct ~90% of the time).
- Reliability diagrams / Expected Calibration Error (ECE) can be used if this is pursued.

## 8. Reporting Format

All evaluation results should be summarized in `ml/evaluation_report.md`, containing:

1. Dataset summary (sizes, sources, class balance)
2. Metrics table (all metrics from Section 3)
3. Visualizations (confusion matrix, ROC/PR curves, score distributions)
4. Failure case discussion (Section 5)
5. Explainability samples (Section 6)
6. Known limitations and suggested future improvements

## 9. Success Thresholds (Suggested Targets)

These are reasonable internal targets to aim for — not hard requirements, since actual achievable performance depends on data quality and quantity:

| Metric | Target |
|---|---|
| Macro F1 (quality label) | ≥ 0.80 |
| ROC-AUC (anomaly detection) | ≥ 0.90 |
| Blur/Exposure/Noise detector F1 | ≥ 0.85 each |

If targets are not met, this should be documented honestly alongside the reasoning (data limitations, task difficulty, time constraints) rather than adjusted after the fact to appear successful.