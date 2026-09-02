import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from ml.baseline_model import BaselineModel
from ml.deep_model import DeepModel
from ml.feature_extraction import extract_image_features


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
EVAL_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"ACCEPTABLE": 0, "DEGRADED": 1, "DEFECTIVE": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_test_data():
    test_dir = DATA_DIR / "test"
    metadata_path = test_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Test metadata not found: {metadata_path}")

    X, y = [], []
    images = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_path = test_dir / item["filename"]
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            features = extract_image_features(image)
            X.append([
                features.sharpness,
                features.brightness,
                features.contrast,
                features.noise_level,
                features.saturation,
                features.texture_complexity,
            ])
            y.append(LABEL_MAP[item["quality_label"]])
            images.append(image)

    if not X:
        raise RuntimeError(f"No valid test images found in {test_dir}")

    return np.array(X, dtype=float), np.array(y, dtype=int), images


def save_confusion_matrix(y_true, y_pred):
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(["ACCEPTABLE", "DEGRADED", "DEFECTIVE"])
    ax.set_yticklabels(["ACCEPTABLE", "DEGRADED", "DEFECTIVE"])
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "black" if value < cm.max() / 2 else "white"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def evaluate_baseline(X_test, y_test):
    baseline_model = BaselineModel.load(str(MODELS_DIR / "baseline_model_random_forest_v1.joblib"))
    pred_labels = []
    for row in X_test:
        label, _ = baseline_model.predict(np.array(row, dtype=float))
        pred_labels.append(LABEL_MAP[label])

    pred_labels = np.array(pred_labels, dtype=int)
    accuracy = accuracy_score(y_test, pred_labels)
    macro_f1 = f1_score(y_test, pred_labels, average="macro")
    report = classification_report(
        y_test,
        pred_labels,
        target_names=["ACCEPTABLE", "DEGRADED", "DEFECTIVE"],
        digits=3,
        output_dict=True,
    )
    save_confusion_matrix(y_test, pred_labels)
    return pred_labels, report, accuracy, macro_f1


def evaluate_deep(images, y_test):
    deep_model = DeepModel.load(str(MODELS_DIR / "deep_model_autoencoder_v1.pt"), device="cpu")
    # Real-world test images have varied dimensions; normalize to the model size
    image_batch = np.stack([cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA) for img in images])
    if image_batch.ndim == 4 and image_batch.shape[-1] == 3 and image_batch.shape[1] != 3:
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))

    scores = deep_model.compute_anomaly_score(image_batch)
    anomaly_target = np.array([1 if label == LABEL_MAP["DEFECTIVE"] else 0 for label in y_test], dtype=int)
    roc_auc = roc_auc_score(anomaly_target, scores)
    return scores, roc_auc


def write_report(accuracy, macro_f1, roc_auc, report):
    lines = [
        "# ClarityAI Evaluation Report",
        "",
        "## Dataset summary",
        f"- Test images: {sum(1 for _ in open(DATA_DIR / 'test' / 'metadata.jsonl', 'r', encoding='utf-8') if _.strip())}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Macro F1: {macro_f1:.4f}",
        f"- ROC-AUC (defective vs non-defective): {roc_auc:.4f}",
        "",
        "## Confusion matrix",
        "![Confusion Matrix](evaluation/confusion_matrix.png)",
        "",
        "## Per-class metrics",
    ]

    for label in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]:
        metrics = report[label]
        lines.append(
            f"- {label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}, support={metrics['support']}"
        )

    lines.extend([
        "",
        "## Conclusion",
        "The baseline classifier and anomaly detector were evaluated on the held-out test set using the dataset generated by the project pipeline.",
    ])

    report_path = ROOT / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved evaluation report: {report_path}")


def main():
    X_test, y_test, images = load_test_data()
    _, report, accuracy, macro_f1 = evaluate_baseline(X_test, y_test)
    _, roc_auc = evaluate_deep(images, y_test)
    write_report(accuracy, macro_f1, roc_auc, report)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion matrix saved to: {EVAL_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
