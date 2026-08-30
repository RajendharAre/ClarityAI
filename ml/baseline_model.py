"""
Baseline Classical ML Model
Trains on engineered features from Phase 1 to classify quality labels.
Follows Design Principle: Interpretability, Reproducibility
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container for saved model artifacts"""
    classifier: Any
    scaler: Any
    feature_names: list
    class_names: list
    metadata: Dict


class BaselineModel:
    """
    Baseline classifier using engineered features from Phase 1.
    
    Architecture:
    - Input: FeatureStats (6 features: sharpness, brightness, contrast, noise, saturation, texture)
    - Processing: StandardScaler normalization
    - Model: Random Forest or Gradient Boosting
    - Output: Quality label (ACCEPTABLE, DEGRADED, DEFECTIVE) + confidence
    
    Design Pattern: Adapter (converts FeatureStats to model input)
    """

    # Quality label mappings
    LABEL_TO_CLASS = {"ACCEPTABLE": 0, "DEGRADED": 1, "DEFECTIVE": 2}
    CLASS_TO_LABEL = {0: "ACCEPTABLE", 1: "DEGRADED", 2: "DEFECTIVE"}

    # Feature names in order (must match FeatureStats order)
    FEATURE_NAMES = [
        "sharpness",
        "brightness",
        "contrast",
        "noise_level",
        "saturation",
        "texture_complexity",
    ]

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize baseline model.
        
        Args:
            model_type: "random_forest" or "gradient_boosting"
        """
        self.model_type = model_type
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False

        if model_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0,
            )
        elif model_type == "gradient_boosting":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"BaselineModel initialized with {model_type}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train baseline model on feature data.
        
        Args:
            X: Feature array (N, 6) - from Phase 1 FeatureStats
            y: Labels array (N,) - class indices [0, 1, 2]
            val_size: Validation split ratio
            random_state: Random seed
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training baseline model ({self.model_type}) on {len(X)} samples")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, stratify=y
        )

        logger.info(
            f"Split: Train={len(X_train)}, Val={len(X_val)}, Classes={np.unique(y_train)}"
        )

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)

        # Evaluate
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_val_pred = self.classifier.predict(X_val_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "model_type": self.model_type,
            "feature_names": self.FEATURE_NAMES,
        }

        logger.info(f"Training complete: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        logger.info(f"\nValidation Classification Report:\n{classification_report(y_val, y_val_pred)}")
        logger.info(
            f"\nValidation Confusion Matrix:\n{confusion_matrix(y_val, y_val_pred)}"
        )

        self.is_trained = True
        return metrics

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict quality label from features.
        
        Args:
            features: Feature array (6,) or (N, 6)
        
        Returns:
            (quality_label, confidence) or list of tuples
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Handle single sample
        single_sample = features.ndim == 1
        if single_sample:
            features = features.reshape(1, -1)

        # Normalize
        features_scaled = self.scaler.transform(features)

        # Predict
        class_indices = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)

        results = []
        for idx, probs in zip(class_indices, probabilities):
            label = self.CLASS_TO_LABEL[idx]
            confidence = float(probs[idx])
            results.append((label, confidence))

        if single_sample:
            return results[0]
        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        importances = self.classifier.feature_importances_
        return {
            name: float(importance)
            for name, importance in zip(self.FEATURE_NAMES, importances)
        }

    def save(self, model_dir: str) -> str:
        """
        Save model artifacts.
        
        Args:
            model_dir: Directory to save model
        
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        model_path = Path(model_dir) / f"baseline_model_{self.model_type}_v1.joblib"
        scaler_path = Path(model_dir) / f"baseline_scaler_{self.model_type}_v1.joblib"

        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "feature_names": self.FEATURE_NAMES,
            "class_names": [self.CLASS_TO_LABEL[i] for i in range(3)],
            "feature_importance": self.get_feature_importance(),
        }

        metadata_path = Path(model_dir) / f"baseline_metadata_{self.model_type}_v1.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved scaler to {scaler_path}")
        logger.info(f"Saved metadata to {metadata_path}")

        return str(model_path)

    @staticmethod
    def load(model_path: str) -> "BaselineModel":
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Loaded BaselineModel instance
        """
        classifier = joblib.load(model_path)
        model_type = "random_forest" if "random_forest" in model_path else "gradient_boosting"

        model = BaselineModel(model_type=model_type)
        model.classifier = classifier

        # Load scaler
        scaler_path = str(model_path).replace("baseline_model", "baseline_scaler")
        model.scaler = joblib.load(scaler_path)

        model.is_trained = True
        logger.info(f"Loaded baseline model from {model_path}")

        return model


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    save_dir: str = "./ml/models",
) -> BaselineModel:
    """
    Convenience function to train and save baseline model.
    
    Args:
        X_train: Training features (N, 6)
        y_train: Training labels (N,)
        model_type: Model type
        save_dir: Directory to save model
    
    Returns:
        Trained BaselineModel
    """
    model = BaselineModel(model_type=model_type)
    metrics = model.train(X_train, y_train)
    model.save(save_dir)
    return model
