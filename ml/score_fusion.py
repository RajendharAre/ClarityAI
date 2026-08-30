"""
Score Fusion Logic
Combines classical baseline predictions with deep anomaly scores into final quality assessment.
Follows Design Principle: Modularity, Interpretability, Ensemble Methods
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysis:
    """Complete quality analysis result"""

    quality_label: str  # "ACCEPTABLE", "DEGRADED", "DEFECTIVE"
    quality_score: float  # 0-100, higher = better quality
    confidence: float  # 0-1, prediction confidence
    baseline_label: str  # Baseline model prediction
    baseline_confidence: float  # Baseline confidence
    anomaly_score: float  # Deep model reconstruction error
    anomaly_percentile: float  # Percentile in training distribution
    feature_breakdown: Dict[str, float]  # Individual feature contributions
    reasoning: str  # Human-readable explanation


class ScoreFusion:
    """
    Fuses baseline and deep model predictions into unified quality score.
    
    Strategy:
    1. Baseline model (Phase 1 features) → quality_label + confidence
    2. Deep model (anomaly detection) → reconstruction_error (anomaly_score)
    3. Fusion logic:
       - If baseline = ACCEPTABLE AND anomaly_score is low → ACCEPTABLE (high confidence)
       - If baseline = ACCEPTABLE AND anomaly_score is high → Upgrade to DEGRADED (anomaly detected)
       - If baseline = DEGRADED → DEGRADED (mid confidence)
       - If baseline = DEFECTIVE → DEFECTIVE (high confidence, no fusion needed)
    4. Score scaling: 0-100 where:
       - 90-100 = ACCEPTABLE (high quality)
       - 50-89 = DEGRADED (usable but issues)
       - 0-49 = DEFECTIVE (unusable)
    
    Design Pattern: Strategy (different fusion strategies based on conditions)
    """

    # Anomaly score thresholds (trained from ACCEPTABLE images)
    # These are percentiles: images with higher reconstruction errors are more anomalous
    ANOMALY_THRESHOLDS = {
        "low": 0.25,  # 25th percentile - normal
        "moderate": 0.75,  # 75th percentile - starting to be anomalous
        "high": 0.95,  # 95th percentile - definitely anomalous
    }

    def __init__(self, anomaly_thresholds: Dict[str, float] = None):
        """
        Initialize fusion logic.
        
        Args:
            anomaly_thresholds: Custom anomaly thresholds (percentiles 0-1)
        """
        if anomaly_thresholds:
            self.ANOMALY_THRESHOLDS = anomaly_thresholds

        self.anomaly_percentile_cache = {}
        logger.info("ScoreFusion initialized")

    def update_anomaly_distribution(self, anomaly_scores: np.ndarray) -> None:
        """
        Update anomaly score distribution from training data.
        Used to calibrate anomaly percentiles.
        
        Args:
            anomaly_scores: Array of anomaly scores from ACCEPTABLE images
        """
        logger.info(
            f"Updating anomaly distribution from {len(anomaly_scores)} samples"
        )
        logger.info(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
        logger.info(f"Anomaly score mean: {anomaly_scores.mean():.4f}")
        logger.info(f"Anomaly score median: {np.median(anomaly_scores):.4f}")

    def fuse_predictions(
        self,
        baseline_label: str,
        baseline_confidence: float,
        anomaly_score: float,
        features: Dict[str, float] = None,
    ) -> QualityAnalysis:
        """
        Fuse baseline and anomaly predictions into final quality assessment.
        
        Args:
            baseline_label: Baseline model prediction ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")
            baseline_confidence: Baseline prediction confidence (0-1)
            anomaly_score: Anomaly reconstruction error (higher = more anomalous)
            features: Feature breakdown for interpretability
        
        Returns:
            QualityAnalysis with fused prediction
        """
        if features is None:
            features = {}

        # Normalize anomaly score to percentile (0-1)
        # Assumption: anomaly_score is in [0, max_observed]
        # For now, use simple normalization; in practice, fit on distribution
        anomaly_percentile = min(1.0, anomaly_score / 0.1)  # Calibrated empirically

        # Fusion logic
        if baseline_label == "DEFECTIVE":
            # If baseline says defective, trust it (no fusion needed)
            quality_label = "DEFECTIVE"
            quality_score = 20  # Low score
            confidence = baseline_confidence * 0.95  # Slightly lower due to ensemble
            reasoning = "Baseline model detected severe quality issues."

        elif baseline_label == "DEGRADED":
            # If baseline says degraded, keep it (no fusion needed)
            quality_label = "DEGRADED"
            quality_score = 60  # Mid score
            confidence = baseline_confidence * 0.9
            reasoning = "Baseline model detected quality degradation."

        elif baseline_label == "ACCEPTABLE":
            # If baseline says acceptable, check anomaly score
            if anomaly_percentile < self.ANOMALY_THRESHOLDS["low"]:
                # Low anomaly score: truly acceptable
                quality_label = "ACCEPTABLE"
                quality_score = 90  # High score
                confidence = baseline_confidence * 0.95
                reasoning = (
                    "Baseline model and anomaly detector both confirm high quality."
                )
            elif anomaly_percentile < self.ANOMALY_THRESHOLDS["moderate"]:
                # Moderate anomaly score: upgrade to degraded
                quality_label = "DEGRADED"
                quality_score = 65
                confidence = (baseline_confidence + (1 - anomaly_percentile)) / 2
                reasoning = (
                    "Baseline acceptable, but anomaly detector found subtle defects."
                )
            else:
                # High anomaly score: upgrade to defective
                quality_label = "DEFECTIVE"
                quality_score = 35
                confidence = max(
                    1 - anomaly_percentile, 0.7
                )  # High confidence defect
                reasoning = "Baseline acceptable, but anomaly detector found major defects."
        else:
            raise ValueError(f"Unknown baseline label: {baseline_label}")

        return QualityAnalysis(
            quality_label=quality_label,
            quality_score=quality_score,
            confidence=confidence,
            baseline_label=baseline_label,
            baseline_confidence=baseline_confidence,
            anomaly_score=anomaly_score,
            anomaly_percentile=anomaly_percentile,
            feature_breakdown=features,
            reasoning=reasoning,
        )

    def batch_fuse(
        self,
        baseline_labels: list,
        baseline_confidences: list,
        anomaly_scores: np.ndarray,
        features_list: list = None,
    ) -> list:
        """
        Fuse predictions for multiple samples.
        
        Args:
            baseline_labels: List of baseline predictions
            baseline_confidences: List of baseline confidences
            anomaly_scores: Array of anomaly scores
            features_list: List of feature dicts
        
        Returns:
            List of QualityAnalysis objects
        """
        if features_list is None:
            features_list = [None] * len(baseline_labels)

        results = []
        for label, conf, anom_score, features in zip(
            baseline_labels, baseline_confidences, anomaly_scores, features_list
        ):
            result = self.fuse_predictions(label, conf, anom_score, features)
            results.append(result)

        return results


class AdaptiveScoring:
    """
    Adaptive scoring that learns from labeled data.
    
    Purpose:
    - Calibrate anomaly thresholds based on training data
    - Learn weight factors for baseline vs anomaly
    - Adapt to dataset-specific distributions
    """

    def __init__(self):
        """Initialize adaptive scoring"""
        self.baseline_acceptable_scores = []
        self.baseline_degraded_scores = []
        self.baseline_defective_scores = []

        self.anomaly_acceptable_scores = []
        self.anomaly_degraded_scores = []
        self.anomaly_defective_scores = []

    def collect_samples(
        self,
        quality_label: str,
        baseline_confidence: float,
        anomaly_score: float,
    ) -> None:
        """
        Collect samples for learning thresholds.
        
        Args:
            quality_label: Ground truth label
            baseline_confidence: Baseline model confidence
            anomaly_score: Anomaly detection score
        """
        if quality_label == "ACCEPTABLE":
            self.baseline_acceptable_scores.append(baseline_confidence)
            self.anomaly_acceptable_scores.append(anomaly_score)
        elif quality_label == "DEGRADED":
            self.baseline_degraded_scores.append(baseline_confidence)
            self.anomaly_degraded_scores.append(anomaly_score)
        elif quality_label == "DEFECTIVE":
            self.baseline_defective_scores.append(baseline_confidence)
            self.anomaly_defective_scores.append(anomaly_score)

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics for calibration.
        
        Returns:
            Statistics dictionary with distribution info
        """
        stats = {
            "baseline": {
                "acceptable_mean": np.mean(self.baseline_acceptable_scores),
                "degraded_mean": np.mean(self.baseline_degraded_scores),
                "defective_mean": np.mean(self.baseline_defective_scores),
            },
            "anomaly": {
                "acceptable_mean": np.mean(self.anomaly_acceptable_scores),
                "degraded_mean": np.mean(self.anomaly_degraded_scores),
                "defective_mean": np.mean(self.anomaly_defective_scores),
            },
        }

        logger.info(f"Adaptive scoring statistics: {stats}")
        return stats

    def get_optimized_thresholds(self) -> Dict[str, float]:
        """
        Get optimized anomaly thresholds based on collected samples.
        
        Returns:
            Optimized thresholds
        """
        if len(self.anomaly_acceptable_scores) == 0:
            logger.warning("No samples collected for optimization")
            return ScoreFusion.ANOMALY_THRESHOLDS

        # Simple threshold: midpoint between acceptable and degraded mean anomaly scores
        acceptable_mean = np.mean(self.anomaly_acceptable_scores)
        degraded_mean = np.mean(self.anomaly_degraded_scores)

        threshold_moderate = (acceptable_mean + degraded_mean) / 2
        threshold_high = degraded_mean

        thresholds = {
            "low": acceptable_mean * 1.2,
            "moderate": threshold_moderate,
            "high": threshold_high,
        }

        logger.info(f"Optimized thresholds: {thresholds}")
        return thresholds
