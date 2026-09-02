"""
Complete Inference Pipeline
Orchestrates feature extraction, baseline model, deep model, and score fusion.
Single entry point for quality analysis of images.
Follows Design Principle: Separation of Concerns, Orchestration
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Union, Tuple, Dict, Any
from dataclasses import asdict

from ml.feature_extraction import ImageFeatureExtractor, FeatureStats, extract_image_features
from ml.image_validation import validate_image, ImageValidationError
from ml.baseline_model import BaselineModel
from ml.deep_model import DeepModel
from ml.score_fusion import ScoreFusion, QualityAnalysis
from ml.issue_detectors import IssueAnalyzer

logger = logging.getLogger(__name__)

# Binary "usable / not-usable" gate.
# An image is considered USABLE only if no per-issue detector reports a
# measurable issue (severity below this threshold). The threshold is set to the
# measured severity ceiling of clean REAL photos across many unseen sources
# (~0.28; driven by naturally bright/textured scenes). A lower value was tuned
# on only the 12 training sources and wrongly rejected ~50-71% of real clean
# photos. At 0.30 the gate never rejects a real good photo while still catching
# every severe degradation (heavy blur/bright/underexpose/noise all exceed it;
# genuine low-light exposure reached 0.39+).
USABLE_SEVERITY_THRESHOLD = 0.30


class QualityAnalyzer:
    """
    End-to-end quality analysis pipeline.
    
    Architecture:
    1. Image validation (Phase 1)
    2. Feature extraction (Phase 1)
    3. Baseline model inference (Phase 3a)
    4. Deep model anomaly scoring (Phase 3b)
    5. Score fusion (Phase 3c)
    6. Return unified QualityAnalysis
    
    Design Pattern: Facade (orchestrates complex pipeline)
    """

    def __init__(
        self,
        baseline_model_path: str = None,
        deep_model_path: str = None,
        device: str = "cpu",
    ):
        """
        Initialize quality analyzer.
        
        Args:
            baseline_model_path: Path to saved baseline model
            deep_model_path: Path to saved deep model
            device: "cpu" or "cuda"
        """
        self.feature_extractor = ImageFeatureExtractor()
        self.baseline_model = None
        self.deep_model = None
        self.fusion = ScoreFusion()
        self.device = device

        # Load models if provided
        if baseline_model_path:
            self.baseline_model = BaselineModel.load(baseline_model_path)
            logger.info(f"Loaded baseline model from {baseline_model_path}")

        if deep_model_path:
            self.deep_model = DeepModel.load(deep_model_path, device=device)
            logger.info(f"Loaded deep model from {deep_model_path}")

        # Load calibrated anomaly thresholds if present (saved by training)
        self._load_anomaly_thresholds(baseline_model_path, deep_model_path)

        self.is_ready = self.baseline_model is not None and self.deep_model is not None
        if not self.is_ready:
            logger.warning("Not all models loaded. Some predictions may be incomplete.")

    def _load_anomaly_thresholds(self, baseline_model_path, deep_model_path) -> None:
        """
        Load persisted anomaly-score thresholds and configure the fusion logic.

        Thresholds are stored alongside the deep model by the training pipeline.
        """
        if deep_model_path:
            thr_path = Path(deep_model_path).parent / "anomaly_thresholds.json"
            if thr_path.exists():
                import json as _json
                with open(thr_path) as _f:
                    thresholds = _json.load(_f)
                self.fusion.set_absolute_thresholds(thresholds)
                logger.info(f"Loaded calibrated anomaly thresholds from {thr_path}")

    def analyze_image(
        self,
        image: Union[str, np.ndarray],
        return_details: bool = False,
    ) -> QualityAnalysis:
        """
        Analyze image quality end-to-end.
        
        Args:
            image: File path (str) or image array (np.ndarray)
            return_details: Whether to include detailed breakdowns
        
        Returns:
            QualityAnalysis with complete assessment
        
        Raises:
            ImageValidationError: If image is invalid
        """
        # Load image if path provided
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ImageValidationError(f"Failed to load image: {image_path}")

        # Validate image
        try:
            validate_image("image", image=image)
        except ImageValidationError as e:
            logger.warning(f"Image validation warning: {e}")
            # Continue anyway (validation is best-effort)

        # Extract features
        feature_stats = extract_image_features(image)
        features_array = np.array([
            feature_stats.sharpness,
            feature_stats.brightness,
            feature_stats.contrast,
            feature_stats.noise_level,
            feature_stats.saturation,
            feature_stats.texture_complexity,
        ])

        # Baseline prediction
        baseline_label = "ACCEPTABLE"  # Default
        baseline_confidence = 0.5

        if self.baseline_model:
            baseline_label, baseline_confidence = self.baseline_model.predict(features_array)

        # Deep model anomaly scoring
        anomaly_score = 0.0  # Default
        if self.deep_model:
            # Prepare image for deep model: (H, W, 3) → (1, 3, H, W)
            if image.shape[2] == 3:
                image_normalized = image.astype(np.float32) / 255.0
                image_for_deep = np.transpose(image_normalized, (2, 0, 1))
                image_batch = np.expand_dims(image_for_deep, 0)

                anomaly_scores = self.deep_model.compute_anomaly_score(image_batch)
                anomaly_score = float(anomaly_scores[0])

        # Per-issue detectors (Approach A): the reliable, content-aware signal.
        # Each detector targets one issue type (blur/exposure/noise/defect) with
        # a content-normalized metric, unlike the content-fragile 6-feature model.
        issue_analyzer = IssueAnalyzer()
        issue_result = issue_analyzer.to_dict(image, anomaly_score=anomaly_score)
        issues = issue_result["issues"]
        issue_qscore = issue_result["quality_score"]

        # Feature breakdown
        feature_breakdown = {
            "sharpness": float(feature_stats.sharpness),
            "brightness": float(feature_stats.brightness),
            "contrast": float(feature_stats.contrast),
            "noise_level": float(feature_stats.noise_level),
            "saturation": float(feature_stats.saturation),
            "texture_complexity": float(feature_stats.texture_complexity),
        }

        # Fuse predictions (baseline + anomaly) for the legacy fields
        quality_analysis = self.fusion.fuse_predictions(
            baseline_label=baseline_label,
            baseline_confidence=baseline_confidence,
            anomaly_score=anomaly_score,
            features=feature_breakdown if return_details else None,
        )

        # Override the aggregated score/label with the per-issue result: the
        # per-issue detectors reliably separate clean from degraded, so they are
        # authoritative. Keep the legacy fields for backward compatibility.
        qscore = issue_qscore
        issues_present = [i for i in issues if i["present"]]

        # Semantically principled bands:
        #   ACCEPTABLE  -> no significant issues (score high)
        #   DEGRADED    -> >=1 noticeable issue, but the frame is still useable
        #                  (a single severe blur/noise is degraded, not broken)
        #   DEFECTIVE   -> multiple concurrent significant issues, OR one
        #                  extreme/catastrophic issue (frame effectively unusable)
        significant = [i for i in issues if i["severity"] >= 0.4]
        worst_sev = max((i["severity"] for i in issues), default=0.0)

        # --- Binary "usable / not-usable" gate (the reliable product signal) ---
        # Based ONLY on the reliable, content-aware detectors (blur, noise,
        # exposure, jpeg-blockiness). The deep anomaly/defect channel is
        # intentionally excluded from this gate: it is trained on very few
        # clean real sources and catastrophically misfires on unseen photograph
        # content (e.g. it flags the clean `basketball` source as defective at
        # severity 0.5), so using it here would reject perfectly good frames.
        #
        # Each detector has its OWN threshold, calibrated against the measured
        # real-clean ceiling for that detector (see below) so that genuine
        # clean photographs are never rejected while detected issues are.
        #   - blur:     0.30  (clean real ceiling ~0.28, e.g. `fruits`)
        #   - noise:    0.30  (clean real ceiling ~0.20, e.g. `baboon`)
        #   - exposure: 0.30  (clean real ceiling ~0.20, e.g. bright `aloeL`,
        #                      dark `messi5` ~0.15)
        #   - jpeg:     0.20  (clean real ceiling ~0.09; re-encoding an
        #                      uncompressed source reliably exceeds ~0.29)
        per_detector_threshold = {
            "blur": USABLE_SEVERITY_THRESHOLD,
            "noise": USABLE_SEVERITY_THRESHOLD,
            "exposure": USABLE_SEVERITY_THRESHOLD,
            "jpeg": 0.20,
        }
        reliable_sev = [
            i for i in issues
            if i["issue_type"] in per_detector_threshold
        ]
        not_usable = any(
            i["severity"] >= per_detector_threshold[i["issue_type"]]
            for i in reliable_sev
        )
        usable = not not_usable

        if qscore >= 80 and not significant:
            label = "ACCEPTABLE"
            confidence = 0.9 if not issues_present else 0.7
            reasoning = "No significant quality issues detected."
        elif len(significant) >= 2 or worst_sev >= 0.95:
            label = "DEFECTIVE"
            confidence = 0.85
            reasons = ", ".join(sorted({i["issue_type"] for i in significant})) or "severe quality problems"
            reasoning = f"Defective due to: {reasons}"
        else:
            label = "DEGRADED"
            confidence = 0.7
            reasoning = "Quality degraded by: " + ", ".join(i["issue_type"] for i in issues_present)
        quality_analysis.quality_label = label
        quality_analysis.quality_score = float(qscore)
        quality_analysis.confidence = float(confidence)
        quality_analysis.reasoning = reasoning
        quality_analysis.issues = issues
        quality_analysis.usable = bool(usable)

        logger.info(
            f"Analysis complete: {quality_analysis.quality_label} "
            f"(score={quality_analysis.quality_score:.1f}, "
            f"confidence={quality_analysis.confidence:.2f})"
        )

        return quality_analysis

    def batch_analyze(
        self,
        images: list,
        return_details: bool = False,
    ) -> list:
        """
        Analyze multiple images.
        
        Args:
            images: List of image paths or arrays
            return_details: Whether to include detailed breakdowns
        
        Returns:
            List of QualityAnalysis objects
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.analyze_image(image, return_details=return_details)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing image {i}: {e}")
                # Create failed analysis
                failed_result = QualityAnalysis(
                    quality_label="UNKNOWN",
                    quality_score=0,
                    confidence=0,
                    baseline_label="ERROR",
                    baseline_confidence=0,
                    anomaly_score=0,
                    anomaly_percentile=0,
                    feature_breakdown={},
                    reasoning=f"Analysis failed: {str(e)}",
                )
                results.append(failed_result)

        return results

    def analyze_directory(
        self,
        directory: str,
        return_details: bool = False,
        recursive: bool = False,
    ) -> Dict[str, QualityAnalysis]:
        """
        Analyze all images in directory.
        
        Args:
            directory: Directory path
            return_details: Whether to include detailed breakdowns
            recursive: Whether to search recursively
        
        Returns:
            Dictionary mapping image paths to QualityAnalysis
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all images
        pattern = "**/*" if recursive else "*"
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = [
            p for p in dir_path.glob(pattern)
            if p.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_paths)} images in {directory}")

        results = {}
        for image_path in image_paths:
            try:
                result = self.analyze_image(str(image_path), return_details=return_details)
                results[str(image_path)] = result
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {e}")

        return results


def infer(image: Union[str, np.ndarray]) -> Dict[str, Any]:
    """
    Convenience function for single image inference.
    Returns results as dictionary.
    
    Args:
        image: Image file path or array
    
    Returns:
        Dictionary with quality assessment
    """
    # Try to load models from default locations
    baseline_model_path = "./ml/models/baseline_model_random_forest_v1.joblib"
    deep_model_path = "./ml/models/deep_model_autoencoder_v1.pt"

    analyzer = QualityAnalyzer(
        baseline_model_path=baseline_model_path if Path(baseline_model_path).exists() else None,
        deep_model_path=deep_model_path if Path(deep_model_path).exists() else None,
    )

    result = analyzer.analyze_image(image, return_details=True)
    return asdict(result)
