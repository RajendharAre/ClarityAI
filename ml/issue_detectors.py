"""
Per-Issue Quality Detectors
Detects individual image-quality issues (blur, exposure, noise, defect) with
interpretable, content-aware metrics. This replaces the single 3-class
classifier with independent, per-issue detectors -- how real quality tools work.

Why per-issue detectors instead of one classifier:
The original pipeline used 6 global features -> one RandomForest -> 3 classes.
Measured on real photographs, ACCEPTABLE vs DEGRADED were statistically
indistinguishable across all 6 features (see temp/diagnostic_stepB.md), because
global aggregates are overwhelmed by natural content variance. Each detector
below uses a content-aware metric targeted at ONE kind of issue, which gives
them far better discrimination.

Follows Design Patterns: Strategy, Facade. Design Principles: Interpretability,
SRP, Modularity.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IssueResult:
    """Result of a single issue detector."""

    issue_type: str  # "blur", "exposure", "noise", "defect"
    severity: float  # 0.0 (none) .. 1.0 (severe)
    present: bool  # severity >= threshold
    confidence: float  # 0..1 how confident in the result
    detail: str = ""  # human-readable explanation
    meta: Dict[str, float] = field(default_factory=dict)  # raw metrics

    def to_dict(self) -> Dict:
        return asdict(self)


class IssueDetector(ABC):
    """Abstract base class for an issue detector."""

    issue_type: str = "generic"

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: severity at/above which the issue is considered present.
        """
        self.threshold = threshold

    @abstractmethod
    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        """Return raw metrics for the issue, including 'severity' (0..1)."""

    def detect(self, image: np.ndarray) -> IssueResult:
        """Run the detector and produce an IssueResult."""
        meta = self.compute_severity(image)
        severity = float(np.clip(meta.get("severity", 0.0), 0.0, 1.0))
        present = severity >= self.threshold
        return IssueResult(
            issue_type=self.issue_type,
            severity=severity,
            present=present,
            confidence=self._confidence(severity),
            detail=self._describe(severity),
            meta=meta,
        )

    def _confidence(self, severity: float) -> float:
        # More confident far from the threshold
        dist = abs(severity - self.threshold)
        return float(np.clip(0.4 + dist, 0.4, 1.0))

    def _describe(self, severity: float) -> str:
        if severity < 0.3:
            return f"Low {self.issue_type} severity"
        if severity < 0.6:
            return f"Moderate {self.issue_type}"
        return f"Severe {self.issue_type} detected"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class BlurDetector(IssueDetector):
    """
    Content-aware blur detector (edge-steepness based).

    Measures how steep the strong edges are, normalized by the image's own
    global contrast: mean(Sobel gradient on strong edge pixels) / std(gray).

    A sharp photo has steep, concentrated edges -> high normalized edge
    strength. Blur smears edges out -> low strength. This is content-robust:
    smooth-but-sharp scenes (e.g. a blank wall, a soft-lit subject) still have
    steep local gradients wherever there IS detail, so they score like other
    sharp photos rather than being mistaken for blur -- unlike raw Laplacian
    variance, which collapses to ~0 on any low-detail scene.
    """

    issue_type = "blur"

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        self._target = 256
        # Severity ramp: normalized edge strength >= sharp_floor -> 0 (sharp),
        # <= blur_floor -> 1 (blurred). Calibrated on real photos:
        #   clean range ~4.0-6.6, blurred(0.95) ~1.3-3.4.
        self._sharp_floor = 4.4
        self._blur_floor = 2.8
        self._edge_rel_threshold = 0.20  # strong edge = grad > 20% of max
        self._min_edge_frac = 0.02  # below this edge density, can't assess blur

    def _edge_steepness(self, image: np.ndarray) -> Tuple[float, float, float]:
        gray = _to_gray(image).astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        strong = grad > (grad.max() + 1e-6) * self._edge_rel_threshold
        edge_frac = float(strong.mean())
        gstd = float(gray.std()) + 1e-6
        edge_str = float(grad[strong].mean()) if edge_frac > 0 else 0.0
        norm_steep = edge_str / gstd
        return norm_steep, edge_frac, edge_str

    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        gray_src = _to_gray(image)
        if gray_src.shape[0] > self._target:
            scale = self._target / gray_src.shape[0]
            gray_src = cv2.resize(gray_src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        norm_steep, edge_frac, edge_str = self._edge_steepness(gray_src)

        # A featureless frame (no edges at all) cannot be judged for blur -- it
        # may be legitimately smooth content (wall, sky, flat product photo).
        # Do not flag it, to avoid the low-detail false-positive.
        if edge_frac < self._min_edge_frac:
            severity = 0.0
        else:
            t = (norm_steep - self._blur_floor) / (self._sharp_floor - self._blur_floor)
            severity = float(np.clip(1.0 - t, 0.0, 1.0))

        return {
            "severity": severity,
            "normalized_edge_strength": norm_steep,
            "edge_fraction": edge_frac,
            "edge_strength": edge_str,
        }

    def _describe(self, severity: float) -> str:
        if severity < 0.3:
            return "Image is sharp (steep edges)"
        if severity < 0.6:
            return "Some loss of sharpness / mild blur"
        return "Image is noticeably blurred"


class ExposureDetector(IssueDetector):
    """
    Exposure detector.
    Flags under- or over-exposure using mean luminance plus clipping.
    A dark *intentional* scene and a truly underexposed shot are distinguished
    by how much of the frame is clipped to black (or white).
    """

    issue_type = "exposure"

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        # Ideal mean L (LAB) for a well-exposed image
        self._ideal = 128.0

    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        if image.ndim == 2:
            gray = image.astype(np.float32)
            mean_l = float(np.mean(gray))
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            mean_l = float(np.mean(lab[:, :, 0]))  # L channel 0-255
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        dark_clip = float(np.mean(gray < 10))  # fraction near black
        bright_clip = float(np.mean(gray > 245))  # fraction near white

        # Distance of mean luminance from ideal, normalized (0..1)
        l_norm = np.clip(abs(mean_l - self._ideal) / 128.0, 0.0, 1.0)

        # Underexposure: dark mean AND significant clipping to black
        under = (mean_l < 95) and (dark_clip > 0.05)
        over = (mean_l > 160) and (bright_clip > 0.05)

        if under or over:
            severity = float(np.clip(0.5 + 0.5 * l_norm, 0.0, 1.0))
            # boost if clipping is extreme
            clip = max(dark_clip, bright_clip)
            severity = float(np.clip(severity + 0.25 * min(clip * 2, 1.0), 0.0, 1.0))
        else:
            # mild deviation but not clearly clipped
            severity = float(np.clip(l_norm * 0.5, 0.0, 1.0))

        return {
            "severity": severity,
            "mean_L": mean_l,
            "dark_clip": dark_clip,
            "bright_clip": bright_clip,
        }

    def _describe(self, severity: float) -> str:
        if severity < 0.35:
            return "Exposure is in normal range"
        if severity < 0.6:
            return "Exposure slightly off (under/over)"
        return "Image is significantly over- or under-exposed"


class NoiseDetector(IssueDetector):
    """
    Noise detector using flat-region denoise-difference.

    Measures noise ONLY in flat (low-local-variance) image regions rather than
    across the whole frame. This solves the classic "texture vs noise"
    ambiguity: fine, legitimate texture (e.g. a brick wall, foliage, a
    checkerboard) has high local variance and would otherwise be misread as
    random noise. Real sensor/compression noise is present everywhere, so
    averaging the denoise residual over smooth areas isolates true noise.
    """

    issue_type = "noise"

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        self._target = 256
        # Flat region = local std below this (on 0-255 gray).
        self._flat_var_thresh = 35.0
        # Severity mapping: noise_magnitude at/below clean_floor -> 0,
        # at/above noisy_floor -> 1 (measured on real photos).
        self._clean_floor = 2.0
        self._noisy_floor = 7.0

    def _flat_region_noise(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        gray = _to_gray(image).astype(np.float32)
        # Local mean and local variance via box blur of squared deviation.
        k = (15, 15)
        local_mean = cv2.GaussianBlur(gray, k, 0)
        local_var = cv2.GaussianBlur((gray - local_mean) ** 2, k, 0)

        # Residual of an edge-preserving median filter = noise estimate.
        denoised = cv2.medianBlur(gray.astype(np.uint8), 5)
        residual = cv2.absdiff(gray.astype(np.uint8), denoised).astype(np.float32)

        flat = (local_var < self._flat_var_thresh).astype(np.float32)
        if flat.mean() < 0.02:  # degenerate: no flat region, fall back to full
            flat[:] = 1.0
        noise_mag = float(np.sum(residual * flat) / max(float(np.sum(flat)), 1.0))
        return noise_mag, flat

    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        if image.shape[0] > self._target:
            scale = self._target / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        noise_mag, flat = self._flat_region_noise(image)

        # Empirical mapping (256px, measured on real photos):
        #   clean photos: ~1-2, noisy(0.95): ~9-30
        t = (noise_mag - self._clean_floor) / (self._noisy_floor - self._clean_floor)
        severity = float(np.clip(t, 0.0, 1.0))

        return {"severity": severity, "noise_magnitude": noise_mag, "flat_fraction": float(flat.mean())}

    def _describe(self, severity: float) -> str:
        if severity < 0.3:
            return "Low image noise"
        if severity < 0.6:
            return "Noticeable noise"
        return "Severe image noise"


class JpegBlockinessDetector(IssueDetector):
    """
    JPEG blockiness detector (8x8-grid seam discontinuity).

    JPEG compression quantizes each 8x8 DCT block independently, producing a
    characteristic grid of boundary discontinuities invisible to the eye but
    measurable: mean edge-response at the 8x8 seams is elevated relative to the
    interior. This is specifically the metric that separates a JPEG-re-encoded
    image from its uncompressed original -- the noise detector cannot, because
    JPEG artifacts are structurally correlated, not random.

    IMPORTANT LIMITATION (documented): the signal only exists when the SOURCE is
    uncompressed (PNG). JPEG-on-JPEG is indistinguishable (the "invisible 6"
    JPEG test-set images on JPEG sources remain physically undetectable). This
    detector is calibrated so clean content maps to ~0 severity and genuine
    re-encoding of an uncompressed source rises with compression severity.
    """

    issue_type = "jpeg"

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold)
        # Severity ramp: blockiness <= 0.10 -> 0 (clean range on real photos:
        # roughly -0.17..0.09), blockiness >= 1.0 -> 1.0 (heavy JPEG forcing of
        # an uncompressed source reaches ~0.7-1.3).
        self._clean_ceiling = 0.10
        self._severe = 1.0

    def _blockiness(self, image: np.ndarray) -> float:
        gray = _to_gray(image)
        if gray.shape[0] > 512:
            scale = 512 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        g = gray.astype(np.float32)
        h, w = g.shape
        dh = np.abs(np.diff(g, axis=1))
        dv = np.abs(np.diff(g, axis=0))

        col_idx = np.arange(w - 1)
        row_idx = np.arange(h - 1)
        col_bound = col_idx % 8 == 7
        row_bound = row_idx % 8 == 7

        bh = float(dh[:, col_bound].mean())
        ih = float(dh[:, ~col_bound].mean())
        bv = float(dv[row_bound, :].mean())
        iv = float(dv[~row_bound, :].mean())

        eh = (bh - ih) / (ih + 1e-6)
        ev = (bv - iv) / (iv + 1e-6)
        return float((eh + ev) / 2)

    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        b = self._blockiness(image)
        severity = float(np.clip((b - self._clean_ceiling) / (self._severe - self._clean_ceiling), 0.0, 1.0))
        return {"severity": severity, "blockiness": b}

    def _describe(self, severity: float) -> str:
        if severity < 0.3:
            return "No noticeable JPEG blocking artifacts"
        if severity < 0.6:
            return "Visible JPEG grid / compression artifacts"
        return "Heavy JPEG compression artifacts"


class DefectDetector(IssueDetector):
    """
    Defect detector wrapping the deep autoencoder anomaly score.

    The anomaly score is used as an absolute severity: 0..(high threshold)
    scaled to 0..1. It is trained only on clean images, so its magnitude is
    measured against the calibrated 'high' threshold.
    """

    issue_type = "defect"

    def __init__(self, threshold: float = 0.5, high_threshold: float = 0.15,
                 max_severity: float = 0.5):
        super().__init__(threshold)
        self.high_threshold = high_threshold  # score at/above this = severe defect
        # The deep anomaly channel is trained on few unique real sources, so it
        # generalizes unreliably to arbitrary photographic content (it can flag
        # normal-but-unseen content as anomalous). Cap its severity so it can
        # downgrade a score toward DEGRADED but never single-handedly declare a
        # photo DEFECTIVE.
        self.max_severity = max_severity

    def compute_severity(self, image: np.ndarray) -> Dict[str, float]:
        # Severity provided externally via set_anomaly_score; default 0.
        # Subclasses/analyzer call set_raw() before detect().
        raw = getattr(self, "_raw_anomaly_score", 0.0)
        severity = float(np.clip(raw / self.high_threshold, 0.0, self.max_severity))
        return {"severity": severity, "anomaly_score": raw}

    def set_raw(self, anomaly_score: float) -> "DefectDetector":
        """Inject the pre-computed autoencoder anomaly score."""
        self._raw_anomaly_score = float(anomaly_score)
        return self

    def _describe(self, severity: float) -> str:
        if severity < 0.4:
            return "No obvious defects detected"
        if severity < 0.7:
            return "Possible subtle defect / anomaly"
        return "Defect or anomaly detected"


class IssueAnalyzer:
    """
    Orchestrates all per-issue detectors.
    Facade pattern: run every detector and aggregate into a final quality score.
    """

    def __init__(self, blur: IssueDetector = None, exposure: IssueDetector = None,
                 noise: IssueDetector = None, defect: IssueDetector = None,
                 jpeg: IssueDetector = None):
        self.blur = blur or BlurDetector()
        self.exposure = exposure or ExposureDetector()
        self.noise = noise or NoiseDetector()
        self.defect = defect or DefectDetector()
        self.jpeg = jpeg or JpegBlockinessDetector()

    def analyze(self, image: np.ndarray, anomaly_score: float = 0.0) -> List[IssueResult]:
        """Run all detectors. anomaly_score optionally injected for defects."""
        self.defect.set_raw(anomaly_score)
        results = [
            self.blur.detect(image),
            self.exposure.detect(image),
            self.noise.detect(image),
            self.jpeg.detect(image),
            self.defect.detect(image),
        ]
        return results

    def quality_score(self, image: np.ndarray, anomaly_score: float = 0.0) -> float:
        """
        Fuse per-issue severities into a 0-100 quality score.
        The overall score is penalized by the worst issue (defective/worst
        condition) more heavily than the average: `worst` dominates, `mean`
        supports.
        """
        results = self.analyze(image, anomaly_score=anomaly_score)
        worst = max(r.severity for r in results)
        mean = sum(r.severity for r in results) / len(results)
        # Blend: worst issue has ~3x weight of the average
        effective = 0.75 * worst + 0.25 * mean
        return float(np.clip(100.0 * (1.0 - effective), 0.0, 100.0))

    def to_dict(self, image: np.ndarray, anomaly_score: float = 0.0) -> Dict:
        """Full serializable result."""
        results = self.analyze(image, anomaly_score=anomaly_score)
        return {
            "quality_score": round(self.quality_score(image, anomaly_score=anomaly_score), 1),
            "issues": [r.to_dict() for r in results],
        }
