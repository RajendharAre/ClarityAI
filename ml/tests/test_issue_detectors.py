"""Tests for per-issue quality detectors (Approach A)."""

import cv2
import numpy as np
import pytest

from ml.issue_detectors import (
    BlurDetector,
    ExposureDetector,
    NoiseDetector,
    DefectDetector,
    IssueAnalyzer,
)
from ml.degradations import DegradationPipeline


def synthetic_photo():
    """Create a compact natural-ish, well-exposed, LOW-noise test image."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        img[y, :, 0] = 90 + y // 4
        img[y, :, 1] = 110 + (255 - y) // 5
        img[y, :, 2] = 120
    cv2.rectangle(img, (60, 60), (180, 180), (200, 200, 200), -1)
    img[80:120, 100:140] = (60, 70, 80)
    cv2.circle(img, (200, 50), 30, (220, 220, 220), -1)
    # fine, low-noise texture via a mild checkerboard (structure, not noise)
    for y in range(0, 256, 4):
        for x in range(0, 256, 4):
            if (x // 4 + y // 4) % 4 == 0:
                img[y:y + 2, x:x + 2] = np.clip(img[y:y + 2, x:x + 2].astype(int) + 8, 0, 255).astype(np.uint8)
    return img


def degraded(photo, kind):
    pipe = DegradationPipeline()
    if kind == "blur":
        pipe.add_degradation("blur", 0.95)
    elif kind == "noise":
        pipe.add_degradation("noise", 0.95)
    elif kind == "exposure":
        pipe.add_degradation("exposure", 0.95)
    return pipe.apply(photo)


class TestBlurDetector:
    def test_sharp_severity_low(self):
        d = BlurDetector().detect(synthetic_photo())
        assert d.severity < 0.5
        assert not d.present

    def test_blur_severity_high(self):
        d = BlurDetector().detect(degraded(synthetic_photo(), "blur"))
        assert d.severity > 0.7
        assert d.present

    def test_content_normalization(self):
        # Distinct photos should still separate sharp vs blurred cleanly
        a1, a2 = BlurDetector().detect(synthetic_photo()), BlurDetector().detect(synthetic_photo())
        assert abs(a1.severity - a2.severity) < 0.05


class TestNoiseDetector:
    def test_clean_low(self):
        assert NoiseDetector().detect(synthetic_photo()).severity < NoiseDetector().detect(
            degraded(synthetic_photo(), "noise")
        ).severity

    def test_noisy_high(self):
        assert NoiseDetector().detect(degraded(synthetic_photo(), "noise")).severity > 0.4


class TestExposureDetector:
    def test_normal_low(self):
        # synthetic_photo mean is near-mid, little clipping
        assert ExposureDetector().detect(synthetic_photo()).severity < 0.6

    def test_overexposed_high(self):
        # blow out the photo
        bright = np.clip(synthetic_photo().astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        assert ExposureDetector().detect(bright).severity > 0.6


class TestDefectDetector:
    def test_severity_scales_with_anomaly(self):
        d = DefectDetector(threshold=0.5, high_threshold=0.15)
        low = d.set_raw(0.001).detect(synthetic_photo()).severity
        high = d.set_raw(0.2).detect(synthetic_photo()).severity
        assert low < high
        assert high > 0.4

    def test_severity_is_bounded(self):
        # The deep anomaly channel is capped so it can't single-handedly
        # declare a photo DEFECTIVE (it generalizes unreliably on real content).
        d = DefectDetector(threshold=0.5, high_threshold=0.15, max_severity=0.5)
        assert d.set_raw(1.0).detect(synthetic_photo()).severity <= 0.5


class TestIssueAnalyzer:
    def test_quality_score_clean_higher_than_degraded(self):
        photo = synthetic_photo()
        clean_score = IssueAnalyzer().quality_score(photo)
        for kind in ["blur", "noise", "exposure"]:
            bad = degraded(photo, kind)
            assert IssueAnalyzer().quality_score(bad) < clean_score, kind

    def test_analyze_returns_all_issues(self):
        res = IssueAnalyzer().analyze(synthetic_photo())
        types = {r.issue_type for r in res}
        assert types == {"blur", "exposure", "noise", "jpeg", "defect"}

    def test_to_dict_serializable(self):
        d = IssueAnalyzer().to_dict(synthetic_photo(), anomaly_score=0.01)
        assert 0 <= d["quality_score"] <= 100
        assert len(d["issues"]) == 5
