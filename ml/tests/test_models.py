"""
Unit Tests for Phase 3 Models
Tests baseline model, deep model, score fusion, and complete inference pipeline.
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from ml.baseline_model import BaselineModel
from ml.deep_model import DeepModel
from ml.score_fusion import ScoreFusion, QualityAnalysis, AdaptiveScoring
from ml.inference import QualityAnalyzer


class TestBaselineModel:
    """Tests for baseline ML model"""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)
        
        # 100 samples with 6 features each
        X = np.random.randn(100, 6) * 10 + 50
        
        # Labels: 33 ACCEPTABLE (0), 33 DEGRADED (1), 34 DEFECTIVE (2)
        y = np.array([0]*33 + [1]*33 + [2]*34)
        
        return X, y

    def test_baseline_initialization(self):
        """Should initialize model"""
        model = BaselineModel("random_forest")
        assert model is not None
        assert not model.is_trained

    def test_baseline_train(self, synthetic_data):
        """Should train model"""
        X, y = synthetic_data
        model = BaselineModel("random_forest")
        metrics = model.train(X, y)
        
        assert model.is_trained
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.5  # Should do better than random

    def test_baseline_predict(self, synthetic_data):
        """Should make predictions"""
        X, y = synthetic_data
        model = BaselineModel("random_forest")
        model.train(X, y)
        
        # Single sample prediction
        features = X[0]
        label, confidence = model.predict(features)
        
        assert label in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
        assert 0 <= confidence <= 1

    def test_baseline_batch_predict(self, synthetic_data):
        """Should batch predict"""
        X, y = synthetic_data
        model = BaselineModel("random_forest")
        model.train(X, y)
        
        # Batch prediction
        features_batch = X[:5]
        results = model.predict(features_batch)
        
        assert len(results) == 5
        assert all(isinstance(r, tuple) for r in results)

    def test_baseline_feature_importance(self, synthetic_data):
        """Should return feature importance"""
        X, y = synthetic_data
        model = BaselineModel("random_forest")
        model.train(X, y)
        
        importance = model.get_feature_importance()
        assert len(importance) == 6
        assert all(isinstance(v, float) for v in importance.values())
        assert sum(importance.values()) > 0

    def test_baseline_save_load(self, synthetic_data):
        """Should save and load model"""
        X, y = synthetic_data
        model = BaselineModel("random_forest")
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            
            # Verify files created
            assert (Path(tmpdir) / "baseline_model_random_forest_v1.joblib").exists()
            assert (Path(tmpdir) / "baseline_scaler_random_forest_v1.joblib").exists()
            
            # Load and verify
            loaded_model = BaselineModel.load(
                str(Path(tmpdir) / "baseline_model_random_forest_v1.joblib")
            )
            assert loaded_model.is_trained
            
            # Predictions should match
            pred1 = model.predict(X[0])
            pred2 = loaded_model.predict(X[0])
            assert pred1[0] == pred2[0]  # Same label


class TestDeepModel:
    """Tests for deep learning model"""

    @pytest.fixture
    def synthetic_images(self):
        """Generate synthetic images"""
        np.random.seed(42)
        images = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
        return images

    def test_deep_initialization(self):
        """Should initialize model"""
        model = DeepModel(device="cpu")
        assert model is not None
        assert not model.is_trained

    def test_deep_train(self, synthetic_images):
        """Should train model"""
        model = DeepModel(device="cpu")
        history = model.train_model(synthetic_images, epochs=2, batch_size=2)
        
        assert model.is_trained
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_deep_anomaly_score(self, synthetic_images):
        """Should compute anomaly scores"""
        model = DeepModel(device="cpu")
        model.train_model(synthetic_images, epochs=2, batch_size=2)
        
        # Score normal images
        scores = model.compute_anomaly_score(synthetic_images)
        
        assert len(scores) == len(synthetic_images)
        assert all(isinstance(s, (float, np.floating)) for s in scores)
        assert all(s >= 0 for s in scores)

    def test_deep_save_load(self, synthetic_images):
        """Should save and load model"""
        model = DeepModel(device="cpu")
        model.train_model(synthetic_images, epochs=2, batch_size=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            
            # Verify files created
            assert (Path(tmpdir) / "deep_model_autoencoder_v1.pt").exists()
            
            # Load and verify
            loaded_model = DeepModel.load(
                str(Path(tmpdir) / "deep_model_autoencoder_v1.pt"),
                device="cpu"
            )
            assert loaded_model.is_trained
            
            # Scores should match
            scores1 = model.compute_anomaly_score(synthetic_images[:2])
            scores2 = loaded_model.compute_anomaly_score(synthetic_images[:2])
            np.testing.assert_array_almost_equal(scores1, scores2, decimal=4)


class TestScoreFusion:
    """Tests for score fusion logic"""

    def test_fusion_initialization(self):
        """Should initialize fusion"""
        fusion = ScoreFusion()
        assert fusion is not None

    def test_fusion_acceptable_clean(self):
        """Should fuse ACCEPTABLE label with low anomaly"""
        fusion = ScoreFusion()
        result = fusion.fuse_predictions(
            baseline_label="ACCEPTABLE",
            baseline_confidence=0.9,
            anomaly_score=0.01,  # Very low anomaly
        )
        
        assert isinstance(result, QualityAnalysis)
        assert result.quality_label == "ACCEPTABLE"
        assert result.quality_score >= 80

    def test_fusion_acceptable_anomalous(self):
        """Should upgrade ACCEPTABLE with high anomaly"""
        fusion = ScoreFusion()
        result = fusion.fuse_predictions(
            baseline_label="ACCEPTABLE",
            baseline_confidence=0.9,
            anomaly_score=0.5,  # High anomaly
        )
        
        assert result.quality_label in ["DEGRADED", "DEFECTIVE"]
        assert result.quality_score < 80

    def test_fusion_degraded(self):
        """Should keep DEGRADED label"""
        fusion = ScoreFusion()
        result = fusion.fuse_predictions(
            baseline_label="DEGRADED",
            baseline_confidence=0.7,
            anomaly_score=0.3,
        )
        
        assert result.quality_label == "DEGRADED"

    def test_fusion_defective(self):
        """Should keep DEFECTIVE label"""
        fusion = ScoreFusion()
        result = fusion.fuse_predictions(
            baseline_label="DEFECTIVE",
            baseline_confidence=0.8,
            anomaly_score=0.9,
        )
        
        assert result.quality_label == "DEFECTIVE"
        assert result.quality_score < 50

    def test_fusion_batch(self):
        """Should batch fuse"""
        fusion = ScoreFusion()
        labels = ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
        confs = [0.9, 0.7, 0.8]
        scores = np.array([0.01, 0.3, 0.9])
        
        results = fusion.batch_fuse(labels, confs, scores)
        
        assert len(results) == 3
        assert all(isinstance(r, QualityAnalysis) for r in results)


class TestAdaptiveScoring:
    """Tests for adaptive scoring"""

    def test_adaptive_initialization(self):
        """Should initialize adaptive scoring"""
        adaptive = AdaptiveScoring()
        assert adaptive is not None

    def test_adaptive_collect_samples(self):
        """Should collect samples"""
        adaptive = AdaptiveScoring()
        
        adaptive.collect_samples("ACCEPTABLE", 0.9, 0.05)
        adaptive.collect_samples("DEGRADED", 0.7, 0.3)
        adaptive.collect_samples("DEFECTIVE", 0.85, 0.8)
        
        assert len(adaptive.anomaly_acceptable_scores) == 1
        assert len(adaptive.anomaly_degraded_scores) == 1
        assert len(adaptive.anomaly_defective_scores) == 1

    def test_adaptive_statistics(self):
        """Should compute statistics"""
        adaptive = AdaptiveScoring()
        
        # Add multiple samples
        for _ in range(5):
            adaptive.collect_samples("ACCEPTABLE", 0.9, np.random.uniform(0.01, 0.1))
            adaptive.collect_samples("DEGRADED", 0.7, np.random.uniform(0.2, 0.4))
            adaptive.collect_samples("DEFECTIVE", 0.8, np.random.uniform(0.6, 0.95))
        
        stats = adaptive.compute_statistics()
        
        assert "baseline" in stats
        assert "anomaly" in stats


class TestQualityAnalyzer:
    """Tests for complete inference pipeline"""

    @pytest.fixture
    def synthetic_image(self):
        """Generate synthetic image"""
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        return image

    def test_analyzer_initialization(self):
        """Should initialize analyzer"""
        analyzer = QualityAnalyzer()
        assert analyzer is not None

    def test_analyzer_analyze_without_models(self, synthetic_image):
        """Should analyze without trained models (graceful degradation)"""
        analyzer = QualityAnalyzer()
        result = analyzer.analyze_image(synthetic_image)
        
        assert isinstance(result, QualityAnalysis)
        # Should have default predictions
        assert result.quality_label in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
        assert 0 <= result.quality_score <= 100

    def test_analyzer_batch(self, synthetic_image):
        """Should batch analyze"""
        analyzer = QualityAnalyzer()
        images = [synthetic_image, synthetic_image, synthetic_image]
        
        results = analyzer.batch_analyze(images)
        
        assert len(results) == 3
        assert all(isinstance(r, QualityAnalysis) for r in results)

    def test_analyzer_from_file(self, synthetic_image):
        """Should analyze from file"""
        analyzer = QualityAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test image
            image_path = Path(tmpdir) / "test.jpg"
            cv2.imwrite(str(image_path), synthetic_image)
            
            # Analyze
            result = analyzer.analyze_image(str(image_path))
            assert isinstance(result, QualityAnalysis)

    def test_analyzer_directory(self, synthetic_image):
        """Should analyze directory"""
        analyzer = QualityAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save test images
            for i in range(3):
                image_path = Path(tmpdir) / f"test_{i}.jpg"
                cv2.imwrite(str(image_path), synthetic_image)
            
            # Analyze directory
            results = analyzer.analyze_directory(tmpdir)
            assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
