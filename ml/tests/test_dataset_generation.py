"""
Unit Tests for Dataset Generation Module
Tests degradations, dataset generation, and train/val/test splitting.
Follows Design Principle: Testing & Quality Assurance
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
import json

from ml.degradations import (
    DegradationFactory,
    DegradationPipeline,
    BlurDegradation,
    ExposureDegradation,
    NoiseDegradation,
)
from ml.dataset_generator import (
    DatasetGenerator,
    DatasetConfig,
    ImageMetadata,
    create_sample_dataset,
)


class TestDegradationFactory:
    """Tests for degradation factory"""

    def test_create_blur_degradation(self):
        """Should create blur degradation"""
        degradation = DegradationFactory.create("blur")
        assert isinstance(degradation, BlurDegradation)
        assert degradation.get_issue_type() == "blur"

    def test_create_all_degradation_types(self):
        """Should create all supported degradation types"""
        for deg_type in DegradationFactory.get_all_types():
            degradation = DegradationFactory.create(deg_type)
            assert degradation is not None

    def test_invalid_degradation_type(self):
        """Should raise error for invalid type"""
        with pytest.raises(ValueError):
            DegradationFactory.create("invalid_type")

    def test_get_all_types(self):
        """Should return all supported types"""
        types = DegradationFactory.get_all_types()
        assert "blur" in types
        assert "exposure" in types
        assert "noise" in types
        assert len(types) >= 5


class TestBlurDegradation:
    """Tests for blur degradation"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_apply_no_blur(self, test_image):
        """Severity 0 should apply minimal blur"""
        degradation = BlurDegradation()
        result = degradation.apply(test_image, 0.0)
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

    def test_apply_heavy_blur(self, test_image):
        """Severity 1 should apply heavy blur"""
        degradation = BlurDegradation()
        result = degradation.apply(test_image, 1.0)
        assert result.shape == test_image.shape

    def test_blur_increases_with_severity(self):
        """Higher severity should produce more blur"""
        # Create sharp image (with edges)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = 255
        image[50:, :] = 0

        degradation = BlurDegradation()
        blur_light = degradation.apply(image, 0.2)
        blur_heavy = degradation.apply(image, 0.8)

        # Variance should decrease with more blur
        var_light = np.var(blur_light)
        var_heavy = np.var(blur_heavy)
        assert var_heavy < var_light


class TestExposureDegradation:
    """Tests for exposure degradation"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_underexposure(self, test_image):
        """Severity < 0.5 should darken image"""
        degradation = ExposureDegradation()
        result = degradation.apply(test_image, 0.2)
        assert np.mean(result) < np.mean(test_image)

    def test_overexposure(self, test_image):
        """Severity > 0.5 should brighten image"""
        degradation = ExposureDegradation()
        result = degradation.apply(test_image, 0.8)
        assert np.mean(result) > np.mean(test_image)

    def test_normal_exposure(self, test_image):
        """Severity 0.5 should keep similar brightness"""
        degradation = ExposureDegradation()
        result = degradation.apply(test_image, 0.5)
        # Should be close to original
        assert abs(np.mean(result) - np.mean(test_image)) < 10


class TestNoiseDegradation:
    """Tests for noise degradation"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_apply_no_noise(self, test_image):
        """Severity 0 should add minimal noise"""
        degradation = NoiseDegradation()
        result = degradation.apply(test_image, 0.0)
        # Should be mostly similar
        mse = np.mean((result.astype(float) - test_image.astype(float)) ** 2)
        assert mse < 100

    def test_apply_heavy_noise(self, test_image):
        """Severity 1 should add heavy noise"""
        degradation = NoiseDegradation()
        result = degradation.apply(test_image, 1.0)
        # Should differ significantly
        mse = np.mean((result.astype(float) - test_image.astype(float)) ** 2)
        assert mse > 500

    def test_noise_increases_with_severity(self, test_image):
        """Higher severity should add more noise"""
        degradation = NoiseDegradation()
        result_light = degradation.apply(test_image, 0.2)
        result_heavy = degradation.apply(test_image, 0.8)

        mse_light = np.mean((result_light.astype(float) - test_image.astype(float)) ** 2)
        mse_heavy = np.mean((result_heavy.astype(float) - test_image.astype(float)) ** 2)

        assert mse_heavy > mse_light


class TestDegradationPipeline:
    """Tests for degradation pipeline"""

    @pytest.fixture
    def test_image(self):
        """Create test image"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_empty_pipeline(self, test_image):
        """Empty pipeline should return original image"""
        pipeline = DegradationPipeline()
        result = pipeline.apply(test_image)
        assert np.array_equal(result, test_image)

    def test_single_degradation(self, test_image):
        """Pipeline with single degradation"""
        pipeline = DegradationPipeline()
        pipeline.add_degradation("blur", 0.5)
        result = pipeline.apply(test_image)
        assert result.shape == test_image.shape

    def test_multiple_degradations(self, test_image):
        """Pipeline with multiple degradations"""
        pipeline = DegradationPipeline()
        pipeline.add_degradation("blur", 0.3)
        pipeline.add_degradation("noise", 0.4)
        pipeline.add_degradation("exposure", 0.7)
        result = pipeline.apply(test_image)
        assert result.shape == test_image.shape

    def test_pipeline_chaining(self, test_image):
        """Should support method chaining"""
        pipeline = (
            DegradationPipeline()
            .add_degradation("blur", 0.3)
            .add_degradation("noise", 0.4)
        )
        result = pipeline.apply(test_image)
        assert result.shape == test_image.shape

    def test_invalid_severity(self):
        """Should reject invalid severity"""
        pipeline = DegradationPipeline()
        with pytest.raises(ValueError):
            pipeline.add_degradation("blur", 1.5)  # > 1.0

    def test_clear_pipeline(self):
        """Should clear degradations"""
        pipeline = DegradationPipeline()
        pipeline.add_degradation("blur", 0.5)
        assert len(pipeline.degradations) == 1
        
        pipeline.clear()
        assert len(pipeline.degradations) == 0


class TestDatasetGenerator:
    """Tests for dataset generator"""

    def test_generator_initialization(self):
        """Should initialize generator"""
        generator = DatasetGenerator()
        assert generator is not None
        assert generator.config.SEED == 42

    def test_quality_labels_configuration(self):
        """Should have proper quality label configuration"""
        config = DatasetConfig()
        assert "ACCEPTABLE" in config.QUALITY_LABELS
        assert "DEGRADED" in config.QUALITY_LABELS
        assert "DEFECTIVE" in config.QUALITY_LABELS

    def test_generate_sample_dataset(self):
        """Should generate sample dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = tmpdir
            create_sample_dataset(output_dir, num_samples=3)
            
            # Check output structure
            assert Path(f"{output_dir}/train").exists()
            assert Path(f"{output_dir}/val").exists()
            assert Path(f"{output_dir}/test").exists()
            
            # Check metadata
            assert Path(f"{output_dir}/train/metadata.jsonl").exists()
            assert Path(f"{output_dir}/dataset_report.json").exists()

    def test_dataset_report_structure(self):
        """Should generate proper report structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = tmpdir
            create_sample_dataset(output_dir, num_samples=2)
            
            # Load and verify report
            report_path = Path(f"{output_dir}/dataset_report.json")
            with open(report_path) as f:
                report = json.load(f)
            
            assert "timestamp" in report
            assert "seed" in report
            assert "total_images" in report
            assert "label_distribution" in report
            assert "train_size" in report
            assert "val_size" in report
            assert "test_size" in report

    def test_no_data_leakage(self):
        """Should ensure no leakage between train/val/test"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = tmpdir
            create_sample_dataset(output_dir, num_samples=10)
            
            # Load metadata for each split
            def load_metadata(split):
                metadata_path = Path(f"{output_dir}/{split}/metadata.jsonl")
                sources = set()
                with open(metadata_path) as f:
                    for line in f:
                        item = json.loads(line)
                        sources.add(item["source_image"])
                return sources

            train_sources = load_metadata("train")
            val_sources = load_metadata("val")
            test_sources = load_metadata("test")
            
            # Check no overlap
            assert len(train_sources & val_sources) == 0, "Train/Val leakage detected"
            assert len(train_sources & test_sources) == 0, "Train/Test leakage detected"
            assert len(val_sources & test_sources) == 0, "Val/Test leakage detected"


class TestImageMetadata:
    """Tests for image metadata"""

    def test_metadata_creation(self):
        """Should create metadata"""
        metadata = ImageMetadata(
            filename="test.jpg",
            source_image="source.jpg",
            quality_label="DEGRADED",
            degradations=[{"type": "blur", "severity": 0.5}],
            generation_timestamp="2024-01-01T00:00:00",
            degradation_seed=42,
        )
        
        assert metadata.filename == "test.jpg"
        assert metadata.quality_label == "DEGRADED"

    def test_metadata_to_dict(self):
        """Should convert to dictionary"""
        from dataclasses import asdict
        
        metadata = ImageMetadata(
            filename="test.jpg",
            source_image="source.jpg",
            quality_label="DEGRADED",
            degradations=[],
            generation_timestamp="2024-01-01T00:00:00",
            degradation_seed=42,
        )
        
        metadata_dict = asdict(metadata)
        assert isinstance(metadata_dict, dict)
        assert "filename" in metadata_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
