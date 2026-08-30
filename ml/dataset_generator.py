"""
Dataset Generation Module
Creates synthetic image dataset with labeled degradations.
Follows Design Principles: Reproducibility, Modularity, Security
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import random

from ml.degradations import DegradationFactory, DegradationPipeline

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for generated image"""
    filename: str
    source_image: str
    quality_label: str  # ACCEPTABLE, DEGRADED, DEFECTIVE
    degradations: List[Dict]  # List of applied degradations
    generation_timestamp: str
    degradation_seed: int


class DatasetConfig:
    """
    Dataset generation configuration.
    Ensures reproducibility.
    """

    # Random seed for reproducibility
    SEED = 42

    # Quality label definitions
    QUALITY_LABELS = {
        "ACCEPTABLE": {
            "description": "No significant quality issues",
            "degradation_types": [],
            "severity_range": (0.0, 0.0),
        },
        "DEGRADED": {
            "description": "One or more quality issues present but still usable",
            "degradation_types": ["blur", "exposure", "noise", "salt_pepper_noise", "jpeg_compression"],
            "severity_range": (0.2, 0.5),
        },
        "DEFECTIVE": {
            "description": "Severe quality problems or visible defects",
            "degradation_types": ["blur", "exposure", "noise", "salt_pepper_noise", "jpeg_compression", "scratch", "spot"],
            "severity_range": (0.5, 1.0),
        },
    }

    # Dataset splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Degradation counts per source image
    DEGRADATIONS_PER_LABEL = {
        "ACCEPTABLE": 1,  # 1 clean copy
        "DEGRADED": 3,    # 3 different degradations
        "DEFECTIVE": 2,   # 2 heavy degradations
    }

    # Output directory structure
    OUTPUT_DIR = "./ml/data"
    TRAIN_DIR = f"{OUTPUT_DIR}/train"
    VAL_DIR = f"{OUTPUT_DIR}/val"
    TEST_DIR = f"{OUTPUT_DIR}/test"


class DatasetGenerator:
    """
    Generates synthetic dataset with labeled degradations.
    Follows Design Patterns: Factory, Builder (GoF)
    """

    def __init__(self, config: DatasetConfig = DatasetConfig()):
        """
        Initialize dataset generator.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        logger.info(f"DatasetGenerator initialized with seed {config.SEED}")

    def generate_dataset(
        self,
        source_images_dir: str,
        output_dir: str = None,
        num_degradations_per_label: Dict[str, int] = None,
    ) -> Dict:
        """
        Generate complete dataset from source images.
        
        Args:
            source_images_dir: Directory containing source images
            output_dir: Output directory (uses config default if None)
            num_degradations_per_label: Override degradation counts
        
        Returns:
            Dictionary with generation statistics
        """
        if output_dir is None:
            output_dir = self.config.OUTPUT_DIR

        if num_degradations_per_label is None:
            num_degradations_per_label = self.config.DEGRADATIONS_PER_LABEL

        logger.info(f"Generating dataset from {source_images_dir}")

        # Load source images
        source_images = self._load_source_images(source_images_dir)
        logger.info(f"Loaded {len(source_images)} source images")

        if not source_images:
            raise ValueError(f"No images found in {source_images_dir}")

        # Create output directories
        self._create_output_structure(output_dir)

        # Generate degraded images
        all_degraded = []
        for quality_label in self.config.QUALITY_LABELS.keys():
            degraded = self._generate_degraded_images(
                source_images,
                quality_label,
                num_degradations_per_label[quality_label],
            )
            all_degraded.extend(degraded)
            logger.info(f"Generated {len(degraded)} {quality_label} images")

        # Split into train/val/test
        train_set, val_set, test_set = self._split_dataset(all_degraded)

        # Save images and metadata
        self._save_dataset(
            train_set, val_set, test_set, output_dir
        )

        # Generate report
        report = self._generate_report(
            all_degraded, train_set, val_set, test_set, output_dir
        )

        logger.info(f"Dataset generation complete: {report}")
        return report

    def _load_source_images(self, source_dir: str) -> List[Tuple[str, np.ndarray]]:
        """Load source images from directory"""
        images = []
        source_path = Path(source_dir)

        if not source_path.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return images

        for image_file in source_path.glob("*"):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                logger.warning(f"Failed to load image: {image_file}")
                continue

            images.append((image_file.name, image))

        return images

    def _generate_degraded_images(
        self,
        source_images: List[Tuple[str, np.ndarray]],
        quality_label: str,
        count_per_image: int,
    ) -> List[Tuple[np.ndarray, ImageMetadata]]:
        """
        Generate degraded images for a quality label.
        
        Args:
            source_images: List of (filename, image) tuples
            quality_label: Quality label (ACCEPTABLE, DEGRADED, DEFECTIVE)
            count_per_image: Number of degradations per source image
        
        Returns:
            List of (degraded_image, metadata) tuples
        """
        label_config = self.config.QUALITY_LABELS[quality_label]
        degraded_images = []

        for source_name, source_image in source_images:
            for i in range(count_per_image):
                # Create degradation pipeline
                pipeline = self._create_degradation_pipeline(
                    quality_label,
                    label_config,
                )

                # Apply degradations
                degraded_image = pipeline.apply(source_image)

                # Create metadata
                metadata = ImageMetadata(
                    filename=f"{source_name.split('.')[0]}_{quality_label.lower()}_{i}.jpg",
                    source_image=source_name,
                    quality_label=quality_label,
                    degradations=[
                        {"type": deg_type, "severity": severity}
                        for deg_type, severity in pipeline.degradations
                    ],
                    generation_timestamp=datetime.now().isoformat(),
                    degradation_seed=random.randint(0, 2**32 - 1),
                )

                degraded_images.append((degraded_image, metadata))

        return degraded_images

    def _create_degradation_pipeline(
        self,
        quality_label: str,
        label_config: Dict,
    ) -> DegradationPipeline:
        """
        Create degradation pipeline for quality label.
        
        Args:
            quality_label: Quality label
            label_config: Label configuration
        
        Returns:
            DegradationPipeline
        """
        pipeline = DegradationPipeline()

        # ACCEPTABLE: no degradations
        if quality_label == "ACCEPTABLE":
            return pipeline

        # Get configuration
        degradation_types = label_config["degradation_types"]
        min_severity, max_severity = label_config["severity_range"]

        # Randomly select 1-2 degradations
        num_degradations = random.randint(1, min(2, len(degradation_types)))
        selected_types = random.sample(degradation_types, num_degradations)

        # Add degradations with random severities
        for deg_type in selected_types:
            severity = random.uniform(min_severity, max_severity)
            pipeline.add_degradation(deg_type, severity)

        return pipeline

    def _split_dataset(
        self,
        all_images: List[Tuple[np.ndarray, ImageMetadata]],
    ) -> Tuple[List, List, List]:
        """
        Split dataset into train/val/test with no leakage.
        Ensures same source image doesn't appear in multiple splits.
        
        Args:
            all_images: All generated images
        
        Returns:
            (train_set, val_set, test_set)
        """
        # Group by source image
        source_groups = {}
        for image, metadata in all_images:
            source = metadata.source_image
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((image, metadata))

        # Split source images
        sources = list(source_groups.keys())
        random.shuffle(sources)

        train_count = int(len(sources) * self.config.TRAIN_RATIO)
        val_count = int(len(sources) * self.config.VAL_RATIO)

        train_sources = sources[:train_count]
        val_sources = sources[train_count:train_count + val_count]
        test_sources = sources[train_count + val_count:]

        # Build sets
        train_set = []
        for source in train_sources:
            train_set.extend(source_groups[source])

        val_set = []
        for source in val_sources:
            val_set.extend(source_groups[source])

        test_set = []
        for source in test_sources:
            test_set.extend(source_groups[source])

        logger.info(
            f"Dataset split: Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}"
        )
        logger.info(
            f"Source split: Train {len(train_sources)}, Val {len(val_sources)}, Test {len(test_sources)}"
        )

        return train_set, val_set, test_set

    def _create_output_structure(self, output_dir: str) -> None:
        """Create output directory structure"""
        for subdir in [
            output_dir,
            f"{output_dir}/train",
            f"{output_dir}/val",
            f"{output_dir}/test",
        ]:
            Path(subdir).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {subdir}")

    def _save_dataset(
        self,
        train_set: List,
        val_set: List,
        test_set: List,
        output_dir: str,
    ) -> None:
        """
        Save images and metadata to disk.
        
        Args:
            train_set: Training set
            val_set: Validation set
            test_set: Test set
            output_dir: Output directory
        """
        # Save each set
        self._save_split(train_set, f"{output_dir}/train")
        self._save_split(val_set, f"{output_dir}/val")
        self._save_split(test_set, f"{output_dir}/test")

    def _save_split(self, image_set: List, split_dir: str) -> None:
        """
        Save images and metadata for a split.
        
        Args:
            image_set: List of (image, metadata) tuples
            split_dir: Split directory
        """
        metadata_list = []

        for image, metadata in image_set:
            # Save image
            image_path = Path(split_dir) / metadata.filename
            cv2.imwrite(str(image_path), image)

            # Collect metadata
            metadata_list.append(asdict(metadata))

        # Save metadata as JSONL (one JSON per line)
        metadata_path = Path(split_dir) / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for item in metadata_list:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Saved {len(image_set)} images to {split_dir}")
        logger.info(f"Saved metadata to {metadata_path}")

    def _generate_report(
        self,
        all_images: List,
        train_set: List,
        val_set: List,
        test_set: List,
        output_dir: str,
    ) -> Dict:
        """
        Generate dataset generation report.
        
        Args:
            all_images: All generated images
            train_set: Training set
            val_set: Validation set
            test_set: Test set
            output_dir: Output directory
        
        Returns:
            Report dictionary
        """
        # Count by label
        label_counts = {}
        for _, metadata in all_images:
            label = metadata.quality_label
            label_counts[label] = label_counts.get(label, 0) + 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.config.SEED,
            "total_images": len(all_images),
            "label_distribution": label_counts,
            "train_size": len(train_set),
            "val_size": len(val_set),
            "test_size": len(test_set),
            "output_directory": output_dir,
        }

        # Save report
        report_path = Path(output_dir) / "dataset_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved dataset report to {report_path}")
        return report


def create_sample_dataset(output_dir: str = "./ml/data", num_samples: int = 5) -> None:
    """
    Create sample dataset with synthetic images for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of sample images to generate
    """
    logger.info(f"Creating sample dataset with {num_samples} sample images")

    # Create temporary directory for source images
    temp_dir = "./ml/data/temp_sources"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Generate synthetic source images
    for i in range(num_samples):
        # Create random image (natural-looking with gradients)
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        # Add some structure (checkerboard pattern)
        for y in range(0, 256, 32):
            for x in range(0, 256, 32):
                if (x // 32 + y // 32) % 2 == 0:
                    image[y:y+32, x:x+32] = 200
                else:
                    image[y:y+32, x:x+32] = 100

        # Save source image
        cv2.imwrite(f"{temp_dir}/sample_{i}.jpg", image)

    logger.info(f"Created {num_samples} sample source images")

    # Generate dataset
    generator = DatasetGenerator()
    report = generator.generate_dataset(temp_dir, output_dir)

    logger.info(f"Sample dataset created: {report}")

    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temporary directory")
