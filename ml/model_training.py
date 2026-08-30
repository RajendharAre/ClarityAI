"""
Model Training Script
Orchestrates training of baseline and deep models.
Generates synthetic dataset, trains both models, saves artifacts.
"""

import logging
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

from ml.dataset_generator import DatasetGenerator, DatasetConfig
from ml.feature_extraction import extract_image_features
from ml.baseline_model import BaselineModel, train_baseline_model
from ml.deep_model import DeepModel
from ml.score_fusion import ScoreFusion, AdaptiveScoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Complete model training pipeline orchestrator.
    
    Steps:
    1. Generate or load dataset
    2. Extract features for baseline model
    3. Train baseline model (Random Forest)
    4. Train deep model (Autoencoder on ACCEPTABLE images only)
    5. Validate models
    6. Save artifacts
    """

    def __init__(self, data_dir: str = "./ml/data", models_dir: str = "./ml/models"):
        """
        Initialize training pipeline.
        
        Args:
            data_dir: Directory with train/val/test splits
            models_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training pipeline initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Models directory: {self.models_dir}")

    def load_dataset(
        self,
        split: str = "train",
    ) -> Tuple[list, list, list]:
        """
        Load dataset from JSONL metadata.
        
        Args:
            split: "train", "val", or "test"
        
        Returns:
            (images, labels, metadata_list)
        """
        split_dir = self.data_dir / split
        metadata_path = split_dir / "metadata.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        images = []
        labels = []
        metadata_list = []

        # Quality label to class mapping
        label_to_class = {"ACCEPTABLE": 0, "DEGRADED": 1, "DEFECTIVE": 2}

        with open(metadata_path) as f:
            for line in f:
                import json
                metadata = json.loads(line)

                # Load image
                image_path = split_dir / metadata["filename"]
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue

                images.append(image)
                labels.append(label_to_class[metadata["quality_label"]])
                metadata_list.append(metadata)

        logger.info(f"Loaded {len(images)} images from {split}")
        return images, labels, metadata_list

    def extract_features_for_baseline(
        self,
        images: list,
    ) -> np.ndarray:
        """
        Extract features from images for baseline model.
        
        Args:
            images: List of image arrays
        
        Returns:
            Feature matrix (N, 6)
        """
        logger.info(f"Extracting features from {len(images)} images")

        features = []
        for i, image in enumerate(images):
            try:
                feature_stats = extract_image_features(image)
                feature_vector = np.array([
                    feature_stats.sharpness,
                    feature_stats.brightness,
                    feature_stats.contrast,
                    feature_stats.noise_level,
                    feature_stats.saturation,
                    feature_stats.texture_complexity,
                ])
                features.append(feature_vector)
            except Exception as e:
                logger.error(f"Error extracting features from image {i}: {e}")
                continue

        features = np.array(features)
        logger.info(f"Extracted features: shape={features.shape}")
        return features

    def prepare_deep_model_data(
        self,
        images: list,
        labels: list,
    ) -> Tuple[np.ndarray, list]:
        """
        Prepare images for deep model training.
        
        Only use ACCEPTABLE (label=0) images for training.
        
        Args:
            images: List of image arrays
            labels: List of labels (0, 1, 2)
        
        Returns:
            (acceptable_images, filtered_labels)
        """
        logger.info(f"Preparing data for deep model (selecting ACCEPTABLE only)")

        acceptable_images = []
        for image, label in zip(images, labels):
            if label == 0:  # ACCEPTABLE
                acceptable_images.append(image)

        logger.info(
            f"Selected {len(acceptable_images)} ACCEPTABLE images for deep model training"
        )

        if len(acceptable_images) == 0:
            logger.warning("No ACCEPTABLE images found for deep model training!")

        # Convert to array
        if acceptable_images:
            # Convert (H, W, 3) → (N, H, W, 3)
            acceptable_images = np.array(acceptable_images)

        return acceptable_images, [0] * len(acceptable_images)

    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "random_forest",
    ) -> BaselineModel:
        """
        Train baseline model.
        
        Args:
            X_train: Training features (N, 6)
            y_train: Training labels (N,)
            model_type: "random_forest" or "gradient_boosting"
        
        Returns:
            Trained BaselineModel
        """
        logger.info(f"Training baseline model ({model_type})")

        model = BaselineModel(model_type=model_type)
        metrics = model.train(X_train, y_train)

        logger.info(f"Baseline training metrics: {metrics}")

        model.save(str(self.models_dir))
        return model

    def train_deep(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> DeepModel:
        """
        Train deep model.
        
        Args:
            X_train: Training images (N, H, W, 3)
            epochs: Number of epochs
            batch_size: Batch size
            device: "cpu" or "cuda"
        
        Returns:
            Trained DeepModel
        """
        logger.info(f"Training deep model for {epochs} epochs")

        model = DeepModel(device=device)
        history = model.train_model(X_train, epochs=epochs, batch_size=batch_size)

        logger.info(f"Deep model training history: {history}")

        model.save(str(self.models_dir))
        return model

    def run_full_pipeline(
        self,
        generate_sample_data: bool = True,
        num_sample_images: int = 10,
        deep_model_epochs: int = 50,
        device: str = "cpu",
    ) -> Dict:
        """
        Run complete training pipeline.
        
        Args:
            generate_sample_data: Whether to generate sample dataset
            num_sample_images: Number of sample images to generate
            deep_model_epochs: Epochs for deep model training
            device: "cpu" or "cuda"
        
        Returns:
            Training report dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting complete model training pipeline")
        logger.info("=" * 60)

        # Step 1: Generate or load dataset
        if generate_sample_data or not (self.data_dir / "train").exists():
            logger.info("Generating sample dataset...")
            from ml.dataset_generator import create_sample_dataset
            create_sample_dataset(str(self.data_dir), num_samples=num_sample_images)

        # Step 2: Load training data
        X_images, y_labels, _ = self.load_dataset("train")
        
        if len(X_images) == 0:
            raise RuntimeError("No training images loaded")

        # Step 3: Extract features for baseline
        X_features = self.extract_features_for_baseline(X_images)

        # Step 4: Train baseline model
        baseline_model = self.train_baseline(X_features, np.array(y_labels))

        # Step 5: Prepare and train deep model
        X_acceptable, _ = self.prepare_deep_model_data(X_images, y_labels)
        
        if len(X_acceptable) > 0:
            deep_model = self.train_deep(
                X_acceptable,
                epochs=deep_model_epochs,
                device=device,
            )
        else:
            logger.warning("Skipping deep model training (no ACCEPTABLE images)")
            deep_model = None

        # Step 6: Validation
        logger.info("\nValidating models on test set...")
        X_test_images, y_test_labels, _ = self.load_dataset("test")
        
        if len(X_test_images) > 0:
            X_test_features = self.extract_features_for_baseline(X_test_images)
            
            baseline_preds = []
            for features in X_test_features:
                label, conf = baseline_model.predict(features)
                baseline_preds.append(label)
            
            # Accuracy
            baseline_correct = sum(
                1 for pred, true in zip(baseline_preds, y_test_labels)
                if BaselineModel.LABEL_TO_CLASS[pred] == true
            )
            baseline_acc = baseline_correct / len(y_test_labels)
            
            logger.info(f"Baseline test accuracy: {baseline_acc:.3f}")

        # Step 7: Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_model_path": str(self.models_dir / "baseline_model_random_forest_v1.joblib"),
            "deep_model_path": str(self.models_dir / "deep_model_autoencoder_v1.pt"),
            "training_samples": len(X_images),
            "test_samples": len(X_test_images) if len(X_test_images) > 0 else 0,
            "baseline_accuracy": float(baseline_acc) if len(X_test_images) > 0 else 0,
            "training_status": "COMPLETE",
        }

        report_path = self.models_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("Training pipeline complete!")
        logger.info("=" * 60)
        logger.info(f"Report: {report}")

        return report


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train ClarityAI models")
    parser.add_argument("--data-dir", default="./ml/data", help="Data directory")
    parser.add_argument("--models-dir", default="./ml/models", help="Models directory")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sample images")
    parser.add_argument("--epochs", type=int, default=50, help="Deep model epochs")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    pipeline = ModelTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )

    report = pipeline.run_full_pipeline(
        generate_sample_data=args.generate_data,
        num_sample_images=args.num_samples,
        deep_model_epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
