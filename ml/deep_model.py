"""
Deep Learning Anomaly Detection Model
Fine-tuned CNN for detecting quality anomalies/defects.
Trained only on "ACCEPTABLE" (normal) images using reconstruction-based anomaly scoring.
Follows Design Principle: Modularity, Interpretability, Reproducibility
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict, List, Any
import json

logger = logging.getLogger(__name__)


class AutoencoderModel(nn.Module):
    """
    Lightweight autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: Conv layers → latent space (128-dim)
    - Decoder: Transpose conv layers → reconstructs image
    - Loss: Reconstruction MSE (low on normal, high on anomalies)
    
    Design: Inspired by PatchCore and simple reconstruction-based anomaly detection
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 128):
        """
        Initialize autoencoder.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            latent_dim: Dimensionality of latent space
        """
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Encoder: Input (B, 3, H, W) → Latent (B, latent_dim)
        self.encoder = nn.Sequential(
            # Layer 1: Conv 3→32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 2: Conv 32→64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 3: Conv 64→128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Global average pooling → (B, 128)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Latent projection (optional: learn a lower-dim representation)
        self.fc_encode = nn.Linear(128, latent_dim)

        # Decoder: Latent (B, latent_dim) → Image (B, 3, H, W)
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)  # (B, 128, 8, 8)

        self.decoder = nn.Sequential(
            # Layer 1: Transpose Conv 128→64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Layer 2: Transpose Conv 64→32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Layer 3: Transpose Conv 32→3
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output normalized to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent vector"""
        features = self.encoder(x)
        latent = self.fc_encode(features)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image"""
        features = self.fc_decode(z)
        features = features.view(-1, 128, 8, 8)
        reconstruction = self.decoder(features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            (reconstruction, latent)
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class DeepModel:
    """
    Deep learning model for anomaly detection.
    
    Training:
    - Trained ONLY on "ACCEPTABLE" (normal) images
    - Loss: Reconstruction MSE
    - Lower reconstruction error = more normal/acceptable
    - Higher reconstruction error = more anomalous/defective
    
    Design Pattern: Adapter (wraps PyTorch model for easy training/inference)
    """

    def __init__(self, device: str = None):
        """
        Initialize deep model.
        
        Args:
            device: "cuda" or "cpu" (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = AutoencoderModel(input_channels=3, latent_dim=128)
        self.model = self.model.to(device)
        self.is_trained = False

        logger.info(f"DeepModel initialized on device: {device}")

    def train_model(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        val_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """
        Train autoencoder on normal images.
        
        Args:
            X_train: Training images (N, 3, H, W) or (N, H, W, 3) - normalized to [0, 1]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
        
        Returns:
            Training history with loss metrics
        """
        logger.info(f"Training deep model for {epochs} epochs on {len(X_train)} images")

        # Convert to torch tensors
        if X_train.shape[-1] == 3 and X_train.shape[1] != 3:
            # Convert (N, H, W, 3) to (N, 3, H, W)
            X_train = np.transpose(X_train, (0, 3, 1, 2))

        # Normalize to [0, 1]
        X_train = X_train.astype(np.float32) / 255.0 if X_train.max() > 1 else X_train.astype(
            np.float32
        )

        # Split validation
        val_idx = int(len(X_train) * (1 - val_split))
        X_train_split = X_train[:val_idx]
        X_val_split = X_train[val_idx:]

        # Create dataloaders
        train_dataset = TensorDataset(torch.from_numpy(X_train_split))
        val_dataset = TensorDataset(torch.from_numpy(X_val_split))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        # Training loop
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()

                recon, _ = self.model(x)
                loss = criterion(recon, x)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    recon, _ = self.model(x)
                    loss = criterion(recon, x)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        logger.info(f"Training complete. Final Val Loss: {val_loss:.4f}")
        self.is_trained = True
        return history

    def compute_anomaly_score(self, images: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for images.
        
        Args:
            images: Image array (N, 3, H, W) or (N, H, W, 3) - [0, 1] normalized
        
        Returns:
            Anomaly scores (N,) - higher = more anomalous
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        # Convert format if needed
        if images.ndim == 4 and images.shape[-1] == 3 and images.shape[1] != 3:
            images = np.transpose(images, (0, 3, 1, 2))

        # Normalize
        images = images.astype(np.float32) / 255.0 if images.max() > 1 else images.astype(
            np.float32
        )

        self.model.eval()
        scores = []

        with torch.no_grad():
            for image in images:
                x = torch.from_numpy(image).unsqueeze(0).to(self.device)
                recon, _ = self.model(x)

                # Reconstruction error = anomaly score
                mse = torch.mean((recon - x) ** 2).item()
                scores.append(mse)

        return np.array(scores)

    def save(self, model_dir: str) -> str:
        """
        Save model weights.
        
        Args:
            model_dir: Directory to save model
        
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        model_path = Path(model_dir) / "deep_model_autoencoder_v1.pt"
        torch.save(self.model.state_dict(), model_path)

        metadata = {
            "model_type": "autoencoder",
            "input_channels": 3,
            "latent_dim": 128,
            "device": self.device,
        }

        metadata_path = Path(model_dir) / "deep_metadata_autoencoder_v1.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved deep model to {model_path}")
        logger.info(f"Saved metadata to {metadata_path}")

        return str(model_path)

    @staticmethod
    def load(model_path: str, device: str = None) -> "DeepModel":
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model
            device: Device to load on
        
        Returns:
            Loaded DeepModel instance
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = DeepModel(device=device)
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        model.is_trained = True

        logger.info(f"Loaded deep model from {model_path}")
        return model
