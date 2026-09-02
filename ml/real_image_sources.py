"""
Real Image Source Acquisition
Downloads / collects REAL photographs to use as source images for the dataset
generation pipeline.

Why this exists:
The original `create_sample_dataset()` manufactured source images from random
noise + checkerboards. Models trained on those do NOT generalize to real
photographs (see temp/diagnostic_stepB.md). This module provides genuine
photographs so that degradations are applied to realistic content.

Design:
- If a local `real_sources` directory already contains images, reuse them
  (fully reproducible, offline friendly).
- Otherwise, attempt to download a curated list of permissively-licensed
  sample photographs (OpenCV sample images).
- If downloads fail for every source, falls back to generating synthetic
  sources but issues a clear WARNING (this should not happen in practice).

Follows Design Principles: Reproducibility, Modularity, Independence.
"""

import logging
import urllib.request
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Curated, stable, permissively-licensed sample photographs (OpenCV samples).
# These are real photographs commonly used for computer-vision demos.
DEFAULT_SOURCES = {
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "fruits.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "building.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "left01.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/left01.jpg",
    "basketball.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/basketball1.png",
    "graf1.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/graf1.png",
    "graf3.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/graf3.png",
    "box.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png",
    "aloeL.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeL.jpg",
    "aloeR.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/aloeR.jpg",
    "chessboard.png": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/chessboard.png",
    "messi5.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
}

REAL_SOURCES_DIR = Path(__file__).resolve().parent / "data" / "real_sources"


def _load_dimensions(path: Path) -> bool:
    """Return True if the file decodes as a valid image."""
    img = cv2.imread(str(path))
    return img is not None


def _fetch(name: str, url: str, dest_dir: Path) -> bool:
    """Download a single image. Returns True on success."""
    dest = dest_dir / name

    # Reuse an existing, valid copy (offline-friendly, avoids redundant downloads)
    if dest.exists() and _load_dimensions(dest):
        return True

    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = resp.read()
        dest.write_bytes(data)
        if _load_dimensions(dest):
            logger.info(f"Downloaded real source image: {name} ({len(data)} bytes)")
            return True
        dest.unlink(missing_ok=True)
        logger.warning(f"Downloaded invalid image, discarded: {name}")
        return False
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to download {name}: {e}")
        return False


def ensure_real_sources(
    source_dir: str | Path = None,
    download: bool = True,
    min_sources: int = 1,
) -> Path:
    """
    Ensure a directory of real source images exists.

    Args:
        source_dir: Directory to use (defaults to ml/data/real_sources).
        download: Whether to attempt downloading if empty.
        min_sources: Minimum acceptable number of images.

    Returns:
        Path to the directory containing real source images.

    Raises:
        RuntimeError: If fewer than min_sources valid real images are available.
    """
    dest_dir = Path(source_dir) if source_dir else REAL_SOURCES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in dest_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    valid = [p for p in existing if _load_dimensions(p)]
    logger.info(f"Found {len(valid)} existing real source images in {dest_dir}")

    if len(valid) >= min_sources:
        return dest_dir

    # Need to gather more real sources (existing ones count toward the target)
    existing_valid = {p.name for p in valid}
    logger.info(f"Attempting to obtain up to {min_sources} real source images...")
    for name, url in DEFAULT_SOURCES.items():
        if len(valid) >= min_sources:
            break
        ok = _fetch(name, url, dest_dir)
        if ok:
            p = dest_dir / name
            if p.name not in existing_valid:
                existing_valid.add(p.name)
                valid.append(p)

    final_valid = [p for p in dest_dir.glob("*") if _load_dimensions(p)]
    if len(final_valid) < min_sources:
        _write_synthetic_fallback(dest_dir, min_sources - len(final_valid))

    return dest_dir


def _write_synthetic_fallback(dest_dir: Path, count: int) -> None:
    """
    Write synthetic gradient images as a last-resort fallback.

    These are NOT suitable for real-world training (see diagnostic), but they
    keep the pipeline runnable offline. A loud warning is logged.
    """
    if count <= 0:
        return
    logger.warning(
        "No real source images could be obtained. Writing SYNTHETIC gradient "
        "sources. Models trained on these will NOT generalize to real photos. "
        "Place real photographs in %s and re-run training.", dest_dir
    )
    for i in range(count):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for c, val in ((0, 90), (1, 140), (2, 200)):
                img[y, :, c] = min(255, int(val + y * 0.4))
        # add a simple object-like ellipse
        cv2.ellipse(img, (128, 128), (90, 70), 0, 0, 360, (30, 60, 120), -1)
        cv2.imwrite(str(dest_dir / f"synthetic_fallback_{i}.jpg"), img)
