"""
Dataset Generation Script
Generates synthetic dataset for ClarityAI training.
Run this script to create train/val/test data.
"""

import logging
import sys
from pathlib import Path
from ml.dataset_generator import DatasetGenerator, create_sample_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main dataset generation entry point"""
    
    logger.info("ClarityAI Dataset Generation Script")
    logger.info("=" * 50)

    # Option 1: Generate from existing images
    source_dir = "./ml/data/source_images"  # User should populate this
    output_dir = "./ml/data"

    # Check if source images directory exists
    if not Path(source_dir).exists():
        logger.warning(f"Source images directory not found: {source_dir}")
        logger.info("Creating sample dataset for demonstration...")
        
        # Create sample dataset
        create_sample_dataset(output_dir, num_samples=5)
        
        logger.info("\n" + "=" * 50)
        logger.info("Sample dataset created successfully!")
        logger.info(f"Dataset location: {output_dir}")
        logger.info("=" * 50)
        return

    # Generate from real source images
    logger.info(f"Generating dataset from: {source_dir}")
    
    try:
        generator = DatasetGenerator()
        report = generator.generate_dataset(source_dir, output_dir)
        
        logger.info("\n" + "=" * 50)
        logger.info("Dataset generation complete!")
        logger.info(f"Report: {report}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
