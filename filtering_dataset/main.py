"""
main.py

Main module for running aircraft image classification using CLIP.
This implementation removes batching for simplicity and provides
more control over the classification process.

Features:
- Configurable via config.yml
- Multiple category classification in a single pass
- Results visualization
- Detailed logging and progress tracking
"""
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Import local modules
from image_filterer import ImageFilterer
import visualization
import pandas as pd

def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Configures and returns a logger instance with timestamped output.

    Args:
        debug: Whether to set logging level to DEBUG.
    """
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)

    return logging.getLogger('main')


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Aircraft image classifier using CLIP"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yml',
        help='Path to YAML configuration file.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging.'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def copy_images_to_categories(
        df: pd.DataFrame,
        dest_root: Path,
        logger: logging.Logger
) -> None:
    """
    Copies images to their respective category folders based on classification.

    Args:
        df: DataFrame with 'path' and 'best_category' columns
        dest_root: Root directory for category folders
        logger: Logger instance
    """
    # Get counts per category for logging
    category_counts = df['best_category'].value_counts().to_dict()

    # Group by category for efficient copying
    for category, group in df.groupby('best_category'):
        # Create category directory
        category_dir = dest_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Copy each image
        for _, row in tqdm(list(group.iterrows()), desc=f"Copying {category} images"):
            src_path = Path(row['path'])
            dst_path = category_dir / src_path.name
            shutil.copy(src_path, dst_path)

        logger.info(f"Copied {len(group)} images to {category_dir}")

    # Log summary
    logger.info("Image distribution by category:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} images")


def main() -> None:
    """
    Main function to run the classification pipeline.
    """
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    logger = setup_logging(args.debug)
    logger.info("Starting aircraft image classification")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Extract path configurations
    dataset_dir = Path(config['paths']['dataset'])
    output_dir = Path(config['paths']['output'])
    results_dir = Path(config['paths']['results'])
    viz_dir = Path(config['paths']['visualizations'])

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Validate dataset exists
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    # Load all image paths
    image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
    image_paths = [str(p) for p in image_files]
    logger.info(f"Found {len(image_paths)} images in dataset")

    if not image_paths:
        logger.error("No images found in dataset directory")
        return

    # Extract model configuration
    model_name = config['model']['name']
    device = config['model']['device'] or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    logger.info(f"Loading CLIP model '{model_name}'...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    logger.info(f"Using device: {device}")

    # Initialize filterer
    filterer = ImageFilterer(model, processor, device)

    # Extract categories for classification
    categories = config['categories']
    default_category = config['classification']['default_category']

    # Start timing
    start_time = time.time()

    # Perform classification
    logger.info("Starting image classification...")
    results_df = filterer.classify_images(
        image_paths,
        categories,
        default_category
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Classification completed in {elapsed_time:.2f} seconds")

    # Save classification results
    if config['classification']['save_results']:
        results_path = results_dir / 'classification_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved classification results to {results_path}")

    # Copy images to category folders if requested
    if config['classification']['copy_to_folders']:
        logger.info("Copying images to category folders...")
        copy_images_to_categories(results_df, output_dir, logger)

    # Generate visualizations if enabled
    if config['visualization']['enabled']:
        logger.info("Generating visualizations...")
        visualization.create_all_visualizations(results_df, viz_dir)

    # Print summary statistics
    category_counts = results_df['best_category'].value_counts()
    logger.info("Classification summary:")
    for category, count in category_counts.items():
        percentage = (count / len(results_df)) * 100
        logger.info(f"  {category}: {count} images ({percentage:.1f}%)")

    logger.info("Classification pipeline completed successfully")


if __name__ == '__main__':
    main()