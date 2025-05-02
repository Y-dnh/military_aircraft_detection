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
- Support for negative prompts and minimum confidence difference
"""
import argparse
import logging
import shutil
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Import local modules
from image_filterer import ImageFilterer
import pandas as pd
import classification_analysis

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


def run_evaluation(
        manual_dir: Path,
        clip_results_df: pd.DataFrame,
        output_dir: Path,
        config: dict,
        logger: logging.Logger
) -> None:
    """
    Evaluates classification performance by comparing with manual classification.

    Args:
        manual_dir: Directory containing manually classified images
        clip_results_df: DataFrame with CLIP classification results
        output_dir: Directory to save evaluation results
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Starting evaluation against manual classification...")

    # Extract evaluation settings from config
    low_quality_category = config.get('evaluation', {}).get('low_quality_category', 'shit_data')
    ignore_categories = config.get('evaluation', {}).get('ignore_in_metrics', [])

    # Ensure the evaluation directory exists
    eval_dir = output_dir / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Run the evaluation
    results = classification_analysis.evaluate_classification(
        manual_dir=manual_dir,
        clip_results_df=clip_results_df,
        output_dir=eval_dir,
        low_quality_category=low_quality_category,
        ignore_in_metrics=ignore_categories
    )

    logger.info(f"Evaluation completed. Report saved to {eval_dir / 'evaluation_report.md'}")

    # Log key metrics
    logger.info(f"Overall accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info(f"F1 score (weighted): {results['metrics']['f1_weighted']:.4f}")

    if results['misclassifications']:
        error_rate = results['misclassifications']['misclassification_rate'] * 100
        logger.info(f"Misclassification rate: {error_rate:.2f}%")


def load_model(model_name: str, logger: logging.Logger):
    """
    Loads a CLIP model. Supports both standard HuggingFace models
    and local fine-tuned models in .pt format.

    Args:
        model_name: Path to model or HuggingFace model identifier
        logger: Logger instance

    Returns:
        tuple: (model, processor)
    """
    # Check if model is a local .pt file
    if model_name.endswith('.pt') and os.path.exists(model_name):
        logger.info(f"Loading local PyTorch model: {model_name}")
        try:
            # First load a base CLIP model to get the architecture
            base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Load the state dictionary
            state_dict = torch.load(model_name,
                                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # If the loaded object is already a model (not just a state dict), use it directly
            if isinstance(state_dict, torch.nn.Module):
                logger.info("Loaded object is already a model instance")
                model = state_dict
            else:
                logger.info("Loaded object is a state dictionary, applying to base model")

                # Handle different saving formats
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    # It's wrapped in a dictionary, common when saving with optimizer states
                    base_model.load_state_dict(state_dict['model_state_dict'])
                else:
                    # Try to load directly, this might fail if keys don't match exactly
                    try:
                        base_model.load_state_dict(state_dict)
                    except Exception as e:
                        logger.warning(f"Direct loading failed: {e}")
                        logger.info("Trying to load with more flexible key matching...")
                        # More flexible loading that ignores missing and unexpected keys
                        base_model.load_state_dict(state_dict, strict=False)

                model = base_model

            logger.info("Model successfully loaded from local file")
            return model, processor

        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise

    # If path points to a directory, try to load as a local HuggingFace model
    elif os.path.isdir(model_name):
        logger.info(f"Loading local HuggingFace model: {model_name}")
        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            return model, processor
        except Exception as e:
            logger.error(f"Error loading model from directory: {str(e)}")
            raise

    # Otherwise, assume it's a HuggingFace model identifier
    else:
        logger.info(f"Loading model from HuggingFace: {model_name}")
        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            return model, processor
        except Exception as e:
            logger.error(f"Error loading model from HuggingFace: {str(e)}")
            raise


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
    try:
        model, processor = load_model(model_name, logger)
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.info("Перевірте, що шлях до моделі вказано правильно в config.yml")
        return

    # Initialize filterer
    filterer = ImageFilterer(model, processor, device)

    # Extract categories for classification
    categories = config['categories']
    default_category = config['classification']['default_category']

    # Extract additional classification settings
    use_negative_prompts = config['classification'].get('use_negative_prompts', False)
    confidence_strategy = config['classification'].get('confidence_strategy', 'fixed')

    logger.info(f"Negative prompts: {'enabled' if use_negative_prompts else 'disabled'}")
    logger.info(f"Confidence strategy: {confidence_strategy}")

    # Start timing
    start_time = time.time()

    # Perform classification
    logger.info("Starting image classification...")
    results_df = filterer.classify_images(
        image_paths,
        categories,
        default_category,
        use_negative_prompts=use_negative_prompts,
        confidence_strategy=confidence_strategy
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
        classification_analysis.create_all_visualizations(results_df, viz_dir)

    # Run evaluation if manual classification is available
    if 'manual_classification_dir' in config.get('evaluation', {}):
        manual_dir = Path(config['evaluation']['manual_classification_dir'])
        if manual_dir.is_dir():
            run_evaluation(
                manual_dir=manual_dir,
                clip_results_df=results_df,
                output_dir=Path(config['paths']['results']),
                config=config,
                logger=logger
            )
        else:
            logger.warning(f"Manual classification directory not found: {manual_dir}")

    # Print summary statistics
    category_counts = results_df['best_category'].value_counts()
    logger.info("Classification summary:")
    for category, count in category_counts.items():
        percentage = (count / len(results_df)) * 100
        logger.info(f"  {category}: {count} images ({percentage:.1f}%)")

    logger.info("Classification pipeline completed successfully")


if __name__ == '__main__':
    main()