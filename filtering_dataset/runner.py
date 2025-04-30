"""
runner.py

Orchestrates a non-destructive, multi-stage filtering pipeline that uses a more robust
classification approach. Images are evaluated against all filter categories simultaneously
and assigned to their best-matching category based on confidence scores and thresholds.
"""
import logging
from pathlib import Path
import argparse
import shutil
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import numpy as np

from image_filterer import ImageFilterer, FilterConfig


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments, providing default config path.
    """
    script_dir = Path(__file__).parent.resolve()
    default_cfg = script_dir / 'prompts.yml'

    parser = argparse.ArgumentParser(
        description="Improved multi-category classifier for military aircraft images"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=str(default_cfg),
        help='Path to YAML configuration file.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for image processing.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging and save classification scores.'
    )
    return parser.parse_args()


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Configures and returns a logger instance with timestamped output.

    Args:
        debug: Whether to set logging level to DEBUG.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('runner')


def evaluate_all_categories(
    image_paths: List[str],
    filterer: ImageFilterer,
    stages: List[FilterConfig],
    batch_size: int
) -> pd.DataFrame:
    """
    Evaluates all images against all filter categories in a single pass.

    Args:
        image_paths: List of paths to images to evaluate
        filterer: ImageFilterer instance
        stages: List of filter configurations
        batch_size: Number of images to process in each batch

    Returns:
        DataFrame with paths and confidence scores for each category
    """
    # Extract all positive prompts from all stages
    all_prompts = []
    stage_names = []
    thresholds = {}

    # Build combined prompt list while tracking which prompts belong to which stages
    prompt_to_stage_map = {}
    prompt_index = 0

    for stage in stages:
        stage_name = stage['name']
        stage_names.append(stage_name)
        thresholds[stage_name] = float(stage['threshold'])

        for prompt in stage['positive_texts']:
            all_prompts.append(prompt)
            prompt_to_stage_map[prompt_index] = stage_name
            prompt_index += 1

    # Process all images against all prompts in a single pass
    results_df = filterer.process_batch(image_paths, all_prompts, batch_size)

    # This will hold our final DataFrame with confidence scores per category
    classified_df = pd.DataFrame()
    classified_df['path'] = results_df['path']
    classified_df['filename'] = results_df['filename']

    # Initialize columns for each category
    for stage_name in stage_names:
        classified_df[f'score_{stage_name}'] = 0.0

    # Group prompt scores by their category
    # The original results_df has one column 'confidence' which contains
    # the maximum confidence across all prompts in all_prompts
    # We need to extract confidence scores for each prompt and map them back to their categories

    # Get detailed scores for each prompt
    detailed_scores = filterer.get_detailed_scores()

    # Map scores back to categories
    for i, row in classified_df.iterrows():
        img_path = row['path']
        if img_path in detailed_scores:
            scores = detailed_scores[img_path]

            # Initialize best score per category
            category_scores = {stage_name: 0.0 for stage_name in stage_names}

            # Find the best score for each category
            for prompt_idx, score in enumerate(scores):
                if prompt_idx in prompt_to_stage_map:
                    category = prompt_to_stage_map[prompt_idx]
                    category_scores[category] = max(category_scores[category], score)

            # Set the best score for each category
            for category, score in category_scores.items():
                classified_df.loc[i, f'score_{category}'] = score

    # Add best_category column based on scores and thresholds
    # Initialize with 'normal_images' as default
    classified_df['best_category'] = 'normal_images'
    classified_df['best_score'] = 0.0

    # Determine best category for each image
    for i, row in classified_df.iterrows():
        best_category = 'normal_images'
        best_score = 0.0

        for stage_name in stage_names:
            score = row[f'score_{stage_name}']
            threshold = thresholds[stage_name]

            # Only consider scores that meet their category's threshold
            if score >= threshold and score > best_score:
                best_category = stage_name
                best_score = score

        # Update the DataFrame
        classified_df.loc[i, 'best_category'] = best_category
        classified_df.loc[i, 'best_score'] = best_score

    return classified_df


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
        for _, row in group.iterrows():
            src_path = Path(row['path'])
            dst_path = category_dir / src_path.name
            shutil.copy(src_path, dst_path)

        logger.info(f"Copied {len(group)} images to {category_dir}")

    # Log summary
    logger.info("Image distribution by category:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} images")


def main():
    args = parse_args()
    logger = setup_logging(args.debug)

    # Define project directories
    filtering_dir = Path(__file__).parent.resolve()
    project_root  = filtering_dir.parent.resolve()
    dataset_dir   = project_root / 'dataset'
    filter_root   = project_root / 'filtering'
    results_dir   = filter_root / 'results'

    # Validate dataset exists
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory missing: {dataset_dir}")
        return

    # Load all image paths
    image_files = list(dataset_dir.glob('*.jpg')) + list(dataset_dir.glob('*.png'))
    image_paths = [str(p) for p in image_files]
    logger.info(f"Loaded {len(image_paths)} images from dataset.")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return
    cfg = yaml.safe_load(config_path.read_text())
    stages = cfg.get('filters', [])
    model_name = cfg.get('model_name', 'openai/clip-vit-base-patch32')

    if not stages:
        logger.error("No filter stages defined in configuration.")
        return

    # Load CLIP model and processor
    logger.info(f"Loading CLIP model '{model_name}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.to(device)
    logger.info(f"Using device: {device}")

    # Initialize filterer
    filterer = ImageFilterer(model, processor, device)

    # Process all images against all categories at once
    logger.info("Classifying all images against all categories...")
    classified_df = evaluate_all_categories(
        image_paths,
        filterer,
        stages,
        args.batch_size
    )

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save classification results for debugging if requested
    if args.debug:
        results_path = results_dir / 'classification_results.csv'
        classified_df.to_csv(results_path, index=False)
        logger.debug(f"Saved detailed classification results to {results_path}")

    # Copy images to their respective category folders
    copy_images_to_categories(classified_df, filter_root, logger)

    logger.info("Classification pipeline complete.")


if __name__ == '__main__':
    main()