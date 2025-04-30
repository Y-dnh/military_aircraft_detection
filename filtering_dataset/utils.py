"""
utils.py

Utility functions for the aircraft image classification pipeline.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensures a directory exists, creating it if necessary.

    Args:
        path: Directory path as string or Path object

    Returns:
        Path object of the ensured directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_yaml(file_path: Union[str, Path]) -> dict:
    """
    Safely loads a YAML file.

    Args:
        file_path: Path to the YAML file

    Returns:
        Dictionary containing the YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {file_path}: {e}")
            raise


def save_yaml(data: dict, file_path: Union[str, Path]) -> None:
    """
    Saves data to a YAML file.

    Args:
        data: Dictionary to save
        file_path: Path where to save the YAML file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """
    Saves data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def analyze_results(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyzes classification results to extract useful statistics.

    Args:
        df: DataFrame with classification results

    Returns:
        Dictionary with analysis results
    """
    total_images = len(df)

    # Count images per category
    category_counts = df['best_category'].value_counts().to_dict()

    # Calculate percentages
    category_percentages = {
        cat: (count / total_images) * 100
        for cat, count in category_counts.items()
    }

    # Calculate average confidence score per category
    avg_scores = {}
    for category in df['best_category'].unique():
        category_df = df[df['best_category'] == category]
        avg_scores[category] = category_df['best_score'].mean()

    # Find most borderline cases (lowest confidence)
    df_sorted = df.sort_values('best_score')
    borderline_cases = df_sorted.head(10)[['filename', 'best_category', 'best_score']]

    # Find highest confidence cases
    confident_cases = df_sorted.tail(10)[['filename', 'best_category', 'best_score']]

    return {
        'total_images': total_images,
        'category_counts': category_counts,
        'category_percentages': category_percentages,
        'average_scores': avg_scores,
        'borderline_cases': borderline_cases.to_dict('records'),
        'confident_cases': confident_cases.to_dict('records')
    }


def save_analysis(analysis: dict, output_dir: Path) -> None:
    """
    Saves analysis results to files.

    Args:
        analysis: Dictionary with analysis results
        output_dir: Directory to save analysis files
    """
    # Ensure directory exists
    ensure_directory(output_dir)

    # Save as JSON
    save_json(analysis, output_dir / 'analysis.json')

    # Create a markdown summary
    with open(output_dir / 'summary.md', 'w', encoding='utf-8') as f:
        f.write("# Classification Results Summary\n\n")

        f.write(f"## Overview\n")
        f.write(f"Total images analyzed: {analysis['total_images']}\n\n")

        f.write("## Category Distribution\n")
        for cat, count in analysis['category_counts'].items():
            percentage = analysis['category_percentages'][cat]
            f.write(f"- **{cat}**: {count} images ({percentage:.1f}%)\n")
        f.write("\n")

        f.write("## Average Confidence Scores\n")
        for cat, score in analysis['average_scores'].items():
            f.write(f"- **{cat}**: {score:.4f}\n")
        f.write("\n")

        f.write("## Borderline Cases (Lowest Confidence)\n")
        f.write("| Filename | Category | Confidence Score |\n")
        f.write("|----------|----------|------------------|\n")
        for case in analysis['borderline_cases']:
            f.write(f"| {case['filename']} | {case['best_category']} | {case['best_score']:.4f} |\n")
        f.write("\n")

        f.write("## Most Confident Cases\n")
        f.write("| Filename | Category | Confidence Score |\n")
        f.write("|----------|----------|------------------|\n")
        for case in analysis['confident_cases']:
            f.write(f"| {case['filename']} | {case['best_category']} | {case['best_score']:.4f} |\n")


def get_image_paths(directory: Union[str, Path],
                    extensions: List[str] = ['jpg', 'jpeg', 'png']) -> List[str]:
    """
    Returns a list of all image paths in the given directory.

    Args:
        directory: Directory to search for images
        extensions: List of file extensions to include

    Returns:
        List of image file paths as strings
    """
    dir_path = Path(directory)
    image_paths = []

    for ext in extensions:
        # Handle extensions with or without the dot
        if not ext.startswith('.'):
            ext = f'.{ext}'

        # Add all matching files
        image_paths.extend([str(p) for p in dir_path.glob(f'*{ext}')])

    return image_paths