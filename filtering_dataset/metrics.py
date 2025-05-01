"""
metrics.py

Module for evaluating classification performance by comparing manual vs. automated categorization.
Provides functions for:
- Loading manual classification (ground truth)
- Computing classification metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Analyzing problematic cases (misclassifications)
- Special handling for low-quality images (shit_data category)
"""
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def load_manual_classification(root_dir: Union[str, Path]) -> Dict[str, Set[str]]:
    """
    Loads the manual classification data by scanning folders.

    Args:
        root_dir: Root directory containing category folders

    Returns:
        Dictionary mapping category names to sets of image filenames
    """
    manual_classes = {}
    root_path = Path(root_dir)

    # Scan all category folders
    for category_dir in root_path.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            # Get all image files (jpg, png) in this category folder
            image_files = set()
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.update(f.name for f in category_dir.glob(f'*{ext}'))

            manual_classes[category] = image_files
            logging.info(f"Found {len(image_files)} images in manual category '{category}'")

    return manual_classes


def create_ground_truth_df(
        manual_classes: Dict[str, Set[str]],
        all_images: List[str]
) -> pd.DataFrame:
    """
    Creates a DataFrame with ground truth labels for all images.

    Args:
        manual_classes: Dictionary mapping category names to sets of image filenames
        all_images: List of all image paths to evaluate

    Returns:
        DataFrame with image paths and ground truth categories
    """
    records = []

    # Create filename to category mapping for faster lookup
    filename_to_category = {}
    for category, filenames in manual_classes.items():
        for filename in filenames:
            filename_to_category[filename] = category

    # Create records for all images
    for img_path in all_images:
        filename = Path(img_path).name
        category = filename_to_category.get(filename, "unknown")

        records.append({
            "path": img_path,
            "filename": filename,
            "manual_category": category
        })

    return pd.DataFrame(records)


def calculate_metrics(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        categories: List[str],
        ignore_categories: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Calculates various classification metrics by comparing ground truth to predictions.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions
        categories: List of category names to consider
        ignore_categories: Optional list of categories to exclude from metrics

    Returns:
        Dictionary with metrics (overall and per-category)
    """
    # Merge dataframes on filename
    merged_df = pd.merge(
        ground_truth_df,
        prediction_df[['filename', 'best_category']],
        on='filename',
        how='inner'
    )

    # Filter out ignored categories if specified
    if ignore_categories:
        before_count = len(merged_df)
        merged_df = merged_df[~merged_df['manual_category'].isin(ignore_categories)]
        logging.info(f"Ignored {before_count - len(merged_df)} images from categories: {ignore_categories}")

    # Extract true and predicted labels
    y_true = merged_df['manual_category'].values
    y_pred = merged_df['best_category'].values

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=categories
    )

    # Calculate per-category metrics
    category_metrics = {}
    for i, category in enumerate(categories):
        if i < len(precision):  # Check if index is valid
            category_metrics[category] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i]) if i < len(support) else 0
            }

    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', labels=categories
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', labels=categories
    )

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_category': category_metrics,
        'total_images': len(merged_df),
        'categories': categories
    }


def plot_confusion_matrix_comparison(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        categories: List[str],
        output_path: Path,
        ignore_categories: Optional[List[str]] = None
) -> None:
    """
    Creates and saves a confusion matrix visualization comparing manual and predicted classes.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions
        categories: List of all categories to include
        output_path: Path to save the visualization
        ignore_categories: Optional list of categories to exclude
    """
    # Merge dataframes
    merged_df = pd.merge(
        ground_truth_df,
        prediction_df[['filename', 'best_category']],
        on='filename',
        how='inner'
    )

    # Filter out ignored categories if specified
    if ignore_categories:
        merged_df = merged_df[~merged_df['manual_category'].isin(ignore_categories)]

    # Extract true and predicted labels
    y_true = merged_df['manual_category'].values
    y_pred = merged_df['best_category'].values

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot normalized confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories
    )

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Manual Category (True)')
    plt.xlabel('CLIP Category (Predicted)')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)
    plt.close()

    # Also save the raw counts version
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=categories,
        yticklabels=categories
    )

    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('Manual Category (True)')
    plt.xlabel('CLIP Category (Predicted)')
    plt.tight_layout()

    # Save raw counts version
    raw_counts_path = output_path.with_name(output_path.stem + '_counts.png')
    plt.savefig(raw_counts_path)
    plt.close()


def analyze_misclassifications(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        output_dir: Path,
        top_n: int = 10
) -> Dict:
    """
    Analyzes misclassified images and identifies common patterns.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions including scores
        output_dir: Directory to save analysis results
        top_n: Number of worst misclassifications to analyze

    Returns:
        Dictionary with misclassification analysis
    """
    # Merge dataframes
    merged_df = pd.merge(
        ground_truth_df,
        prediction_df,
        on='filename',
        how='inner',
        suffixes=('', '_pred')
    )

    # Find misclassifications
    misclassified = merged_df[merged_df['manual_category'] != merged_df['best_category']]

    # Count misclassifications by category pair
    error_types = Counter()
    for _, row in misclassified.iterrows():
        true_cat = row['manual_category']
        pred_cat = row['best_category']
        error_types[(true_cat, pred_cat)] += 1

    # Get most common misclassification patterns
    common_errors = [
        {
            'true_category': true_cat,
            'predicted_category': pred_cat,
            'count': count
        }
        for (true_cat, pred_cat), count in error_types.most_common(top_n)
    ]

    # Find the worst misclassifications (highest confidence incorrect predictions)
    misclassified['confidence_error'] = misclassified['best_score']  # Higher confidence = worse error
    worst_errors = misclassified.sort_values('confidence_error', ascending=False).head(top_n)

    # Save the worst misclassifications to CSV
    worst_errors_path = output_dir / 'worst_misclassifications.csv'
    worst_errors[['filename', 'manual_category', 'best_category', 'best_score']].to_csv(
        worst_errors_path, index=False
    )

    # Create a visualization of worst misclassifications
    fig, axes = plt.subplots(min(5, len(worst_errors)), 2, figsize=(12, 3 * len(worst_errors)))

    for i, (_, row) in enumerate(worst_errors.iterrows()):
        if i >= 5:  # Limit to 5 examples
            break

        # Try to load and display the image
        try:
            img = Image.open(row['path'])
            if len(worst_errors) == 1:
                ax = axes[0]
            else:
                ax = axes[i, 0]

            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Filename: {row['filename']}")

            # Create text description in second column
            if len(worst_errors) == 1:
                ax = axes[1]
            else:
                ax = axes[i, 1]

            ax.axis('off')
            ax.text(0.1, 0.5,
                    f"True: {row['manual_category']}\n"
                    f"Pred: {row['best_category']}\n"
                    f"Confidence: {row['best_score']:.4f}",
                    fontsize=12, va='center')

        except Exception as e:
            logging.warning(f"Could not load image {row['path']}: {e}")

    plt.tight_layout()
    plt.savefig(output_dir / 'worst_misclassifications.png')
    plt.close()

    return {
        'total_misclassified': len(misclassified),
        'misclassification_rate': len(misclassified) / len(merged_df),
        'common_errors': common_errors,
        'worst_errors': worst_errors[['filename', 'manual_category', 'best_category', 'best_score']].to_dict('records')
    }


def analyze_low_quality_category(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        low_quality_category: str,
        output_dir: Path
) -> Dict:
    """
    Analyzes how the model handles low-quality images (e.g. 'shit_data').

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions
        low_quality_category: Name of the category with low-quality images
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with low-quality image analysis
    """
    # Merge dataframes
    merged_df = pd.merge(
        ground_truth_df,
        prediction_df,
        on='filename',
        how='inner',
        suffixes=('', '_pred')
    )

    # Find low-quality images
    low_quality = merged_df[merged_df['manual_category'] == low_quality_category]

    if len(low_quality) == 0:
        logging.warning(f"No images found in category: {low_quality_category}")
        return {'count': 0}

    # Analyze where the model placed these images
    predictions_distribution = low_quality['best_category'].value_counts().to_dict()

    # Calculate average confidence scores
    avg_confidence = low_quality.groupby('best_category')['best_score'].mean().to_dict()

    # Create visualization of distribution
    plt.figure(figsize=(10, 6))

    # Plot where low-quality images ended up
    sns.countplot(y='best_category', data=low_quality, order=low_quality['best_category'].value_counts().index)

    plt.title(f'Where {low_quality_category} Images Were Classified')
    plt.xlabel('Count')
    plt.ylabel('Predicted Category')
    plt.tight_layout()

    plt.savefig(output_dir / f'{low_quality_category}_distribution.png')
    plt.close()

    # Create visualization of confidence scores
    plt.figure(figsize=(10, 6))

    sns.boxplot(x='best_category', y='best_score', data=low_quality)

    plt.title(f'Confidence Scores for {low_quality_category} Images')
    plt.xlabel('Predicted Category')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / f'{low_quality_category}_confidence.png')
    plt.close()

    return {
        'count': len(low_quality),
        'distribution': predictions_distribution,
        'avg_confidence': avg_confidence,
        'percent_of_dataset': len(low_quality) / len(ground_truth_df) * 100
    }


def evaluate_classification(
        manual_dir: Union[str, Path],
        clip_results_df: pd.DataFrame,
        output_dir: Union[str, Path],
        low_quality_category: Optional[str] = 'shit_data',
        ignore_in_metrics: Optional[List[str]] = None
) -> Dict:
    """
    Performs a comprehensive evaluation of classification performance.

    Args:
        manual_dir: Directory containing manually classified images
        clip_results_df: DataFrame with CLIP classification results
        output_dir: Directory to save evaluation results
        low_quality_category: Name of the low-quality image category
        ignore_in_metrics: Categories to ignore when computing metrics

    Returns:
        Dictionary with complete evaluation results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manual classification
    logging.info(f"Loading manual classification from {manual_dir}")
    manual_classes = load_manual_classification(manual_dir)

    # Get all unique image paths from CLIP results
    all_images = clip_results_df['path'].unique().tolist()

    # Create ground truth DataFrame
    ground_truth_df = create_ground_truth_df(manual_classes, all_images)

    # Get all categories
    all_categories = list(manual_classes.keys())

    # Calculate metrics (optionally excluding low-quality category)
    metrics_categories = [cat for cat in all_categories if cat not in (ignore_in_metrics or [])]

    metrics = calculate_metrics(
        ground_truth_df,
        clip_results_df,
        metrics_categories,
        ignore_in_metrics
    )

    # Create confusion matrix visualization
    confusion_matrix_path = output_path / 'confusion_matrix.png'
    plot_confusion_matrix_comparison(
        ground_truth_df,
        clip_results_df,
        metrics_categories,
        confusion_matrix_path,
        ignore_in_metrics
    )

    # Analyze misclassifications
    misclassification_analysis = analyze_misclassifications(
        ground_truth_df,
        clip_results_df,
        output_path
    )

    # Analyze low-quality category if specified
    low_quality_analysis = None
    if low_quality_category and low_quality_category in manual_classes:
        low_quality_analysis = analyze_low_quality_category(
            ground_truth_df,
            clip_results_df,
            low_quality_category,
            output_path
        )

    # Save results to JSON and markdown
    results = {
        'metrics': metrics,
        'misclassifications': misclassification_analysis,
        'low_quality_analysis': low_quality_analysis,
        'total_images': len(ground_truth_df),
        'images_evaluated': len(ground_truth_df[ground_truth_df['manual_category'] != 'unknown'])
    }

    # Create summary markdown report
    create_evaluation_report(results, output_path / 'evaluation_report.md')

    return results


def create_evaluation_report(results: Dict, output_path: Path) -> None:
    """
    Creates a formatted markdown report from evaluation results.

    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the markdown report
    """
    metrics = results['metrics']

    with open(output_path, 'w') as f:
        f.write("# Image Classification Evaluation Report\n\n")

        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write(f"- **Total images evaluated**: {results['images_evaluated']}\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **Macro F1 Score**: {metrics['f1_macro']:.4f}\n")
        f.write(f"- **Weighted F1 Score**: {metrics['f1_weighted']:.4f}\n\n")

        # Per-category metrics
        f.write("## Per-Category Metrics\n\n")
        f.write("| Category | Precision | Recall | F1 Score | Support |\n")
        f.write("|----------|-----------|--------|----------|--------|\n")

        for category, cat_metrics in metrics['per_category'].items():
            f.write(
                f"| {category} | {cat_metrics['precision']:.4f} | {cat_metrics['recall']:.4f} | {cat_metrics['f1']:.4f} | {cat_metrics['support']} |\n")

        f.write("\n")

        # Misclassification analysis
        if results['misclassifications']:
            misclass = results['misclassifications']
            f.write("## Misclassification Analysis\n\n")
            f.write(f"- **Total misclassified images**: {misclass['total_misclassified']}\n")
            f.write(
                f"- **Misclassification rate**: {misclass['misclassification_rate']:.4f} ({misclass['misclassification_rate'] * 100:.2f}%)\n\n")

            f.write("### Common Error Patterns\n\n")
            f.write("| True Category | Predicted Category | Count |\n")
            f.write("|--------------|-------------------|-------|\n")

            for error in misclass['common_errors']:
                f.write(f"| {error['true_category']} | {error['predicted_category']} | {error['count']} |\n")

            f.write("\n")

        # Low-quality image analysis
        if results['low_quality_analysis']:
            low_qual = results['low_quality_analysis']
            f.write("## Low-Quality Image Analysis\n\n")
            f.write(f"- **Total low-quality images**: {low_qual['count']}\n")
            f.write(f"- **Percentage of dataset**: {low_qual['percent_of_dataset']:.2f}%\n\n")

            f.write("### Distribution of Low-Quality Images\n\n")
            f.write("| Predicted Category | Count | Avg. Confidence |\n")
            f.write("|-------------------|-------|----------------|\n")

            for category, count in low_qual['distribution'].items():
                avg_conf = low_qual['avg_confidence'].get(category, 0)
                f.write(f"| {category} | {count} | {avg_conf:.4f} |\n")

            f.write("\n")

        f.write("## Conclusion and Recommendations\n\n")
        f.write("Based on the evaluation results, consider the following recommendations:\n\n")

        # Generate recommendations based on results
        accuracy = metrics['accuracy']
        if accuracy < 0.7:
            f.write(
                "- The overall accuracy is quite low. Consider refining the CLIP prompts or adjusting thresholds.\n")
        elif accuracy < 0.85:
            f.write("- The accuracy is moderate. Focus on improving the most common misclassification patterns.\n")
        else:
            f.write("- The accuracy is good. Focus on specific edge cases if further improvement is needed.\n")

        # Add recommendation for low-quality images if they exist
        if results['low_quality_analysis'] and results['low_quality_analysis']['count'] > 0:
            lq_percent = results['low_quality_analysis']['percent_of_dataset']
            if lq_percent > 10:
                f.write(
                    f"- Low-quality images make up {lq_percent:.2f}% of your dataset. Consider preprocessing to filter these out automatically.\n")
            else:
                f.write(
                    f"- Low-quality images make up only {lq_percent:.2f}% of your dataset. Their impact on overall metrics is minimal.\n")

        # Add visualization references
        f.write("\n## Visualizations\n\n")
        f.write("The following visualizations were generated:\n\n")
        f.write("- `confusion_matrix.png` - Shows how manual categories map to predicted categories\n")
        f.write("- `confusion_matrix_counts.png` - Raw counts version of the confusion matrix\n")
        f.write("- `worst_misclassifications.png` - Examples of the most confident incorrect predictions\n")

        if results['low_quality_analysis'] and results['low_quality_analysis']['count'] > 0:
            f.write("- `shit_data_distribution.png` - Shows where low-quality images were classified\n")
            f.write("- `shit_data_confidence.png` - Shows confidence scores for low-quality images\n")