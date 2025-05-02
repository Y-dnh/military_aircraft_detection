"""
classification_evaluation.py

Comprehensive module for evaluating and visualizing image classification performance.
Integrates metrics calculation and visualization functionality for complete analysis.

Key features:
- Loading and comparing manual vs. automated classification
- Computing comprehensive classification metrics (accuracy, precision, recall, F1, etc.)
- Advanced confusion matrix visualization and analysis
- Distribution analysis of classification results and scores
- Borderline case detection and analysis
- Low-quality image handling
- Visualization of results across multiple models/thresholds
- Sample image grid generation for visual inspection
"""
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----- Setup Functions -----

def setup_plotting():
    """
    Configure matplotlib for high-quality plots.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


# ----- Data Loading Functions -----

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


# ----- Core Metrics Calculation Functions -----

def calculate_metrics(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        categories: List[str],
        ignore_categories: Optional[List[str]] = None,
        model_name: str = "model"
) -> Dict[str, Union[float, Dict]]:
    """
    Calculates various classification metrics by comparing ground truth to predictions.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions
        categories: List of category names to consider
        ignore_categories: Optional list of categories to exclude from metrics
        model_name: Name of the model being evaluated (for multi-model comparison)

    Returns:
        Dictionary with metrics (overall and per-category)
    """
    # Merge dataframes on filename
    merged_df = pd.merge(
        ground_truth_df,
        prediction_df[['filename', 'best_category', 'best_score']],
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

    # Calculate Matthews Correlation Coefficient (MCC) for multi-class
    # This requires scikit-learn's matthews_corrcoef
    try:
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(y_true, y_pred)
    except ImportError:
        mcc = None
        logging.warning("scikit-learn's matthews_corrcoef not available, MCC not calculated")

    # Calculate Cohen's Kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa = cohen_kappa_score(y_true, y_pred)
    except ImportError:
        kappa = None
        logging.warning("scikit-learn's cohen_kappa_score not available, Kappa not calculated")

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'mcc': mcc,
        'kappa': kappa,
        'per_category': category_metrics,
        'total_images': len(merged_df),
        'categories': categories
    }


# ----- Visualization Functions -----

def plot_confusion_matrix_comparison(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        categories: List[str],
        output_path: Path,
        ignore_categories: Optional[List[str]] = None,
        cmap: str = 'Blues',
        normalize: bool = True,
        title: str = 'Confusion Matrix'
) -> None:
    """
    Creates and saves a confusion matrix visualization comparing manual and predicted classes.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions
        categories: List of all categories to include
        output_path: Path to save the visualization
        ignore_categories: Optional list of categories to exclude
        cmap: Colormap for the heatmap
        normalize: Whether to normalize the confusion matrix
        title: Plot title
    """
    # Set up the plotting style
    setup_plotting()

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

    # Normalize if requested
    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        vmax = 1.0
        title = f'Normalized {title}'
    else:
        cm_plot = cm
        fmt = 'd'
        vmax = cm.max()
        title = f'{title} (Counts)'

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot confusion matrix
    heatmap = sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=categories,
        yticklabels=categories,
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Normalized frequency' if normalize else 'Count'}
    )

    plt.title(title)
    plt.ylabel('Manual Category (True)')
    plt.xlabel('Predicted Category')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path)
    plt.close()

    # If normalized version created, also save the raw counts version
    if normalize:
        raw_counts_path = output_path.with_name(output_path.stem + '_counts.png')
        plot_confusion_matrix_comparison(
            ground_truth_df, prediction_df, categories, raw_counts_path,
            ignore_categories, cmap, normalize=False
        )

    # Add diagonal values to get overall accuracy
    cm_diag = np.diag(cm)
    cm_total = np.sum(cm)
    cm_accuracy = np.sum(cm_diag) / cm_total if cm_total > 0 else 0

    logging.info(f"Confusion matrix saved to {output_path}")
    logging.info(f"Overall accuracy from confusion matrix: {cm_accuracy:.4f}")


def plot_category_distribution(
        df: pd.DataFrame,
        output_path: Path,
        title: str = "Distribution of Images by Category"
) -> None:
    """
    Create a bar chart showing the distribution of images across categories.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        title: Plot title
    """
    # Set up the plotting style
    setup_plotting()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Count images in each category
    category_counts = df['best_category'].value_counts().sort_values(ascending=False)

    # Plot horizontal bars with custom colors
    colors = sns.color_palette('viridis', len(category_counts))
    bars = ax.barh(y=category_counts.index, width=category_counts.values, color=colors)

    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (width * 0.01),
                bar.get_y() + bar.get_height() / 2,
                f'{int(width)}',
                va='center')

    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Category')

    # Add percentage to y-axis labels
    total_images = len(df)
    y_labels = [f"{cat} ({count / total_images:.1%})" for cat, count in
                zip(category_counts.index, category_counts.values)]
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Add grid lines for readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_score_distributions(
        df: pd.DataFrame,
        output_path: Path,
        category_columns: Optional[List[str]] = None
) -> None:
    """
    Create density plots showing the distribution of confidence scores for each category.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        category_columns: Optional list of score column names to plot
    """
    # Set up the plotting style
    setup_plotting()

    # If category columns not specified, find all columns starting with 'score_'
    if category_columns is None:
        category_columns = [col for col in df.columns if col.startswith('score_')]

    if not category_columns:
        logging.warning("No score columns found for visualization")
        return

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the density for each category's scores
    for col in category_columns:
        # Extract the category name from the column name
        category = col.replace('score_', '')

        # Plot the distribution
        sns.kdeplot(df[col], label=category, ax=ax, fill=True, alpha=0.3)

    # Add vertical lines at each category's threshold
    for col in category_columns:
        category = col.replace('score_', '')
        # Find images actually classified as this category
        category_images = df[df['best_category'] == category]
        if not category_images.empty:
            # Get the minimum score that was still classified as this category
            min_score = category_images[col].min()
            ax.axvline(x=min_score, linestyle='--', alpha=0.7,
                       color='gray', label=f"{category} threshold")

    # Add titles and labels
    ax.set_title("Distribution of Confidence Scores by Category")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")

    # Add legend
    ax.legend(title="Categories", loc='upper left')

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_category_confusion_matrix(
        df: pd.DataFrame,
        output_path: Path,
        category_columns: Optional[List[str]] = None
) -> None:
    """
    Create a heatmap showing the average score of each category's images
    for every other category's prompts.

    This helps identify categories that might be getting confused.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        category_columns: Optional list of score column names to use
    """
    # Set up the plotting style
    setup_plotting()

    # If category columns not specified, find all columns starting with 'score_'
    if category_columns is None:
        category_columns = [col for col in df.columns if col.startswith('score_')]

    if not category_columns:
        logging.warning("No score columns found for visualization")
        return

    # Extract category names
    categories = [col.replace('score_', '') for col in category_columns]

    # Create a confusion matrix of average scores
    confusion_matrix = np.zeros((len(categories), len(categories)))

    # Fill the confusion matrix
    for i, true_cat in enumerate(categories):
        # Get images classified as this category
        cat_images = df[df['best_category'] == true_cat]
        if cat_images.empty:
            continue

        # Calculate average scores for each category prompt
        for j, score_col in enumerate(category_columns):
            confusion_matrix[i, j] = cat_images[score_col].mean()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=categories, yticklabels=categories, ax=ax)

    # Add titles and labels
    ax.set_title("Category Confusion Matrix (Average Scores)")
    ax.set_xlabel("Prompt Category")
    ax.set_ylabel("Assigned Category")

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_borderline_cases(
        df: pd.DataFrame,
        output_path: Path,
        threshold_margin: float = 0.05
) -> pd.DataFrame:
    """
    Plot the distribution of borderline cases - images whose scores
    are close to the decision boundary between categories.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        threshold_margin: Margin around the threshold to consider borderline

    Returns:
        DataFrame containing only the borderline cases
    """
    # Set up the plotting style
    setup_plotting()

    # Find score columns
    score_columns = [col for col in df.columns if col.startswith('score_')]
    categories = [col.replace('score_', '') for col in score_columns]

    # For each image, find the difference between its top two category scores
    df_with_margins = df.copy()

    # For each row, find the top two scores
    for i, row in df.iterrows():
        scores = [row[f'score_{cat}'] for cat in categories if f'score_{cat}' in row]
        scores.sort(reverse=True)

        if len(scores) >= 2:
            # Calculate the margin between top two scores
            df_with_margins.loc[i, 'score_margin'] = scores[0] - scores[1]
        else:
            df_with_margins.loc[i, 'score_margin'] = 1.0  # Maximum margin if only one category

    # Consider images with small margins as borderline
    borderline = df_with_margins[df_with_margins['score_margin'] <= threshold_margin]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram of margins
    sns.histplot(df_with_margins['score_margin'], bins=30, kde=True, ax=ax)

    # Add a vertical line at the threshold margin
    ax.axvline(x=threshold_margin, color='red', linestyle='--',
               label=f'Threshold ({threshold_margin})')

    # Add shading for the borderline region
    ax.axvspan(0, threshold_margin, alpha=0.2, color='red')

    # Count borderline cases
    borderline_count = len(borderline)
    borderline_percent = borderline_count / len(df) * 100

    # Add annotation for count of borderline cases
    ax.text(0.98, 0.95,
            f"Borderline Cases: {borderline_count}\n({borderline_percent:.1f}% of dataset)",
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add titles and labels
    ax.set_title("Distribution of Decision Margins Between Top Categories")
    ax.set_xlabel("Score Margin (difference between top two category scores)")
    ax.set_ylabel("Number of Images")

    ax.legend()

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return borderline


def create_sample_grid(
        df: pd.DataFrame,
        output_path: Path,
        samples_per_category: int = 5,
        random_seed: int = 42
) -> None:
    """
    Create a grid of sample images from each category for visual inspection.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        samples_per_category: Number of sample images to display per category
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get list of categories
    categories = df['best_category'].unique()
    num_categories = len(categories)

    # Determine grid dimensions
    grid_cols = min(samples_per_category, 5)  # Maximum 5 columns
    grid_rows = num_categories

    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 3 * grid_rows))

    # If there's only one category, axes won't be a 2D array
    if num_categories == 1:
        axes = np.array([axes])

    # If only one column, ensure axes is 2D for consistent indexing
    if grid_cols == 1:
        axes = axes.reshape(-1, 1)

    # Iterate through categories
    for i, category in enumerate(categories):
        # Get images from this category
        cat_images = df[df['best_category'] == category]

        # Sample images from this category
        sample_count = min(len(cat_images), samples_per_category)
        if sample_count > 0:
            samples = cat_images.sample(sample_count)

            # Plot each sample
            for j, (_, row) in enumerate(samples.iterrows()):
                if j >= grid_cols:
                    break

                # Load and display image
                try:
                    img = Image.open(row['path'])

                    # Turn off axes for images
                    ax = axes[i, j]
                    ax.imshow(img)
                    ax.axis('off')

                    # Add score as text
                    score = row['best_score']
                    ax.text(10, 10, f"Score: {score:.2f}",
                            bbox=dict(facecolor='white', alpha=0.7),
                            fontsize=8)

                    # First image in each row gets a category label
                    if j == 0:
                        ax.set_title(f"Category: {category}")

                except Exception as e:
                    logging.warning(f"Error loading image {row['path']}: {e}")
                    ax = axes[i, j]
                    ax.text(0.5, 0.5, "Image load error",
                            ha='center', va='center', fontsize=10)
                    ax.axis('off')

            # If we have fewer samples than columns, turn off remaining axes
            for j in range(sample_count, grid_cols):
                axes[i, j].axis('off')
        else:
            # No samples for this category, show a message
            ax = axes[i, 0]
            ax.text(0.5, 0.5, f"No samples for {category}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

            # Turn off remaining axes in this row
            for j in range(1, grid_cols):
                axes[i, j].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        output_path: Path,
        categories: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Create ROC curves for each category (one-vs-rest approach).

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions and scores
        output_path: Path to save the visualization
        categories: Optional list of categories to include

    Returns:
        Dictionary mapping category names to AUC scores
    """
    # Set up plotting
    setup_plotting()

    # Merge dataframes
    merged_df = pd.merge(
        ground_truth_df[['filename', 'manual_category']],
        prediction_df,
        on='filename',
        how='inner'
    )

    # If categories not specified, get all unique categories
    if categories is None:
        categories = merged_df['manual_category'].unique().tolist()

    # Create figure
    plt.figure(figsize=(10, 8))

    # Calculate and plot ROC curve for each category
    auc_scores = {}
    for category in categories:
        # Create binary labels (1 for this category, 0 for others)
        y_true = (merged_df['manual_category'] == category).astype(int)

        # Get scores for this category
        score_col = f'score_{category}'
        if score_col in merged_df.columns:
            y_score = merged_df[score_col]

            # Calculate ROC
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            auc_scores[category] = roc_auc

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{category} (AUC = {roc_auc:.2f})')
        else:
            logging.warning(f"Score column '{score_col}' not found for ROC curve")

    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')

    # Add legend and grid
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # Adjust limits and save
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return auc_scores


def plot_precision_recall_curve(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        output_path: Path,
        categories: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Create Precision-Recall curves for each category.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions and scores
        output_path: Path to save the visualization
        categories: Optional list of categories to include

    Returns:
        Dictionary mapping category names to average precision scores
    """
    # Set up plotting
    setup_plotting()

    # Merge dataframes
    merged_df = pd.merge(
        ground_truth_df[['filename', 'manual_category']],
        prediction_df,
        on='filename',
        how='inner'
    )

    # If categories not specified, get all unique categories
    if categories is None:
        categories = merged_df['manual_category'].unique().tolist()

    # Create figure
    plt.figure(figsize=(10, 8))

    # Calculate and plot PR curve for each category
    ap_scores = {}
    for category in categories:
        # Create binary labels (1 for this category, 0 for others)
        y_true = (merged_df['manual_category'] == category).astype(int)

        # Get scores for this category
        score_col = f'score_{category}'
        if score_col in merged_df.columns:
            y_score = merged_df[score_col]

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            average_precision = average_precision_score(y_true, y_score)
            ap_scores[category] = average_precision

            # Plot PR curve
            plt.plot(recall, precision, lw=2,
                     label=f'{category} (AP = {average_precision:.2f})')
        else:
            logging.warning(f"Score column '{score_col}' not found for PR curve")

    # Set labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')

    # Add legend and grid
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    # Adjust limits and save
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return ap_scores


# ----- Advanced Analysis Functions -----

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

    if len(misclassified) == 0:
        logging.info("No misclassifications found!")
        return {
            'total_misclassified': 0,
            'misclassification_rate': 0,
            'common_errors': [],
            'worst_errors': []
        }

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
        for (true_cat, pred_cat),count in error_types.most_common(10)
    ]

    # Find worst errors - those with highest confidence but wrong prediction
    misclassified['confidence_error'] = misclassified['best_score']
    worst_errors = misclassified.sort_values('confidence_error', ascending=False).head(top_n)

    worst_list = []
    for _, row in worst_errors.iterrows():
        worst_list.append({
            'filename': row['filename'],
            'path': row['path'],
            'true_category': row['manual_category'],
            'predicted_category': row['best_category'],
            'confidence': row['best_score']
        })

    # Save misclassifications to CSV
    misclassified_path = output_dir / 'misclassifications.csv'
    misclassified.to_csv(misclassified_path, index=False)

    # Save worst misclassifications to separate CSV
    worst_errors_path = output_dir / 'worst_misclassifications.csv'
    worst_errors.to_csv(worst_errors_path, index=False)

    # Create visualization of common error types
    if error_types:
        error_viz_path = output_dir / 'common_errors.png'
        setup_plotting()
        plt.figure(figsize=(12, 8))

        # Extract data for plotting
        error_labels = [f"{true} → {pred}" for (true, pred), _ in error_types.most_common(10)]
        error_counts = [count for _, count in error_types.most_common(10)]

        # Create bar chart
        bars = plt.barh(error_labels, error_counts, color='salmon')

        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center')

        plt.title('Most Common Misclassification Patterns')
        plt.xlabel('Number of Images')
        plt.ylabel('Misclassification Type (True → Predicted)')
        plt.tight_layout()
        plt.savefig(error_viz_path)
        plt.close()

    return {
        'total_misclassified': len(misclassified),
        'misclassification_rate': len(misclassified) / len(merged_df),
        'common_errors': common_errors,
        'worst_errors': worst_list
    }


def analyze_low_quality_category(
        ground_truth_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        low_quality_category: str,
        output_dir: Path
) -> Dict:
    """
    Analyzes how the model handles low-quality images.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_df: DataFrame with model predictions including scores
        low_quality_category: Name of the category for low-quality images
        output_dir: Directory to save analysis

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
    true_low_quality = merged_df[merged_df['manual_category'] == low_quality_category]
    predicted_low_quality = merged_df[merged_df['best_category'] == low_quality_category]

    # Calculate metrics
    true_positive = len(merged_df[(merged_df['manual_category'] == low_quality_category) &
                                 (merged_df['best_category'] == low_quality_category)])

    false_positive = len(merged_df[(merged_df['manual_category'] != low_quality_category) &
                                  (merged_df['best_category'] == low_quality_category)])

    false_negative = len(merged_df[(merged_df['manual_category'] == low_quality_category) &
                                  (merged_df['best_category'] != low_quality_category)])

    # Calculate precision and recall
    precision = true_positive / len(predicted_low_quality) if len(predicted_low_quality) > 0 else 0
    recall = true_positive / len(true_low_quality) if len(true_low_quality) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Analyze where low-quality images were misclassified
    if false_negative > 0:
        false_neg_df = merged_df[(merged_df['manual_category'] == low_quality_category) &
                                 (merged_df['best_category'] != low_quality_category)]

        misclassified_categories = Counter(false_neg_df['best_category'])
        misclassified_distribution = [
            {'category': cat, 'count': count}
            for cat, count in misclassified_categories.most_common()
        ]
    else:
        misclassified_distribution = []

    # Analyze score distributions
    scores_path = output_dir / f'{low_quality_category}_score_distribution.png'
    setup_plotting()

    try:
        plt.figure(figsize=(10, 6))

        # Get the column name for low quality scores
        low_qual_score_col = f'score_{low_quality_category}'

        # Plot score distributions
        if low_qual_score_col in merged_df.columns:
            # True low quality
            if len(true_low_quality) > 0:
                sns.kdeplot(
                    true_low_quality[low_qual_score_col],
                    label=f'True {low_quality_category}',
                    fill=True, alpha=0.3
                )

            # Non-low quality
            non_low_quality = merged_df[merged_df['manual_category'] != low_quality_category]
            if len(non_low_quality) > 0:
                sns.kdeplot(
                    non_low_quality[low_qual_score_col],
                    label=f'Not {low_quality_category}',
                    fill=True, alpha=0.3
                )

            plt.axvline(
                x=merged_df[merged_df['best_category'] == low_quality_category][low_qual_score_col].min(),
                color='red', linestyle='--', label='Classification Threshold'
            )

            plt.title(f'Score Distribution for {low_quality_category} Category')
            plt.xlabel(f'Score for {low_quality_category}')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(scores_path)
        else:
            logging.warning(f"Score column for {low_quality_category} not found")
    except Exception as e:
        logging.error(f"Error creating low quality score distribution: {e}")
    finally:
        plt.close()

    return {
        'true_low_quality_count': len(true_low_quality),
        'predicted_low_quality_count': len(predicted_low_quality),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'misclassified_distribution': misclassified_distribution
    }


def compare_models(
        ground_truth_df: pd.DataFrame,
        prediction_dfs: Dict[str, pd.DataFrame],
        categories: List[str],
        output_dir: Path,
        ignore_categories: Optional[List[str]] = None
) -> Dict:
    """
    Compares multiple classification models against the same ground truth.

    Args:
        ground_truth_df: DataFrame with manual classifications
        prediction_dfs: Dictionary mapping model names to prediction DataFrames
        categories: List of category names to consider
        output_dir: Directory to save comparison results
        ignore_categories: Optional list of categories to exclude

    Returns:
        Dictionary with model comparison results
    """
    # Calculate metrics for each model
    model_metrics = {}
    for model_name, pred_df in prediction_dfs.items():
        model_metrics[model_name] = calculate_metrics(
            ground_truth_df, pred_df, categories,
            ignore_categories, model_name=model_name
        )

    # Create comparison visualizations
    setup_plotting()

    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(model_metrics.keys())
    accuracies = [metrics['accuracy'] for metrics in model_metrics.values()]

    bars = plt.bar(model_names, accuracies, color='skyblue')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

    plt.title('Accuracy Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_accuracy_comparison.png')
    plt.close()

    # F1 Score comparison (macro)
    plt.figure(figsize=(10, 6))
    f1_scores = [metrics['f1_macro'] for metrics in model_metrics.values()]

    bars = plt.bar(model_names, f1_scores, color='lightgreen')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')

    plt.title('F1 Score (Macro) Comparison Across Models')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, max(f1_scores) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_f1_comparison.png')
    plt.close()

    # Per-category comparison for selected metrics
    per_category_metrics = {}
    for category in categories:
        category_metrics = {}
        for model_name, metrics in model_metrics.items():
            if category in metrics['per_category']:
                category_metrics[model_name] = metrics['per_category'][category]
        per_category_metrics[category] = category_metrics

    # Save comparison data as CSV
    comparison_data = []
    for model_name, metrics in model_metrics.items():
        row = {
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_weighted': metrics['f1_weighted']
        }
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)

    return {
        'model_metrics': model_metrics,
        'per_category_metrics': per_category_metrics
    }


# ----- Integration Functions -----

def create_evaluation_report(results: Dict, output_path: Path) -> None:
    """
    Creates a formatted markdown report from evaluation results.

    Args:
        results: Dictionary with evaluation results
        output_path: Path to save markdown report
    """
    report = []

    # Add title
    report.append("# Image Classification Evaluation Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Add overall metrics
    report.append("## Overall Metrics\n")
    metrics = results.get('metrics', {})

    if metrics:
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Accuracy | {metrics.get('accuracy', 0):.4f} |")
        report.append(f"| Precision (macro) | {metrics.get('precision_macro', 0):.4f} |")
        report.append(f"| Recall (macro) | {metrics.get('recall_macro', 0):.4f} |")
        report.append(f"| F1 Score (macro) | {metrics.get('f1_macro', 0):.4f} |")
        report.append(f"| Precision (weighted) | {metrics.get('precision_weighted', 0):.4f} |")
        report.append(f"| Recall (weighted) | {metrics.get('recall_weighted', 0):.4f} |")
        report.append(f"| F1 Score (weighted) | {metrics.get('f1_weighted', 0):.4f} |")

        if 'mcc' in metrics and metrics['mcc'] is not None:
            report.append(f"| Matthews Correlation Coefficient | {metrics['mcc']:.4f} |")

        if 'kappa' in metrics and metrics['kappa'] is not None:
            report.append(f"| Cohen's Kappa | {metrics['kappa']:.4f} |")

        report.append(f"| Total Images | {metrics.get('total_images', 0)} |")
        report.append("")

    # Add per-category metrics
    report.append("## Per-Category Metrics\n")
    per_category = metrics.get('per_category', {})

    if per_category:
        report.append("| Category | Precision | Recall | F1 Score | Support |")
        report.append("|----------|-----------|--------|----------|---------|")

        for category, cat_metrics in per_category.items():
            report.append(f"| {category} | " +
                         f"{cat_metrics.get('precision', 0):.4f} | " +
                         f"{cat_metrics.get('recall', 0):.4f} | " +
                         f"{cat_metrics.get('f1', 0):.4f} | " +
                         f"{cat_metrics.get('support', 0)} |")
        report.append("")

    # Add misclassification analysis
    misclass = results.get('misclassifications', {})

    if misclass:
        report.append("## Misclassification Analysis\n")

        total = misclass.get('total_misclassified', 0)
        rate = misclass.get('misclassification_rate', 0)
        report.append(f"Total misclassified images: {total} ({rate:.2%} of all images)\n")

        # Common errors
        common_errors = misclass.get('common_errors', [])
        if common_errors:
            report.append("### Most Common Misclassification Patterns\n")
            report.append("| True Category | Predicted Category | Count |")
            report.append("|---------------|-------------------|-------|")

            for error in common_errors:
                report.append(f"| {error.get('true_category', '')} | " +
                             f"{error.get('predicted_category', '')} | " +
                             f"{error.get('count', 0)} |")
            report.append("")

    # Add low-quality image analysis
    low_quality = results.get('low_quality_analysis', {})

    if low_quality:
        report.append("## Low-Quality Image Analysis\n")

        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| True Low-Quality Count | {low_quality.get('true_low_quality_count', 0)} |")
        report.append(f"| Predicted Low-Quality Count | {low_quality.get('predicted_low_quality_count', 0)} |")
        report.append(f"| True Positives | {low_quality.get('true_positive', 0)} |")
        report.append(f"| False Positives | {low_quality.get('false_positive', 0)} |")
        report.append(f"| False Negatives | {low_quality.get('false_negative', 0)} |")
        report.append(f"| Precision | {low_quality.get('precision', 0):.4f} |")
        report.append(f"| Recall | {low_quality.get('recall', 0):.4f} |")
        report.append(f"| F1 Score | {low_quality.get('f1', 0):.4f} |")
        report.append("")

        # Misclassified distribution
        misclass_dist = low_quality.get('misclassified_distribution', [])
        if misclass_dist:
            report.append("### Where Low-Quality Images Were Misclassified\n")
            report.append("| Category | Count |")
            report.append("|----------|-------|")

            for item in misclass_dist:
                report.append(f"| {item.get('category', '')} | {item.get('count', 0)} |")
            report.append("")

    # Add recommendations
    report.append("## Recommendations\n")

    # Generate some basic recommendations based on metrics
    if metrics:
        # Find worst-performing categories
        worst_cats = []
        for cat, cat_metrics in per_category.items():
            if cat_metrics.get('f1', 0) < 0.7 and cat_metrics.get('support', 0) > 10:
                worst_cats.append((cat, cat_metrics.get('f1', 0)))

        worst_cats.sort(key=lambda x: x[1])

        if worst_cats:
            report.append("### Improving Category Classification\n")
            report.append("The following categories show lower performance and may need attention:\n")

            for cat, f1 in worst_cats[:3]:  # Top 3 worst
                report.append(f"- **{cat}** (F1: {f1:.4f}): Consider adding more training examples or refining prompts\n")

        # Check for low recall in important categories
        low_recall_cats = []
        for cat, cat_metrics in per_category.items():
            if cat_metrics.get('recall', 0) < 0.7 and cat_metrics.get('precision', 0) > 0.8:
                low_recall_cats.append(cat)

        if low_recall_cats:
            report.append("### Categories with Low Recall\n")
            report.append("These categories have good precision but low recall, suggesting the model is missing valid examples:\n")
            for cat in low_recall_cats:
                report.append(f"- **{cat}**: Review false negatives and consider lowering confidence thresholds\n")

    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))


def evaluate_classification(
        manual_dir: Union[str, Path],
        clip_results_df: pd.DataFrame,
        output_dir: Union[str, Path],
        low_quality_category: Optional[str] = 'shit_data',
        ignore_in_metrics: Optional[List[str]] = None
) -> Dict:
    """
    Performs comprehensive evaluation of classification performance.

    Args:
        manual_dir: Directory containing manually classified images
        clip_results_df: DataFrame with CLIP classification results
        output_dir: Directory to save evaluation results
        low_quality_category: Name of low-quality category
        ignore_in_metrics: Categories to ignore in metrics calculation

    Returns:
        Dictionary with evaluation results
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load manual classifications
    manual_classes = load_manual_classification(manual_dir)
    categories = list(manual_classes.keys())

    # Get list of all image paths from CLIP results
    all_images = clip_results_df['path'].tolist()

    # Create ground truth DataFrame
    ground_truth_df = create_ground_truth_df(manual_classes, all_images)

    # Create subdirectories for different analyses
    metrics_dir = output_dir / 'metrics'
    viz_dir = output_dir / 'visualizations'
    analysis_dir = output_dir / 'analysis'

    for directory in [metrics_dir, viz_dir, analysis_dir]:
        directory.mkdir(exist_ok=True)

    # Calculate metrics
    metrics = calculate_metrics(
        ground_truth_df, clip_results_df, categories, ignore_in_metrics
    )

    # Save metrics as JSON
    import json
    with open(metrics_dir / 'metrics.json', 'w') as f:
        # Convert numpy values to Python native types
        serializable_metrics = json.loads(
            json.dumps(metrics, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
        )
        json.dump(serializable_metrics, f, indent=2)

    # Create confusion matrix
    plot_confusion_matrix_comparison(
        ground_truth_df, clip_results_df, categories,
        metrics_dir / 'confusion_matrix.png',
        ignore_categories=ignore_in_metrics
    )

    # Create ROC curves
    roc_auc_scores = plot_roc_curve(
        ground_truth_df, clip_results_df,
        metrics_dir / 'roc_curves.png',
        categories=categories
    )

    # Create precision-recall curves
    pr_ap_scores = plot_precision_recall_curve(
        ground_truth_df, clip_results_df,
        metrics_dir / 'pr_curves.png',
        categories=categories
    )

    # Create visualizations
    plot_category_distribution(
        clip_results_df, viz_dir / 'category_distribution.png'
    )

    plot_score_distributions(
        clip_results_df, viz_dir / 'score_distributions.png'
    )

    plot_category_confusion_matrix(
        clip_results_df, viz_dir / 'category_confusion.png'
    )

    borderline_cases = plot_borderline_cases(
        clip_results_df, viz_dir / 'borderline_cases.png'
    )

    # Save borderline cases to CSV
    borderline_cases.to_csv(analysis_dir / 'borderline_cases.csv', index=False)

    # Create sample image grid
    try:
        create_sample_grid(
            clip_results_df, viz_dir / 'sample_grid.png'
        )
    except Exception as e:
        logging.error(f"Failed to create sample grid: {e}")

    # Analyze misclassifications
    misclassifications = analyze_misclassifications(
        ground_truth_df, clip_results_df, analysis_dir
    )

    # Analyze low-quality category if specified
    low_quality_analysis = None
    if low_quality_category and low_quality_category in categories:
        low_quality_analysis = analyze_low_quality_category(
            ground_truth_df, clip_results_df, low_quality_category, analysis_dir
        )

    # Create comprehensive evaluation report
    results = {
        'metrics': metrics,
        'roc_auc_scores': roc_auc_scores,
        'pr_ap_scores': pr_ap_scores,
        'misclassifications': misclassifications,
        'low_quality_analysis': low_quality_analysis
    }

    create_evaluation_report(results, output_dir / 'evaluation_report.md')

    return results


def create_all_visualizations(
        results_df: pd.DataFrame,
        output_dir: Path
) -> Dict:
    """
    Creates and saves all visualizations to the specified directory.

    Args:
        results_df: DataFrame with classification results
        output_dir: Directory to save all visualizations

    Returns:
        Dictionary with paths to all created visualizations
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize paths dictionary
    viz_paths = {}

    # Category distribution
    category_dist_path = output_dir / 'category_distribution.png'
    plot_category_distribution(results_df, category_dist_path)
    viz_paths['category_distribution'] = category_dist_path

    # Score distributions
    score_dist_path = output_dir / 'score_distributions.png'
    plot_score_distributions(results_df, score_dist_path)
    viz_paths['score_distributions'] = score_dist_path

    # Category confusion matrix
    category_conf_path = output_dir / 'category_confusion.png'
    plot_category_confusion_matrix(results_df, category_conf_path)
    viz_paths['category_confusion'] = category_conf_path

    # Borderline cases
    borderline_path = output_dir / 'borderline_cases.png'
    borderline_cases = plot_borderline_cases(results_df, borderline_path)
    viz_paths['borderline_cases'] = borderline_path

    # Save borderline cases to CSV
    borderline_csv_path = output_dir / 'borderline_cases.csv'
    borderline_cases.to_csv(borderline_csv_path, index=False)
    viz_paths['borderline_cases_csv'] = borderline_csv_path

    # Sample grid
    try:
        sample_grid_path = output_dir / 'sample_grid.png'
        create_sample_grid(results_df, sample_grid_path)
        viz_paths['sample_grid'] = sample_grid_path
    except Exception as e:
        logging.error(f"Failed to create sample grid: {e}")

    # Create confidence score boxplots by category
    try:
        boxplot_path = output_dir / 'confidence_boxplot.png'
        setup_plotting()
        plt.figure(figsize=(12, 8))

        sns.boxplot(x='best_category', y='best_score', data=results_df)
        plt.title('Confidence Score Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(boxplot_path)
        plt.close()

        viz_paths['confidence_boxplot'] = boxplot_path
    except Exception as e:
        logging.error(f"Failed to create confidence boxplot: {e}")

    return viz_paths


# ----- Main Function -----

def run_evaluation(
        manual_dir: Union[str, Path],
        results_df: pd.DataFrame,
        output_dir: Union[str, Path],
        low_quality_category: Optional[str] = 'shit_data',
) -> Dict:
    """
    Main function to run the complete evaluation pipeline.

    Args:
        manual_dir: Directory with manually classified images
        results_df: DataFrame with classification results
        output_dir: Directory to save all results and visualizations
        low_quality_category: Name of the low-quality category

    Returns:
        Dictionary with all evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logging.info(f"Starting evaluation with {len(results_df)} images")

    logging.info("Running full evaluation with metrics")
    return evaluate_classification(
        manual_dir, results_df, output_path, low_quality_category
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate image classification results")
    parser.add_argument("--manual-dir", required=True, help="Directory with manually classified images")
    parser.add_argument("--results-csv", required=True, help="CSV file with classification results")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--low-quality-category", default="shit_data", help="Name of low-quality category")

    args = parser.parse_args()

    # Load results DataFrame
    results_df = pd.read_csv(args.results_csv)

    # Run evaluation
    run_evaluation(
        args.manual_dir,
        results_df,
        args.output_dir,
        args.low_quality_category,
    )