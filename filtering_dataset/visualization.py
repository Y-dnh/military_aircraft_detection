"""
visualization.py

Module for creating visualizations of classification results.
Provides functions for:
 - Distribution of images across categories
 - Score distributions within categories
 - Confusion analysis by examining borderline cases
 - Sample image display for each category
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


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


def plot_confusion_matrix(
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
) -> None:
    """
    Plot the distribution of borderline cases - images whose scores
    are close to the decision boundary between categories.

    Args:
        df: DataFrame with classification results
        output_path: Path to save the visualization
        threshold_margin: Margin around the threshold to consider borderline
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
        scores = [row[f'score_{cat}'] for cat in categories]
        scores.sort(reverse=True)

        if len(scores) >= 2:
            # Calculate the margin between top two scores
            df_with_margins.loc[i, 'score_margin'] = scores[0] - scores[1]

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

    # Iterate through categories
    for i, category in enumerate(categories):
        # Get images from this category
        cat_images = df[df['best_category'] == category]

        # Sample images from this category
        sample_count = min(len(cat_images), samples_per_category)
        samples = cat_images.sample(sample_count)

        # Plot each sample
        for j, (_, row) in enumerate(samples.iterrows()):
            if j >= grid_cols:
                break

            # Load and display image
            try:
                img = Image.open(row['path'])

                # Turn off axes for images
                if num_categories == 1:
                    ax = axes[j]
                else:
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
                print(f"Error loading image {row['path']}: {e}")

        # If we have fewer samples than columns, turn off remaining axes
        if sample_count < grid_cols:
            for j in range(sample_count, grid_cols):
                if num_categories == 1:
                    axes[j].axis('off')
                else:
                    axes[i, j].axis('off')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_all_visualizations(
        results_df: pd.DataFrame,
        output_dir: Path
) -> None:
    """
    Create all visualizations and save them to the output directory.

    Args:
        results_df: DataFrame with classification results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual visualizations
    plot_category_distribution(
        results_df,
        output_dir / 'category_distribution.png'
    )

    plot_score_distributions(
        results_df,
        output_dir / 'score_distributions.png'
    )

    plot_confusion_matrix(
        results_df,
        output_dir / 'confusion_matrix.png'
    )

    borderline_cases = plot_borderline_cases(
        results_df,
        output_dir / 'borderline_cases.png'
    )

    # Save borderline cases to CSV for further analysis
    if borderline_cases is not None and not borderline_cases.empty:
        borderline_cases.to_csv(output_dir / 'borderline_cases.csv', index=False)

    create_sample_grid(
        results_df,
        output_dir / 'sample_images.png'
    )

    # Additional visualization: Confidence score distribution by category
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='best_category', y='best_score', data=results_df, ax=ax)
    ax.set_title('Confidence Score Distribution by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_confidence.png')
    plt.close()

    print(f"All visualizations saved to {output_dir}")