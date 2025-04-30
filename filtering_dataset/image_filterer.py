"""
image_filterer.py

Core module providing the ImageFilterer class. This class handles:
 - caching and tokenizing prompts
 - batching image preprocessing
 - running CLIP inference
 - computing confidence scores for multiple categories simultaneously
 - tracking detailed scores for advanced classification
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Tuple, Any, Union

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import pandas as pd
from tqdm import tqdm


class FilterConfig(TypedDict):
    """TypedDict describing a single filter stage."""
    name: str
    positive_texts: List[str]
    threshold: float


class ImageFilterer:
    """
    Encapsulates batch scoring and file operations for image filtering.

    Attributes:
        model        : Pre-trained CLIPModel on specified device.
        processor    : CLIPProcessor for text+image preprocessing.
        device       : 'cuda' or 'cpu'.
        logger       : Standard logger for informational messages.
        _text_cache  : Internal cache for tokenized prompts.
        _score_cache : Cache for detailed scoring results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: Optional[str] = None
    ):
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device and set to eval mode
        self.model = model.to(self.device)
        self.model.eval()

        # Store processor and initialize logger
        self.processor = processor
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache for prompt tokenization and detailed scores
        self._text_cache = {}
        self._score_cache = {}  # Maps image path to list of prompt scores

    def _tokenize_prompts(self, prompts: List[str]) -> dict:
        """
        Tokenize and cache text prompts to avoid redundant work.

        Args:
            prompts: List of text prompts describing desired images.

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        key = tuple(prompts)
        if key not in self._text_cache:
            tokens = self.processor.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            self._text_cache[key] = tokens
        return self._text_cache[key]

    def get_detailed_scores(self) -> Dict[str, List[float]]:
        """
        Returns the detailed per-prompt scores cached during batch processing.

        Returns:
            Dictionary mapping image paths to lists of confidence scores,
            where each list index corresponds to a prompt.
        """
        return self._score_cache

    def process_batch(
        self,
        image_paths: List[str],
        positive_texts: List[str],
        batch_size: int = 8
    ) -> pd.DataFrame:
        """
        Processes images in batches and computes a confidence score per image.
        Also stores detailed per-prompt scores for each image for later category analysis.

        Args:
            image_paths    : List[str] of image file paths.
            positive_texts : List[str] prompts for target detection.
            batch_size     : Number of images per inference batch.

        Returns:
            pd.DataFrame with columns ['path', 'filename', 'confidence'].
        """
        records = []
        # Clear any previous detailed scores
        self._score_cache.clear()

        # Cache tokenized prompts for reuse
        text_tokens = self._tokenize_prompts(positive_texts)
        num_prompts = len(positive_texts)

        # Iterate over images in streaming batches
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
            batch_paths = image_paths[start : start + batch_size]
            images, valid_paths = [], []

            # Load and convert images safely
            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        images.append(img.convert("RGB"))
                        valid_paths.append(path)
                except (OSError, UnidentifiedImageError) as err:
                    self.logger.warning(f"Skipping unreadable image {path}: {err}")

            if not images:
                continue

            # Preprocess images to pixel tensors
            image_inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Merge image and text inputs
            model_inputs = {
                "pixel_values": image_inputs["pixel_values"],
                **text_tokens
            }

            # Perform inference without gradient tracking
            with torch.no_grad():
                # Shape [batch_size, num_texts]
                logits = self.model(**model_inputs).logits_per_image
                # Convert logits to probabilities
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            # Process and store results for each image
            for idx, path in enumerate(valid_paths):
                # Store detailed per-prompt scores for this image
                prompt_scores = probabilities[idx, :num_prompts].tolist()
                self._score_cache[path] = prompt_scores

                # For backward compatibility, compute max confidence across all prompts
                conf_score = float(max(prompt_scores))

                records.append({
                    "path": path,
                    "filename": Path(path).name,
                    "confidence": conf_score  # Maximum confidence across all prompts
                })

        # Return results as DataFrame
        return pd.DataFrame(records)

    def filter_by_category(
        self,
        df: pd.DataFrame,
        categories: List[Dict[str, Any]],
        default_category: str = "normal_images"
    ) -> pd.DataFrame:
        """
        Assigns each image to its best matching category based on confidence scores.

        Args:
            df              : DataFrame from process_batch containing image paths.
            categories      : List of category configs with name, prompts, threshold.
            default_category: Name for images not matching any category.

        Returns:
            DataFrame with added 'category' column.
        """
        # Initialize result DataFrame with category column
        result_df = df.copy()
        result_df['category'] = default_category

        # Process each image
        for idx, row in result_df.iterrows():
            path = row['path']

            # Skip if image not in score cache
            if path not in self._score_cache:
                continue

            prompt_scores = self._score_cache[path]

            # Track best category match
            best_category = default_category
            best_score = 0.0

            # Check each category
            prompt_idx = 0
            for category in categories:
                category_name = category['name']
                threshold = float(category['threshold'])
                num_prompts = len(category['positive_texts'])

                # Calculate max score for this category's prompts
                category_scores = prompt_scores[prompt_idx:prompt_idx+num_prompts]
                category_max_score = max(category_scores) if category_scores else 0.0

                # Update if this is the best match so far
                if category_max_score >= threshold and category_max_score > best_score:
                    best_category = category_name
                    best_score = category_max_score

                prompt_idx += num_prompts

            # Assign best category
            result_df.loc[idx, 'category'] = best_category

        return result_df

    def move_matches(
        self,
        df: pd.DataFrame,
        threshold: float,
        dest_folder: Path
    ) -> List[str]:
        """
        Moves images with confidence >= threshold into a destination folder.

        Note: This is included for backward compatibility.
        The new approach uses filter_by_category for more granular control.

        Args:
            df          : DataFrame from process_batch.
            threshold   : Float cutoff to select images.
            dest_folder : Path to move selected images into.

        Returns:
            List of file paths moved.
        """
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Identify which images meet the threshold
        matches = df[df["confidence"] >= threshold]["path"].tolist()

        # Move each matched image
        for src_path in matches:
            src = Path(src_path)
            dst = dest_folder / src.name
            shutil.move(str(src), str(dst))

        self.logger.info(f"Moved {len(matches)} images to {dest_folder}")
        return matches