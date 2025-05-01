"""
image_filterer.py

Core module providing the ImageFilterer class. This class handles:
 - caching and tokenizing prompts
 - image preprocessing without batching for simplicity
 - running CLIP inference
 - computing confidence scores for multiple categories simultaneously
 - tracking detailed scores for advanced classification
 - handling negative prompts for improved classification
 - implementing min_difference checks for category confidence
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import pandas as pd
from tqdm import tqdm


class ImageFilterer:
    """
    Encapsulates image scoring and classification using the CLIP model.

    This implementation processes images one by one without batching for simplicity
    and greater control over the process. It stores detailed scoring results
    for each image and prompt combination.

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
        Returns the detailed per-prompt scores cached during processing.

        Returns:
            Dictionary mapping image paths to lists of confidence scores,
            where each list index corresponds to a prompt.
        """
        return self._score_cache

    def process_images(
        self,
        image_paths: List[str],
        positive_texts: List[str],
        negative_texts: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Processes images one by one and computes a confidence score per image.
        Also stores detailed per-prompt scores for each image for category analysis.

        Args:
            image_paths    : List[str] of image file paths.
            positive_texts : List[str] prompts for target detection.
            negative_texts : Optional List[str] of negative prompts to penalize.

        Returns:
            pd.DataFrame with columns ['path', 'filename', 'confidence'].
        """
        records = []
        # Clear any previous detailed scores
        self._score_cache.clear()

        # Cache tokenized prompts for reuse
        text_tokens = self._tokenize_prompts(positive_texts)
        num_positive_prompts = len(positive_texts)

        # Process negative texts if provided
        negative_text_tokens = None
        if negative_texts and len(negative_texts) > 0:
            negative_text_tokens = self._tokenize_prompts(negative_texts)
            self.logger.info(f"Using {len(negative_texts)} negative prompts for classification")

        # Process images one by one for simplicity
        for path in tqdm(image_paths, desc="Processing Images"):
            try:
                # Load and convert image safely
                with Image.open(path) as img:
                    image = img.convert("RGB")
            except (OSError, UnidentifiedImageError) as err:
                self.logger.warning(f"Skipping unreadable image {path}: {err}")
                continue

            # Preprocess image to pixel tensor
            image_inputs = self.processor(
                images=[image],  # Single image in a list
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Process positive prompts
            positive_scores = self._process_single_image(image_inputs, text_tokens, num_positive_prompts)

            # Process negative prompts if provided
            negative_scores = []
            if negative_text_tokens is not None:
                negative_scores = self._process_single_image(
                    image_inputs,
                    negative_text_tokens,
                    len(negative_texts)
                )

            # Store detailed scores for this image (positive scores only)
            self._score_cache[path] = positive_scores

            # Compute confidence score: max positive score minus max negative score if negative exists
            pos_conf_score = float(max(positive_scores)) if positive_scores else 0.0
            neg_conf_score = float(max(negative_scores)) if negative_scores else 0.0

            # Final confidence is positive score reduced by negative matches
            # Only penalize if negative scores exist
            if negative_scores:
                # Use a penalty weight of 0.5 to balance the influence of negative prompts
                neg_penalty_weight = 0.5
                conf_score = pos_conf_score - (neg_conf_score * neg_penalty_weight)
                # Ensure score doesn't go below 0
                conf_score = max(0.0, conf_score)
            else:
                conf_score = pos_conf_score

            records.append({
                "path": path,
                "filename": Path(path).name,
                "confidence": conf_score,
                "pos_score": pos_conf_score,
                "neg_score": neg_conf_score if negative_scores else 0.0
            })

        # Return results as DataFrame
        return pd.DataFrame(records)

    def _process_single_image(
        self,
        image_inputs: dict,
        text_tokens: dict,
        num_prompts: int
    ) -> List[float]:
        """
        Helper method to process a single image against a set of text prompts.

        Args:
            image_inputs: Preprocessed image inputs
            text_tokens: Tokenized text prompts
            num_prompts: Number of prompts to process

        Returns:
            List of scores for each prompt
        """
        # Merge image and text inputs
        model_inputs = {
            "pixel_values": image_inputs["pixel_values"],
            **text_tokens
        }

        # Perform inference without gradient tracking
        with torch.no_grad():
            # Shape [1, num_texts] since we're processing one image
            logits = self.model(**model_inputs).logits_per_image
            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        # Return scores for all prompts
        return probabilities[0, :num_prompts].tolist()

    def classify_images(
        self,
        image_paths: List[str],
        categories: List[Dict[str, Union[str, List[str], float]]],
        default_category: str = "normal_images",
        use_negative_prompts: bool = True,
        confidence_strategy: str = "fixed"
    ) -> pd.DataFrame:
        """
        Processes all images against all categories and assigns each to its best match.

        This is a more streamlined approach that combines processing and classification
        in a single function call for simpler usage. Now supports negative prompts and
        minimum confidence difference between categories.

        Args:
            image_paths         : List of paths to images to evaluate
            categories          : List of category configs with name, prompts, threshold
            default_category    : Name for images not matching any category
            use_negative_prompts: Whether to use negative prompts in classification
            confidence_strategy : Strategy for confidence thresholds ("fixed" or "adaptive")

        Returns:
            DataFrame with paths, scores for each category, and best matching category
        """
        # Extract all positive prompts from all categories
        all_positive_prompts = []
        all_negative_prompts = []
        category_names = []
        thresholds = {}
        min_differences = {}

        # Build combined prompt list while tracking category information
        prompt_to_category_map = {}
        prompt_index = 0

        for category in categories:
            category_name = category['name']
            category_names.append(category_name)
            thresholds[category_name] = float(category['threshold'])

            # Store minimum difference requirement if provided
            if 'min_difference' in category:
                min_differences[category_name] = float(category['min_difference'])
            else:
                min_differences[category_name] = 0.0

            # Collect positive prompts
            for prompt in category['positive_texts']:
                all_positive_prompts.append(prompt)
                prompt_to_category_map[prompt_index] = category_name
                prompt_index += 1

            # Collect negative prompts if enabled and available
            if use_negative_prompts and 'negative_texts' in category and category['negative_texts']:
                for neg_prompt in category['negative_texts']:
                    all_negative_prompts.append(neg_prompt)

        # Process all images against all prompts
        self.logger.info(f"Processing {len(image_paths)} images against {len(all_positive_prompts)} positive prompts")
        if use_negative_prompts and all_negative_prompts:
            self.logger.info(f"Using {len(all_negative_prompts)} negative prompts for refinement")
            results_df = self.process_images(
                image_paths,
                all_positive_prompts,
                all_negative_prompts
            )
        else:
            results_df = self.process_images(image_paths, all_positive_prompts)

        # Create output DataFrame with scores for each category
        classified_df = pd.DataFrame()
        classified_df['path'] = results_df['path']
        classified_df['filename'] = results_df['filename']

        # Initialize score columns for each category
        for category_name in category_names:
            classified_df[f'score_{category_name}'] = 0.0

        # Map detailed scores back to their categories
        detailed_scores = self.get_detailed_scores()

        for i, row in classified_df.iterrows():
            img_path = row['path']
            if img_path in detailed_scores:
                scores = detailed_scores[img_path]

                # Initialize best score per category
                category_scores = {name: 0.0 for name in category_names}

                # Find the best score for each category
                for prompt_idx, score in enumerate(scores):
                    if prompt_idx in prompt_to_category_map:
                        category = prompt_to_category_map[prompt_idx]
                        category_scores[category] = max(category_scores[category], score)

                # Set the best score for each category
                for category, score in category_scores.items():
                    classified_df.loc[i, f'score_{category}'] = score

        # Add best_category column based on scores and thresholds
        # Initialize with default category
        classified_df['best_category'] = default_category
        classified_df['best_score'] = 0.0
        classified_df['second_best_category'] = ''
        classified_df['second_best_score'] = 0.0

        # Determine best category for each image
        for i, row in classified_df.iterrows():
            best_category = default_category
            best_score = 0.0
            second_best_category = ''
            second_best_score = 0.0

            # Get all category scores for this image
            category_scores = [(cat, row[f'score_{cat}']) for cat in category_names]

            # Sort by score in descending order
            category_scores.sort(key=lambda x: x[1], reverse=True)

            # Check if best category meets threshold
            if category_scores:
                top_category, top_score = category_scores[0]
                if top_score >= thresholds[top_category]:
                    # We have a potential match, but need to check min_difference if required
                    if len(category_scores) > 1:
                        # Get second best category and score
                        second_category, second_score = category_scores[1]
                        second_best_category = second_category
                        second_best_score = second_score

                        # Check if difference meets min_difference requirement
                        min_diff_required = min_differences[top_category]
                        actual_diff = top_score - second_score

                        if actual_diff >= min_diff_required:
                            # Passes min_difference check
                            best_category = top_category
                            best_score = top_score
                        else:
                            # Fails min_difference check, use default category
                            self.logger.debug(
                                f"Image {row['filename']} narrowly missed {top_category} " +
                                f"(diff: {actual_diff:.3f}, required: {min_diff_required:.3f})"
                            )
                    else:
                        # Only one category, no need for min_difference check
                        best_category = top_category
                        best_score = top_score

            # Update the DataFrame
            classified_df.loc[i, 'best_category'] = best_category
            classified_df.loc[i, 'best_score'] = best_score
            classified_df.loc[i, 'second_best_category'] = second_best_category
            classified_df.loc[i, 'second_best_score'] = second_best_score

        return classified_df