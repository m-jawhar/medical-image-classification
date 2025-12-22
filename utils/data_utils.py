#!/usr/bin/env python3
"""
Data Utilities for Medical Image Classification
Helper functions for data loading, preprocessing, and handling.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str, mode: str = "RGB") -> np.ndarray:
    """Load image from path."""
    try:
        image = Image.open(image_path).convert(mode)
        return np.array(image)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image to specified dimensions."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def normalize_image(
    image: np.ndarray,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> np.ndarray:
    """Normalize image with mean and std."""
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def load_dataset_from_csv(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} samples from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {csv_path}: {str(e)}")
        return None


def get_class_distribution(
    df: pd.DataFrame, label_column: str = "label"
) -> Dict[int, int]:
    """Get class distribution from dataframe."""
    return df[label_column].value_counts().to_dict()


def create_balanced_subset(
    df: pd.DataFrame, samples_per_class: int, label_column: str = "label"
) -> pd.DataFrame:
    """Create balanced subset of dataset."""
    balanced_dfs = []
    for class_label in df[label_column].unique():
        class_df = df[df[label_column] == class_label]
        if len(class_df) >= samples_per_class:
            balanced_dfs.append(class_df.sample(n=samples_per_class, random_state=42))
        else:
            balanced_dfs.append(class_df)

    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)


def verify_image_paths(
    df: pd.DataFrame, image_column: str = "image_path"
) -> Tuple[pd.DataFrame, List[str]]:
    """Verify that all image paths in dataframe exist."""
    invalid_paths = []
    valid_indices = []

    for idx, row in df.iterrows():
        image_path = row[image_column]
        if Path(image_path).exists():
            valid_indices.append(idx)
        else:
            invalid_paths.append(image_path)

    if invalid_paths:
        logger.warning(f"Found {len(invalid_paths)} invalid image paths")

    return df.loc[valid_indices], invalid_paths


def get_image_statistics(image_paths: List[str]) -> Dict[str, any]:
    """Calculate statistics for a collection of images."""
    dimensions = []
    intensities = []
    file_sizes = []

    for path in image_paths:
        if Path(path).exists():
            # Image dimensions
            img = cv2.imread(path)
            if img is not None:
                dimensions.append(img.shape)
                intensities.extend(img.flatten())

            # File size
            file_sizes.append(Path(path).stat().st_size)

    stats = {
        "num_images": len(image_paths),
        "dimensions": dimensions,
        "mean_intensity": np.mean(intensities) if intensities else 0,
        "std_intensity": np.std(intensities) if intensities else 0,
        "mean_file_size_mb": np.mean(file_sizes) / (1024**2) if file_sizes else 0,
    }

    return stats


def save_predictions_to_csv(predictions: List[Dict], output_path: str):
    """Save predictions to CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)

    return weights
