#!/usr/bin/env python3
"""
Medical Image Preprocessing Pipeline
Advanced preprocessing techniques for chest X-ray images including CLAHE,
normalization, lung segmentation, and medical-specific augmentations.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageEnhance
import yaml
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Comprehensive medical image preprocessing pipeline optimized for chest X-rays.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.image_size = tuple(self.config["data"]["image_size"])
        self.setup_transforms()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def setup_transforms(self):
        """Setup image transformation pipelines."""
        # Medical-specific preprocessing transforms
        self.medical_transforms = A.Compose(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.CLAHE(
                    clip_limit=self.config["preprocessing"]["clahe_clip_limit"],
                    tile_grid_size=(8, 8),
                    p=0.8,
                ),
                A.Normalize(
                    mean=[0.485], std=[0.229]
                ),  # ImageNet stats adapted for grayscale
                ToTensorV2(),
            ]
        )

        # Training augmentation transforms
        aug_config = self.config["augmentation"]
        self.augmentation_transforms = A.Compose(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=aug_config["rotation_range"], p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=aug_config["rotation_range"],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=0.7
                ),
                A.GaussianBlur(blur_limit=(3, 3), p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    p=0.3,
                ),
                A.CLAHE(
                    clip_limit=self.config["preprocessing"]["clahe_clip_limit"],
                    tile_grid_size=(8, 8),
                    p=0.8,
                ),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )

        # Validation/Test transforms (no augmentation)
        self.validation_transforms = A.Compose(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.CLAHE(
                    clip_limit=self.config["preprocessing"]["clahe_clip_limit"],
                    tile_grid_size=(8, 8),
                    p=1.0,
                ),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ]
        )

    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)

    def enhance_lung_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance lung region contrast specifically for medical imaging."""
        # Apply histogram equalization
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply adaptive histogram equalization
        equalized = cv2.equalizeHist(image)

        # Combine original and equalized image
        enhanced = cv2.addWeighted(image, 0.4, equalized, 0.6, 0)

        return enhanced

    def preprocess_medical_image(
        self, image_path: str, mode: str = "train"
    ) -> np.ndarray:
        """Preprocess a single medical image."""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to 3-channel for consistency with pre-trained models
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply appropriate transforms based on mode
        if mode == "train" and self.config["augmentation"]["enabled"]:
            transformed = self.augmentation_transforms(image=image)
        else:
            transformed = self.validation_transforms(image=image)

        return transformed["image"]

    def create_data_splits(
        self, data_dir: str, output_dir: str
    ) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits with balanced distribution."""
        data_path = Path(data_dir)
        splits_path = Path(output_dir)
        splits_path.mkdir(parents=True, exist_ok=True)

        # Collect all image paths and labels
        image_data = []

        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                label = 1 if "pneumonia" in class_name else 0

                for img_path in class_dir.glob("*.jpeg"):
                    image_data.append(
                        {
                            "image_path": str(img_path),
                            "label": label,
                            "class_name": class_name,
                        }
                    )

        df = pd.DataFrame(image_data)
        logger.info(f"Total images found: {len(df)}")
        logger.info(f"Class distribution:\n{df['class_name'].value_counts()}")

        # Stratified split
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config["data"]["test_split"],
            stratify=df["label"],
            random_state=42,
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config["data"]["val_split"]
            / (1 - self.config["data"]["test_split"]),
            stratify=train_val_df["label"],
            random_state=42,
        )

        # Save splits
        splits = {"train": train_df, "validation": val_df, "test": test_df}

        for split_name, split_df in splits.items():
            split_df.to_csv(splits_path / f"{split_name}.csv", index=False)
            logger.info(f"{split_name.capitalize()} set: {len(split_df)} images")
            logger.info(
                f"{split_name.capitalize()} class distribution:\n{split_df['class_name'].value_counts()}"
            )

        return splits

    def analyze_dataset_statistics(self, data_dir: str) -> Dict[str, any]:
        """Analyze dataset statistics for medical images."""
        data_path = Path(data_dir)
        stats = {
            "total_images": 0,
            "class_distribution": {},
            "image_dimensions": [],
            "pixel_intensities": [],
        }

        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_count = len(list(class_dir.glob("*.jpeg")))
                stats["class_distribution"][class_name] = image_count
                stats["total_images"] += image_count

                # Sample images for dimension and intensity analysis
                for i, img_path in enumerate(class_dir.glob("*.jpeg")):
                    if i >= 100:  # Sample first 100 images per class
                        break

                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        stats["image_dimensions"].append(img.shape)
                        stats["pixel_intensities"].extend(img.flatten())

        return stats

    def visualize_preprocessing_effects(
        self, image_path: str, save_dir: str = "results/visualizations"
    ):
        """Visualize the effects of different preprocessing techniques."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load original image
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply different preprocessing techniques
        clahe_applied = self.apply_clahe(original)
        enhanced = self.enhance_lung_contrast(original)

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(original, cmap="gray")
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # CLAHE applied
        axes[0, 1].imshow(clahe_applied, cmap="gray")
        axes[0, 1].set_title("CLAHE Applied")
        axes[0, 1].axis("off")

        # Enhanced contrast
        axes[0, 2].imshow(enhanced, cmap="gray")
        axes[0, 2].set_title("Enhanced Contrast")
        axes[0, 2].axis("off")

        # Histograms
        axes[1, 0].hist(original.flatten(), bins=50, alpha=0.7, color="blue")
        axes[1, 0].set_title("Original Histogram")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].hist(clahe_applied.flatten(), bins=50, alpha=0.7, color="green")
        axes[1, 1].set_title("CLAHE Histogram")
        axes[1, 1].set_xlabel("Pixel Intensity")

        axes[1, 2].hist(enhanced.flatten(), bins=50, alpha=0.7, color="red")
        axes[1, 2].set_title("Enhanced Histogram")
        axes[1, 2].set_xlabel("Pixel Intensity")

        plt.tight_layout()
        plt.savefig(
            save_path / "preprocessing_effects.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Preprocessing visualization saved to {save_path / 'preprocessing_effects.png'}"
        )


def main():
    """Main function to run preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor()

    # Example usage - update paths as needed
    # data_dir = "data/raw/chest_xray"  # Update with actual path
    # splits = preprocessor.create_data_splits(data_dir, "data/splits")

    # Analyze dataset statistics
    # stats = preprocessor.analyze_dataset_statistics(data_dir)
    # logger.info(f"Dataset statistics: {stats}")

    logger.info("Preprocessing pipeline initialized successfully!")
    logger.info("To use: update data paths in main() function and run again.")


if __name__ == "__main__":
    main()
