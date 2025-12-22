#!/usr/bin/env python3
"""
Visualization Utilities for Medical Image Classification
Functions for creating plots, charts, and visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_class_distribution(
    class_counts: Dict,
    title: str = "Class Distribution",
    save_path: Optional[str] = None,
):
    """Plot class distribution as bar chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = ax.bar(
        classes, counts, color=["#3498db", "#e74c3c"], alpha=0.7, edgecolor="black"
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sample_images(
    image_paths: List[str],
    labels: List[int],
    class_names: List[str],
    n_samples: int = 16,
    save_path: Optional[str] = None,
):
    """Plot grid of sample images."""
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    for idx in range(min(n_samples, len(image_paths))):
        img = cv2.imread(image_paths[idx])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{class_names[labels[idx]]}")
            axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history["train_losses"]) + 1)

    # Loss curves
    axes[0, 0].plot(
        epochs, history["train_losses"], "b-", label="Training Loss", linewidth=2
    )
    if "val_losses" in history:
        axes[0, 0].plot(
            epochs, history["val_losses"], "r-", label="Validation Loss", linewidth=2
        )
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    if "train_accuracies" in history:
        axes[0, 1].plot(
            epochs,
            history["train_accuracies"],
            "b-",
            label="Training Accuracy",
            linewidth=2,
        )
    if "val_accuracies" in history:
        axes[0, 1].plot(
            epochs,
            history["val_accuracies"],
            "r-",
            label="Validation Accuracy",
            linewidth=2,
        )
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC curves
    if "val_aucs" in history:
        axes[1, 0].plot(
            epochs, history["val_aucs"], "g-", label="Validation AUC", linewidth=2
        )
        axes[1, 0].set_title("Validation AUC-ROC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC-ROC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    if "learning_rates" in history:
        axes[1, 1].plot(epochs, history["learning_rates"], "purple", linewidth=2)
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_comparison(
    images: List[np.ndarray], titles: List[str], save_path: Optional[str] = None
):
    """Plot comparison of multiple images."""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

    if n_images == 1:
        axes = [axes]

    for idx, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:
            axes[idx].imshow(img, cmap="gray")
        else:
            axes[idx].imshow(img)
        axes[idx].set_title(title)
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_medical_report_visualization(
    image: np.ndarray,
    prediction: Dict,
    cam: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
):
    """Create comprehensive medical report visualization."""
    if cam is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    axes[0].set_title("Original X-Ray Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Prediction information
    axes[1].axis("off")
    prediction_text = f"""
    PREDICTION RESULTS

    Diagnosis: {prediction['class_name']}
    Confidence: {prediction['confidence']:.1%}

    CLASS PROBABILITIES:
    • Normal: {prediction.get('probabilities', {}).get('Normal', 0):.1%}
    • Pneumonia: {prediction.get('probabilities', {}).get('Pneumonia', 0):.1%}

    CLINICAL RECOMMENDATION:
    {prediction.get('recommendation', 'N/A')}
    """
    axes[1].text(
        0.1,
        0.5,
        prediction_text,
        fontsize=12,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )

    # Grad-CAM if available
    if cam is not None:
        from matplotlib import cm as colormap

        heatmap = colormap.jet(cam)[:, :, :3]
        overlayed = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image,
            0.5,
            (heatmap * 255).astype(np.uint8),
            0.5,
            0,
        )
        axes[2].imshow(overlayed)
        axes[2].set_title("Attention Heatmap", fontsize=14, fontweight="bold")
        axes[2].axis("off")

    plt.suptitle("Medical Image Analysis Report", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
