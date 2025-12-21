#!/usr/bin/env python3
"""
Training Pipeline for Medical Image Classification
Comprehensive training script with support for different models, optimizers,
and medical-specific evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import time
import os

# Setup logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import optional dependencies
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Logging to wandb will be disabled.")

# Import custom modules
from model_architectures import ModelFactory
from data_preprocessing import MedicalImagePreprocessor


class MedicalImageDataset(Dataset):
    """Dataset class for medical images."""

    def __init__(self, csv_file: str, transform=None, mode: str = "train"):
        """Initialize dataset."""
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item from dataset."""
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in medical datasets."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MedicalTrainer:
    """Comprehensive trainer for medical image classification."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.preprocessor = MedicalImagePreprocessor(config_path)
        self.best_val_auc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

        # Create directories
        self._create_directories()

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_device(self) -> torch.device:
        """Get appropriate device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _create_directories(self):
        """Create necessary directories."""
        dirs = ["models/saved_models", "logs", "results", "results/visualizations"]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        if WANDB_AVAILABLE and self.config["logging"].get("wandb", False):
            wandb.init(
                project="medical-image-classification",
                config=self.config,
                name=f"experiment_{int(time.time())}",
            )

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""

        # Define transforms
        train_transform = transforms.Compose(
            [
                transforms.Resize(self.config["data"]["image_size"]),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_test_transform = transforms.Compose(
            [
                transforms.Resize(self.config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets
        train_dataset = MedicalImageDataset(
            "data/splits/train.csv", transform=train_transform, mode="train"
        )
        val_dataset = MedicalImageDataset(
            "data/splits/val.csv", transform=val_test_transform, mode="val"
        )
        test_dataset = MedicalImageDataset(
            "data/splits/test.csv", transform=val_test_transform, mode="test"
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=True,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model = ModelFactory.create_model(
            architecture=self.config["model"]["architecture"],
            num_classes=self.config["model"]["num_classes"],
            pretrained=self.config["model"]["pretrained"],
            dropout_rate=self.config["model"]["dropout_rate"],
        )

        return model.to(self.device)

    def create_loss_function(self, train_loader: DataLoader) -> nn.Module:
        """Create loss function with class weights if specified."""
        loss_config = self.config["loss"]

        if loss_config["type"] == "focal_loss":
            return FocalLoss(
                alpha=loss_config["focal_alpha"], gamma=loss_config["focal_gamma"]
            )

        # Calculate class weights if specified
        if self.config["training"]["class_weights"] == "balanced":
            # Calculate class weights
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy())

            class_weights = compute_class_weight(
                "balanced", classes=np.unique(all_labels), y=all_labels
            )

            class_weights = torch.FloatTensor(class_weights).to(self.device)
            logger.info(f"Using class weights: {class_weights}")

            return nn.CrossEntropyLoss(weight=class_weights)

        return nn.CrossEntropyLoss()

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config["training"]["optimizer"].lower()
        lr = self.config["training"]["learning_rate"]
        weight_decay = self.config["training"]["weight_decay"]

        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def create_scheduler(
        self, optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config["training"]["scheduler"].lower()

        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["training"]["epochs"]
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5, factor=0.5
            )
        else:
            return None

    def calculate_medical_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate medical-specific metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] > 1 else 0.0

        # Medical-specific metrics
        tn, fp, fn, tp = np.bincount(y_true * 2 + y_pred, minlength=4)

        sensitivity = (
            tp / (tp + fn) if (tp + fn) > 0 else 0.0
        )  # Recall for positive class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True negative rate

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
    ) -> float:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/Training")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            # Update progress bar
            current_acc = correct_predictions / total_predictions
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{current_acc:.4f}"}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions

        return epoch_loss, epoch_acc

    def validate_epoch(
        self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module, epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc=f"Epoch {epoch+1}/Validation"):
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = criterion(output, target)

                running_loss += loss.item()

                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        epoch_loss = running_loss / len(val_loader)

        # Calculate metrics
        all_probabilities = np.array(all_probabilities)
        metrics = self.calculate_medical_metrics(
            np.array(all_targets), np.array(all_predictions), all_probabilities
        )

        return epoch_loss, metrics

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_auc: float,
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = Path("models/saved_models") / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = Path("models/saved_models") / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with AUC: {val_auc:.4f}")

    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info("Starting training process...")

        # Create data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Create model, loss, optimizer, scheduler
        model = self.create_model()
        criterion = self.create_loss_function(train_loader)
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)

        # Training loop
        patience_counter = 0
        best_val_auc = 0.0

        for epoch in range(self.config["training"]["epochs"]):
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )

            # Validate
            val_loss, val_metrics = self.validate_epoch(
                model, val_loader, criterion, epoch
            )

            # Update learning rate
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["auc_roc"])
                else:
                    scheduler.step()

            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}:")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}"
            )
            logger.info(
                f"Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}"
            )

            # Log to wandb
            if WANDB_AVAILABLE and self.config["logging"].get("wandb", False):
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_auc": val_metrics["auc_roc"],
                        "sensitivity": val_metrics["sensitivity"],
                        "specificity": val_metrics["specificity"],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            # Save checkpoint
            is_best = val_metrics["auc_roc"] > best_val_auc
            if is_best:
                best_val_auc = val_metrics["auc_roc"]
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(
                model, optimizer, epoch, val_metrics["auc_roc"], is_best
            )

            # Early stopping
            if patience_counter >= self.config["training"]["early_stopping_patience"]:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

            # Store metrics for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_metrics["auc_roc"])

        # Plot training curves
        self.plot_training_curves()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_aucs": self.val_aucs,
        }

    def plot_training_curves(self):
        """Plot training curves."""
        epochs = range(1, len(self.train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(epochs, self.train_losses, "b-", label="Training Loss")
        ax1.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # AUC curve
        ax2.plot(epochs, self.val_aucs, "g-", label="Validation AUC")
        ax2.set_title("Validation AUC-ROC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC-ROC")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            "results/visualizations/training_curves.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            "Training curves saved to results/visualizations/training_curves.png"
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train Medical Image Classification Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model", type=str, default="resnet50", help="Model architecture to use"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = MedicalTrainer(args.config)

    # Override model architecture if provided
    if args.model:
        trainer.config["model"]["architecture"] = args.model

    # Start training
    training_history = trainer.train()

    logger.info("Training completed successfully!")
    logger.info(f"Best validation AUC: {max(training_history['val_aucs']):.4f}")


if __name__ == "__main__":
    main()
