#!/usr/bin/env python3
"""
Transfer Learning for Medical Image Classification
Implementation of transfer learning techniques using pre-trained models
optimized for medical imaging applications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from pathlib import Path
import yaml
from model_architectures import ModelFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferLearningManager:
    """
    Comprehensive transfer learning manager for medical image classification.
    Handles model adaptation, fine-tuning strategies, and optimization.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize transfer learning manager."""
        self.config = self._load_config(config_path)
        self.device = self._get_device()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_device(self) -> torch.device:
        """Get appropriate device for training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def create_transfer_learning_model(
        self,
        architecture: str,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_strategy: str = "partial",
    ) -> nn.Module:
        """Create transfer learning model with specified freezing strategy."""

        # Create base model
        model = ModelFactory.create_model(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=self.config["model"].get("freeze_backbone", False),
            fine_tune_layers=self.config["model"].get("fine_tune_layers", 10),
        )

        # Apply freezing strategy
        self._apply_freeze_strategy(model, freeze_strategy)

        return model.to(self.device)

    def _apply_freeze_strategy(self, model: nn.Module, strategy: str):
        """Apply different freezing strategies to the model."""

        if strategy == "none":
            # No freezing - fine-tune all layers
            for param in model.parameters():
                param.requires_grad = True

        elif strategy == "full":
            # Freeze all layers except classifier
            for name, param in model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif strategy == "partial":
            # Freeze early layers, fine-tune later layers
            layers = list(model.children())
            freeze_count = len(layers) // 2

            for i, layer in enumerate(layers):
                if i < freeze_count:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True

        elif strategy == "gradual":
            # Gradual unfreezing strategy (to be used during training)
            # Start with all frozen, gradually unfreeze
            for param in model.parameters():
                param.requires_grad = False

            # Only classifier is trainable initially
            for name, param in model.named_parameters():
                if "classifier" in name or "fc" in name:
                    param.requires_grad = True

        logger.info(f"Applied freeze strategy: {strategy}")
        self._log_trainable_parameters(model)

    def _log_trainable_parameters(self, model: nn.Module):
        """Log information about trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    def setup_progressive_unfreezing(self, model: nn.Module) -> Dict[int, List[str]]:
        """Setup progressive unfreezing schedule."""
        # Define unfreezing schedule by epoch
        unfreeze_schedule = {
            0: ["classifier", "fc"],  # Start with classifier only
            10: ["layer4", "classifier", "fc"],  # Unfreeze last ResNet block
            20: ["layer3", "layer4", "classifier", "fc"],  # Unfreeze more blocks
            30: [],  # Unfreeze all (empty list means all layers)
        }

        return unfreeze_schedule

    def apply_progressive_unfreezing(
        self, model: nn.Module, epoch: int, unfreeze_schedule: Dict[int, List[str]]
    ):
        """Apply progressive unfreezing based on epoch."""

        # Find the appropriate unfreezing stage
        current_stage = None
        for schedule_epoch in sorted(unfreeze_schedule.keys(), reverse=True):
            if epoch >= schedule_epoch:
                current_stage = unfreeze_schedule[schedule_epoch]
                break

        if current_stage is None:
            return

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze specified layers
        if len(current_stage) == 0:  # Unfreeze all
            for param in model.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                for layer_name in current_stage:
                    if layer_name in name:
                        param.requires_grad = True
                        break

        logger.info(f"Epoch {epoch}: Progressive unfreezing applied")
        self._log_trainable_parameters(model)

    def setup_differential_learning_rates(
        self, model: nn.Module, base_lr: float = 0.001
    ) -> optim.Optimizer:
        """Setup differential learning rates for different parts of the network."""

        # Define different learning rates for different parts
        param_groups = [
            {
                "params": [],
                "lr": base_lr * 0.1,  # Lower LR for frozen/backbone layers
                "name": "backbone",
            },
            {
                "params": [],
                "lr": base_lr,  # Higher LR for classifier layers
                "name": "classifier",
            },
        ]

        # Assign parameters to groups
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "classifier" in name or "fc" in name:
                    param_groups[1]["params"].append(param)
                else:
                    param_groups[0]["params"].append(param)

        # Remove empty groups
        param_groups = [group for group in param_groups if group["params"]]

        # Create optimizer with parameter groups
        optimizer = optim.Adam(
            param_groups, weight_decay=self.config["training"]["weight_decay"]
        )

        logger.info(f"Setup differential learning rates: {len(param_groups)} groups")
        for i, group in enumerate(param_groups):
            logger.info(
                f"Group {i} ({group['name']}): LR={group['lr']}, Params={len(group['params'])}"
            )

        return optimizer

    def create_ensemble_transfer_models(
        self, architectures: List[str], num_classes: int = 2
    ) -> List[nn.Module]:
        """Create ensemble of transfer learning models."""
        models = []

        for arch in architectures:
            model = self.create_transfer_learning_model(
                architecture=arch, num_classes=num_classes, freeze_strategy="partial"
            )
            models.append(model)
            logger.info(f"Created {arch} model for ensemble")

        return models

    def fine_tune_medical_features(
        self, model: nn.Module, medical_layer_names: List[str]
    ):
        """Fine-tune specific layers that are important for medical imaging."""

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze medical-relevant layers
        for name, param in model.named_parameters():
            for medical_layer in medical_layer_names:
                if medical_layer in name:
                    param.requires_grad = True
                    logger.info(f"Unfroze medical layer: {name}")
                    break

        # Always keep classifier trainable
        for name, param in model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True

    def adapt_model_for_medical_imaging(self, model: nn.Module) -> nn.Module:
        """Adapt pre-trained model specifically for medical imaging."""

        # Add medical-specific preprocessing layers if needed
        if hasattr(model, "backbone"):
            # Add attention mechanism for medical feature focus
            original_features = model.backbone.avgpool

            # Medical attention module
            medical_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(2048, 512, 1),  # Assuming ResNet50
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 2048, 1),
                nn.Sigmoid(),
            )

            # Replace avgpool with attention-enhanced version
            model.backbone.avgpool = nn.Sequential(medical_attention, original_features)

        return model

    def evaluate_transfer_learning_effectiveness(
        self,
        pretrained_model: nn.Module,
        scratch_model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the effectiveness of transfer learning vs training from scratch."""

        results = {}
        models = {"pretrained": pretrained_model, "from_scratch": scratch_model}

        for model_name, model in models.items():
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            accuracy = correct / total
            results[f"{model_name}_accuracy"] = accuracy
            logger.info(f"{model_name.capitalize()} model accuracy: {accuracy:.4f}")

        # Calculate improvement
        improvement = results["pretrained_accuracy"] - results["from_scratch_accuracy"]
        results["transfer_learning_improvement"] = improvement

        logger.info(f"Transfer learning improvement: {improvement:.4f}")

        return results


def main():
    """Main function for testing transfer learning functionality."""
    # Initialize transfer learning manager
    tl_manager = TransferLearningManager()

    # Test model creation
    model = tl_manager.create_transfer_learning_model(
        architecture="resnet50", num_classes=2, freeze_strategy="partial"
    )

    logger.info("Transfer learning model created successfully!")

    # Test progressive unfreezing schedule
    schedule = tl_manager.setup_progressive_unfreezing(model)
    logger.info(f"Progressive unfreezing schedule: {schedule}")

    # Test differential learning rates
    optimizer = tl_manager.setup_differential_learning_rates(model)
    logger.info("Differential learning rates setup completed!")


if __name__ == "__main__":
    main()
