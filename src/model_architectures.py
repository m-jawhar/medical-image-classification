#!/usr/bin/env python3
"""
Medical Image Classification Model Architectures
Custom CNN architectures and transfer learning models optimized for chest X-ray classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """
    Custom CNN architecture designed specifically for medical image classification.
    Features: Multiple convolutional blocks, batch normalization, dropout, and attention.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super(CustomCNN, self).__init__()

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Fourth convolutional block
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1),
            nn.Sigmoid(),
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


class ResNetMedical(nn.Module):
    """
    ResNet-based model for medical image classification with custom modifications.
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        fine_tune_layers: int = 10,
    ):
        super(ResNetMedical, self).__init__()

        # Load pre-trained ResNet
        if architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Modify first convolution for medical images (optional enhancement)
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Fine-tune only last N layers
            layers = list(self.backbone.children())
            for layer in layers[:-fine_tune_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Medical-specific attention module
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),  # For ResNet50
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass through ResNet with attention."""
        # Extract features up to avgpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights

        # Average pooling and classification
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x


class DenseNetMedical(nn.Module):
    """
    DenseNet-based model for medical image classification.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(DenseNetMedical, self).__init__()

        # Load pre-trained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)

        # Replace classifier
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Forward pass through DenseNet."""
        return self.backbone(x)


class EfficientNetMedical(nn.Module):
    """
    EfficientNet-based model for medical image classification.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super(EfficientNetMedical, self).__init__()

        # Load pre-trained EfficientNet
        if architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
        elif architecture == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """Forward pass through EfficientNet."""
        return self.backbone(x)


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures for improved performance.
    """

    def __init__(self, models_list: list, num_classes: int = 2):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.num_models = len(models_list)

        # Weighted combination layer
        self.combination = nn.Linear(num_classes * self.num_models, num_classes)

    def forward(self, x):
        """Forward pass through ensemble."""
        outputs = []

        for model in self.models:
            output = model(x)
            outputs.append(output)

        # Concatenate outputs
        combined = torch.cat(outputs, dim=1)

        # Final prediction
        final_output = self.combination(combined)

        return final_output


class ModelFactory:
    """
    Factory class for creating different model architectures.
    """

    @staticmethod
    def create_model(
        architecture: str, num_classes: int = 2, pretrained: bool = True, **kwargs
    ) -> nn.Module:
        """Create model based on architecture name."""

        if architecture.lower() == "custom_cnn":
            return CustomCNN(
                num_classes=num_classes, dropout_rate=kwargs.get("dropout_rate", 0.5)
            )

        elif architecture.lower() in ["resnet50", "resnet101"]:
            return ResNetMedical(
                architecture=architecture.lower(),
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=kwargs.get("freeze_backbone", False),
                fine_tune_layers=kwargs.get("fine_tune_layers", 10),
            )

        elif architecture.lower() == "densenet121":
            return DenseNetMedical(num_classes=num_classes, pretrained=pretrained)

        elif architecture.lower() in ["efficientnet_b0", "efficientnet_b3"]:
            return EfficientNetMedical(
                architecture=architecture.lower(),
                num_classes=num_classes,
                pretrained=pretrained,
            )

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @staticmethod
    def create_ensemble(
        architectures: list, num_classes: int = 2, pretrained: bool = True, **kwargs
    ) -> EnsembleModel:
        """Create ensemble model from multiple architectures."""
        models_list = []

        for arch in architectures:
            model = ModelFactory.create_model(
                arch, num_classes=num_classes, pretrained=pretrained, **kwargs
            )
            models_list.append(model)

        return EnsembleModel(models_list, num_classes)


# Model summary utility
def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model summary."""
    summary = {
        "total_parameters": count_parameters(model),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters())
        / (1024**2),
        "architecture": model.__class__.__name__,
    }

    return summary


if __name__ == "__main__":
    # Test model creation
    logger.info("Testing model architectures...")

    # Test custom CNN
    custom_model = ModelFactory.create_model("custom_cnn")
    logger.info(f"Custom CNN: {get_model_summary(custom_model)}")

    # Test ResNet
    resnet_model = ModelFactory.create_model("resnet50")
    logger.info(f"ResNet50: {get_model_summary(resnet_model)}")

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = custom_model(dummy_input)
        logger.info(f"Output shape: {output.shape}")

    logger.info("Model architectures tested successfully!")
