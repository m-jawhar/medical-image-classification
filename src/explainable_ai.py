#!/usr/bin/env python3
"""
Explainable AI for Medical Image Classification
Implementation of Grad-CAM, attention visualization, and interpretability
techniques for medical imaging applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

# Import custom modules
from model_architectures import ModelFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation
    for visualizing which regions of the image are important for predictions.
    """

    def __init__(self, model: nn.Module, target_layer: str = None):
        """Initialize Grad-CAM."""
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find target layer
        target_layer = self._find_target_layer()

        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            logger.info(f"Hooks registered on layer: {target_layer}")

    def _find_target_layer(self) -> Optional[nn.Module]:
        """Find the target layer for Grad-CAM visualization."""
        # For ResNet architectures
        if hasattr(self.model, "backbone"):
            if hasattr(self.model.backbone, "layer4"):
                return self.model.backbone.layer4[-1]  # Last block of layer4

        # For custom CNN
        if hasattr(self.model, "conv_block4"):
            return self.model.conv_block4

        # Generic approach: find last convolutional layer
        conv_layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(module)

        if conv_layers:
            return conv_layers[-1]

        logger.warning("Could not find suitable target layer for Grad-CAM")
        return None

    def generate_cam(
        self, input_image: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Class Activation Map."""
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_image)

        # Target class for backward pass
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            logger.error("Gradients or activations not captured")
            return None

        # Calculate weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU to get only positive influences
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def visualize_cam(
        self, original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay CAM on original image."""
        # Resize CAM to match image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))

        # Apply colormap
        heatmap = cm.jet(cam_resized)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Ensure original image is RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

        # Overlay
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

        return overlayed


class MedicalImageExplainer:
    """
    Comprehensive explainability toolkit for medical image classification.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize explainer."""
        self.model = model
        self.device = device
        self.grad_cam = GradCAM(model)
        self.class_names = ["Normal", "Pneumonia"]

    def explain_prediction(self, image_path: str, preprocess_fn=None) -> Dict:
        """Generate comprehensive explanation for a single prediction."""
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if preprocess_fn:
            input_tensor = preprocess_fn(image_path)
        else:
            # Default preprocessing
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            input_tensor = transform(original_image)

        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Generate Grad-CAM
        cam = self.grad_cam.generate_cam(input_tensor, predicted_class)

        # Create visualization
        if cam is not None:
            overlayed = self.grad_cam.visualize_cam(original_image, cam)
        else:
            overlayed = original_image

        explanation = {
            "predicted_class": predicted_class,
            "class_name": self.class_names[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities.cpu().numpy()[0],
            "original_image": original_image,
            "cam": cam,
            "overlayed_image": overlayed,
        }

        return explanation

    def batch_explain(
        self, image_paths: List[str], save_dir: str = "results/explanations"
    ):
        """Generate explanations for multiple images."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")

            try:
                explanation = self.explain_prediction(image_path)
                self.save_explanation(
                    explanation,
                    save_path / f"explanation_{idx}.png",
                    Path(image_path).name,
                )
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")

    def save_explanation(
        self, explanation: Dict, save_path: Path, original_filename: str
    ):
        """Save explanation visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(explanation["original_image"])
        axes[0].set_title(f"Original Image\n{original_filename}")
        axes[0].axis("off")

        # Grad-CAM heatmap
        if explanation["cam"] is not None:
            axes[1].imshow(explanation["cam"], cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap\n(Red = High Importance)")
        axes[1].axis("off")

        # Overlayed visualization
        axes[2].imshow(explanation["overlayed_image"])
        pred_class = explanation["class_name"]
        confidence = explanation["confidence"]
        axes[2].set_title(f"Prediction: {pred_class}\nConfidence: {confidence:.2%}")
        axes[2].axis("off")

        # Add probability bar chart
        plt.figtext(
            0.5,
            0.05,
            f"Normal: {explanation['probabilities'][0]:.2%} | "
            f"Pneumonia: {explanation['probabilities'][1]:.2%}",
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Explanation saved to {save_path}")

    def compare_predictions(self, image_paths: List[str], save_path: str):
        """Compare predictions and explanations for multiple images."""
        n_images = len(image_paths)
        fig, axes = plt.subplots(n_images, 3, figsize=(15, 5 * n_images))

        if n_images == 1:
            axes = axes.reshape(1, -1)

        for idx, image_path in enumerate(image_paths):
            explanation = self.explain_prediction(image_path)

            # Original
            axes[idx, 0].imshow(explanation["original_image"])
            axes[idx, 0].set_title(f"Image {idx+1}: {Path(image_path).name}")
            axes[idx, 0].axis("off")

            # Heatmap
            if explanation["cam"] is not None:
                axes[idx, 1].imshow(explanation["cam"], cmap="jet")
                axes[idx, 1].set_title("Attention Heatmap")
            axes[idx, 1].axis("off")

            # Overlayed
            axes[idx, 2].imshow(explanation["overlayed_image"])
            pred = explanation["class_name"]
            conf = explanation["confidence"]
            axes[idx, 2].set_title(f"Prediction: {pred} ({conf:.1%})")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Comparison saved to {save_path}")

    def generate_attention_statistics(self, image_paths: List[str]) -> Dict:
        """Generate statistics about attention patterns across images."""
        attention_maps = []
        predictions = []

        for image_path in image_paths:
            explanation = self.explain_prediction(image_path)
            if explanation["cam"] is not None:
                attention_maps.append(explanation["cam"])
                predictions.append(explanation["predicted_class"])

        # Calculate statistics
        attention_maps = np.array(attention_maps)

        stats = {
            "mean_attention": np.mean(attention_maps, axis=0),
            "std_attention": np.std(attention_maps, axis=0),
            "max_attention_per_image": [np.max(am) for am in attention_maps],
            "attention_center_of_mass": [
                self._calculate_center_of_mass(am) for am in attention_maps
            ],
            "predictions": predictions,
        }

        return stats

    def _calculate_center_of_mass(
        self, attention_map: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate center of mass of attention map."""
        h, w = attention_map.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        total_mass = np.sum(attention_map)
        if total_mass == 0:
            return (h / 2, w / 2)

        y_center = np.sum(y_coords * attention_map) / total_mass
        x_center = np.sum(x_coords * attention_map) / total_mass

        return (y_center, x_center)

    def visualize_attention_statistics(self, stats: Dict, save_path: str):
        """Visualize attention statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Mean attention map
        im1 = axes[0, 0].imshow(stats["mean_attention"], cmap="jet")
        axes[0, 0].set_title("Mean Attention Across All Images")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0])

        # Standard deviation
        im2 = axes[0, 1].imshow(stats["std_attention"], cmap="viridis")
        axes[0, 1].set_title("Attention Variability (Std Dev)")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1])

        # Max attention distribution
        axes[1, 0].hist(stats["max_attention_per_image"], bins=20, edgecolor="black")
        axes[1, 0].set_title("Distribution of Maximum Attention")
        axes[1, 0].set_xlabel("Maximum Attention Value")
        axes[1, 0].set_ylabel("Frequency")

        # Center of mass scatter
        centers = np.array(stats["attention_center_of_mass"])
        predictions = np.array(stats["predictions"])

        for pred_class in np.unique(predictions):
            mask = predictions == pred_class
            axes[1, 1].scatter(
                centers[mask, 1],
                centers[mask, 0],
                label=self.class_names[pred_class],
                alpha=0.6,
                s=50,
            )

        axes[1, 1].set_title("Attention Center of Mass by Class")
        axes[1, 1].set_xlabel("X Position")
        axes[1, 1].set_ylabel("Y Position")
        axes[1, 1].legend()
        axes[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Attention statistics visualization saved to {save_path}")


def main():
    """Main function for testing explainability features."""
    import torch

    # Load model (example)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This is just a test - you would load your actual trained model
    model = ModelFactory.create_model("resnet50", num_classes=2, pretrained=True)
    model.to(device)
    model.eval()

    # Create explainer
    explainer = MedicalImageExplainer(model, device)

    logger.info("Explainable AI module initialized successfully!")
    logger.info(
        "To use: provide image paths and call explain_prediction() or batch_explain()"
    )


if __name__ == "__main__":
    main()
