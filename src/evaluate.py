#!/usr/bin/env python3
"""
Evaluation Pipeline for Medical Image Classification
Comprehensive evaluation including medical metrics, ROC curves,
confusion matrices, and clinical interpretation support.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
import json

# Import custom modules
from model_architectures import ModelFactory
from train import MedicalImageDataset
from torchvision import transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalEvaluator:
    """Comprehensive evaluator for medical image classification models."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.class_names = ["Normal", "Pneumonia"]

        # Create results directory
        Path("results/evaluation").mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _get_device(self) -> torch.device:
        """Get appropriate device for evaluation."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Get model configuration from checkpoint
        model_config = checkpoint.get("config", self.config)

        # Create model
        model = ModelFactory.create_model(
            architecture=model_config["model"]["architecture"],
            num_classes=model_config["model"]["num_classes"],
            pretrained=False,  # We're loading trained weights
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Best validation AUC: {checkpoint.get('val_auc', 'unknown')}")

        return model

    def create_test_loader(self) -> DataLoader:
        """Create test data loader."""
        transform = transforms.Compose(
            [
                transforms.Resize(self.config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        test_dataset = MedicalImageDataset(
            "data/splits/test.csv", transform=transform, mode="test"
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
        )

        logger.info(f"Test dataset size: {len(test_dataset)}")
        return test_loader

    def get_predictions(
        self, model: nn.Module, test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on test set."""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Getting predictions"):
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return (
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities),
        )

    def calculate_comprehensive_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive medical metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Overall metrics
        precision_macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[0]
        recall_macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[1]
        f1_macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[2]

        # AUC-ROC
        auc_roc = roc_auc_score(y_true, y_prob[:, 1])

        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Medical-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

        # Additional clinical metrics
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Likelihood ratios
        positive_lr = (
            sensitivity / false_positive_rate
            if false_positive_rate > 0
            else float("inf")
        )
        negative_lr = false_negative_rate / specificity if specificity > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision_normal": precision[0],
            "precision_pneumonia": precision[1],
            "precision_macro": precision_macro,
            "recall_normal": recall[0],
            "recall_pneumonia": recall[1],
            "recall_macro": recall_macro,
            "f1_normal": f1[0],
            "f1_pneumonia": f1[1],
            "f1_macro": f1_macro,
            "auc_roc": auc_roc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "positive_likelihood_ratio": positive_lr,
            "negative_likelihood_ratio": negative_lr,
            "support_normal": support[0],
            "support_pneumonia": support[1],
        }

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Add percentage annotations
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm_percent[i, j]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="blue", linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")

        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(
            fpr[optimal_idx],
            tpr[optimal_idx],
            "ro",
            markersize=8,
            label=f"Optimal Threshold = {optimal_threshold:.3f}",
        )
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"ROC curve saved to {save_path}")
        return optimal_threshold

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str
    ):
        """Plot and save precision-recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, color="blue", linewidth=2, label="Precision-Recall Curve"
        )

        # Add no-skill line
        no_skill = len(y_true[y_true == 1]) / len(y_true)
        plt.plot(
            [0, 1],
            [no_skill, no_skill],
            color="red",
            linestyle="--",
            label=f"No Skill (AP = {no_skill:.3f})",
        )

        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision (PPV)")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Precision-Recall curve saved to {save_path}")

    def plot_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str
    ):
        """Plot and save calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob[:, 1], n_bins=10
        )

        plt.figure(figsize=(8, 6))
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            color="blue",
            linewidth=2,
            label="Model",
        )
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot (Reliability Curve)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Calibration curve saved to {save_path}")

    def create_clinical_report(self, metrics: Dict[str, float], save_path: str):
        """Create clinical interpretation report."""
        report = (
            f"""
# Medical Image Classification - Clinical Evaluation Report

## Model Performance Summary

### Overall Performance
- **Accuracy**: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)
- **AUC-ROC**: {metrics['auc_roc']:.3f}

### Clinical Metrics
- **Sensitivity (True Positive Rate)**: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)
  - Ability to correctly identify pneumonia cases
- **Specificity (True Negative Rate)**: {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%)
  - Ability to correctly identify normal cases

### Predictive Values
- **Positive Predictive Value (PPV)**: {metrics['ppv']:.3f} ({metrics['ppv']*100:.1f}%)
  - Probability that a positive test indicates pneumonia
- **Negative Predictive Value (NPV)**: {metrics['npv']:.3f} ({metrics['npv']*100:.1f}%)
  - Probability that a negative test indicates normal

### Error Rates
- **False Positive Rate**: {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}%)
  - Rate of incorrectly diagnosing pneumonia
- **False Negative Rate**: {metrics['false_negative_rate']:.3f} ({metrics['false_negative_rate']*100:.1f}%)
  - Rate of missing pneumonia cases

### Likelihood Ratios
- **Positive Likelihood Ratio**: {metrics['positive_likelihood_ratio']:.2f}
  - How much more likely a positive test is in pneumonia vs normal
- **Negative Likelihood Ratio**: {metrics['negative_likelihood_ratio']:.3f}
  - How much more likely a negative test is in normal vs pneumonia

## Clinical Interpretation

### Strengths
"""
            + (
                f"- High sensitivity ({metrics['sensitivity']:.1%}) indicates excellent ability to detect pneumonia cases\n"
                if metrics["sensitivity"] >= 0.95
                else ""
            )
            + (
                f"- High specificity ({metrics['specificity']:.1%}) indicates excellent ability to identify normal cases\n"
                if metrics["specificity"] >= 0.95
                else ""
            )
            + f"- AUC-ROC of {metrics['auc_roc']:.3f} indicates {'excellent' if metrics['auc_roc'] >= 0.9 else 'good' if metrics['auc_roc'] >= 0.8 else 'fair'} discriminative ability\n"
        )

        if metrics["sensitivity"] < 0.9:
            report += f"\n### Areas for Improvement\n- Sensitivity ({metrics['sensitivity']:.1%}) could be improved to reduce missed pneumonia cases\n"

        if metrics["specificity"] < 0.9:
            report += f"- Specificity ({metrics['specificity']:.1%}) could be improved to reduce false alarms\n"

        report += f"""

## Recommendations

### Clinical Use
{'- Model shows excellent performance for pneumonia screening' if metrics['auc_roc'] >= 0.95 else '- Model shows good performance but should be used as a diagnostic aid, not replacement'}
- Always combine AI results with clinical judgment
- Consider patient history and symptoms in final diagnosis

### Quality Assurance
- Regular model performance monitoring recommended
- Periodic retraining with new data
- Validation on diverse patient populations

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(save_path, "w") as f:
            f.write(report)

        logger.info(f"Clinical report saved to {save_path}")

    def evaluate_model(self, model_path: str) -> Dict[str, any]:
        """Comprehensive model evaluation."""
        logger.info("Starting comprehensive model evaluation...")

        # Load model and data
        model = self.load_model(model_path)
        test_loader = self.create_test_loader()

        # Get predictions
        y_true, y_pred, y_prob = self.get_predictions(model, test_loader)

        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_prob)

        # Create visualizations
        viz_dir = Path("results/evaluation")
        self.plot_confusion_matrix(y_true, y_pred, viz_dir / "confusion_matrix.png")
        optimal_threshold = self.plot_roc_curve(
            y_true, y_prob, viz_dir / "roc_curve.png"
        )
        self.plot_precision_recall_curve(
            y_true, y_prob, viz_dir / "precision_recall_curve.png"
        )
        self.plot_calibration_curve(y_true, y_prob, viz_dir / "calibration_curve.png")

        # Generate reports
        self.create_clinical_report(metrics, viz_dir / "clinical_report.md")

        # Save detailed results
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            elif hasattr(obj, "tolist"):  # numpy array
                return obj.tolist()
            else:
                return obj

        detailed_results = {
            "metrics": convert_to_native(metrics),
            "optimal_threshold": float(optimal_threshold),
            "predictions": {
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist(),
            },
        }

        with open(viz_dir / "evaluation_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.3f}")
        logger.info(f"Specificity: {metrics['specificity']:.3f}")
        logger.info(f"PPV: {metrics['ppv']:.3f}")
        logger.info(f"NPV: {metrics['npv']:.3f}")
        logger.info("=" * 50)

        return detailed_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Medical Image Classification Model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/saved_models/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MedicalEvaluator(args.config)

    # Evaluate model
    results = evaluator.evaluate_model(args.model_path)

    logger.info("Evaluation completed successfully!")
    logger.info(
        "Check 'results/evaluation/' directory for detailed results and visualizations."
    )


if __name__ == "__main__":
    main()
