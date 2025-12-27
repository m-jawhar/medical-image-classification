#!/usr/bin/env python3
"""
Prediction Pipeline for Medical Image Classification
Inference script for making predictions on new chest X-ray images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import json
import cv2

# Import custom modules
from model_architectures import ModelFactory
from explainable_ai import MedicalImageExplainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalImagePredictor:
    """
    Predictor for medical image classification with explainability support.
    """

    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """Initialize predictor."""
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model = self._load_model(model_path)
        self.class_names = ["Normal", "Pneumonia"]
        self.explainer = MedicalImageExplainer(self.model, self.device)

        # Setup preprocessing
        self.transform = self._setup_transforms()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {"data": {"image_size": [224, 224]}}

    def _get_device(self) -> torch.device:
        """Get appropriate device for inference."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Get model configuration
        model_config = checkpoint.get("config", self.config)

        # Create model
        model = ModelFactory.create_model(
            architecture=model_config["model"]["architecture"],
            num_classes=model_config["model"]["num_classes"],
            pretrained=False,
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.info(f"Model loaded successfully from {model_path}")
        return model

    def _setup_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms."""
        image_size = tuple(self.config["data"]["image_size"])

        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess single image for inference."""
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict_single(
        self, image_path: str, return_probabilities: bool = True
    ) -> Dict:
        """Make prediction on a single image."""
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # Prepare result
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "class_name": self.class_names[predicted_class],
            "confidence": confidence,
        }

        if return_probabilities:
            result["probabilities"] = {
                "Normal": probabilities[0][0].item(),
                "Pneumonia": probabilities[0][1].item(),
            }

        return result

    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """Make predictions on multiple images."""
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            for image_path in batch_paths:
                try:
                    result = self.predict_single(image_path)
                    results.append(result)
                    logger.info(
                        f"Processed: {Path(image_path).name} - "
                        f"{result['class_name']} ({result['confidence']:.2%})"
                    )
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    results.append({"image_path": image_path, "error": str(e)})

        return results

    def predict_with_explanation(
        self,
        image_path: str,
        save_visualization: bool = True,
        output_dir: str = "results/predictions",
    ) -> Dict:
        """Make prediction with explainability visualization."""
        # Get basic prediction
        prediction = self.predict_single(image_path)

        # Get explanation
        explanation = self.explainer.explain_prediction(image_path)

        # Combine results
        result = {
            **prediction,
            "explanation": {
                "cam_available": explanation["cam"] is not None,
                "attention_map": explanation["cam"],
            },
        }

        # Save visualization if requested
        if save_visualization:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            filename = Path(image_path).stem
            viz_path = output_path / f"{filename}_prediction.png"

            self.explainer.save_explanation(
                explanation, viz_path, Path(image_path).name
            )

            result["visualization_path"] = str(viz_path)

        return result

    def predict_directory(
        self,
        directory: str,
        pattern: str = "*.jpeg",
        output_file: str = "predictions.json",
    ) -> List[Dict]:
        """Make predictions on all images in a directory."""
        # Find all images
        dir_path = Path(directory)
        image_paths = list(dir_path.glob(pattern))

        if not image_paths:
            logger.warning(f"No images found in {directory} with pattern {pattern}")
            return []

        logger.info(f"Found {len(image_paths)} images in {directory}")

        # Make predictions
        results = self.predict_batch([str(p) for p in image_paths])

        # Save results to JSON
        output_path = dir_path / output_file
        with open(output_path, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = []
            for r in results:
                serializable_result = {}
                for key, value in r.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_result[key] = value.item()
                    elif isinstance(value, dict):
                        serializable_result[key] = {
                            k: (
                                v.item()
                                if isinstance(v, (np.integer, np.floating))
                                else v
                            )
                            for k, v in value.items()
                        }
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Predictions saved to {output_path}")

        return results

    def get_prediction_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from prediction results."""
        successful_predictions = [r for r in results if "error" not in r]

        if not successful_predictions:
            return {"error": "No successful predictions"}

        # Count predictions by class
        class_counts = {
            "Normal": sum(
                1 for r in successful_predictions if r["predicted_class"] == 0
            ),
            "Pneumonia": sum(
                1 for r in successful_predictions if r["predicted_class"] == 1
            ),
        }

        # Calculate average confidence
        avg_confidence = np.mean([r["confidence"] for r in successful_predictions])

        # Find high and low confidence predictions
        sorted_by_confidence = sorted(
            successful_predictions, key=lambda x: x["confidence"], reverse=True
        )

        summary = {
            "total_images": len(results),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(results) - len(successful_predictions),
            "class_distribution": class_counts,
            "average_confidence": float(avg_confidence),
            "highest_confidence": {
                "image": Path(sorted_by_confidence[0]["image_path"]).name,
                "class": sorted_by_confidence[0]["class_name"],
                "confidence": sorted_by_confidence[0]["confidence"],
            },
            "lowest_confidence": {
                "image": Path(sorted_by_confidence[-1]["image_path"]).name,
                "class": sorted_by_confidence[-1]["class_name"],
                "confidence": sorted_by_confidence[-1]["confidence"],
            },
        }

        return summary

    def assess_clinical_urgency(self, prediction: Dict) -> Dict:
        """Assess clinical urgency based on prediction."""
        urgency_levels = {
            "high": {"color": "red", "action": "Immediate review recommended"},
            "medium": {"color": "orange", "action": "Review within 24 hours"},
            "low": {"color": "green", "action": "Routine follow-up"},
        }

        # Determine urgency
        if prediction["class_name"] == "Pneumonia":
            if prediction["confidence"] >= 0.9:
                urgency = "high"
            elif prediction["confidence"] >= 0.7:
                urgency = "medium"
            else:
                urgency = "low"
        else:  # Normal
            if prediction["confidence"] >= 0.95:
                urgency = "low"
            else:
                urgency = "medium"  # Low confidence normal should be reviewed

        return {
            "urgency_level": urgency,
            **urgency_levels[urgency],
            "recommendation": self._generate_recommendation(prediction, urgency),
        }

    def _generate_recommendation(self, prediction: Dict, urgency: str) -> str:
        """Generate clinical recommendation based on prediction."""
        if prediction["class_name"] == "Pneumonia":
            if urgency == "high":
                return (
                    f"High confidence ({prediction['confidence']:.1%}) pneumonia detection. "
                    "Recommend immediate clinical assessment, consider antibiotic therapy, "
                    "and follow-up imaging."
                )
            elif urgency == "medium":
                return (
                    f"Moderate confidence ({prediction['confidence']:.1%}) pneumonia detection. "
                    "Recommend clinical correlation with symptoms and patient history. "
                    "Consider additional imaging if clinically indicated."
                )
            else:
                return (
                    f"Low confidence ({prediction['confidence']:.1%}) pneumonia indication. "
                    "Recommend clinical evaluation and correlation with symptoms. "
                    "May warrant additional imaging modalities."
                )
        else:  # Normal
            if prediction["confidence"] >= 0.95:
                return (
                    f"High confidence ({prediction['confidence']:.1%}) normal chest X-ray. "
                    "Routine follow-up as clinically appropriate."
                )
            else:
                return (
                    f"Moderate confidence ({prediction['confidence']:.1%}) normal finding. "
                    "Consider clinical correlation and repeat imaging if symptoms persist."
                )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Predict on Medical Images")
    parser.add_argument(
        "--model",
        type=str,
        default="models/saved_models/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--directory", type=str, help="Path to directory of images")
    parser.add_argument(
        "--explain", action="store_true", help="Include explainability visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = MedicalImagePredictor(args.model, args.config)

    # Make predictions
    if args.image:
        # Single image prediction
        if args.explain:
            result = predictor.predict_with_explanation(args.image)
        else:
            result = predictor.predict_single(args.image)

        # Display result
        logger.info("\n" + "=" * 50)
        logger.info("PREDICTION RESULT")
        logger.info("=" * 50)
        logger.info(f"Image: {Path(args.image).name}")
        logger.info(f"Prediction: {result['class_name']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")

        if "probabilities" in result:
            logger.info("\nClass Probabilities:")
            for class_name, prob in result["probabilities"].items():
                logger.info(f"  {class_name}: {prob:.2%}")

        # Clinical assessment
        urgency = predictor.assess_clinical_urgency(result)
        logger.info(f"\nClinical Urgency: {urgency['urgency_level'].upper()}")
        logger.info(f"Recommendation: {urgency['recommendation']}")
        logger.info("=" * 50)

    elif args.directory:
        # Directory prediction
        results = predictor.predict_directory(args.directory)

        # Generate and display summary
        summary = predictor.get_prediction_summary(results)

        logger.info("\n" + "=" * 50)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Successful: {summary['successful_predictions']}")
        logger.info(f"Failed: {summary['failed_predictions']}")
        logger.info(f"\nClass Distribution:")
        for class_name, count in summary["class_distribution"].items():
            logger.info(f"  {class_name}: {count}")
        logger.info(f"\nAverage Confidence: {summary['average_confidence']:.2%}")
        logger.info("=" * 50)

    else:
        logger.error("Please provide either --image or --directory argument")
        parser.print_help()


if __name__ == "__main__":
    main()
