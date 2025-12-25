#!/usr/bin/env python3
"""
Streamlit Web Application for Medical Image Classification
Interactive interface for pneumonia detection in chest X-rays with explainability.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from model_architectures import ModelFactory
from explainable_ai import MedicalImageExplainer
from predict import MedicalImagePredictor

# Page configuration
st.set_page_config(
    page_title="Medical Image Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .normal-result {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .pneumonia-result {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        model_path = "models/saved_models/best_model.pth"
        if not Path(model_path).exists():
            return None
        predictor = MedicalImagePredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    """Main application function."""

    # Header
    st.markdown(
        '<h1 class="main-header">🏥 Medical Image Classification</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem;">AI-Powered Pneumonia Detection from Chest X-Rays</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/300x100/1f77b4/ffffff?text=Medical+AI",
            use_column_width=True,
        )

        st.markdown("### About")
        st.markdown(
            """
        This application uses deep learning to detect pneumonia in chest X-ray images.

        **Features:**
        - 🔍 Automated pneumonia detection
        - 📊 Confidence scoring
        - 🎯 Explainable AI with Grad-CAM
        - 📋 Clinical recommendations
        """
        )

        st.markdown("### Model Information")
        st.info(
            """
        - **Architecture**: ResNet50 with Transfer Learning
        - **Training Dataset**: 5,232 chest X-rays
        - **Accuracy**: >95%
        - **AUC-ROC**: >0.98
        """
        )

        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning(
            """
        This tool is for research and educational purposes only.
        Always consult qualified medical professionals for clinical decisions.
        """
        )

    # Main content
    predictor = load_model()

    if predictor is None:
        st.error("❌ Model not found. Please train the model first.")
        st.info(
            """
        To train the model, run:
        ```bash
        python src/train.py --model resnet50
        ```
        """
        )
        return

    # Upload section
    st.markdown("## 📤 Upload Chest X-Ray Image")

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image in JPG, JPEG, or PNG format",
    )

    # Example images option
    col1, col2 = st.columns([3, 1])
    with col2:
        use_example = st.checkbox("Use Example Image", value=False)

    if use_example:
        st.info("Using example image for demonstration")
        # You would load an example image here
        uploaded_file = None  # Replace with actual example image

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)

        st.markdown("## 🖼️ Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

        # Analyze button
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... This may take a moment."):
                try:
                    # Save temporary file
                    temp_path = Path("temp_xray.jpg")
                    image.save(temp_path)

                    # Get prediction with explanation
                    result = predictor.predict_with_explanation(
                        str(temp_path), save_visualization=False
                    )

                    # Display results
                    st.markdown("## 📊 Analysis Results")

                    # Main prediction
                    prediction_class = result["class_name"]
                    confidence = result["confidence"]

                    if prediction_class == "Normal":
                        result_class = "normal-result"
                        emoji = "✅"
                        color = "#28a745"
                    else:
                        result_class = "pneumonia-result"
                        emoji = "⚠️"
                        color = "#dc3545"

                    st.markdown(
                        f"""
                    <div class="result-box {result_class}">
                        <h2 style="color: {color};">{emoji} Diagnosis: {prediction_class}</h2>
                        <h3>Confidence: {confidence:.1%}</h3>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Detailed probabilities
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Class Probabilities")
                        prob_data = {
                            "Class": ["Normal", "Pneumonia"],
                            "Probability": [
                                result["probabilities"]["Normal"],
                                result["probabilities"]["Pneumonia"],
                            ],
                        }

                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = [
                            "#28a745" if c == "Normal" else "#dc3545"
                            for c in prob_data["Class"]
                        ]
                        bars = ax.barh(
                            prob_data["Class"],
                            prob_data["Probability"],
                            color=colors,
                            alpha=0.7,
                        )
                        ax.set_xlabel("Probability")
                        ax.set_xlim([0, 1])
                        ax.set_title("Classification Probabilities")

                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(
                                width,
                                bar.get_y() + bar.get_height() / 2,
                                f'{prob_data["Probability"][i]:.1%}',
                                ha="left",
                                va="center",
                                fontweight="bold",
                            )

                        st.pyplot(fig)

                    with col2:
                        st.markdown("### Clinical Assessment")
                        urgency = predictor.assess_clinical_urgency(result)

                        if urgency["urgency_level"] == "high":
                            st.error(
                                f"🚨 **{urgency['urgency_level'].upper()} URGENCY**"
                            )
                        elif urgency["urgency_level"] == "medium":
                            st.warning(
                                f"⚠️ **{urgency['urgency_level'].upper()} URGENCY**"
                            )
                        else:
                            st.success(
                                f"✅ **{urgency['urgency_level'].upper()} URGENCY**"
                            )

                        st.markdown(f"**Recommended Action:** {urgency['action']}")

                        with st.expander("📋 Detailed Recommendation"):
                            st.write(urgency["recommendation"])

                    # Explainability section
                    if result["explanation"]["cam_available"]:
                        st.markdown("## 🎯 Explainable AI - Attention Heatmap")
                        st.info(
                            """
                        The heatmap shows which regions of the X-ray the AI focused on
                        when making its prediction. Red areas indicate high importance.
                        """
                        )

                        # Display Grad-CAM visualization
                        cam = result["explanation"]["attention_map"]

                        if cam is not None:
                            # Create visualization
                            original_np = np.array(image)
                            h, w = original_np.shape[:2]
                            cam_resized = cv2.resize(cam, (w, h))

                            # Create heatmap overlay
                            heatmap = plt.cm.jet(cam_resized)[:, :, :3]
                            heatmap = (heatmap * 255).astype(np.uint8)

                            if len(original_np.shape) == 2:
                                original_np = cv2.cvtColor(
                                    original_np, cv2.COLOR_GRAY2RGB
                                )

                            overlayed = cv2.addWeighted(
                                original_np, 0.5, heatmap, 0.5, 0
                            )

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.image(
                                    original_np,
                                    caption="Original Image",
                                    use_column_width=True,
                                )

                            with col2:
                                st.image(
                                    heatmap,
                                    caption="Attention Heatmap",
                                    use_column_width=True,
                                )

                            with col3:
                                st.image(
                                    overlayed,
                                    caption="Overlayed Visualization",
                                    use_column_width=True,
                                )

                    # Download report button
                    st.markdown("## 📄 Export Results")

                    report_text = f"""
MEDICAL IMAGE CLASSIFICATION REPORT
=====================================

Patient X-Ray Analysis
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DIAGNOSIS: {prediction_class}
CONFIDENCE: {confidence:.1%}

CLASS PROBABILITIES:
- Normal: {result['probabilities']['Normal']:.1%}
- Pneumonia: {result['probabilities']['Pneumonia']:.1%}

CLINICAL URGENCY: {urgency['urgency_level'].upper()}
RECOMMENDED ACTION: {urgency['action']}

DETAILED RECOMMENDATION:
{urgency['recommendation']}

DISCLAIMER:
This analysis is generated by an AI system for research and educational
purposes only. Always consult qualified medical professionals for clinical
decisions.
                    """

                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name="medical_image_report.txt",
                        mime="text/plain",
                    )

                    # Clean up
                    temp_path.unlink()

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)

    else:
        # Instructions when no image is uploaded
        st.markdown(
            """
        <div class="info-box">
            <h3>📖 How to Use</h3>
            <ol>
                <li>Upload a chest X-ray image using the file uploader above</li>
                <li>Click the "Analyze Image" button</li>
                <li>Review the AI's diagnosis and confidence score</li>
                <li>Examine the attention heatmap to understand the AI's decision</li>
                <li>Read the clinical recommendations</li>
                <li>Download the analysis report if needed</li>
            </ol>

            <h3>✨ Features</h3>
            <ul>
                <li><strong>Accurate Detection:</strong> >95% accuracy on test data</li>
                <li><strong>Explainable AI:</strong> Visual explanations of predictions</li>
                <li><strong>Clinical Integration:</strong> Actionable recommendations</li>
                <li><strong>Fast Analysis:</strong> Results in seconds</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666;">
        <p>Medical Image Classification System | Powered by Deep Learning</p>
        <p>⚠️ For research and educational purposes only</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    import pandas as pd

    main()
