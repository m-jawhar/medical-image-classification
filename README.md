# Medical Image Classification - Pneumonia Detection

## 🏥 Project Overview

A deep learning application to assist radiologists in detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning techniques.

## 🎯 Objectives

- **High Impact Healthcare**: Assist radiologists in pneumonia detection
- **CNN Implementation**: Custom and pre-trained models
- **Transfer Learning**: Leverage models trained on ImageNet
- **Explainable AI**: Provide interpretable results using Grad-CAM
- **Real-world Application**: Production-ready model deployment

## 📊 Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset:

- **Normal**: 1,349 images
- **Pneumonia**: 3,883 images (bacterial and viral)
- **Total**: 5,232 chest X-ray images
- **Source**: Kaggle Chest X-Ray Images Dataset

## 🏗️ Project Structure

```
├── data/
│   ├── raw/                    # Raw dataset
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/Val/Test splits
├── models/
│   ├── architectures/          # CNN model definitions
│   ├── pretrained/            # Transfer learning models
│   └── saved_models/          # Trained model weights
├── src/
│   ├── data_preprocessing.py   # Image preprocessing pipeline
│   ├── model_architectures.py  # CNN architectures
│   ├── transfer_learning.py    # Pre-trained models
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   ├── predict.py             # Inference pipeline
│   └── explainable_ai.py      # Grad-CAM visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_explainable_ai.ipynb
├── utils/
│   ├── data_utils.py          # Data handling utilities
│   ├── visualization.py       # Plotting functions
│   └── metrics.py             # Custom metrics
├── config/
│   └── config.yaml            # Configuration parameters
├── requirements.txt           # Dependencies
└── app.py                     # Streamlit web application
```

## 🚀 Key Features

### 1. **Advanced Image Preprocessing**

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Image normalization and augmentation
- Lung segmentation preprocessing
- Multi-scale image processing

### 2. **Multiple CNN Architectures**

- Custom CNN from scratch
- ResNet50/101 transfer learning
- DenseNet121 transfer learning
- EfficientNet transfer learning
- Ensemble methods

### 3. **Explainable AI**

- Grad-CAM visualizations
- Feature map analysis
- Attention mechanisms
- Clinical interpretation support

### 4. **Comprehensive Evaluation**

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Confusion matrices
- Cross-validation
- Clinical metrics (Sensitivity/Specificity)

## 🛠️ Installation & Setup

```bash
# Clone and navigate to project
cd "Medical Image Classification"

# Install dependencies
pip install -r requirements.txt

# Download dataset (instructions in notebooks)
# Run data preprocessing
python src/data_preprocessing.py
```

## 🏃 Quick Start

1. **Data Exploration**: `notebooks/01_data_exploration.ipynb`
2. **Preprocessing**: `notebooks/02_preprocessing.ipynb`
3. **Training**: `notebooks/03_model_training.ipynb`
4. **Evaluation**: `notebooks/04_evaluation.ipynb`
5. **Explainable AI**: `notebooks/05_explainable_ai.ipynb`

Or run the complete pipeline:

```bash
# Train models
python src/train.py --model resnet50

# Evaluate model
python src/evaluate.py --model_path models/saved_models/best_model.pth

# Launch web app
streamlit run app.py
```

## 📈 Expected Results

- **Accuracy**: >95% on test set
- **Sensitivity**: >96% (crucial for medical applications)
- **Specificity**: >94%
- **AUC-ROC**: >0.98

## 🔬 Clinical Impact

- **Faster Diagnosis**: Reduce interpretation time
- **Second Opinion**: Support radiologist decisions
- **Resource Optimization**: Prioritize critical cases
- **Accessibility**: Aid in resource-limited settings

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines and ensure medical accuracy in any changes.

## 📄 License

MIT License - See LICENSE file for details.

## 🏥 Medical Disclaimer

This tool is for research and educational purposes. Always consult qualified medical professionals for clinical decisions.

## 📚 References

- Kermany, D. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning
- Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
- WHO Pneumonia Guidelines and Clinical Standards
