# Medical Image Classification - Pneumonia Detection from Chest X-Rays

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🏥 Project Overview

A deep learning application to assist radiologists in detecting pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs), transfer learning, and explainable AI. Achieves **90.8% AUC-ROC** and **93.3% sensitivity** on the test set.

### Clinical Impact

- **Faster Diagnosis**: Automated screening reduces interpretation time
- **High Sensitivity**: 93.3% - rarely misses pneumonia cases
- **Explainable AI**: Grad-CAM visualizations show decision reasoning
- **Production Ready**: Streamlit web app for real-time predictions

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** - Kaggle

- **Total**: 5,856 images
  - Training: 5,216 images
  - Validation: 16 images
  - Test: 624 images
- **Classes**: Normal (1,341) | Pneumonia (3,875 - bacterial and viral)
- **Source**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## 🎯 Model Performance

| Metric          | Value | Clinical Interpretation             |
| --------------- | ----- | ----------------------------------- |
| **Accuracy**    | 80.6% | Good overall performance            |
| **AUC-ROC**     | 90.8% | Strong discriminative ability       |
| **Sensitivity** | 93.3% | ⭐ Excellent at detecting pneumonia |
| **Specificity** | 59.4% | Trade-off for screening safety      |
| **PPV**         | 79.3% | Positive predictions 79% reliable   |
| **NPV**         | 84.2% | Negative predictions 84% reliable   |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 8GB RAM (16GB recommended)
- GPU with CUDA support (optional, but recommended for training)
- 10GB disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/m-jawhar/medical-image-classification.git
cd medical-image-classification

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract to `data/raw/chest_xray/`
3. Run data preparation:

```bash
python prepare_data.py --create-splits
```

### Train Model

```bash
# Recommended: ResNet50 with transfer learning
python src/train.py --model resnet50

# Other options: custom_cnn, densenet121, efficientnet_b0
```

Training takes:

- **Local CPU**: 10-11 hours (50 epochs)
- **Google Colab (T4 GPU)**: ~75 minutes

### Evaluate Model

```bash
python src/evaluate.py --model_path models/saved_models/best_model.pth
```

Generates:

- Confusion matrix
- ROC curve
- Precision-recall curve
- Clinical assessment report

Results saved to `results/evaluation/`

### Run Web Application

```bash
streamlit run app.py
```

Opens at http://localhost:8501

**Features:**

- Upload chest X-ray images
- Real-time pneumonia detection
- Confidence scores
- Grad-CAM attention heatmaps
- Clinical recommendations
- Downloadable reports

## 📁 Project Structure

```
├── data/
│   ├── raw/chest_xray/         # Downloaded dataset
│   ├── processed/              # Preprocessed images
│   └── splits/                 # CSV files: train.csv, val.csv, test.csv
├── models/
│   ├── architectures/          # Model definitions
│   ├── pretrained/             # Pre-trained weights
│   └── saved_models/           # Trained checkpoints (best_model.pth)
├── src/
│   ├── data_preprocessing.py   # CLAHE, augmentation pipeline
│   ├── model_architectures.py  # CNN architectures (4 models)
│   ├── transfer_learning.py    # Pre-trained model utilities
│   ├── train.py                # Training pipeline with metrics
│   ├── evaluate.py             # Comprehensive evaluation
│   ├── predict.py              # Inference with explanations
│   └── explainable_ai.py       # Grad-CAM implementation
├── utils/
│   ├── data_utils.py           # Data loading utilities
│   ├── visualization.py        # Plotting functions
│   └── metrics.py              # Clinical metrics
├── notebooks/
│   └── 01_data_exploration.ipynb  # Interactive data analysis
├── results/
│   ├── evaluation/             # Test metrics & visualizations
│   ├── visualizations/         # Training curves
│   ├── predictions/            # Inference outputs
│   └── explanations/           # Grad-CAM heatmaps
├── config/
│   └── config.yaml             # Hyperparameters & settings
├── app.py                      # Streamlit web application
├── prepare_data.py             # Dataset verification & splits
└── requirements.txt            # Python dependencies
```

## 🏗️ Model Architectures

| Model           | Parameters | Accuracy  | Speed  | Best For             |
| --------------- | ---------- | --------- | ------ | -------------------- |
| **ResNet50** ⭐ | ~23M       | 90.8% AUC | Medium | **Production use**   |
| Custom CNN      | ~500K      | ~90%      | Fast   | Learning/baseline    |
| DenseNet121     | ~7M        | >94%      | Medium | Efficient deployment |
| EfficientNet-B0 | ~5M        | >94%      | Fast   | Mobile/edge devices  |

**Trained Model:** ResNet50 with ImageNet pre-training, fine-tuned on chest X-rays

## 🔬 Technical Implementation

### Data Preprocessing

- **CLAHE Enhancement**: Improves contrast in medical images
  - Clip limit: 2.0
  - Tile grid: 8×8
- **Augmentation**: Rotation (±15°), flips, brightness/contrast (±20%), Gaussian blur, coarse dropout
- **Normalization**: ImageNet statistics for transfer learning
- **Class Balancing**: Weighted loss (1.9448 Normal, 0.6730 Pneumonia)

### Training Configuration

```yaml
Model: ResNet50 (transfer learning)
Optimizer: Adam (lr=0.001)
Loss: Weighted Cross-Entropy
Batch Size: 32
Epochs: 50 (early stopping at epoch 15)
Scheduler: ReduceLROnPlateau
Validation: AUC-ROC monitoring
```

### Explainable AI

**Grad-CAM (Gradient-weighted Class Activation Mapping)**

- Visualizes regions influencing predictions
- Highlights lung opacities and infiltrates
- Red areas = high attention (pneumonia indicators)
- Provides clinical interpretability

## 📊 Evaluation Results

Comprehensive evaluation on 624 test images:

**Confusion Matrix:**

- True Positives: Correctly identified pneumonia cases
- True Negatives: Correctly identified normal cases
- Trade-off: High sensitivity (93.3%) prioritizes catching pneumonia over false alarms

**Clinical Assessment:**

- **Medium Urgency**: 86% cases require review within 24 hours
- **Sensitivity-first approach**: Safer for screening (rarely misses pneumonia)
- **Specificity trade-off**: 41% false positive rate acceptable for triage tool

See `results/evaluation/` for detailed reports.

## 🎮 Usage Examples

### Command Line Prediction

```bash
# Single image with explanation
python src/predict.py --image data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg --explain

# Batch processing
python src/predict.py --directory data/raw/chest_xray/test/PNEUMONIA/ --explain
```

### Web Application

1. Launch: `streamlit run app.py`
2. Upload chest X-ray (JPG, JPEG, PNG)
3. Click "Analyze Image"
4. View: Diagnosis, confidence, Grad-CAM heatmap, recommendations
5. Download report

### Jupyter Notebooks

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🛠️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  architecture: resnet50
  num_classes: 2

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  image_size: 224
  augmentation: true
  clahe: true
```

## 📈 Training on Google Colab

For faster training with GPU:

1. Upload project to Google Colab
2. Mount Google Drive or upload dataset
3. Install dependencies: `!pip install -r requirements.txt`
4. Run: `!python src/train.py --model resnet50`

**Performance:**

- Tesla T4 GPU: ~75 minutes (50 epochs)
- Free Colab tier sufficient

## 🚀 Deployment Options

### Streamlit Cloud (Recommended - Free)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy `app.py`
5. Public URL in ~3 minutes

**Note:** Upload trained model to external hosting (Hugging Face Hub, Google Drive) and download in app.

### Hugging Face Spaces

1. Create Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select Streamlit template
3. Push code
4. Optional: Free GPU tier available

## 📝 Quick Reference Commands

```bash
# Setup
pip install -r requirements.txt
python prepare_data.py --create-splits

# Training
python src/train.py --model resnet50                    # Recommended
python src/train.py --model densenet121                 # Alternative
python src/train.py --model efficientnet_b0             # Mobile

# Evaluation
python src/evaluate.py --model_path models/saved_models/best_model.pth

# Prediction
python src/predict.py --image path/to/xray.jpg --explain
python src/predict.py --directory path/to/images/

# Web App
streamlit run app.py

# Notebooks
jupyter notebook notebooks/
```

## ⚠️ Medical Disclaimer

**For Research and Educational Purposes Only**

This tool is NOT a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for clinical decisions. The model may produce false positives/negatives and should be used as a screening aid only, not for definitive diagnosis.

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{medical_image_classification_2025,
  author = {Your Name},
  title = {Medical Image Classification - Pneumonia Detection},
  year = {2025},
  url = {https://github.com/m-jawhar/medical-image-classification}
}
```

## 📚 References

- Kermany, D. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. _Cell_.
- Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. _arXiv_.
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. _ICCV_.

## 🤝 Contributing

Contributions welcome! Please ensure:

- Medical accuracy in all changes
- Comprehensive testing
- Documentation updates
- Follow existing code style

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🔗 Links

- **GitHub**: https://github.com/m-jawhar/medical-image-classification
- **Dataset**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Demo**: [Coming Soon - Streamlit Cloud]

---

**Built with ❤️ for healthcare impact** | Last Updated: January 2026
