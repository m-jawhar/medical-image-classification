# Medical Image Classification - Getting Started Guide

Welcome to the Medical Image Classification project for pneumonia detection! This guide will help you set up and run the project.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- At least 8GB RAM
- 10GB free disk space

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Navigate to project directory
cd "d:\projects\Computer Vision\Medical Image Classification"

# Install required packages
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle:

- **URL**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Size**: ~2GB

Extract the dataset to:

```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Explore the Data

Open and run the data exploration notebook:

```powershell
jupyter notebook "notebooks/01_data_exploration.ipynb"
```

### 4. Preprocess the Data

Run the preprocessing script:

```powershell
python src/data_preprocessing.py
```

This will:

- Create balanced train/validation/test splits
- Save split information to `data/splits/`
- Generate preprocessing visualizations

### 5. Train a Model

Train a model using one of the available architectures:

```powershell
# Train with ResNet50 (recommended)
python src/train.py --model resnet50

# Or try other architectures
python src/train.py --model densenet121
python src/train.py --model efficientnet_b0
python src/train.py --model custom_cnn
```

Training configuration can be modified in `config/config.yaml`

### 6. Evaluate the Model

Evaluate the trained model on the test set:

```powershell
python src/evaluate.py --model_path models/saved_models/best_model.pth
```

This generates:

- Confusion matrix
- ROC curve
- Precision-Recall curve
- Calibration curve
- Clinical evaluation report

Results are saved to `results/evaluation/`

### 7. Make Predictions

#### Single Image Prediction:

```powershell
python src/predict.py --image path/to/xray.jpg --explain
```

#### Batch Prediction:

```powershell
python src/predict.py --directory path/to/images/ --explain
```

### 8. Launch Web Application

Run the interactive Streamlit web app:

```powershell
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## 📁 Project Structure

```
Medical Image Classification/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── raw/                     # Raw dataset
│   ├── processed/               # Preprocessed images
│   └── splits/                  # Train/val/test split files
├── models/
│   ├── architectures/           # Model definitions
│   ├── pretrained/              # Pre-trained weights
│   └── saved_models/            # Trained model checkpoints
├── src/
│   ├── data_preprocessing.py    # Data preprocessing pipeline
│   ├── model_architectures.py   # CNN architectures
│   ├── transfer_learning.py     # Transfer learning utilities
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation pipeline
│   ├── predict.py               # Inference pipeline
│   └── explainable_ai.py        # Grad-CAM & visualization
├── utils/
│   ├── data_utils.py            # Data handling utilities
│   ├── visualization.py         # Plotting functions
│   └── metrics.py               # Custom metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_explainable_ai.ipynb
├── results/                     # Output directory
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🎯 Available Models

### 1. Custom CNN

- Built from scratch for medical imaging
- ~500K parameters
- Good for understanding CNN fundamentals

```powershell
python src/train.py --model custom_cnn
```

### 2. ResNet50 (Recommended)

- Pre-trained on ImageNet
- ~23M parameters
- Excellent accuracy (>95%)

```powershell
python src/train.py --model resnet50
```

### 3. DenseNet121

- Dense connections between layers
- ~7M parameters
- Efficient feature reuse

```powershell
python src/train.py --model densenet121
```

### 4. EfficientNet-B0

- State-of-the-art efficiency
- ~5M parameters
- Best accuracy/computation tradeoff

```powershell
python src/train.py --model efficientnet_b0
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Training Parameters:

```yaml
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  optimizer: adam
  early_stopping_patience: 10
```

### Model Settings:

```yaml
model:
  architecture: resnet50
  pretrained: true
  dropout_rate: 0.5
  freeze_backbone: false
```

### Data Augmentation:

```yaml
augmentation:
  enabled: true
  rotation_range: 15
  horizontal_flip: true
  brightness_range: [0.8, 1.2]
```

## 📊 Expected Results

With the recommended configuration:

- **Accuracy**: >95%
- **Sensitivity**: >96% (crucial for medical diagnosis)
- **Specificity**: >94%
- **AUC-ROC**: >0.98

## 🔍 Explainable AI

The project includes Grad-CAM visualization to understand model decisions:

```python
from explainable_ai import MedicalImageExplainer

explainer = MedicalImageExplainer(model, device)
explanation = explainer.explain_prediction("path/to/xray.jpg")
```

Visualizations show:

- Which regions the model focuses on
- Attention heatmaps overlayed on original images
- Confidence scores and predictions

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config/config.yaml`:

```yaml
data:
  batch_size: 16 # or smaller
```

### Slow Training

- Enable mixed precision training
- Use fewer workers for data loading
- Consider using a smaller model

### Import Errors

Ensure all dependencies are installed:

```powershell
pip install -r requirements.txt --upgrade
```

### Dataset Not Found

Verify the dataset path in notebooks and scripts matches your actual data location.

## 📚 Jupyter Notebooks Workflow

Follow these notebooks in order:

1. **01_data_exploration.ipynb**

   - Load and examine dataset
   - Visualize sample images
   - Analyze class distribution

2. **02_preprocessing.ipynb**

   - Apply CLAHE enhancement
   - Create data splits
   - Implement augmentation

3. **03_model_training.ipynb**

   - Train different architectures
   - Compare performance
   - Tune hyperparameters

4. **04_evaluation.ipynb**

   - Comprehensive evaluation
   - Generate metrics and plots
   - Clinical interpretation

5. **05_explainable_ai.ipynb**
   - Grad-CAM visualizations
   - Attention analysis
   - Model interpretability

## 🏥 Clinical Use Guidelines

### Model Strengths:

✅ High sensitivity for pneumonia detection
✅ Fast analysis (seconds per image)
✅ Consistent performance
✅ Explainable predictions

### Limitations:

⚠️ Training data diversity
⚠️ Edge cases handling
⚠️ Cannot replace radiologist expertise

### Recommendations:

1. Use as a screening tool, not diagnostic replacement
2. Always combine with clinical judgment
3. Review model attention maps
4. Monitor performance on new data
5. Regular model updates and validation

## 🔄 Model Updates

To retrain with new data:

1. Add new images to appropriate directories
2. Re-run preprocessing:
   ```powershell
   python src/data_preprocessing.py
   ```
3. Retrain model:
   ```powershell
   python src/train.py --model resnet50
   ```
4. Evaluate on new test set:
   ```powershell
   python src/evaluate.py
   ```

## 📞 Support & Resources

### Documentation:

- Project README: `README.md`
- Code documentation: Inline docstrings
- Jupyter notebooks: Step-by-step guides

### External Resources:

- PyTorch documentation: https://pytorch.org/docs/
- Medical imaging best practices
- Transfer learning guides

## ⚖️ Medical Disclaimer

**IMPORTANT**: This tool is for research and educational purposes only.

- NOT FDA approved
- NOT a substitute for professional medical advice
- Always consult qualified healthcare professionals
- Validate results with clinical findings

## 🎓 Learning Resources

### Concepts Covered:

1. **Image Preprocessing**: CLAHE, normalization, augmentation
2. **CNNs**: Convolutional layers, pooling, activation functions
3. **Transfer Learning**: Fine-tuning pre-trained models
4. **Explainable AI**: Grad-CAM, attention visualization
5. **Medical Metrics**: Sensitivity, specificity, PPV, NPV

### Next Steps:

- Experiment with different architectures
- Try ensemble methods
- Implement uncertainty estimation
- Explore multi-class classification (bacterial vs viral pneumonia)
- Deploy as REST API

## ✅ Verification Checklist

- [ ] Python environment set up
- [ ] Dependencies installed
- [ ] Dataset downloaded and extracted
- [ ] Data exploration completed
- [ ] Model training successful
- [ ] Evaluation metrics generated
- [ ] Predictions working
- [ ] Web app running

## 🚀 Advanced Features

### 1. Ensemble Models

```python
from model_architectures import ModelFactory

ensemble = ModelFactory.create_ensemble(
    ['resnet50', 'densenet121', 'efficientnet_b0'],
    num_classes=2
)
```

### 2. Progressive Unfreezing

```python
from transfer_learning import TransferLearningManager

manager = TransferLearningManager()
schedule = manager.setup_progressive_unfreezing(model)
```

### 3. Custom Metrics

```python
from utils.metrics import calculate_clinical_metrics

metrics = calculate_clinical_metrics(y_true, y_pred, y_prob)
```

## 📈 Performance Monitoring

Track model performance over time:

- Log predictions and ground truth
- Calculate metrics on new data
- Monitor for distribution shift
- Retrain when performance degrades

---

**Happy Learning and Building! 🎉**

For questions or issues, refer to the inline documentation or create detailed error reports with:

- Error messages
- Steps to reproduce
- System information
- Configuration used
