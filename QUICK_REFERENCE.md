# Medical Image Classification - Quick Reference

## 🚀 Quick Commands

### Setup

```powershell
pip install -r requirements.txt
```

### Training

```powershell
# Basic training with ResNet50
python src/train.py --model resnet50

# Training with custom architecture
python src/train.py --model custom_cnn

# Training with specific config
python src/train.py --model densenet121 --config config/config.yaml
```

### Evaluation

```powershell
# Evaluate best model
python src/evaluate.py

# Evaluate specific checkpoint
python src/evaluate.py --model_path models/saved_models/checkpoint_epoch_25.pth
```

### Prediction

```powershell
# Single image
python src/predict.py --image path/to/xray.jpg

# With explanation
python src/predict.py --image path/to/xray.jpg --explain

# Batch prediction
python src/predict.py --directory path/to/images/

# Directory with explanation
python src/predict.py --directory path/to/images/ --explain
```

### Web Application

```powershell
streamlit run app.py
```

### Jupyter Notebooks

```powershell
jupyter notebook notebooks/
```

## 📊 Model Architectures

| Model           | Parameters | Speed  | Accuracy | Best For                       |
| --------------- | ---------- | ------ | -------- | ------------------------------ |
| Custom CNN      | ~500K      | Fast   | ~90%     | Learning, Simple cases         |
| ResNet50        | ~23M       | Medium | >95%     | **Recommended** - Best balance |
| DenseNet121     | ~7M        | Medium | >94%     | Efficient, Good accuracy       |
| EfficientNet-B0 | ~5M        | Fast   | >94%     | Mobile deployment              |

## 🎯 Key Metrics

### Medical Metrics

- **Sensitivity (Recall)**: Ability to detect pneumonia cases
- **Specificity**: Ability to identify normal cases
- **PPV**: Probability positive prediction is correct
- **NPV**: Probability negative prediction is correct

### Target Performance

- Accuracy: >95%
- Sensitivity: >96% (critical for medical diagnosis)
- Specificity: >94%
- AUC-ROC: >0.98

## 📁 Important Files

### Configuration

- `config/config.yaml` - Main configuration file

### Source Code

- `src/train.py` - Training pipeline
- `src/evaluate.py` - Evaluation pipeline
- `src/predict.py` - Prediction/inference
- `src/explainable_ai.py` - Grad-CAM visualizations
- `src/data_preprocessing.py` - Data preprocessing
- `src/model_architectures.py` - Model definitions
- `src/transfer_learning.py` - Transfer learning utilities

### Utilities

- `utils/data_utils.py` - Data handling functions
- `utils/visualization.py` - Plotting functions
- `utils/metrics.py` - Custom metrics

### Application

- `app.py` - Streamlit web application

## ⚙️ Configuration Parameters

### Training

```yaml
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  optimizer: adam
  weight_decay: 0.0001
  early_stopping_patience: 10
```

### Model

```yaml
model:
  architecture: resnet50
  pretrained: true
  num_classes: 2
  dropout_rate: 0.5
  freeze_backbone: false
  fine_tune_layers: 10
```

### Data

```yaml
data:
  image_size: [224, 224]
  batch_size: 32
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

### Augmentation

```yaml
augmentation:
  enabled: true
  rotation_range: 15
  horizontal_flip: true
  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]
```

## 🔍 Explainable AI

### Grad-CAM Usage

```python
from explainable_ai import MedicalImageExplainer

explainer = MedicalImageExplainer(model, device)
explanation = explainer.explain_prediction("xray.jpg")
explainer.save_explanation(explanation, "output.png")
```

### Batch Explanation

```python
explainer.batch_explain(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    save_dir="results/explanations"
)
```

## 📊 Evaluation Outputs

After running evaluation, find results in `results/evaluation/`:

- `confusion_matrix.png` - Confusion matrix
- `roc_curve.png` - ROC curve with AUC
- `precision_recall_curve.png` - PR curve
- `calibration_curve.png` - Model calibration
- `clinical_report.md` - Clinical interpretation
- `evaluation_results.json` - Detailed metrics (JSON)

## 🐛 Common Issues & Solutions

### CUDA Out of Memory

```yaml
# In config/config.yaml, reduce:
data:
  batch_size: 16 # or 8
```

### Slow Training

```yaml
# Enable mixed precision
hardware:
  mixed_precision: true

# Reduce workers
data:
  num_workers: 2
```

### Import Errors

```powershell
pip install -r requirements.txt --upgrade
```

### Model Not Found

```powershell
# Train a model first
python src/train.py --model resnet50
```

## 📈 Monitoring Training

### TensorBoard (if enabled)

```powershell
tensorboard --logdir logs/
```

### Weights & Biases (if enabled)

```yaml
# In config/config.yaml
logging:
  wandb: true
```

## 🔄 Workflow

1. **Explore Data**: Run `notebooks/01_data_exploration.ipynb`
2. **Preprocess**: Run `python src/data_preprocessing.py`
3. **Train**: Run `python src/train.py --model resnet50`
4. **Evaluate**: Run `python src/evaluate.py`
5. **Predict**: Run `python src/predict.py --image xray.jpg`
6. **Deploy**: Run `streamlit run app.py`

## 🎓 Learning Path

### Beginners

1. Start with `01_data_exploration.ipynb`
2. Understand data preprocessing
3. Train `custom_cnn` model
4. Study evaluation metrics
5. Explore Grad-CAM visualizations

### Intermediate

1. Experiment with transfer learning models
2. Tune hyperparameters in config
3. Implement custom augmentations
4. Try different loss functions
5. Build ensemble models

### Advanced

1. Implement uncertainty estimation
2. Multi-class classification (bacterial vs viral)
3. Deploy as REST API
4. Optimize for mobile/edge devices
5. Integrate with PACS systems

## 💡 Tips & Best Practices

### Training

- Start with small learning rate (0.001)
- Use early stopping to prevent overfitting
- Monitor validation metrics, not just training loss
- Save checkpoints regularly
- Use class weights for imbalanced data

### Evaluation

- Always test on held-out test set
- Focus on sensitivity for medical applications
- Review false negatives carefully
- Validate on diverse patient populations
- Compare with radiologist performance

### Deployment

- Set appropriate confidence thresholds
- Implement uncertainty estimation
- Log all predictions for monitoring
- Regular model updates with new data
- Clinical validation before deployment

## 📞 Resources

### Documentation

- [PyTorch](https://pytorch.org/docs/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Streamlit](https://docs.streamlit.io/)

### Papers

- ResNet: "Deep Residual Learning for Image Recognition"
- DenseNet: "Densely Connected Convolutional Networks"
- Grad-CAM: "Grad-CAM: Visual Explanations from Deep Networks"
- CheXNet: "Radiologist-Level Pneumonia Detection on Chest X-Rays"

## ⚖️ Medical Disclaimer

**IMPORTANT**:

- For research and educational purposes only
- NOT for clinical diagnosis
- NOT FDA approved
- Always consult medical professionals
- Validate all predictions clinically

## 📝 Checklist

Before deployment:

- [ ] Model accuracy >95% on test set
- [ ] Sensitivity >96%
- [ ] Validated on diverse patient data
- [ ] Explainability features working
- [ ] Error handling implemented
- [ ] Logging and monitoring setup
- [ ] Clinical validation performed
- [ ] Ethical approval obtained
- [ ] Documentation complete
- [ ] User training provided

---

**Last Updated**: December 2025
**Version**: 1.0.0
