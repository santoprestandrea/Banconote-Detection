# Banknote Authenticity Detection using Computer Vision

## Overview

This project focuses on detecting whether a banknote is genuine or fake using Computer Vision techniques.

The goal is to build a complete pipeline starting from image preprocessing, feature extraction, and classification, and ending with performance evaluation.

The system processes an input image of a banknote and predicts its authenticity using a trained machine learning model.

---

## Features

- Automatic banknote detection and alignment
- Image preprocessing (noise reduction, perspective correction)
- Feature extraction using:
  - HOG (Histogram of Oriented Gradients)
  - LBP (Local Binary Patterns)
  - Color histograms
  - Statistical features
- **Two classification approaches**:
  - Support Vector Machine (SVM) - Classical Computer Vision
  - Convolutional Neural Network (CNN - MobileNetV2) - Deep Learning
- Performance evaluation with standard metrics
- Model comparison: SVM vs CNN side-by-side

---

## Project Structure

```
## Project Structure

```bash
progetto_bancnote/
│
├── data/train/
│       ├── genuine/
│       └── fake/
│
├── data/test/
│       ├── genuine/
│       └── fake/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   ├── dataset.py
│   ├── train_svm.py          # Training SVM
│   ├── train_cnn.py          # Training CNN (MobileNetV2)
│   ├── compare_models.py     # Comparazione SVM vs CNN
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   └── codice_fake.py
│
├── models/                   # Modelli salvati
├── outputs/                  # Risultati e visualizzazioni
├── COMPARE_MODELS.md         # Guida comparazione
├── Santo.md                  # Documentazione tecnica
├── Santo.pdf                 # PDF tecnico
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd progetto_bancnote
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## Quick Start

### Train SVM model:
```bash
python src/train_svm.py
```

### Train CNN model:
```bash
python src/train_cnn.py
```

### Evaluate SVM model:
```bash
python src/evaluate.py
```

### Compare SVM vs CNN (side-by-side metrics):
```bash
python src/compare_models.py
```

Generates `outputs/comparison_svm_vs_cnn.txt` and `outputs/comparison_table.png` with:
- Accuracy, Precision, Recall, F1-Score comparison
- Confusion matrices for both models
- Classification reports
- Best model summary

### Predict on new image:
```bash
python src/predict.py --image path/to/image.jpg
```


## Pipeline Description

The system follows a standard Computer Vision pipeline with **two different classification approaches**:

### 1. Data Acquisition & Preprocessing

- Convert image to grayscale
- Blur and edge detection
- Contour detection
- Perspective correction
- Image standardization

### 2. Feature Extraction

- **HOG** for shape representation
- **LBP** for texture features
- Color histograms
- Statistical descriptors (mean, std, min, max)
- Quality features (sharpness, contrast, entropy)

### 3. Classification - Two Approaches

**Approach A: Classical Computer Vision (SVM)**
- Support Vector Machine with RBF kernel
- Hyperparameter tuning via GridSearchCV
- Fast training, interpretable, works well on small datasets

**Approach B: Deep Learning (CNN)**
- MobileNetV2 pre-trained on ImageNet
- Transfer learning: frozen feature extractor + custom classifier
- Automatic feature learning, scalable to larger datasets

### 4. Post-processing & Evaluation

- Prediction probability analysis
- Confidence level assignment (LOW/MEDIUM/HIGH)
- Performance evaluation with standard metrics

---

## Results

The model performance is evaluated using standard classification metrics:

- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives
- **F1-score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Visual breakdown of predictions

### Model Comparison

To compare SVM vs CNN performance side-by-side, run:

```bash
python src/compare_models.py
```

This generates:
- `comparison_svm_vs_cnn.txt` - Detailed report with metrics table
- `comparison_table.png` - Visual comparison chart

**Key findings from Santo.md:**
- Classical SVM performs well on small datasets
- CNN requires larger datasets but offers better scalability
- Both approaches have complementary strengths



> **Note:** Results depend on dataset quality and size. Current evaluation is based on a small test set.

---

## Limitations

- Small dataset size (68 training, 2 test images)
- Fake banknotes are artificially generated - not real counterfeits
- SVM performance depends on manual feature engineering
- CNN requires larger dataset to fully exploit its potential
- Preprocessing may fail in difficult lighting conditions
- Results may not generalize to real-world scenarios without larger, more diverse data

---

## Improvements & Future Work

- Collect real-world fake banknote datasets
- Increase test set size for statistical significance
- Fine-tune both models with cross-validation
- Implement ensemble methods (SVM + CNN combination)
- Add real-time optimization
- Deploy as web application or API
- Test with different CNN architectures (ResNet, EfficientNet)