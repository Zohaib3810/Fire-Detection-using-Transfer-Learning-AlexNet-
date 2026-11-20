
# Fire Detection using Transfer Learning (AlexNet)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Computer Vision pipeline implementing binary Fire vs. Non-Fire classification utilizing two transfer learning strategies with AlexNet (pre-trained on ImageNet-1k):
1. Feature Extraction + Classical ML Classifier: Deep features extraction via pre-trained AlexNet layers mapped to a Logistic Regression head.
2. Classifier Head Fine-Tuning: Layer freezing with PyTorch optimization of a customized linear layer outputting binary predictions.

---

## Project Workflow & Tasks

### 1. Data Preparation & Splitting
- Dataset Extraction: Unpacks dataset containing fire and non-fire images.
- Preprocessing: Resizes all images to (227, 227, 3), normalizes inputs to [0, 1], and standardizes using standard ImageNet mean and standard deviations ([0.485, 0.456, 0.406] & [0.229, 0.224, 0.225]).
- Splitting: Stratified splitting partition (80% Train, 20% Test).

### 2. Feature Extraction & Logistic Regression
- Bottleneck Activation: Strips final classifier layer out of AlexNet to expose a 4096-dimensional feature vector.
- Inference: Pass dataset through fixed network layers without gradient updates.
- Classification: Train a Scikit-Learn LogisticRegression classifier on the extracted high-level features.

### 3. Fine-Tuning with PyTorch
- Freeze convolutional features and adaptive pooling layers to protect learned representations.
- Substitute final fully connected layer with nn.Linear(in_features, 2).
- Train head over 10 epochs using Adam optimizer and CrossEntropyLoss.

### 4. Metrics & Evaluation
- Plots comprehensive Training and Validation curves (Accuracy & Loss).
- Highlights model capabilities via Confusion Matrices and Scikit-Learn Classification Reports.

---

## Experimental Results

### Strategy A: Logistic Regression on Extracted Features
* Accuracy: 96.33%
* Precision: 95.16%
* Recall: 98.33%
* F1-Score: 96.72%

#### Classification Report:
```text
              precision    recall  f1-score   support

  Non-Fire         0.98      0.94      0.96        49
      Fire         0.95      0.98      0.97        60
```

### Strategy B: PyTorch Linear Head Fine-Tuning
* Stable Validation Accuracy: ~95.41% (reaching peak at 97.25% on Epoch 7).
* Trained using Adam Optimizer with a learning rate of 0.001 over 10 epochs.

---

## Quick Setup & Usage

### Dependencies
Ensure Python and standard ML libraries are installed:
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn opencv-python
```

### How to Run
1. Place your fire.zip in the root workspace directory.
2. Execute the Jupyter/Colab notebook to run through data ingestion, feature extraction, PyTorch optimization, and visual evaluation steps.

---

## Technologies Used
- Core Frameworks: PyTorch, torchvision, Scikit-Learn
- Processing: NumPy, OpenCV
- Visualization: Matplotlib, Seaborn
```
