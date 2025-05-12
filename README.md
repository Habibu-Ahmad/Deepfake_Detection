# 🧠 Deepfake Detection System using ML & DL

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project implements a hybrid deepfake detection system using both machine learning and deep learning techniques. The goal is to accurately classify facial images as **real** or **fake** using handcrafted image features and a CNN-based architecture. A custom feature extraction pipeline is combined with benchmark ML models and a deep learning model to compare performance and identify the most effective approach. The CNN model used is **Meso4**, optimized for detecting artifacts introduced during deepfake generation.

---

## 📁 Dataset & Preprocessing

- **Dataset**: [WildDeepfake](https://huggingface.co/datasets/faceforensics/wilddeepfake)  
- **Format**: `.tar.gz` archives containing `.png` facial images  
- **Total samples**: 7,314 face sequences from 707 videos (balanced between real and fake)  
- **Source**: Real-world deepfakes collected from the internet (diverse in resolution, lighting, angles)

### 🔧 Preprocessing Pipeline
1. Recursively extracted `.tar.gz` files using custom Python scripts.
2. Filtered and organized only `.png` files into labeled directories.
3. Extracted six handcrafted features per image:
   - Entropy
   - Blur
   - Noise
   - Keypoints
   - Blobs
   - Phase unwrapping
4. Stored extracted features and labels in a structured DataFrame (`.csv`) for ML models.

---

## 🧠 Methodology

### ✅ Machine Learning Models (Scikit-learn)
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Trained on the handcrafted features. Feature importance was analyzed via ablation testing to determine their contribution.

### 🔬 Deep Learning Model (TensorFlow/Keras)
- **Meso4 CNN** (optimized for deepfake detection)
- Trained directly on raw images resized and normalized
- Able to learn discriminative features without manual extraction

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| **Meso4 (CNN)**     | 94.00%   | 93.80%    | 94.20% | 94.00%   | 0.98 |
| **SVM**             | 91.50%   | 90.30%    | 92.50% | 91.37%   | 0.92 |
| **Random Forest**   | 88.00%   | 87.20%    | 89.00% | 88.10%   | 0.89 |
| **Logistic Reg.**   | 85.00%   | 83.50%    | 86.00% | 84.75%   | 0.87 |

---

## 💻 Setup & Run

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
