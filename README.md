#  Deepfake Detection System using ML & DL

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

This project implements a deepfake detection system using a hybrid of machine learning and deep learning models. It classifies facial images as either **real** or **fake** based on both handcrafted features and raw image data. The system was designed to evaluate the performance of multiple traditional ML classifiers alongside a CNN-based deep learning model to determine which approach is most effective for deepfake detection.

---

##  Dataset & Preprocessing

The dataset used is [WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake/tree/main/deepfake_in_the_wild), a real-world collection of over 7,000 facial sequences sourced from deepfake videos on the internet. It contains an equal distribution of real and fake face images with variations in angle, resolution, lighting, and manipulation techniques.

The original files are provided as `.tar.gz` archives containing `.png` images. A custom preprocessing pipeline was developed to recursively extract all `.png` images, filter out irrelevant data, and organize them into labeled directories (`real` or `fake`). Six handcrafted visual features—**entropy, blur, noise, keypoints, blobs**, and **phase unwrapping**—were computed for each image and compiled into a structured DataFrame used for training ML models.

---

## Methodology

### Machine Learning Models (using extracted features)
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **Logistic Regression**

These models were trained on a numerical dataset of handcrafted image features. Feature importance was tested using ablation, and model performance was evaluated using standard classification metrics.

### Deep Learning Model
- **Meso4 CNN (Keras/TensorFlow)**  
The Meso4 model was trained directly on raw facial images, resized and normalized, allowing it to learn discriminative patterns automatically without manual feature engineering.

---

##  Results

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| **Meso4 (CNN)**      | 0.9400   | 0.9380    | 0.9420 | 0.9400   |
| **Random Forest**    | 0.9162   | 0.9155    | 0.9250 | 0.9202   |
| **Logistic Regression** | 0.8884 | 0.8935    | 0.8929 | 0.8932   |
| **XGBoost**          | 0.8065   | 0.7817    | 0.8737 | 0.8251   |
| **SVM**              | 0.7246   | 0.7060    | 0.8106 | 0.7547   |

>  The Meso

---

##  Setup & Run

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
