#  Deepfake Detection System using ML & DL

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

This project implements a deepfake detection system using both traditional machine learning and deep learning approaches. Facial images are processed and classified as **real** or **fake** using handcrafted visual features (for ML models) and deep neural representations. Alongside developing and evaluating classical ML models, the system incorporates insights from the [Meso4 CNN architecture](https://arxiv.org/pdf/1809.00888), a state-of-the-art deepfake detection framework. Performance comparisons are drawn between the implemented ML models and theoretical benchmarks from literature .

---

##  Dataset & Preprocessing

The dataset used in this project is a **large, real-world deepfake dataset** collected from various internet sources. It contains compressed `.tar.gz` archives hosting thousands of facial images extracted from videos. The dataset is publicly available on Hugging Face:  
🔗 [WildDeepfake on Hugging Face](https://huggingface.co/datasets/xingjunm/WildDeepfake/tree/main/deepfake_in_the_wild)

Custom preprocessing scripts were developed to:
- Extract `.png` images from `.tar.gz` archives
- Organize and clean the images into labeled directories
- Extract six meaningful image-based features:
  - Entropy
  - Blur
  - Noise
  - Keypoints
  - Blobs
  - Phase unwrapping

These features were compiled into a structured dataset for machine learning models.

---

##  Methodology

###  Traditional Machine Learning Models (Scikit-learn)
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**

These models were trained using the extracted features and evaluated using standard performance metrics.

###  Deep Learning Model
- **Meso4 CNN (TensorFlow/Keras)**  
  A compact convolutional neural network that processes raw images and learns spatial representations without manual feature engineering.

---

##  Results

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| **Meso4 (CNN)**      | **0.9400**   | 0.9380    | 0.9420 | 0.9400   |
| **Random Forest**    | 0.9162   | 0.9155    | 0.9250 | 0.9202   |
| **KNN**              | 0.8884   | 0.8935    | 0.8929 | 0.8932   |
| **Logistic Regression** | 0.8884 | 0.8935    | 0.8929 | 0.8932   |
| **XGBoost**          | 0.8065   | 0.7817    | 0.8737 | 0.8251   |
| **SVM**              | 0.7246   | 0.7060    | 0.8106 | 0.7547   |

Among the classical ML models, Random Forest stood out as the best-performing one, coming second only to Meso-4, the deep learning-based model. This demonstrates that classical machine learning—when carefully feature-engineered and fine-tuned—can serve as a viable alternative to deep learning-based models, especially under computational constraints.
