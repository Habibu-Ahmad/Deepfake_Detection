#  Deepfake Detection System using ML & DL

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

##  Overview

This project implements a deepfake detection system using both machine learning and deep learning approaches. It processes facial images to classify them as either **real** or **fake** by leveraging handcrafted visual features and deep neural representations. Traditional ML models were developed and evaluated alongside a deep learning baseline based on the [Meso4 CNN architecture](https://arxiv.org/pdf/1809.00888), commonly used in prior deepfake research.

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
| **Meso4 (CNN)**      | 0.9400   | 0.9380    | 0.9420 | 0.9400   |
| **Random Forest**    | 0.9162   | 0.9155    | 0.9250 | 0.9202   |
| **KNN**              | 0.8884   | 0.8935    | 0.8929 | 0.8932   |
| **Logistic Regression** | 0.8884 | 0.8935    | 0.8929 | 0.8932   |
| **XGBoost**          | 0.8065   | 0.7817    | 0.8737 | 0.8251   |
| **SVM**              | 0.7246   | 0.7060    | 0.8106 | 0.7547   |

>  The baseline Meso4 CNN model demonstrated the strongest performance, highlighting the strength of convolutional features in deepfake detection tasks.

---

##  Setup & Run

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
