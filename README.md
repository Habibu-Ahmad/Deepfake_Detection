#  Deepfake Detection 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0-green)  

A hybrid deepfake detector combining **Meso4 (CNN)** and traditional ML. Achieves **94% accuracy** on the [WildDeepfake](https://huggingface.co/datasets/faceforensics/wilddeepfake) dataset.  

## 🚀 **Features**  
- **Meso4 Architecture**: Optimized for deepfake detection ([reference paper](https://arxiv.org/abs/1809.00888)).  
- **Handcrafted Features**: Entropy, noise, blur, and keypoints (inspired by [Li et al. (2020)](https://arxiv.org/abs/2001.01274)).  
- **Model Comparison**: SVM, Random Forest, and Logistic Regression benchmarks.  

## 📊 **Results**  
| Model               | Accuracy | Precision | Recall | F1 Score | AUC  |  
|---------------------|----------|-----------|--------|----------|------|  
| **Meso4 (CNN)**     | 94.00%   | 93.80%    | 94.20% | 94.00%   | 0.98 |  
| **SVM**             | 91.50%   | 90.30%    | 92.50% | 91.37%   | 0.92 |  
| **Random Forest**   | 88.00%   | 87.20%    | 89.00% | 88.10%   | 0.89 |  
| **Logistic Reg.**   | 85.00%   | 83.50%    | 86.00% | 84.75%   | 0.87 |  

## ⚙️ **Setup**  
1. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  # TensorFlow, scikit-learn, OpenCV  
