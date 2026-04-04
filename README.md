# 🔒 Phishing Detection System Using Machine Learning

An intelligent machine learning system to detect phishing websites by analyzing URL and domain-based features with **98.7% accuracy**.

## 🎯 Features

- ✅ **Real-time Phishing Detection** - Analyze any URL instantly
- ✅ **Risk Score Calculator** - 0-100% score with color-coded alerts (🟢/🟡/🔴)
- ✅ **Multiple ML Models** - Logistic Regression, Decision Tree, Random Forest
- ✅ **Hyperparameter Tuning** - GridSearchCV optimized Random Forest
- ✅ **Pattern Discovery** - Unsupervised K-Means clustering
- ✅ **Interactive Dashboard** - Streamlit web interface with 5 tabs
- ✅ **Batch Processing** - Analyze multiple URLs from CSV file
- ✅ **Automatic Feature Extraction** - Extracts 48 features from any URL

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest (Tuned)** | **98.7%** | **98.8%** | **98.6%** | **98.7%** |
| Random Forest | 98.5% | 98.6% | 98.4% | 98.5% |
| Decision Tree | 96.2% | 95.8% | 96.5% | 96.2% |
| Logistic Regression | 94.5% | 94.1% | 94.8% | 94.4% |

**ROC-AUC Score:** 0.9989 (Near-perfect discrimination)

## 📁 Project Structure
phishing-detection/
│
├── data/ # Dataset folder
│ └── Phishing_Legitimate_full.csv
│
├── src/ # Source code modules
│ ├── data_preprocessing.py # Load, clean, split, scale data
│ ├── supervised_learning.py # Train & evaluate ML models
│ ├── unsupervised_learning.py # K-Means clustering
│ └── url_feature_extractor.py # Extract 48 features from URLs
│
├── models/ # Saved trained models
│ ├── phishing_detector_model.pkl
│ ├── kmeans_clustering.pkl
│ └── scaler.pkl
│
├── visualizations/ # Generated plots
│ ├── class_distribution.png
│ ├── correlation_heatmap.png
│ ├── model_comparison.png
│ ├── roc_curves.png
│ ├── feature_importance.png
│ ├── cluster_optimization.png
│ └── cluster_pca_visualization.png
│
├── main_pipeline.py # Main execution script
├── ui_app.py # Streamlit web interface
├── check_data.py # Dataset validation script
├── requirements.txt # Python dependencies
└── README.md # This file


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/phishing-detection.git
cd phishing-detection

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Place dataset
Download the dataset from Kaggle and place it in the data/ folder as Phishing_Legitimate_full.csv

💻 Usage
Run the complete pipeline
python main_pipeline.py

Launch web interface
streamlit run ui_app.py

Check your dataset
python check_data.py

