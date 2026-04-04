**Phishing Detection System Using Machine Learning**
**Project Overview**
This project develops an intelligent machine learning system to automatically detect phishing websites by analyzing URL and domain-based features. The system helps prevent cyberattacks by identifying malicious websites before users can be harmed.

**Key Features**
 Real-time Phishing Detection - Analyze any URL instantly

 Risk Score Calculation - Get 0-100% risk score with color-coded alerts

 Multiple ML Models - Logistic Regression, Decision Tree, Random Forest

 Hyperparameter Tuning - Optimized Random Forest (98.7% accuracy)

Pattern Discovery - Unsupervised K-Means clustering

 Interactive Dashboard - Streamlit web interface

 Batch Processing - Analyze multiple URLs from CSV

 Feature Extraction - Automatic 48-feature extraction from any URL

**Model Performance**
Model	Accuracy	Precision	Recall	F1-Score
Random Forest (Tuned)	98.7%	98.8%	98.6%	98.7%
Random Forest	98.5%	98.6%	98.4%	98.5%
Decision Tree	96.2%	95.8%	96.5%	96.2%
Logistic Regression	94.5%	94.1%	94.8%	94.4%


**Project Structure**
text
phishing-detection/
│
├── data/                          # Dataset folder
│   └── Phishing_Legitimate_full.csv
│
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Load, clean, split, scale data
│   ├── supervised_learning.py     # Train & evaluate ML models
│   ├── unsupervised_learning.py   # K-Means clustering
│   └── url_feature_extractor.py   # Extract 48 features from URLs
│
├── models/                        # Saved trained models
│   ├── phishing_detector_model.pkl
│   ├── kmeans_clustering.pkl
│   └── scaler.pkl
│
├── visualizations/                # Generated plots
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── cluster_optimization.png
│   └── cluster_pca_visualization.png
│
├── main_pipeline.py               # Main execution script
├── ui_app.py                      # Streamlit web interface
├── check_data.py                  # Dataset validation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file


****Installation**
Prerequisites**
Python 3.8 or higher

pip package manager

**Step 1:** Clone or Download Project
bash
cd "C:\Users\rypri\OneDrive\文件\machine learning sem 4\phishing-detection"
**Step 2**: Install Dependencies
bash
pip install -r requirements.txt
Or install manually:

bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib tldextract
**Step 3:** Place Dataset
Ensure your dataset Phishing_Legitimate_full.csv is in the data/ folder.

**Usage**
**Option 1:** Run Complete Pipeline
bash
python main_pipeline.py
This will:

Load and preprocess data

Train all 4 models

Perform hyperparameter tuning

Generate risk scores

Run K-Means clustering

Save models and visualizations

**Option 2:** Launch Web Interface
bash
streamlit run ui_app.py
Then open your browser to http://localhost:8501

**O****ption 3**: Check Dataset
bash
python check_data.py
Web Interface Tabs
Tab	Function
🌐 URL Analysis	Enter any URL for real-time phishing detection
🎯 Single URL Detection	Manual feature input (demo mode)
📊 Batch Processing	Upload CSV for bulk analysis
📈 Model Performance	View all model metrics and charts
🔍 K-Means Clustering	Explore discovered patterns
Risk Score Interpretation
Risk Score	Color	Meaning
0-30%	🟢 Green	Low Risk - Likely legitimate website
30-70%	🟡 Yellow	Medium Risk - Suspicious, proceed with caution
70-100%	🔴 Red	High Risk - Likely phishing website
**Example URLs to Test**
**Legitimate **(Expected Low Risk)
text
https://www.google.com
https://www.github.com
https://www.microsoft.com
**Suspicious** (Expected Medium-High Risk)
text
http://paypal.com.verify-account.xyz/login
http://193.168.1.100/~secure/bankofamerica/login.php
https://appleid.apple.com.verify.duckdns.org
**How It Works**
1. Data Preprocessing
Loads 10,000 websites with 48 URL features

Removes ID column, handles missing values

Splits into 80% training / 20% testing

Scales features using StandardScaler

2. Model Training
Trains Logistic Regression, Decision Tree, Random Forest

Hyperparameter tuning with GridSearchCV (5-fold CV)

Best parameters: n_estimators=200, max_depth=20

3. Risk Score Calculation
text
Phishing Risk Score = (1 - Probability of Legitimate) × 100
0-30%: 🟢 Safe

30-70%: 🟡 Suspicious

70-100%: 🔴 Phishing

4. Unsupervised Learning
K-Means clustering finds 2 natural groups

Cluster 0: 64% Phishing, 36% Legitimate

Cluster 1: 43% Phishing, 57% Legitimate

Purity: 59.6%

5. URL Feature Extraction
Extracts 48 features from any URL:

Basic: URL length, number of dots, dashes

Security: HTTPS presence, IP address usage

Structure: Subdomain level, path length

Content: Sensitive words, brand names

**Key Findings**
Best Model: Random Forest with hyperparameter tuning (98.7% accuracy)

Most Important Features: URL Length, Number of Dots, HTTPS presence

Clustering Results: 2 clusters with significant overlap (phishing mimics legitimate)

ROC-AUC: 0.9989 (near-perfect discrimination)

Troubleshooting
Common Issues and Solutions
Issue	Solution
File not found	Ensure dataset is in data/ folder
Import errors	Run pip install -r requirements.txt
Models not loading	Run python main_pipeline.py first
Plot display issues	Code uses matplotlib.use('Agg') to save files
Slow spectral clustering	Use K-Means instead (already configured)
Technologies Used
Technology	Purpose
Python 3.13	Core programming language
Pandas	Data manipulation
NumPy	Numerical operations
Scikit-learn	ML models (Random Forest, K-Means, etc.)
Matplotlib/Seaborn	Data visualization
Streamlit	Web interface
Plotly	Interactive gauges and charts
Joblib	Model persistence
Future Improvements
Add more training data for better generalization

Implement deep learning (LSTM/CNN) for URL analysis

Add browser extension for real-time protection

Include SSL certificate validation

Add website screenshot analysis

Deploy as REST API service

Output Files
Models (saved in models/)
phishing_detector_model.pkl - Trained Random Forest

kmeans_clustering.pkl - K-Means clustering model

scaler.pkl - StandardScaler for feature normalization

Visualizations (saved in visualizations/)
class_distribution.png - Dataset balance

model_comparison.png - Model performance comparison

roc_curves.png - ROC curves for all models

feature_importance.png - Top 20 important features

cluster_optimization.png - Elbow method graph

cluster_pca_visualization.png - 2D cluster visualization

License
This project is for educational purposes as part of a Machine Learning course project.

Author
Machine Learning Project - Semester 4

Acknowledgments
Dataset: Phishing Dataset for Machine Learning

Scikit-learn documentation

Streamlit documentation

Quick Start Commands
bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main_pipeline.py

# 3. Launch the web interface
streamlit run ui_app.py

# 4. (Optional) Check your data
python check_data.py
