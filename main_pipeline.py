"""
Main Pipeline - Phishing Detection System
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from supervised_learning import SupervisedLearning
from unsupervised_learning import UnsupervisedLearning

def find_dataset_file():
    """Find the dataset CSV file"""
    import glob
    
    # Check multiple locations
    search_paths = [
        "data/*.csv",
        "*.csv",
        "../*.csv",
        "../../*.csv"
    ]
    
    for pattern in search_paths:
        files = glob.glob(pattern)
        if files:
            print(f"✅ Found dataset: {files[0]}")
            return files[0]
    
    print("❌ No CSV file found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files here: {os.listdir('.')}")
    
    if os.path.exists('data'):
        print(f"Files in data/: {os.listdir('data')}")
    
    return None

def main():
    print("="*60)
    print("PHISHING DETECTION SYSTEM - MACHINE LEARNING PIPELINE")
    print("="*60)
    
    # Find data
    data_path = find_dataset_file()
    if data_path is None:
        print("\nPlease place your CSV file in the 'data' folder or current directory")
        return None, None
    
    # Step 1: Preprocessing
    print("\n" + "🔹"*30)
    print("STEP 1: DATA PREPROCESSING")
    print("🔹"*30)
    
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load_data()
    preprocessor.explore_data()
    X, y = preprocessor.preprocess_data()
    X_train, X_test, y_train, y_test, scaler = preprocessor.split_and_scale()
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Step 2: Supervised Learning
    print("\n" + "🔹"*30)
    print("STEP 2: SUPERVISED LEARNING")
    print("🔹"*30)
    
    sl = SupervisedLearning(X_train, X_test, y_train, y_test)
    sl.train_models()
    results = sl.evaluate_models()
    sl.hyperparameter_tuning()
    
    # Risk scores
    y_proba = sl.best_model.predict_proba(X_test)[:, 1]
    risk_scores, categories = sl.get_risk_score(y_proba)
    
    print("\n" + "="*50)
    print("SAMPLE RISK SCORES")
    print("="*50)
    for i in range(min(5, len(risk_scores))):
        actual = "Legit" if y_test[i] == 1 else "Phish"
        print(f"Sample {i+1}: Actual={actual} | Risk={risk_scores[i]:.1f}% | {categories[i]}")
    
    sl.plot_results()
    sl.save_model()
    
    # Step 3: Unsupervised Learning
    print("\n" + "🔹"*30)
    print("STEP 3: UNSUPERVISED LEARNING")
    print("🔹"*30)
    
    sample_size = min(5000, len(X_train))
    ul = UnsupervisedLearning(X_train[:sample_size], y_train[:sample_size])
    optimal_k, _ = ul.find_optimal_clusters(max_clusters=6)
    ul.perform_clustering(optimal_k)
    ul.analyze_clusters_with_labels()
    ul.visualize_clusters()
    ul.save_clustering_model()
    
    # Summary
    print("\n" + "🔹"*30)
    print("PIPELINE COMPLETED!")
    print("🔹"*30)
    print(f"\n✅ Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"✅ Models saved in 'models/'")
    print(f"✅ Visualizations saved in 'visualizations/'")
    print(f"\nRun: streamlit run ui_app.py")
    
    return sl, ul

if __name__ == "__main__":
    main()
