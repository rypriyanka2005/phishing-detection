"""
Module for supervised learning models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class SupervisedLearning:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize supervised learning models
        
        Parameters:
        X_train, X_test, y_train, y_test: Training and testing data
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_models(self):
        """Train multiple supervised learning models"""
        print("\n" + "="*50)
        print("SUPERVISED LEARNING - MODEL TRAINING")
        print("="*50)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # 2. Decision Tree
        print("2. Training Decision Tree...")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt
        
        # 3. Random Forest
        print("3. Training Random Forest...")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        print("✓ All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n{'-'*40}")
            print(f"Model: {name}")
            print(f"{'-'*40}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
            print(f"FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
        
        return self.results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for Random Forest"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING - Random Forest")
        print("="*50)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1', 
            n_jobs=-1, verbose=1
        )
        
        print("Performing Grid Search with 5-fold CV...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
        
        # Train best model
        self.best_model = grid_search.best_estimator_
        self.models['Random Forest (Tuned)'] = self.best_model
        
        # Evaluate tuned model
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\nTuned Model Performance on Test Set:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_pred):.4f}")
        
        # Add tuned model to results
        self.results['Random Forest (Tuned)'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'model': self.best_model,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.best_model
    
    def get_risk_score(self, probabilities):
        """
        Convert probabilities to phishing risk scores
        
        Parameters:
        probabilities: Prediction probabilities (0-1)
        
        Returns:
        risk_scores: Risk scores (0-100)
        """
        risk_scores = probabilities * 100
        
        # Categorize risk levels
        risk_categories = []
        for score in risk_scores:
            if score < 30:
                risk_categories.append('Low Risk (Legitimate)')
            elif score < 70:
                risk_categories.append('Medium Risk (Suspicious)')
            else:
                risk_categories.append('High Risk (Phishing)')
        
        return risk_scores, risk_categories
    
    def plot_results(self):
        """Create visualizations for model results and save to files"""
        os.makedirs('visualizations', exist_ok=True)
        
        print("\nGenerating model visualizations...")
        
        # 1. Model Comparison Bar Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy Comparison', 'Precision Comparison', 
                 'Recall Comparison', 'F1-Score Comparison']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = idx // 2, idx % 2
            values = [self.results[name][metric] for name in self.results.keys()]
            bars = axes[row, col].bar(self.results.keys(), values, 
                                      color=['skyblue', 'lightgreen', 'salmon', 'gold'])
            axes[row, col].set_title(title)
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_ylim([0, 1])
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/roc_curves.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance (Random Forest)
        if 'Random Forest' in self.models:
            plt.figure(figsize=(12, 8))
            rf_model = self.models['Random Forest']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.barh(range(20), importances[indices])
            plt.yticks(range(20), [f'Feature_{i}' for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Most Important Features - Random Forest')
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        print("✓ Visualizations saved to 'visualizations/' folder")
    
    def save_model(self, model_name='phishing_detector_model.pkl'):
        """Save the best model"""
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/{model_name}'
        joblib.dump(self.best_model, model_path)
        print(f"✓ Model saved to {model_path}")
        return model_path
