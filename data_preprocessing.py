"""
Module for data preprocessing and EDA
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()[:10]}...")
        return self.df
    
    def explore_data(self):
        """Perform EDA on the dataset"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        print("\n1. Dataset Info:")
        print(self.df.info())
        
        print("\n2. Missing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print("\n3. Class Distribution:")
        if 'CLASS_LABEL' in self.df.columns:
            class_dist = self.df['CLASS_LABEL'].value_counts()
            print(class_dist)
            print(f"\nClass mapping: 0 = Phishing, 1 = Legitimate")
        else:
            print("CLASS_LABEL column not found!")
        
        self._create_visualizations()
        
        return class_dist if 'CLASS_LABEL' in self.df.columns else None
    
    def _create_visualizations(self):
        """Create EDA visualizations"""
        os.makedirs('visualizations', exist_ok=True)
        print("\nGenerating visualizations...")
        
        if 'CLASS_LABEL' in self.df.columns:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            colors = ['red', 'green']
            self.df['CLASS_LABEL'].value_counts().plot(kind='bar', color=colors)
            plt.title('Class Distribution (0=Phishing, 1=Legitimate)')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['Phishing (0)', 'Legitimate (1)'], rotation=0)
            
            plt.subplot(1, 2, 2)
            self.df['CLASS_LABEL'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                        colors=colors)
            plt.title('Class Distribution Percentage')
            plt.tight_layout()
            plt.savefig('visualizations/class_distribution.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # Correlation Heatmap
        plt.figure(figsize=(12, 10))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Remove id and CLASS_LABEL for correlation
        plot_cols = [col for col in numeric_cols if col not in ['id', 'CLASS_LABEL']][:20]
        if len(plot_cols) > 1:
            correlation_matrix = self.df[plot_cols].corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap (Top 20 Features)')
            plt.tight_layout()
            plt.savefig('visualizations/correlation_heatmap.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        print("✓ Visualizations saved to 'visualizations/' folder")
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Remove ID column if exists
        if 'id' in self.df.columns:
            print("Removing 'id' column...")
            self.df = self.df.drop('id', axis=1)
        
        # Check for CLASS_LABEL
        if 'CLASS_LABEL' not in self.df.columns:
            raise ValueError("CLASS_LABEL column not found!")
        
        # Separate features and target
        self.X = self.df.drop('CLASS_LABEL', axis=1)
        self.y = self.df['CLASS_LABEL'].copy()
        
        # Your data already has 0 and 1, no conversion needed
        print(f"Class distribution before preprocessing:")
        print(f"  Legitimate (1): {(self.y == 1).sum()}")
        print(f"  Phishing (0): {(self.y == 0).sum()}")
        
        # Ensure no NaN values
        if self.y.isnull().any():
            valid_idx = ~self.y.isnull()
            self.X = self.X[valid_idx]
            self.y = self.y[valid_idx]
        
        # Convert to integers
        self.y = self.y.astype(int)
        
        print(f"\nFeatures shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Final class distribution:\n{self.y.value_counts()}")
        
        return self.X, self.y
    
    def split_and_scale(self, test_size=0.2, random_state=42):
        """Split data and apply scaling"""
        print("\n" + "="*50)
        print("TRAIN-TEST SPLIT & SCALING")
        print("="*50)
        
        # Final check - ensure no NaN
        if self.X.isnull().any().any():
            print("Filling remaining NaN in features...")
            self.X = self.X.fillna(self.X.median())
        
        # Convert y to numpy array
        y_clean = self.y.values.flatten()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, y_clean, test_size=test_size, random_state=random_state, 
            stratify=y_clean
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Training class distribution: {np.bincount(self.y_train)}")
        print(f"Test class distribution: {np.bincount(self.y_test)}")
        
        # Scale features
        print("Applying StandardScaler...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Data preprocessing completed!")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test, self.scaler
