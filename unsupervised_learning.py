"""
Module for unsupervised learning (clustering)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedLearning:
    def __init__(self, X, y=None):
        """
        Initialize unsupervised learning module
        
        Parameters:
        X: Feature matrix
        y: Target labels (optional, for comparison)
        """
        self.X = X
        self.y = y
        self.kmeans = None
        self.labels = None
        self.pca_result = None
        
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using Elbow method and Silhouette Score"""
        print("\n" + "="*50)
        print("UNSUPERVISED LEARNING - CLUSTERING ANALYSIS")
        print("="*50)
        
        inertias = []
        silhouette_scores = []
        
        print("Finding optimal number of clusters...")
        for k in range(2, max_clusters + 1):
            print(f"  Testing k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            inertias.append(kmeans.inertia_)
            
            score = silhouette_score(self.X, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Create directory for plots
        os.makedirs('visualizations', exist_ok=True)
        
        # Plot Elbow Method
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow curve
        ax1.plot(range(2, max_clusters + 1), inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Optimal k')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_optimization.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Determine optimal k
        optimal_k = np.argmax(silhouette_scores) + 2
        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"Best Silhouette Score: {max(silhouette_scores):.4f}")
        
        return optimal_k, max(silhouette_scores)
    
    def perform_clustering(self, n_clusters=3):
        """Perform K-Means clustering"""
        print(f"\nPerforming K-Means clustering with {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.X)
        
        # Calculate silhouette score
        sil_score = silhouette_score(self.X, self.labels)
        print(f"Silhouette Score: {sil_score:.4f}")
        
        # Analyze cluster distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nCluster Distribution:")
        for cluster, count in zip(unique, counts):
            print(f"  Cluster {cluster}: {count} samples ({count/len(self.labels)*100:.1f}%)")
        
        return self.labels, sil_score
    
    def analyze_clusters_with_labels(self):
        """Analyze clusters against true labels if available"""
        if self.y is None:
            print("No true labels available for analysis")
            return
        
        print("\nCluster vs True Label Analysis:")
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(self.y, self.labels)
        print("\nCross-tabulation (True Labels vs Clusters):")
        print(cross_tab)
        
        # Calculate purity
        purity = np.sum(np.max(cross_tab, axis=0)) / len(self.labels)
        print(f"\nCluster Purity: {purity:.4f}")
        
        # Visualize
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cluster distribution
        ax[0].bar(range(len(np.unique(self.labels))), 
                  np.bincount(self.labels))
        ax[0].set_xlabel('Cluster')
        ax[0].set_ylabel('Count')
        ax[0].set_title('Cluster Size Distribution')
        
        # Cross-tabulation heatmap
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax[1])
        ax[1].set_xlabel('Cluster')
        ax[1].set_ylabel('True Label (0=Phishing, 1=Legitimate)')
        ax[1].set_title('True Labels vs Clusters')
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_label_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    def visualize_clusters(self):
        """Visualize clusters using PCA for dimensionality reduction"""
        print("\nVisualizing clusters using PCA...")
        
        # Apply PCA
        pca = PCA(n_components=2)
        self.pca_result = pca.fit_transform(self.X)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA with clusters
        scatter1 = ax1.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                              c=self.labels, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.set_title('Clusters Visualization (PCA)')
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # If true labels available, plot them too
        if self.y is not None:
            scatter2 = ax2.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                                  c=self.y, cmap='coolwarm', alpha=0.6)
            ax2.set_xlabel('First Principal Component')
            ax2.set_ylabel('Second Principal Component')
            ax2.set_title('True Labels Visualization (PCA)')
            plt.colorbar(scatter2, ax=ax2, label='Label (0=Phish, 1=Legit)')
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_pca_visualization.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 3D visualization if enough samples
        if self.X.shape[0] > 100 and self.X.shape[1] >= 3:
            try:
                from mpl_toolkits.mplot3d import Axes3D
                pca_3d = PCA(n_components=3)
                pca_result_3d = pca_3d.fit_transform(self.X)
                
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(pca_result_3d[:, 0], pca_result_3d[:, 1], 
                                   pca_result_3d[:, 2], c=self.labels, 
                                   cmap='viridis', alpha=0.6)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')
                ax.set_title('3D Clusters Visualization')
                plt.colorbar(scatter, label='Cluster')
                plt.savefig('visualizations/cluster_3d_visualization.png', dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not create 3D plot: {e}")
        
        print("✓ Cluster visualizations saved!")
    
    def save_clustering_model(self, model_name='kmeans_clustering.pkl'):
        """Save the clustering model"""
        os.makedirs('models', exist_ok=True)
        
        model_path = f'models/{model_name}'
        joblib.dump(self.kmeans, model_path)
        print(f"✓ Clustering model saved to {model_path}")
        return model_path
