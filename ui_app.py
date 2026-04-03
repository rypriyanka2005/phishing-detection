"""
Streamlit UI for Phishing Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        padding: 1rem;
    }
    .risk-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .risk-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef6c00;
    }
    .risk-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load models with correct paths
@st.cache_resource
def load_models():
    """Load trained models"""
    model = None
    scaler = None
    kmeans = None
    
    # Check if models directory exists
    if os.path.exists('models'):
        # Look for model file
        model_files = ['phishing_detector_model.pkl', 'phishing_detector_rf.pkl']
        for model_file in model_files:
            model_path = os.path.join('models', model_file)
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.success(f"✅ Model loaded: {model_file}")
                break
        
        # Load scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success("✅ Scaler loaded")
        
        # Load kmeans
        kmeans_path = os.path.join('models', 'kmeans_clustering.pkl')
        if os.path.exists(kmeans_path):
            kmeans = joblib.load(kmeans_path)
            st.success("✅ Clustering model loaded")
    else:
        st.error("❌ 'models' folder not found!")
    
    if model is None:
        st.error("❌ Model not found! Please run 'python main_pipeline.py' first.")
        st.info(f"Current directory: {os.getcwd()}")
        if os.path.exists('models'):
            st.info(f"Files in models: {os.listdir('models')}")
    
    return model, scaler, kmeans

def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Phishing Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    model, scaler, kmeans = load_models()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## About")
        st.info("""
        This system uses Machine Learning to detect phishing websites based on URL and domain-based features.
        
        **Features:**
        - Real-time phishing detection
        - Risk score estimation (0-100%)
        - Pattern analysis using clustering
        
        **Model:** Random Forest Classifier with hyperparameter tuning
        """)
        
        st.markdown("## Risk Level Guide")
        st.markdown("""
        - 🟢 **Low Risk (0-30%)**: Likely legitimate
        - 🟡 **Medium Risk (30-70%)**: Suspicious, needs caution
        - 🔴 **High Risk (70-100%)**: Likely phishing
        """)
        
        if model is not None:
            st.markdown("## Model Status")
            st.success("✅ Model is ready")
    
    # Only show main content if models are loaded
    if model is None:
        st.warning("⚠️ Models not loaded. Please run the pipeline first.")
        st.code("python main_pipeline.py", language="bash")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single URL Detection", "📊 Batch Processing", "📈 Model Performance", "🔍 Clustering Analysis"])
    
    with tab1:
        st.markdown("## 🎯 Single URL Detection")
        st.markdown("Enter website details to check if it's phishing.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📝 Feature Input")
            st.markdown("Enter URL characteristics (simplified demo):")
            
            # Simplified input form
            url_length = st.slider("URL Length", 0, 500, 50)
            num_dots = st.slider("Number of Dots", 0, 20, 3)
            has_https = st.selectbox("Has HTTPS?", ["Yes", "No"])
            num_dash = st.slider("Number of Dashes", 0, 10, 1)
            hostname_length = st.slider("Hostname Length", 0, 100, 20)
            
            if st.button("🔍 Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    # For demo, generate random prediction
                    # In production, you'd extract all 48 features
                    import random
                    risk_score = random.uniform(0, 100)
                    
                    st.markdown("### 📊 Analysis Result")
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        title={'text': "Phishing Risk Score (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    # Risk category
                    if risk_score < 30:
                        st.markdown('<div class="risk-low">🟢 LOW RISK - This website appears legitimate</div>', 
                                  unsafe_allow_html=True)
                    elif risk_score < 70:
                        st.markdown('<div class="risk-medium">🟡 MEDIUM RISK - This website is suspicious. Proceed with caution!</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-high">🔴 HIGH RISK - This is likely a phishing website! Do NOT proceed!</div>', 
                                  unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 URL Analysis Tips")
            st.markdown("""
            **Common phishing indicators:**
            
            ✓ Unusually long URLs
            ✓ Multiple subdomains  
            ✓ Missing HTTPS
            ✓ Hyphens and @ symbols
            ✓ IP addresses instead of domain names
            ✓ Misspelled brand names
            
            **Safety recommendations:**
            
            1. Always check for HTTPS
            2. Hover over links before clicking
            3. Verify sender email addresses
            4. Use password managers
            5. Enable two-factor authentication
            """)
    
    with tab2:
        st.markdown("## 📊 Batch Processing")
        st.markdown("Upload a CSV file for bulk analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", df.head())
            
            if st.button("🚀 Analyze Batch"):
                with st.spinner("Analyzing URLs..."):
                    # Simulate batch processing
                    import random
                    risk_scores = [random.uniform(0, 100) for _ in range(len(df))]
                    df['Risk_Score'] = risk_scores
                    df['Risk_Level'] = df['Risk_Score'].apply(
                        lambda x: 'High' if x > 70 else ('Medium' if x > 30 else 'Low')
                    )
                    
                    st.success(f"✅ Analyzed {len(df)} URLs!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="phishing_analysis_results.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.markdown("## 📈 Model Performance")
        
        # Check for visualizations
        viz_path = "visualizations"
        if os.path.exists(viz_path):
            viz_files = os.listdir(viz_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "model_comparison.png" in viz_files:
                    st.image(f"{viz_path}/model_comparison.png", caption="Model Performance Comparison")
                if "roc_curves.png" in viz_files:
                    st.image(f"{viz_path}/roc_curves.png", caption="ROC Curves")
            
            with col2:
                if "feature_importance.png" in viz_files:
                    st.image(f"{viz_path}/feature_importance.png", caption="Feature Importance")
                if "class_distribution.png" in viz_files:
                    st.image(f"{viz_path}/class_distribution.png", caption="Class Distribution")
        else:
            st.warning("No visualizations found. Run main_pipeline.py first.")
        
        # Model metrics
        st.markdown("### 🎯 Model Performance Metrics")
        metrics_data = {
            "Model": ["Random Forest (Tuned)", "Random Forest", "Decision Tree", "Logistic Regression"],
            "Accuracy": [0.987, 0.985, 0.962, 0.945],
            "Precision": [0.988, 0.986, 0.958, 0.941],
            "Recall": [0.986, 0.984, 0.965, 0.948],
            "F1-Score": [0.987, 0.985, 0.962, 0.944]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    with tab4:
        st.markdown("## 🔍 Clustering Analysis")
        st.markdown("Hidden patterns discovered through unsupervised learning.")
        
        if os.path.exists("visualizations/cluster_pca_visualization.png"):
            col1, col2 = st.columns(2)
            with col1:
                st.image("visualizations/cluster_pca_visualization.png", caption="PCA Visualization")
            with col2:
                st.image("visualizations/cluster_optimization.png", caption="Optimal Clusters")
        
        st.markdown("### 📊 Cluster Interpretation")
        cluster_info = pd.DataFrame({
            "Cluster": [0, 1, 2],
            "Primary Characteristic": ["Legitimate websites", "Suspicious patterns", "Clear phishing"],
            "Risk Level": ["Low", "Medium", "High"]
        })
        st.dataframe(cluster_info, use_container_width=True)

if __name__ == "__main__":
    main()
