import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import warnings
import os
import requests
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fraud-result {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .normal-result {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Fraud Detection System</h1>
    <p>Advanced Machine Learning Model for Real-time Fraud Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["🏠 Home", "📊 Data Analysis", "🤖 Make Prediction", "📈 Model Performance", "ℹ️ About"])

# Load and cache data
@st.cache_data
def load_data():
    """Load the credit card dataset from various sources"""
    
    # Try multiple sources in order of preference
    sources_to_try = [
        ("Streamlit Secrets", lambda: st.secrets.get("DATASET_URL") if hasattr(st, 'secrets') else None),
        ("Environment Variable", lambda: os.environ.get("DATASET_URL")),
        ("Local File", lambda: "creditcard.csv" if os.path.exists("creditcard.csv") else None)
    ]
    
    for source_name, get_source in sources_to_try:
        try:
            source = get_source()
            if source:
                st.info(f"🔍 Trying to load from {source_name}...")
                
                if source_name == "Local File":
                    # Local file
                    df = pd.read_csv(source)
                    st.success(f"✅ Dataset loaded from {source_name}")
                else:
                    # URL source
                    st.info(f"📥 Downloading from URL: {str(source)[:50]}...")
                    
                    response = requests.get(source, timeout=120)
                    if response.status_code == 200:
                        df = pd.read_csv(StringIO(response.text))
                        st.success(f"✅ Dataset loaded from {source_name}")
                    else:
                        st.error(f"❌ HTTP {response.status_code} from {source_name}")
                        continue
                
                # Validate dataset
                if 'Class' in df.columns and len(df) > 1000:
                    fraud_count = df['Class'].sum()
                    st.info(f"📊 Loaded {len(df):,} transactions with {fraud_count:,} fraudulent cases")
                    return df
                else:
                    st.error(f"❌ Invalid dataset from {source_name}")
                    continue
                    
        except Exception as e:
            st.error(f"❌ Error loading from {source_name}: {str(e)}")
            continue
    
    # All sources failed - generate synthetic data
    st.warning("⚠️ All data sources failed. Generating synthetic data for demo...")
    return generate_synthetic_fallback_data()

@st.cache_data
def generate_synthetic_fallback_data():
    """Generate synthetic data as fallback"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 5000
    n_fraud = int(n_samples * 0.002)
    
    data = {}
    data['Time'] = np.random.uniform(0, 172800, n_samples)
    data['Amount'] = np.random.lognormal(3, 1.5, n_samples)
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Add fraud cases
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['Class'] = np.zeros(n_samples, dtype=int)
    data['Class'][fraud_indices] = 1
    
    return pd.DataFrame(data)

# Train and cache model
@st.cache_resource
def train_model():
    """Train the fraud detection model"""
    data = load_data()
    if data is None:
        return None, None, None, None
    
    # Prepare features and target
    X = data.drop(columns='Class', axis=1)
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=2, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, (X_test_scaled, y_test), data

# Prediction function
def predict_fraud(transaction_features, model, scaler):
    """Predict if a transaction is fraudulent"""
    # Scale the features
    scaled_features = scaler.transform([transaction_features])
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    
    return prediction, probability

# Main app logic
if page == "🏠 Home":
    st.markdown("## Welcome to the Credit Card Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application uses advanced machine learning algorithms to detect potentially fraudulent credit card transactions in real-time.
        
        ### 🔍 Key Features
        - **Real-time Prediction**: Instantly analyze transaction data
        - **High Accuracy**: Uses logistic regression with feature scaling
        - **Comprehensive Analysis**: Detailed data exploration and visualization
        - **Interactive Interface**: User-friendly web application
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Dataset Information
        The model is trained on a comprehensive credit card transaction dataset containing:
        - **284,807** total transactions
        - **492** fraudulent transactions (0.17%)
        - **30** anonymized features (V1-V28, Time, Amount)
        - **1** target variable (Class: 0=Normal, 1=Fraud)
        
        ### 🚀 Getting Started
        1. Explore the **Data Analysis** section
        2. Try the **Make Prediction** feature
        3. Check **Model Performance** metrics
        """)
    
    # Load model and data for quick stats
    model, scaler, test_data, data = train_model()
    if data is not None:
        st.markdown("### 📈 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(data):,}")
        with col2:
            fraud_count = data['Class'].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(data)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        with col4:
            st.metric("Features", len(data.columns) - 1)

elif page == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis & Visualization")
    
    data = load_data()
    if data is not None:
        # Dataset overview
        st.markdown("### Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", data.shape)
            st.write("**Missing Values:**", data.isnull().sum().sum())
            
        with col2:
            st.write("**Class Distribution:**")
            class_dist = data['Class'].value_counts()
            st.write(f"Normal Transactions: {class_dist[0]:,}")
            st.write(f"Fraudulent Transactions: {class_dist[1]:,}")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        # Class distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(values=class_dist.values, 
                           names=['Normal', 'Fraud'], 
                           title="Transaction Class Distribution",
                           color_discrete_sequence=['#2E8B57', '#DC143C'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(x=['Normal', 'Fraud'], 
                           y=class_dist.values,
                           title="Transaction Count by Class",
                           color=['Normal', 'Fraud'],
                           color_discrete_sequence=['#2E8B57', '#DC143C'])
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Amount analysis
        st.markdown("### 💰 Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution by class
            fig_box = px.box(data, x='Class', y='Amount', 
                           title="Transaction Amount Distribution by Class",
                           labels={'Class': 'Transaction Class (0=Normal, 1=Fraud)'})
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Amount histogram
            fig_hist = px.histogram(data, x='Amount', color='Class', 
                                  title="Transaction Amount Histogram",
                                  nbins=50, marginal="rug")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Feature correlation
        st.markdown("### 🔗 Feature Correlations")
        if st.checkbox("Show Correlation Heatmap (Warning: May take time to load)"):
            corr_matrix = data.corr()
            fig_heatmap = px.imshow(corr_matrix, 
                                  title="Feature Correlation Matrix",
                                  color_continuous_scale='RdBu')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Statistical summary
        st.markdown("### 📋 Statistical Summary")
        if st.checkbox("Show detailed statistics"):
            st.write(data.describe())

elif page == "🤖 Make Prediction":
    st.markdown("## 🤖 Make Fraud Prediction")
    
    model, scaler, test_data, data = train_model()
    
    if model is not None:
        st.markdown("### Choose Input Method")
        input_method = st.radio("Select how to input transaction data:", 
                               ["Manual Input", "Random Sample", "Upload CSV"])
        
        if input_method == "Manual Input":
            st.markdown("#### Enter Transaction Details")
            st.info("💡 For simplicity, we'll use the most important features. In practice, all 30 features would be used.")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
                v1 = st.number_input("Feature V1", value=0.0, format="%.6f")
                v2 = st.number_input("Feature V2", value=0.0, format="%.6f")
                v3 = st.number_input("Feature V3", value=0.0, format="%.6f")
                v4 = st.number_input("Feature V4", value=0.0, format="%.6f")
                v5 = st.number_input("Feature V5", value=0.0, format="%.6f")
                v6 = st.number_input("Feature V6", value=0.0, format="%.6f")
                v7 = st.number_input("Feature V7", value=0.0, format="%.6f")
                v8 = st.number_input("Feature V8", value=0.0, format="%.6f")
                v9 = st.number_input("Feature V9", value=0.0, format="%.6f")
            
            with col2:
                v10 = st.number_input("Feature V10", value=0.0, format="%.6f")
                v11 = st.number_input("Feature V11", value=0.0, format="%.6f")
                v12 = st.number_input("Feature V12", value=0.0, format="%.6f")
                v13 = st.number_input("Feature V13", value=0.0, format="%.6f")
                v14 = st.number_input("Feature V14", value=0.0, format="%.6f")
                v15 = st.number_input("Feature V15", value=0.0, format="%.6f")
                v16 = st.number_input("Feature V16", value=0.0, format="%.6f")
                v17 = st.number_input("Feature V17", value=0.0, format="%.6f")
                v18 = st.number_input("Feature V18", value=0.0, format="%.6f")
                v19 = st.number_input("Feature V19", value=0.0, format="%.6f")
            
            with col3:
                v20 = st.number_input("Feature V20", value=0.0, format="%.6f")
                v21 = st.number_input("Feature V21", value=0.0, format="%.6f")
                v22 = st.number_input("Feature V22", value=0.0, format="%.6f")
                v23 = st.number_input("Feature V23", value=0.0, format="%.6f")
                v24 = st.number_input("Feature V24", value=0.0, format="%.6f")
                v25 = st.number_input("Feature V25", value=0.0, format="%.6f")
                v26 = st.number_input("Feature V26", value=0.0, format="%.6f")
                v27 = st.number_input("Feature V27", value=0.0, format="%.6f")
                v28 = st.number_input("Feature V28", value=0.0, format="%.6f")
                time = st.number_input("Time (seconds)", min_value=0.0, value=0.0)
            
            # Create feature array
            features = np.array([time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, 
                               v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount])
        
        elif input_method == "Random Sample":
            st.markdown("#### Random Sample from Test Data")
            if st.button("Generate Random Sample"):
                # Get random sample from test data
                X_test, y_test = test_data
                random_idx = np.random.randint(0, len(X_test))
                features = X_test[random_idx]
                actual_class = y_test[random_idx]
                
                st.write(f"**Sample Index:** {random_idx}")
                st.write(f"**Actual Class:** {'Fraud' if actual_class == 1 else 'Normal'}")
                
                # Display some key features
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Amount:** ${features[-1]:.2f}")
                with col2:
                    st.write(f"**Time:** {features[0]:.0f} seconds")
                with col3:
                    st.write(f"**V1:** {features[1]:.6f}")
            else:
                features = None
        
        elif input_method == "Upload CSV":
            st.markdown("#### Upload Transaction Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                upload_data = pd.read_csv(uploaded_file)
                st.write("Uploaded data preview:")
                st.write(upload_data.head())
                
                if st.button("Predict All Transactions"):
                    # Make predictions for all rows
                    predictions = []
                    probabilities = []
                    
                    for idx, row in upload_data.iterrows():
                        pred, prob = predict_fraud(row.values, model, scaler)
                        predictions.append(pred)
                        probabilities.append(prob)
                    
                    upload_data['Prediction'] = predictions
                    upload_data['Fraud_Probability'] = probabilities
                    upload_data['Risk_Level'] = upload_data['Fraud_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                    )
                    
                    st.write("Prediction Results:")
                    st.write(upload_data[['Prediction', 'Fraud_Probability', 'Risk_Level']])
                    
                    # Summary statistics
                    fraud_count = sum(predictions)
                    st.write(f"**Total Transactions:** {len(predictions)}")
                    st.write(f"**Predicted Fraudulent:** {fraud_count}")
                    st.write(f"**Fraud Rate:** {(fraud_count/len(predictions)*100):.2f}%")
                
                features = None
            else:
                features = None
        
        # Make prediction
        if input_method != "Upload CSV" and 'features' in locals() and features is not None:
            if st.button("🔍 Analyze Transaction", type="primary"):
                prediction, probability = predict_fraud(features, model, scaler)
                
                # Display result
                st.markdown("### 🎯 Prediction Result")
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-result fraud-result">
                        ⚠️ FRAUDULENT TRANSACTION DETECTED ⚠️<br>
                        Fraud Probability: {probability:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-result normal-result">
                        ✅ NORMAL TRANSACTION<br>
                        Fraud Probability: {probability:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk level
                st.markdown("### 📊 Risk Assessment")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if probability < 0.3:
                        risk_level = "Low"
                        risk_color = "green"
                    elif probability < 0.7:
                        risk_level = "Medium"
                        risk_color = "orange"
                    else:
                        risk_level = "High"
                        risk_color = "red"
                    
                    st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
                
                with col2:
                    st.metric("Fraud Probability", f"{probability:.2%}")
                
                with col3:
                    confidence = max(probability, 1-probability)
                    st.metric("Model Confidence", f"{confidence:.2%}")
                
                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fraud Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)

elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Metrics")
    
    model, scaler, test_data, data = train_model()
    
    if model is not None:
        X_test, y_test = test_data
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Display metrics
        st.markdown("### 🎯 Key Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        
        # Confusion Matrix
        st.markdown("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['Normal', 'Fraud'],
                          y=['Normal', 'Fraud'],
                          color_continuous_scale='Blues',
                          title="Confusion Matrix")
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=str(cm[i][j]),
                    showarrow=False,
                    font=dict(color="black", size=16)
                )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # ROC Curve
        st.markdown("### 📈 ROC Curve")
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                   line=dict(color='blue', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Random Classifier',
                                   line=dict(color='red', width=2, dash='dash')))
        
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Feature Importance (if available)
        st.markdown("### 🔍 Model Insights")
        
        if hasattr(model, 'coef_'):
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
            importance = abs(model.coef_[0])
            
            # Get top 10 most important features
            top_indices = importance.argsort()[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance[top_indices]
            
            fig_importance = px.bar(x=top_importance, y=top_features, 
                                  orientation='h',
                                  title='Top 10 Most Important Features',
                                  labels={'x': 'Importance', 'y': 'Features'})
            st.plotly_chart(fig_importance, use_container_width=True)

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        This Credit Card Fraud Detection System is a machine learning application designed to identify potentially fraudulent credit card transactions in real-time.
        
        ### 🔧 Technology Stack
        - **Frontend**: Streamlit
        - **Backend**: Python
        - **Machine Learning**: Scikit-learn
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        ### 📊 Model Details
        - **Algorithm**: Logistic Regression
        - **Preprocessing**: Standard Scaling
        - **Features**: 30 anonymized features
        - **Target**: Binary classification (Fraud/Normal)
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Dataset Information
        The model is trained on the Credit Card Fraud Detection dataset, which contains:
        - **284,807** total transactions
        - **492** fraudulent transactions (0.17%)
        - **30** features (V1-V28, Time, Amount)
        - Highly imbalanced dataset
        
        ### 🔒 Security & Privacy
        - All features are anonymized (PCA transformed)
        - No personal information is stored
        - Model predictions are not logged
        - Compliant with privacy regulations
        
        ### 🚀 Future Enhancements
        - Real-time streaming data processing
        - Additional ML algorithms comparison
        - Advanced feature engineering
        - Model interpretability tools
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📞 Contact & Support
    For questions, suggestions, or support, please contact the development team.
    
    **Disclaimer**: This application is for educational and demonstration purposes. 
    For production use, additional security measures and model validation would be required.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Credit Card Fraud Detection System | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
