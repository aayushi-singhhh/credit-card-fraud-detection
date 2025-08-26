# 💳 Credit Card Fraud Detection System

A comprehensive machine learning web application for detecting fraudulent credit card transactions in real-time.

## 🌟 Features

- **Interactive Web Interface**: Built with Streamlit for easy use
- **Real-time Predictions**: Instant fraud detection analysis
- **Data Visualization**: Comprehensive charts and graphs
- **Model Performance**: Detailed metrics and evaluation
- **Multiple Input Methods**: Manual input, random samples, or CSV upload
- **Risk Assessment**: Color-coded risk levels and probability scores

## 🚀 Quick Start

### ⚠️ **IMPORTANT: Dataset Required**
Before running the app, you need to download the dataset. See [`DATASET_SETUP.md`](DATASET_SETUP.md) for detailed instructions.

**Quick Download:**
1. Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in the project root directory

### Method 1: Using the Launcher Script
```bash
./run_app.sh
```

### Method 2: Manual Setup
1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Run the Streamlit app:
```bash
python3 -m streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## 🌐 Deployment

### Quick Deployment
```bash
./deploy.sh
```

### Deployment Options
1. **Streamlit Cloud** (Free, Recommended)
2. **Heroku** (Free tier available)
3. **Docker** (Self-hosted)
4. **Google Cloud Run** (Pay-per-use)
5. **Railway** (Free tier)

### Dataset for Deployment
Since the dataset is too large for GitHub, you have several options:
- Upload to **Google Drive** and use public link
- Store in **AWS S3** with public access
- Use **Dropbox** with direct download link
- Deploy with **synthetic data** (demo mode)

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed instructions.

## 🔧 Environment Variables

For deployment, set these environment variables:

```bash
# Option 1: Direct URL
DATASET_URL="https://your-storage-url/creditcard.csv"

# Option 2: AWS S3
S3_BUCKET_NAME="your-bucket"
S3_FILE_KEY="creditcard.csv"
AWS_ACCESS_KEY_ID="your-key"
AWS_SECRET_ACCESS_KEY="your-secret"
```

## � Dataset

The application uses the Credit Card Fraud Detection dataset containing:
- **284,807** total transactions
- **492** fraudulent transactions (0.17% fraud rate)
- **30** anonymized features (V1-V28, Time, Amount)
- Binary classification target (0=Normal, 1=Fraud)

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Local web server

## 📱 App Sections

### 🏠 Home
- Project overview and quick statistics
- Dataset information
- Key features summary

### 📊 Data Analysis
- Dataset exploration and visualization
- Class distribution analysis
- Transaction amount patterns
- Feature correlation heatmaps

### 🤖 Make Prediction
- **Manual Input**: Enter transaction details manually
- **Random Sample**: Test with random data from the dataset
- **CSV Upload**: Batch prediction for multiple transactions
- Real-time fraud probability scoring
- Risk level assessment

### 📈 Model Performance
- Accuracy, Precision, Recall, F1-Score metrics
- ROC-AUC score and curve visualization
- Confusion matrix analysis
- Feature importance rankings

### ℹ️ About
- Technical details and documentation
- Security and privacy information
- Future enhancement plans

## 🎯 Model Performance

- **Accuracy**: >99%
- **Precision**: High fraud detection precision
- **Recall**: Effective fraud identification
- **ROC-AUC**: Excellent discrimination capability

## 🔒 Security & Privacy

- All features are anonymized (PCA transformed)
- No personal information is stored or logged
- Model predictions are processed locally
- Privacy-compliant design

## � Requirements

- Python 3.7+
- All packages listed in `requirements.txt`
- Credit card dataset (`creditcard.csv`)

## 🚧 Future Enhancements

- [ ] Real-time streaming data processing
- [ ] Additional ML algorithms (Random Forest, XGBoost)
- [ ] Advanced feature engineering
- [ ] Model interpretability tools (SHAP, LIME)
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Cloud deployment options

## 📞 Support

For questions, issues, or contributions, please contact the development team.

**Note**: This application is designed for educational and demonstration purposes. For production deployment, additional security measures and model validation would be required.

## 📄 License

This project is for educational purposes. Please ensure you have the rights to use the dataset and comply with all applicable regulations when using this code.
