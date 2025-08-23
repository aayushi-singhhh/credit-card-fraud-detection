💳 Credit Card Fraud Detection
📌 Project Overview

This project builds a machine learning model to detect fraudulent credit card transactions. Fraudulent cases are highly rare (<1% of total), making this a classic imbalanced classification problem. The project applies preprocessing, resampling, and ML models to improve fraud detection performance.

📂 Dataset

Source: Kaggle Credit Card Fraud Detection

Records: 284,807 transactions

Features:

V1–V28 → anonymized PCA features

Time → seconds from first transaction

Amount → transaction amount

Class → target (0 = genuine, 1 = fraud)

⚙️ Project Workflow

Exploratory Data Analysis (EDA)

Distribution of genuine vs fraud transactions

Visualizations of transaction amounts and time trends

Data Preprocessing

Normalization of Amount and Time

Dropping redundant columns

Train-test split with stratification

Imbalance Handling

Random undersampling

Oversampling with SMOTE

Class weight adjustments

Model Training

Logistic Regression (baseline)

Random Forest Classifier

XGBoost

Model Evaluation

Precision, Recall, F1-score

Confusion Matrix

ROC-AUC Curve

🚀 Tech Stack

Python (NumPy, Pandas, Matplotlib, Seaborn)

Scikit-learn (Logistic Regression, Random Forest)

Imbalanced-learn (SMOTE)

XGBoost

📊 Results

Logistic Regression → good baseline, interpretable

Random Forest → higher recall for fraud cases

XGBoost → best overall performance on imbalanced dataset

📁 Repository Structure
📂 Credit-Card-Fraud-Detection
 ├── creditcard.csv          # Dataset (not included due to size)
 ├── main.ipynb              # Jupyter Notebook with code
 ├── requirements.txt        # Dependencies
 └── README.md               # Project documentation

📝 Future Work

Hyperparameter tuning with GridSearchCV

Deploy model as a REST API (Flask/FastAPI)

Streamlit dashboard for real-time fraud detection