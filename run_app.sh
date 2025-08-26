#!/bin/bash

# Credit Card Fraud Detection Streamlit App Launcher

echo "🚀 Starting Credit Card Fraud Detection App..."
echo "📋 Installing required packages..."

# Install required packages
pip install -r requirements.txt

echo "✅ Dependencies installed successfully!"
echo "🌐 Launching Streamlit app..."

# Launch the Streamlit app
streamlit run app.py

echo "🔗 Open your browser and go to: http://localhost:8501"
