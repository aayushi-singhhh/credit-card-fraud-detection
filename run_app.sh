#!/bin/bash

# Credit Card Fraud Detection Streamlit App Launcher

echo "🚀 Starting Credit Card Fraud Detection App..."
echo "📋 Installing required packages..."

# Install required packages
pip3 install -r requirements.txt

echo "✅ Dependencies installed successfully!"
echo "🌐 Launching Streamlit app..."

# Launch the Streamlit app
python3 -m streamlit run app.py

echo "🔗 Open your browser and go to: http://localhost:8501"
