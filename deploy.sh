#!/bin/bash

# 🚀 Quick Deployment Script for Credit Card Fraud Detection App

echo "🚀 Credit Card Fraud Detection - Deployment Helper"
echo "=================================================="

# Check if dataset URL is provided
if [ -z "$DATASET_URL" ]; then
    echo "⚠️  Warning: DATASET_URL environment variable not set"
    echo "   The app will use synthetic data for demo purposes"
    echo ""
fi

# Function to deploy to different platforms
deploy_streamlit_cloud() {
    echo "📱 Deploying to Streamlit Cloud..."
    echo "1. Push your code to GitHub"
    echo "2. Go to share.streamlit.io"
    echo "3. Connect your GitHub repo"
    echo "4. Add DATASET_URL in app secrets"
    echo "5. Deploy!"
}

deploy_heroku() {
    echo "🔧 Deploying to Heroku..."
    
    # Check if Heroku CLI is installed
    if ! command -v heroku &> /dev/null; then
        echo "❌ Heroku CLI not found. Install it first:"
        echo "   https://devcenter.heroku.com/articles/heroku-cli"
        return 1
    fi
    
    echo "Creating Heroku app..."
    read -p "Enter app name: " app_name
    
    heroku create $app_name
    
    if [ ! -z "$DATASET_URL" ]; then
        echo "Setting dataset URL..."
        heroku config:set DATASET_URL="$DATASET_URL" --app $app_name
    fi
    
    echo "Deploying..."
    git push heroku main
    
    echo "✅ Deployed to: https://$app_name.herokuapp.com"
}

deploy_docker() {
    echo "🐳 Building Docker image..."
    
    # Build the image
    docker build -t fraud-detection-app .
    
    echo "🚀 Starting container..."
    
    # Run with environment variables
    if [ ! -z "$DATASET_URL" ]; then
        docker run -p 8501:8501 -e DATASET_URL="$DATASET_URL" fraud-detection-app
    else
        docker run -p 8501:8501 fraud-detection-app
    fi
}

# Main menu
echo "Choose deployment option:"
echo "1) Streamlit Cloud (Recommended)"
echo "2) Heroku"
echo "3) Docker (Local)"
echo "4) Show deployment guide"
echo "5) Exit"

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        deploy_streamlit_cloud
        ;;
    2)
        deploy_heroku
        ;;
    3)
        deploy_docker
        ;;
    4)
        echo "📖 Opening deployment guide..."
        if command -v open &> /dev/null; then
            open DEPLOYMENT.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open DEPLOYMENT.md
        else
            echo "Please read DEPLOYMENT.md for detailed instructions"
        fi
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment process completed!"
echo "📖 For detailed instructions, see DEPLOYMENT.md"
