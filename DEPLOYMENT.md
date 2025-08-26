# Deployment Guide for Credit Card Fraud Detection App

## 🚀 Deployment Options

### **Option 1: Streamlit Cloud (Recommended)**

#### Step 1: Upload Dataset to Cloud Storage

**Option A: Google Drive**
1. Upload `creditcard.csv` to Google Drive
2. Make it publicly accessible
3. Get the direct download link:
   - Right-click → Get link → Anyone with the link
   - Change URL format from: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - To: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Option B: Dropbox**
1. Upload `creditcard.csv` to Dropbox
2. Get shareable link
3. Change `?dl=0` to `?dl=1` at the end

**Option C: AWS S3 (Most Professional)**
```bash
# Upload to S3
aws s3 cp creditcard.csv s3://your-bucket-name/
aws s3api put-object-acl --bucket your-bucket-name --key creditcard.csv --acl public-read
```

#### Step 2: Deploy to Streamlit Cloud
1. Push your code to GitHub (without the CSV file)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in the Streamlit Cloud dashboard:
   - Go to your app settings
   - Add environment variables/secrets
   - Example: `DATASET_URL = "your_direct_download_url"`

#### Step 3: Configure Environment Variables
In Streamlit Cloud secrets, add:
```toml
DATASET_URL = "https://your-cloud-storage-url/creditcard.csv"
```

### **Option 2: Heroku Deployment**

#### Step 1: Create Heroku Files

**Procfile:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt:**
```
python-3.9.18
```

#### Step 2: Deploy
```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set DATASET_URL="your_dataset_url"
git push heroku main
```

### **Option 3: Docker + Cloud Run/ECS**

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2: Deploy to Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-detection
gcloud run deploy --image gcr.io/PROJECT_ID/fraud-detection --platform managed
```

### **Option 4: Railway**
1. Connect GitHub repo to Railway
2. Add environment variable: `DATASET_URL`
3. Railway auto-deploys from GitHub

### **Option 5: Render**
1. Connect GitHub repo to Render
2. Add environment variable in Render dashboard
3. Auto-deploy from GitHub

## 🔧 **Dataset Storage Solutions**

### **Free Options:**
1. **Google Drive** (15GB free)
2. **Dropbox** (2GB free)
3. **GitHub LFS** (1GB free)
4. **OneDrive** (5GB free)

### **Professional Options:**
1. **AWS S3** (Pay per use)
2. **Google Cloud Storage** (Pay per use)
3. **Azure Blob Storage** (Pay per use)

## 📋 **Quick Setup Commands**

### For Google Drive:
```bash
# Set environment variable
export DATASET_URL="https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
```

### For AWS S3:
```bash
# Upload file
aws s3 cp creditcard.csv s3://your-bucket/
# Set environment variables
export S3_BUCKET_NAME="your-bucket"
export S3_FILE_KEY="creditcard.csv"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

## 🔐 **Security Best Practices**

1. **Never commit secrets** to GitHub
2. **Use environment variables** for all sensitive data
3. **Set up proper IAM roles** for cloud storage access
4. **Use HTTPS** for all data transfers
5. **Consider data encryption** for sensitive datasets

## 📱 **Mobile-Responsive Deployment**

The app is already mobile-responsive, but for better mobile experience:

1. Add to your Streamlit config:
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

2. Test on mobile devices after deployment

## 🚨 **Troubleshooting**

**Issue: "Dataset not found"**
- Check environment variable is set correctly
- Verify URL is accessible
- Check file permissions

**Issue: "Memory error"**
- Use data chunking for large files
- Consider data preprocessing/sampling
- Increase deployment memory limits

**Issue: "Slow loading"**
- Implement data caching
- Consider data compression
- Use CDN for data delivery
