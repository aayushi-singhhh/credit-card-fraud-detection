# 🚀 Streamlit Cloud Deployment Guide

## Step-by-Step Deployment with DATASET_URL

### Step 1: Upload Your Dataset to Cloud Storage

#### Option A: Google Drive (Easiest - Recommended)

1. **Upload to Google Drive:**
   - Go to [drive.google.com](https://drive.google.com)
   - Upload your `creditcard.csv` file
   - Right-click on the uploaded file → "Get link"
   - Change permissions to "Anyone with the link can view"

2. **Get Direct Download URL:**
   - Copy the shareable link (looks like): 
     `https://drive.google.com/file/d/1ABC123DEF456GHI789/view?usp=sharing`
   - Convert it to direct download URL:
     `https://drive.google.com/uc?export=download&id=1ABC123DEF456GHI789`
   - The ID is the part between `/d/` and `/view`

#### Option B: Dropbox

1. Upload `creditcard.csv` to Dropbox
2. Get shareable link
3. Change `?dl=0` to `?dl=1` at the end

#### Option C: GitHub Release (for files < 100MB)

If you have a smaller version of the dataset:
1. Create a GitHub release
2. Attach the CSV file
3. Use the download URL from the release

### Step 2: Deploy to Streamlit Cloud

1. **Push Code to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account if not already connected
   - Select your repository: `credit-card-fraud-detection`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add Environment Variables (Secrets):**
   - After deployment starts, click "⚙️ Settings" 
   - Go to "Secrets" tab
   - Add your dataset URL:
   ```toml
   DATASET_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
   ```

### Step 3: Test Your Deployment

1. **Wait for deployment** (usually 2-5 minutes)
2. **Visit your app** at: `https://your-app-name.streamlitapp.com`
3. **Check the logs** if there are any issues

### Step 4: Troubleshooting

#### Common Issues:

**Issue: "Dataset not found"**
- Check that DATASET_URL is correctly set in secrets
- Verify the Google Drive link is public
- Test the URL in your browser

**Issue: "Permission denied"**
- Make sure Google Drive file is set to "Anyone with link can view"
- For Dropbox, ensure the link ends with `?dl=1`

**Issue: "Memory error"**
- Streamlit Cloud has memory limits
- Consider using a smaller sample of your dataset
- The app will fall back to synthetic data if needed

### Step 5: Using Alternative Storage

#### AWS S3 (Professional Option)
If you want to use AWS S3:

1. Upload to S3:
   ```bash
   aws s3 cp creditcard.csv s3://your-bucket-name/
   aws s3api put-object-acl --bucket your-bucket-name --key creditcard.csv --acl public-read
   ```

2. Add to Streamlit secrets:
   ```toml
   S3_BUCKET_NAME = "your-bucket-name"
   S3_FILE_KEY = "creditcard.csv"
   ```

#### Google Cloud Storage
1. Upload to GCS bucket
2. Make it publicly accessible
3. Add to secrets:
   ```toml
   GCS_BUCKET_NAME = "your-bucket"
   GCS_FILE_NAME = "creditcard.csv"
   ```

### Step 6: Monitor Your App

- **View logs:** Check Streamlit Cloud dashboard for any errors
- **Update secrets:** You can modify environment variables anytime
- **Redeploy:** Push new commits to auto-redeploy

## 🎯 Quick Commands

### Generate sample data for testing:
```bash
python3 generate_sample_data.py --samples 5000
```

### Test locally with environment variable:
```bash
export DATASET_URL="your_url_here"
python3 -m streamlit run app.py
```

### Deploy script:
```bash
./deploy.sh
# Choose option 1 for Streamlit Cloud
```

## 📱 Your Live App

Once deployed, your fraud detection app will be live at:
`https://your-chosen-name.streamlitapp.com`

## 🔒 Security Notes

- Never commit secrets to GitHub
- Use environment variables for all sensitive data
- Regularly rotate access keys if using cloud storage APIs
- Monitor usage if using paid storage services
