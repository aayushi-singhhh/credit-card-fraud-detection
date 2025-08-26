"""
Cloud storage configuration for large dataset deployment
"""
import os
import pandas as pd
import streamlit as st
from urllib.request import urlretrieve
import boto3
from google.cloud import storage
import requests

class DatasetManager:
    """Manages dataset loading from various sources"""
    
    def __init__(self):
        self.dataset_cache = None
    
    @st.cache_data
    def load_from_url(_self, url):
        """Load dataset from a direct URL"""
        try:
            return pd.read_csv(url)
        except Exception as e:
            st.error(f"Error loading from URL: {e}")
            return None
    
    @st.cache_data
    def load_from_s3(_self, bucket_name, file_key, aws_access_key=None, aws_secret_key=None):
        """Load dataset from AWS S3"""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key or os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=aws_secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            
            # Download to temporary file
            temp_file = '/tmp/creditcard.csv'
            s3_client.download_file(bucket_name, file_key, temp_file)
            return pd.read_csv(temp_file)
        except Exception as e:
            st.error(f"Error loading from S3: {e}")
            return None
    
    @st.cache_data
    def load_from_gcs(_self, bucket_name, file_name, credentials_path=None):
        """Load dataset from Google Cloud Storage"""
        try:
            if credentials_path:
                client = storage.Client.from_service_account_json(credentials_path)
            else:
                client = storage.Client()
            
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_name)
            
            # Download to temporary file
            temp_file = '/tmp/creditcard.csv'
            blob.download_to_filename(temp_file)
            return pd.read_csv(temp_file)
        except Exception as e:
            st.error(f"Error loading from GCS: {e}")
            return None
    
    @st.cache_data
    def load_from_dropbox(_self, dropbox_url):
        """Load dataset from Dropbox direct link"""
        try:
            # Convert Dropbox share URL to direct download URL
            if 'dropbox.com' in dropbox_url and '?dl=0' in dropbox_url:
                direct_url = dropbox_url.replace('?dl=0', '?dl=1')
            else:
                direct_url = dropbox_url
            
            return pd.read_csv(direct_url)
        except Exception as e:
            st.error(f"Error loading from Dropbox: {e}")
            return None
    
    def load_dataset(self):
        """Load dataset from configured source"""
        if self.dataset_cache is not None:
            return self.dataset_cache
        
        # Try different sources in order of preference
        dataset_url = os.environ.get('DATASET_URL')
        s3_bucket = os.environ.get('S3_BUCKET_NAME')
        s3_key = os.environ.get('S3_FILE_KEY')
        gcs_bucket = os.environ.get('GCS_BUCKET_NAME')
        gcs_file = os.environ.get('GCS_FILE_NAME')
        dropbox_url = os.environ.get('DROPBOX_URL')
        
        # Try loading from different sources
        if dataset_url:
            st.info("📥 Loading dataset from URL...")
            self.dataset_cache = self.load_from_url(dataset_url)
        elif s3_bucket and s3_key:
            st.info("📥 Loading dataset from AWS S3...")
            self.dataset_cache = self.load_from_s3(s3_bucket, s3_key)
        elif gcs_bucket and gcs_file:
            st.info("📥 Loading dataset from Google Cloud Storage...")
            self.dataset_cache = self.load_from_gcs(gcs_bucket, gcs_file)
        elif dropbox_url:
            st.info("📥 Loading dataset from Dropbox...")
            self.dataset_cache = self.load_from_dropbox(dropbox_url)
        elif os.path.exists('creditcard.csv'):
            st.info("📥 Loading local dataset...")
            self.dataset_cache = pd.read_csv('creditcard.csv')
        else:
            st.warning("⚠️ No dataset source configured. Using synthetic data...")
            self.dataset_cache = self.generate_synthetic_data()
        
        return self.dataset_cache
    
    def generate_synthetic_data(self):
        """Generate synthetic data as fallback"""
        import numpy as np
        
        np.random.seed(42)
        n_samples = 10000
        n_fraud = int(n_samples * 0.002)
        
        # Generate data similar to your existing generator
        data = {}
        data['Time'] = np.random.uniform(0, 172800, n_samples)
        data['Amount'] = np.random.lognormal(3, 1.5, n_samples)
        
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        # Add fraud cases
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        data['Class'] = np.zeros(n_samples)
        data['Class'][fraud_indices] = 1
        
        return pd.DataFrame(data)
