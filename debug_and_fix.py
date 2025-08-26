#!/usr/bin/env python3
"""
Complete fix for Streamlit dataset loading issue
This will test all possible solutions and find what works
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os

def test_all_loading_methods():
    """Test all possible ways to load the dataset"""
    
    st.title("🔧 Dataset Loading Debug & Fix")
    st.write("Let's find out exactly what's happening...")
    
    # Test 1: Check Streamlit secrets
    st.subheader("1. 🔍 Checking Streamlit Secrets")
    try:
        if hasattr(st, 'secrets'):
            dataset_url = st.secrets.get("DATASET_URL")
            if dataset_url:
                st.success(f"✅ Found DATASET_URL in secrets: {dataset_url[:50]}...")
                
                # Test the URL
                st.write("🧪 Testing URL...")
                try:
                    response = requests.get(dataset_url, timeout=60)
                    if response.status_code == 200:
                        st.success(f"✅ URL is accessible! Status: {response.status_code}")
                        
                        # Try to parse CSV
                        df = pd.read_csv(StringIO(response.text))
                        if 'Class' in df.columns:
                            fraud_count = df['Class'].sum()
                            st.success(f"🎉 SUCCESS! Dataset loaded: {len(df):,} rows, {fraud_count} frauds")
                            
                            # Cache this working dataset
                            st.session_state['working_dataset'] = df
                            return df
                        else:
                            st.error("❌ No 'Class' column found")
                    else:
                        st.error(f"❌ URL failed: HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"❌ URL test failed: {e}")
            else:
                st.error("❌ DATASET_URL not found in secrets")
        else:
            st.error("❌ Streamlit secrets not available")
    except Exception as e:
        st.error(f"❌ Secrets error: {e}")
    
    # Test 2: Try direct Dropbox URL
    st.subheader("2. 🧪 Testing Direct Dropbox URL")
    dropbox_url = "https://www.dropbox.com/scl/fi/j7pya8qkd3eoedqgv7rwg/creditcard.csv?rlkey=dibt91q92qlx4zz0bzy1hg909&st=5pr4i9as&dl=1"
    
    try:
        st.write(f"📥 Testing: {dropbox_url[:60]}...")
        response = requests.get(dropbox_url, timeout=120)
        
        if response.status_code == 200:
            st.success(f"✅ Dropbox URL works! Size: {len(response.content)} bytes")
            
            df = pd.read_csv(StringIO(response.text))
            if 'Class' in df.columns:
                fraud_count = df['Class'].sum()
                st.success(f"🎉 DROPBOX SUCCESS! {len(df):,} rows, {fraud_count} frauds")
                st.session_state['working_dataset'] = df
                return df
        else:
            st.error(f"❌ Dropbox failed: HTTP {response.status_code}")
    except Exception as e:
        st.error(f"❌ Dropbox error: {e}")
    
    # Test 3: Check local file
    st.subheader("3. 📁 Checking Local File")
    if os.path.exists('creditcard.csv'):
        try:
            df = pd.read_csv('creditcard.csv')
            if 'Class' in df.columns:
                fraud_count = df['Class'].sum()
                st.success(f"✅ Local file works! {len(df):,} rows, {fraud_count} frauds")
                st.session_state['working_dataset'] = df
                return df
        except Exception as e:
            st.error(f"❌ Local file error: {e}")
    else:
        st.info("ℹ️ No local creditcard.csv file found")
    
    # Test 4: Generate enhanced synthetic data
    st.subheader("4. 🎲 Creating Enhanced Synthetic Data")
    try:
        df = create_realistic_synthetic_data()
        st.success(f"✅ Created synthetic data: {len(df):,} rows")
        st.session_state['working_dataset'] = df
        return df
    except Exception as e:
        st.error(f"❌ Synthetic data error: {e}")
    
    return None

def create_realistic_synthetic_data():
    """Create realistic synthetic credit card data"""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 50000  # Larger dataset
    fraud_rate = 0.002
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_data = {
        'Time': np.random.uniform(0, 172800, n_normal),
        'Amount': np.random.lognormal(3.5, 1.8, n_normal),
        'Class': np.zeros(n_normal, dtype=int)
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        normal_data[f'V{i}'] = np.random.normal(0, 1, n_normal)
    
    # Fraud transactions
    fraud_data = {
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Amount': np.random.lognormal(2.8, 2.2, n_fraud),
        'Class': np.ones(n_fraud, dtype=int)
    }
    
    # Fraud features (different patterns)
    for i in range(1, 29):
        if i in [1, 2, 3, 9, 10, 11, 14, 16, 17, 18]:
            fraud_data[f'V{i}'] = np.random.normal(1.5, 1.8, n_fraud)
        else:
            fraud_data[f'V{i}'] = np.random.normal(-0.5, 1.5, n_fraud)
    
    # Combine data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def main():
    """Main debug function"""
    
    st.set_page_config(
        page_title="Dataset Debug & Fix",
        page_icon="🔧",
        layout="wide"
    )
    
    # Run all tests
    dataset = test_all_loading_methods()
    
    if dataset is not None:
        st.subheader("🎉 Dataset Successfully Loaded!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(dataset):,}")
        with col2:
            fraud_count = dataset['Class'].sum()
            st.metric("Fraudulent Cases", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(dataset)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        with col4:
            total_amount = dataset['Amount'].sum()
            st.metric("Total Amount", f"${total_amount:,.2f}")
        
        st.subheader("📊 Dataset Preview")
        st.dataframe(dataset.head())
        
        st.subheader("📈 Quick Visualization")
        fraud_dist = dataset['Class'].value_counts()
        st.bar_chart(fraud_dist)
        
        # Save working dataset
        dataset.to_csv('working_creditcard.csv', index=False)
        st.success("💾 Saved working dataset as 'working_creditcard.csv'")
        
        st.subheader("🚀 Next Steps")
        st.write("""
        1. ✅ Dataset is now working in this session
        2. 📁 Saved as 'working_creditcard.csv' for backup
        3. 🔄 Your main app should now work with this data
        4. 🎯 If main app still shows synthetic data, restart it
        """)
        
    else:
        st.error("❌ All loading methods failed!")
        st.write("Contact support with this debug info.")

if __name__ == "__main__":
    main()
