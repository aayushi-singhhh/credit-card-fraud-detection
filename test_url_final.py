#!/usr/bin/env python3
"""
Test if your Dropbox URL actually works
"""

import pandas as pd
import requests
from io import StringIO

def test_dropbox_url():
    """Test the Dropbox URL that you provided"""
    
    # Your Dropbox URL (converted to direct download)
    url = "https://www.dropbox.com/scl/fi/j7pya8qkd3eoedqgv7rwg/creditcard.csv?rlkey=dibt91q92qlx4zz0bzy1hg909&st=5pr4i9as&dl=1"
    
    print("🧪 Testing your Dropbox URL...")
    print(f"📋 URL: {url[:80]}...")
    
    try:
        print("📥 Downloading dataset...")
        response = requests.get(url, timeout=120)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📏 Content Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            # Try to parse as CSV
            try:
                df = pd.read_csv(StringIO(response.text))
                print(f"✅ Successfully loaded CSV!")
                print(f"📊 Dataset shape: {df.shape}")
                print(f"📋 Columns: {list(df.columns)[:10]}")
                
                if 'Class' in df.columns:
                    fraud_count = df['Class'].sum()
                    total_count = len(df)
                    fraud_rate = (fraud_count / total_count) * 100
                    
                    print(f"🎉 PERFECT! This is the real dataset!")
                    print(f"📈 Total transactions: {total_count:,}")
                    print(f"🚨 Fraudulent cases: {fraud_count:,}")
                    print(f"📊 Fraud rate: {fraud_rate:.3f}%")
                    
                    # Save a local copy
                    df.to_csv('creditcard.csv', index=False)
                    print(f"💾 Saved local copy as 'creditcard.csv'")
                    
                    return True, url
                else:
                    print("❌ No 'Class' column found - not the right dataset")
                    return False, None
                    
            except Exception as e:
                print(f"❌ CSV parsing error: {e}")
                # Show first 500 chars of response
                print(f"📝 Response preview: {response.text[:500]}")
                return False, None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False, None
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False, None

def main():
    print("🔍 Final Dataset URL Test")
    print("=" * 30)
    
    success, working_url = test_dropbox_url()
    
    if success:
        print(f"\n🎉 SUCCESS! Your URL works perfectly!")
        print(f"\n📋 Working URL:")
        print(f"{working_url}")
        print(f"\n🚀 Next step: Update Streamlit Cloud secrets with this URL")
        print(f"✅ Local dataset also saved for testing!")
    else:
        print(f"\n😔 URL didn't work. Let me generate local data instead...")
        return False
    
    return True

if __name__ == "__main__":
    main()
