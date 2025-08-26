#!/usr/bin/env python3
"""
Helper script to convert Google Drive shareable URL to direct download URL
"""

import re
import sys

def convert_drive_url(share_url):
    """Convert Google Drive share URL to direct download URL"""
    
    # Extract file ID from various Google Drive URL formats
    patterns = [
        r'https://drive\.google\.com/file/d/([a-zA-Z0-9-_]+)',
        r'https://drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)',
        r'id=([a-zA-Z0-9-_]+)'
    ]
    
    file_id = None
    for pattern in patterns:
        match = re.search(pattern, share_url)
        if match:
            file_id = match.group(1)
            break
    
    if not file_id:
        return None
    
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def main():
    print("🔗 Google Drive URL Converter for Streamlit Deployment")
    print("=" * 55)
    
    if len(sys.argv) > 1:
        # URL provided as command line argument
        share_url = sys.argv[1]
    else:
        # Ask user for URL
        print("📋 Paste your Google Drive shareable URL:")
        share_url = input().strip()
    
    if not share_url:
        print("❌ No URL provided!")
        return
    
    # Convert the URL
    direct_url = convert_drive_url(share_url)
    
    if direct_url:
        print("\n✅ Conversion successful!")
        print(f"📥 Original URL: {share_url}")
        print(f"📤 Direct download URL: {direct_url}")
        print("\n📋 For Streamlit Cloud secrets, add:")
        print(f'DATASET_URL = "{direct_url}"')
        print("\n💡 Steps to use:")
        print("1. Copy the direct download URL above")
        print("2. Go to your Streamlit Cloud app settings")
        print("3. Add it to secrets as DATASET_URL")
        print("4. Redeploy your app")
    else:
        print("❌ Could not extract file ID from URL!")
        print("Please make sure you're using a valid Google Drive shareable URL")
        print("\n📝 Expected format:")
        print("https://drive.google.com/file/d/FILE_ID/view?usp=sharing")

if __name__ == "__main__":
    main()
