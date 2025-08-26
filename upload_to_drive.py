#!/usr/bin/env python3
"""
Simple script to upload dataset to Google Drive and get shareable URL
Requires: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    """Authenticate and return Google Drive service"""
    creds = None
    
    # Token file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def upload_file_to_drive(service, file_path):
    """Upload file to Google Drive and return shareable URL"""
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': []  # Upload to root directory
    }
    
    media = MediaFileUpload(file_path, resumable=True)
    
    print(f"📤 Uploading {file_path} to Google Drive...")
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    file_id = file.get('id')
    
    # Make the file publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    
    service.permissions().create(
        fileId=file_id,
        body=permission
    ).execute()
    
    # Get the direct download URL
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"✅ File uploaded successfully!")
    print(f"📋 File ID: {file_id}")
    print(f"🔗 Direct download URL: {direct_url}")
    print(f"🌐 Shareable URL: https://drive.google.com/file/d/{file_id}/view")
    
    return direct_url

def main():
    """Main function"""
    print("🚀 Google Drive Dataset Uploader")
    print("=" * 40)
    
    # Check if dataset file exists
    dataset_file = 'creditcard.csv'
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file '{dataset_file}' not found!")
        print("Please ensure the file is in the current directory.")
        return
    
    # Check if credentials file exists
    if not os.path.exists('credentials.json'):
        print("❌ Google API credentials file 'credentials.json' not found!")
        print("📋 To get credentials:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable Google Drive API")
        print("4. Create credentials (OAuth 2.0)")
        print("5. Download and save as 'credentials.json'")
        return
    
    try:
        # Authenticate and upload
        service = authenticate_google_drive()
        direct_url = upload_file_to_drive(service, dataset_file)
        
        print("\n🎉 Upload completed!")
        print(f"💡 Set this environment variable for deployment:")
        print(f"   DATASET_URL='{direct_url}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your credentials and try again.")

if __name__ == "__main__":
    main()
