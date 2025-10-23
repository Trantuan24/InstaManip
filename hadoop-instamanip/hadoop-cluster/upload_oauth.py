#!/usr/bin/env python3
"""
InstaManip Google Drive Backup Script
Uploads processed batch files from Hadoop to Google Drive using OAuth Client
"""

import os
import sys
import json
import pickle
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path

try:
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    print("❌ Google API libraries not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "google-api-python-client", "google-auth-oauthlib", "google-auth-httplib2", "--break-system-packages"], check=True)
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class GoogleDriveUploader:
    def __init__(self, credentials_file):
        """Initialize Google Drive service with OAuth credentials"""
        self.service = None
        self.authenticate(credentials_file)
    
    def authenticate(self, credentials_file):
        """Authenticate with Google Drive using OAuth"""
        creds = None
        token_file = 'token.pickle'
        
        # Load existing token
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("🔄 Refreshing expired token...")
                creds.refresh(Request())
            else:
                print("🔐 Starting OAuth authentication flow...")
                print("📋 A browser window will open for authentication")
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Successfully authenticated with Google Drive")
    
    def create_folder(self, name, parent_id=None):
        """Create a folder in Google Drive"""
        try:
            file_metadata = {
                'name': name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            print(f"📁 Created folder: {name} (ID: {folder_id})")
            return folder_id
        except Exception as e:
            print(f"❌ Failed to create folder {name}: {e}")
            return None
    
    def upload_file(self, file_path, filename=None, parent_id=None):
        """Upload a file to Google Drive"""
        try:
            if not filename:
                filename = os.path.basename(file_path)
            
            file_metadata = {'name': filename}
            if parent_id:
                file_metadata['parents'] = [parent_id]
            
            # Get file size for progress reporting
            file_size = os.path.getsize(file_path)
            print(f"📤 Uploading: {filename} ({file_size / 1024 / 1024:.1f} MB)")
            
            media = MediaFileUpload(file_path, resumable=True)
            file_obj = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()
            
            file_id = file_obj.get('id')
            file_link = file_obj.get('webViewLink')
            print(f"✅ Uploaded: {filename} (ID: {file_id})")
            return file_id, file_link
        except Exception as e:
            print(f"❌ Failed to upload {filename}: {e}")
            return None, None

def run_hdfs_command(command):
    """Run HDFS command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ HDFS command failed: {command}")
        print(f"❌ Error: {e.stderr}")
        return None

def get_batch_files_from_hdfs():
    """Get list of batch files from HDFS"""
    print("📋 Listing batch files from HDFS...")
    
    # List files in /datasets/final/ directory
    files_output = run_hdfs_command("hdfs dfs -ls /datasets/final/ | grep -E '\\.tar\\.gz$'")
    if not files_output:
        print("❌ No batch files found in HDFS /datasets/final/")
        return []
    
    batch_files = []
    for line in files_output.split('\n'):
        if '.tar.gz' in line:
            # Extract file path from hdfs ls output
            file_path = line.split()[-1]
            batch_files.append(file_path)
    
    print(f"📦 Found {len(batch_files)} batch files")
    return batch_files

def download_file_from_hdfs(hdfs_path, local_path):
    """Download file from HDFS to local filesystem"""
    try:
        cmd = f"hdfs dfs -get {hdfs_path} {local_path}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"📥 Downloaded: {os.path.basename(hdfs_path)}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to download: {hdfs_path}")
        return False

def main():
    print("🌐 InstaManip Batch Uploader (OAuth Version)")
    print("Purpose: Upload processed batches to Google Drive for cloud deployment\n")
    
    # Check for OAuth credentials file
    credentials_file = "client_secrets.json"
    if not os.path.exists(credentials_file):
        print(f"❌ Error: {credentials_file} not found!")
        print("📋 Please follow these steps:")
        print("1. Go to https://console.developers.google.com/")
        print("2. Create project, enable Drive API")
        print("3. Create OAuth 2.0 Client ID credentials (Desktop Application)")
        print("4. Download as client_secrets.json to current directory")
        return 1
    
    try:
        print("🚀 Starting Google Drive Upload...")
        
        # Initialize uploader
        uploader = GoogleDriveUploader(credentials_file)
        
        # Create main folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"InstaManip_ProcessedData_{timestamp}"
        main_folder_id = uploader.create_folder(folder_name)
        
        if not main_folder_id:
            print("❌ Failed to create main folder")
            return 1
        
        # Get batch files from HDFS
        batch_files = get_batch_files_from_hdfs()
        if not batch_files:
            print("❌ No batch files to upload")
            return 1
        
        # Create temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary directory: {temp_dir}")
            
            uploaded_files = {}
            
            # Download and upload each batch file
            for hdfs_file in batch_files:
                filename = os.path.basename(hdfs_file)
                local_file = os.path.join(temp_dir, filename)
                
                # Download from HDFS
                if download_file_from_hdfs(hdfs_file, local_file):
                    # Upload to Google Drive
                    file_id, file_link = uploader.upload_file(local_file, parent_id=main_folder_id)
                    if file_id:
                        uploaded_files[filename] = {
                            "file_id": file_id,
                            "file_link": file_link,
                            "hdfs_path": hdfs_file
                        }
            
            # Try to upload batch_index.json if it exists
            index_files = ["/datasets/final/batch_index.json", "/datasets/batch_index.json"]
            for index_path in index_files:
                check_cmd = f"hdfs dfs -test -e {index_path}"
                if subprocess.run(check_cmd, shell=True, capture_output=True).returncode == 0:
                    local_index = os.path.join(temp_dir, "batch_index.json")
                    if download_file_from_hdfs(index_path, local_index):
                        # Update index with Google Drive information
                        try:
                            with open(local_index, 'r') as f:
                                index_data = json.load(f)
                            
                            # Add Google Drive info
                            index_data["google_drive"] = {
                                "folder_id": main_folder_id,
                                "folder_name": folder_name,
                                "uploaded_files": uploaded_files,
                                "upload_timestamp": datetime.now().isoformat()
                            }
                            
                            # Save updated index
                            with open(local_index, 'w') as f:
                                json.dump(index_data, f, indent=2)
                            
                            # Upload updated index
                            uploader.upload_file(local_index, "batch_index.json", main_folder_id)
                            
                        except Exception as e:
                            print(f"⚠️ Warning: Could not update batch index: {e}")
                            # Upload original index anyway
                            uploader.upload_file(local_index, "batch_index.json", main_folder_id)
                    break
            
            # Try to upload batch_creation_report.json if it exists
            report_files = ["/datasets/final/batch_creation_report.json", "/datasets/batch_creation_report.json"]
            for report_path in report_files:
                check_cmd = f"hdfs dfs -test -e {report_path}"
                if subprocess.run(check_cmd, shell=True, capture_output=True).returncode == 0:
                    local_report = os.path.join(temp_dir, "batch_creation_report.json")
                    if download_file_from_hdfs(report_path, local_report):
                        uploader.upload_file(local_report, "batch_creation_report.json", main_folder_id)
                    break
        
        # Final summary
        print(f"\n🎉 Upload Complete!")
        print(f"📁 Folder: {folder_name}")
        print(f"📤 Files uploaded: {len(uploaded_files)}")
        print(f"🔗 Google Drive Folder: https://drive.google.com/drive/folders/{main_folder_id}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)