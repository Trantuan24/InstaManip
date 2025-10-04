#!/usr/bin/env python3
"""
Data Download & Preprocessing Pipeline
Tiền xử lý dữ liệu cho InstaManip training
"""

import os
import sys
import json
import tarfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import requests

def download_from_drive(file_id, filename):
    """Tải batch files từ Google Drive"""
    print(f"📥 Downloading {filename} từ Google Drive...")
    
    # Real Google Drive download URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded {filename} successfully")
        return filename
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None

def extract_batch_file(batch_path, extract_dir):
    """Giải nén các file tar.gz chứa groups đã được xử lý sẵn"""
    print(f"📂 Extracting {batch_path}...")
    
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with tarfile.open(batch_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        print(f"✅ Extracted {batch_path} to {extract_dir}")
        return extract_dir
        
    except Exception as e:
        print(f"❌ Error extracting {batch_path}: {e}")
        return None

def reorganize_data_structure(extracted_dir, output_dir="./data/processed"):
    """Tổ chức lại theo cấu trúc pipeline, mỗi group chứa source/target pairs và instruction tương ứng"""
    print("🔄 Reorganizing data structure...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    samples_data = []
    group_count = 0
    
    # Scan extracted directory for groups
    for item in os.listdir(extracted_dir):
        item_path = os.path.join(extracted_dir, item)
        
        if os.path.isdir(item_path):
            # This is a group folder
            group_id = f"group_{group_count:04d}"
            group_output_path = os.path.join(output_dir, group_id)
            
            # Copy group to output directory
            shutil.copytree(item_path, group_output_path, dirs_exist_ok=True)
            
            # Process images in group
            images = [f for f in os.listdir(group_output_path) if f.endswith('.jpg')]
            
            # Create sample pairs
            source_images = [img for img in images if img.endswith('_0.jpg')]
            
            for source_img in source_images:
                base_name = source_img.replace('_0.jpg', '')
                target_img = f"{base_name}_1.jpg"
                
                if target_img in images:
                    # Read instruction if available
                    instruction_file = os.path.join(group_output_path, 'instruction.txt')
                    if os.path.exists(instruction_file):
                        with open(instruction_file, 'r') as f:
                            instruction = f.read().strip()
                    else:
                        instruction = "Transform the image"  # Default instruction
                    
                    samples_data.append({
                        'group_id': group_id,
                        'sample_id': f"{group_count:04d}_{len(samples_data):04d}",
                        'source_path': os.path.join(group_output_path, source_img),
                        'target_path': os.path.join(group_output_path, target_img),
                        'instruction': instruction
                    })
            
            group_count += 1
    
    print(f"✅ Organized {len(samples_data)} samples from {group_count} groups")
    
    # Save samples metadata
    metadata_file = os.path.join(output_dir, 'samples_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(samples_data, f, indent=2)
    
    print(f"💾 Saved metadata to {metadata_file}")
    return samples_data

def main():
    """Main data download and preprocessing pipeline"""
    print("""
🚀 DATA DOWNLOAD & PREPROCESSING PIPELINE
========================================
    """)
    
    # Batch files configuration (example IDs)
    batch_files = {
        "batch_0000.tar.gz": "1-XSfs7Mop-Pr8tyWynC1o9im4JlnxYAR",
        "batch_0001.tar.gz": "1tILKFX4AyzHundDEZ2DyUU6mccehtXAI",
        "batch_0002.tar.gz": "1qfSS9LDZEU4QXrPY1BNXWb2u1Wt4m_5w",
        "batch_0003.tar.gz": "1Fb0SDA3DJ0vkeUQ8tTvLF1-iMiyEhq7r"
    }
    
    # Create directories
    download_dir = "./downloads"
    extract_dir = "./extracted"
    output_dir = "./data/processed"
    
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Download and extract batches
    all_samples = []
    
    for batch_name, file_id in batch_files.items():
        # Download batch file
        batch_path = os.path.join(download_dir, batch_name)
        
        # Skip if already downloaded
        if not os.path.exists(batch_path):
            downloaded_file = download_from_drive(file_id, batch_path)
            if downloaded_file is None:
                continue
        
        # Extract batch
        batch_extract_dir = os.path.join(extract_dir, batch_name.replace('.tar.gz', ''))
        extract_batch_file(batch_path, batch_extract_dir)
        
        # Reorganize data
        batch_samples = reorganize_data_structure(batch_extract_dir, 
                                                  os.path.join(output_dir, batch_name.replace('.tar.gz', '')))
        all_samples.extend(batch_samples)
    
    # Final summary
    print(f"""
📊 DATA PREPROCESSING COMPLETED!
==============================

📁 Downloaded: {len(batch_files)} batch files
📂 Extracted: {len(batch_files)} batch directories  
📋 Total samples: {len(all_samples)}
📍 Output directory: {output_dir}

✅ Data ready for training pipeline!
    """)
    
    return all_samples

if __name__ == "__main__":
    samples = main()
