#!/usr/bin/env python3
"""
JOB 1: Image Preprocessing Mapper
Purpose: Resize raw images to 1024x1024, maintain repo-compatible structure
"""

import sys
import os
import json
from PIL import Image
import glob

def process_group_images(group_path):
    """Process all images in a group folder"""
    group_id = os.path.basename(group_path)
    processed_count = 0
    errors = []
    
    # Find all JPG files in group
    image_files = glob.glob(os.path.join(group_path, "*.jpg"))
    
    for image_file in image_files:
        try:
            # Load and resize image
            image = Image.open(image_file)
            
            # Convert to RGB (handle RGBA, grayscale)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 1024x1024 (InstaManip standard)
            image = image.resize((1024, 1024), Image.LANCZOS)
            
            # Keep original filename for repo compatibility
            filename = os.path.basename(image_file)
            
            # Create output path (maintain group structure)
            output_dir = f"/datasets/processed/images/{group_id}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            # Save processed image
            image.save(output_path, 'JPEG', quality=95)
            processed_count += 1
            
        except Exception as e:
            errors.append(f"{image_file}: {str(e)}")
            continue
    
    return processed_count, errors

def main():
    """Mapper main function"""
    for line in sys.stdin:
        group_path = line.strip()
        
        if not os.path.exists(group_path):
            continue
            
        # Process images in this group
        processed_count, errors = process_group_images(group_path)
        
        # Emit results (group_id -> processed_count)
        group_id = os.path.basename(group_path)
        
        # Output: key=group_id, value=json with stats
        result = {
            'processed_images': processed_count,
            'errors': len(errors),
            'error_details': errors[:5]  # First 5 errors only
        }
        
        print(f"{group_id}\t{json.dumps(result)}")

if __name__ == "__main__":
    main()
