#!/usr/bin/env python3
"""
OPTIMIZED Job 1: Fast Image Preprocessing Mapper
Purpose: Quick resize to 1024x1024 with minimal overhead
"""

import sys, os, json, subprocess, tempfile
try:
    from PIL import Image
    HAS_PIL = True
except:
    HAS_PIL = False

def process_group(group_path):
    """Fast process all images in a group"""
    group_id = os.path.basename(group_path)
    processed = 0
    errors = []
    
    # Get JPG files from HDFS
    try:
        result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-ls', group_path], 
                               capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return 0, ["Cannot list directory"]
    except:
        return 0, ["HDFS ls timeout"]
    
    # Extract JPG paths
    jpg_files = [line.split()[-1] for line in result.stdout.split('\n') 
                 if '.jpg' in line and not line.startswith('d')]
    
    if not jpg_files:
        return 0, ["No JPG files found"]
    
    # Create output directory once
    out_dir = f"/datasets/processed/images/{group_id}"
    subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-mkdir', '-p', out_dir], 
                   capture_output=True, timeout=10)
    
    # Process each image quickly
    for jpg_path in jpg_files:
        filename = os.path.basename(jpg_path)
        
        # Fix: Use manual temp files instead of context manager
        import uuid
        tmp_in = f"/tmp/in_{uuid.uuid4().hex}.jpg"
        tmp_out = f"/tmp/out_{uuid.uuid4().hex}.jpg"
        
        try:
            # Fast download
            if subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-get', jpg_path, tmp_in], 
                             timeout=15).returncode != 0:
                errors.append(f"Download failed: {filename}")
                continue
            
            # Quick resize if PIL available, else copy
            if HAS_PIL:
                img = Image.open(tmp_in)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.resize((1024, 1024), Image.LANCZOS).save(tmp_out, 'JPEG', quality=90)
            else:
                subprocess.run(['cp', tmp_in, tmp_out])
            
            # Fast upload
            if subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', tmp_out, f"{out_dir}/{filename}"],
                             timeout=15).returncode == 0:
                processed += 1
            else:
                errors.append(f"Upload failed: {filename}")
                
        except Exception as e:
            errors.append(f"Process error: {filename}")
        finally:
            # Quick cleanup
            for f in [tmp_in, tmp_out]:
                if os.path.exists(f):
                    os.unlink(f)
    
    return processed, errors

def main():
    """Main mapper function"""
    for line in sys.stdin:
        group_path = line.strip()
        if not group_path:
            continue
            
        processed, errors = process_group(group_path)
        group_id = os.path.basename(group_path)
        
        result = {
            'processed': processed,
            'errors': len(errors),
            'details': errors[:2]  # Only first 2 errors
        }
        
        print(f"{group_id}\t{json.dumps(result)}")

if __name__ == "__main__":
    main()