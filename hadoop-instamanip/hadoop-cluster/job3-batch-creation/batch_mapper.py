#!/usr/bin/env python3
"""
OPTIMIZED JOB 3: Batch Creation Mapper
Purpose: Create repo-compatible training batches for cloud deployment (HDFS)
"""

import sys
import os
import json
import subprocess
import tarfile
import tempfile
import uuid

def extract_instructions_for_groups(group_ids):
    """Extract instructions for specific groups from HDFS"""
    group_instructions = []
    
    try:
        # Read instructions from HDFS
        result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-cat', 
                                '/datasets/processed/instructions/processed_dataset.jsonl'],
                               capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return group_instructions
            
        # Filter instructions for our groups
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            try:
                instruction = json.loads(line.strip())
                # Check if instruction belongs to any of our groups
                source_img = instruction.get('source_image', '')
                group_id = source_img.split('/')[0] if '/' in source_img else ''
                
                if group_id in group_ids:
                    group_instructions.append(instruction)
                    
            except json.JSONDecodeError:
                continue
                    
    except Exception as e:
        pass
        
    return group_instructions

def create_batch(batch_id, group_ids):
    """Create a training batch with repo-compatible structure from HDFS"""
    try:
        # Create unique temporary batch directory
        temp_id = uuid.uuid4().hex[:8]
        batch_dir = f"/tmp/batch_{temp_id}_{batch_id}"
        os.makedirs(f"{batch_dir}/ip2p", exist_ok=True)      # Repo expects "ip2p" folder
        os.makedirs(f"{batch_dir}/train", exist_ok=True)     # Repo expects "train" folder
        
        copied_groups = []
        total_image_count = 0
        
        # Download image groups from HDFS
        for group_id in group_ids:
            source_hdfs_path = f"/datasets/processed/images/{group_id}"
            target_group_path = f"{batch_dir}/ip2p/{group_id}"
            
            # Create local group directory
            os.makedirs(target_group_path, exist_ok=True)
            
            # Download all images in this group from HDFS
            try:
                result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-get', 
                                        f"{source_hdfs_path}/*", target_group_path + "/"],
                                       capture_output=True, timeout=30)
                if result.returncode == 0:
                    copied_groups.append(group_id)
                    # Count downloaded images
                    image_count = len([f for f in os.listdir(target_group_path) if f.endswith('.jpg')])
                    total_image_count += image_count
                    
            except Exception:
                continue
        
        # Extract and write instructions for these groups
        batch_instructions = extract_instructions_for_groups(copied_groups)
        
        # Write batch instruction file
        batch_jsonl_path = f"{batch_dir}/train/batch_data.jsonl"
        with open(batch_jsonl_path, 'w') as f:
            for instruction in batch_instructions:
                f.write(json.dumps(instruction) + '\n')
        
        # Create compressed archive locally first
        temp_compressed_path = f"/tmp/{batch_id}.tar.gz"
        
        with tarfile.open(temp_compressed_path, 'w:gz') as tar:
            tar.add(batch_dir, arcname=batch_id)
        
        # Upload compressed batch to HDFS
        hdfs_final_path = f"/datasets/final/{batch_id}.tar.gz"
        
        # Ensure HDFS final directory exists
        subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-mkdir', '-p', '/datasets/final'],
                      capture_output=True, timeout=15)
        
        # Upload to HDFS
        upload_result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', 
                                       temp_compressed_path, hdfs_final_path],
                                      capture_output=True, timeout=60)
        
        # Get file size
        compressed_size_mb = os.path.getsize(temp_compressed_path) / (1024 * 1024)
        
        # Cleanup temp files
        if os.path.exists(temp_compressed_path):
            os.unlink(temp_compressed_path)
        if os.path.exists(batch_dir):
            subprocess.run(['rm', '-rf', batch_dir], capture_output=True)
        
        if upload_result.returncode == 0:
            return {
                'batch_id': batch_id,
                'groups_included': len(copied_groups),
                'total_samples': len(batch_instructions),
                'total_images': total_image_count,
                'compressed_size_mb': round(compressed_size_mb, 2),
                'output_file': hdfs_final_path,
                'status': 'success'
            }
        else:
            return {
                'batch_id': batch_id,
                'error': 'Upload to HDFS failed',
                'groups_included': len(copied_groups),
                'total_samples': len(batch_instructions)
            }
        
    except Exception as e:
        return {
            'batch_id': batch_id,
            'error': str(e),
            'groups_included': 0,
            'total_samples': 0
        }

def main():
    """Mapper main function"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse input: batch_info as JSON
            batch_info = json.loads(line)
            batch_id = batch_info.get('batch_id', 'unknown')
            group_ids = batch_info.get('groups', [])
            
            # Create batch
            result = create_batch(batch_id, group_ids)
            
            # Emit result
            print(f"{batch_id}\t{json.dumps(result)}")
            
        except Exception as e:
            print(f"error\t{json.dumps({'error': str(e), 'line': line})}")

if __name__ == "__main__":
    main()