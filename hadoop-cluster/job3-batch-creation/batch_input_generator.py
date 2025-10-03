#!/usr/bin/env python3
"""
Batch Input Generator
Purpose: Generate input for batch creation job (groups per batch)
"""

import os
import json
import sys
import subprocess

def generate_batch_inputs(groups_per_batch=30):
    """Generate batch input data from HDFS"""
    # Get list of processed image groups from HDFS
    images_hdfs_dir = "/datasets/processed/images"
    
    try:
        # List directories in HDFS
        result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-ls', images_hdfs_dir],
                               capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print("Error: Cannot list HDFS processed images directory", file=sys.stderr)
            return
    except Exception as e:
        print(f"Error accessing HDFS: {e}", file=sys.stderr)
        return
    
    # Extract group IDs from HDFS ls output
    group_ids = []
    for line in result.stdout.split('\n'):
        if 'drw' in line:  # Directory line
            path = line.split()[-1]  # Last column is the path
            group_id = os.path.basename(path)
            if group_id and group_id != "processing_stats.json":
                group_ids.append(group_id)
    
    group_ids.sort()  # Consistent ordering
    
    # Create batches
    batch_num = 0
    for i in range(0, len(group_ids), groups_per_batch):
        batch_groups = group_ids[i:i + groups_per_batch]
        
        batch_info = {
            'batch_id': f'batch_{batch_num:04d}',
            'groups': batch_groups
        }
        
        print(json.dumps(batch_info))
        batch_num += 1

if __name__ == "__main__":
    # Default: 30 groups per batch (~120 samples, ~100MB compressed)
    groups_per_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    generate_batch_inputs(groups_per_batch)