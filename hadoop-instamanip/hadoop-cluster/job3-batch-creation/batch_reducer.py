#!/usr/bin/env python3
"""
JOB 3: Batch Creation Reducer
Purpose: Collect batch creation statistics and generate final report
"""

import sys
import json
import os
import subprocess
import tempfile

def main():
    """Reducer main function"""
    successful_batches = []
    failed_batches = []
    total_groups = 0
    total_samples = 0
    total_size_mb = 0
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse mapper output: batch_id \t result_json
            batch_id, result_json = line.split('\t', 1)
            result = json.loads(result_json)
            
            if batch_id == "error":
                failed_batches.append(result)
                continue
                
            if 'error' in result:
                failed_batches.append(result)
            else:
                successful_batches.append(result)
                total_groups += result.get('groups_included', 0)
                total_samples += result.get('total_samples', 0)
                total_size_mb += result.get('compressed_size_mb', 0)
                
        except Exception as e:
            continue
    
    # Generate final summary
    summary = {
        'job': 'batch_creation',
        'successful_batches': len(successful_batches),
        'failed_batches': len(failed_batches),
        'total_groups_processed': total_groups,
        'total_samples_created': total_samples,
        'total_compressed_size_gb': round(total_size_mb / 1024, 2),
        'average_batch_size_mb': round(total_size_mb / len(successful_batches), 2) if successful_batches else 0,
        'output_directory': '/datasets/final',
        'ready_for_cloud_deployment': len(successful_batches) > 0
    }
    
    # Write summary to output
    print(json.dumps(summary, indent=2))
    
    # Write detailed batch report to HDFS
    try:
        batch_report = {
            'summary': summary,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'batch_files': [
                {
                    'filename': f"{batch['batch_id']}.tar.gz",
                    'size_mb': batch.get('compressed_size_mb', 0),
                    'samples': batch.get('total_samples', 0),
                    'groups': batch.get('groups_included', 0)
                }
                for batch in successful_batches
            ]
        }
        
        # Write to temp file then upload to HDFS
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_report:
            json.dump(batch_report, temp_report, indent=2)
            temp_report_path = temp_report.name
        
        subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', temp_report_path, 
                       '/datasets/final/batch_creation_report.json'], capture_output=True, timeout=15)
        os.unlink(temp_report_path)
        
        # Create batch index for PyDrive upload
        batch_index = {
            'dataset_info': {
                'name': 'InstructPix2Pix_InstaManip_Processed',
                'total_batches': len(successful_batches),
                'total_samples': total_samples,
                'total_size_gb': round(total_size_mb / 1024, 2),
                'format': 'repo_compatible'
            },
            'batch_files': {
                f"{batch['batch_id']}.tar.gz": "DRIVE_FILE_ID_TO_BE_FILLED"
                for batch in successful_batches
            }
        }
        
        # Write index to temp file then upload to HDFS
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_index:
            json.dump(batch_index, temp_index, indent=2)
            temp_index_path = temp_index.name
        
        subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', temp_index_path, 
                       '/datasets/final/batch_index.json'], capture_output=True, timeout=15)
        os.unlink(temp_index_path)
        
    except Exception as e:
        pass

if __name__ == "__main__":
    main()