#!/usr/bin/env python3
"""
OPTIMIZED JOB 2: Instruction Processing Reducer
Purpose: Combine all instructions into repo-compatible JSONL file
"""

import sys
import json
import os
import subprocess
import tempfile

def main():
    """Reducer main function"""
    instructions = []
    errors = []
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse mapper output: key \t json_data
            key, data_json = line.split('\t', 1)
            data = json.loads(data_json)
            
            if key == "instruction":
                instructions.append(data)
            elif key == "error":
                errors.append(data)
                
        except Exception as e:
            continue
    
    # Write repo-compatible JSONL file to temp then upload to HDFS
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
        for instruction in instructions:
            temp_file.write(json.dumps(instruction) + '\n')
        temp_jsonl_path = temp_file.name
    
    # Ensure HDFS directory exists
    subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-mkdir', '-p', '/datasets/processed/instructions'],
                  capture_output=True, timeout=15)
    
    # Upload JSONL file to HDFS
    hdfs_output_path = '/datasets/processed/instructions/processed_dataset.jsonl'
    subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', temp_jsonl_path, hdfs_output_path],
                  capture_output=True, timeout=15)
    
    os.unlink(temp_jsonl_path)
    
    # Generate processing summary
    summary = {
        'job': 'instruction_processing',
        'total_instructions': len(instructions),
        'total_groups_with_errors': len(errors),
        'total_errors': sum(err.get('error_count', 0) for err in errors),
        'output_file': hdfs_output_path,
        'sample_instruction': instructions[0] if instructions else None,
        'error_groups': [err.get('group_id') for err in errors[:10]]  # First 10 error groups
    }
    
    # Write summary
    print(json.dumps(summary, indent=2))
    
    # Write detailed error log to HDFS
    if errors:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_error_file:
            json.dump(errors, temp_error_file, indent=2)
            temp_error_path = temp_error_file.name
        
        subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', temp_error_path, 
                       '/datasets/processed/instructions/error_log.json'], capture_output=True, timeout=10)
        os.unlink(temp_error_path)

if __name__ == "__main__":
    main()