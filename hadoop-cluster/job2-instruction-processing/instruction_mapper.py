#!/usr/bin/env python3
"""
OPTIMIZED JOB 2: Instruction Processing Mapper
Purpose: Convert raw metadata.jsonl + prompt.json to repo-compatible JSONL format
"""

import sys
import os
import json
import subprocess
import tempfile

def process_group_instructions(group_path):
    """Process metadata and prompt files in a group from HDFS (simplified)"""
    group_id = os.path.basename(group_path)
    instructions = []
    errors = []
    
    try:
        # Read prompt.json directly from HDFS using cat
        prompt_hdfs_path = f"{group_path}/prompt.json"
        result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-cat', prompt_hdfs_path],
                               capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            errors.append(f"Cannot read prompt.json for {group_id}")
            return instructions, errors
        
        # Parse prompt data
        prompt_data = json.loads(result.stdout)
        edit_instruction = prompt_data.get('edit', 'Unknown edit')
        
        # Read metadata.jsonl directly from HDFS using cat
        metadata_hdfs_path = f"{group_path}/metadata.jsonl"
        result = subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-cat', metadata_hdfs_path],
                               capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            errors.append(f"Cannot read metadata.jsonl for {group_id}")
            return instructions, errors
        
        # Process each line in metadata
        for line_num, line in enumerate(result.stdout.strip().split('\n'), 1):
            if not line.strip():
                continue
                
            try:
                sample_data = json.loads(line.strip())
                image_id = str(sample_data.get('seed', f'unknown_{line_num}'))
                
                # Create repo-compatible instruction record
                instruction_record = {
                    "source_image": f"{group_id}/{image_id}_0.jpg",
                    "target_image": f"{group_id}/{image_id}_1.jpg", 
                    "instruction": edit_instruction,
                    "caption_before": f"Original image {image_id}",
                    "caption_after": f"Edited: {edit_instruction}"
                }
                
                instructions.append(instruction_record)
                
            except json.JSONDecodeError as e:
                errors.append(f"JSON error in {group_id} line {line_num}")
                continue
            except Exception as e:
                errors.append(f"Processing error in {group_id} line {line_num}")
                continue
                
    except Exception as e:
        errors.append(f"Group processing error {group_id}: {str(e)}")
        
    return instructions, errors

def main():
    """Mapper main function"""
    for line in sys.stdin:
        group_path = line.strip()
        
        if not group_path:
            continue
            
        # Process instructions in this group
        instructions, errors = process_group_instructions(group_path)
        
        # Emit each instruction separately
        for instruction in instructions:
            # Output: key="instruction", value=instruction_json
            print(f"instruction\t{json.dumps(instruction)}")
        
        # Also emit error summary
        if errors:
            group_id = os.path.basename(group_path)
            error_summary = {
                'group_id': group_id,
                'error_count': len(errors),
                'errors': errors[:2]  # First 2 errors only
            }
            print(f"error\t{json.dumps(error_summary)}")

if __name__ == "__main__":
    main()