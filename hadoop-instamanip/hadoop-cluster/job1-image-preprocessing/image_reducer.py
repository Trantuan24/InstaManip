#!/usr/bin/env python3
"""
JOB 1: Image Preprocessing Reducer
Purpose: Collect processing statistics and verify output
"""

import sys
import json
import os

def main():
    """Reducer main function"""
    current_group = None
    total_processed = 0
    total_errors = 0
    all_error_details = []
    group_stats = {}
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse mapper output: group_id \t json_stats
            group_id, stats_json = line.split('\t', 1)
            stats = json.loads(stats_json)
            
            # Collect statistics
            processed = stats.get('processed_images', 0)
            errors = stats.get('errors', 0)
            error_details = stats.get('error_details', [])
            
            total_processed += processed
            total_errors += errors
            all_error_details.extend(error_details)
            
            group_stats[group_id] = {
                'processed': processed,
                'errors': errors
            }
            
        except Exception as e:
            continue
    
    # Generate final report
    summary = {
        'job': 'image_preprocessing',
        'total_groups_processed': len(group_stats),
        'total_images_processed': total_processed,
        'total_errors': total_errors,
        'success_rate': f"{(total_processed/(total_processed+total_errors)*100):.1f}%" if (total_processed+total_errors) > 0 else "0%",
        'sample_errors': all_error_details[:10]  # First 10 errors
    }
    
    # Write summary to output
    print(json.dumps(summary, indent=2))
    
    # Write detailed stats to file
    output_dir = "/datasets/processed/images"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/processing_stats.json", 'w') as f:
        json.dump({
            'summary': summary,
            'group_details': group_stats
        }, f, indent=2)

if __name__ == "__main__":
    main()
