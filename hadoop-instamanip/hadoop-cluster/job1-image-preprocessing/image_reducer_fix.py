#!/usr/bin/env python3
"""
OPTIMIZED Job 1: Fast Image Preprocessing Reducer
Purpose: Quick stats collection
"""

import sys, json, subprocess, tempfile

def main():
    """Fast reducer function"""
    total_processed = 0
    total_errors = 0
    group_count = 0
    all_errors = []
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
                
            group_id, stats_json = parts
            stats = json.loads(stats_json)
            
            # Quick stats collection
            processed = stats.get('processed', 0)
            errors = stats.get('errors', 0)
            details = stats.get('details', [])
            
            total_processed += processed
            total_errors += errors
            group_count += 1
            all_errors.extend(details)
            
        except:
            continue
    
    # Quick summary
    total_attempts = total_processed + total_errors
    success_rate = f"{(total_processed/total_attempts*100):.1f}%" if total_attempts > 0 else "0%"
    
    summary = {
        'job': 'image_preprocessing_optimized',
        'groups': group_count,
        'processed': total_processed,
        'errors': total_errors,
        'success_rate': success_rate,
        'sample_errors': all_errors[:5]
    }
    
    print(json.dumps(summary, indent=2))
    
    # Quick stats save to HDFS
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(summary, f, indent=2)
            temp_file = f.name
        
        subprocess.run(['/opt/hadoop/bin/hdfs', 'dfs', '-put', temp_file, 
                       '/datasets/processed/images/processing_stats.json'], timeout=15)
        subprocess.run(['rm', temp_file])
    except:
        pass

if __name__ == "__main__":
    main()