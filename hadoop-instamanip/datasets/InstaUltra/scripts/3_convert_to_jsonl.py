#!/usr/bin/env python3
"""
Script 3: Convert UltraEdit sang JSONL format giống IP2P
Mục tiêu: Tạo ultraedit_1.jsonl, ultraedit_2.jsonl, ... giống ip2p_1.jsonl
"""

import os
import json
from tqdm import tqdm
import argparse
from PIL import Image
import shutil


def convert_to_jsonl(input_file, output_dir, image_output_dir, chunk_size=5000):
    """
    Convert UltraEdit JSON sang JSONL format
    
    Args:
        input_file: Input JSON file
        output_dir: Output directory cho JSONL files
        image_output_dir: Directory để lưu images
        chunk_size: Số samples per JSONL file
    """
    print("=" * 80)
    print("CONVERTING TO JSONL FORMAT")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    with open(input_file, 'r') as f:
        samples = json.load(f)
    
    print(f"✓ Loaded {len(samples)} samples")
    
    # Convert to JSONL format
    print("\nConverting to JSONL format...")
    print(f"Chunk size: {chunk_size} samples per file")
    
    num_chunks = (len(samples) + chunk_size - 1) // chunk_size
    print(f"Will create {num_chunks} JSONL files")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(samples))
        chunk_samples = samples[start_idx:end_idx]
        
        # Output file
        output_file = os.path.join(output_dir, f"ultraedit_{chunk_idx + 1}.jsonl")
        
        print(f"\n[{chunk_idx + 1}/{num_chunks}] Processing chunk {chunk_idx + 1}...")
        print(f"  Samples: {start_idx} - {end_idx}")
        print(f"  Output: {output_file}")
        
        with open(output_file, 'w') as f:
            for i, sample in enumerate(tqdm(chunk_samples, desc=f"Chunk {chunk_idx + 1}")):
                # Convert to IP2P format
                # IP2P format: {"id": "...", "instruction": "...", "source_image": "...", "target_image": "..."}
                
                sample_id = str(start_idx + i).zfill(10)
                
                # Extract fields
                instruction = sample.get('instruction', sample.get('edit', ''))
                
                # Image paths - UltraEdit có thể có nhiều format khác nhau
                # Chúng ta sẽ lưu image paths tương đối
                source_image = f"ultraedit/{sample_id}_source.jpg"
                target_image = f"ultraedit/{sample_id}_target.jpg"
                
                # Tạo entry theo format IP2P
                entry = {
                    "id": sample_id,
                    "instruction": instruction,
                    "source_image": source_image,
                    "target_image": target_image
                }
                
                # Optional fields
                if 'source_caption' in sample:
                    entry['source_caption'] = sample['source_caption']
                if 'target_caption' in sample:
                    entry['target_caption'] = sample['target_caption']
                if 'edit_type' in sample:
                    entry['edit_type'] = sample['edit_type']
                
                # Write to JSONL
                f.write(json.dumps(entry) + '\n')
                
                # Save images if available
                # Note: Images trong HuggingFace dataset thường ở dạng PIL Image hoặc bytes
                # Chúng ta sẽ handle trong script riêng nếu cần
        
        print(f"  ✓ Saved {len(chunk_samples)} samples to {output_file}")
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETED!")
    print("=" * 80)
    print(f"\nCreated {num_chunks} JSONL files in: {output_dir}")
    print(f"Total samples: {len(samples)}")
    
    # Summary
    print("\nFiles created:")
    for i in range(num_chunks):
        filename = f"ultraedit_{i + 1}.jsonl"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            # Count lines
            with open(filepath, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"  {filename}: {num_lines} samples")
    
    print("\n" + "=" * 80)
    print("IMPORTANT NOTES:")
    print("=" * 80)
    print("1. Images chưa được download/save")
    print("2. Bạn cần download images từ HuggingFace dataset")
    print("3. Hoặc sử dụng streaming mode khi load dataset")
    print("\nNext step: Run script 4_validate_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Convert to JSONL format")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../raw_data/ultraedit_raw.json",
        help="Input JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../processed_data/train",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--image_output_dir",
        type=str,
        default="../processed_data/images",
        help="Output directory for images"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of samples per JSONL file"
    )
    
    args = parser.parse_args()
    
    convert_to_jsonl(
        args.input_file,
        args.output_dir,
        args.image_output_dir,
        args.chunk_size
    )


if __name__ == "__main__":
    main()

