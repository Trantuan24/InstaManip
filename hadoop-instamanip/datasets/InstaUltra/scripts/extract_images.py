#!/usr/bin/env python3
"""
Extract images from Parquet files and save as individual image files
"""

import os
import glob
from tqdm import tqdm
import argparse

try:
    import pyarrow.parquet as pq
    from PIL import Image
    from io import BytesIO
except ImportError:
    print("Error: Required packages not installed")
    print("Run: pip install pyarrow Pillow")
    exit(1)


def extract_images(parquet_dir, output_dir):
    """Extract all images from Parquet files"""
    print("=" * 80)
    print("EXTRACTING IMAGES FROM PARQUET FILES")
    print("=" * 80)
    print(f"Input: {parquet_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} Parquet files")
    
    global_idx = 0
    total_images = 0
    skipped = 0
    
    print("\nExtracting images...")
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        # Read parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Process each row
        for _, row in df.iterrows():
            sample_id = str(global_idx).zfill(10)  # Format: 0000000000
            
            # Extract source image
            source_path = os.path.join(output_dir, f"{sample_id}_source.jpg")
            if not os.path.exists(source_path):
                try:
                    source_img_dict = row['source_image']
                    if isinstance(source_img_dict, dict) and 'bytes' in source_img_dict:
                        img_bytes = source_img_dict['bytes']
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            img.save(source_path, 'JPEG')
                            total_images += 1
                except Exception as e:
                    print(f"\nError extracting source image {sample_id}: {e}")
            else:
                skipped += 1
            
            # Extract target image
            target_path = os.path.join(output_dir, f"{sample_id}_target.jpg")
            if not os.path.exists(target_path):
                try:
                    target_img_dict = row['edited_image']
                    if isinstance(target_img_dict, dict) and 'bytes' in target_img_dict:
                        img_bytes = target_img_dict['bytes']
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            img.save(target_path, 'JPEG')
                            total_images += 1
                except Exception as e:
                    print(f"\nError extracting target image {sample_id}: {e}")
            else:
                skipped += 1
            
            global_idx += 1
    
    print(f"\n✓ Extracted {total_images} images")
    print(f"✓ Skipped {skipped} existing images")
    print(f"✓ Total samples: {global_idx}")
    print(f"✓ Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from Parquet files")
    parser.add_argument("--parquet_dir", type=str, default="../raw_data",
                        help="Directory containing Parquet files")
    parser.add_argument("--output_dir", type=str, default="../processed_data/images",
                        help="Output directory for extracted images")
    
    args = parser.parse_args()
    
    extract_images(args.parquet_dir, args.output_dir)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

