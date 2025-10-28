#!/usr/bin/env python3
"""
Extract images from Parquet files with HIGH QUALITY (JPEG quality=100)
This version preserves maximum image quality to meet >= 10 GB requirement
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


def extract_images_high_quality(parquet_dir, output_dir, quality=100, format='JPEG'):
    """Extract all images from Parquet files with high quality"""
    print("=" * 80)
    print("EXTRACTING IMAGES WITH HIGH QUALITY")
    print("=" * 80)
    print(f"Input: {parquet_dir}")
    print(f"Output: {output_dir}")
    print(f"Format: {format}")
    print(f"Quality: {quality}/100")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} Parquet files")
    
    global_idx = 0
    total_images = 0
    skipped = 0
    total_size = 0
    
    print("\nExtracting images...")
    for parquet_file in tqdm(parquet_files, desc="Processing files"):
        # Read parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        # Process each row
        for _, row in df.iterrows():
            sample_id = str(global_idx).zfill(10)  # Format: 0000000000
            
            # Determine file extension
            ext = 'jpg' if format == 'JPEG' else 'png'
            
            # Extract source image
            source_path = os.path.join(output_dir, f"{sample_id}_source.{ext}")
            if not os.path.exists(source_path):
                try:
                    source_img_dict = row['source_image']
                    if isinstance(source_img_dict, dict) and 'bytes' in source_img_dict:
                        img_bytes = source_img_dict['bytes']
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            if format == 'JPEG':
                                img.save(source_path, 'JPEG', quality=quality)
                            else:
                                img.save(source_path, 'PNG')
                            total_images += 1
                            total_size += os.path.getsize(source_path)
                except Exception as e:
                    print(f"\nError extracting source image {sample_id}: {e}")
            else:
                skipped += 1
                total_size += os.path.getsize(source_path)
            
            # Extract target image
            target_path = os.path.join(output_dir, f"{sample_id}_target.{ext}")
            if not os.path.exists(target_path):
                try:
                    target_img_dict = row['edited_image']
                    if isinstance(target_img_dict, dict) and 'bytes' in target_img_dict:
                        img_bytes = target_img_dict['bytes']
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes))
                            if format == 'JPEG':
                                img.save(target_path, 'JPEG', quality=quality)
                            else:
                                img.save(target_path, 'PNG')
                            total_images += 1
                            total_size += os.path.getsize(target_path)
                except Exception as e:
                    print(f"\nError extracting target image {sample_id}: {e}")
            else:
                skipped += 1
                total_size += os.path.getsize(target_path)
            
            global_idx += 1
    
    print(f"\n✓ Extracted {total_images} images")
    print(f"✓ Skipped {skipped} existing images")
    print(f"✓ Total samples: {global_idx}")
    print(f"✓ Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"✓ Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract images from Parquet files with high quality")
    parser.add_argument("--parquet_dir", type=str, default="../raw_data",
                        help="Directory containing Parquet files")
    parser.add_argument("--output_dir", type=str, default="../processed_data/images_hq",
                        help="Output directory for extracted images")
    parser.add_argument("--quality", type=int, default=100,
                        help="JPEG quality (1-100, default: 100)")
    parser.add_argument("--format", type=str, default="JPEG", choices=["JPEG", "PNG"],
                        help="Image format (JPEG or PNG, default: JPEG)")
    
    args = parser.parse_args()
    
    extract_images_high_quality(args.parquet_dir, args.output_dir, args.quality, args.format)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

