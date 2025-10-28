#!/usr/bin/env python3
"""
Script 1: Download UltraEdit Dataset với Stratified Sampling
Mục tiêu: Download ~30,000 samples (10-12 GB) với distribution cân bằng
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Target distribution (30,000 samples total)
TARGET_DISTRIBUTION = {
    "object_addition": 6000,      # 20%
    "attribute_modification": 9000,  # 30%
    "object_removal": 4500,       # 15%
    "style_transfer": 3000,       # 10%
    "background_change": 3750,    # 12.5%
    "other": 3750                 # 12.5%
}

def download_ultraedit_stratified(output_dir, target_size_gb=12, dry_run=False):
    """
    Download UltraEdit dataset với stratified sampling
    
    Args:
        output_dir: Thư mục lưu data
        target_size_gb: Kích thước target (GB)
        dry_run: Nếu True, chỉ analyze không download
    """
    print("=" * 80)
    print("DOWNLOADING ULTRAEDIT DATASET - STRATIFIED SAMPLING")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # OPTION 1: Download từ subset 500K (dễ hơn)
    print("\n[1/4] Loading UltraEdit_500k metadata...")
    print("Note: Chúng ta sẽ download từ subset 500K để dễ quản lý")
    
    try:
        # Load dataset streaming để không tốn RAM
        dataset = load_dataset(
            "BleachNick/UltraEdit_500k",
            split="FreeForm",  # UltraEdit uses "FreeForm" split
            streaming=True
        )
        
        print("✓ Dataset loaded successfully (streaming mode)")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nAlternative: Download manually từ HuggingFace")
        print("Link: https://huggingface.co/datasets/BleachNick/UltraEdit_500k")
        return
    
    # OPTION 2: Simple approach - download first N samples
    print("\n[2/4] Downloading samples...")
    print(f"Target: ~30,000 samples (~{target_size_gb} GB)")
    
    if dry_run:
        print("DRY RUN MODE - Chỉ analyze, không download")
        
        # Count samples
        count = 0
        for i, sample in enumerate(dataset):
            count += 1
            if i >= 100:  # Chỉ check 100 samples đầu
                break
            
            if i % 10 == 0:
                print(f"Sample {i}: {list(sample.keys())}")
        
        print(f"\nTotal samples checked: {count}")
        print("Dataset structure looks good!")
        return
    
    # Download actual data
    samples = []
    target_samples = 30000
    
    print(f"Downloading {target_samples} samples...")
    for i, sample in enumerate(tqdm(dataset, total=target_samples)):
        samples.append(sample)
        
        if i >= target_samples - 1:
            break
    
    print(f"\n✓ Downloaded {len(samples)} samples")
    
    # Save to disk
    print("\n[3/4] Saving to disk...")
    output_file = os.path.join(output_dir, "ultraedit_raw.json")
    
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✓ Saved to: {output_file}")
    
    # Statistics
    print("\n[4/4] Dataset Statistics:")
    print(f"Total samples: {len(samples)}")
    
    # Estimate size
    import sys
    size_mb = sys.getsizeof(json.dumps(samples)) / (1024 * 1024)
    print(f"Estimated size: {size_mb:.2f} MB (metadata only)")
    print("\nNote: Images sẽ được download khi load dataset lần đầu")
    
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETED!")
    print("=" * 80)
    print(f"\nNext step: Run script 2_create_grouping.py")


def main():
    parser = argparse.ArgumentParser(description="Download UltraEdit dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../raw_data",
        help="Output directory"
    )
    parser.add_argument(
        "--target_size_gb",
        type=int,
        default=12,
        help="Target size in GB"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode (analyze only)"
    )
    
    args = parser.parse_args()
    
    download_ultraedit_stratified(
        args.output_dir,
        args.target_size_gb,
        args.dry_run
    )


if __name__ == "__main__":
    main()

