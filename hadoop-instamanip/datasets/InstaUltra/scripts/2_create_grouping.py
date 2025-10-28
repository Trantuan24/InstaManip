#!/usr/bin/env python3
"""
Script 2: Tạo Grouping Structure cho In-Context Learning
Mục tiêu: Tạo file ultraedit_group_instruct.json giống ip2p_group_instruct.json
"""

import os
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not installed. Will use simple grouping.")


def simple_grouping(samples, output_file):
    """
    Simple grouping: Group theo edit_type có sẵn
    Nhanh nhưng ít chính xác hơn CLIP clustering
    """
    print("\n[METHOD 1] Simple Grouping by Edit Type")
    print("=" * 80)
    
    groups = defaultdict(list)
    
    print("Grouping samples by edit_type...")
    for i, sample in enumerate(tqdm(samples)):
        # Lấy edit_type nếu có, không thì dùng "unknown"
        edit_type = sample.get('edit_type', 'unknown')
        
        # Thêm sample ID vào group
        sample_id = str(i).zfill(10)  # Format: 0000000001
        groups[edit_type].append(sample_id)
    
    print(f"\n✓ Created {len(groups)} groups")
    
    # Statistics
    print("\nGroup Statistics:")
    for edit_type, sample_ids in sorted(groups.items()):
        print(f"  {edit_type}: {len(sample_ids)} samples")
    
    # Convert to format giống ip2p_group_instruct.json
    # Format: {"0000000": ["id1", "id2", ...], "0000001": [...], ...}
    grouping_dict = {}
    for group_idx, (edit_type, sample_ids) in enumerate(groups.items()):
        group_key = str(group_idx).zfill(7)  # Format: 0000000
        grouping_dict[group_key] = sample_ids
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(grouping_dict, f, indent=4)
    
    print(f"\n✓ Saved grouping to: {output_file}")
    print(f"Total groups: {len(grouping_dict)}")
    
    return grouping_dict


def clip_grouping(samples, output_file, n_clusters=500):
    """
    Advanced grouping: CLIP clustering
    Chậm hơn nhưng chính xác hơn
    """
    print("\n[METHOD 2] CLIP-based Clustering")
    print("=" * 80)
    
    if not CLIP_AVAILABLE:
        print("✗ transformers not installed. Falling back to simple grouping.")
        return simple_grouping(samples, output_file)
    
    print("Loading CLIP model...")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ CLIP model loaded on {device}")
    
    # Extract instructions
    print("\nExtracting instructions...")
    instructions = []
    for sample in tqdm(samples):
        instruction = sample.get('instruction', sample.get('edit', ''))
        instructions.append(instruction)
    
    # Encode instructions
    print("\nEncoding instructions with CLIP...")
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size)):
            batch = instructions[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"✓ Encoded {len(embeddings)} instructions")
    print(f"Embedding shape: {embeddings.shape}")
    
    # K-means clustering
    print(f"\nClustering into {n_clusters} groups...")
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    labels = kmeans.fit_predict(embeddings)
    
    print(f"✓ Clustering completed")
    
    # Create groups
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        sample_id = str(i).zfill(10)
        groups[label].append(sample_id)
    
    # Convert to format giống ip2p_group_instruct.json
    grouping_dict = {}
    for group_idx, sample_ids in groups.items():
        group_key = str(group_idx).zfill(7)
        grouping_dict[group_key] = sample_ids
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(grouping_dict, f, indent=4)
    
    print(f"\n✓ Saved grouping to: {output_file}")
    print(f"Total groups: {len(grouping_dict)}")
    
    # Statistics
    group_sizes = [len(ids) for ids in grouping_dict.values()]
    print(f"\nGroup size statistics:")
    print(f"  Min: {min(group_sizes)}")
    print(f"  Max: {max(group_sizes)}")
    print(f"  Mean: {np.mean(group_sizes):.1f}")
    print(f"  Median: {np.median(group_sizes):.1f}")
    
    return grouping_dict


def create_grouping(input_file, output_file, method="simple", n_clusters=500):
    """
    Main function để tạo grouping structure
    
    Args:
        input_file: File JSON chứa raw data
        output_file: Output grouping file
        method: "simple" hoặc "clip"
        n_clusters: Số clusters cho CLIP method
    """
    print("=" * 80)
    print("CREATING GROUPING STRUCTURE FOR IN-CONTEXT LEARNING")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    with open(input_file, 'r') as f:
        samples = json.load(f)
    
    print(f"✓ Loaded {len(samples)} samples")
    
    # Create grouping
    if method == "simple":
        grouping_dict = simple_grouping(samples, output_file)
    elif method == "clip":
        grouping_dict = clip_grouping(samples, output_file, n_clusters)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("\n" + "=" * 80)
    print("GROUPING COMPLETED!")
    print("=" * 80)
    print(f"\nGrouping file: {output_file}")
    print(f"Total groups: {len(grouping_dict)}")
    print(f"\nNext step: Run script 3_convert_to_jsonl.py")


def main():
    parser = argparse.ArgumentParser(description="Create grouping structure")
    parser.add_argument(
        "--input_file",
        type=str,
        default="../raw_data/ultraedit_raw.json",
        help="Input JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../processed_data/ultraedit_group_instruct.json",
        help="Output grouping file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["simple", "clip"],
        default="simple",
        help="Grouping method (simple=fast, clip=accurate)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=500,
        help="Number of clusters for CLIP method"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    create_grouping(
        args.input_file,
        args.output_file,
        args.method,
        args.n_clusters
    )


if __name__ == "__main__":
    main()

