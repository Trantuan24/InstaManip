# InstaEdit-30K Dataset

Dataset for InstaManip project (CVPR 2025) - In-context learning for image manipulation.

## 📊 Dataset Info

- **Samples:** 30,788
- **Images:** 61,576 (source + target pairs)
- **Groups:** 90 (for in-context learning)
- **Size:** 10.99 GB
- **Quality:** 92% of original (JPEG quality=100)
- **Source:** UltraEdit (NeurIPS 2024)

## 📁 Structure

```
ultraedit_processing/
├── processed_data/          # Final dataset (10.99 GB)
│   ├── train/
│   │   ├── ultraedit_1.jsonl       # 5,000 samples
│   │   ├── ultraedit_2.jsonl       # 5,000 samples
│   │   ├── ultraedit_3.jsonl       # 5,000 samples
│   │   ├── ultraedit_4.jsonl       # 5,000 samples
│   │   ├── ultraedit_5.jsonl       # 5,000 samples
│   │   ├── ultraedit_6.jsonl       # 5,000 samples
│   │   └── ultraedit_7.jsonl       # 788 samples
│   ├── images/                      # 61,576 images
│   │   ├── 0000000000_source.jpg
│   │   ├── 0000000000_target.jpg
│   │   └── ...
│   └── ultraedit_group_instruct.json  # 90 groups
│
├── raw_data/                # Original Parquet files (11.16 GB)
│   └── FreeForm-*.parquet   # 24 files
│
├── scripts/                 # Processing scripts
│   ├── 1_download_dataset.py
│   ├── 2_create_grouping.py
│   ├── 3_convert_to_jsonl.py
│   ├── extract_images.py
│   └── extract_images_high_quality.py
│
└── requirements.txt
```

## 🚀 Quick Start

### Use Dataset

```python
import json
from PIL import Image

# Load JSONL
with open('processed_data/train/ultraedit_1.jsonl', 'r') as f:
    for line in f:
        sample = json.loads(line)
        print(sample['id'])
        print(sample['instruction'])
        print(sample['source_image'])
        print(sample['target_image'])

# Load image
img = Image.open('processed_data/images/0000000000_source.jpg')
```

### Load Grouping

```python
import json

# Load groups for in-context learning
with open('processed_data/ultraedit_group_instruct.json', 'r') as f:
    groups = json.load(f)
```

## 📋 JSONL Format

```json
{
  "id": "0000000000",
  "instruction": "Replace the pizza with miniature burgers",
  "source_image": "images/0000000000_source.jpg",
  "target_image": "images/0000000000_target.jpg"
}
```

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total samples | 30,788 |
| Total images | 61,576 |
| Groups | 90 |
| Image resolution | 512x512 |
| Avg image size | 186.95 KB |
| Total size | 10.99 GB |

## 🎯 Usage in InstaManip

```python
# In InstaManip config
data_dir: "ultraedit_processing/processed_data/train"
image_dir: "ultraedit_processing/processed_data"
data_group_path: "ultraedit_processing/processed_data/ultraedit_group_instruct.json"
```

---

**Dataset:** InstaEdit-30K  
**Project:** InstaManip (CVPR 2025)  
**Source:** UltraEdit (NeurIPS 2024)  
**License:** CC-BY-4.0
