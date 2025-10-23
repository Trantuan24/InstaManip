# Thống Kê Dataset InstructPix2Pix

**Đường dẫn:** `D:\big_data\dataset\InstructPix2Pix_share00`  
**Dự án:** InstaManip - Image Manipulation with LLM

---

## 📊 Tổng Quan Dataset

| Thông Tin               | Giá Trị            |
| ----------------------- | ------------------ |
| **Tổng số thư mục con** | 10,434 thư mục     |
| **Tổng số file**        | 93,120 files       |
| **Tổng kích thước**     | ~14.04 GB          |
| **Mật độ trung bình**   | ~8.9 files/thư mục |

---

## 📁 Phân Loại File

| Loại File | Số Lượng     | Tỷ Lệ | Mô Tả                                    |
| --------- | ------------ | ----- | ---------------------------------------- |
| **JPG**   | 72,252 files | 77.6% | Ảnh nguồn và ảnh đích (cặp before/after) |
| **JSON**  | 10,434 files | 11.2% | File metadata chứa thông tin prompt      |
| **JSONL** | 10,434 files | 11.2% | File metadata dạng JSON Lines            |

---

## 🏗️ Cấu Trúc Thư Mục

### Quy Ước Đặt Tên

```
InstructPix2Pix_share00/
├── 0000132/
├── 0000145/
├── 0000231/
├── 0000336/
└── ... (10,434 thư mục con với tên dạng số 7 chữ số)
```

### Cấu Trúc Mỗi Thư Mục Con

Mỗi thư mục chứa:

- **Nhiều cặp ảnh**: `{id}_0.jpg` (ảnh gốc) và `{id}_1.jpg` (ảnh sau chỉnh sửa)
- **1 file `metadata.jsonl`**: Chứa thông tin về các phép biến đổi
- **1 file `prompt.json`**: Chứa instruction prompts

### Ví Dụ Thư Mục Mẫu (0000132)

```
0000132/
├── 127380018_0.jpg     (278.7 KB)
├── 127380018_1.jpg     (235.7 KB)
├── 1818564940_0.jpg    (257.9 KB)
├── 1818564940_1.jpg    (239.8 KB)
├── 3491690273_0.jpg    (262.8 KB)
├── 3491690273_1.jpg    (219.6 KB)
├── 3573774691_0.jpg    (237.7 KB)
├── 3573774691_1.jpg    (221.7 KB)
├── metadata.jsonl      (1.0 KB)
└── prompt.json         (0.3 KB)
```

---

## 📈 Phân Tích Thống Kê

### Thống Kê File Ảnh

- **Tổng số cặp ảnh**: ~36,126 cặp (72,252 ảnh / 2)
- **Trung bình mỗi thư mục**: ~6.9 cặp ảnh
- **Kích thước trung bình**: ~14.04 GB / 93,120 files ≈ 158 KB/file

### Phân Bố Dữ Liệu

```
Cấu trúc:
- Mỗi thư mục = 1 batch huấn luyện
- Mỗi cặp ảnh = 1 sample training
- JSON files = Metadata và prompts cho training
```

---

## 💾 Thông Tin Chi Tiết

Để xem thống kê chi tiết từng thư mục con, vui lòng tham khảo file:
**`dataset_detailed_statistics.csv`**

File CSV bao gồm các cột:

- `Folder_Name`: Tên thư mục
- `JPG_Files`: Số lượng file ảnh JPG
- `JSON_Files`: Số lượng file JSON
- `JSONL_Files`: Số lượng file JSONL
- `Total_Size_MB`: Tổng kích thước (MB)
- `Total_Files`: Tổng số file

---

## 🎯 Mục Đích Sử Dụng

Dataset này được sử dụng cho:

1. **Training MLLM (Multimodal Large Language Model)** - Image manipulation
2. **InstructPix2Pix Pipeline** - Image-to-image translation với text instructions
3. **Research** - Visual understanding và image editing tasks

---

## 📝 Ghi Chú

- Dataset đã được tổ chức sẵn theo batch để thuận tiện cho quá trình training
- Mỗi thư mục con có thể được xử lý độc lập
- Tương thích với training pipeline trong `src/train/train_model.py`

---
