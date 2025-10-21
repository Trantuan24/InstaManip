# Cấu hình nhẹ cho Kaggle

Repo gốc InstaManip yêu cầu mô hình 17B + SDXL ⇒ vượt khả năng GPU T4/P100 trên Kaggle. Bộ **configs_kaggle** này cung cấp pipeline “mock” giúp bạn thử nghiệm luồng huấn luyện với mô hình nhỏ gọn (CNN + MLP) mà không cần tải các checkpoint lớn.

## Thành phần

- `mock_training.yaml`: thông số cho script huấn luyện nhẹ (`scripts/train_kaggle_mock.py`). Bạn có thể chỉnh dữ liệu đầu vào, batch size, số epoch, kích thước ảnh, v.v.
- `scripts/train_kaggle_mock.py`: script độc lập, không phụ thuộc Hydra hay Deepspeed, dùng PyTorch thuần để huấn luyện mô hình mini nhằm kiểm thử pipeline.

## Cách chạy trên Kaggle Notebook

1. Chuẩn bị dữ liệu theo notebook demo (symlink 1.500 nhóm IP2P vào `data/ip2p/`, gộp JSONL thành `data/train/ip2p_subset.jsonl`).
2. Chạy lệnh huấn luyện nhẹ:

```bash
!cd /kaggle/working/InstaManip && python scripts/train_kaggle_mock.py \
    --config configs_kaggle/mock_training.yaml
```

> Lưu ý: script sẽ tự phát hiện `cuda` nếu khả dụng; bạn có thể ép dùng CPU bằng `--device cpu`.

3. Kết quả (checkpoint `.pt` + log JSON) sẽ được lưu trong thư mục `output_dir` chỉ định trong YAML (mặc định `/kaggle/working/train_output_kaggle_mock`). Bạn có thể zip thư mục này để tải về.

## Tuỳ chỉnh chính

- `max_samples`: số mẫu đọc từ JSONL (giúp giảm thời gian nếu muốn thử nhanh).
- `batch_size`, `num_epochs`, `learning_rate`: điều chỉnh nhịp train.
- `image_size`, `embed_dim`: kiểm soát kích thước ảnh và độ lớn embedding của mô hình.
- `train_split`: tỉ lệ chia train/val (ví dụ 0.9 tức 90% train, 10% val).

## Giới hạn

Pipeline này **không** tái tạo InstaManip gốc; nó dùng mô hình mini để kiểm thử luồng dữ liệu, logging và xuất checkpoint. Khi bạn có hạ tầng GPU mạnh, hãy quay lại các config gốc trong `configs/` để huấn luyện mô hình đầy đủ.*** End Patch*** End Patch
