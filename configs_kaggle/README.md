# Config nhẹ cho Kaggle

Repo InstaManip gốc yêu cầu mô hình 17B + SDXL nên vượt khả năng GPU T4/P100. Thư mục **configs_kaggle** cung cấp pipeline “mini” và “mock” để bạn thử nghiệm luồng dữ liệu, logging, checkpoint mà không cần tải checkpoint lớn.

## Thành phần

- `mock_training.yaml`: cấu hình mặc định (nhanh, embed 256, 3 epoch).
- `mock_training_large.yaml`: cấu hình “sâu hơn” (nhiều epoch, embed 512, batch lớn hơn) dành cho khi bạn muốn huấn luyện lâu hơn trên Kaggle.
- `scripts/train_kaggle_mock.py`: script PyTorch thuần, không phụ thuộc Hydra/DeepSpeed, dùng cấu hình YAML ở trên để huấn luyện mô hình mini.

## Cách chạy trên Kaggle Notebook

1. Chuẩn bị dữ liệu IP2P subset như notebook `kaggle_instamanip_mock.py`:
   - Symlink 1.500 nhóm vào `data/ip2p/` trong repo.
   - Gộp metadata/prompt thành `data/train/ip2p_subset.jsonl` và `data/ip2p_group_instruct.json`.
2. Chạy huấn luyện mock:

```bash
!cd /kaggle/working/InstaManip && python scripts/train_kaggle_mock.py \
    --config configs_kaggle/mock_training.yaml
```

Muốn chạy cấu hình sâu hơn: thay `mock_training.yaml` bằng `mock_training_large.yaml`. Script tự nhận `cuda` nếu có; có thể ép dùng CPU bằng `--device cpu`. Tham số `--max-samples` giúp giới hạn số mẫu đọc từ JSONL.

3. Kết quả (checkpoint `.pt`, `training_log.json`) nằm trong `output_dir` được chỉ định trong YAML (mặc định `/kaggle/working/train_output_kaggle_mock` hoặc `_mock_large`). Bạn có thể zip thư mục này để tải xuống.

## Tuỳ chỉnh chính

- `max_samples`: số mẫu đọc từ JSONL. `mock_training_large.yaml` mặc định là `null` (dùng toàn bộ subset).
- `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`: điều chỉnh nhịp train.
- `image_size`, `embed_dim`: điều chỉnh kích thước ảnh và độ lớn embedding.
- `train_split`: tỉ lệ train/val (ví dụ 0.9 nghĩa là 90% train, 10% val).

## Lưu ý

Pipeline này **không** tái hiện InstaManip gốc; đây chỉ là mô hình nhỏ để kiểm thử quy trình. Khi có GPU đủ mạnh, hãy quay lại các config chính trong `configs/` để huấn luyện mô hình đầy đủ.
