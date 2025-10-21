#!/usr/bin/env python3
"""
Huấn luyện mô hình mini (mock) cho InstaManip trên Kaggle.

- Đọc JSONL (định dạng source/target/instruction) và ảnh tương ứng.
- Áp dụng visual encoder nhỏ + text embedding đơn giản.
- Mục tiêu: dự đoán embedding ảnh target từ ảnh source + instruction (MSE).

Phù hợp để kiểm thử pipeline trên GPU T4/CPU mà không cần SEED-X 17B hay SDXL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm.auto import tqdm
import yaml
import re


# -----------------------------
# cấu hình & tiện ích
# -----------------------------

@dataclass
class TrainConfig:
    dataset_path: str
    image_root: str
    output_dir: str
    max_samples: int | None = None
    train_split: float = 0.9
    batch_size: int = 8
    num_workers: int = 2
    num_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    image_size: int = 128
    embed_dim: int = 256
    vocab_max_size: int = 4000
    device: str = "auto"
    seed: int = 42
    log_interval: int = 20
    val_interval: int = 1
    save_model: bool = True
    save_every_epochs: int = 1

    @staticmethod
    def from_yaml(path: Path) -> "TrainConfig":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device(device_str: str) -> torch.device:
    if device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# -----------------------------
# xử lý ngôn ngữ đơn giản
# -----------------------------

TOKEN_REGEX = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_REGEX.findall(text.lower())


class Vocabulary:
    def __init__(self, max_size: int = 4000) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.max_size = max_size
        self.token_to_idx: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_token: List[str] = [self.pad_token, self.unk_token]

    def add_sentence(self, text: str) -> None:
        for token in simple_tokenize(text):
            if token not in self.token_to_idx:
                if len(self.token_to_idx) >= self.max_size:
                    return
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    def encode(self, text: str) -> List[int]:
        tokens = simple_tokenize(text)
        unk_idx = self.token_to_idx[self.unk_token]
        return [self.token_to_idx.get(tok, unk_idx) for tok in tokens] or [unk_idx]

    @property
    def size(self) -> int:
        return len(self.idx_to_token)

    @property
    def pad_index(self) -> int:
        return self.token_to_idx[self.pad_token]


# -----------------------------
# Dataset
# -----------------------------

class MockManipDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        image_root: Path,
        image_size: int = 128,
        max_samples: int | None = None,
        vocab_max_size: int = 4000,
    ) -> None:
        self.image_root = image_root
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

        self.records: List[Dict[str, str]] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {"source_image", "target_image", "instruction"}
                if not required.issubset(record):
                    continue
                self.records.append(record)
                if max_samples is not None and len(self.records) >= max_samples:
                    break

        if not self.records:
            raise ValueError(f"Không tìm thấy dữ liệu hợp lệ trong {jsonl_path}")

        self.vocab = Vocabulary(max_size=vocab_max_size)
        for rec in self.records:
            self.vocab.add_sentence(rec.get("instruction", ""))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        source_path = self.image_root / rec["source_image"]
        target_path = self.image_root / rec["target_image"]

        source_img = self._load_image(source_path)
        target_img = self._load_image(target_path)

        instruction_ids = torch.tensor(self.vocab.encode(rec.get("instruction", "")), dtype=torch.long)

        return {
            "source": source_img,
            "target": target_img,
            "instruction_ids": instruction_ids,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_idx: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    sources = torch.stack([item["source"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])

    lengths = [item["instruction_ids"].shape[0] for item in batch]
    max_len = max(lengths)

    tokens = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        ids = item["instruction_ids"]
        tokens[i, : ids.shape[0]] = ids
        mask[i, : ids.shape[0]] = 1.0

    return {
        "source": sources,
        "target": targets,
        "instruction_tokens": tokens,
        "instruction_mask": mask,
    }


# -----------------------------
# Mô hình nhỏ
# -----------------------------

class TinyVisualEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.proj(x)
        return x


class TinyFusionModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = TinyVisualEncoder(embed_dim=embed_dim)
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.text_embedding(tokens)  # (B, L, D)
        mask = mask.unsqueeze(-1)  # (B, L, 1)
        summed = (embeddings * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1.0)
        avg = summed / lengths
        return self.text_norm(avg)

    def forward(
        self,
        source: torch.Tensor,
        instruction_tokens: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        source_feat = self.visual_encoder(source)
        text_feat = self.encode_text(instruction_tokens, instruction_mask)
        combined = torch.cat([source_feat, text_feat], dim=-1)
        pred = self.predictor(combined)
        return pred, source_feat


# -----------------------------
# Huấn luyện
# -----------------------------

def evaluate(model: TinyFusionModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            tokens = batch["instruction_tokens"].to(device)
            mask = batch["instruction_mask"].to(device)

            pred, _ = model(source, tokens, mask)
            target_feat = model.visual_encoder(target)
            loss = F.mse_loss(pred, target_feat)
            total_loss += loss.item()
            total_batches += 1
    return {"mse": total_loss / max(total_batches, 1)}


def train_loop(cfg: TrainConfig) -> None:
    device = build_device(cfg.device)
    set_seed(cfg.seed)

    dataset_path = Path(cfg.dataset_path)
    image_root = Path(cfg.image_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MockManipDataset(
        jsonl_path=dataset_path,
        image_root=image_root,
        image_size=cfg.image_size,
        max_samples=cfg.max_samples,
        vocab_max_size=cfg.vocab_max_size,
    )

    train_len = int(len(dataset) * cfg.train_split)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))

    pad_idx = dataset.vocab.pad_index

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_batch(b, pad_idx=pad_idx),
    )

    model = TinyFusionModel(vocab_size=dataset.vocab.size, embed_dim=cfg.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    log_path = output_dir / "training_log.json"
    history: Dict[str, List[float]] = {"train_loss": [], "val_mse": []}

    global_step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}"), start=1):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            tokens = batch["instruction_tokens"].to(device)
            mask = batch["instruction_mask"].to(device)

            pred, _ = model(source, tokens, mask)
            target_feat = model.visual_encoder(target)
            loss = F.mse_loss(pred, target_feat)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if step % cfg.log_interval == 0:
                avg_loss = running_loss / cfg.log_interval
                history["train_loss"].append(avg_loss)
                running_loss = 0.0

        if epoch % cfg.val_interval == 0:
            metrics = evaluate(model, val_loader, device)
            history["val_mse"].append(metrics["mse"])
            print(f"[Validation] Epoch {epoch}: MSE={metrics['mse']:.6f}")

        if cfg.save_model and (epoch % cfg.save_every_epochs == 0):
            ckpt_path = output_dir / f"mock_model_epoch{epoch}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab": dataset.vocab.token_to_idx,
                    "config": cfg.__dict__,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Đã lưu checkpoint tại {ckpt_path}")

        with log_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("Huấn luyện hoàn tất. Log lưu ở", log_path)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tiny InstaManip mock model (phù hợp Kaggle).")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn YAML cấu hình (ví dụ configs_kaggle/mock_training.yaml).")
    parser.add_argument("--device", type=str, default=None, help='Ghi đè device (vd "cuda", "cpu").')
    parser.add_argument("--max-samples", type=int, default=None, help="Ghi đè số mẫu tối đa đọc từ JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(Path(args.config))

    if args.device:
        cfg.device = args.device
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples

    train_loop(cfg)


if __name__ == "__main__":
    main()
