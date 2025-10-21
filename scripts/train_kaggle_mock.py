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
import time


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
    num_visual_blocks: int = 4
    num_transformer_layers: int = 1
    num_transformer_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    loss_mse_weight: float = 1.0
    loss_contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
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
    def __init__(self, embed_dim: int = 256, num_blocks: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = 3
        channels = base_channels
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = channels
            channels = min(channels * 2, embed_dim)
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.proj(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyFusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_visual_blocks: int = 4,
        num_transformer_layers: int = 1,
        num_transformer_heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = TinyVisualEncoder(embed_dim=embed_dim, num_blocks=num_visual_blocks)
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_transformer_heads,
            dim_feedforward=embed_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.text_embedding(tokens)
        embeddings = self.positional_encoding(embeddings)
        key_padding = mask == 0
        transformed = self.text_transformer(embeddings, src_key_padding_mask=key_padding)
        mask_float = mask.unsqueeze(-1)
        summed = (transformed * mask_float).sum(dim=1)
        lengths = mask_float.sum(dim=1).clamp_min(1.0)
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


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.transpose(0, 1) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_i + loss_t)


# -----------------------------
# Huấn luyện
# -----------------------------

def evaluate(model: TinyFusionModel, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    mse_accum = 0.0
    contrastive_accum = 0.0
    with torch.no_grad():
        for batch in loader:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            tokens = batch["instruction_tokens"].to(device)
            mask = batch["instruction_mask"].to(device)

            pred, _ = model(source, tokens, mask)
            target_feat = model.visual_encoder(target)
            loss_val = 0.0
            if cfg.loss_mse_weight > 0:
                mse = F.mse_loss(pred, target_feat)
                loss_val += cfg.loss_mse_weight * mse
                mse_accum += mse.item()
            if cfg.loss_contrastive_weight > 0:
                cl = contrastive_loss(pred, target_feat, temperature=cfg.contrastive_temperature)
                loss_val += cfg.loss_contrastive_weight * cl
                contrastive_accum += cl.item()
            total_loss += loss_val.item()
            total_batches += 1
    return {
        "val_loss": total_loss / max(total_batches, 1),
        "val_mse": mse_accum / max(total_batches, 1) if total_batches > 0 else 0.0,
        "val_contrastive": contrastive_accum / max(total_batches, 1) if total_batches > 0 else 0.0,
    }


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

    model = TinyFusionModel(
        vocab_size=dataset.vocab.size,
        embed_dim=cfg.embed_dim,
        num_visual_blocks=cfg.num_visual_blocks,
        num_transformer_layers=cfg.num_transformer_layers,
        num_transformer_heads=cfg.num_transformer_heads,
        ff_multiplier=cfg.ff_multiplier,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    log_path = output_dir / "training_log.json"
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_mse": [], "val_contrastive": []}

    global_step = 0
    best_val = float("inf")
    best_epoch = 0
    best_ckpt: Path | None = None
    start_time = time.time()

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
            loss = 0.0
            if cfg.loss_mse_weight > 0:
                loss = loss + cfg.loss_mse_weight * F.mse_loss(pred, target_feat)
            if cfg.loss_contrastive_weight > 0:
                loss = loss + cfg.loss_contrastive_weight * contrastive_loss(
                    pred, target_feat, temperature=cfg.contrastive_temperature
                )

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
            metrics = evaluate(model, val_loader, device, cfg)
            history["val_loss"].append(metrics["val_loss"])
            history["val_mse"].append(metrics["val_mse"])
            history["val_contrastive"].append(metrics["val_contrastive"])
            print(f"[Validation] Epoch {epoch}: loss={metrics['val_loss']:.6f}")
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                best_epoch = epoch

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
            if epoch == best_epoch:
                best_ckpt = ckpt_path

        with log_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    elapsed = time.time() - start_time
    summary = {
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "total_steps": global_step,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
        "elapsed_seconds": elapsed,
    }
    history["summary"] = summary
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Huấn luyện hoàn tất. Log lưu ở", log_path)
    print("Tóm tắt:", summary)


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
