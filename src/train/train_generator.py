# src/train/train_generator.py
"""生成器 MSE 预训练脚本"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from src.data.dataset import SRDataset
from src.models.generator import Generator


def train():
    cfg = Config()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = cfg.device
    use_amp = (device.type == "cuda")

    # ---------- 数据集 ----------
    train_ds = SRDataset("train", augment=True)
    val_ds = SRDataset("val", augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---------- 模型 / 损失 / 优化器 ----------
    generator = Generator(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(generator.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    train_losses, val_losses = [], []

    for epoch in range(1, cfg.epochs + 1):
        generator.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    sr = generator(lr)
                    loss = criterion(sr, hr)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr = generator(lr)
                loss = criterion(sr, hr)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # ---------- 验证 ----------
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device, non_blocking=True)
                hr = hr.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        sr = generator(lr)
                        val_loss += criterion(sr, hr).item()
                else:
                    sr = generator(lr)
                    val_loss += criterion(sr, hr).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:3d} | Train {avg_train:.6f} | Val {avg_val:.6f}")

        # 每 10 轮保存一次中间权重
        if epoch % 10 == 0:
            ckpt = f"checkpoints/generator_epoch{epoch}.pth"
            torch.save(generator.state_dict(), ckpt)
            print(f"  已保存 {ckpt}")

    # ---------- 保存最终权重 ----------
    torch.save(generator.state_dict(), cfg.generator_weights)
    print(f"预训练完成，模型保存至 {cfg.generator_weights}")

    # ---------- 损失曲线 ----------
    plt.figure()
    plt.plot(range(1, cfg.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, cfg.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("logs/generator_pretrain_loss.png")
    plt.close()
    print("损失曲线已保存至 logs/generator_pretrain_loss.png")


if __name__ == "__main__":
    train()