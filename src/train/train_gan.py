# src/train/train_gan.py
"""GAN 联合训练脚本：加载 MSE 预训练的生成器，与判别器对抗训练"""
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
from src.models.discriminator import ConditionalDiscriminator
from src.losses.losses import RegionWeightedMSELoss


def train():
    cfg = Config()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    device = cfg.device
    use_amp = (device.type == "cuda")

    # ---------- 数据集 ----------
    train_ds = SRDataset("train", augment=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )

    # ---------- 模型 ----------
    generator = Generator(cfg).to(device)
    discriminator = ConditionalDiscriminator().to(device)

    # 加载 MSE 预训练的生成器权重
    if os.path.exists(cfg.pretrained_generator):
        generator.load_state_dict(torch.load(cfg.pretrained_generator, map_location=device))
        print(f"已加载预训练生成器: {cfg.pretrained_generator}")
    else:
        print(f"[警告] 找不到 {cfg.pretrained_generator}，将从头开始训练生成器")

    # ---------- 损失 / 优化器 ----------
    mse_loss = RegionWeightedMSELoss()
    disc_criterion = nn.BCEWithLogitsLoss()
    g_optimizer = Adam(generator.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=cfg.disc_lr, betas=(0.9, 0.999))

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    g_losses, d_losses = [], []

    for epoch in range(1, cfg.joint_epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch}/{cfg.joint_epochs}")
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            batch_size = lr.size(0)

            # 全 1 掩码（暂时没有真实语义图）
            mask = torch.ones(batch_size, 1, cfg.hr_size, cfg.hr_size,
                              device=device, dtype=lr.dtype)

            # ---------- 1. 训练判别器 ----------
            d_optimizer.zero_grad()
            with torch.no_grad():
                sr = generator(lr)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    real_pred = discriminator(hr, mask)
                    fake_pred = discriminator(sr.detach(), mask)
                    real_loss = disc_criterion(real_pred, torch.ones_like(real_pred))
                    fake_loss = disc_criterion(fake_pred, torch.zeros_like(fake_pred))
                    d_loss = (real_loss + fake_loss) / 2
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                real_pred = discriminator(hr, mask)
                fake_pred = discriminator(sr.detach(), mask)
                real_loss = disc_criterion(real_pred, torch.ones_like(real_pred))
                fake_loss = disc_criterion(fake_pred, torch.zeros_like(fake_pred))
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()

            # ---------- 2. 训练生成器 ----------
            g_optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    sr = generator(lr)
                    adv = disc_criterion(discriminator(sr, mask), torch.ones_like(real_pred))
                    mse = mse_loss(sr, hr, mask=None)
                    g_loss = cfg.lambda_mse * mse + cfg.lambda_adv * adv
                scaler.scale(g_loss).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                sr = generator(lr)
                adv = disc_criterion(discriminator(sr, mask), torch.ones_like(real_pred))
                mse = mse_loss(sr, hr, mask=None)
                g_loss = cfg.lambda_mse * mse + cfg.lambda_adv * adv
                g_loss.backward()
                g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"Epoch {epoch:2d} | G Loss {avg_g:.6f} | D Loss {avg_d:.6f}")

        # 每 5 轮保存一次检查点
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_gan_epoch{epoch}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch{epoch}.pth")
            print(f"  已保存 epoch {epoch} 检查点")

    # ---------- 保存最终权重 ----------
    torch.save(generator.state_dict(), "checkpoints/generator_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator_final.pth")
    print("GAN 联合训练完成。")

    # ---------- 损失曲线 ----------
    plt.figure()
    plt.plot(range(1, cfg.joint_epochs + 1), g_losses, label="Generator Loss")
    plt.plot(range(1, cfg.joint_epochs + 1), d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("logs/gan_training_loss.png")
    plt.close()
    print("损失曲线已保存至 logs/gan_training_loss.png")


if __name__ == "__main__":
    train()