import os
import torch


class Config:
    # ---------- 数据路径 ----------
    dataset_dir = "dataset"
    raw_train_dir = os.path.join(dataset_dir, "DIV2K_train_HR")
    raw_valid_dir = os.path.join(dataset_dir, "DIV2K_valid_HR")
    output_root = "processed"
    data_root = output_root

    # ---------- 图像参数 ----------
    scale_factor = 4
    hr_size = 256
    lr_size = 64
    hr_patch_size = 256
    lr_patch_size = 64
    crop_stride = 128
    norm_mean = 0.5
    norm_std = 0.5

    # ---------- 模型结构----------
    n_res_blocks = 4
    n_feats = 64
    growth_channel = 32

    # ---------- 训练参数 ----------
    batch_size = 16
    epochs = 50
    joint_epochs = 20
    lr = 1e-4
    disc_lr = 1e-4
    lambda_mse = 1.0
    lambda_adv = 1e-3
    lambda_percep = 1e-2

    # ---------- DataLoader ----------
    num_workers = 4
    pin_memory = True
    prefetch_factor = 2

    # ---------- 权重路径 ----------
    generator_weights = "checkpoints/generator_pretrain.pth"
    pretrained_generator = "checkpoints/generator_epoch20.pth"
    deploy_generator = "checkpoints/generator_gan_epoch20.pth"

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")