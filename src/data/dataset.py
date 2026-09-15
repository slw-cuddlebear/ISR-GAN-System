# src/data/dataset.py
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config


class SRDataset(Dataset):
    """超分数据集：按索引读取 HR/LR 图像对，归一化到 [-1, 1]"""

    def __init__(self, split="train", augment=True):
        cfg = Config()
        self.cfg = cfg
        self.split = split
        self.augment = augment and (split == "train")

        index_path = f"{cfg.data_root}/{split}_index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)["pairs"]

    def __len__(self):
        return len(self.pairs)

    def _normalize(self, img):
        """uint8 [0,255] -> float32 [-1,1]"""
        img = img.astype(np.float32) / 255.0
        img = (img - self.cfg.norm_mean) / self.cfg.norm_std
        return img

    def _augment_pair(self, hr, lr):
        """对 HR 和 LR 做相同的随机翻转 / 旋转"""
        if random.random() > 0.5:
            hr = np.flip(hr, axis=1).copy()
            lr = np.flip(lr, axis=1).copy()
        if random.random() > 0.5:
            hr = np.flip(hr, axis=0).copy()
            lr = np.flip(lr, axis=0).copy()
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            hr = np.rot90(hr, k, axes=(0, 1)).copy()
            lr = np.rot90(lr, k, axes=(0, 1)).copy()
        return hr, lr

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        hr = cv2.cvtColor(cv2.imread(pair["hr_path"]), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(pair["lr_path"]), cv2.COLOR_BGR2RGB)

        hr = self._normalize(hr)
        lr = self._normalize(lr)

        if self.augment:
            hr, lr = self._augment_pair(hr, lr)

        hr = torch.from_numpy(hr).permute(2, 0, 1).float()
        lr = torch.from_numpy(lr).permute(2, 0, 1).float()
        return lr, hr