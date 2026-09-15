# backend/model_loader.py
"""加载训练好的生成器权重，供后端推理使用"""
import os
import sys
import torch

# 把项目根目录加入 sys.path，保证能 import config 和 src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config
from src.models.generator import Generator


def load_generator(weights_path, device):
    """实例化 Generator 并加载权重，返回 eval 模式的模型"""
    cfg = Config()
    generator = Generator(cfg).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重文件: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator