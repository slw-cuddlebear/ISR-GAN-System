# backend/utils.py
"""图像预处理 / 后处理工具，支持任意尺寸输入"""
import cv2
import numpy as np
import torch


def _round_to_multiple(x, m=4):
    """向上取整到 m 的倍数"""
    return ((x + m - 1) // m) * m


def preprocess_image(img_bgr, max_side=512, min_side=32, multiple=4):
    """
    把任意尺寸的 BGR 图像调整为模型可以接受的输入。

    规则：
    1. 短边小于 min_side：等比放大
    2. 长边大于 max_side：等比缩小
    3. 保证宽高都是 multiple 的倍数（PixelShuffle 要求）

    返回：
        tensor: (1, 3, H, W) 范围 [-1, 1]
        (H, W): 调整后的尺寸，供后续参考
    """
    h, w = img_bgr.shape[:2]

    # 1. 短边太小 → 放大
    short = min(h, w)
    if short < min_side:
        scale = min_side / short
        h, w = int(round(h * scale)), int(round(w * scale))

    # 2. 长边太大 → 缩小
    long_side = max(h, w)
    if long_side > max_side:
        scale = max_side / long_side
        h, w = int(round(h * scale)), int(round(w * scale))

    # 3. 对齐到 multiple 的倍数
    h = _round_to_multiple(h, multiple)
    w = _round_to_multiple(w, multiple)

    # 4. resize + BGR→RGB
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5. 归一化到 [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, (h, w)


def tensor_to_bgr(tensor):
    """
    把 (1,3,H,W) 范围 [-1,1] 的 tensor 转回 uint8 BGR 图像
    """
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def encode_png_base64(img_bgr):
    """把 BGR 图像编码成 base64 PNG 字符串（不含 data: 前缀）"""
    import base64
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return base64.b64encode(buf).decode("utf-8")