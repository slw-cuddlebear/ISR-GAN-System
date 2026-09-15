# backend/app.py
"""Flask 后端：加载生成器模型，提供超分接口，并托管前端页面"""
import os
import sys
import torch
from flask import Flask, request, jsonify, send_from_directory

# 把项目根目录加入 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config
from backend.model_loader import load_generator
from backend.utils import preprocess_image, tensor_to_bgr, encode_png_base64

cfg = Config()

# ---------- Flask 应用 ----------
# static_folder 指向前端目录，static_url_path="" 让 / 直接返回 index.html
frontend_dir = os.path.join(ROOT, "frontend")
app = Flask(__name__, static_folder=frontend_dir, static_url_path="")

# ---------- 设备与模型 ----------
device = cfg.device
generator = None


def resolve_weights_path():
    """按优先级寻找可用的权重文件"""
    candidates = [
        cfg.deploy_generator,
        "checkpoints/generator_gan_epoch20.pth",
        "checkpoints/generator_final.pth",
        "checkpoints/generator_pretrain.pth",
        "checkpoints/generator_epoch50.pth",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def init_model():
    global generator
    weights_path = resolve_weights_path()
    if weights_path is None:
        print("[警告] 未找到任何权重文件，接口将无法使用。")
        print("       请把训练好的 .pth 放到 checkpoints/ 目录下。")
        return
    print(f"正在加载模型: {weights_path}")
    generator = load_generator(weights_path, device)
    print(f"模型加载完成，运行设备: {device}")


# 启动时加载模型
init_model()


# ---------- 路由 ----------
@app.route("/")
def index():
    """返回前端首页"""
    index_file = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_file):
        return "前端页面还没建好，请先创建 frontend/index.html", 404
    return send_from_directory(frontend_dir, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": generator is not None,
        "device": str(device),
    })


@app.route("/superres_base64", methods=["POST"])
def superres_base64():
    """接收图片，返回超分后的 base64 PNG"""
    if generator is None:
        return jsonify({"error": "模型未加载"}), 503

    if "image" not in request.files:
        return jsonify({"error": "请求中没有 image 字段"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "文件为空"}), 400

    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"error": "文件内容为空"}), 400

    import cv2
    import numpy as np
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "无法解析图片，请确认格式"}), 400

    # 预处理（任意尺寸 → 合理尺寸）
    tensor, (in_h, in_w) = preprocess_image(img_bgr)
    tensor = tensor.to(device)

    # 推理
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                sr_tensor = generator(tensor)
        else:
            sr_tensor = generator(tensor)

    # 后处理
    sr_bgr = tensor_to_bgr(sr_tensor)
    out_h, out_w = sr_bgr.shape[:2]

    img_base64 = encode_png_base64(sr_bgr)
    return jsonify({
        "image_base64": img_base64,
        "input_size": [in_w, in_h],
        "output_size": [out_w, out_h],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)