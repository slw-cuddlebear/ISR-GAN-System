# src/data/preprocess.py
"""
DIV2K 数据预处理：
1. 收集并清洗原始高分辨率图像
2. 滑动窗口裁剪成 HR patch
3. 双三次下采样生成对应 LR patch
4. 划分 train / val / test
5. 生成 JSON 索引文件
"""
import os
import json
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import Config


def check_image_validity(img_path, min_size=64):
    """检查图像是否能读取，且尺寸不小于 min_size"""
    img = cv2.imread(img_path)
    if img is None:
        return False, "无法读取"
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False, f"尺寸过小 {w}x{h}"
    return True, ""


def sliding_window_crop(img_path, save_hr_dir, save_lr_dir, img_id):
    """对一张图做滑动窗口裁剪，同时生成 HR/LR 图像对"""
    cfg = Config()
    img = cv2.imread(img_path)
    if img is None:
        return 0

    h, w = img.shape[:2]

    # 图太小就先放大到至少能裁一个 patch
    if h < cfg.hr_patch_size or w < cfg.hr_patch_size:
        scale = cfg.hr_patch_size / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]

    count = 0
    for y in range(0, h - cfg.hr_patch_size + 1, cfg.crop_stride):
        for x in range(0, w - cfg.hr_patch_size + 1, cfg.crop_stride):
            hr = img[y:y + cfg.hr_patch_size, x:x + cfg.hr_patch_size]
            lr = cv2.resize(hr, (cfg.lr_patch_size, cfg.lr_patch_size),
                            interpolation=cv2.INTER_CUBIC)

            fname = f"{img_id}_{y}_{x}.png"
            cv2.imwrite(os.path.join(save_hr_dir, fname), hr)
            cv2.imwrite(os.path.join(save_lr_dir, fname), lr)
            count += 1
    return count


def create_index(split_name, hr_dir, lr_dir, out_path):
    """根据 HR/LR 目录生成 JSON 索引文件"""
    cfg = Config()
    pairs = []
    for f in sorted(os.listdir(hr_dir)):
        if not f.endswith(".png"):
            continue
        lr_path = os.path.join(lr_dir, f)
        if not os.path.exists(lr_path):
            continue
        pairs.append({
            "id": f.replace(".png", ""),
            "hr_path": os.path.join(hr_dir, f).replace("\\", "/"),
            "lr_path": lr_path.replace("\\", "/"),
            "hr_size": [cfg.hr_patch_size, cfg.hr_patch_size],
            "lr_size": [cfg.lr_patch_size, cfg.lr_patch_size],
        })

    index = {"split": split_name, "num_pairs": len(pairs), "pairs": pairs}
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(index, fp, indent=2, ensure_ascii=False)
    print(f"[{split_name}] 生成 {len(pairs)} 对图像")


def main():
    cfg = Config()

    # 1. 建目录
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(cfg.output_root, s, "HR"), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_root, s, "LR"), exist_ok=True)

    # 2. 收集所有原始图
    all_imgs = []
    for d in [cfg.raw_train_dir, cfg.raw_valid_dir]:
        if not os.path.exists(d):
            print(f"[警告] 找不到目录: {d}")
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                p = os.path.join(d, f)
                ok, msg = check_image_validity(p, cfg.lr_patch_size)
                if ok:
                    all_imgs.append(p)
                else:
                    print(f"[跳过] {f}: {msg}")

    print(f"有效原始图像: {len(all_imgs)} 张")
    if not all_imgs:
        print("没有可用图像，请先把 DIV2K 放到 dataset/ 下")
        return

    # 3. 划分 train / val / test
    n = len(all_imgs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_imgs, temp = train_test_split(all_imgs, train_size=n_train, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, train_size=n_val, random_state=42)

    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    # 4. 裁剪
    for name, imgs in split_map.items():
        hr_dir = os.path.join(cfg.output_root, name, "HR")
        lr_dir = os.path.join(cfg.output_root, name, "LR")
        for p in tqdm(imgs, desc=f"裁剪 {name}"):
            img_id = os.path.splitext(os.path.basename(p))[0]
            sliding_window_crop(p, hr_dir, lr_dir, img_id)

    # 5. 生成索引
    for name in splits:
        hr_dir = os.path.join(cfg.output_root, name, "HR")
        lr_dir = os.path.join(cfg.output_root, name, "LR")
        out = os.path.join(cfg.output_root, f"{name}_index.json")
        create_index(name, hr_dir, lr_dir, out)

    print("预处理完成。")


if __name__ == "__main__":
    main()