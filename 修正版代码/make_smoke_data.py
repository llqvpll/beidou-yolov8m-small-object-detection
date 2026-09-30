"""生成一个极小的合成数据集，仅用于"冒烟测试"——验证训练链路能否跑通。

为什么需要它：
    完整的 VisDrone2019-DET 约 1.5GB，下载/解压都麻烦。而冒烟测试的目的只是确认
    「自定义模块 → 建模 → 前向 → 反向 → 验证」这条链路是通的，不需要真实数据。
    所以这里生成几十张 640x640 的随机图 + 对应的 YOLO 格式标签，
    类别名与 VisDrone 完全一致（10 类），保证模型结构（Detect nc=10）与正式实验一致。

产出：
    冒烟测试数据集/
      ├── images/{train,val}/*.jpg
      ├── labels/{train,val}/*.txt
      └── smoke.yaml

用法：
    python make_smoke_data.py            # 默认 24 训练 / 8 验证
    python make_smoke_data.py --train 40 --val 10
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "冒烟测试数据集"
SIZE = 640
CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def make_one(seed: int) -> tuple[Image.Image, list[str]]:
    """造一张图 + 若干小目标框（返回归一化后的 YOLO 标签行）。"""
    rng = random.Random(seed)
    npr = np.random.default_rng(seed)

    # 背景：低频随机噪声，模拟"地面/天空"色块
    bg = npr.integers(60, 200, size=(SIZE // 32, SIZE // 32, 3), dtype=np.uint8)
    img = Image.fromarray(bg).resize((SIZE, SIZE), Image.BILINEAR)
    draw = ImageDraw.Draw(img)

    labels: list[str] = []
    n_obj = rng.randint(2, 8)
    for _ in range(n_obj):
        # 小目标：短边 12~28 px（VisDrone 里大量目标都小于 32px）
        w = rng.randint(12, 28)
        h = rng.randint(12, 28)
        x = rng.randint(0, SIZE - w - 1)
        y = rng.randint(0, SIZE - h - 1)
        color = tuple(int(c) for c in rng.choices(range(40, 256), k=3))
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(255, 255, 255))

        cx = (x + w / 2) / SIZE
        cy = (y + h / 2) / SIZE
        labels.append(f"{rng.randrange(len(CLASSES))} {cx:.6f} {cy:.6f} {w / SIZE:.6f} {h / SIZE:.6f}")
    return img, labels


def build(split: str, count: int, base_seed: int) -> None:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img, labels = make_one(base_seed + i)
        stem = f"{split}_{i:04d}"
        img.save(OUT / "images" / split / f"{stem}.jpg", quality=88)
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def write_yaml() -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (OUT / "smoke.yaml").write_text(
        "# 合成冒烟测试数据集（自动生成，仅供链路验证，不代表任何精度）\n"
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "# 与 VisDrone2019-DET 完全一致的 10 类，保证 Detect nc=10 与正式实验一致\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="生成冒烟测试用的极小合成数据集")
    p.add_argument("--train", type=int, default=24)
    p.add_argument("--val", type=int, default=8)
    args = p.parse_args()

    build("train", args.train, base_seed=1000)
    build("val", args.val, base_seed=9000)
    write_yaml()
    print(f"[ok] 已生成：{OUT}")
    print(f"     train {args.train} 张 / val {args.val} 张，{len(CLASSES)} 类")
    print(f"     配置：{OUT / 'smoke.yaml'}")


if __name__ == "__main__":
    main()
