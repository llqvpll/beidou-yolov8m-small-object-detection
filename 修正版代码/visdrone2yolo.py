r"""
VisDrone2019-DET 标注 -> YOLO 格式 转换脚本

为什么需要它：
    VisDrone2019-DET 的标注 **不是** YOLO 格式。官方每行是 8 个逗号分隔的字段：

        bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

    而 ultralytics 训练需要的是每行 5 个字段、且都归一化到 [0,1] 的 YOLO 格式：

        class_id  cx  cy  w  h

    所以必须转换一次，否则 VisDrone.yaml 指向的 labels/ 根本不存在。

类别的处理：
    官方 category 共 12 个值（0=ignored regions, 1~10=有效类, 11=others）。
    score=0 表示「该区域应被忽略」。本脚本按通行做法：
      - 丢掉 score == 0 的行
      - 丢掉 category == 0（ignored regions）和 category == 11（others）
      - category 1~10 线性映射为 0~9，与 VisDrone.yaml 的 names 一一对应

用法：
    # 最常见：一次把 train / val 都转好，图片直接软链接（不额外占磁盘）
    python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --dst D:/datasets/VisDrone

    # 图片想真正复制一份（跨盘、或软链接失败时用）
    python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --dst D:/datasets/VisDrone --mode copy

    # 只转其中一个 split
    python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --only train

转换完成后，把 VisDrone.yaml 里的 path 指到 --dst 即可：
    path: D:/datasets/VisDrone
"""
import argparse
import os
import shutil
import sys

# 官方 category -> YOLO class id（其余一律丢弃）
CATEGORY_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

# VisDrone 官方目录名 -> 输出 split 名
SPLIT_DIRS = {
    "train": ["VisDrone2019-DET-train", "train"],
    "val": ["VisDrone2019-DET-val", "val"],
    "test": ["VisDrone2019-DET-test-dev", "VisDrone2019-DET-test", "test"],
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def find_split_dir(src: str, split: str):
    """在 src 下找出某个 split 的目录（官方命名 / 简写都认）。"""
    for cand in SPLIT_DIRS[split]:
        p = os.path.join(src, cand)
        if os.path.isdir(p):
            return p
    return None


def image_size(path: str):
    """取图片宽高。优先 PIL，退化到 cv2，再退化到手动读 JPEG 头。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        pass
    try:
        import cv2
        im = cv2.imread(path)
        if im is not None:
            h, w = im.shape[:2]
            return (w, h)
    except Exception:
        pass
    return None


def link_or_copy(src_file: str, dst_file: str, mode: str):
    """把图片放到输出目录。mode=symlink 用软链接省磁盘；失败则自动退回复制。"""
    if os.path.exists(dst_file):
        return "skip"
    if mode == "symlink":
        try:
            os.symlink(os.path.abspath(src_file), dst_file)
            return "link"
        except (OSError, NotImplementedError):
            # Windows 上普通用户无权限创建符号链接时会走到这里 / 或需要开发者模式
            try:
                shutil.copy2(src_file, dst_file)
                return "copy"
            except Exception:
                return "fail"
    try:
        shutil.copy2(src_file, dst_file)
        return "copy"
    except Exception:
        return "fail"


def convert_split(src_root, dst_root, split, mode, verbose=True):
    src_dir = find_split_dir(src_root, split)
    if src_dir is None:
        print(f"[skip] 未找到 {split} 目录（在 {src_root} 下）")
        return None

    img_dir = os.path.join(src_dir, "images")
    ann_dir = os.path.join(src_dir, "annotations")
    if not os.path.isdir(img_dir):
        print(f"[skip] {src_dir} 下没有 images/")
        return None

    out_img = os.path.join(dst_root, "images", split)
    out_lbl = os.path.join(dst_root, "labels", split)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    has_ann = os.path.isdir(ann_dir)
    if not has_ann:
        print(f"[warn] {src_dir} 没有 annotations/（test 集通常就没有标注）—— 只拷图片。")

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT)]
    files.sort()
    n_img = n_box = n_drop = n_empty = 0
    n_link = n_copy = n_fail = 0

    for i, fname in enumerate(files, 1):
        stem, _ = os.path.splitext(fname)
        src_img = os.path.join(img_dir, fname)
        dst_img = os.path.join(out_img, fname)

        act = link_or_copy(src_img, dst_img, mode)
        if act == "fail":
            n_fail += 1
            continue
        n_link += act == "link"
        n_copy += act == "copy"
        n_img += 1

        if not has_ann:
            continue

        ann_path = os.path.join(ann_dir, stem + ".txt")
        wh = image_size(src_img)
        if wh is None:
            print(f"  [warn] 读不出尺寸，跳过: {fname}")
            continue
        W, H = wh
        if W <= 0 or H <= 0:
            continue

        lines = []
        if os.path.isfile(ann_path):
            with open(ann_path, "r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    parts = raw.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        x, y, w, h = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                        score = int(float(parts[4]))
                        cat = int(float(parts[5]))
                    except ValueError:
                        continue

                    if score == 0:                    # 官方约定：0 = 忽略区域
                        n_drop += 1
                        continue
                    if cat not in CATEGORY_MAP:       # 0=ignored regions, 11=others
                        n_drop += 1
                        continue
                    if w <= 1 or h <= 1:              # 退化的框
                        n_drop += 1
                        continue

                    # 裁剪到图像范围内
                    x1 = max(0.0, x)
                    y1 = max(0.0, y)
                    x2 = min(float(W), x + w)
                    y2 = min(float(H), y + h)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 1 or bh <= 1:
                        n_drop += 1
                        continue

                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    nw = bw / W
                    nh = bh / H
                    cls = CATEGORY_MAP[cat]
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    n_box += 1

        with open(os.path.join(out_lbl, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")
        if not lines:
            n_empty += 1

        if verbose and (i % 500 == 0 or i == len(files)):
            print(f"  [{split}] {i}/{len(files)} 张，已写入 {n_box} 个框")

    return {
        "split": split, "images": n_img, "boxes": n_box,
        "empty": n_empty, "dropped": n_drop,
        "link": n_link, "copy": n_copy, "fail": n_fail,
    }


def main():
    ap = argparse.ArgumentParser(
        description="把 VisDrone2019-DET 标注转换成 YOLO 格式（class cx cy w h）")
    ap.add_argument("--src", required=True,
                    help="VisDrone2019-DET 根目录（下面有 VisDrone2019-DET-train / -val / -test-dev）")
    ap.add_argument("--dst", required=True,
                    help="输出根目录，将被 VisDrone.yaml 的 path 指向")
    ap.add_argument("--mode", default="symlink", choices=["symlink", "copy"],
                    help="图片处理方式：symlink=软链接省磁盘（Windows 可能需开发者模式）、copy=真复制")
    ap.add_argument("--only", default=None, choices=["train", "val", "test"],
                    help="只转换其中一个 split（默认 train + val + test）")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isdir(src):
        print(f"[ERROR] 源目录不存在: {src}")
        sys.exit(1)

    splits = [args.only] if args.only else ["train", "val", "test"]

    print("=" * 60)
    print("VisDrone2019-DET -> YOLO 转换")
    print(f"  源: {src}")
    print(f"  目标: {dst}")
    print(f"  模式: {args.mode}   split: {', '.join(splits)}")
    print("=" * 60)

    os.makedirs(dst, exist_ok=True)
    results = []
    for sp in splits:
        r = convert_split(src, dst, sp, args.mode)
        if r:
            results.append(r)

    print()
    print("-" * 60)
    for r in results:
        print(f"[{r['split']:5s}] 图片 {r['images']:>6d} 张 | 标注框 {r['boxes']:>7d} 个 | "
              f"空标注 {r['empty']:>5d} 张 | 丢弃 {r['dropped']:>6d} 个 | "
              f"链接 {r['link']} / 复制 {r['copy']}" + (f" / 失败 {r['fail']}" if r["fail"] else ""))
    print("-" * 60)

    # 顺手校验一下命名是否一一对应
    for r in results:
        sp = r["split"]
        lbl_dir = os.path.join(dst, "labels", sp)
        if not os.path.isdir(lbl_dir):
            continue
        n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])
        flag = "OK" if n_lbl == r["images"] else "!! 数量不一致"
        print(f"[check] {sp}: images={r['images']}  labels={n_lbl}   {flag}")

    print()
    print("下一步：把 VisDrone.yaml 里的 path 改成下面这行，然后就能训练了")
    print(f"    path: {dst}")


if __name__ == "__main__":
    main()
