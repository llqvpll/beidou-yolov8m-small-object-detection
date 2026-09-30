# -*- coding: utf-8 -*-
"""汇总报告用的真实实验数据（全部来自本机可复现产物，不引用任何外部数字）。

产出：`报告数据.json` + 控制台表格。

做三件事：
  1. 逐个 run 解析 results.csv → best(mAP50) / best_fitness / 末轮 P/R / 轮数
  2. 逐个权重跑一次 model.val()（VisDrone val 548 张）→ 权威 P/R/mAP + 逐类 AP
  3. 记录参数量、层数、GFLOPs，供消融表互校

⚠ 一律显式用 Python310；自定义模块需已由 install_custom_modules.py 注册。
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUNS = Path(r"C:/yolo_runs/train")
DATA = str(HERE / "VisDrone.yaml")
OUT = HERE / "报告数据.json"

# 待评的三个权重：本项目的改进版（v1 4尺度）、推荐版（v4 3尺度）、外部标准基线
WEIGHTS = [
    ("v1_improved_4scale", "v1 改进版（P2 四尺度 + DySample/EMA/DualConv）",
     RUNS / "visdrone_v1/weights/best.pt"),
    ("v4_slimneck_nop5", "v4 推荐版（Slim-Neck + 剪 P5，三尺度）",
     RUNS / "visdrone_v4/weights/best.pt"),
    ("std_yolov8m_visdrone", "标准 YOLOv8m-VisDrone（外部现成权重）",
     Path(r"C:/visdrone_weights_eval/best(yolov8m-visdrone).pt")),
]


def parse_csv(run_dir: Path) -> dict:
    f = run_dir / "results.csv"
    rows = list(csv.DictReader(io.open(f, encoding="utf-8")))

    def g(r, k):
        try:
            return float(r[k])
        except Exception:
            return None

    pts = [(g(r, "metrics/mAP50(B)"), g(r, "metrics/mAP50-95(B)"), int(r["epoch"])) for r in rows]
    pts = [p for p in pts if p[0] is not None]
    bm = max(pts, key=lambda p: p[0])
    bf = max(pts, key=lambda p: 0.1 * p[0] + 0.9 * p[1])
    last = rows[-1]
    return {
        "epochs": len(rows),
        "best_map50": round(bm[0], 5),
        "best_map50_epoch": bm[2],
        "best_fitness_map50": round(bf[0], 5),
        "best_fitness_epoch": bf[2],
        "best_map50_95": round(bm[1], 5),
        "last_precision": round(g(last, "metrics/precision(B)"), 4),
        "last_recall": round(g(last, "metrics/recall(B)"), 4),
        "last_map50": round(g(last, "metrics/mAP50(B)"), 5),
        "last_map50_95": round(g(last, "metrics/mAP50-95(B)"), 5),
    }


def main() -> int:
    from ultralytics import YOLO  # noqa: PLC0415

    out: dict = {"runs": {}, "val": {}}

    print("=" * 78)
    print("[1/2] 解析 results.csv")
    print("=" * 78)
    for d in sorted(RUNS.iterdir()):
        if (d / "results.csv").is_file():
            out["runs"][d.name] = parse_csv(d)
            r = out["runs"][d.name]
            print(f"{d.name:20s} epochs={r['epochs']:4d}  "
                  f"best mAP50={r['best_map50']:.5f}@{r['best_map50_epoch']:3d}  "
                  f"fitness mAP50={r['best_fitness_map50']:.5f}@{r['best_fitness_epoch']:3d}")

    print()
    print("=" * 78)
    print("[2/2] 逐个权重跑 model.val()（VisDrone val）")
    print("=" * 78)
    tmpdir = HERE / "_val_tmp"
    for key, label, wp in WEIGHTS:
        if not wp.is_file():
            print(f"[跳过] 权重不存在：{wp}")
            continue
        # 外部权重文件名带括号，先拷成干净名字，避免路径解析踩坑
        tmp = tmpdir / f"{key}.pt"
        tmpdir.mkdir(exist_ok=True)
        if tmp.resolve() != wp.resolve():
            shutil.copy2(wp, tmp)
        print(f"\n--- {label} ---")
        print(f"    权重 {wp}  ({wp.stat().st_size / 1e6:.1f} MB)")
        model = YOLO(str(tmp))
        n_p = sum(p.numel() for p in model.model.parameters())
        n_l = len(list(model.model.modules()))
        try:
            gflops = model.model.get_flops(640)
        except Exception:
            gflops = None
        print(f"    参数量 {n_p:,}   层数 {n_l}   GFLOPs {gflops}")

        m = model.val(data=DATA, split="val", imgsz=640, batch=4, conf=0.001,
                      iou=0.6, plots=False, verbose=False)
        per_class = {}
        names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
        for i, ap in enumerate(m.box.ap50):
            per_class[names[i]] = round(float(ap), 4)
        rec = {
            "label": label,
            "weight": str(wp),
            "params": n_p,
            "layers": n_l,
            "gflops": gflops,
            # VisDrone images/val 实为 548 张 jpg（目录里另有 548 个 .npy 预处理残留，
            # `ls | wc -l` 会数成 1096 —— 别拿它当图片数）
            "images": 548,
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "map50": round(float(m.box.map50), 5),
            "map50_95": round(float(m.box.map), 5),
            "per_class_ap50": per_class,
        }
        out["val"][key] = rec
        print(f"    P={rec['precision']:.4f} R={rec['recall']:.4f} "
              f"mAP50={rec['map50']:.5f} mAP50-95={rec['map50_95']:.5f}")
        print(f"    逐类 AP50: {per_class}")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n已写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
