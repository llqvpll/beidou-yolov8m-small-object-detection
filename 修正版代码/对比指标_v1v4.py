# -*- coding: utf-8 -*-
"""v1(基线) vs v4(推荐) 消融对比：读两份 results.csv，打印指标表 + 出对比图。

产出：
  <project>/对比_v1_v4.png   —— 4 联图（mAP50 / mAP50-95 / 参数量-精度 / 收敛曲线）
  <project>/对比_v1_v4.txt   —— 可直接粘进论文的指标表

用法：
    python 对比指标_v1v4.py
    python 对比指标_v1v4.py --project C:/yolo_runs/train --epochs 150
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体：Windows 自带微软雅黑；找不到就退回默认（会显示成方块，但不影响出图流程）
for f in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

VARIANTS = [("v1", "v1 基线\n(DualConv, 4尺度)", "#5E82A8"),
            ("v4", "v4 推荐\n(Slim-Neck + 剪P5, 3尺度)", "#C25E58")]

# 真机实测参数量（test_modules.py 输出），用于表格里的模型复杂度一行
PARAMS = {"v1": 24_892_928, "v4": 17_552_134}
GFLOPS = {"v1": None, "v4": 79.3}


def read_results(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    return rows


def _g(row, name):
    """按列名取值（容忍列名两侧空格）。"""
    for k in row:
        if k.strip() == name:
            return float(row[k] or 0)
    raise KeyError(name)


def best_row(rows, key):
    """取 key 最大的那一行。"""
    return max(rows, key=lambda r: _g(r, key))


def col(rows, name):
    return [_g(r, name) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("YOLO_RUNS_DIR", r"C:\yolo_runs\train"))
    p.add_argument("--epochs", type=int, default=0, help="用于横轴标注；0=按实际行数")
    a = p.parse_args()

    data = {}
    for v, _, _ in VARIANTS:
        csv_path = os.path.join(a.project, "visdrone_%s" % v, "results.csv")
        rows = read_results(csv_path)
        if not rows:
            print("[缺] %s —— 还没跑完或路径不对" % csv_path)
            continue
        data[v] = rows

    if not data:
        print("两组都没有结果，先跑 连续训练_v1v4.py")
        return 1

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 84)
    out("v1(基线) vs v4(推荐) 消融对比")
    out("=" * 84)
    out("%-26s %10s %10s %10s %10s %12s" % ("变体", "mAP50(best)", "mAP50-95", "P", "R", "参数量"))
    out("-" * 84)
    summary = {}
    for v, label, _ in VARIANTS:
        if v not in data:
            continue
        rows = data[v]
        m50 = col(rows, "metrics/mAP50(B)")
        m5095 = col(rows, "metrics/mAP50-95(B)")
        bp = max(col(rows, "metrics/precision(B)"))
        br = max(col(rows, "metrics/recall(B)"))
        i_best = m50.index(max(m50))
        summary[v] = dict(mAP50=m50[i_best], best_epoch=i_best + 1,
                          final_epoch=len(rows),
                          mAP5095=m5095[i_best], P=bp, R=br,
                          params=PARAMS.get(v))
        out("%-26s %10.4f %10.4f %10.4f %10.4f %12s"
            % (label.replace("\n", " "), summary[v]["mAP50"], summary[v]["mAP5095"], bp, br,
               "{:,}".format(PARAMS[v]) if PARAMS.get(v) else "-"))
    out("-" * 84)
    for v in summary:
        out("  %s：best mAP50 出现在第 %d epoch，共 %d epoch"
            % (v, summary[v]["best_epoch"], summary[v]["final_epoch"]))

    if "v1" in summary and "v4" in summary:
        d50 = summary["v4"]["mAP50"] - summary["v1"]["mAP50"]
        dp = (summary["v4"]["params"] - summary["v1"]["params"]) / summary["v1"]["params"] * 100
        out("v4 相对 v1：")
        out("  mAP50        %+.4f  (%+.2f pp)" % (d50, d50 * 100))
        out("  mAP50-95     %+.4f" % (summary["v4"]["mAP5095"] - summary["v1"]["mAP5095"]))
        out("  参数量       %+.1f%%   (%s → %s)"
            % (dp, "{:,}".format(summary["v1"]["params"]), "{:,}".format(summary["v4"]["params"])))
        out("  训练轮数     %d vs %d" % (summary["v1"]["final_epoch"], summary["v4"]["final_epoch"]))
        if summary["v1"]["final_epoch"] != summary["v4"]["final_epoch"]:
            out("  ⚠ 两组轮数不一致，对比不严谨，建议同轮数重跑")
    out("=" * 84)

    txt = os.path.join(a.project, "对比_v1_v4.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---------- 出图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        ax.set_facecolor("#F7FAFD")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=.25, linestyle="--", linewidth=.7)

    # 1) mAP50 收敛曲线
    for v, label, color in VARIANTS:
        if v not in data:
            continue
        y = col(data[v], "metrics/mAP50(B)")
        axes[0].plot(range(1, len(y) + 1), y, color=color, linewidth=2,
                     label=label.replace("\n", " "))
    axes[0].set_title("mAP50 收敛曲线", fontsize=13, color="#25313F")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mAP50")
    axes[0].legend(fontsize=9, frameon=False)

    # 2) mAP50-95 收敛曲线
    for v, label, color in VARIANTS:
        if v not in data:
            continue
        y = col(data[v], "metrics/mAP50-95(B)")
        axes[1].plot(range(1, len(y) + 1), y, color=color, linewidth=2,
                     label=label.replace("\n", " "))
    axes[1].set_title("mAP50-95 收敛曲线", fontsize=13, color="#25313F")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mAP50-95")
    axes[1].legend(fontsize=9, frameon=False)

    # 3) 参数量 vs 精度
    if "v1" in summary and "v4" in summary:
        xs = [summary["v1"]["params"] / 1e6, summary["v4"]["params"] / 1e6]
        ys = [summary["v1"]["mAP50"], summary["v4"]["mAP50"]]
        for i, (v, label, color) in enumerate(VARIANTS):
            axes[2].scatter(xs[i], ys[i], s=190, color=color, zorder=3)
            axes[2].annotate("%s\n%.2fM" % (v, xs[i]), (xs[i], ys[i]),
                             textcoords="offset points", xytext=(0, 14),
                             ha="center", fontsize=9, color="#25313F")
        axes[2].plot(xs, ys, color="#8B97A8", linestyle="--", linewidth=1.4, zorder=2)
        axes[2].set_title("参数量 — 精度权衡", fontsize=13, color="#25313F")
        axes[2].set_xlabel("参数量 (M)"); axes[2].set_ylabel("mAP50 (best)")
        axes[2].margins(x=.25, y=.3)

    fig.suptitle("VisDrone2019-DET 消融：v1 基线 vs v4 Slim-Neck+剪P5",
                 fontsize=15, color="#0F3B63", y=1.02)
    fig.tight_layout()
    png = os.path.join(a.project, "对比_v1_v4.png")
    fig.savefig(png, bbox_inches="tight", facecolor="#FFFFFF")
    print("\n已输出：")
    print("  指标表 %s" % txt)
    print("  对比图 %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
