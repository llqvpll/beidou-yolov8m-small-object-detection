# -*- coding: utf-8 -*-
"""一键重出「论文配图」：把训练曲线与验证图重画成当前这一份权重对应的版本。

为什么需要这个脚本
------------------
服务端「分析 · 模型与训练」里那一块配图，默认取自当前权重所属 run 目录里的
ultralytics 原图。那些图是**训练结束时**画的，和页面 KPI 不是同一次测量：

    KPI / 收敛曲线  取 max(mAP50)          → 本项目第 74 轮，0.44573
    best.pt         取 max(fitness)        → 本项目第 66 轮，0.44383
                    fitness = 0.1*mAP50 + 0.9*mAP50-95

PR / F1 / 混淆矩阵都是拿 best.pt 复评生成的，所以图上写 0.442、KPI 写 44.6%，
同屏出现就像数据是拼凑的。本脚本把配图全部重做成与当前权重严格对应的一批，
放进 <代码目录>/figures/；后端**优先**展示这个目录（为空才回落 run 原图）。

两件事分别由谁做
----------------
  MATLAB          画 results.csv 里有的东西：mAP / P / R / 损失 / 学习率
  ultralytics val 画 results.csv 里**没有**的东西：PR / F1 / P-R 曲线、混淆矩阵
                  （需要逐类预测，无法从 csv 反推）

用法
----
    python 生成论文配图.py                     # 自动挑 results.csv 最新的 run
    python 生成论文配图.py --run visdrone_v4
    python 生成论文配图.py --no-matlab          # 只重跑验证
    python 生成论文配图.py --no-val             # 只出 MATLAB 曲线
    python 生成论文配图.py --project D:/runs --out D:/figs
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 重跑验证产出 -> 项目 figures 目录里的目标名（名字带 val_ 前缀，一眼区分来源）
VAL_FILES = {
    "BoxPR_curve.png": "val_pr_curve.png",
    "BoxF1_curve.png": "val_f1_curve.png",
    "BoxP_curve.png": "val_p_curve.png",
    "BoxR_curve.png": "val_r_curve.png",
    "confusion_matrix.png": "val_confusion_matrix.png",
    "confusion_matrix_normalized.png": "val_confusion_matrix_norm.png",
    # val_batch0_pred.jpg（验证集预测样例）已按用户要求从界面撤下。
    # 这里仍保留生成（重跑验证时顺手产出，留档用），但后端 metrics.py 的
    # HIDDEN_ARTIFACTS 会把它从配图列表中过滤掉 —— 生成与展示是两回事。
    "val_batch0_pred.jpg": "val_batch0_pred.jpg",
}

# MATLAB 常见安装位置（按版本从新到旧）
MATLAB_HINTS = [
    r"C:\Program Files\MATLAB",
    r"C:\Program Files (x86)\MATLAB",
]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def find_matlab() -> str | None:
    """找 matlab.exe：先 PATH，再扫常见安装目录里版本号最大的那个。"""
    exe = shutil.which("matlab")
    if exe:
        return exe
    cands: list[Path] = []
    for root in MATLAB_HINTS:
        p = Path(root)
        if not p.is_dir():
            continue
        cands += [c / "bin" / "matlab.exe" for c in p.iterdir() if c.is_dir()]
    cands = [c for c in cands if c.is_file()]
    if not cands:
        return None
    # R2025b > R2024a … 直接按名字倒序，够用
    return str(sorted(cands, key=lambda c: c.parent.parent.name, reverse=True)[0])


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def looks_like_smoke(name: str) -> bool:
    """冒烟 / 自检 / 临时目录 —— 能跑通流程，但指标没有意义。"""
    n = name.lower()
    return n.startswith(("smoke", "gpu_check", "debug", "_")) or "smoke" in n


def pick_run(project: Path, run: str | None, use_backend: bool = True) -> tuple[Path | None, str]:
    """挑一个 run 目录 → ``(目录, 依据说明)``。

    **优先用后端自己的权重候选表**（``Settings.weights_candidates``：
    RUN_NAMES 按优先级、FALLBACK_RUN_NAMES 单独兜底）。这样"脚本出的图"与
    "后端实际加载的权重"必定是同一个 run，不会出现图和页面各说各话。

    ⚠ 千万不要按 ``results.csv`` 的 mtime 挑"最新"：冒烟测试 ``smoke_visinit``
    的 csv 比 ``visdrone_v4`` 新，会把 3 个 epoch 的玩具模型图**覆盖**掉真图。
    实测踩过一次（本项目历史上"冒烟模型静默顶替真实权重"的同一类错误，
    只是换了个地方复现）。

    ``use_backend=False`` 时跳过第 1 步，只走扫描兜底 —— 测试用它去掉对
    本机训练目录的依赖。
    """
    if run:
        cand = Path(run)
        if not cand.is_dir():
            cand = project / run
        if (cand / "results.csv").is_file():
            return cand, "命令行指定"
        return None, f"指定的 run 不存在或没有 results.csv：{cand}"

    # 1) 后端口径：第一个真实存在的权重候选
    if use_backend:
        try:
            sys.path.insert(0, str(HERE))
            from server.config import Settings  # noqa: PLC0415

            for c in Settings().weights_candidates:
                p = Path(c)
                if p.is_file():
                    return p.parent.parent, "后端权重候选表（与页面加载的是同一个）"
        except Exception as e:  # noqa: BLE001
            say(f"[提示] 读后端配置失败（{e}），退回扫描目录。")

    # 2) 兜底：扫 project，先排除冒烟目录，再按 mtime 取最新
    cands = [c for c in (project.iterdir() if project.is_dir() else [])
             if c.is_dir() and (c / "results.csv").is_file()]
    real = [c for c in cands if not looks_like_smoke(c.name)]
    pool = real or cands
    if not pool:
        return None, f"{project} 下没有含 results.csv 的 run 目录"
    newest = max(pool, key=lambda c: (c / "results.csv").stat().st_mtime)
    note = "扫描目录取最新" if real else "扫描目录取最新（**只找到冒烟目录**）"
    return newest, note


def describe(run_dir: Path) -> str:
    """一行摘要：epoch 数 + best mAP50 + 参数量提示，让选错 run 一眼可见。"""
    rows = read_rows(run_dir / "results.csv")
    if not rows:
        return "（读不到 results.csv）"
    best = 0.0
    for r in rows:
        try:
            best = max(best, float(r.get("metrics/mAP50(B)") or 0))
        except ValueError:
            pass
    w = run_dir / "weights" / "best.pt"
    mb = f"{w.stat().st_size / 1024 / 1024:.1f} MB" if w.is_file() else "无权重"
    return f"{len(rows)} epoch · best mAP50 {best:.4f} · best.pt {mb}"


def run_matlab(mx: str, run_dir: Path, out_dir: Path) -> bool:
    mfile = HERE / "matlab"
    if not (mfile / "make_figures.m").is_file():
        say(f"[跳过] 找不到 MATLAB 脚本 {mfile / 'make_figures.m'}")
        return False
    csv_path = run_dir / "results.csv"
    say(f"[MATLAB] {mx}")
    say(f"         读 {csv_path}")
    cmd = [
        mx, "-batch",
        f"addpath('{mfile}'); make_figures('{csv_path}', '{out_dir}');",
    ]
    try:
        # MATLAB 启动约 40 s，别用默认短超时
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=900)
    except subprocess.TimeoutExpired:
        say("[失败] MATLAB 超时（>15 min）")
        return False
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith(("WROTE", "OK ", "epochs=", "mAP50_peak", "fitness_best", "out=")):
            say("         " + line.strip())
    if r.returncode != 0:
        say(f"[失败] MATLAB 退出码 {r.returncode}")
        say(out[-1500:])
        return False
    return True


def run_validation(weights: Path, data: Path, project: Path, name: str,
                   imgsz: int, batch: int) -> Path | None:
    """用当前权重重跑一次验证，产出 PR/F1/P/R 曲线与混淆矩阵。"""
    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        say(f"[失败] 没装 ultralytics：{e}")
        return None

    say(f"[验证] 权重 {weights}")
    say(f"       数据 {data}")
    model = YOLO(str(weights))
    res = model.val(
        data=str(data), split="val", imgsz=imgsz, batch=batch,
        conf=0.001, iou=0.6, plots=True, save_json=False, verbose=False,
        project=str(project), name=name, exist_ok=True,
    )
    d = res.results_dict or {}
    say("       读数 " + "  ".join(
        f"{k.split('/')[-1].rstrip('(B)')}={v:.3f}"
        for k, v in d.items() if isinstance(v, (int, float))))
    return Path(res.save_dir)


def report(run_dir: Path) -> None:
    """把两个「最优」口径打出来 —— 这正是页面图上数字对不上的原因。"""
    rows = read_rows(run_dir / "results.csv")
    if not rows:
        return

    def f(r, k):
        try:
            return float(r.get(k, "") or 0)
        except ValueError:
            return 0.0

    b_map = max(rows, key=lambda r: f(r, "metrics/mAP50(B)"))
    b_fit = max(rows, key=lambda r: 0.1 * f(r, "metrics/mAP50(B)") + 0.9 * f(r, "metrics/mAP50-95(B)"))
    say()
    say("两个「最优」口径（页面上的 44.6% 与 44.2% 就是这么来的）：")
    say(f"  mAP50 峰值   epoch {b_map.get('epoch')}   mAP50={f(b_map, 'metrics/mAP50(B)'):.5f}"
        "   ← KPI 与收敛曲线用它")
    say(f"  fitness 最优 epoch {b_fit.get('epoch')}   mAP50={f(b_fit, 'metrics/mAP50(B)'):.5f}"
        "   ← best.pt 存的是这一轮，PR 曲线/混淆矩阵用它")
    if b_map.get("epoch") != b_fit.get("epoch"):
        say("  两者不是同一轮 —— 图与 KPI 数字不同是正常的，不是数据错误。")


def main() -> int:
    ap = argparse.ArgumentParser(description="重出论文配图（MATLAB 曲线 + 重跑验证）")
    ap.add_argument("--project", default="C:/yolo_runs/train", help="训练根目录（含各 run）")
    ap.add_argument("--run", default=None, help="run 名或路径；默认取 results.csv 最新的")
    ap.add_argument("--weights", default=None, help="显式指定权重；默认 <run>/weights/best.pt")
    ap.add_argument("--out", default=None, help="输出目录；默认 <project>/figures")
    ap.add_argument("--data", default=None, help="数据集 yaml；默认 <project>/VisDrone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--no-matlab", action="store_true", help="跳过 MATLAB 曲线")
    ap.add_argument("--no-val", action="store_true", help="跳过重跑验证")
    ap.add_argument("--dry-run", action="store_true", help="只看会选哪个 run、会写哪些文件，不动盘")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="允许用冒烟/自检 run 出图（默认拒绝，避免覆盖真图）")
    args = ap.parse_args()

    proj_root = HERE
    out_dir = Path(args.out) if args.out else (proj_root / "figures")
    data = Path(args.data) if args.data else (proj_root / "VisDrone.yaml")
    runs_root = Path(args.project)

    say("=" * 62)
    say("  生成论文配图 —— 让页面上的图与当前权重严格对应")
    say("=" * 62)

    run_dir, why = pick_run(runs_root, args.run)
    if run_dir is None:
        say(f"[失败] {why}")
        say("       用 --project 指定训练根目录，或 --run 指定 run 名。")
        return 1
    say(f"run    {run_dir}")
    say(f"依据   {why}")
    say(f"概况   {describe(run_dir)}")

    # 冒烟 / 自检产物：指标没有意义，默认拒绝出图 —— 否则会把真图覆盖成玩具模型的图。
    if looks_like_smoke(run_dir.name) and not args.allow_smoke:
        say()
        say(f"[拒绝] 「{run_dir.name}」看起来是冒烟/自检产物，指标没有意义。")
        say("       拿它出图会把已有的真图覆盖掉。要强行出图加 --allow-smoke，")
        say("       或者用 --run 指定正确的 run，例如 --run visdrone_v4。")
        return 1

    weights = Path(args.weights) if args.weights else (run_dir / "weights" / "best.pt")
    if not weights.is_file():
        say(f"[警告] 权重不存在：{weights}（只出 MATLAB 曲线，跳过验证）")
        args.no_val = True
    say(f"权重   {weights}")
    say(f"输出   {out_dir}")
    if args.dry_run:
        say()
        say("[dry-run] 只报告，不写任何文件。")
        report(run_dir)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    say()

    ok_any = False

    if not args.no_matlab:
        mx = find_matlab()
        if mx is None:
            say("[跳过] 没找到 MATLAB（PATH 与 C:\\Program Files\\MATLAB 都没扫到）。")
            say("       可以先只重跑验证：python 生成论文配图.py --no-matlab")
            say("       也可手动执行：")
            say(f"         matlab -batch \"addpath('{HERE / 'matlab'}'); "
                f"make_figures('{run_dir / 'results.csv'}', '{out_dir}');\"")
        else:
            ok_any |= run_matlab(mx, run_dir, out_dir)
        say()

    if not args.no_val:
        tmp_root = runs_root / "_figs_val"
        save_dir = run_validation(weights, data, tmp_root, "current",
                                  args.imgsz, args.batch)
        if save_dir is None:
            say("[失败] 验证没跑起来。")
        else:
            copied, missing = [], []
            for src_name, dst_name in VAL_FILES.items():
                src = save_dir / src_name
                if src.is_file():
                    shutil.copy2(src, out_dir / dst_name)
                    copied.append(dst_name)
                else:
                    missing.append(src_name)
            say(f"       写入 {len(copied)} 个文件到 {out_dir}")
            if missing:
                say(f"       [注意] 验证目录里没有：{', '.join(missing)}")
            ok_any |= bool(copied)
        say()

    report(run_dir)

    say()
    say("=" * 62)
    files = sorted(p.name for p in out_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    say(f"  {out_dir} 现有 {len(files)} 张图：")
    for n in files:
        say("    - " + n)
    say()
    say("  后端会自动优先展示这个目录（为空才回落到 run 目录原图）。")
    say("  重启后端或点「分析 · 模型与训练 · 重读」即可看到新图。")
    say("=" * 62)
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
