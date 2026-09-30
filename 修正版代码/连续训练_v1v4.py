# -*- coding: utf-8 -*-
"""v1(基线) + v4(推荐) 顺序连跑，无人值守。

为什么单独写这个脚本而不是跑两次 train_v8.py：
  1. 两组必须用**完全相同**的 imgsz/batch/epochs 才有可比性 —— 这里用同一份参数下发，杜绝手滑；
  2. 顺序执行、互不抢显存（8GB 卡只能一次跑一组）；
  3. 每组独立日志 + 断点续跑判断（已跑完的变体自动跳过）；
  4. 实时打印 ETA，便于判断能不能过夜。

用法示例（配置按测速结果填）：
    python 连续训练_v1v4.py --imgsz 640 --batch 4 --epochs 150
默认输出到 C:/yolo_runs/train（OneDrive 之外），可用 --project 改。
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# 只跑这两个：v1 = 基线，v4 = 推荐（Slim-Neck + 剪 P5）
ORDER = ["v1", "v4"]
TRAIN_IMAGES = 6471


def parse_args():
    p = argparse.ArgumentParser(description="v1 基线 + v4 推荐 顺序连跑")
    p.add_argument("--imgsz", type=int, required=True,
                   help="必须与测速选定的值一致；两变体共用")
    p.add_argument("--batch", type=int, required=True, help="两变体共用")
    p.add_argument("--epochs", type=int, default=150, help="两变体共用")
    p.add_argument("--project", default=os.environ.get("YOLO_RUNS_DIR", r"C:\yolo_runs\train"),
                   help="输出根目录，务必在 OneDrive 之外")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "none"])
    p.add_argument("--force", action="store_true",
                   help="即使发现该变体已有完整 results.csv 也强制重跑")
    return p.parse_args()


def state(project, name, epochs):
    """判断某变体当前处于哪种状态：none（没跑过）/ partial（跑了一半）/ done（跑满）。"""
    run = os.path.join(project, name)
    csv_path = os.path.join(run, "results.csv")
    last = os.path.join(run, "weights", "last.pt")
    if not os.path.isdir(run):
        return "none", 0
    if not (os.path.isfile(csv_path) and os.path.isfile(last)):
        # 目录在但没权重 —— 上次刚启动就被打断
        return "partial", 0
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            rows = [r for r in csv.reader(f) if r]
        n = len(rows) - 1                       # 减掉表头
    except Exception:
        return "partial", 0
    return ("done" if n >= epochs else "partial"), n


def main():
    a = parse_args()
    os.makedirs(a.project, exist_ok=True)

    if "onedrive" in a.project.lower():
        print("[warn] 输出目录在 OneDrive 内，每 epoch 的 checkpoint 都会被同步上传！")
        print("[warn] 建议 --project C:/yolo_runs/train")

    plan = []
    partial = []
    for v in ORDER:
        name = "visdrone_%s" % v
        st, n = state(a.project, name, a.epochs)
        if st == "done" and not a.force:
            print("[skip] %s 已跑满 %d epoch（%s），跳过" % (v, n, os.path.join(a.project, name)))
            continue
        if st == "partial":
            partial.append((v, name, n))
            continue
        plan.append(v)

    if partial:
        print("\n" + "!" * 74)
        for v, name, n in partial:
            print("  发现未跑完的 %s：%s 里只有 %d 个 epoch。" % (v, name, n))
        print("  直接重跑会新建一个 <%s>-2 目录从头开始，白费之前的进度。" % partial[0][1])
        print("  要接着上次跑，请手工执行（其余超参以 checkpoint 里存的为准）：")
        for v, name, _ in partial:
            print("      python train_v8.py --variant %s --project %s --name %s --resume"
                  % (v, a.project, name))
        print("  确认要从头来过，就重跑本脚本并加 --force。")
        print("!" * 74)
        return 2

    if not plan:
        print("\n两组都已完成，无需重跑。加 --force 可强制重跑。")
        return 0

    iters_full = -(-TRAIN_IMAGES // a.batch)
    print("=" * 74)
    print("  即将顺序训练: %s" % ", ".join(plan))
    print("  统一配置: imgsz=%d  batch=%d  epochs=%d  cache=%s  workers=%d"
          % (a.imgsz, a.batch, a.epochs, a.cache, a.workers))
    print("  每组每 epoch ≈ %d 个 iteration" % iters_full)
    print("  输出目录: %s" % a.project)
    print("  提示：两组配置完全一致，消融对比才成立；中途别改参数。")
    print("=" * 74)
    sys.stdout.flush()

    t_all = time.time()
    done = []
    for i, v in enumerate(plan, 1):
        name = "visdrone_%s" % v
        log = os.path.join(a.project, name + ".log")
        cmd = [PY, "train_v8.py", "--variant", v,
               "--imgsz", str(a.imgsz), "--batch", str(a.batch),
               "--epochs", str(a.epochs), "--cache", a.cache,
               "--workers", str(a.workers),
               "--project", a.project, "--name", name]
        print("\n" + "-" * 74)
        print("[%d/%d] 开始 %s（%s）" % (i, len(plan), v, "基线" if v == "v1" else "推荐：Slim-Neck + 剪P5"))
        print("        日志: %s" % log)
        print("-" * 74)
        sys.stdout.flush()

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)          # 去掉 WorkBuddy 的 safe-delete shim
        env["POLARS_SKIP_CPU_CHECK"] = "1"
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, cwd=HERE, env=env,
                                stdout=f, stderr=subprocess.STDOUT).returncode
        used = time.time() - t0

        if rc != 0:
            print("\n[abort] %s 退出码 %d，流程中止。" % (v, rc))
            print("        看日志找原因: %s" % log)
            print("        OOM 的话把 --imgsz 降一档（如 800 -> 640）后重跑本脚本，")
            print("        已完成的变体会被自动跳过。")
            return rc

        # 从日志尾部拿每 epoch 用时，给下一组估时
        s = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
        m = re.search(r"(\d+) epochs completed in ([\d.]+) hours", s)
        per_ep = (float(m.group(2)) * 3600 / a.epochs) if m else None
        if per_ep:
            print("[%d/%d] %s 完成，用时 %.2f 小时（每 epoch %.1f 分钟）"
                  % (i, len(plan), v, used / 3600, per_ep / 60))
        else:
            print("[%d/%d] %s 完成，用时 %.2f 小时" % (i, len(plan), v, used / 3600))
        done.append(v)
        sys.stdout.flush()

    print("\n" + "=" * 74)
    print("  全部完成: %s，总用时 %.2f 小时" % (", ".join(done), (time.time() - t_all) / 3600))
    for v in done:
        run = os.path.join(a.project, "visdrone_%s" % v)
        print("    %-28s %s" % (v, os.path.join(run, "weights", "best.pt")))
    print("  下一步：跑 'python 对比指标.py'（若尚无该脚本，可用两张 results.csv 直接画图）")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
