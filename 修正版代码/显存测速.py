# -*- coding: utf-8 -*-
"""显存悬崖测速：一次量出「哪个 imgsz/batch 组合能装下 + 每 epoch 要多久」。

为什么需要它（2026-09-12 实测踩出来的）：
    RTX 5060 Ti 只有 8GB。VisDrone 每张图目标数在几十到 900 之间波动，
    批量一大会让 PyTorch 分配器的 reserved 一路爬升；一旦越过物理显存，
    Windows WDDM 会**悄悄溢出到共享内存** —— 不报 OOM，只是速度塌方。
    实测同一份代码同一张卡：

        imgsz/batch   起始→末显存     稳定吞吐    图像/秒
        ------------------------------------------------------
        1024 / 2      4.83G → 8.33G   0.87 it/s   1.7     ← 崩
        1024 / 1      3.34G → 13.1G   1.7  it/s   1.7     ← 崩
        800  / 2      3.5G  → 5.29G   4.2  it/s   8.5
        640  / 4      3.36G → 5.38G   5.3  it/s   21.2    ← 最优
        640  / 4 (v1) 3.27G → 5.51G   5.3  it/s   21.2

    → 结论：吞吐不是被算力卡住，是被显存溢出卡住的。所以换配置前先跑本脚本，
      别拍脑袋。8GB 卡上把 imgsz 从 1024 降到 640，快 12 倍。

用法：
    python 显存测速.py                       # 默认测三组
    python 显存测速.py --variant v1          # 换变体（v1 更重，P2 尺度更吃显存）
    python 显存测速.py --imgsz 640 800 --batch 4 2

注意：
    1. `--fraction` 会把 C:\\datasets\\VisDrone\\labels\\train.cache 覆盖成子集缓存，
       正式训练时会自动重建全量缓存，不影响正确性，只是多花几十秒。
    2. 每组跑得短（约 160 个 iteration），显存碎片还没长满，所以结果是**乐观上界**；
       要让结论更硬，把 --iters 提到 500 再来一轮。
    3. 四个消融变体必须用同一组 imgsz/batch，所以按最重的那个变体（v1）选配置即可。
"""
import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMAGES = 6471


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="v4", help="用哪个变体测（建议用最重的 v1 定上限）")
    p.add_argument("--imgsz", type=int, nargs="+", default=[640, 800, 1024],
                   help="要测的输入尺寸列表")
    p.add_argument("--batch", type=int, nargs="+", default=[4, 2, 2],
                   help="与 --imgsz 一一对应的批大小列表")
    p.add_argument("--iters", type=int, default=160,
                   help="每组目标 iteration 数（越多越准，越慢）")
    p.add_argument("--project", default=r"C:\yolo_runs\_speedtest")
    return p.parse_args()


def parse_log(log):
    """从日志里抠出显存曲线与稳定吞吐。"""
    s = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    if "OutOfMemoryError" in s or "CUDA out of memory" in s:
        return {"oom": True}

    mems, last = [], None
    for ln in s.split("\n"):
        if "%" not in ln:
            continue
        head = re.search(r"(\d+)/(\d+)\s+([\d.]+)G\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\d+)\s+(\d+):\s*(\d+)%", ln)
        rates = re.findall(r"(\d+)/(\d+)\s+([\d.]+)(it/s|s/it)\s+(\d+):(\d+)", ln)
        if not head or not rates:
            continue
        it, tot, rate, unit, mm, ss = rates[-1]
        rate = float(rate)
        mems.append(float(head.group(3)))
        last = dict(it=int(it), tot=int(tot),
                    ips=rate if unit == "it/s" else (1.0 / rate if rate else 0.0),
                    sec=int(mm) * 60 + int(ss))
    if not last:
        return {"oom": False, "no_rate": True}
    eps = re.search(r"(\d+) epochs completed in ([\d.]+) hours", s)
    last.update(oom=False, mem_first=mems[0] if mems else None,
                mem_last=mems[-1] if mems else None, mem_max=max(mems) if mems else None,
                hours=float(eps.group(2)) if eps else None, n=len(mems))
    return last


def main():
    a = parse_args()
    if len(a.batch) != len(a.imgsz):
        print("--imgsz 与 --batch 个数必须一致")
        return 1
    os.makedirs(a.project, exist_ok=True)

    py = sys.executable
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # 去掉 WorkBuddy 注入的 safe-delete shim
    env["POLARS_SKIP_CPU_CHECK"] = "1"

    rows = []
    for i, (imgsz, batch) in enumerate(zip(a.imgsz, a.batch), 1):
        n_img = a.iters * batch
        frac = max(round(n_img / TRAIN_IMAGES, 4), 0.002)
        tag = "sp%d_b%d_%s" % (imgsz, batch, a.variant)
        log = os.path.join(a.project, tag + ".log")
        print("\n%s\n[%d/%d] %s  imgsz=%d batch=%d fraction=%.4f（约 %d iters）"
              % ("=" * 74, i, len(a.imgsz), tag, imgsz, batch, frac, a.iters))

        cmd = [py, "train_v8.py", "--variant", a.variant, "--epochs", "1",
               "--imgsz", str(imgsz), "--batch", str(batch),
               "--fraction", str(frac), "--project", a.project.replace("\\", "/"),
               "--name", tag]
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(cmd, cwd=HERE, env=env, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        r = parse_log(log)
        r.update(tag=tag, imgsz=imgsz, batch=batch, rc=rc)
        rows.append(r)
        print("    完成，退出码 %d，墙钟 %.1f 分钟" % (rc, (time.time() - t0) / 60))
        sys.stdout.flush()

    print("\n" + "=" * 90)
    print("测速结果（%s，VisDrone train=%d 张）" % (a.variant, TRAIN_IMAGES))
    print("=" * 90)
    print("%-14s %6s %6s %16s %10s %10s %14s" % (
        "配置", "imgsz", "batch", "起始→末显存", "训练it/s", "图像/秒", "推算每epoch"))
    print("-" * 90)
    best = None
    for r in rows:
        if r.get("oom"):
            print("%-14s %6d %6d   OOM —— 装不下，淘汰" % (r["tag"], r["imgsz"], r["batch"]))
            continue
        if r.get("no_rate"):
            print("%-14s %6d %6d   没取到速率，看日志 %s" % (r["tag"], r["imgsz"], r["batch"], r["tag"]))
            continue
        img_s = r["ips"] * r["batch"]
        iters_full = -(-TRAIN_IMAGES // r["batch"])
        per_ep = iters_full / r["ips"] / 60
        print("%-14s %6d %6d  %6.2fG→%6.2fG %10.2f %10.2f %11.1f 分" % (
            r["tag"], r["imgsz"], r["batch"], r["mem_first"], r["mem_last"],
            r["ips"], img_s, per_ep))
        # 选：能装下（末显存留 2G 余量）且每 epoch 最快的
        if r["mem_last"] and r["mem_last"] < 6.0 and (best is None or per_ep < best[1]):
            best = (r["tag"], per_ep, r["mem_last"])
    print("-" * 90)
    print("说明：'推算每epoch'未含验证（验证约再加 10~20 秒/epoch）。")
    print("      判据：末显存必须明显低于物理上限（8GB 卡 ≈ 7.66G 可用），")
    print("      留 2G 余量才安全 —— 逼近上限就会溢出到共享内存，吞吐直接塌。")
    if best:
        print("\n→ 推荐：%s（每 epoch 约 %.1f 分钟，末显存 %.2fG）" % best)
        print("  再确认：用最重的 v1 也测一遍，避免 v1 的 P2 尺度把显存顶穿。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
