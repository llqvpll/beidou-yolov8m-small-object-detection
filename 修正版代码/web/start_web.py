#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北斗 + 改进 YOLOv8m 小目标识别辅助系统 · Web 启动器
================================================
职责：
  1) 把自定义模块（EMA/DySample/DualConv/GSConv/VoVGSCSP）注册进 ultralytics；
  2) 检查 Web 运行所需依赖（flask/cv2/numpy）；
  3) 自动探测 CUDA：有卡用 GPU，无卡回落 CPU；
  4) 启动 Flask（若 ultralytics/权重缺失，自动进入「演示模式」，页面仍可用）。

用法：
    python start_web.py            # 默认 5000 端口
    PORT=8080 python start_web.py  # 自定义端口
"""
import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent          # web/
ROOT = HERE.parent                               # 修正版代码/

PY = sys.executable


def step(msg):
    print("\n>>> " + msg)


def run_install_modules():
    step("步骤1/4：注册自定义模块到 ultralytics")
    try:
        subprocess.run([PY, str(ROOT / "install_custom_modules.py")],
                       check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError:
        print("  [警告] 自定义模块注册失败，请确认已 `pip install ultralytics`。")
        print("          Web 将以「演示模式」启动（界面可用，但无真实推理）。")


def check_web_deps():
    step("步骤2/4：检查 Web 依赖")
    missing = []
    for mod in ("flask", "cv2", "numpy", "PIL"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"  [缺失] {missing} —— 请先安装：")
        print(f"          pip install {' '.join(missing)}")
        sys.exit(1)
    print("  [OK] flask / cv2 / numpy / PIL 就绪")


def report_env():
    step("步骤3/4：运行环境")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA: 可用 -> {torch.cuda.get_device_name(0)}（将使用 GPU 推理）")
        else:
            print("  CUDA: 不可用 -> 回落 CPU 推理（速度较慢，建议在有显卡的机器上跑）")
    except Exception as e:  # noqa: BLE001
        print(f"  torch 未安装或导入失败（{e}），将以「演示模式」启动。")


def start():
    step("步骤4/4：启动 Web 服务")
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    import app as webapp
    webapp.load_model()
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  访问地址: http://localhost:{port}")
    print("  按 Ctrl+C 停止。\n")
    webapp.app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    run_install_modules()
    check_web_deps()
    report_env()
    start()
