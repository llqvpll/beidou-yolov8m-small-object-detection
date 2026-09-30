"""
模块自检脚本：验证 5 个自定义模块 + 损失 + 模型结构能否正确构建与前后向。

用法（在有 torch + ultralytics 的环境里）：
    python test_modules.py

退出码：全部通过 = 0；有任何 [FAIL] = 1（方便 .bat / 环境验证.py 判断）。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 同 train_v8.py：跳过 polars 的 CPU 自检（本机 cpuid 探测失败会误报 sse3）
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

from custom_modules.dualconv import DualConv          # noqa: E402
from custom_modules.dysample import DySample          # noqa: E402
from custom_modules.ema_attention import EMAAttention  # noqa: E402

FAILED = []


def check_shape(tag, out, expect):
    ok = tuple(out.shape) == tuple(expect)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {tuple(out.shape)} (期望 {expect})")
    if not ok:
        FAILED.append(tag)
    return ok


def main():
    x = torch.randn(2, 96, 32, 32)

    # 1) 单模块形状 / 数值健全性
    m = EMAAttention(96, 96, groups=32).eval()
    check_shape("EMAAttention", m(x), (2, 96, 32, 32))

    d = DySample(96, 96, scale=2).eval()
    check_shape("DySample(2x)", d(x), (2, 96, 64, 64))

    c = DualConv(96, 96, k=3, s=1, g=4).eval()
    check_shape("DualConv(s=1)", c(x), (2, 96, 32, 32))
    check_shape("DualConv(s=2)", DualConv(96, 96, 3, 2).eval()(x), (2, 96, 16, 16))

    # 1b) Slim-Neck 模块（依赖 ultralytics 的 Conv，故单独 try）
    try:
        from custom_modules.gsconv import GSConv, VoVGSCSP
        check_shape("GSConv(s=1)", GSConv(96, 96, 3, 1).eval()(x), (2, 96, 32, 32))
        check_shape("GSConv(s=2)", GSConv(96, 96, 3, 2).eval()(x), (2, 96, 16, 16))
        check_shape("VoVGSCSP(n=3)", VoVGSCSP(96, 96, 3).eval()(x), (2, 96, 32, 32))
    except ImportError as e:
        print(f"[SKIP] Slim-Neck 模块需要已安装 ultralytics：{e}")

    # 2) NWD 损失
    from custom_loss import NWDLoss
    p = torch.tensor([[10.0, 10.0, 4.0, 4.0], [10.0, 10.0, 4.0, 4.0]])
    g = torch.tensor([[10.5, 10.5, 4.0, 4.0], [10.0, 10.0, 4.0, 4.0]])
    v = NWDLoss(C=12.0)(p, g)
    print("NWD:", v.tolist(), "(第一个应 > 0，第二个应 == 0)")
    if not (v[1].abs() < 1e-3 and v[0] > 0):
        print("[FAIL] NWD 数值不符合预期")
        FAILED.append("NWDLoss")

    # 3) 四个消融变体能否全部构建 + 前向（需先装好自定义模块）
    #    期望的检测尺度数是关键判据：v1/v2 是 4（含 P2），v3/v4 是 3（剪了 P5）
    VARIANTS = [
        ("yolov8m_visdrone_improved.yaml", 4),
        ("yolov8m_visdrone_slimneck.yaml", 4),
        ("yolov8m_visdrone_p2_nop5.yaml", 3),
        ("yolov8m_visdrone_slimneck_nop5.yaml", 3),
    ]
    try:
        from custom_modules.installer import is_installed
        if not is_installed():
            print("[WARN] 自定义模块尚未安装，请先运行 `python install_custom_modules.py` 并重启进程")
            FAILED.append("custom_modules 未安装")
        else:
            from ultralytics import YOLO
            for yml, expect_nl in VARIANTS:
                try:
                    model = YOLO(yml)
                    n_param = sum(p.numel() for p in model.model.parameters())
                    n_scale = model.model.model[-1].nl
                    out = model.model(torch.randn(1, 3, 640, 640))
                    ok = n_scale == expect_nl
                    print(f"[{'PASS' if ok else 'FAIL'}] {yml}: 检测尺度={n_scale}"
                          f"(期望 {expect_nl}), 参数={n_param:,}, 前向 OK")
                    if not ok:
                        FAILED.append(yml)
                except Exception as e:
                    print(f"[FAIL] {yml}: {type(e).__name__}: {e}")
                    FAILED.append(yml)
    except Exception as e:  # pragma: no cover
        print("[WARN] 模型构建未通过:", e)
        FAILED.append("模型构建")

    print()
    if FAILED:
        print(f"自检结束 —— {len(FAILED)} 项失败: {', '.join(FAILED)}")
        sys.exit(1)
    print("自检结束 —— 全部通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
