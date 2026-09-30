r"""
显卡体检 —— 上机第一件事就跑这个

为什么需要它：
    RTX 50 系（Blackwell，GB20x）是 **sm_120** 计算能力。如果装到的 PyTorch 是
    用较老的 CUDA 编的（cu118/cu121 之类），它的 kernel 列表里没有 sm_120，
    运行时不会在安装阶段报错，而是在**第一次 GPU 运算时**才炸：

        CUDA error: no kernel image is available for execution on the device

    所以「torch.cuda.is_available() == True」并不能证明能训练，必须真的跑一次运算。

这个脚本做四件事：
    1. 打印 torch / CUDA / 显卡信息
    2. 判断当前 torch 是否编进了你这张卡的 compute capability
    3. 在 GPU 上真的做一次 forward + backward（含 conv / BN / grid_sample / 池化）
    4. 如果不行，直接给出对应的修复命令

用法：
    python 显卡检查.py
    python 显卡检查.py --full      # 额外用真实模型跑一次前向+反向（更彻底，需先打过补丁）
"""
import argparse
import sys


def line(c="="):
    print(c * 66)


def check_basic():
    ok = True
    line()
    print("1) 环境基本信息")
    line()
    try:
        import torch
    except ImportError:
        print("  [FAIL] 没有装 torch。先 pip install torch torchvision")
        return None, False

    print(f"  torch          : {torch.__version__}")
    cuda_build = getattr(torch.version, "cuda", None)
    print(f"  torch CUDA 编译版: {cuda_build}")
    avail = torch.cuda.is_available()
    print(f"  cuda.is_available: {avail}")

    if not avail:
        print()
        print("  [FAIL] 看不到 CUDA 设备。可能原因：")
        print("    - 装的是 CPU 版 torch（版本号带 '+cpu'）")
        print("    - NVIDIA 驱动没装 / 太旧")
        print("    - 这是笔记本，独显被电源策略或 MUX 关掉了")
        print()
        print("  修复：装带 CUDA 的 torch。先看 https://pytorch.org/get-started/locally/ 给的命令，")
        print("        或者用 PyTorch 官方索引（下面这条是 CUDA 12.8 版，Blackwell 可用）：")
        print("          pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        return torch, False

    n = torch.cuda.device_count()
    print(f"  卡数量          : {n}")
    caps = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024 ** 3
        caps.append(f"sm_{cap[0]}{cap[1]}")
        print(f"  卡[{i}]          : {name}")
        print(f"                    计算能力 {cap[0]}.{cap[1]}  (sm_{cap[0]}{cap[1]})"
              f"  显存 {total_gb:.1f} GB")

    # torch 编译进去的架构列表里有没有这张卡的 sm_xxx
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception as e:
        arch_list = []
        print(f"  (取不到 arch_list: {e})")
    print(f"  torch 支持架构   : {', '.join(arch_list) if arch_list else '(未知)'}")

    line()
    print("2) 架构匹配检查")
    line()
    missing = [c for c in caps if not any(a.startswith(c) for a in arch_list)] if arch_list else []
    if missing:
        ok = False
        print(f"  [FAIL] torch 里没有编进 {', '.join(missing)} 的 kernel！")
        print()
        print("  RTX 50 系（含 5060 Ti）是 Blackwell sm_120，需要 **CUDA 12.8 及以上**的 torch 轮子。")
        print("  修复（二选一，装完要重启 Python 进程）：")
        print("    pip uninstall -y torch torchvision")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("  具体索引名以 https://pytorch.org/get-started/locally/ 当前给的为准。")
    elif arch_list:
        print(f"  [PASS] {', '.join(caps)} 在 torch 的支持列表里")
    if cuda_build:
        major = int(str(cuda_build).split(".")[0])
        if major < 12:
            ok = False
            print(f"  [FAIL] torch 是用 CUDA {cuda_build} 编的，太老（Blackwell 要 12.8+）")
    return torch, ok


def check_kernels(torch):
    line()
    print("3) 真实 GPU 运算测试（forward + backward）")
    line()
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda"
    try:
        # 覆盖本项目实际会用到的主要算子：conv / BN / 激活 / 池化 / 双线性缩放 / grid_sample
        net = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 1), nn.BatchNorm2d(32), nn.SiLU(),
        ).to(dev).half()
        x = torch.randn(2, 8, 256, 256, device=dev, dtype=torch.float16)
        y = net(x)
        print(f"  conv/bn/silu      : {tuple(y.shape)}")

        pool = F.adaptive_avg_pool2d(y, 1)
        print(f"  adaptive_avg_pool : {tuple(pool.shape)}")

        up = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        print(f"  interpolate 2x    : {tuple(up.shape)}")

        grid = torch.rand(2, 32, 32, 2, device=dev, dtype=torch.float16) * 2 - 1
        gs = F.grid_sample(y, grid, mode="bilinear", align_corners=False)
        print(f"  grid_sample       : {tuple(gs.shape)}")

        loss = gs.float().sum() + up.float().sum() + pool.float().sum()
        loss.backward()
        grad_ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
        print(f"  backward          : {'梯度已回传' if grad_ok else '梯度缺失!'}")
        torch.cuda.synchronize()

        # 显存实测
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
        resv = torch.cuda.max_memory_reserved() / 1024 ** 2
        print(f"  峰值显存占用      : 已分配 {alloc:.1f} MB / 预留 {resv:.1f} MB")
        line()
        print("  [PASS] GPU kernel 全部正常执行 —— 这张卡可以拿来训练")
        return True
    except RuntimeError as e:
        msg = str(e)
        print(f"  [FAIL] {msg[:400]}")
        if "no kernel image" in msg:
            print()
            print("  → 这就是 sm_120 没被编译进去的典型报错。按第 2 步给的命令重装 torch。")
        elif "out of memory" in msg.lower():
            print()
            print("  → 显存不足。这不影响卡的可用性，只是别用这么大的 batch / imgsz。")
        return False
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return False


def check_real_model(torch):
    line()
    print("4) 真实模型前向 + 反向（--full）")
    line()
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
    try:
        from ultralytics import YOLO
        from custom_modules.register import setup as setup_custom
    except Exception as e:
        print(f"  [SKIP] 依赖没就绪（{type(e).__name__}: {e}）")
        print("         先跑 python install_custom_modules.py，并确认 ultralytics 已装。")
        return

    try:
        setup_custom()
        yml = os.path.join(here, "yolov8m_visdrone_slimneck_nop5.yaml")
        model = YOLO(yml)
        model.model.to("cuda").half().train()
        import torch.nn.functional as F
        x = torch.randn(2, 3, 1024, 1024, device="cuda", dtype=torch.float16)
        out = model.model(x)
        print(f"  前向输出尺度数    : {len(out[0]) if isinstance(out, (list, tuple)) else 'n/a'}")
        s = sum(o.float().sum() for o in (out[0] if isinstance(out[0], (list, tuple)) else [out]))
        s.backward()
        torch.cuda.synchronize()
        print(f"  imgsz=1024 batch=2 峰值显存: "
              f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print("  [PASS] 真实模型在 GPU 上跑通（含 DySample 的 grid_sample / EMA / DualConv）")
        print()
        print("  说明：这只是前向+反向，训练还会额外占优化器状态与数据增强的显存，")
        print("        实际训练请用 --batch -1 让 ultralytics 自动选 batch。")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {str(e)[:400]}")


def main():
    ap = argparse.ArgumentParser(description="显卡体检：确认这张卡能不能用来训练")
    ap.add_argument("--full", action="store_true",
                    help="额外用真实 v4 模型跑一次前向+反向（需先打过自定义模块补丁）")
    args = ap.parse_args()

    torch, ok = check_basic()
    if torch is None:
        sys.exit(2)
    if not torch.cuda.is_available():
        sys.exit(2)

    k_ok = check_kernels(torch)
    if args.full and k_ok:
        check_real_model(torch)

    line()
    if k_ok and ok:
        print("结论：显卡可用，去跑  python train_v8.py --variant v4 --quick  验证一遍链路。")
        sys.exit(0)
    print("结论：显卡**当前不可用**，按上面的提示修好再继续。")
    sys.exit(1)


if __name__ == "__main__":
    main()
