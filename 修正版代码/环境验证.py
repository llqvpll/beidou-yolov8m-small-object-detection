r"""
环境验证 —— 复制到新电脑后，第一件事就跑这个

它把「能不能开始训练」需要的所有前置条件串起来检查一遍，最后给一个明确结论：

    [1] Python 解释器版本
    [2] 依赖包（装没装 / 能不能 import —— 专抓"包被删一半"的情况）
    [3] 显卡与 CUDA（含 RTX 50 系的 sm_120 检查 + 真实 GPU 运算测试）
    [4] 自定义模块补丁（EMA / DySample / DualConv / GSConv / VoVGSCSP 有没有写进 ultralytics 源码）
    [5] 预训练权重 yolov8m.pt（存在性 + 参数量鉴定）
    [6] 数据集配置（VisDrone.yaml 的 path 对不对、images/labels 是否配对、标签格式抽检）
    [7] 代码文件完整性（4 份 yaml + 各个脚本）

可选加餐：
    --full     额外跑 test_modules.py（构建 4 个变体 + 前向，约 1~3 分钟）
    --bench    在目标 imgsz 上实测训练速度，估算跑完要多久（需要显卡或耐心）

用法：
    python 环境验证.py                      # 日常：全套基础检查
    python 环境验证.py --full                # 再加上模型构建前向
    python 环境验证.py --bench --imgsz 1024  # 顺便测速，估算训练时长
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
import time

# 让输出在 GBK 控制台上遇到不能编码的字符也不会崩（顶多显示成 ?）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# 必须在 import ultralytics 之前设置：本机 cpuid 探测失败会让 polars 误报 sse3
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

W = 70
RESULTS = []          # [(status, step, detail)]
CTX = {}              # 各步产出的上下文，供后续步骤复用

PASS, FAIL, WARN, SKIP, INFO = "PASS", "FAIL", "WARN", "SKIP", "INFO"


def head(title, idx=None, total=None):
    print()
    tag = f"[{idx}/{total}] " if idx else ""
    print("-" * W)
    print(f"--- {tag}{title} " + "-" * max(0, W - len(title) - len(tag) - 6))
    print("-" * W)


def rec(status, step, detail=""):
    RESULTS.append((status, step, detail))
    print(f"  [{status}] {step}" + (f"  -- {detail}" if detail else ""))
    return status


# ---------------------------------------------------------------- 1. Python
def check_python():
    head("Python 解释器", 1, 7)
    v = sys.version_info
    print(f"  executable : {sys.executable}")
    print(f"  version    : {platform.python_version()}")
    print(f"  platform   : {platform.system()} {platform.release()} "
          f"({platform.machine() or platform.processor() or '未知架构'})")
    ok = (3, 8) <= (v.major, v.minor) <= (3, 12)
    if ok:
        rec(PASS, f"Python {platform.python_version()} 在 ultralytics 支持范围内 (3.8~3.12)")
    elif (v.major, v.minor) >= (3, 13):
        rec(WARN, f"Python {platform.python_version()} 偏新",
            "ultralytics / torch 在 3.13 上偶有轮子缺失，建议 3.10 或 3.11")
    else:
        rec(FAIL, f"Python {platform.python_version()} 太旧", "ultralytics 需要 3.8+")
    CTX["python_ok"] = ok


# ------------------------------------------------------------- 2. 依赖包
# (pip 名, import 名, 级别)
#   core     —— 缺了肯定跑不起来
#   declared —— ultralytics 在 METADATA 里声明的依赖，但代码里是惰性导入
#               （不做就 FAIL，只在少数功能路径上才需要）
#   optional —— 非本项目所需
# 注：这份清单按 ultralytics 8.4.147 的 Requires-Dist 逐条核对过。
#     8.4.x 起已经**不再**依赖 scipy / pandas / seaborn，别按老版本经验判断。
PACKAGES = [
    ("torch", "torch", "core"),
    ("torchvision", "torchvision", "core"),
    ("ultralytics", "ultralytics", "core"),
    ("numpy", "numpy", "core"),
    ("opencv-python", "cv2", "core"),
    ("Pillow", "PIL", "core"),
    ("PyYAML", "yaml", "core"),
    ("matplotlib", "matplotlib", "core"),
    ("polars", "polars", "core"),
    ("psutil", "psutil", "core"),
    ("requests", "requests", "core"),
    ("filelock", "filelock", "core"),
    ("nvidia-ml-py", "pynvml", "core"),
    # 上游声明了，但都是惰性导入，缺了不影响训练主链路
    ("cloudpickle", "cloudpickle", "declared"),      # 只在 DDP 分布式训练里用
    ("ultralytics-thop", "thop", "declared"),        # 只用于统计 GFLOPs
    ("tqdm", "tqdm", "declared"),
    ("pandas", "pandas", "optional"),
    ("scipy", "scipy", "optional"),
    ("seaborn", "seaborn", "optional"),
    ("py-cpuinfo", "cpuinfo", "optional"),
]


def check_packages():
    head("依赖包", 2, 7)
    bad_core, bad_decl, missing_optional = [], [], []
    for dist, mod, tier in PACKAGES:
        try:
            installed = md.version(dist)
        except Exception:
            installed = None
        try:
            importlib.import_module(mod)
            import_ok, err = True, ""
        except Exception as e:
            import_ok, err = False, f"{type(e).__name__}: {e}"

        if import_ok:
            print(f"  [ok]     {dist:<18} {installed or '(版本未知)'}")
        elif installed:
            # 最阴的情况：dist-info 说装了，但模块文件被删了
            print(f"  [BROKEN] {dist:<18} 版本记录 {installed}，但 import 失败")
            print(f"           {err[:150]}")
            (bad_core if tier == "core" else bad_decl).append(dist)
        elif tier == "core":
            print(f"  [MISS]   {dist:<18} 未安装（必需）")
            bad_core.append(dist)
        elif tier == "declared":
            print(f"  [miss]   {dist:<18} 未安装（上游声明，本项目主链路用不到）")
            bad_decl.append(dist)
        else:
            print(f"  [skip]   {dist:<18} 未安装（可选）")
            missing_optional.append(dist)

    if not bad_core:
        rec(PASS, f"核心依赖齐全（{sum(1 for p in PACKAGES if p[2] == 'core')} 个）")
    else:
        rec(FAIL, f"{len(bad_core)} 个核心依赖有问题: {', '.join(bad_core)}",
            "若是'装了却 import 不到'，运行  python 环境自检.py --fix")
    if bad_decl:
        rec(WARN, f"上游声明但缺失: {', '.join(bad_decl)}",
            "训练主链路不需要；要补就 pip install " + " ".join(bad_decl))
    CTX["bad_pkgs"] = bad_core


# -------------------------------------------------------- 3. 显卡 / CUDA
def check_torch_gpu() -> bool:
    head("显卡与 CUDA", 3, 7)
    try:
        import torch
    except Exception as e:
        rec(FAIL, "无法 import torch", str(e)[:120])
        CTX["gpu_ok"] = False
        CTX["cuda"] = False
        return False

    print(f"  torch          : {torch.__version__}")
    print(f"  torch CUDA 编译版: {getattr(torch.version, 'cuda', None)}")
    cuda = torch.cuda.is_available()
    CTX["cuda"] = cuda
    print(f"  cuda.is_available: {cuda}")

    if not cuda:
        print("  [INFO] 没检测到 CUDA 设备：")
        print("         - 装的是 CPU 版 torch（版本号带 '+cpu'）-> 只能跑冒烟测试")
        print("         - 或驱动没装好 / 卡被禁用")
        print("         冒烟测试仍可继续： python train_v8.py --variant v4 --quick")
        rec(WARN, "没有可用的 CUDA 设备", "本地只能跑冒烟测试；正式训练见 README 的 Kaggle 一节")
        CTX["gpu_ok"] = False
        return False

    n = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    vram_gb = props.total_memory / 1024 ** 3
    print(f"  显卡           : {torch.cuda.get_device_name(0)}")
    print(f"  计算能力       : {cap[0]}.{cap[1]}  ({sm})")
    print(f"  显存           : {vram_gb:.1f} GB")
    print(f"  卡数量         : {n}")
    CTX["vram_gb"] = vram_gb
    CTX["sm"] = sm

    # --- 架构匹配：RTX 50 系最容易在这里翻车 ---
    arch_list = []
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        pass
    print(f"  torch 支持架构 : {', '.join(arch_list) if arch_list else '(取不到)'}")
    arch_ok = True
    if arch_list and not any(a.startswith(sm) for a in arch_list):
        arch_ok = False
        rec(FAIL, f"torch 没编进 {sm} 的 kernel",
            "RTX 50 系需要 CUDA 12.8+ 的 torch，见下方修复命令")

    # --- 真的在 GPU 上跑一遍（这才是唯一可信的验证）---
    print("  在 GPU 上实跑 forward + backward ...")
    gpu_run_ok = False
    try:
        import torch.nn as nn
        import torch.nn.functional as F

        net = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 32, 1), nn.BatchNorm2d(32), nn.SiLU(),
        ).cuda().half()
        x = torch.randn(2, 8, 256, 256, device="cuda", dtype=torch.float16)
        y = net(x)
        F.adaptive_avg_pool2d(y, 1)
        F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        # DySample 依赖 grid_sample，必须测到
        grid = torch.rand(2, 32, 32, 2, device="cuda", dtype=torch.float16) * 2 - 1
        gs = F.grid_sample(y, grid, mode="bilinear", align_corners=False)
        (gs.float().sum() + y.float().sum()).backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        gpu_run_ok = True
        rec(PASS, f"GPU 运算正常（conv/BN/SiLU/pool/interpolate/grid_sample）",
            f"峰值显存 {peak:.0f} MB")
    except RuntimeError as e:
        msg = str(e)
        if "no kernel image" in msg:
            rec(FAIL, "GPU 运算失败：no kernel image is available", "就是 sm_120 没被编译进去")
        elif "out of memory" in msg.lower():
            rec(WARN, "GPU 运算时显存不足", "卡本身可用，只是测试张量偏大")
            gpu_run_ok = True
        else:
            rec(FAIL, f"GPU 运算失败: {msg[:120]}")
    except Exception as e:
        rec(FAIL, f"GPU 运算失败: {type(e).__name__}: {e}"[:150])

    if not gpu_run_ok or not arch_ok:
        print()
        print("  修复（装完记得重启 Python 进程）：")
        print("    pip uninstall -y torch torchvision")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("  索引名以 https://pytorch.org/get-started/locally/ 当前给出的为准，")
        print("  关键是 CUDA 12.8 或更高；更详细的排查见  python 显卡检查.py")

    # --- 按显存给配置建议 ---
    if vram_gb < 6:
        tip = "6GB 以下：建议 --imgsz 640"
    elif vram_gb < 10:
        tip = "8GB 级：--batch -1（AutoBatch）配 --imgsz 1024，若 batch<2 再降 800/640"
    elif vram_gb < 20:
        tip = "16GB 级：--batch -1 配 --imgsz 1024 没问题"
    else:
        tip = "24GB 以上：可以放宽 batch，或提高 imgsz"
    print(f"  显存建议       : {tip}")

    CTX["gpu_ok"] = bool(gpu_run_ok and arch_ok)
    return CTX["gpu_ok"]


# --------------------------------------------------- 4. 自定义模块补丁
CLASS_NAMES = ("EMAAttention", "DySample", "DualConv", "GSConv", "VoVGSCSP")
REPEAT_NAMES = ("VoVGSCSP",)


def _frozenset_body(src: str, anchor: str):
    """把 `anchor = frozenset( [注释] { ... }` 的花括号内容抠出来。"""
    m = re.search(rf"{re.escape(anchor)}\s*=\s*frozenset\(\s*(?:\#[^\n]*\s*)*\{{", src)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
    return None


def check_patch():
    head("自定义模块补丁", 4, 7)
    try:
        import ultralytics  # noqa: F401
        import ultralytics.nn.tasks as tk
    except Exception as e:
        rec(FAIL, "ultralytics 不可用，无法检查补丁", str(e)[:120])
        CTX["patch_ok"] = False
        return

    ult_dir = os.path.dirname(os.path.abspath(tk.__file__))
    tasks_file = os.path.join(ult_dir, "tasks.py")
    init_file = os.path.join(ult_dir, "modules", "__init__.py")
    print(f"  ultralytics    : {os.path.dirname(os.path.dirname(ult_dir))}")

    try:
        src = open(tasks_file, encoding="utf-8").read()
        init_src = open(init_file, encoding="utf-8").read()
    except Exception as e:
        rec(FAIL, "读不到 ultralytics 源码", str(e)[:120])
        CTX["patch_ok"] = False
        return

    base = _frozenset_body(src, "base_modules")
    rep = _frozenset_body(src, "repeat_modules")

    if base is None:
        rec(FAIL, "在 tasks.py 里找不到 base_modules 的 frozenset",
            "ultralytics 版本可能差异过大，需手动打补丁")
        CTX["patch_ok"] = False
        return

    miss_base = [n for n in CLASS_NAMES if not re.search(rf"\b{n}\b", base)]
    miss_init = [n for n in CLASS_NAMES if n not in init_src]
    miss_rep = []
    if rep is not None:
        miss_rep = [n for n in REPEAT_NAMES if not re.search(rf"\b{n}\b", rep)]

    for n in CLASS_NAMES:
        where = []
        where.append("base_modules" if n not in miss_base else "base_modules(缺)")
        if n in REPEAT_NAMES:
            where.append("repeat_modules" if n not in miss_rep else "repeat_modules(缺)")
        print(f"  {n:<14} {' + '.join(where)}")

    ok = not (miss_base or miss_init or miss_rep)
    if ok:
        rec(PASS, "5 个自定义模块已正确注册进 ultralytics 源码")
    else:
        detail = []
        if miss_base:
            detail.append(f"base_modules 缺 {miss_base}")
        if miss_init:
            detail.append(f"modules/__init__.py 缺 {miss_init}")
        if miss_rep:
            detail.append(f"repeat_modules 缺 {miss_rep}")
        rec(FAIL, "自定义模块补丁不完整", "; ".join(detail))
        print()
        print("  修复：python install_custom_modules.py")
        print("        然后【重启 Python 进程】再跑本脚本（补丁在内存里不会自动生效）")
    CTX["patch_ok"] = ok


# ------------------------------------------------------- 5. 预训练权重
def check_weights():
    head("预训练权重", 5, 7)
    pt = os.path.join(_HERE, "yolov8m.pt")
    if not os.path.isfile(pt):
        rec(FAIL, "本目录下没有 yolov8m.pt",
            "脚本按名称对齐加载预训练权重，缺失会退化为从头训练")
        print()
        print("  下载：curl -L --ssl-no-revoke -o yolov8m.pt \\")
        print("        https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt")
        print("        或者直接双击 一键运行.bat，第 3 步会自动下")
        CTX["pt_ok"] = False
        return

    size_mb = os.path.getsize(pt) / 1024 ** 2
    print(f"  文件           : {pt}")
    print(f"  大小           : {size_mb:.1f} MB")
    if size_mb < 30:
        rec(FAIL, f"yolov8m.pt 只有 {size_mb:.1f} MB，明显不完整",
            "删掉重下（正常约 52 MB）")
        CTX["pt_ok"] = False
        return

    # 参数量鉴定：普通 yolov8m = 25,902,640 参数。这是判断权重真伪最可靠的办法。
    n_param = None
    try:
        import torch
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        m = ck.get("model") if isinstance(ck, dict) else ck
        if m is not None and hasattr(m, "parameters"):
            n_param = sum(p.numel() for p in m.parameters())
    except Exception as e:
        print(f"  (读取 checkpoint 失败，跳过参数量鉴定: {type(e).__name__})")

    if n_param is not None:
        print(f"  参数量         : {n_param:,}")
        if n_param == 25_902_640:
            rec(PASS, "是官方 yolov8m 预训练权重（25,902,640 参数）")
        else:
            rec(WARN, f"参数量 {n_param:,}，不是官方 yolov8m 的 25,902,640",
                "若是你自己训练产出的权重，忽略即可")
    else:
        rec(PASS, f"yolov8m.pt 存在（{size_mb:.1f} MB）")
    CTX["pt_ok"] = True


# --------------------------------------------------------- 6. 数据集
def _img2label(p: str) -> str:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in p:
        return p.replace(sa, sb)
    return p.replace(f"{os.sep}images", f"{os.sep}labels")


def check_dataset():
    head("数据集配置", 6, 7)
    yml = os.path.join(_HERE, "VisDrone.yaml")
    if not os.path.isfile(yml):
        rec(FAIL, "找不到 VisDrone.yaml", "train_v8.py 的默认数据源就是这个文件")
        CTX["data_ok"] = False
        return

    try:
        import yaml
        cfg = yaml.safe_load(open(yml, encoding="utf-8"))
    except Exception as e:
        rec(FAIL, f"VisDrone.yaml 解析失败: {e}")
        CTX["data_ok"] = False
        return

    root = cfg.get("path")
    names = cfg.get("names") or {}
    print(f"  yaml           : {yml}")
    print(f"  path           : {root}")
    print(f"  names          : {len(names)} 类")

    if not root:
        rec(FAIL, "VisDrone.yaml 里没有 path:", "填上数据集根目录（建议绝对路径）")
        CTX["data_ok"] = False
        return
    if not os.path.isabs(str(root)):
        root = os.path.normpath(os.path.join(_HERE, str(root)))
        print(f"  (相对路径解析为) : {root}")

    if not os.path.isdir(root):
        # 注意：这里判 WARN 而不是 FAIL。
        # "数据集还没拷过来" 和 "环境配置错了" 是两回事——环境本身没问题，
        # 只是还不能开始正式训练。真正属于配置错误的（路径指向了一个存在但结构不对的目录）
        # 会在下面判 FAIL。
        print()
        print("  两种可能：")
        print("    A. 数据集还没拷到这台机器 / 还没下载（最常见）")
        print("    B. VisDrone.yaml 的 path 还是旧的（两台机器路径不一样是常态）")
        print("       改 VisDrone.yaml:  path: <这台机器上的数据集根目录>")
        print("   另：VisDrone 官方标注不是 YOLO 格式，若还没转过，先跑：")
        print("       python visdrone2yolo.py --src <VisDrone2019-DET> --dst <输出目录> --mode copy")
        rec(WARN, "数据集目录还不存在（环境本身没问题，只是还不能正式训练）", str(root))
        CTX["data_ok"] = False
        return

    if len(names) != 10:
        rec(WARN, f"names 有 {len(names)} 类，VisDrone2019-DET 应该是 10 类")

    all_ok = True
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            continue
        img_dir = os.path.join(root, str(rel))
        lbl_dir = _img2label(img_dir)
        if not os.path.isdir(img_dir):
            rec(FAIL, f"{split}: 图片目录不存在", img_dir)
            all_ok = False
            continue
        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        n_lbl = 0
        if os.path.isdir(lbl_dir):
            n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])
        if n_lbl == 0:
            rec(FAIL, f"{split}: 找不到标签文件", lbl_dir)
            print("        VisDrone 官方标注不是 YOLO 格式，必须先用 visdrone2yolo.py 转换")
            all_ok = False
            continue
        flag = "" if n_lbl == len(imgs) else f"  <-- 与图片数 {len(imgs)} 不一致"
        rec(PASS if n_lbl == len(imgs) else WARN,
            f"{split}: {len(imgs)} 张图 / {n_lbl} 个标签", flag.strip())

        # 标签格式抽检
        checked = bad_line = out_of_range = 0
        for f in sorted(os.listdir(lbl_dir))[:20]:
            if not f.endswith(".txt"):
                continue
            try:
                for line in open(os.path.join(lbl_dir, f), encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        bad_line += 1
                        continue
                    cls = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                    checked += 1
                    if not (0 <= cls <= 9) or any(not (0.0 <= v <= 1.0) for v in vals):
                        out_of_range += 1
            except Exception:
                bad_line += 1
        if checked == 0:
            rec(FAIL, f"{split}: 抽检的标签文件里没有任何有效标注行", "格式可能不对")
            all_ok = False
        elif bad_line or out_of_range:
            rec(WARN, f"{split}: 抽检 {checked} 行，异常 {bad_line + out_of_range} 行",
                "字段数不对 = 没转换；坐标>1 = 没归一化")
            all_ok = False
        else:
            print(f"         标签格式抽检 {checked} 行 -> 全部为 class cx cy w h 且已归一化")

    CTX["data_ok"] = all_ok
    if all_ok:
        rec(PASS, "数据集配置正确，可以训练")


# --------------------------------------------------------- 7. 代码文件
def check_files():
    head("代码文件完整性", 7, 7)
    need = [
        "train_v8.py", "custom_loss.py", "install_custom_modules.py", "test_modules.py",
        "visdrone2yolo.py", "make_smoke_data.py", "VisDrone.yaml",
        "yolov8m_visdrone_improved.yaml", "yolov8m_visdrone_slimneck.yaml",
        "yolov8m_visdrone_p2_nop5.yaml", "yolov8m_visdrone_slimneck_nop5.yaml",
        "custom_modules/__init__.py", "custom_modules/register.py",
        "custom_modules/installer.py", "custom_modules/ema_attention.py",
        "custom_modules/dysample.py", "custom_modules/dualconv.py",
        "custom_modules/gsconv.py",
    ]
    missing = [f for f in need if not os.path.exists(os.path.join(_HERE, f.replace("/", os.sep)))]
    print(f"  检查 {len(need)} 个文件")
    if missing:
        rec(FAIL, f"缺少 {len(missing)} 个文件", ", ".join(missing))
    else:
        rec(PASS, "代码文件齐全")

    # yaml 里的 nc 是否和 VisDrone.yaml 的 names 数一致（盲审"论文=代码"要看这个）
    try:
        import yaml
        n_names = len(yaml.safe_load(open(os.path.join(_HERE, "VisDrone.yaml"),
                                          encoding="utf-8")).get("names") or {})
        bad = []
        for f in ("yolov8m_visdrone_improved.yaml", "yolov8m_visdrone_slimneck.yaml",
                  "yolov8m_visdrone_p2_nop5.yaml", "yolov8m_visdrone_slimneck_nop5.yaml"):
            d = yaml.safe_load(open(os.path.join(_HERE, f), encoding="utf-8"))
            nc = d.get("nc")
            if nc != n_names:
                bad.append(f"{f}: nc={nc}")
        if bad:
            rec(WARN, "yaml 的 nc 与 VisDrone.yaml 的类别数不一致", "; ".join(bad))
        else:
            rec(PASS, f"4 份 yaml 的 nc 都等于 {n_names}")
    except Exception as e:
        rec(WARN, f"nc 一致性检查跳过: {type(e).__name__}")


# ------------------------------------------------------------ 可选：full
def run_full():
    head("深度检查：构建 4 个变体并前向 (--full)")
    script = os.path.join(_HERE, "test_modules.py")
    if not os.path.isfile(script):
        rec(SKIP, "找不到 test_modules.py")
        return None
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = ""          # 摆脱外部注入的 sitecustomize
    p = subprocess.run([sys.executable, script], cwd=_HERE, env=env,
                       capture_output=True, text=True, errors="replace")
    print(p.stdout.rstrip() if p.stdout else "(无输出)")
    if p.stderr.strip():
        print("  --- stderr ---")
        print(p.stderr.rstrip()[:2000])
    dt = time.time() - t0
    if p.returncode == 0 and "[FAIL]" not in (p.stdout or ""):
        rec(PASS, f"模型自检通过，用时 {dt:.0f}s")
        return True
    rec(FAIL, f"模型自检未通过（退出码 {p.returncode}）", f"用时 {dt:.0f}s")
    return False


# ----------------------------------------------------------- 可选：bench
def run_bench(imgsz, variant, train_images):
    head(f"速度基准 (--bench)  imgsz={imgsz}")
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        rec(SKIP, "缺少 torch / ultralytics，无法测速", str(e)[:100])
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        rec(WARN, "没有显卡，CPU 测速没有参考意义", "跳过")
        return

    var_map = {
        "v1": "yolov8m_visdrone_improved.yaml",
        "v2": "yolov8m_visdrone_slimneck.yaml",
        "v3": "yolov8m_visdrone_p2_nop5.yaml",
        "v4": "yolov8m_visdrone_slimneck_nop5.yaml",
    }
    yml = os.path.join(_HERE, var_map.get(variant, var_map["v4"]))
    try:
        model = YOLO(yml).model.to(dev).train()
    except Exception as e:
        rec(SKIP, f"建图失败（补丁没打？）: {e}"[:150])
        return

    print(f"  模型           : {os.path.basename(yml)}")
    print(f"  设备           : {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")

    # 找一个装得下的 batch：从大到小试，OOM 就退一档
    torch.cuda.reset_peak_memory_stats()
    chosen, sec_per_iter = None, None
    for bs in (8, 6, 4, 2, 1):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            opt = torch.optim.SGD(model.parameters(), lr=0.001)
            x = torch.randn(bs, 3, imgsz, imgsz, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                out = model(x)
            loss = sum(o.float().mean() for o in (out[0] if isinstance(out[0], (list, tuple)) else [out]))
            loss.backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # 计 3 次
            n_rep = 3
            t0 = time.time()
            for _ in range(n_rep):
                with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                    out = model(x)
                loss = sum(o.float().mean() for o in (out[0] if isinstance(out[0], (list, tuple)) else [out]))
                loss.backward()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            sec_per_iter = (time.time() - t0) / n_rep
            chosen = bs
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch={bs} -> 显存不足，退一档")
                try:
                    del x
                except Exception:
                    pass
                continue
            rec(SKIP, f"测速失败: {str(e)[:150]}")
            return

    if chosen is None:
        rec(FAIL, "连 batch=1 都跑不动", "降低 --imgsz 再试")
        return

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    iters = -(-train_images // chosen)          # 向上取整
    sec_epoch = sec_per_iter * iters
    print(f"  可用 batch     : {chosen}   (峰值显存 {peak:.2f} GB)")
    print(f"  每次迭代       : {sec_per_iter * 1000:.0f} ms")
    print(f"  每 epoch       : {sec_epoch / 60:.1f} 分钟  (按 {train_images} 张图算)")
    print()
    print("  预估总时长：")
    for ep in (100, 200):
        h = sec_epoch * ep / 3600
        print(f"    {ep:>3d} epoch          : {h:.1f} 小时")
    h4 = sec_epoch * 100 * 4 / 3600
    print(f"    4 个变体 x 100 epoch : {h4:.1f} 小时  ({h4 / 24:.1f} 天)")
    rec(PASS, f"测速完成：batch={chosen}, {sec_epoch / 60:.1f} min/epoch @ imgsz={imgsz}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="环境验证：确认这台机器能不能开始训练 YOLOv8m 改进模型")
    ap.add_argument("--full", action="store_true",
                    help="额外运行 test_modules.py（构建 4 个变体 + 前向，约 1~3 分钟）")
    ap.add_argument("--bench", action="store_true",
                    help="额外做训练速度基准，估算跑完要多久")
    ap.add_argument("--imgsz", type=int, default=1024, help="测速用的输入尺寸（默认 1024）")
    ap.add_argument("--variant", default="v4", choices=["v1", "v2", "v3", "v4"],
                    help="测速用哪个变体（默认 v4）")
    ap.add_argument("--train-images", type=int, default=6471,
                    help="训练集图片数（VisDrone2019-DET 训练集默认 6471）")
    args = ap.parse_args()

    t_start = time.time()
    print("=" * W)
    print(" YOLOv8m 改进模型 —— 环境验证")
    print(f" 目录   : {_HERE}")
    print(f" 时间   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 解释器 : {sys.executable}")
    print("=" * W)

    check_python()
    check_packages()
    check_torch_gpu()
    check_patch()
    check_weights()
    check_dataset()
    check_files()

    if args.full:
        run_full()
    if args.bench:
        run_bench(args.imgsz, args.variant, args.train_images)

    # ---------------- 汇总 ----------------
    print()
    print("=" * W)
    print(" 汇总")
    print("=" * W)
    n_fail = n_warn = n_pass = 0
    for status, step, detail in RESULTS:
        if status == FAIL:
            n_fail += 1
        elif status == WARN:
            n_warn += 1
        elif status == PASS:
            n_pass += 1
        if status in (FAIL, WARN):
            print(f"  [{status}] {step}" + (f"  -- {detail}" if detail else ""))
    print(f"\n  PASS {n_pass}   WARN {n_warn}   FAIL {n_fail}   用时 {time.time() - t_start:.0f}s")

    print()
    print("-" * W)
    data_ready = CTX.get("data_ok", False)
    if n_fail == 0:
        cuda = CTX.get("cuda", False)
        if cuda and CTX.get("gpu_ok"):
            print(" 结论：环境就绪，显卡可用。")
        elif cuda:
            print(" 结论：环境就绪，但显卡有问题（见上面 [3]）。")
        else:
            print(" 结论：环境就绪（无可用显卡，只能本地跑冒烟测试）。")
        if not data_ready:
            print()
            print(" 还差一步才能正式训练：数据集没就位。")
            print("   1) 准备好 VisDrone2019-DET，用 visdrone2yolo.py 转成 YOLO 格式")
            print("   2) 把 VisDrone.yaml 的 path: 改成数据集根目录")
            print("   3) 再跑一次本脚本，[6] 变成 PASS 就可以开训了")
        print()
        print(" 现在就能做（不需要数据集）：冒烟测试，验证代码链路")
        print("   python train_v8.py --variant v4 --quick")
        if data_ready:
            print()
            print(" 开始正式训练：")
            print("   python train_v8.py --variant v4          # 建议先看 README 的 imgsz 建议")
        sys.exit(0)
    print(f" 结论：有 {n_fail} 项必须解决，先按上面的提示修，再重跑本脚本。")
    print("       常见对应关系：")
    print("         [2] 包有问题            -> python 环境自检.py --fix")
    print("         [3] no kernel image     -> 重装 CUDA 12.8+ 版 torch")
    print("         [4] 补丁不完整          -> python install_custom_modules.py 然后重启进程")
    print("         [5] 缺 yolov8m.pt       -> 双击 一键运行.bat 自动下载")
    print("         [6] 有图无标签/格式错   -> visdrone2yolo.py 转换，或检查 VisDrone.yaml")
    sys.exit(1)


if __name__ == "__main__":
    main()
