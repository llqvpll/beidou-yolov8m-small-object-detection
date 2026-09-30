"""
YOLOv8m 改进模型训练脚本（修正版）

相对原附录 train_v8.py 的改动：
  [P0-1] 修正模型装配：原 `YOLO(pre_model_name, override_cfg)` 的第二个位置参数是
         task 而非配置，导致自定义结构与损失从未生效。现改为
         `YOLO(yaml).load(weights)`，并显式挂接模块与 NWD 损失。
  [P0-3] 自定义模块的注册：未注册模块在 parse_model 里会走 `else: c2 = ch[f]` 分支，
         args 原样传入、不做通道缩放，因此必须把类加入 ultralytics 的 base_modules。
         `setup_custom()` 会检查并自动安装（写入源码，需重启一次进程）。
  [P0-2] 结构配置合并为一份完整可加载的 v8 yaml（见 yolov8m_visdrone_improved.yaml）。
  [P2-1] imgsz 由 list 改为 int（list 形式在 train 阶段非标准，易被静默忽略）。
  [P2-2] cache: True -> 'disk'（VisDrone 高分辨率全量进内存极易 OOM）。
  [P2-3] mixup: 0.5 -> 0.0（MixUp 混叠小目标，对 VisDrone 通常有害）。
  [P2-4] 删除非法参数 augment=True。
  [P2-5] copy_paste 仅在数据集含分割标注时有效，此处保留并注明。
  [P2-6] exist_ok: True -> False，避免覆盖历史实验产物。
  [P2-7] patience 注释与取值统一（20 -> 50）。
  [P2-8] warmup_epochs 10 -> 5；AdamW 初始学习率 1e-3 -> 5e-4。
  [P2-9] 增加 deterministic=True，配合 seed 提升可复现性。

另：`--quick` 提供 CPU 冒烟测试模式（3% 数据 / 3 epoch / 640 / cpu），
    几分钟即可验证「装补丁 → 建图 → 前向 → 反向 → 验证」全链路是否打通，
    不追求精度；真实训练请在有 GPU 的机器上跑完整配置。

[P0-4] device 自适应：原脚本 device 写死 '0'，在没有 NVIDIA 显卡的机器上直接抛
       `ValueError: Invalid CUDA 'device=0' requested`。现改为先探测 torch.cuda，
       无卡时自动回落 cpu；若此时跑的又是正式训练（非 --quick），则明确提示
       「CPU 跑 VisDrone 不现实」并退出（如需硬跑可加 --force-cpu）。
"""
import os
import sys

# 让脚本无论从哪个目录运行都能找到本目录下的自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# polars(ultralytics 的依赖) 会做 CPU 指令集自检。本机取不到 cpuid，
# 映射表为空 → 误报 "unknown feature flag: 'sse3'"（实际 CPU 支持 sse3/ssse3/avx2）。
# polars 官方提供了跳过开关，必须在 ultralytics 之前设置（ultralytics 是延迟导入 polars 的）。
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

# 防显存碎片：VisDrone 每批实例数波动大（上百~近千），分配尺寸随之变化，
# PyTorch 默认分配器的 reserved 会碎片化爬升（实测 batch3@1024 从 7.5G 爬到
# 11.8G），在 Windows WDDM 下溢出到共享内存后训练速度塌方（0.7 s/it → 8+ s/it）。
# expandable_segments 让预留段随需伸缩，必须在 torch 首次 import 前设置。
#
# ⚠ 实测（本机 RTX 5060 Ti 8GB + torch 2.9.0+cu128）：
# ① Windows 上 expandable_segments **不被支持**，只打一句 UserWarning 然后忽略；
# ② 千万别再加 `max_split_size_mb:128` —— 2026-09-12 实测，加了以后同样的
#    v4 @ imgsz800/batch2，reserved 从 8.33G 直接飙到 **15.6G**（物理显存的两倍），
#    吞吐从 ~1.7 img/s 掉到 ~0.8 img/s。原因是大块激活（>128MB）无法再切分，
#    每次都得重新 cudaMalloc，碎片反而更严重。
# → 结论：这条环境变量在 Windows 上别折腾，压显存只能靠 imgsz / batch。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ultralytics import YOLO  # noqa: E402

from custom_modules.register import setup as setup_custom  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_YAML = "yolov8m_visdrone_improved.yaml"   # 默认结构（可用 --model 切换）
PRETRAINED = "yolov8m.pt"                       # 预训练权重（按名称对齐加载）

# 默认数据配置：用本目录自带的 VisDrone.yaml（nc=10），绝对路径，换工作目录也不会丢。
# 注意：该文件里的 `path:` 需要指向你实际的 VisDrone2019-DET 数据集根目录。
DATA_YAML = os.path.join(_HERE, "VisDrone.yaml")

# 冒烟测试用的合成小数据集（由 make_smoke_data.py 生成）。
# --quick 时若没显式指定 --data，就自动用它，这样无需下载 1.5GB 的 VisDrone 也能验证链路。
SMOKE_YAML = os.path.join(_HERE, "冒烟测试数据集", "smoke.yaml")

# 输出根目录的默认值：**必须放在 OneDrive 之外**。
# 本模型每 epoch 都会写 last.pt / best.pt（约 272MB），外加每 save_period 个 epoch
# 一个 epochN.pt。放在 OneDrive 同步目录里会被反复上传，既拖慢磁盘/进程，又白烧带宽。
# 可用环境变量 YOLO_RUNS_DIR 覆盖（换机器时改这一处即可）。
DEFAULT_PROJECT = os.environ.get("YOLO_RUNS_DIR", r"C:\yolo_runs\train")

# 2x2 消融矩阵（见 ABLATION.md）
VARIANTS = {
    "v1": "yolov8m_visdrone_improved.yaml",          # 基线：DualConv，4 尺度
    "v2": "yolov8m_visdrone_slimneck.yaml",          # +Slim-Neck，4 尺度
    "v3": "yolov8m_visdrone_p2_nop5.yaml",           # 剪 P5，3 尺度
    "v4": "yolov8m_visdrone_slimneck_nop5.yaml",     # 推荐：Slim-Neck + 剪 P5
}


def build_model(model_yaml: str):
    """先按自定义结构建图，再加载预训练权重。"""
    model = YOLO(model_yaml)
    model.load(PRETRAINED)      # 结构不同的层会被自动跳过并给出提示
    return model


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="YOLOv8m VisDrone 改进模型训练")
    p.add_argument("--variant", default="v1", choices=sorted(VARIANTS),
                   help="消融矩阵中的变体：v1 基线 / v2 +SlimNeck / v3 剪P5 / v4 两者")
    p.add_argument("--model", default=None, help="直接指定 yaml 路径（优先于 --variant）")
    p.add_argument("--data", default=DATA_YAML)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=-1,
                   help="批大小。-1 = AutoBatch（ultralytics 自动按显存挑最大的，目标占用 60%%），"
                        "8GB 显存（如 RTX 5060 Ti）建议就用 -1，别手写 8/16，否则几乎必然 OOM。")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="输入尺寸。默认 1024（与原代码一致，利于消融可比）。"
                        "8GB 显存若 AutoBatch 给出的 batch < 2，建议降到 800 或 640 再跑，"
                        "但四个变体必须用同一个值。")
    p.add_argument("--device", default="0",
                   help="训练设备。默认 '0'（第 0 号 CUDA 卡）；本机无 NVIDIA GPU 时会"
                        "自动回落为 'cpu'，不会直接报错。")
    p.add_argument("--force-cpu", action="store_true",
                   help="本机无 GPU 时，仍坚持在 CPU 上做正式训练（默认会直接退出并给出指引）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default=None)
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help="输出根目录。必须是绝对路径——ultralytics 的 get_save_dir() 对"
                        "相对 project 会强行前置 'runs/<task>/'，导致路径多套一层。"
                        "默认放在 OneDrive 之外（每 epoch 都有 272MB checkpoint 落盘，"
                        "放同步盘里会被反复上传）。切勿用反斜杠结尾/未加引号的 Windows 路径，"
                        "cmd/bash 会吃掉分隔符，把 C:\\\\datasets\\\\runs\\\\train 变成 "
                        "'C:datasetsrunstrain' 这种怪目录。")
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                   help="数据缓存：disk / ram / none（none = 不缓存，最省内存）")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="使用数据集的比例（调试用，如 0.05 表示只用 5%%）")
    p.add_argument("--resume", action="store_true",
                   help="从上次中断处继续训练。要求 <project>/<name>/weights/last.pt 存在，"
                        "且 --project/--name 与上次一致（续训时其余超参一律以 checkpoint 里存的为准）")
    p.add_argument("--quick", action="store_true",
                   help="CPU 冒烟测试：3%% 数据 / 3 epoch / imgsz=640 / device=cpu，几分钟跑完，只验证链路")
    return p.parse_args()


def apply_quick(args):
    """--quick：把一切压到最小，只为了验证『代码能跑通』，不追求精度。

    若用户没有显式指定 --data，则自动改用 make_smoke_data.py 生成的合成小数据集，
    这样不必先下载 1.5GB 的 VisDrone 就能把整条链路跑通。
    """
    using_smoke = False
    if args.data == DATA_YAML and os.path.exists(SMOKE_YAML):
        args.data = SMOKE_YAML
        using_smoke = True

    args.epochs = 3
    args.imgsz = 640
    args.batch = 4
    args.device = "cpu"
    args.workers = 0        # 冒烟测试用 0：避免 Windows 下多进程 dataloader 偶发卡死
    args.cache = "none"

    # 有显卡就让冒烟测试也跑在显卡上 —— 否则"换到新电脑"这一步根本没验证到 CUDA/sm_120 是否正常。
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        pass
    if cuda_ok:
        args.device = "0"
        args.batch = 8
        print("[train] 检测到可用显卡，冒烟测试将在 GPU 上跑（这样才能一并验证 CUDA kernel 正常）")

    # 合成数据集本身就只有几十张，不用再抽样；真实数据集才需要采样
    args.fraction = 1.0 if using_smoke else (0.03 if args.fraction == 1.0 else args.fraction)
    if not args.name:
        args.name = f"smoke_{args.variant}"

    print("[train] QUICK 模式：%d epoch / imgsz %d / device %s / fraction %.3f"
          % (args.epochs, args.imgsz, args.device, args.fraction))
    print("[train] 数据配置: %s%s" % (args.data, "   ← 合成冒烟数据集" if using_smoke else ""))
    print("[train] 目的只是验证链路是否跑通；真实精度请到 GPU 上用 VisDrone 跑完整配置。")
    if not using_smoke:
        print("[train] 提示：本目录若已有 `冒烟测试数据集/smoke.yaml`，quick 模式会自动使用它。")
    return args


def probe_batch(model, imgsz):
    """batch=-1 时自测安全批大小（替代 ultralytics AutoBatch）。

    为什么不用 AutoBatch：Windows 的 WDDM 驱动允许显存不够时溢出到系统内存
    （"共享 GPU 内存"），torch.cuda 报告的可用量会虚高，AutoBatch 据此挑出
    的批大小实际训练时一边溢出一边变慢，甚至 profiling 阶段就 RuntimeError。
    这里改成实测前向+反向的峰值显存，乘 1.3 安全系数（EMA/优化器/验证开销）
    后仍不超过"当前空闲显存 - 0.8GB（上下文/桌面波动余量）"才采纳，从大到小
    试到第一个能过的为止。
    """
    import gc

    import torch

    candidates = [16, 12, 8, 6, 4, 3, 2, 1]
    net = model.model.cuda().train()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    budget = free_gb - 0.8
    print("[batch] 自测安全批大小 @ imgsz=%d（空闲 %.2f GB，预算 %.2f GB）" % (imgsz, free_gb, budget))
    for b in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        try:
            x = torch.randn(b, 3, imgsz, imgsz, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
            feats = out if isinstance(out, (list, tuple)) else ([out] if torch.is_tensor(out) else list(out.values()))
            loss = sum(f.float().sum() for f in feats if torch.is_tensor(f) and f.is_floating_point())
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            del x, out, loss
            print("[batch]   batch %2d → 峰值 %.2f GB ×1.3 = %.2f GB" % (b, peak, peak * 1.3))
            if peak * 1.3 <= budget:
                print("[batch] 采纳 batch = %d" % b)
                model.model.cpu()
                torch.cuda.empty_cache()
                return b
        except torch.cuda.OutOfMemoryError:
            print("[batch]   batch %2d → OOM，跳过" % b)
        except Exception as e:
            print("[batch]   batch %2d → 探测异常（%s: %s），跳过" % (b, type(e).__name__, e))
    model.model.cpu()
    torch.cuda.empty_cache()
    print("[batch] 没有候选能通过预算，回退 batch = 1（建议降低 --imgsz）")
    return 1


def resolve_device(args):
    """按实际硬件决定训练设备，避免 `ValueError: Invalid CUDA 'device=0' requested`。

    原脚本把 device 写死成 '0'，在没装 N 卡的机器上会直接崩。这里统一处理：
      - 有 CUDA           → 用用户指定的卡（默认 0）
      - 无 CUDA           → 自动回落 cpu（绝不会再因为 device 报错）
      - 无 CUDA + 正式训练 → 额外提示：CPU 跑 VisDrone 不现实，建议去 GPU 平台
    """
    import torch  # ultralytics 已依赖 torch，这里一定可用

    cuda_ok = torch.cuda.is_available()
    want_cpu = str(args.device).strip().lower().startswith("cpu")

    if cuda_ok and not want_cpu:
        # 把卡的名字和显存打出来，方便一眼确认用的是哪张卡、够不够
        try:
            idx = 0
            dev_s = str(args.device)
            if dev_s.replace(",", "").isdigit() and dev_s:
                idx = int(dev_s.split(",")[0])
            props = torch.cuda.get_device_properties(idx)
            cap = torch.cuda.get_device_capability(idx)
            print("[device] %s | sm_%d%d | 显存 %.1f GB | torch %s (CUDA %s)"
                  % (props.name, cap[0], cap[1], props.total_memory / 1024 ** 3,
                     torch.__version__, getattr(torch.version, "cuda", "?")))
            arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
            sm = "sm_%d%d" % (cap[0], cap[1])
            if arch_list and not any(a.startswith(sm) for a in arch_list):
                print("[device] ⚠ torch 未编入 %s 的 kernel（RTX 50 系需 CUDA 12.8+ 的 torch）。" % sm)
                print("[device]   先跑 `python 显卡检查.py` 确认，或重装 CUDA 版 torch 后再训练。")
        except Exception as e:
            print("[device] (查显卡信息失败: %s)" % e)
        if args.batch == -1:
            print("[device] batch = -1 → 稍后用自研显存探测挑批大小")
        return args

    if cuda_ok or want_cpu:
        if not cuda_ok and not args.quick:
            # CPU + 正式训练：不阻止，但把并行度压到合理值
            args.workers = min(args.workers, 2)
        return args

    # —— 以下：想用 CUDA 但本机没有 ——
    print("=" * 62)
    print("[device] 本机未检测到可用的 NVIDIA GPU（torch.cuda.is_available() = False）")
    print("[device] device 已自动回落为 'cpu'。")
    if args.quick:
        print("[device] 当前是 --quick 冒烟模式，CPU 跑几分钟没问题，继续。")
        args.device = "cpu"
        return args

    print()
    print("  ⚠ 正式训练（v1~v4）需要有 GPU 的机器。")
    print("     VisDrone2019-DET @ imgsz=1024 / 200 epoch，CPU 上预计要跑几十天，不现实。")
    print()
    print("  推荐做法（二选一）：")
    print("    A. 先用冒烟模式验证整条链路是否通：")
    print("         python train_v8.py --variant v4 --quick")
    print("       （或双击 一键运行.bat，在菜单里按 s）")
    print("    B. 到免费 GPU 平台跑正式训练（Kaggle 每周 30h 免费 P100）：")
    print("       把「修正版代码」整个目录上传即可，见 KAGGLE.md")
    print()
    print("  若你确实要用 CPU 硬跑（不推荐，仅供小数据调试）：")
    print("         python train_v8.py --variant v1 --force-cpu")
    print("=" * 62)
    if args.force_cpu:
        args.device = "cpu"
        args.workers = min(args.workers, 2)
        args.amp = False
        print("[device] --force-cpu 已开启，将在 CPU 上训练（amp 自动关闭）。")
        return args
    sys.exit(3)


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args = apply_quick(args)
    args = resolve_device(args)
    cache = False if args.cache == "none" else args.cache

    # 0) 输出目录体检：每 epoch 都会写 272MB 级 checkpoint，落在 OneDrive 里会被反复上传。
    if "onedrive" in str(args.project).lower():
        print("[warn] --project 落在 OneDrive 同步目录内：%s" % args.project)
        print("[warn] 每 epoch 的 last.pt / best.pt（约 272MB）都会被同步上传，"
              "既拖慢训练又占带宽。")
        print("[warn] 强烈建议改到本地盘，例如：--project C:/yolo_runs/train")
    print("[run] 输出目录: %s\\%s" % (str(args.project).rstrip("\\/"),
                                     args.name or ("visdrone_" + args.variant)))

    # 1) 注册自定义模块（EMA/DySample/DualConv/GSConv/VoVGSCSP）并启用 NWD 损失
    setup_custom()

    # 2) 建模
    model_yaml = args.model or VARIANTS[args.variant]
    print(f"[train] 结构配置: {model_yaml}")
    model = build_model(model_yaml)

    # 2.5) batch=-1：用自研探测挑安全批大小（AutoBatch 在 Windows WDDM 上不可靠）
    #      注意：torch 只在函数内部 import 过，这里必须先显式导入，否则 batch=-1 时
    #      会直接 NameError: name 'torch' is not defined（--batch -1 恰是 help 里推荐的默认）。
    if args.batch == -1:
        import torch

        if torch.cuda.is_available() and not str(args.device).strip().lower().startswith("cpu"):
            args.batch = probe_batch(model, args.imgsz)

    # 3) 训练
    if args.resume:
        # 续训：必须从 last.pt 加载（ultralytics 的 resume 会从 checkpoint 里读回上次的全部参数），
        # 因此上面那份按 yaml+预训练权重建的 model 在这里用不上，这是正常的。
        run_name = args.name or f"visdrone_{args.variant}"
        last_pt = os.path.join(args.project, run_name, "weights", "last.pt")
        if not os.path.isfile(last_pt):
            print("[resume] 找不到 %s" % last_pt)
            print("[resume] 没有可续的 checkpoint；请去掉 --resume 重新开始，")
            print("[resume] 或检查 --project / --name 是否与中断那次一致。")
            sys.exit(4)
        print("[resume] 从 %s 继续训练" % last_pt)
        YOLO(last_pt).train(resume=True)
        sys.exit(0)

    model.train(
        data=args.data,          # 数据配置（nc 必须与 yaml 中的 10 一致）
        epochs=args.epochs,
        batch=args.batch,        # 视显存调整，建议 8~16
        imgsz=args.imgsz,        # 单一尺度，多尺度交给增强机制
        optimizer="AdamW",
        lr0=5e-4,                # AdamW 初始学习率（原 1e-3 偏高）
        lrf=0.05,                # 终了学习率 = lr0 * lrf
        weight_decay=0.01,
        warmup_epochs=1 if args.quick else 5,   # 原 10 偏长；quick 模式再压到 1
        warmup_bias_lr=0.1,
        cos_lr=True,
        seed=42,
        deterministic=True,      # 配合 seed 保证可复现
        mosaic=1.0,              # 默认值，保留
        close_mosaic=10 if args.epochs >= 20 else 0,   # 最后 10 epoch 关闭 Mosaic（epoch 少时关闭该机制）
        mixup=0.0,               # 关闭 MixUp（小目标任务）
        copy_paste=0.3,          # 仅当数据集含 segment 标注时生效
        fliplr=0.5,
        degrees=10.0,
        scale=0.5,
        fraction=args.fraction,  # 1.0 = 全量；quick 模式只用 3%
        cache=cache,             # False / 'disk' / 'ram'（避免高分辨率全量进内存导致 OOM）
        device=args.device,
        amp=getattr(args, "amp", True),   # CPU 上自动关闭（见 resolve_device）
        workers=args.workers,
        save_period=10,
        project=args.project,    # 绝对路径，避免被前置 runs/<task>/
        name=args.name or f"visdrone_{args.variant}",
        exist_ok=False,          # 不覆盖历史实验
        patience=50,             # 50 个 epoch 无提升则早停（注释与取值一致）
    )
