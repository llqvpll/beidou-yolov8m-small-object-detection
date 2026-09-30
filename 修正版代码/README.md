# 修正版代码说明（YOLOv8m 改进模型）

对应分析报告：`../YOLOv8m改进模型_代码优化分析报告.md`
消融矩阵说明：`ABLATION.md`

## 目录结构
```
修正版代码/
├── train_v8.py                             # 训练入口（--variant 切换消融变体）
├── custom_loss.py                          # NWD + DFL 损失（公式对齐、去 in-place、除零保护）
├── ABLATION.md                             # 2×2 消融矩阵：改了什么、怎么跑、预期
├── yolov8m_visdrone_improved.yaml          # v1 基线：DualConv，4 尺度
├── yolov8m_visdrone_slimneck.yaml          # v2：+Slim-Neck，4 尺度
├── yolov8m_visdrone_p2_nop5.yaml           # v3：剪 P5，3 尺度
├── yolov8m_visdrone_slimneck_nop5.yaml     # v4：Slim-Neck + 剪 P5（推荐）
├── VisDrone.yaml                           # 数据配置（nc=10，train_v8.py 默认数据源）
├── visdrone2yolo.py                        # 把 VisDrone 官方标注转成 YOLO 格式（必跑一次）
├── install_custom_modules.py               # 一键把自定义模块写入 ultralytics 源码
├── test_modules.py                         # 自检脚本（形状 / 损失 / 建模）
├── make_smoke_data.py                      # 生成合成小数据集（冒烟测试用，免下载 VisDrone）
├── 冒烟测试数据集/                          # ↑ 生成物：images/labels + smoke.yaml
├── 环境自检.py                             # 排障：找出被删空的包并覆盖式修复
├── 环境验证.py                             # 换机器后第一件事：一次跑完环境/显卡/补丁/权重/数据集检查
├── 显卡检查.py                             # 显卡专项：确认这张卡真能用来训练（含 sm_120 检查）
├── KAGGLE.md                               # 上 Kaggle 跑正式训练的操作手册（逐 Cell 可复制）
├── 一键运行.bat                            # Windows 一键：装环境 → 下权重 → 打补丁 → 自检 → 训练
└── custom_modules/
    ├── __init__.py
    ├── ema_attention.py                    # 注意力模块（补 BN+SiLU+分组激活）
    ├── dysample.py                         # 动态上采样（可学习偏移 + grid_sample）
    ├── dualconv.py                         # 双路轻量卷积（并联 + channel shuffle）
    ├── gsconv.py                           # Slim-Neck：GSConv / GSBottleneck / VoVGSCSP
    ├── installer.py                        # 源码补丁安装器（幂等 + 备份）
    └── register.py                         # NWD 损失挂接 + 安装检查
```

## 怎么跑
```bash
pip install ultralytics                # 建议 8.1.x ~ 8.4.x（已在 8.4.147 上逐行核对）
python install_custom_modules.py       # 1) 把自定义模块写进 ultralytics（只需一次）
# —— 关掉当前 Python 进程，重新开一个终端 ——
python test_modules.py                 # 2) 自检：形状 / 损失 / 建模

# 3) 数据：VisDrone 官方标注不是 YOLO 格式，先转一次（只需一次）
python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --dst D:/datasets/VisDrone --mode copy

# 4) 训练：--variant 选消融变体（v1 基线 / v2 +SlimNeck / v3 剪P5 / v4 两者）
python train_v8.py --variant v4 --name visdrone_v4
```
> 嫌麻烦直接双击 **`一键运行.bat`**，它会自动找 Python、装 ultralytics、打补丁、自检、再让你选变体开训。
> `install_custom_modules.py` 会备份原文件为 `*.bak`，并且可以重复执行不会重复插入。
> 若中途想还原：把 `ultralytics/nn/tasks.py.bak`、`ultralytics/nn/modules/__init__.py.bak` 覆盖回去即可。

## 数据配置（默认已指向本目录）
`train_v8.py` 的 `--data` 默认是**本目录下的 `VisDrone.yaml`**（绝对路径，换工作目录也不会丢），
不再依赖 `ultralytics/cfg/datasets/` 里的那份。所以只需改一处：

```yaml
# VisDrone.yaml
path: D:/datasets/VisDrone   # ← 改成你本机 VisDrone2019-DET 的根目录，建议绝对路径
train: images/train
val:   images/val
test:  images/test
```
目录结构要求：`<path>/images/{train,val,test}/*.jpg` 与 `<path>/labels/{train,val,test}/*.txt`（YOLO 格式、同名）。

⚠️ **VisDrone 官方标注不是 YOLO 格式**，直接指过去只会得到 "No labels found" 或 mAP 恒为 0。
官方每行是 8 个逗号分隔字段：
```
bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion
```
用 `visdrone2yolo.py` 转一次即可：
```bash
python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --dst D:/datasets/VisDrone --mode copy
# 参数：--mode symlink 用软链接省磁盘；--only train 只转一个 split
```
它会自动丢掉 `score==0`（官方"忽略区域"）、`category==0`（ignored regions）、
`category==11`（others），并把 `category 1~10` 映射成 `0~9`；越界框先裁到画布内再归一化。
转完把 `VisDrone.yaml` 的 `path:` 指到 `--dst` 就行。

> `--mode symlink` 在打包上传云平台时**会拿到坏链接**，那种场合要用 `--mode copy`。

想临时换一份数据：`python train_v8.py --variant v4 --data /path/to/other.yaml`。

第一轮训练前还需把 `yolov8m.pt` 放到本目录（脚本按名称对齐加载预训练权重）。

## 没有 GPU 怎么办（重要）
CPU 上跑 YOLOv8m + imgsz=1024 + VisDrone 是**不现实**的（一个 epoch 可能就是几小时到十几小时）。
所以脚本内置了**设备自适应**，不会让你在一台没卡的机器上撞墙：

- `train_v8.py` 会先探测 `torch.cuda.is_available()`：
  - 有卡 → 用 `--device 0`（默认）；
  - **没卡 + `--quick`** → 静默回落 `cpu`，顺带把 `workers` 降到 ≤2、关掉 `amp`，继续跑；
  - **没卡 + 正式训练（v1~v4）** → 打印"CPU 跑不现实 + 去 Kaggle"的指引后**退出（码 3）**，
    不会傻跑几十天。`一键运行.bat` 认得这个退出码，会提示"已跳过"而不是报错。
- 确实想用 CPU 硬跑（小数据调试）：
  ```bash
  python train_v8.py --variant v1 --force-cpu
  ```
- 以前写死 `--device 0` 时，在没有 N 卡的机器上会直接抛
  `ValueError: Invalid CUDA 'device=0' requested` —— 现在不会了。

分两步走：

**第一步：本地冒烟测试，不需要任何数据集、也不需要 GPU**（几分钟）
```bash
python make_smoke_data.py          # 生成合成小数据集（24 训练 / 8 验证，10 类）
python train_v8.py --variant v4 --quick
```
`--quick` 会自动改用 `冒烟测试数据集/smoke.yaml`（不用先下 1.5GB 的 VisDrone），
参数压到 3 epoch / imgsz 640 / batch 4 / device cpu / workers 0。
产物在 `runs/train/smoke_v4/`。

**判据**：能完整跑完 3 个 epoch + 验证 + 存出 `weights/best.pt`，
就说明**自定义模块、NWD 损失、数据管线、验证链路全部是通的** —— 论文里"代码可复现"这句就有据可依。
（合成数据上的 mAP 当然是 0，那是随机噪声，没有意义；看的是"跑得通"而不是"跑得准"。）

> 本机实测（Python 3.10.8 + ultralytics 8.4.147 + torch 2.14.0+cpu）已跑通：
> v4 = **17,552,134** 参数 / 79.3 GFLOPs / 321 层，3 epoch 用时约 0.09 h。

**第二步：去云 GPU 跑正式实验**（消融矩阵 4 个变体）
**完整操作手册见 `KAGGLE.md`**（逐 Cell 可复制，含数据集打包、路径改写、四组消融循环、踩坑清单）。
- Kaggle Notebooks：每周 30 小时 P100，免费，不用绑卡；
- AutoDL / 恒源云：按小时租 3090/4090，国内直连快；
- Colab：免费 T4（时长不稳，适合调试）。
传上去后同样先 `pip install ultralytics==8.4.147 && python install_custom_modules.py`，
再 `python train_v8.py --variant vN`。**四组超参必须一致**，否则消融没有可比性（见 `ABLATION.md`）。

---

## 换到一台有显卡的电脑上跑（例如 RTX 5060 Ti 8GB + i5-14600KF）

### 拷贝什么
把整个 `修正版代码` 目录拷过去即可。**建议先删掉 `runs/`**（是训练产物，几百 MB，新机器会重新生成）。

拷过去之后，有三件事**必须在新机器上重做**：

| 事项 | 为什么 |
|---|---|
| 重跑 `install_custom_modules.py` | 补丁是写进**那台机器的 ultralytics 源码**里的，不是写在本目录里 |
| 改 `VisDrone.yaml` 的 `path:` | 两台机器上数据集的盘符/目录通常不一样 |
| 重装 ultralytics / torch | 环境不能跨机器搬 |

> 直接双击 `一键运行.bat` 就会自动做完前两件里的第一件（打补丁）+ 装环境，所以最省事。
> 但 `VisDrone.yaml` 的路径要你自己改。

### 第 0 步：先跑环境验证（一条命令说清所有问题）

拷过去之后，**别急着训练，先跑这个**：

```bash
python 环境验证.py
```

它把"能不能开始训练"需要的所有前置条件检查一遍，最后给明确结论（退出码 0 = 就绪，1 = 有必须解决的问题）：

| 步骤 | 检查内容 |
|---|---|
| [1] | Python 版本（ultralytics 要 3.8~3.12） |
| [2] | 13 个核心依赖 + 上游声明依赖 + 可选依赖，逐个 `import` 实测 |
| [3] | 显卡 / CUDA / **sm_120 架构匹配** / 真实 GPU 运算 / 按显存给出配置建议 |
| [4] | 5 个自定义模块有没有正确注册进 `base_modules` 和 `repeat_modules` |
| [5] | `yolov8m.pt` 是否存在，并**加载权重数参数量**鉴定真假（官方 25,902,640） |
| [6] | `VisDrone.yaml` 的 `path` 是否有效、images/labels 是否配对、标签格式抽检 |
| [7] | 代码文件齐全 + 4 份 yaml 的 `nc` 是否都等于 10 |

加餐参数：

```bash
python 环境验证.py --full                 # 再跑 test_modules.py：构建 4 个变体并前向（约 1~3 分钟）
python 环境验证.py --bench --imgsz 1024   # 顺便实测训练速度，直接估算"要跑多久"
```

`--bench` 会从 batch=8 往下试、OOM 就退一档，找到装得下的 batch，然后报出
"每次迭代耗时 / 每 epoch 耗时 / 100 epoch 总时长 / 四变体总时长"。

> 说明：这个脚本是**只读检查**，不会改你的环境。要修的话它会明确告诉你是哪一条命令。

### ⚠️ 第一件事：确认显卡真的能用（RTX 50 系特有）

RTX 5060 Ti 是 **Blackwell 架构、计算能力 sm_120**。
如果装到的 PyTorch 是用较老的 CUDA 编的，**安装阶段不会报错，`torch.cuda.is_available()` 也是 True**，
但在**第一次 GPU 运算时**才会炸：

```
CUDA error: no kernel image is available for execution on the device
```

所以上机第一件事是跑体检（`一键运行.bat` 检测到显卡会自动帮你跑）：

```bash
python 显卡检查.py          # 基础检查 + 真实 GPU forward/backward
python 显卡检查.py --full    # 再加上用真实 v4 模型跑一遍（需先打过补丁）
```

它会打印卡名、显存、`sm_xxx`、torch 支持的架构列表，并真的在 GPU 上跑一次
conv / BN / SiLU / 池化 / 双线性缩放 / **grid_sample**（DySample 用到的）的 forward + backward。
如果 sm_120 没被编进 torch，它会直接给出修复命令：

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
（索引名以 https://pytorch.org/get-started/locally/ 当前给的为准；关键是 **CUDA 12.8 及以上**。）

### 8GB 显存怎么配

`train_v8.py` 的默认值已经按"小显存也不炸"调过：

```bash
python train_v8.py --variant v4            # batch 默认 -1 = 自研显存探测，imgsz 默认 1024
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch -1` | ✅ | **自研显存探测**：实测前向+反向峰值显存 ×1.3 安全系数后取最大可容纳的 batch。不用 ultralytics AutoBatch——Windows WDDM 会把共享内存当显存虚报，AutoBatch 挑出的 batch 实际会溢出变慢甚至 profiling 直接崩。1024 上一般落到 **3~4** |
| `--imgsz 1024` | ✅ | 与原代码一致，利于消融可比。若探测给出的 batch < 2，就降到 `--imgsz 800` 或 `640` |
| `--workers 8` | ✅ | i5-14600KF 有 14 核，够用；数据增强不会成为瓶颈 |
| `--cache disk` | ✅ | 别用 `ram`，VisDrone 全量进内存容易爆 |

**四条铁律**：`imgsz`、`epochs`、`optimizer`、`lr0` 在四个变体之间必须完全一样，否则消融不成立。

> ⚠️ 一个容易忽略的点：我们的结构带 **P2 检测分支**（stride=4），
> 在 imgsz=1024 时 P2 特征图是 256×256，激活值占用比普通 yolov8m 大不少。
> 这就是 8GB 上 batch 只能给到 2~4 的原因。

### 时间预估（VisDrone2019-DET train = 6,471 张）

以下按 RTX 5060 Ti 8GB + AMP 半精度估算，**仅供参考**（实测：1024/batch3 ≈ 60 min/epoch），
实际请先用下面的"测速"步骤量一次：

| imgsz | 约 min/epoch | 100 epoch | 200 epoch | 四个变体（各 100 epoch） |
|---|---|---|---|---|
| 640  | 4 ~ 5   | 7 ~ 9 h    | 14 ~ 18 h | 约 1.5 天 |
| 800  | 7 ~ 8   | 12 ~ 14 h  | 24 ~ 28 h | 约 2 天   |
| 1024 | 11 ~ 15 | 18 ~ 25 h  | 36 ~ 50 h | 约 3 ~ 4 天 |

**先测速再决定**（10 分钟，比猜准得多）：

```bash
python train_v8.py --variant v1 --epochs 3 --imgsz 1024 --batch -1 --name _speedtest
```
跑完看控制台里每个 epoch 的用时（或 `runs/train/_speedtest/results.csv`），
乘以你要的总 epoch 数即可。测完把 `runs/train/_speedtest/` 删掉。

结论：**这台机器完全跑得动，慢在 8GB 显存限制了 batch**。
赶时间的话两条路并行 —— 本机跑 `imgsz=800` 的两组，Kaggle 免费 P100 跑另外两组（见 `KAGGLE.md`）。
注意 Kaggle 的 P100 算力其实不比 5060 Ti 强，优势只在 16GB 显存和可以并行开多个会话。

> `一键运行.bat` 已经做了自动适配：检测到没有 NVIDIA 显卡时，会先装 **CPU 版 torch**（省掉约 2GB 的 CUDA 运行时）；
> 会自动下载 `yolov8m.pt`（用 `curl --ssl-no-revoke`，并带两个镜像兜底）；
> 并在第 6 步给一个 `s` 选项，一条龙生成合成数据集 + 跑冒烟测试。

## 排障一：`pip list` 里明明装了，`import` 却 ModuleNotFoundError
症状举例：`ModuleNotFoundError: No module named 'cv2'`，但 `pip list` 显示 opencv-python 已安装。
这是 **site-packages 里的包目录被部分/整体删掉、只剩 dist-info** 造成的（磁盘清理工具、杀软隔离、
或被中断的 pip 卸载都可能留下这种"半残"状态；pip 卸载会先把包改名成 `~xxx` 再删，中断就留下 `~xxx` 孤儿目录）。

直接用脚本体检 + 修复：
```bash
python 环境自检.py          # 只体检：扫描 dist-info/RECORD + 关键模块导入检查
python 环境自检.py --fix    # 修复：覆盖式重装受损包（不卸载其他包）
```
核心修复命令就是（**别用 `--force-reinstall`**，它会走卸载流程、中途失败会再留半残）：
```bash
pip install --ignore-installed --no-deps opencv-python certifi kiwisolver mpmath polars requests
```

## 排障二：`polars` 报 `unknown feature flag: 'sse3'`
`polars` 启动时会做 CPU 指令集自检；部分机器上取不到 `cpuid`，映射表为空 →
任何指令集都会被误判成"未知"（报错文案就是 `sse3`）。本机 CPU 实际支持 sse3/ssse3/avx2，
所以这是**误报**，用官方开关跳过即可：
```bash
set POLARS_SKIP_CPU_CHECK=1        # cmd；bash 用 export
```
`train_v8.py` / `test_modules.py` / `环境自检.py` 内部都已自动设置这个变量，`一键运行.bat` 也设了，
正常跑不需要手动处理。只有你在别的脚本里 `import polars` 时才会遇到。

## 为什么必须装进 ultralytics 源码
`parse_model()` 内部：
```python
if m in base_modules:                 # 内置模块
    c1, c2 = ch[f], args[0]
    c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]        # 自动补通道 + 按 width 缩放
else:
    c2 = ch[f]                        # 自定义模块：args 原样传入，不缩放
```
自定义模块若不进 `base_modules`，就会拿到未经缩放的参数、且无法得知真实输入通道。
所以 `install_custom_modules.py` 做三件事：复制模块文件、在模块包 `__init__.py` 导出、
把类名插入 `tasks.py` 的 `base_modules` 集合并导入到 `tasks` 命名空间。

**额外一处（容易漏）**：`VoVGSCSP` 这类"会重复堆叠"的模块还必须进 `repeat_modules`，
否则 yaml 里的 `repeats`（如 `[-1, 3, VoVGSCSP, [512]]` 的 3）不会被插入构造函数，
只会建一层——**结构静默变浅，不报错**。`installer.py` 的 `REPEAT_NAMES` 已处理。

## 相对原附录改了什么（逐条对应）

| 编号 | 原问题 | 修正 |
|---|---|---|
| P0-1 | `YOLO(weights, cfg)` 把配置塞进 `task`，改进从未生效 | 改为 `YOLO(yaml).load(weights)`，并显式 `setup_custom()` 挂接模块与损失 |
| P0-2 | 两份 yaml 均不可加载（`backbone: ...` 占位、v5 字段） | 合并为一份完整 v8 结构，显式 `scale: m` |
| P0-3 | 自定义模块未注册 → 落到 `else: c2=ch[f]` 分支，参数不缩放 | `install_custom_modules.py` 幂等写入 `base_modules` 并导出（带 `.bak` 备份） |
| P0-4 | `nc: 80` | 改为 `nc: 10`（VisDrone） |
| P0-5 | 多 Detect 头 + anchors（v5 写法） | 单 Detect 输出 4 个尺度，anchor-free |
| P0-6 | `device` 写死 `'0'`，无 N 卡机器直接 `ValueError` | 探测 `torch.cuda` 后自适应：有卡用卡，无卡冒烟走 cpu、正式训练给指引并退出（码 3），另留 `--force-cpu` |
| P1-1 | NWD 公式漏 `/4`、缺 `sqrt` | `W2 = 中心距² + 宽高差²/4`，`1 - exp(-sqrt(W2)/C)` |
| P1-2 | `clamp_` 就地改输入 | 改为 `clamp` |
| P1-3 | 除零 / 空正样本批次出 NaN | `target_scores_sum = max(..., 1)`，空批次返回 0 |
| P1-4 | DySample 门控只衰减、卷积放错位置 | 改可学习偏移上采样；1x1 卷积前移到低分辨率（省 ~75% FLOPs） |
| P1-5 | DualConv 串联、无 shuffle、无激活 | 改并联 + channel shuffle + BN/SiLU |
| P1-6 | EMAAttention 缺 BN/激活 | 补 BN + SiLU；命名建议改为“EMA-lite / 注意力引导” |
| P2-1 | `imgsz=[1024,1280]` | `imgsz=1024` |
| P2-2 | `cache=True` 易 OOM | `cache='disk'` |
| P2-3 | `mixup=0.5` 伤小目标 | `mixup=0.0` |
| P2-4 | `augment=True` 非法参数 | 删除 |
| P2-6 | `exist_ok=True` 覆盖实验 | `exist_ok=False` |
| P2-7 | patience 注释/取值不一致 | 统一为 50 |
| P2-8 | warmup 偏长 / AdamW lr 偏高 | `warmup_epochs=5`、`lr0=5e-4` |
| P2-9 | 不可复现 | 增加 `deterministic=True` |

## 仍需你确认/调参的点
1. **NWD 的 `C`**：ultralytics 内部 bbox 以 stride 归一化后的“格点”为单位，`C=12` 是经验值，建议扫描 `C ∈ {6,8,12,16,20}` 做消融。
2. **`copy_paste=0.3`**：只有数据集含 `segment` 标注时才生效；纯检测标注可删。
3. **P2 的延迟问题**：已提供两个解法——(a) 用 `v3/v4` **剪掉 P5 分支**；(b) 若要完全去掉 P2，删掉 yaml 中 node 18~20 并把 `Detect` 输入改为 `[23,26,29]`。建议用 `v1 vs v3` 的实测延迟来量化。
4. **`DySample` 命名**：本实现是 DySample 思想的轻量化版本；若论文严格要求复现原论文，需补 `groups` 与 `pixel_unshuffle('pl')` 分支。
5. **Slim-Neck 的引用**：`gsconv.py` 按 *Slim-neck by GSConv* 论文结构整理，来源 `github.com/AlanLi1997/slim-neck-by-gsconv`，请按其 LICENSE 使用并在论文中引用。
6. **ultralytics 版本**：`parse_model` 的通道逻辑在不同版本略有差异，若报通道相关错误，把报错贴回来即可定位。

## 关于结果可复现的提醒
原代码的 `nc=80` 如果是真实训练配置，则**论文指标可能并非来自这份代码**；修正后请重跑关键实验，确保“论文描述 = 代码 = 权重”三者一致。

