# 上 Kaggle 跑正式训练（免费 GPU）

本机没有 NVIDIA 显卡，`v1~v4` 的正式训练只能在 GPU 平台上跑。
Kaggle 是目前最省事的：**每周 30 小时免费 GPU（P100 16GB 或 T4×2）**，不用绑卡。

本文件是"照着做就行"的操作手册。

---

## 0. 先在本机把要上传的东西准备好

需要上传两样：**代码** 和 **数据集**。

### 0.1 代码包

把整个 `修正版代码` 目录压成 zip（**不要**把 `runs/` 和 `yolov8m.pt` 塞进去，体积大且白占 20GB 磁盘配额）：

```bash
# 在 修正版代码 的上一级目录执行
# 下面这条会把 runs/、__pycache__/、yolov8m.pt 排除掉
python -c "
import os, zipfile
src = r'修正版代码'
skip_dirs = {'runs', '__pycache__', '冒烟测试数据集', '.workbuddy'}
skip_files = {'yolov8m.pt'}
with zipfile.ZipFile('修正版代码.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f in skip_files or f.endswith('.pyc'):
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, '.'))
print('ok')
"
```

### 0.2 数据集

**关键前提**：先把 VisDrone 标注转成 YOLO 格式，否则训练时会找不到 labels。

```bash
python visdrone2yolo.py --src D:/datasets/VisDrone2019-DET --dst D:/datasets/VisDrone --mode copy
```

> Kaggle 数据集上传后是**只读**的，而且要求上传目录结构原样保留，所以这里用 `--mode copy`
> 生成一份干净的 `images/ + labels/`，直接打包上传。用 `symlink` 的话打包会拿到坏链接。

然后把它压成 zip（约 1.5GB 压缩后更小）：

```bash
cd D:/datasets/VisDrone
# Windows 下用资源管理器右键"压缩为 zip"也行
```

> 如果 Kaggle 上已经有人传过 VisDrone（搜 `VisDrone`），也可以不自己传，
> 直接 `Add Input` 加别人的，但**要确认它已经是 YOLO 格式**，否则还是得自己转。
> 不确认就用自己转的那份，最稳。

---

## 1. 在 Kaggle 创建数据集

1. 打开 https://www.kaggle.com/datasets → **New Dataset**
2. 上传 `VisDrone.zip`（或直接拖整个 `VisDrone` 文件夹）→ 标题填 `visdrone-yolo` → **Create**
3. 同样再建一个 `yolov8m-code`，上传 `修正版代码.zip`

> 上传后目录会被解压到 `/kaggle/input/<数据集名>/`，zip 里的顶层文件夹名会保留。

---

## 2. 建 Notebook 并开 GPU

1. https://www.kaggle.com/code → **New Notebook**
2. 右侧 **Settings**：
   - **Accelerator** → `GPU P100`（或 `GPU T4 x2`）
   - **Internet** → `On`（装依赖必须开；手机号验证过的账号才能开）
   - **Persistence** → `Files only`（可选）
3. 右侧 **Add Input** → 把 `visdrone-yolo` 和 `yolov8m-code` 两个都加上

---

## 3. Notebook 里要跑的 Cell（按顺序）

### Cell 1 — 装环境

```python
# Kaggle 镜像自带 torch，这里只补 ultralytics，并锁定版本与本机一致
!pip install -q ultralytics==8.4.147
import ultralytics, torch
print('ultralytics', ultralytics.__version__)
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())
assert torch.cuda.is_available(), '没有 GPU —— 检查 Settings 里 Accelerator 是否开了'
```

### Cell 2 — 把代码拷到可写目录

```python
import shutil, os, sys
# /kaggle/input 只读，必须拷到 /kaggle/working 才能装补丁、写输出
SRC = '/kaggle/input/yolov8m-code/修正版代码'   # 名字不对就 !ls /kaggle/input 看一眼
WORK = '/kaggle/working/修正版代码'
if os.path.exists(WORK):
    shutil.rmtree(WORK)
shutil.copytree(SRC, WORK)
os.chdir(WORK)
sys.path.insert(0, WORK)
!ls
```

### Cell 3 — 指好数据集路径 + 打补丁 + 自检

```python
DATA = '/kaggle/input/visdrone-yolo/VisDrone'   # 用 !ls /kaggle/input/visdrone-yolo 确认
yaml_path = '/kaggle/working/修正版代码/VisDrone.yaml'
txt = open(yaml_path).read()
import re
txt = re.sub(r'^path:.*$', f'path: {DATA}', txt, flags=re.M)
open(yaml_path, 'w').write(txt)
print(txt[:400])
```

```python
# 把自定义模块打进 ultralytics 源码（EMA / DySample / DualConv / GSConv / VoVGSCSP）
!python install_custom_modules.py
# 逐模块自检：前向形状 + NWD 数值 + 4 份 yaml 的检测尺度
!python test_modules.py
```

`test_modules.py` 全部 `[PASS]` 再往下走。注意 **`v1`/`v2` 是 4 尺度、`v3`/`v4` 是 3 尺度**，
输出里的"检测尺度"必须对得上。

### Cell 4 — 下载预训练权重

```python
# Kaggle 开了 Internet，直接让 ultralytics 自己下；也可以提前传到代码数据集里
!cd /kaggle/working/修正版代码 && python -c "from ultralytics.utils.downloads import safe_download; safe_download('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt', file='yolov8m.pt')"
!ls -la /kaggle/working/修正版代码/yolov8m.pt
```

### Cell 5 — 跑消融（v1 → v4）

```python
import os
os.environ['POLARS_SKIP_CPU_CHECK'] = '1'
os.chdir('/kaggle/working/修正版代码')
# 输出统一放 /kaggle/working，否则关掉 notebook 就没了
!python train_v8.py --variant v1 --project /kaggle/working/runs --name visdrone_v1
```

**重要：Kaggle 单次会话上限 12 小时**，200 epoch @ imgsz=1024 在 P100 上大概率跑不完一个变体。
两个办法：

- **降配置**（推荐先这样把四组都跑出来）：
  ```python
  !python train_v8.py --variant v1 --epochs 100 --imgsz 800 --batch 8 \
      --project /kaggle/working/runs --name visdrone_v1
  ```
- **断点续训**：`train_v8.py` 目前**没有内置 `--resume`**（它每次都从 yaml 重新建图）。
  真要续训，得手动改成从 checkpoint 恢复，或者直接用 ultralytics 原生方式：

  ```python
  from ultralytics import YOLO
  m = YOLO('/kaggle/working/runs/train/visdrone_v1/weights/last.pt')
  m.train(resume=True)   # 注意：这条路径加载的 task 来自 checkpoint
  ```

  > 更省事的做法：一次会话只跑一个变体，跑完立刻把 `best.pt` / `results.csv` 拷出来。

> **四组超参必须完全一致**，否则消融没有可比性（见 `ABLATION.md`）。
> 建议写成一个循环，顺序跑，每组跑完立刻把结果拷出来：

```python
import shutil, os
for v in ['v1', 'v2', 'v3', 'v4']:
    print('='*20, v, '='*20)
    !python train_v8.py --variant {v} --epochs 100 --imgsz 800 --batch 8 \
        --project /kaggle/working/runs --name visdrone_{v}
    # 只留关键产物，避免 /kaggle/working 被 20GB 配额撑爆
    src = f'/kaggle/working/runs/train/visdrone_{v}'
    for f in ['results.csv', 'results.png', 'weights/best.pt', 'weights/last.pt']:
        p = os.path.join(src, f)
        if os.path.exists(p):
            print('keep', p)
```

---

## 4. 把结果拿回来

`/kaggle/working` 里的文件在 notebook 关掉后仍会保留，可以直接下载：

- `runs/train/visdrone_v*/results.csv` —— **论文里画曲线就靠它**（每个 epoch 的 loss / mAP50 / mAP50-95）
- `runs/train/visdrone_v*/weights/best.pt` —— 验证/推理用
- `args.yaml` —— 超参留档，盲审时证明四组一致

下载方式：Notebook 右侧 **Output** 面板可以逐个下载，或者：

```python
import shutil
shutil.make_archive('/kaggle/working/results_all', 'zip', '/kaggle/working/runs')
# 然后在 Output 面板下载 results_all.zip
```

---

## 5. 踩坑清单

| 现象 | 原因 / 处理 |
|---|---|
| `ValueError: Invalid CUDA 'device=0' requested` | 没开 Accelerator。Settings → Accelerator 选 GPU |
| `Add Input` 开了但仍 `ModuleNotFoundError` | 自定义模块补丁没打。先跑 `install_custom_modules.py`，**改完源码需要新进程**才能生效 |
| `No labels found` 或 mAP 恒为 0 | `VisDrone.yaml` 的 `path:` 没指对，或数据集不是 YOLO 格式 → 先跑 `visdrone2yolo.py` |
| `[ERROR] ... repeat_modules` | `install_custom_modules.py` 没跑到。ultralytics 版本不对就退回 `8.4.147` |
| 磁盘满 / 会话被 kill | Kaggle `/kaggle/working` 只有 20GB。跑完一个变体就删掉中间产物（`train_batch*.jpg`、`epoch*.pt`） |
| 一断网就崩 | Kaggle 开了 Internet 后，长训练中途断网会让 `results.csv` 写入失败。关键结果勤 `make_archive` |

---

## 6. 备选平台

| 平台 | 免费额度 | 备注 |
|---|---|---|
| **Kaggle** | 30 h/周，P100 16GB | 最省事，不用绑卡，单会话 12h |
| **AutoDL** | 新人券 / 按量付费 | 国内直连快，T4/3090 都有；批量跑四组最舒服 |
| **Colab** | 免费 T4，时不常断 | 交互式，跑长任务不稳，适合调试 |
| **ModelScope 创空间** | 免费 GPU 额度 | 国内，适合短期实验 |

四组消融的算力需求不大（每组 100 epoch @ 800px，P100 上大约 6~10 小时），
**Kaggle 一周的 30 小时额度够跑完好几轮**。
