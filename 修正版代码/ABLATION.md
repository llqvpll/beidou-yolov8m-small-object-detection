# 改动说明 · 2×2 消融矩阵

> 本次在 `修正版代码/` 里新增了两项结构改进，并做成了**四组可对比的配置**，
> 目的：让你在盲审里能拿出一张干净的消融表，而不是"我加了四个模块，涨了"。

---

## 1. 这次改了什么（两件事）

### 改动 A：Slim-Neck（GSConv / VoVGSCSP）—— 轻量化
- **把 Neck 里的 `C2f` 换成 `VoVGSCSP`**，**把下采样 `DualConv` 换成 `GSConv`**。
- 来源：论文 *Slim-neck by GSConv: A lightweight-design for real-time detector architectures*，
  官方实现 `github.com/AlanLi1997/slim-neck-by-gsconv`。
- 为什么换：`DualConv` 是你自己写的模块（无文献背书），而 GSConv 是**已发表、被大量复现**的方法。
  盲审里"引入 Slim-Neck"比"自己设计了个 DualConv"更站得住脚。
  而且两者都是"Neck 轻量化"，消融对比非常自然。
- 新增文件：`custom_modules/gsconv.py`（含 `GSConv` / `GSBottleneck` / `VoVGSCSP`）。

### 改动 B：剪枝 P5 检测分支 —— 抵消 P2 的延迟
- **保留 P2（stride=4）分支，删掉 P5（stride=32）分支**，检测头由 4 尺度变 3 尺度。
- 来源思路：DRS-YOLO（Meas. Sci. Technol. 36 095413, 2025）——"增加高分辨率分支，同时剪枝低分辨率分支"。
- 为什么：VisDrone 里几乎没有大目标，P5 基本在浪费；
  而 P2 是**实打实的延迟大户**（有开源实测：Jetson 上 31.8ms → 60.6ms）。
  加 P2 又砍 P5，等于"把算力从没用的地方挪到有用的地方"。

---

## 2. 四组配置（就是你的消融表）

| 变体 | 文件 | Slim-Neck | P5 分支 | 检测尺度 | 说明 |
|---|---|---|---|---|---|
| **v1** | `yolov8m_visdrone_improved.yaml` | ✗（用 DualConv） | ✓ | P2/P3/P4/P5 | 基线（上一轮成果） |
| **v2** | `yolov8m_visdrone_slimneck.yaml` | ✓ | ✓ | P2/P3/P4/P5 | 只换轻量 Neck |
| **v3** | `yolov8m_visdrone_p2_nop5.yaml` | ✗ | ✗ | P2/P3/P4 | 只剪 P5 |
| **v4** | `yolov8m_visdrone_slimneck_nop5.yaml` | ✓ | ✗ | P2/P3/P4 | **推荐主方案** |

**唯一变量原则**：v2 相对 v1 只动了 Neck 的卷积块；v3 相对 v1 只动了检测尺度。
所以每一格的差异都能干净地归因到一个改动上——这正是消融该有的样子。

### 预期（方向性，需你实跑确认）
| 对比 | 预期参数量/GFLOPs | 预期 mAP | 预期延迟 |
|---|---|---|---|
| v1 → v2 | ↓（Neck 变轻） | ≈ 或 ↑小幅 | ↓ |
| v1 → v3 | ↓（少一个尺度头） | ≈ 或 ↑小幅（小目标召回更集中） | ↓↓ |
| v1 → v4 | ↓↓ | ↑ | ↓↓ |

> 具体数字必须你自己跑。**不要引用别人的数字放进你的表**（不同 baseline / 训练配置不可比）。

---

## 3. 怎么跑

```bash
pip install ultralytics
python install_custom_modules.py     # 已升级：现在会一并注册 GSConv / VoVGSCSP
# —— 重启终端 ——
python test_modules.py               # 自检

# 四组依次跑（--variant 切换）
python train_v8.py --variant v1 --name visdrone_v1
python train_v8.py --variant v2 --name visdrone_v2
python train_v8.py --variant v3 --name visdrone_v3
python train_v8.py --variant v4 --name visdrone_v4
```

跑完看 `runs/train/visdrone_v*/results.csv` 与 `weights/best.pt`，
再用 `yolo val model=runs/train/visdrone_v4/weights/best.pt data=VisDrone.yaml` 出指标。

> **显存/时间提示**：imgsz=1024 + P2 分支很吃显存。如果 v1/v2/v4 跑不动，
> 先把 `--batch 4` 或 `--imgsz 768` 降下来，四组**用同一套超参**才可比。

---

## 4. 技术上做了什么保证

因为本机没有 torch，不能实跑，所以我做了两层静态验证：

1. **结构推演**：模拟 ultralytics 的 `parse_model` 通道计算，确认四个 yaml
   - 所有 `from` 索引合法；
   - `EMAAttention` / `DySample` 的 `in == out` 约束成立；
   - `GSConv` 输出通道恒为偶数（它内部要 `c2 // 2` 对半分）；
   - 检测头输入通道递增（`[96,192,384,576]` / `[96,192,384]`），与官方 YOLOv8 的规律一致。
   结果：**四份全部通过**。

2. **安装器升级验证**：在临时假目录上实跑补丁，确认
   - `GSConv / VoVGSCSP` 正确写入 `base_modules`；
   - **`VoVGSCSP` 同时写入 `repeat_modules`**（否则 yaml 里的 `repeats=3` 不会被传进构造函数
     —— 这是新踩到的坑，已修）；
   - 补丁后文件 `compile()` 通过；重复执行幂等。

**还没验证的**：实跑训练、真实 mAP、真实延迟。这三项只能在你机器上确认。

---

## 5. 还没做、但建议做的（按性价比排序）

1. **SAHI 切片推理**：`github.com/obss/sahi`，不改模型不用重训，VisDrone AP 有公开的 +6.8%（切片微调累计 +12.7%）。
   而且能和你"辅助系统"的定位绑定（切片坐标 → 北斗地理坐标），比单纯刷 mAP 更有故事。
2. **NWD 用满**：现在只用了"损失"这一处，论文原意是**标签分配 + NMS + 损失**三处都能用。
3. **MPDIoU / Focaler-WIoU**：换掉 NWD 做对比实验，证明 NWD 在你的数据上更好（消融的说服力）。
4. **部署导出**：`model.export(format='onnx')` → TensorRT，P2 的延迟问题只有实测才说得清。

要哪一项直接说，我接着改。
