# 北斗 + 改进 YOLOv8m 小目标识别辅助系统

> 面向无人机航拍（VisDrone2019-DET）场景的小目标检测改进方案，并作为北斗（BDS）高精度定位 / 短报文回传能力的前端感知模块。
> 作者：llqvpll。

## 项目简介

在无人机对地观测中，行人、车辆等目标在画面里占比极小、易被漏检。本项目在 **YOLOv8m** 上做了一系列面向小目标的改进，
并把它接到北斗终端上：检测出的目标框经北斗短报文回传坐标，结合北斗高精度定位实现"通、导、授、救"一体化的辅助系统。

### 模型改进点（消融可复现）

| 模块 | 作用 |
|---|---|
| **EMA 注意力（EMA-lite）** | 跨通道-空间注意力，强化小目标响应 |
| **DySample 动态上采样** | 可学习偏移的上采样，替换 CARAFE，更轻 |
| **DualConv 双路轻量卷积** | 并联 + channel shuffle，提质不暴涨参数量 |
| **Slim-Neck（GSConv）** | 颈部轻量化，降参数量/算力 |
| **NWD 损失** | 归一化 Gaussian Wasserstein 距离，缓解小目标 IoU 不敏感 |
| **P2 检测分支** | 增加大分辨率检测头，专治极小目标 |

四个消融变体见 [`修正版代码/ABLATION.md`](修正版代码/ABLATION.md)：
`v1` 基线（DualConv，4 尺度）→ `v2`（+Slim-Neck）→ `v3`（剪 P5，3 尺度）→ `v4`（两者，推荐）。

实测（ultralytics 8.4.147，CPU 冒烟）：v4 参数量 **17,552,134**、比 v1 少 **29.5%**，结构可跑通、可复现。

## 目录结构

```
.
├── 修正版代码/                 # 全部可运行代码 + 说明（详见其 README.md）
│   ├── train_v8.py             #   训练入口，--variant 切换消融变体
│   ├── custom_loss.py          #   NWD + DFL 损失
│   ├── custom_modules/         #   自定义模块（EMA / DySample / DualConv / GSConv / 安装器）
│   ├── *.yaml                  #   4 份变体结构 + VisDrone 数据配置
│   ├── visdrone2yolo.py        #   VisDrone 官方标注 → YOLO 格式转换（必跑一次）
│   ├── install_custom_modules.py  # 把自定义模块写入 ultralytics 源码（幂等 + 备份）
│   ├── test_modules.py / 环境验证.py / 显卡检查.py  # 自检 / 排障
│   ├── KAGGLE.md / 一键运行.bat   # 云 GPU 手册 / Windows 一键
│   └── server/                 # 生产级推理后端（FastAPI）
├── YOLOv8m改进模型_代码优化分析报告.md   # 问题分级清单（P0/P1/P2）+ 修复说明
├── 技术分析_北斗YOLOv8m小目标识别辅助系统.md  # 技术分析报告
├── 开源项目参考清单.md                    # 参考的开源项目 / 论文
├── LICENSE
└── .gitignore
```

## 快速开始

```bash
pip install ultralytics==8.4.147
cd 修正版代码
python install_custom_modules.py     # 1) 自定义模块写入 ultralytics（一次）
# 重新开一个终端
python test_modules.py               # 2) 自检
python visdrone2yolo.py --src <VisDrone2019-DET> --dst <dst> --mode copy  # 3) 转数据
python train_v8.py --variant v4      # 4) 训练（需 GPU；CPU 仅建议 --quick 冒烟）
```

无 GPU 时先本地冒烟（无需数据集）：
```bash
python make_smoke_data.py
python train_v8.py --variant v4 --quick
```
完整说明、显存配置、Kaggle / 云 GPU 流程见 [`修正版代码/README.md`](修正版代码/README.md)。

## 许可与引用

- 本项目以 **MIT** 许可发布（见 [LICENSE](LICENSE)）。
- `custom_modules/gsconv.py` 整理自 [Slim-Neck by GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv)，请遵守其许可并在论文中引用。
- 检测基座为 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0），本仓库不 bundled 其源码。
- VisDrone2019-DET 数据集版权归 VisDrone 挑战赛方，须按其许可使用；本仓库不含数据。

如用于论文，请引用 VisDrone、YOLOv8、Slim-Neck(GSConv)、NWD、DySample、EMA 等相关工作。
