# 相关工作与开源生态调研（Related Works）

> 检索时间：2026-10-01
> 检索方式：GitHub Search API（`sort=stars`，关键词见各节）
> 用途：作为参赛技术报告 / 论文「相关工作」章节的素材；star 数为检索时快照，会随时间变化。

## 1. 总体态势：成熟但同质化

| 检索关键词 | 命中仓库数 |
|---|---|
| `VisDrone YOLO` | 231 |
| `YOLOv8 UAV detection` | 298 |
| `small object detection YOLOv8` | 118 |
| `drone aerial object detection YOLO` | 52 |

结论：**「YOLO + 无人机航拍 + 小目标」是高度成熟的赛道**，纯检测精度改进难以形成差异化。
真正稀少的是 **YOLOv8m（中等体量）+ VisDrone** 这一组合——全网仅 2 个仓库命中（其中之一为本项目，
另一个为 `jontyroades2006-lang/Drone-Traffic-Object-Detection-and-Tracking`，用 YOLOv8m + ByteTrack + VisDrone）。

## 2. 分类梳理

### 2.1 VisDrone + YOLO：数据集工具与训练基线
| Stars | 仓库 | 一句话定位 | 与本项目异同 |
|---|---|---|---|
| 84 | [dronefreak/VisDrone-dataset-python-toolkit](https://github.com/dronefreak/VisDrone-dataset-python-toolkit) | 33 个模型的训练/评测/视频推理/标注转换工具箱 | 同数据、同目标；它是「工具箱」，本项目是「带北斗回传的改进模型」 |
| 61 | [xuanandsix/VisDrone-yolov8](https://github.com/xuanandsix/VisDrone-yolov8) | 多 yolov8 变体在 VisDrone 上训练 | 基线性质，无结构改进与回传 |
| 47 | [adityatandon/VisDrone2YOLO](https://github.com/adityatandon/VisDrone2YOLO) | VisDrone 标注转 YOLO 格式 | 与本项目 `visdrone2yolo.py` 同功能，可互为参考 |
| 11 | [SEMHAQ/YOLO11_VisDrone](https://github.com/SEMHAQ/YOLO11_VisDrone) | YOLO11 改进上 VisDrone | 更新架构，但同样无回传闭环 |
| 9 | [Qxy661/yolo-applications](https://github.com/Qxy661/yolo-applications) | VisDrone 低空检测 + 钢珠/风电缺陷全流程 | 多场景闭环思路可借鉴 |

### 2.2 无人机空中检测系统：仿真 / 边缘部署
| Stars | 仓库 | 一句话定位 | 与本项目异同 |
|---|---|---|---|
| 402 | [monemati/PX4-ROS2-Gazebo-YOLOv8](https://github.com/monemati/PX4-ROS2-Gazebo-YOLOv8) | PX4 飞控 + ROS2 + Gazebo 仿真的完整空中检测系统 | 工程最完整；用 GPS/IMU 而非北斗 |
| 91 | [alebal123bal/khadas_yolov8n_multithread](https://github.com/alebal123bal/khadas_yolov8n_multithread) | RK3588 NPU 上 YOLOv8n 跑 46 FPS / ~140MB | 边缘部署标杆；本项目未做 NPU 量化 |
| 38 | [anhuipl2010/SOD-YOLO](https://github.com/anhuipl2010/SOD-YOLO) | 在 YOLOv8 上做小目标专用增强 | 改进点与本项目的 EMA/DySample/Slim-Neck 思路重叠最多 |
| 17 | [MartinMohammed/object-tracking-yolo-v8-mil](https://github.com/MartinMohammed/object-tracking-yolo-v8-mil) | YOLOv8 + MIL 跟踪器做无人机目标跟踪 | 多目标跟踪视角，可补本项目后处理 |

### 2.3 小目标检测通用方法（YOLOv8）
| Stars | 仓库 | 一句话定位 | 与本项目异同 |
|---|---|---|---|
| 556 | [Koldim2001/YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) | SAHI 切片推理库，小目标检测最火工程方案 | 推理侧互补；本项目是训练侧结构改进 |
| 35 | [DmitriyKras/YOLOv8-for-small-objects](https://github.com/DmitriyKras/YOLOv8-for-small-objects) | 论文 "Improved YOLOv8 for Small Objects" 复现 | 学术基线 |
| 32 | [zhuty2001/YOLOv8n-SMALL-OBJECTS-DETECTION](https://github.com/zhuty2001/YOLOv8n-SMALL-OBJECTS-DETECTION) | 改 yolov8n 结构优化小目标 | 同方向不同模块组合 |

### 2.4 特定场景应用（应用层灵感）
| Stars | 仓库 | 一句话定位 | 与本项目异同 |
|---|---|---|---|
| 16 | [InvictusRex/Autonomous-Drone-MSAR](https://github.com/InvictusRex/Autonomous-Drone-MSAR) | 无人机海上搜救（GPS/IMU + Jetson/RPi 边缘） | 与本项目「救」的卖点最接近，但用 GPS 非北斗 |
| 27 | [kbhujbal/GeoIntel-satellite_military_asset_classification_CNN](https://github.com/kbhujbal/GeoIntel-satellite_military_asset_classification_CNN) | SAHI 切片处理 4000×4000 卫星图小目标 | 大图切片思路可借鉴 |

## 3. 对比分析：本项目的差异化定位

| 维度 | 典型开源项目 | 本项目 |
|---|---|---|
| 数据集 | VisDrone / 通用航拍 / 卫星 | VisDrone2019-DET（同 SOTA 基线） |
| 检测模型 | YOLOv5/8/10/11 各变体 | **改进 YOLOv8m**（EMA/DySample/DualConv/Slim-Neck/NWD/P2） |
| 小目标优化 | 部分有（SOD-YOLO、切片推理） | 有（结构改进 + NWD 损失） |
| 边缘/仿真部署 | monemati、alebal 等做得深 | 未做 NPU 量化（可后续扩展） |
| **定位与回传闭环** | **几乎都没有** | **北斗高精度定位 + 短报文回传坐标** |
| **通导授救一体化** | **无** | **有（项目核心叙事）** |

**护城河**：上述所有项目都停留在「检测 / 追踪 / 仿真 / 边缘部署」，**没有任何一家把检测结果经北斗短报文回传坐标、实现「通、导、授、救」一体化**。这是本项目区别于全行业同类工作的最关键差异点，也是技术报告应重点展开的部分。

## 4. 建议作为「相关工作」引用的基线
1. `monemati/PX4-ROS2-Gazebo-YOLOv8` — 空中检测系统的工程上限参照
2. `dronefreak/VisDrone-dataset-python-toolkit` — VisDrone 训练/评测事实标准工具箱
3. `anhuipl2010/SOD-YOLO` — 小目标 YOLO 改进的直接对标
4. `Koldim2001/YOLO-Patch-Based-Inference` — 切片推理（SAHI）互补方案

## 5. 一句话结论
本项目的价值不在「又训练了一个 VisDrone 检测器」，而在于**把改进检测模型接入北斗定位 / 短报文回传，形成无人机前端感知 + 北斗通导授救的闭环**——这在 GitHub 现有开源生态中尚无同类实现。
