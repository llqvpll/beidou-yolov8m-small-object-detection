# 北斗 · 改进 YOLOv8m 小目标识别辅助系统 · Web 平台

把「修正版代码」里的命令行推理封装成一个**可交互的网页前端**，方便盲审演示、答辩展示和日常使用：
上传图片/视频，实时看到检测框、类别、置信度，并带一个北斗定位面板（示意地理映射）。

> 设计要点：**双模式自适应**。
> - **真实模式**：检测到 `ultralytics` + 权重即自动加载改进模型（含 EMA/DySample/GSConv/VoVGSCSP），做真实推理；自动探测 CUDA，有卡用 GPU、无卡回落 CPU。
> - **演示模式**：环境缺 `ultralytics` 或权重缺失时，自动降级，页面照常打开；点「示例演示」可查看完整交互（演示数据，界面明确标注）。答辩现场无网/无卡也能演示 UI。

---

## 目录结构

```
修正版代码/
├── app 相关（原命令行脚本：train_v8.py / visdrone2yolo.py / 环境自检.py …）
├── install_custom_modules.py      # 把自定义模块注册进 ultralytics（必须先跑一次）
├── runs/train/smoke_v4/weights/best.pt   # 已有真实权重（v4 变体），可直接加载
└── web/
    ├── app.py            # Flask 后端（懒加载重依赖，真实/演示双模式）
    ├── start_web.py      # 启动器：注册模块 → 自检 → 启服务
    ├── requirements.txt  # Web 依赖（flask/opencv/pillow/numpy）
    ├── static/index.html # 零依赖单页前端（内联 CSS/JS，离线可用）
    ├── output/           # 视频推理结果落盘处（自动生成）
    └── README_web.md
```

---

## 快速开始（在你的本机 / 台式机）

### 1. 安装依赖
```bash
cd 修正版代码
pip install -r web/requirements.txt
pip install ultralytics torch   # torch 按机器选：RTX 50 系需 CUDA 12.8+ 版（sm_120）
```

### 2. 注册自定义模块（关键，只需一次）
```bash
python install_custom_modules.py
```
> 这会把 `EMAAttention / DySample / DualConv / GSConv / VoVGSCSP` 写入 ultralytics 源码。
> 换机器后必须重跑（补丁写进的是那台机的 ultralytics）。

### 3. 启动
```bash
cd web
python start_web.py
# 或：python app.py
```
启动后打开浏览器：**http://localhost:5000**

---

## 功能

- **三种输入源**
  - 图片：点击/拖拽上传，返回原图 + 标注结果；右侧统计各类数量、检测列表、示意经纬度。
  - 视频：上传 mp4（建议 < 50MB），服务端逐帧推理并输出带框视频 + 统计。
  - 摄像头：浏览器调起摄像头，定时截帧推理（实时模式依赖 GPU，CPU 下帧率较低）。
- **置信度滑块实时过滤**：后端一次性返回全部候选框（低阈值），前端滑块即时过滤，**不重新推理**，拖拽零延迟。
- **北斗定位面板**：显示经纬度/海拔/卫星数；检测结果按像素偏移示意映射到地理坐标（演示估算，部署时可接入真实北斗读数）。
- **真实/演示双模式**：徽章实时显示当前模式、设备（GPU/CPU）、模型变体。

---

## 替换为你自己正式训练的权重

后端按以下优先级自动选择权重（第一个存在的即用）：

1. 环境变量 `WEIGHTS` 指定的路径
2. `runs/train/visdrone_v4/weights/best.pt`  ← **正式训练产物放这里即自动启用**
3. `runs/train/smoke_v4/weights/best.pt`    ← 当前已有（v4 冒烟权重，仅 3 epoch + 小数据，效果有限，用作跑通验证）
4. `yolov8m.pt`（官方 COCO 预训练，80 类，非 VisDrone 类别，仅兜底）

> 盲审要求「论文 = 代码 = 权重」自洽：把正式训练好的 `best.pt` 放到 `visdrone_v4/weights/` 下即可，无需改任何代码。

---

## 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `WEIGHTS` | 强制指定权重路径 | 自动探测 |
| `PORT` | Web 端口 | 5000 |
| `BEIDOU_LAT` / `BEIDOU_LON` / `BEIDOU_ALT` | 北斗坐标（演示） | 28.169 / 112.944 / 50.0（长沙·岳麓） |
| `BEIDOU_SATS` | 卫星数（展示） | 12 |
| `BEIDOU_REAL` | 设为 `1` 标注为真实坐标 | 0（演示） |

示例：`BEIDOU_REAL=1 BEIDOU_LAT=28.17 BEIDOU_LON=112.94 PORT=8080 python app.py`

---

## 已知限制 & 后续扩展（按性价比）

1. **SAHI 切片推理**：VisDrone 小目标利器，不改模型。可把切片坐标与北斗定位绑定，做成「切片 → 地理坐标」的故事线。
2. **真实北斗接入**：当前坐标为示意映射；接入真实北斗模块读数后即可做像素→地理的标定（需相机内参）。
3. **TensorRT 导出**：`model.export(format='engine')` 解决 P2 分支的延迟问题，部署更快。
4. **NMS/标签分配处 NWD**：目前只在损失用了一处，论文原意三处皆可用，可做对比消融。
