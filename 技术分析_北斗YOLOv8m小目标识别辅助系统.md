# 北斗 · 改进 YOLOv8m 小目标识别辅助系统 —— 技术分析

> 分析对象：`C:\Users\Administrator\OneDrive\Desktop\新建文件夹\修正版代码\`
> 证据口径：全部结论来自仓库内源码、配置与 `报告数据.json`，不引用任何外部材料或历史旧稿数字。
> 代码规模：后端 Python 约 5236 行（不含测试）+ 测试 2866 行（13 个文件，当前 251 passed）+ 前端约 8650 行（HTML 1076 / CSS 1777 / JS 5796）。

---

## 1. 设计目标与约束

### 1.1 目标

系统面向**无人机航拍场景下的小目标识别**，把三件原本分离的事合到一条流水线上：

1. **看得见小目标**：VisDrone 图像中 `pedestrian`、`bicycle`、`motor` 等目标常只有十余像素，标准 YOLOv8m 的三尺度头（P3/P4/P5）在 P5（stride 32）上对这类目标几乎无响应。
2. **知道目标在哪**：把检测结果与北斗/GNSS 定位绑定，给出目标的地理坐标。
3. **辅助态势判断**：在检测结果之上做人群密度估计与分级预警。

### 1.2 三条硬约束

约束贯穿全栈，是理解本系统所有反常规设计的钥匙。

**（1）「没有数据」必须可表达。** 定位、密度、地理坐标，数据不足一律返回空值 + 说明原因，**绝不回退到写死默认值**。这条约束在代码中反复出现：

```python
# server/services/beidou.py —— geo_map()
if not fix.usable or fix.lat is None or fix.lon is None:
    return None, None          # 不是返回默认坐标
gsd, _calibrated, _estimated = self.gsd()
if gsd <= 0:
    return None, None
```

```python
# server/services/crowd.py —— analyze()
area = None
if area_m2 and area_m2 > 0:      area = float(area_m2)
elif meters_per_px and meters_per_px > 0:  area = w * h * (meters_per_px ** 2)
density = (total / area) if (area and area > 0) else None   # 未标定 -> None，不硬算
```

理由是明确的：**若「模块没插好」与「定位正常」在接口上长得一样，用户就会拿到"看起来正常的错误结果"。** 这一原则直接决定了接口契约的形状（字段恒存在、值为 `None`），而不是靠"字段缺失"来表达。

**（2）任何单点失败不得让服务不可用。** 模型缺失、ultralytics 未安装、北斗模块未插、编码器不可用，都要降级而非崩溃（详见 §3.3、§5.4）。

**（3）工程产物与学术材料同源。** 页面上展示的曲线、KPI、配图必须来自同一次测量，不允许拼接（详见 §5.10 的"配图优先重绘图目录"机制）。

---

## 2. 整体架构

### 2.1 分层

```
┌──────────────────────────────────────────────────────────────┐
│ 前端 SPA（零依赖原生 JS）                                      │
│   index.html + assets/{css/app.css, js/core.js, visual.js,    │
│   app.js}  ——  5 个视图：检测/任务/分析/北斗/设置               │
└─────────────────────────── 同源 HTTP ─────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ 路由层  server/routers/  （8 个 router，统一 /api/v1 前缀）     │
│  health │ meta │ detect │ video │ jobs │ files │ metrics │track│
├──────────────────────────────────────────────────────────────┤
│ 契约层  server/schemas/  pydantic v2 模型                      │
│ 横切层  server/core/     deps(注入) errors(统一信封) security  │
├──────────────────────────────────────────────────────────────┤
│ 服务层  server/services/                                      │
│  model_registry │ inference │ jobs │ crowd │ metrics │        │
│  beidou → gnss → nmea（实时） │ track_store → gnss_track（离线）│
├──────────────────────────────────────────────────────────────┤
│ 配置层  server/config.py  pydantic-settings，APP_ 前缀注入      │
├──────────────────────────────────────────────────────────────┤
│ 模型层  custom_modules/（EMAT/DySample/DualConv/GSConv）        │
│         custom_loss.py（NWD）  +  ultralytics YOLOv8m          │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 部署形态

**单进程、单端口、前后端同源。** `main.py` 用 app 工厂构建 FastAPI，`lifespan` 内完成三件事（模型预热、任务库加载、定位源启动），最后把前端 `StaticFiles` 挂在 `/`：

```python
# main.py —— 必须最后挂载："/" 会兜住所有未匹配路径
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
```

带来的直接收益是前端**无需 CORS、无需另起静态服务器**，双击 `启动系统.bat` 即可用。代价是：静态挂载会吞掉未匹配路径（前端 `Api.probe()` 专门处理了"404/405 才试下一候选"的探测逻辑）。

同时提供 `Dockerfile`（python:3.10-slim，默认 CPU 版 torch，需换 CUDA 版则改一行）与 `docker-compose.yml`，但主交付形态是 Windows 本地进程。

### 2.3 请求流（两条典型路径）

**图像检测（同步）**：`POST /detect` → `InferenceService.detect()` → `asyncio.Semaphore(1)` → `asyncio.to_thread(_detect_sync)` → `model.predict()` → 结果组装（含北斗帧 + 人群密度）→ JSON（`annotated` 为 base64 JPEG）。

**视频检测（异步）**：`POST /video` 立即返回 `202 + job_id` → 后台 `asyncio.Task` → `to_thread(predict_video)` 逐帧推理写盘 → `GET /jobs/{id}` 轮询进度 → `GET /files/{id}/result_*.mp4` 取结果。

---

## 3. 核心模块与职责

| 模块 | 文件 | 行数 | 职责 |
|---|---|---|---|
| 应用入口 | `server/main.py` | 128 | app 工厂、lifespan、CORS + 关联 ID 中间件、路由与静态资源装配 |
| 配置 | `server/config.py` | 232 | pydantic-settings 强类型配置；权重候选表、GSD 换算、切片/人群阈值等派生属性 |
| 依赖注入 | `server/core/deps.py` | 72 | 模块级懒单例（registry / beidou / inference / store / metrics / track） |
| 异常体系 | `server/core/errors.py` | 137 | 统一错误信封 + 4 类全局异常处理器 |
| 模型注册表 | `services/model_registry.py` | 245 | 单例加载/缓存/切换权重，CUDA 探测，双模式降级 |
| 推理服务 | `services/inference.py` | 686 | 图像/切片/视频推理、NMS、编码器探测与回读校验 |
| 任务管理 | `services/jobs.py` | 132 | 内存 + JSON 持久化的任务库与后台协程 |
| 人群密度 | `services/crowd.py` | 197 | 人数/密度/分级/逐帧序列抽稀 |
| 北斗服务 | `services/beidou.py` | 114 | 像素→地理坐标换算（GSD 唯一来源） |
| 定位源 | `services/gnss.py` | 576 | 串口/回放/模拟源的选取、读线程、过期判定 |
| NMEA 解析 | `services/nmea.py` | 443 | GGA/RMC/GSA/GSV 解析、校验和、卫星明细 |
| 离线轨迹 | `services/gnss_track.py` | 629 | 存储卡日志导入、时间索引、二分插值查询、对齐、诊断 |
| 轨迹服务 | `services/track_store.py` | 167 | 单份轨迹的进程内持有与抽稀缓存 |
| 训练指标 | `services/metrics.py` | 420 | **只读**解析 results.csv / args.yaml / 对比表 / 配图 |
| 自定义模块 | `custom_modules/*.py` | 468 | EMA-lite / DySample / DualConv / GSConv，及注入 ultralytics 的安装器 |
| 损失函数 | `custom_loss.py` | 114 | NWD 损失 + 去 in-place 的 DFL + 除零保护 |

---

## 4. 技术选型理由

| 选型 | 理由 | 被否决的替代方案及原因 |
|---|---|---|
| **FastAPI + pydantic v2** | 自带 OpenAPI、依赖注入、类型即校验；`asyncio.to_thread` 能把同步 torch 推理包成非阻塞 | Flask：无原生类型校验与异步，需手工拼装；Django：过重 |
| **pydantic-settings** | 配置强类型 + `APP_` 前缀 + `.env` 加载，启动即 fail-fast；派生属性（候选表、GSD、体积上限）集中在一处 | 散落的 `os.getenv`：无类型、无默认值约束，历史上导致过前后端常数不一致 |
| **ultralytics YOLOv8m** | 生态成熟、yaml 可声明式改结构、训练/验证/导出一体；VisDrone 上有公开基线可对照 | 自研检测框架：训练与验证口径需自建，成本高且无法与公开基线对齐 |
| **零依赖原生前端** | 交付只需一个 bat；无构建链、无 npm，避免"装环境"失败 | Vue/React + Vite：需要构建产物与额外部署步骤，与"双击即用"冲突 |
| **内置切片推理** | 切片是小目标检测最直接的增益手段，且**不改模型**；自带实现零依赖 | 强依赖 sahi：多一个重型依赖；实现为可选回退，两全 |
| **JSON 任务库** | 单机部署下够用，无 Redis/DB 依赖 | Celery + Redis：引入 Broker 与 worker 进程，与单机定位不符（见 §7 局限） |
| **OpenCV VideoWriter** | 与 cv2 解码同库，避免额外转码链路 | ffmpeg 子进程：需打包二进制；imageio：编码器可控性差 |
| **NMEA-0183 文本协议** | 北斗/GNSS 模块通用输出，串口文本最易调试与回放 | 二进制协议（UBX/RTCM）：解析复杂、无法直接回放与人工检查 |

---

## 5. 关键实现思路与算法

### 5.1 模型侧改动（v4 为主方案）

v4 的网络定义见 `yolov8m_visdrone_slimneck_nop5.yaml`，叠加两项改动：

**（a）Slim-Neck 轻量化**：Neck 中 `C2f → VoVGSCSP`、下采样 `Conv → GSConv`。

`GSConv` 是"半标准卷积 + 半深度可分离卷积 + channel shuffle"：

```python
x1 = self.cv1(x)                       # 标准卷积支路 -> c_ = c2//2
x2 = torch.cat((x1, self.cv2(x1)), 1)  # 深度可分离支路（k=5）拼接
y = x2.reshape(b, 2, n//2, h, w).permute(1,0,2,3,4)   # shuffle
return torch.cat((y[0], y[1]), 1)
```

`VoVGSCSP` 作为 `C2f` 的轻量替代，保留双路 concat 聚合结构，可在 yaml 中同位置替换，消融改动干净。

**（b）剪枝 P5**：去掉 stride=32 分支，检测头为 `[[20, 23, 26], 1, Detect, [nc]]`，即 P2/P3/P4 三尺度，用"剪掉的大尺度分支"抵消"新增 P2 分支"的延迟。

其余组件：
- **P2 四尺度头**（v1 主方案）：增加 stride=4 分支，直接提升小目标召回。
- **EMA-lite 注意力**：沿 H/W 双方向全局池化 → 1×1 融合（BN+SiLU）→ 3×3 分组卷积 → Sigmoid 门控回乘。模块 docstring 明确要求论文中**如实命名为"注意力引导模块 / EMA-lite"**，因其是坐标注意力的轻量化变体，而非原论文 EMA 的三支路完整结构。
- **DySample 上采样**：学习采样偏移 + `grid_sample`。两处关键修正：原实现的 `interpolate(x) * sigmoid(mask)` 门控恒 <1 只能衰减激活；且 1×1 卷积放在 2× 分辨率后 FLOPs 是低分辨率的 4 倍。现实现把 pointwise 放回低分辨率，理论上省约 75% 计算量，偏移分支零初始化（初始等价于标准半像素对齐上采样）。
- **DualConv**：`3×3 分组卷积 ∥ 1×1 点卷积` 并联相加 + channel shuffle + BN + SiLU。参数量约 `3.25·in·out`（g=4），约为标准 3×3 卷积的 36%。

### 5.2 自定义模块如何注入 ultralytics（关键工程细节）

ultralytics 的 `parse_model()` 对**未注册**模块不做通道推导：

```python
if m in base_modules:            # 内置模块：自动补 c1/c2 并按 width 缩放
    c1, c2 = ch[f], args[0]
    c2 = make_divisible(min(c2, max_channels) * width, 8)
    args = [c1, c2, *args[1:]]
else:                            # 未知模块：args 原样传入，拿不到正确 c1
    c2 = ch[f]
```

因此 `custom_modules/installer.py` 用**源码补丁**方式（幂等、带 `.bak` 备份、可 `is_installed()` 检查）完成四步：复制模块文件 → 在 `modules/__init__.py` 追加导入并扩展 `__all__` → 把类名插入 `tasks.py` 的 `base_modules`（以及 `repeat_modules`，供 `VoVGSCSP` 接收 yaml 的 repeats）→ 末尾追加导入使类进入 `parse_model` 的 `globals()` 查找空间。

补丁用正则定位 `frozenset({`，并显式兼容两种写法（`base_modules = frozenset({` 与 `repeat_modules = frozenset(  # 注释\n {`）。定位失败时会抛错并给出手工修改指引，而不是静默跳过。

NWD 损失**不需要改源码**：`v8DetectionLoss.__init__` 在调用时才从模块全局解析 `BboxLoss`，因此 `loss_mod.BboxLoss = BboxLoss` 直接替换即可生效。

### 5.3 图像推理管线

```python
async with self._sem:                          # Semaphore(INFERENCE_CONCURRENCY)，默认 1
    result = await asyncio.to_thread(self._detect_sync, ...)   # 同步 torch 推理不阻塞事件循环
```

- **切片推理**（零依赖）：按 `SLICE_SIZE=640` / `SLICE_OVERLAP=0.2` 生成重叠网格，边缘不足处零填充对齐，逐 tile `predict` 后把坐标平移回原图，最后**按类 NMS**合并（`_nms()`，纯 numpy，`order = conf.argsort()[::-1]` 贪心抑制）。
- **三级回退**：`sahi`（可选）→ 内置切片 → 标准推理，每级失败仅告警不抛出。
- **定位帧只取一次**：一张图可能几百个目标，逐个取定位帧会重复加锁重建卫星列表，`_to_detections()` 中先取 `fix = self.beidou.info()` 再复用。

### 5.4 视频链路：自管读写 + 编码器实证

这一段集中体现了"不轻信框架"的工程取向。两个踩过的坑写入注释：

1. 把 `save_dir` 交给 ultralytics 时，目录已存在会**自增**成 `<job_id>-2/`，`glob` 会抓回上传的原片当结果（表现为"结果视频与输入字节数一模一样"）；因此读写路径全部自控，结果文件名固定 `result_<job>.<ext>`，**绝不与上传件 `input.<ext>` 同名**。
2. `VideoWriter.isOpened()` 返回 True 不代表真能写（OpenH264 缺失时会写出 1KB 坏文件）。因此：

```python
def _probe_codec(fourcc, suffix) -> bool:   # 试写 3 帧合成视频并回读帧数 > 0 才算可用
def _count_frames(path) -> int:             # 回读真实可解码帧数，唯一可信判据
def _read_fourcc(path) -> str:              # 回读实际 fourcc（OpenCV 会静默替换未知编码器）
```

`_codec_order()` 只保留探测通过者，并**忽略不在候选表中的配置值**（否则会对外汇报一个根本没生效的编码器名）。

Windows 中文路径问题单独处理：`avc1` 走 libopenh264 分支打不开含非 ASCII 的路径，因此目标路径非 ASCII 时先写 `_ascii_tmp_dir()`（纯 ASCII，含用户名是中文的兜底逻辑），校验通过后再 `shutil.move` 到最终位置。文件名也必须 ASCII（OpenCV 按 ANSI 处理路径，中文文件名会乱码落盘，导致 `Path.exists()` 为 False 而 cv2 自己却能读到）。

### 5.5 北斗实时链路

**源选择**（`gnss.py` `_pick_source()`）：`auto` 先试串口、再试回放文件，都没有则 `unavailable`（**不静默编假坐标**）；`mock` 需显式指定，界面标注"模拟"。

**串口自动扫描**：`list_ports.comports()` 对 `description+manufacturer+device` 统计关键词命中（gnss/gps/beidou/bds/ch340/ch341/cp210/ftdi/uart/usb-serial），按 `(-score, device)` 排序；**多串口且都不像 GNSS 时返回 None**，不做"随便挑一个"的猜测。

**线程模型**：`daemon=True` 的 `gnss-reader` 线程持续 `readline()`，`_lock` 保护共享状态；`snapshot()` 同样持锁。源报错即退出循环（拔线容错）。

**过期判定**：`age > GNSS_STALE_S(5s)` 判为 stale，`> GNSS_MAX_AGE_S(30s)` 判为 dead —— 注释写明"不把最后一次坐标一直挂在那儿骗人"。`usable = fq in {1,2,3,4,5} and lat/lon 非空`（排除航位推算/人工/模拟）；`real` 仅在 `serial/file` 源且 usable 时为真。

**降级提示分级**：unavailable / 读到 0 行 / 0 条有效语句（提示波特率不匹配）/ fq=0（收星中）/ 超 max_age / 兜底文案。

**热插拔**：`POST /beidou/reload` 重建 `GnssService` 重扫设备，无需重启（注意配置是进程启动时的快照，改 `.env` 仍需重启）。

### 5.6 NMEA 解析

支持 GGA/RMC/GSA/GSV（不支持 VTG），talker 与语句类型按地址位切分（`addr[:2]` / `addr[2:5]`），不区分星座。校验和逐字符 XOR 验证，失败即丢弃（无 `*` 的语句接受但计数）。GSV 多句按 talker 分桶拼接，卫星从字段下标 3 起每 4 个一组，`snr<=0` 记 None。度分格式 `ddmm.mmmm` 按 `deg = int(f//100)` 换算，不靠字符串长度猜测；两位年按 `yy<70 → 2000+yy` 处理世纪折点。空字段一律返回 `None` 而非 0。

### 5.7 存储卡日志离线链路（真实主线）

用户的实际路线是"模块接单片机 → MCU 把 NMEA + 时间写进存储卡 → 事后把视频与日志一起导入"，因此离线链路是主线，实时串口只是便利功能。

- **时间戳格式自动识别**：`iso`（绝对）/ `unix`(≥1e9) / `unix_ms`(≥1e12) / `hms`（当天时刻，非绝对）/ `ticks`（上电毫秒，非绝对）/ `none`。分隔符支持 `,;\t|` 或无分隔行首。
- **收点时机**：在 `feed()` **之前**遇到 GGA/RMC 第二次出现即收点，防止"提前带上秒坐标"。
- **时间索引**：有缺失时间戳时用相邻差中位数还原节奏（≥100 判毫秒），最后排序；连读数都没有才退化为"每点 1 秒"并如实标记 `absolute=False`。
- **查询**：`at()` 边界钳位 + 二分查找（`while lo+1<hi`），对连续量（经纬度/高度/速度/HDOP…）线性插值并保留 7 位小数，离散量（定位质量/卫星数）取更近端，结果标 `interpolated=True`。
- **对齐**：`align(origin_s)` 设定"视频第 0 秒对应的日志时刻"，`align_by_delta(delta_s)` 整体平移；**未对齐时 `at_video_time()` 退化为"视频 0 秒 = 日志起点"并在结果里标 `aligned=False`**，界面据此提示，而不是假装已对齐。三入参 `utc > origin_s > delta_s` 优先级取用并回传 `used`（"静默忽略用户输入是更糟的选择"）。
- **抽稀与内联**：`scrub_points()` 等间隔取样且末点必留（默认 12000 点），卫星列表用 `json.dumps(sort_keys=True)` 去重内联成 `sats` 表，行内存下标，避免几万个点塞进 JSON。
- **诊断回执**：`diagnose()` 返回 stamp_kind 及分布、是否绝对时间、历元间隔、时钟偏移（日志时间戳与语句 UTC 之差的中位数，用于发现 MCU 时区偏 8 小时）、跳过行数、样例行与 hints。格式对不上时用户能立刻看到差在哪。

### 5.8 像素 → 地理坐标

换算基准只有**一个**参数：GSD（米/像素），来自 `Settings.geo_gsd`。注释记录了历史教训：后端曾用 `0.00001*(w/640)` 度/像素（≈2.94 m/px @1920），前端 ENU 散布写死 0.021 m/px，**同一张图上相差约 140 倍**，表格里"经度偏移"和"东(m)"两列互相矛盾。现在两侧共用同一取值。

```python
d = self.enu(cx, cy, w, h)                       # 画面中心为原点的东-北坐标（米）
dlat = d["north_m"] / 111320.0
cos_lat = math.cos(math.radians(fix.lat))
if cos_lat < 1e-6: return None, None             # 极区经度放大到无意义，直接不给
dlon = d["east_m"] / (111320.0 * cos_lat)
return round(fix.lon + dlon, 7), round(fix.lat + dlat, 7)   # 7 位 ≈ 1cm
```

未标定且 `GEO_ALLOW_ESTIMATE=True`（默认）时使用 `GEO_ESTIMATE_GSD=0.021` 并标 `estimated=true`，前端同时标注"估算值"。

### 5.9 人群密度与聚集预警

- **计数口径**：`pedestrian`(0) + `people`(1) 都算人；`people` 是"密集人群"标注（一框代表多人），按 `people_weight`（默认 1.0）折算 —— **保守低估，不夸大**；两类原始计数分别上报，便于自行调整口径。
- **密度必须有标定**：无 `area_m2` / `meters_per_px` 时只报人数与画面占比，不硬算密度。
- **尺度无关参考量** `occupancy_ratio`（人群框面积 / 画面面积），无需标定，但受拍摄高度影响，不能当绝对密度。
- **分级**（人/m²）参考 Fruin 服务水平与踩踏事故复盘：`<1 正常 | 1~2 关注 | 2~4 警戒 | >4 危险`，仅"警戒"及以上算预警事件。
- **模块自身承认的局限**（写在 docstring 里）：**遮挡悖论** —— 人群越密遮挡越严重、检测器越容易漏检，最需要准确的时候精度反而最低，高密度下计数会系统性偏低。因此定位为"中低密度的态势感知与相对趋势判断"，不宜作为精确人数计量依据。

### 5.10 训练指标只读服务

**只读**解析 `results.csv` / `args.yaml` / `对比_*.txt` / 配图，单例 + 惰性解析 + 缓存。字段缺失返回 `None` 而非抛异常（接口永不 500）。

一个值得称道的细节：**配图优先取 `figures/` 重绘图目录**。原因写在配置注释里 —— run 目录里的 `results.png` / PR 曲线 / 混淆矩阵是训练结束时画的，与页面 KPI 不同源（KPI 取 mAP50 峰值轮，而 `best.pt` 按 fitness `0.1*mAP50+0.9*mAP50-95` 选的是另一轮），同屏会出现 44.6% 与 44.2% 这种"看着像数据拼凑"的读数。重绘图由 MATLAB 从当前 run 的 `results.csv` 重绘，或用当前权重重跑验证生成，与页面同源。另有 `HIDDEN_ARTIFACTS` 显式排除清单（注释特别说明：只删 `FIGURE_CAPTIONS` 条目没用，因为不在表内的图会走 `extra` 分支照样列出）。

### 5.11 前端

`index.html`（1076 行骨架，内联 SVG 图标精灵 + 5 个 `<section class="view">`）+ 外链 `app.css`(1777) / `core.js`(1297) / `visual.js`(1242) / `app.js`(3257)，均 IIFE 挂在 `window.BDS` 命名空间，无构建。

- **视图切换**：`go(view)` 只做 class 切换，无 hash 路由；支持 `Alt+1…5` 与 `Ctrl+K` 命令面板。
- **状态**：`Store` 单一状态树 + 发布订阅，但 `app.js` 顶部直接 `const S = Store.state` 读写，重绘靠 `refreshView()` 手动触发（未用 subscribe）。
- **网络层**：原生 `fetch` + `AbortController` 超时（120s），`Api.request()` 统一返回 `{ok,status,data,requestId,ms,error,code}` 并回显 `x-request-id`；`probe()` 依 `/api/v1/status → /api/status → /status` 候选表探测，命中即锁定前缀。
- **置信度过滤在前端**：后端固定传 `conf=0.05`（低阈候选框一次性返回），滑块只做前端过滤、不重新推理。`params` 持久化在 `localStorage['bds.prefs.v1']`，滑块拉满会永久滤掉所有目标 —— `emptyReason()` 专门诊断该状态并提示恢复默认值。
- **渲染**：`Overlay`（Canvas 画框，zoom 限 0.04–24，stage <8px 时跳过 fit 防负缩放）、`Charts`（SVG DOM 构造）、`Beidou`（星空图）。动态值经 `U.esc()` 转义（74 处 innerHTML 均过转义，但仍属拼接式渲染）。缺值渲染"—"而非 0。
- **演示模式**：后端不可用时程序化生成航拍图与配套检测框（`demoSeed=20260913`），状态栏与顶部 banner 明确标注。

---

## 6. 性能与实测数据

### 6.1 模型对比（同协议复评：VisDrone val 548 张，imgsz 640，conf 0.001 / iou 0.6）

| 模型 | 参数量 | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|---|
| 标准 YOLOv8m-VisDrone | 25,862,110 | 78.7 | 0.42619 | 0.25182 | 0.5596 | 0.4332 |
| v1 四尺度（P2 + DySample/EMA-lite/DualConv） | 24,892,928 | 99.1 | **0.45469** | 0.25390 | 0.5713 | 0.4528 |
| v4 Slim-Neck + 剪 P5（三尺度） | **17,552,134** | 78.7 | 0.44235 | 0.24646 | 0.5566 | 0.4418 |

相对基线：

| 对比 | ΔmAP@0.5 | ΔmAP@0.5:0.95 | 参数量 | GFLOPs |
|---|---|---|---|---|
| v1 vs 标准 | **+2.85 pp** | +0.21 pp | −3.7% | +25.9% |
| v4 vs 标准 | **+1.62 pp** | −0.54 pp | **−32.1%** | 0% |
| v4 vs v1 | −1.23 pp | — | −29.5% | −20.6% |

训练实测（两个 run 请求 150 轮，分别于 118 / 117 轮早停；imgsz 640、batch 4、AdamW、lr0 5e-4、RTX 5060 Ti 8GB）：

| run | 峰值 mAP@0.5（轮次） | fitness 选中轮 mAP@0.5（轮次） | mAP@0.5:0.95 |
|---|---|---|---|
| visdrone_v1 | 0.45627（76） | 0.45560（68） | 0.24750 |
| visdrone_v4 | 0.44573（74） | 0.44383（66） | 0.24666 |

注意两列轮次不同（76 vs 68、74 vs 66）—— 这正是 §5.10 中"KPI 与 best.pt 不同源"问题的量化体现。

### 6.2 类别级增益（v1 vs 标准，AP@0.5，pp）

| 类别 | 标准 | v1 | Δ | v4 | Δ(v4 vs 标准) |
|---|---|---|---|---|---|
| pedestrian | 0.4598 | 0.5215 | **+6.17** | 0.5040 | +4.42 |
| people | 0.3543 | 0.4272 | **+7.29** | 0.4206 | +6.63 |
| motor | 0.4832 | 0.5430 | **+5.98** | 0.5271 | +4.39 |
| car | 0.7896 | 0.8303 | +4.07 | 0.8295 | +3.99 |
| bus | 0.6359 | 0.6050 | **−3.09** | 0.5975 | −3.84 |
| bicycle | 0.1858 | 0.2016 | +1.58 | 0.1665 | **−1.93** |

结论：增益集中在行人/人群/摩托等小目标与大目标的 `car`；代价在 `bus`，且剪 P5 后 `bicycle`、`truck`(−1.48 pp) 进一步回落 —— 说明 P5 分支对尺度偏大的目标仍有贡献，剪枝收益与代价并存。

### 6.3 吞吐与可扩展性

- **并发模型**：`INFERENCE_CONCURRENCY` 默认 1，用信号量把 GPU 推理串行化以避免 OOM；CPU 可调大。同步 torch 调用经 `asyncio.to_thread` 卸载，事件循环不被阻塞。
- **视频任务**：`asyncio.Task` + `to_thread`，长耗时与 HTTP 连接解耦，避免长连接超时。
- **实测缺口**：仓库内**没有端到端 FPS / 显存占用的归档结果**（`显存测速.py`、`显卡检查.py` 为临时脚本，无产物落盘）。当前只能以 GFLOPs 作为算力代理指标（v4 与标准持平 78.7 GFLOPs，参数少 32%），**不能直接给出实时性结论**。
- **扩展边界**：单进程、内存态任务库、本地磁盘输出 —— 天然是"单机单用户"形态（见 §7）。

---

## 7. 潜在风险与局限

按严重度排列，均附代码依据。

**P0-1 · 视频任务未纳入推理信号量，多任务并发会争抢 GPU。**
图像检测走 `async with self._sem`，但 `run_video_job()` 直接 `asyncio.to_thread(inference.predict_video, ...)`，**未过信号量**。同时提交两个视频任务即两路并发推理，在 8GB 显存上可能 OOM；且视频任务与图像检测之间也无互斥。

**P0-2 · 上传体积校验发生在整包读入内存之后。**
`data = await file.read()` 后才比较 `len(data) > max_video_bytes`。500MB 的视频会先完整驻留内存再被拒绝，多请求并发即可打爆内存。

**P0-3 · 进程重启后遗留"僵尸任务"。**
`JobStore` 是内存 dict + 全量重写 `jobs.json`；`asyncio.Task` 只存在于进程内。重启后状态为 `processing` 的任务永远停在 `processing`，前端轮询无终点。

**P0-4 · 模型热切换无并发保护。**
`switch_model()` 原地替换 `self.models["default"]`，没有锁、不感知在途推理/视频任务：切换瞬间显存峰值接近翻倍（旧模型未释放即加载新模型），并发请求可能拿到半初始化对象。失败回滚逻辑存在，但"回滚成功"不等于"在途推理安全"。

**P1-5 · 指标缓存不随模型切换失效。**
`MetricsService._cache` 常驻，而 `/metrics` 无 `force` 参数；`switch_model()` 后 `/metrics` 仍返回**切换前 run** 的训练指标，直到进程重启。这与 §5.10 "同源"目标直接冲突，是静默错误。

**P1-6 · 安全默认值为开发态。**
`ENABLE_AUTH=False`、`CORS_ORIGINS=["*"]`；鉴权通过 `require_api_key` 依赖可选挂载。直接暴露到非可信网络即有未授权访问与视频/日志上传风险。

**P1-7 · 安装器改写 site-packages 源码。**
`installer.py` 修改 ultralytics 的 `tasks.py` 与 `modules/__init__.py`。升级/重装 ultralytics 后补丁丢失（需重跑）；`is_installed()` 仅检测类名字符串是否存在，属于脆弱判据；多环境（venv/系统 Python）间容易装错目标。

**P1-8 · GSD 未标定时默认给出"估算"坐标。**
`GEO_ALLOW_ESTIMATE=True` + `GEO_ESTIMATE_GSD=0.021` 会产出带 `estimated=true` 的坐标。机制上诚实，但默认值是"开"，部署者若不标定就直接拿到估算经纬度，材料引用时有误读风险。

**P1-9 · 单项模块增益未做消融。**
P2 分支、EMA-lite、DySample、DualConv、NWD 各自的贡献**没有单独消融数据**；NWD 的超参 `C=12` 亦未扫描（代码注释建议扫 `C ∈ {6,8,12,16,20}`）。因此"+2.85 pp"只能归因于整体改动，不能拆解。

**P1-10 · 切片推理逐 tile 串行，无 batch。**
每个 tile 单独 `predict()`，未组 batch；4K 图像在 640/0.2 重叠下 tile 数量可观，延迟线性增长。且切片的端到端增益同样**未做独立评测**。

**P2-11 · 前端 `app.js` 单文件 3257 行。** 承载视图路由 + 5 个视图逻辑 + 快捷键 + 命令面板，无模块边界、无测试；`Store` 的 subscribe 机制形同虚设（重绘靠手工调用）。

**P2-12 · 任务轮询无退避、无上限。** `Jobs.poll()` 递归 `setTimeout(…, 1500)`，仅在 completed/failed 停止；切换任务不清旧 timer，长时间失败任务会持续轮询。

**P2-13 · 输出目录无清理策略。** `OUTPUT_DIR/<job_id>/` 保留上传原片与结果视频，`tracks/` 保留导入日志（单份上限 200MB），磁盘只增不减。

**P2-14 · 视频/日志时间对齐依赖人工。** 无自动首帧 UTC 元数据提取，需人工 `align`（utc / origin_s / delta_s 三选一），对齐错误的代价是定位整体偏移且不易察觉（`aligned=False` 仅在未对齐时提示）。

**P2-15 · 领域适用性边界。** 模型只认 VisDrone 航拍视角：屏幕录制、室内监控等非航拍素材必然检出 0。这不是缺陷但常被误判为"标注坏了"或"模型坏了"。

---

## 8. 优化与改进建议

### P0（建议在下一迭代内处理）

1. **统一 GPU 准入**：把 `predict_video` 也纳入 `InferenceService._sem`（或在 `JobStore` 层加一个跨类型的 GPU 队列），使"图像 + 视频 + 多视频"共享同一并发额度。更进一步可做成显式任务队列（单 worker），彻底消除竞争。
2. **上传体积前置校验**：先检查 `Content-Length`（`UploadFile.headers` / `request.headers`）超限即 413；再改为分块写入临时文件（`await file.seek` + 流式 `write`），避免整包驻留内存。
3. **任务库持久化加固**：启动时 reconcile，把孤儿 `processing` 标为 `failed`（原因：进程重启）；写入改为原子替换（`tmp + os.replace`）；中期可换 SQLite（单机零依赖、支持并发与查询）。
4. **模型切换并发保护**：加 `asyncio.Lock`，切换前等待/拒绝在途推理（`_sem` 全部 acquire 后再换），切换成功后**主动失效 `MetricsService._cache`**（连带修 P1-5）。

### P1（中期）

5. **导出与加速**：导出 ONNX / TensorRT（FP16），把 v4 的 78.7 GFLOPs 与实际帧率挂钩；切片推理 tile 组 batch 一次 `predict`。同时**补齐实测**：固定素材 + 固定分辨率下的端到端 FPS 与显存占用，归档进 `报告数据.json`，取代 GFLOPs 代理指标。
6. **补消融实验**：P2 分支 / EMA-lite / DySample / DualConv / NWD 逐项开关，NWD 的 `C` 超参扫描；切片推理独立评测（开/关 × 分辨率）。这是把"整体 +2.85 pp"拆解为可归因结论的唯一途径，也是论文口径能否站住的关键。
7. **生产安全基线**：默认 `ENABLE_AUTH=True` 或至少在 `.env.example` 中显著提示；`CORS_ORIGINS` 显式白名单；上传文件类型白名单（当前 `track/import` 只提示未限制）；输出目录 TTL 清理任务。
8. **安装器健壮性**：把"改源码"降级为优先方案——可改用 ultralytics 支持的 `custom_modules` 注册入口（若版本支持），或在启动时校验补丁版本指纹（记录 ultralytics 版本号，不匹配即报错重装），并保留一键卸载（从 `.bak` 还原）。
9. **前端工程化**：按视图拆 ES Module + Vite 构建（保留同源挂载方式），引入 Vitest 覆盖 `Store` / `Api` / 视图渲染；轮询加指数退避与上限，切换任务清理 timer。

### P2（长期 / 研究向）

10. **GSD 标定流程化**：支持从 EXIF/飞控高度 + 像元角，或地面控制点反算 GSD；未标定时建议默认 `GEO_ALLOW_ESTIMATE=False`（由部署方显式开启），把"估算"变成主动选择。
11. **人群计数升级**：接入 ByteTrack 等多目标跟踪做跨帧去重，缓解遮挡悖论下的漏检；高密度场景改用密度回归（密度图）而非框计数，并给出置信区间。同时研究 `people_weight` 的场景化标定方法。
12. **时间同步自动化**：视频首帧写入绝对 UTC（录制端加 GPS 授时或 NTP/PPS），后端自动完成对齐，减少人工 `align`；保留 `clock_offset_s` 诊断作为兜底。
13. **离线/在线一体化**：把实时串口链路也写入环形缓冲，与离线日志共用同一 `NmeaTrack` 索引，使"实时查看"与"事后回放"共用一套查询接口。

---

## 9. 结论

这是一套**工程完整度高于算法新颖度**的系统，其真正的价值集中在三处：

1. **诚实的失败语义。** 定位缺失返回 `None`、密度未标定不给密度、编码器不可用就降级并如实汇报实际 fourcc、未对齐就标 `aligned=False` —— 全栈贯穿"宁可不给，也不给看起来正常的错误结果"。这在一个"到处都是默认值"的领域里是稀缺品质。
2. **把踩过的坑写进代码。** 视频自增目录、中文路径编码器失效、`isOpened()` 假成功、GSD 双常数相差 140 倍、KPI 与 best.pt 不同源 —— 每个都有注释说明根因，而不是留一个神秘的补丁。
3. **模型改动可归因、数据同源。** v4 以 −32.1% 参数量取得 +1.62 pp，v1 以 +25.9% GFLOPs 取得 +2.85 pp，增益集中在行人与人群；页面数字全部来自 `报告数据.json` 与只读解析的训练产物。

主要短板同样清晰：**系统停留在单机单进程形态**（内存任务库、无 GPU 准入统一、无鉴权默认值），**算法侧缺少消融与实时性实测**（无法拆解模块贡献、无法给出 FPS 结论），**前端已到需要模块化的规模**。

若按 §8 的 P0 四项（GPU 准入、上传前置校验、任务库 reconcile、模型切换加锁 + 指标缓存失效）先行处理，系统的可靠性会有立竿见影的提升；P1 的消融实验与实测数据，则决定了这套改进在学术口径上能走多远。
