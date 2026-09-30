# 检测工作台前端 · 设计说明与接口契约

> 面向「北斗 · 改进 YOLOv8m 小目标识别辅助系统」的 Web 前端。
> 零运行时依赖：无 CDN、无构建步骤、可离线运行；图表 / 星图 / 画布标注全部为手写 SVG 与 Canvas。

```
server/static/
├── index.html              单页应用骨架（含图标精灵）
├── assets/css/app.css      设计系统（令牌 → 组件 → 视图）
├── assets/js/core.js       工具 / 状态 / 接口适配层 / 演示场景生成
├── assets/js/visual.js     画布标注渲染器 / SVG 图表 / 北斗星空图
└── assets/js/app.js        视图路由 / 交互 / 快捷键 / 各视图逻辑
```

---

## 一、如何运行

### 1. 独立预览（不需要后端）

```bash
cd server/static
python -m http.server 8777
# 打开 http://127.0.0.1:8777
```

页面会自动进入**离线演示模式**，并载入一幅程序化生成的航拍场景（含与之严格对应的检测结果），
所有交互、图表、北斗视图均可完整体验。也可以直接用浏览器打开 `index.html`（`file://`）。

### 2. 由后端挂载（推荐，**当前项目已启用**）

`server/main.py` 已把本目录挂在根路径 `/`（`StaticFiles(directory=..., html=True)`），
启动后端后直接打开 `http://localhost:8000/` 即可，**不需要另起静态服务器，也不涉及 CORS**。

关键点（踩过的坑）：

```python
# 1) mount 必须在所有 include_router 之后 —— "/" 会兜住一切未匹配路径，
#    挂早了会把 /api/v1/* 全部吃掉。
# 2) 服务信息路由不能再占用 "/"，否则页面被 JSON 抢先命中（返回 application/json）。
#    本项目已把它移到 "/info"。
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
```

若前后端分离部署，把本目录交给 Nginx/CDN，并在「设置 → 后端连接 → API 基址」填写后端地址即可。

---

## 二、后端接口契约

前端内置**适配层**，会按顺序探测并自动兼容两种后端形态，任一可用即切换为真实推理：

| 形态 | 前缀 | 状态 |
|---|---|---|
| 新版 FastAPI | `/api/v1` | 首选 |
| 旧版 Flask | `/api` | 兼容回退 |

探测顺序：`/api/v1/status` → `/api/status` → `/status`。

### 需要的端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/v1/status` | 系统状态（见下） |
| `GET` | `/api/v1/health` | 健康检查（可选） |
| `POST` | `/api/v1/detect` | 图像检测，**仅 multipart**（`file` 字段） |
| `POST` | `/api/v1/detect/json` | 图像检测，**仅 JSON**（`{image: dataURL}`） |
| `GET` | `/api/v1/metrics` | 训练指标（设置 → 训练成果，见 §四） |
| `GET` | `/api/v1/metrics/artifact/{name}` | 训练配图原图（后端直出） |
| `POST` | `/api/v1/videos` | 提交视频任务 |
| `GET` | `/api/v1/videos/{job_id}` | 查询视频任务 |
| `GET` | `/api/v1/videos` | 任务列表（可选） |

> ⚠ **multipart 与 JSON 是两个不同端点**，不要合并。把 JSON 发到 `/detect` 会得到
> `400 请上传 file 或提供 image(base64)`（该端点的 `file`/`image` 是 `Form(...)`）。
> 表现是「点了推理没反应、结果还是旧的」，很难排查。
> 前端已按输入类型分流：`File/Blob → /detect`，dataURL → `/detect/json`。

### `GET /api/v1/status` 期望返回

与 `server/schemas/detection.py::StatusResponse` 一致：

```json
{
  "mode": "real",
  "device": "0",
  "model": { "weights": ".../best.pt", "device": "0", "cuda_name": "RTX 4060",
             "classes": 10, "loaded": true },
  "classes": [ { "id": 0, "name_en": "pedestrian", "name_zh": "行人", "color": "#32CD32" } ],
  "beidou": { "lat": 28.169, "lon": 112.944, "alt": 50.0, "satellites": 12, "real": false },
  "engine_available": true,
  "slicing_available": true,
  "message": null
}
```

> 旧版 Flask 的扁平结构（`classes_zh` / `classes_en` / `class_colors`）同样被识别，无需改造旧后端。

### `POST /api/v1/detect` 期望返回

与 `DetectResponse` 一致：

```json
{
  "mode": "real", "width": 1600, "height": 1100,
  "detections": [
    { "class_id": 3, "name_en": "car", "name_zh": "小汽车", "conf": 0.91,
      "bbox": { "x1": 120.5, "y1": 430.0, "x2": 154.5, "y2": 448.0 },
      "lon": 112.9441, "lat": 28.1692 }
  ],
  "original": "data:image/jpeg;base64,...",
  "annotated": null,
  "device": "0", "engine": "standard",
  "beidou": { "lat": 28.169, "lon": 112.944, "alt": 50.0, "satellites": 12, "real": false },
  "elapsed_ms": 42.7
}
```

**关键约定**

- `conf` 后端按低阈值（建议 `0.05`）**一次性返回全部候选框**，前端用滑块实时过滤，**不重新推理**，拖拽零延迟。
- `original` 回传原图（data URL）时前端自行绘制标注，因此**不依赖** `annotated`，可省去服务端绘图开销。
- 前端的类别配色是 UI 层呈现，与 `classes[].color` 解耦（「设置 → 类别配色」可切换三套方案）。

### 错误信封

前端已适配统一错误结构，直接读 `error.message` 展示：

```json
{ "error": { "code": "MODEL_NOT_READY", "message": "推理引擎不可用：ultralytics 未安装", "details": null },
  "request_id": "a1b2c3" }
```

`request_id` 会显示在底部状态栏，便于与服务端日志串联。

### `POST /api/v1/videos` 两种返回形态均支持

- **异步（推荐）**：返回 `{ "job_id", "status", "status_url" }` → 前端轮询 `status_url`，在「任务」视图展示进度条与结果视频。
- **同步**：返回 `{ "video_url", "stats" }` → 前端直接展示结果。

---

## 三、设计系统

### 设计语言来源（均为 GitHub 高收藏项目）

| 参考项目 | 借鉴点 |
|---|---|
| **shadcn/ui** | 令牌驱动的中性色板、统一圆角阶梯、`focus-visible` 焦点环 |
| **Vercel Geist** | 单色精度、发丝级 1px 边框、等宽数字（`tabular-nums`） |
| **Linear** | 深色表面层次、命令面板、键盘优先、克制的 150–300ms 动效 |
| **Radix / Base UI** | 完整状态覆盖（hover / active / focus / disabled / aria） |
| **Tremor · Grafana** | 高密度数据磁贴、迷你趋势线、KPI 卡 |
| **supervision** | 画布标注范式：角标式检测框、标签芯片、置信度条 |
| **Tailwind CSS** | 4px 间距栅格纪律 |

### 关键取舍

- **不用大面积毛玻璃**。玻璃拟态只保留在命令面板、Toast 等浮层；主界面靠**层次 + 发丝线**建立结构，避免"炫技式"廉价感。
- **面板贴合、以 1px 线分区**（VS Code / Linear 式），而不是一堆悬浮卡片堆叠。
- **画布自绘**：检测框用"四角加粗 + 细边框 + 置信度条"，标签带指向三角，比整块实心矩形更专业、遮挡更少。
- **明暗双主题**：深色为默认（值守 / 暗光场景），浅色用于投影与打印；10 类配色在浅色下自动切换到更高对比度的色阶。
- **等宽数字**：所有计数、坐标、耗时使用 `font-variant-numeric: tabular-nums`，避免数字跳动。

### 设计令牌

```
间距   4 / 8 / 12 / 16 / 20 / 24 / 32 px
圆角   5 / 7 / 9 / 12 / 16 / full
动效   120ms(微) · 190ms(常规) · 300ms(入场)，ease-out
强调色 深色 #4D9EFF / 浅色 #1D6FE0   （北斗域辅色 #8B7CFF）
语义色 ok / warn / danger / info
```

---

## 四、功能一览

**检测台**

- 三种输入源：图片（批量）、视频（异步任务）、摄像头（实时逐帧）
- 拖拽上传、队列管理、胶片条快速切换
- 置信度 / IoU / 输入尺寸 / 最大目标数 / SAHI 切片推理
- 舞台：滚轮缩放、拖拽平移、双击聚焦目标、适应窗口、实际像素
- 显示开关：检测框 / 类别标签 / 序号 / 置信度条 / 目标遮罩 / 原图对比 / 参考栅格
- 结果面板：KPI 磁贴、类别分布（点击可隐藏类别）、可搜索可排序的检测列表、选中目标详情（含经纬度与本地 ENU 坐标）
- 导出：JSON / CSV / 合成标注 PNG

**任务** — 视频任务列表、进度条、统计、结果视频播放与下载

**分析** — 累计帧数 / 检出数 / 平均置信度 / P95 耗时；类别环形图、检出量条形图、耗时趋势线、置信度直方图、类别明细表（会话数据存于本机）

**北斗定位** — 经纬度 / 海拔 / 可见卫星数；卫星星空图（方位角-高度角极坐标，含 BDS/GPS/GAL/GLO 分系统着色与扫描动效）；目标 ENU 本地坐标散布图；目标定位明细表

**设置** — 主题、类别配色（柔和 / 鲜明 / 单色）、界面密度、动效开关；API 基址与连接测试；推理默认值；会话数据导出与清空

**设置 → 训练成果**（数据全部来自后端 `/api/v1/metrics`，前端**不写死任何指标**）

- KPI 磁贴：最佳 mAP50（含 best epoch）、mAP50-95（含 P / R）、参数量、训练轮数（含计划轮数）
- 收敛曲线：`mAP50 / mAP50-95 / precision / recall` 四序列折线，epoch 横轴，圆点标注 mAP50 峰值轮次
- 消融对比：从 `对比_*.txt` 解析出的变体表（当前 run 高亮 + 「当前」标签），下附结论与风险提示（⚠ 行标黄）
- 训练配图：`results.png` / PR·F1·P·R 曲线 / 混淆矩阵 / `labels.jpg` 缩略图，点击放大
- 来源备注：run 目录、权重路径、架构 yaml、imgsz / batch / optimizer / lr0

> 换了模型或重跑训练，这里的内容会自动跟着变 —— 因为全部读自权重所在 run 的训练产物。

**通用** — 命令面板（`Ctrl/⌘ K`，含「训练成果」直达）、Toast 通知、快捷键、离线演示模式

---

## 五、快捷键

| 分组 | 操作 | 键 |
|---|---|---|
| 全局 | 命令面板 | `Ctrl/⌘ K` |
| | 快捷键帮助 | `?` |
| | 切换主题 | `D` |
| | 切换视图 | `Alt 1` … `Alt 5` |
| 检测台 | 开始 / 重新推理 | `R` |
| | 切换输入源 | `1` `2` `3` |
| | 适应窗口 / 实际像素 | `F` / `Shift 1` |
| | 放大 / 缩小 | `+` / `-` |
| | 框 / 标签 / 序号 | `B` / `L` / `N` |
| | 置信度条 / 遮罩 / 栅格 | `C` / `M` / `G` |
| | 原图对比 | `V` |
| | 列表上下移动选中 | `↑` `↓` |
| 上传 | 打开文件选择 / 载入演示 | `O` / `Shift D` |

---

## 六、可定制点

| 想改什么 | 改哪里 |
|---|---|
| 类别名称 / 颜色 | `core.js` → `BDS.CLASSES`、`BDS.PALETTES` |
| 主题色 / 间距 / 圆角 | `app.css` → `:root` 令牌 |
| 后端路径 | `core.js` → `Api.PATHS` |
| 演示场景外观 | `core.js` → `BDS.Demo.build()` |
| 新增视图 | `index.html` 加 `<section class="view" id="view-x">`，`app.js` → `VIEWS` 数组 |
| 图表样式 | `visual.js` → `BDS.Charts`（`donut` / `hbar` / `sparkline` / `histogram` / **`curve`**）；北斗用 `BDS.Beidou`（`skyplot` / `scatter`） |
| 训练成果卡片 | `app.js` → `Train` 模块；字段来自后端 `/api/v1/metrics`，改后端即改前端 |

> `Charts.curve(host, series, opts)` 是多序列折线图，专为训练曲线写：
> 支持小数刻度（`fmt`）、指定值域（`yMin/yMax`）、epoch 横轴（`xStart/xTicks`）、
> 图例与 best 峰值标注。`sparkline` 只画单序列整数刻度，画 mAP（0.44 这种）会被四舍五入成 0。

---

## 七、已知边界

- 离线演示模式的检测框为前端预置数据，界面已明确标注；接入后端后自动切换为真实推理。
- 视频结果播放依赖浏览器对编码格式的支持，建议服务端输出 H.264 的 MP4。
- 北斗坐标为示意映射（图像中心 ↔ 基准坐标 + 像素偏移），接入真实相机内参与北斗读数后可做严格标定。

### 响应式行为

| 断点 | 变化 |
|---|---|
| ≤1420px | 左右栏宽度收窄（278 / 318） |
| ≤1240px | 左右栏进一步收窄（258 / 296），顶栏副标题隐藏 |
| ≤1080px | 右栏转为右侧抽屉（工具栏最右按钮开合）；北斗/任务视图改单列 |
| ≤900px | 左栏转为左侧抽屉（工具栏出现专属入口）；顶栏搜索框收为图标；KPI 磁贴改 2×2 |
| ≤700px | 隐藏左侧图标栏，主区变单列，设置项改单列 |

抽屉可用 `Esc` 依次关闭。跨断点缩放窗口时会自动复位抽屉状态，避免面板停在屏幕外。

窄屏下工具栏按钮仍然全部保留（实测 660px 宽不会溢出），但按钮较密，建议配合快捷键与命令面板使用。

---

## 八、开发工具（`dev/`）

纯本地开发辅助，不参与线上运行，`index.html` 不引用它们。

| 文件 | 用途 |
|---|---|
| `dev/_check.js` | 交叉校验 DOM id / 图标符号 / CSS 类的引用一致性，跑 `node dev/_check.js`。输出里的 `btn-job-go` / `btn-geo-go` 是空状态里动态生成的 id，属预期误报。 |
| `dev/_shot.js` | 无头浏览器截图 / 探针。复用系统已装的 Edge，以 CDP 直连，**不依赖 agent-browser**。 |
| `dev/_e.js` | hash 驱动的界面状态探针，供 `_shot.js` 注入。 |

### 截图用法

先让 Edge 以无头模式常驻（**必须用独立 `--user-data-dir` 并加 `--disable-extensions`**，
否则会加载用户浏览器里的暗色类扩展，导致截图配色被覆盖、与真实渲染不符）：

```bash
msedge.exe --headless=new --remote-debugging-port=9333 \
  --user-data-dir=<一个空目录> --disable-extensions --no-first-run about:blank
```

```bash
# node dev/_shot.js <url> <out.png> [w] [h] [scale] [waitMs] [evalFile]
node dev/_shot.js "http://127.0.0.1:8777/index.html#view=geo" shot.png 1600 1000 1 5000 dev/_e.js
```

`dev/_e.js` 支持的 hash 参数：

| 参数 | 作用 |
|---|---|
| `view` / `theme` / `zoom` / `palette` | 切视图 / 主题 / 缩放 / 打开命令面板并搜索 |
| `select` / `hidecls` / `params` / `mask` | 选中首个检测、隐藏类别、改参数、开显示开关 |
| `keys` / `about` / `toast` / `drawer=l\|r` | 弹窗与抽屉 |
| `mockjobs` / `jobsel=N` | 注入 5 条假任务（四种状态）/ 选中第 N 行 |
| `run` | 点「开始推理」 |
| `train` / `lb=<配图名>` | 滚到「训练成果」卡片 / 点开某张配图放大 |
| `state` / `trainstat` / `detstat` / `spytoast` | **回读**：全局状态 / 训练卡片渲染结果 / 当前帧推理结果 / 拦截全部 toast |
| `wait=<ms>` | 让探针返回 Promise（`_shot.js` 以 `awaitPromise` 等待），异步动作落定即返回 |

> **异步动作必须配 `wait`**。`run` / `lb` / `callrun` 都是异步的，直接回读会拿到旧值。
> `wait` 会轮询等待，动作完成或超时即返回。
> `callrun` 比 `run` 更可靠：它会**先轮询等演示场景生成完毕**再发起推理 ——
> 无头软件渲染下 `Demo.build()`（2× 超采样 + 模糊阴影）可能要十几秒，
> 不等就点按钮只会拿到「没有可推理的输入」。

**端到端真实推理验收**（一条命令跑通「前端→后端→GPU→回填」）：

```bash
node dev/_shot.js "http://127.0.0.1:8000/#view=detect&spytoast=1&callrun=1&wait=90000" \
  e2e.png 1600 1000 1 8000 dev/_e.js
# 期望：runDone.engine=standard, device=0, mode=real, n>0；CONSOLE: clean
```

两个坑：

1. **截图输出路径不要含中文**。项目目录名里有中文时 `fs.writeFileSync` 会失败，先写到纯 ASCII 路径。
2. **改了 CSS/JS 后务必在 URL 上加 `?t=<时间戳>`**。`_shot.js` 已开 `Network.setCacheDisabled`，但仍建议加参数以排除缓存干扰。
3. **`theme=light` 会经 `localStorage` 持久化到 profile**。下一次截图若不显式指定主题，会沿用上次的偏好 —— 不是 bug，但要心里有数。
4. **别用 `curl -o /dev/null`**（Git Bash 下会 `exit 23` 写失败），改用临时文件。

