# 北斗 · 改进 YOLOv8m 小目标识别辅助系统 — 推理后端（FastAPI）

> **给前端团队**：本文件是后端与前端之间的**唯一契约**。前端是一个独立任务，
> 只需消费下面定义的 `GET/POST /api/v1/...` 接口即可。后端已做到生产级：
> 分层架构、统一异常、关联 ID、双模式（真实/演示）优雅降级、异步视频任务、健康检查、OpenAPI 文档。

---

## 1. 快速开始

> **只想跑起来？** 双击项目根目录下的 **`启动系统.bat`** 即可：自动找代码目录、
> 挑一个装齐依赖的 Python、起后端、等健康检查通过再打开前端页面，全程无需命令行。
> 换端口用 `set APP_PORT=8080`；不想自动开浏览器设 `BDS_NO_BROWSER=1`。
>
> 它是**两个文件**：
> - `启动系统.bat` —— 纯 ASCII 的引导器，只负责「找到一个 python」，避免批处理
>   处理中文 / 路径结尾反斜杠时的各种静默坑；
> - `start_system.py` —— 真正的逻辑（找代码目录、选解释器、起后端、轮询、开浏览器）。
>   放 Python 里是因为批处理的决策点无法自动化验证，出过「目录扫到了却判成找不到」的 bug。

```bash
# 1) 安装依赖（推理框架 ultralytics 与「修正版代码」共用）
pip install -r server/requirements.txt
#    torch 需单独装：CPU 版见 requirements.txt 注释；RTX 50 系用 cu128 版

# 2) 启动（会自动把自定义模块注册进 ultralytics，再起 uvicorn）
python server/run.py
#    自定义端口： APP_PORT=8080 python server/run.py

# 3) 文档
#    Swagger UI : http://localhost:8000/docs
#    ReDoc      : http://localhost:8000/redoc
#    OpenAPI    : http://localhost:8000/openapi.json
```

**Docker：**

```bash
docker compose -f server/docker-compose.yml up --build
```

---

## 2. 架构（high-star 设计思路）

```
server/
├── main.py            # create_app() 工厂 + lifespan + 中间件 + 路由装配
├── run.py             # 启动器（先注册自定义模块，再起 uvicorn）
├── config.py          # 类型安全配置（pydantic-settings，APP_ 前缀）
├── logging_setup.py   # 控制台 / 可选 JSON 日志，注入 request_id
├── core/
│   ├── errors.py      # 统一异常体系 + 全局异常处理器（错误信封）
│   ├── security.py    # 可选 API Key 鉴权
│   └── deps.py        # 依赖注入（配置 / 单例服务）
├── services/          # 业务逻辑（不依赖 FastAPI）
│   ├── model_registry.py  # 模型注册表（类比 roboflow/inference ModelManager）
│   ├── inference.py       # 推理：标准 / 内置切片 / sahi / 视频
│   ├── beidou.py          # 北斗定位服务
│   ├── drawing.py         # 中文标签绘制
│   ├── image_io.py        # 图像编解码 / data URL
│   ├── classes.py         # VisDrone 10 类元信息（中英文 + 颜色）
│   └── jobs.py            # 视频任务库（落盘 + 进度）
├── schemas/           # Pydantic 契约（前端对接依据）
│   ├── common.py / detection.py / video.py
├── routers/           # 薄控制器
│   ├── health.py meta.py detect.py video.py jobs.py files.py
└── tests/             # pytest（demo 合约 + real 集成，详见 §9）
```

设计借鉴：[tiangolo/full-stack-fastapi-template](https://github.com/tiangolo/full-stack-fastapi-template)
（分层 + DI + 类型安全配置）与 [roboflow/inference](https://github.com/roboflow/inference)
（ModelManager 模型注册 + X-Request-ID + 标准 CV API + app 工厂）。

---

## 3. 双模式（前端务必据此切换 UI）

后端启动时自动探测环境，进入两种模式之一，接口契约**始终不变**：

| 模式 | 触发条件 | `/detect` 行为 | `/video` 行为 |
|------|----------|----------------|---------------|
| `real` | 装了 `ultralytics` + 自定义模块已注册 + 找到权重 | 返回真实检测框 | 异步跑视频推理 |
| `demo` | 缺少 `ultralytics` / 权重 / 强制 `APP_FORCE_DEMO=true` | `mode:"demo"`、`detections:[]` + 提示文案 | `503 MODEL_NOT_READY` |

**前端做法**：每个响应都带 `"mode"` 字段，`GET /api/v1/status` 也返回 `"mode"`。
UI 应以 `mode` 决定：真实模式展示框；演示模式展示占位/演示数据（接口永不崩）。

---

## 4. 接口清单（全部挂载在 `/api/v1` 下，除探针）

### 4.1 健康检查 / 元信息

| 方法 | 路径 | 说明 | 响应 |
|------|------|------|------|
| GET | `/` | **前端页面**（`server/static/index.html`，见 §8） | `text/html` |
| GET | `/info` | 服务信息 | `{project, version, docs, openapi, api_prefix, frontend, note}` |
| GET | `/healthz` | 存活探针（k8s liveness） | `{"status":"ok"}` |
| GET | `/health` | 健康检查（前端探测别名） | `{"status":"ok","service":"…"}` |
| GET | `/readyz` | 就绪探针（k8s readiness） | `{"status":"ok"}` |
| GET | `/api/v1/health` | `/health` 的带前缀别名（不进 OpenAPI） | `{"status":"ok"}` |
| GET | `/api/v1/status` | 运行状态总览 | `StatusResponse` |
| GET | `/api/v1/classes` | 检测类别（中英文+颜色） | `list[ClassInfo]` |
| GET | `/api/v1/beidou` | 北斗定位信息 | `BeidouInfo` |
| GET | `/api/v1/models` | 权重候选与加载情况 | `{mode, loaded, loaded_variant, device, error, candidates, candidates_exist}` |

> ⚠ **`/` 归前端**：服务信息已从 `/` 移到 `/info`。若把前端挂在根路径又保留 `/` 路由，
> 页面会被 JSON 路由抢先命中（返回 `application/json` 而非 `index.html`）。

**`StatusResponse`**
```json
{
  "mode": "real | demo",
  "device": "cpu | 0 | null",
  "model": {"weights": "…/best.pt", "device": "0", "cuda_name": "NVIDIA GeForce RTX 5060 Ti",
            "classes": 10, "loaded": true, "run": "visdrone_v4", "params": 17552134, "layers": 540},
  "classes": [ {"id":0,"name_en":"pedestrian","name_zh":"行人","color":"#32CD32"}, … ],
  "beidou": {"lat":28.169,"lon":112.944,"alt":50.0,"satellites":12,"real":false},
  "engine_available": true,
  "slicing_available": true,
  "message": null
}
```

> `model.run` = 权重所属训练产物目录名；`model.params` / `model.layers` 由**直接统计已加载权重张量**得到
> （不依赖训练侧记录），可用于与论文指标互校。

**`ClassInfo`**  `{ "id":int, "name_en":str, "name_zh":str, "color":"#RRGGBB" }`
—— 共 10 类，顺序与训练/权重严格一致：`行人 人群 自行车 小汽车 厢式货车 卡车 三轮车 带棚三轮车 公交车 摩托车`。

### 4.2 图像检测

#### POST `/api/v1/detect`  （multipart 表单，浏览器/上传用）

表单字段：

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `file` | file | — | 图片文件（与 `image` 二选一） |
| `image` | string | — | 备用：base64 data URL |
| `conf` | float | `0.25` | 置信度阈值（前端滑块直接传此值） |
| `iou` | float | `0.45` | NMS IoU |
| `imgsz` | int | `640` | 推理尺寸 |
| `max_det` | int | `300` | 最大框数 |
| `slice` | bool | `false` | 启用切片推理（小目标利器） |
| `draw` | bool | `false` | 是否返回服务端绘制图（`annotated`） |
| `strict` | bool | `false` | `true` 时演示模式返回 `503` 而非降级 |
| `return_original` | bool | `false` | 是否回显原图（`original`） |

#### POST `/api/v1/detect/json`  （JSON body，程序化 / 摄像头逐帧用）

```json
{ "image": "data:image/jpeg;base64,....", "conf":0.25, "iou":0.45,
  "imgsz":640, "max_det":300, "slice":false, "draw":false,
  "strict":false, "return_original":false }
```

#### POST `/api/v1/detect/batch`  （多图，最多 16 张）

`files: file[]` + `conf,iou,imgsz,max_det,slice`（表单）。
响应：`{ "count":int, "results":[ { "filename":str, ...DetectResponse } ] }`

**`DetectResponse`**（三个检测接口通用）
```json
{
  "mode": "real | demo",
  "width": 640, "height": 480,
  "detections": [
    {
      "class_id": 3,
      "name_en": "car",
      "name_zh": "小汽车",
      "conf": 0.87,
      "bbox": {"x1": 120.5, "y1": 80.0, "x2": 210.3, "y2": 160.2},
      "lon": 112.94401, "lat": 28.16899
    }
  ],
  "original": "data:image/jpeg;base64,.... | null",
  "annotated": "data:image/jpeg;base64,.... | null",
  "device": "cpu",
  "engine": "standard | slice | sahi | null",
  "beidou": {"lat":28.169,"lon":112.944,"alt":50.0,"satellites":12,"real":false},
  "message": null,
  "elapsed_ms": 366.1
}
```

> 注：`original` / `annotated` 仅在请求时显式 `return_original=true` / `draw=true` 才返回
> **data URL（base64）**，否则为 `null`。前端若只想自己画框，拿 `detections[]` 即可，不必让后端画。

### 4.3 视频检测（异步任务）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/video` | 提交视频，立即 `202` + `job_id` |
| GET | `/api/v1/jobs?limit=20` | 列出近期任务 |
| GET | `/api/v1/jobs/{job_id}` | 查询任务状态/结果 |
| GET | `/api/v1/files/{job_id}/{filename}` | 获取结果视频文件 |

**提交流程**
1. `POST /api/v1/video`（表单：`file` 视频，`conf,iou,imgsz,max_det,slice`）→ `202`：
   ```json
   { "job_id":"a1b2c3d4e5f6", "status":"queued",
     "status_url":"/api/v1/jobs/a1b2c3d4e5f6",
     "message":"已加入队列，轮询 status_url 获取进度与结果" }
   ```
   - 若 `APP_ENABLE_AUTH=true`，需带请求头 `X-API-Key: <APP_API_KEY>`。
   - 演示模式下返回 `503 MODEL_NOT_READY`（不支持视频推理）。
2. 前端轮询 `GET /api/v1/jobs/{job_id}`，直至 `status:"completed"`：
   ```json
   {
     "job_id":"a1b2c3d4e5f6", "status":"completed",
     "created_at":"2026-09-13T12:00:00", "updated_at":"2026-09-13T12:01:30",
     "params":{"conf":0.25,"iou":0.45,"imgsz":640,"max_det":300,"slice":false},
     "progress":{"done":300,"total":300},
     "result":{
        "video_url":"/api/v1/files/a1b2c3d4e5f6/result_a1b2c3d4e5f6.mp4",
        "stats":{"total":1234,"frames":300,"readback_frames":300,"avg_per_frame":4.11,
                 "max_per_frame":12,"class_count":[10,3,0,120,5,1,0,0,7,2],
                 "fps":30.0,"size":[1280,720],"codec":"h264",
                 "requested_codec":"avc1","bytes":12345678},
        "codec":"h264", "container":".mp4"
     },
     "error":null
   }
   ```
   `status` 取值：`queued → processing → completed | failed`。
   `progress.total` 由推理侧从视频容器读出（`CAP_PROP_FRAME_COUNT`），不是上传时的预估值。
3. 结果视频用 `result.video_url` 直接 `<video src=...>` 播放（`/files/...` 接口返回文件流）。

**结果文件的两个硬性约定**

| 约定 | 原因 |
|------|------|
| 文件名固定为 **`result_<job>.<ext>`**，绝不与上传件 `input.<ext>` 同名 | 早期实现让 ultralytics 用 `save=True` 写到输入同目录，它见目录已存在就自增成 `<job>-2/`，于是任务目录里只剩上传的原片，glob 一把抓回原片当结果 —— 表现为"结果视频和输入字节数一模一样"，**静默且极难查** |
| 逐帧**自己控读写**（cv2 `VideoCapture`/`VideoWriter`），不再用 ultralytics 的 `save=True` | 同上；顺带可以按帧回调进度、逐帧画中文标签 |

**编码器：先探测、写完再回读校验**

```
APP_VIDEO_CODEC=avc1    # 候选表 avc1(→H.264) > mp4v > XVID
```
1. **探测**：进程内先写 3 帧 128×128 小样并回读，通过才排进候选（结果按 fourcc 缓存）。
   不能只看 `VideoWriter.isOpened()` —— OpenH264 缺失时它返回 `True` 却写出 1KB 坏文件。
2. **回读校验**：整片写完后再数一遍可解码帧数，`readback_frames` 与 `frames` 不符就换下一个编码器。
3. **报真实编码**：`codec` 取回读到的 fourcc，不是请求值 —— OpenCV 遇到不认识的 fourcc
   会**静默替换**成默认编码器，照单汇报等于对外撒谎（`requested_codec` 单独保留请求值）。

> **Windows 中文路径坑（本机实测）**：`avc1` 在**含非 ASCII 字符**的路径上 `isOpened()` 直接返回
> `False`，而 `mp4v`/`XVID` 走另一条 FFmpeg 分支却正常。本项目目录常带中文，不处理就会被
> **静默降级成 mp4v（FMP4，浏览器播不了）**。处理办法：目标路径含非 ASCII 时，先写到
> **全 ASCII 临时路径**（目录名和文件名都要 ASCII —— 中文文件名会被 OpenCV 按 ANSI 落盘成乱码，
> 导致 `Path.exists()` 为 False 而 cv2 自己还能读到），校验通过后再 `shutil.move` 到最终位置。
>
> 另外：输出体积偏大（720p/30fps 约 26 Mbps，21 秒 → 约 70 MB）。OpenCV 没有暴露码率/CRF 控制
> （实测 `VIDEOWRITER_PROP_QUALITY` 各档体积完全一致），要压体积只能外接 `ffmpeg` 重编码。

### 4.4 训练指标（论文数据直出）

把「已加载权重」所属训练产物目录**只读**解析出来，供前端展示真实收敛曲线与消融对比 ——
前端不再硬编码任何指标，模型换了指标就跟着换。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/metrics` | 已加载权重对应 run 的指标快照 |
| GET | `/api/v1/metrics/runs` | 列出各训练根目录下所有可用 run（含 best mAP50） |
| GET | `/api/v1/metrics/artifact/{name}` | 训练配图原图（`results.png` / `BoxPR_curve.png` / 混淆矩阵…） |

**数据来源（全部只读，绝不改写训练产物）**

| 文件 | 用途 |
|------|------|
| `<run>/results.csv` | 逐 epoch 指标（ultralytics 官方列名） |
| `<run>/args.yaml` | 本次超参（白名单字段） |
| `<runs_root>/对比_*.txt` | 多版本消融对比表（可选，自动探测） |

**`/api/v1/metrics` 响应（以 `visdrone_v4` 为例）**
```json
{
  "available": true, "run": "visdrone_v4",
  "run_dir": "C:\\yolo_runs\\train\\visdrone_v4",
  "weights": "C:\\yolo_runs\\train\\visdrone_v4\\weights\\best.pt", "weights_mb": 34.0,
  "epochs": 117, "params": 17552134, "layers": 540,
  "best": {"epoch": 74, "mAP50": 0.44573, "mAP50_95": 0.24666, "precision": 0.55813, "recall": 0.43154},
  "last": {"epoch": 117, "mAP50": 0.43432, "…": "…"},
  "curves": {"epoch":[…117], "mAP50":[…], "mAP50_95":[…], "precision":[…], "recall":[…],
             "box_loss":[…], "cls_loss":[…], "dfl_loss":[…], "lr0":[…]},
  "args": {"model":"yolov8m_visdrone_slimneck_nop5.yaml", "epochs":150, "batch":4, "imgsz":640,
           "optimizer":"AdamW", "lr0":0.0005, "seed":42, "device":"0", "copy_paste":0.3, "…":"…"},
  "artifacts": [{"name":"results.png","url":"/api/v1/metrics/artifact/results.png"}, "…10 张"],
  "comparison": {
    "name": "对比_v1_v4.txt",
    "items": [
      {"name":"v1 基线 (DualConv, 4尺度)","mAP50":0.4563,"mAP50_95":0.2475,
       "precision":0.593,"recall":0.4404,"params":24892928},
      {"name":"v4 推荐 (Slim-Neck + 剪P5, 3尺度)","mAP50":0.4457,"mAP50_95":0.2467,
       "precision":0.575,"recall":0.4327,"params":17552134}
    ],
    "notes": ["v1：best mAP50 第 76 epoch", "v4：best mAP50 第 74 epoch",
              "mAP50 -0.0105  (-1.05 pp)", "参数量 -29.5%   (24,892,928 → 17,552,134)",
              "⚠ 两组轮数不一致，对比不严谨，建议同轮数重跑"]
  },
  "runs": [{"name":"visdrone_v1","epochs":118,"best_mAP50":0.45627,"has_weights":true}, "…"]
}
```

**要点**
- `best` 按 **mAP50 最大值**取，与 ultralytics 判定 `best.pt` 的口径一致（v4 = 第 74 epoch）。
- `params` 来自**已加载权重**的张量统计，与 `对比_v1_v4.txt` 的 `17,552,134` 完全吻合 —— 可交叉验证。
- 权重不在标准产物目录（如手放的单文件）时 `available:false` 并给出 `reason`，接口不报错。
- `artifact` 走**文件名白名单 + 路径穿越校验**，`../` 与不存在文件均返回 `404`。
- 结果带内存缓存（训练产物不会中途变化），`MetricsService.snapshot(force=True)` 可强制刷新。

---

## 5. 统一错误信封

所有错误（含校验失败、未捕获异常）都返回同一结构，便于前端统一拦截：

```json
{ "error": { "code": "MODEL_NOT_READY", "message": "推理引擎不可用：ultralytics 未安装", "details": null },
  "request_id": "a1b2c3..." }
```

每个响应（含错误）都带 `X-Request-ID` 响应头，可透传用于日志串联。

**错误码 → HTTP 状态**

| code | HTTP | 含义 |
|------|------|------|
| `BAD_REQUEST` | 400 | 参数缺失/非法（如未上传文件） |
| `UNAUTHORIZED` | 401 | 缺少/无效 API Key（启用鉴权时） |
| `NOT_FOUND` | 404 | 任务或文件不存在 |
| `UNSUPPORTED_MEDIA` | 415 | 不支持的媒体类型 |
| `PAYLOAD_TOO_LARGE` | 413 | 图片/视频超过 `APP_MAX_UPLOAD_MB` / `APP_MAX_VIDEO_MB` |
| `MODEL_NOT_READY` | 503 | 演示模式不支持该操作（如视频推理） |
| `VALIDATION_ERROR` | 422 | 请求体/表单校验失败（`details` 含字段级错误） |
| `INTERNAL_ERROR` | 500 | 服务器内部错误 |
| `HTTP_<code>` | 其它 | 透传的 Starlette HTTP 异常 |

---

## 6. 配置（环境变量，前缀 `APP_`，可用 `.env`）

完整清单见 `server/.env.example`。常用项：

| 变量 | 默认 | 说明 |
|------|------|------|
| `APP_PORT` / `APP_HOST` | `8000` / `0.0.0.0` | 监听 |
| `APP_CORS_ORIGINS` | `*` | 前端域名白名单（生产请显式指定） |
| `APP_PRELOAD_MODEL` | `true` | 启动即加载权重（false=首次请求懒加载） |
| `APP_DEVICE` | `auto` | `auto\|cpu\|cuda\|0` |
| `APP_INFERENCE_CONCURRENCY` | `1` | GPU 建议 1（防 OOM）；CPU 可调大 |
| `APP_FORCE_DEMO` | `false` | `true`=强制演示模式（不加载模型，纯 UI 演示部署） |
| `APP_BEIDOU_LAT/LON/ALT/SATELLITES/REAL` | 长沙·岳麓 | 北斗演示坐标 |
| `APP_SLICE_ENABLED` / `SLICE_SIZE` / `SLICE_OVERLAP` / `SLICE_ENGINE` | `true`/`640`/`0.2`/`builtin` | 切片推理（小目标）；`builtin`=零依赖，`sahi`=需装 sahi |
| `APP_MAX_UPLOAD_MB` / `APP_MAX_VIDEO_MB` | `50` / `500` | 上传上限 |
| `APP_ENABLE_AUTH` / `APP_API_KEY` | `false` / `""` | 可选视频接口鉴权 |
| `APP_WEIGHTS` | 空 | **显式指定权重路径，优先级最高**。训练产物不在标准位置时用它兜底 |
| `APP_RUNS_DIRS` | `C:/yolo_runs/train;<项目>/runs/train` | 训练产物**根目录**，多个用 `,` 或 `;` 分隔 |
| `APP_RUN_NAMES` | `visdrone_v4,visdrone_v3,visdrone_v2,visdrone_v1` | 正式 run 名（优先级从高到低） |
| `APP_FALLBACK_RUN_NAMES` | `smoke_v4,smoke_v4-2,gpu_check` | 兜底 run 名（冒烟/自检产物，**仅当正式 run 全缺时才用**） |
| `APP_METRICS_ENABLED` | `true` | 是否暴露 §4.4 训练指标接口 |
| `APP_METRICS_COMPARE_GLOB` | `对比_*.txt` | 消融对比文件名（相对 run 目录的上一级） |
| `APP_VIDEO_CODEC` | `avc1` | 视频输出首选编码器：`avc1`(→H.264，浏览器可播)｜`mp4v`｜`XVID`。不可用会自动降级，见 §4.3 |

**权重解析顺序（`weights_candidates`）**

```
APP_WEIGHTS（若设置）
→ 对 RUN_NAMES 中的每个名字，遍历 RUNS_DIRS 各根目录： <根>/<run名>/weights/best.pt
→ 再对 FALLBACK_RUN_NAMES 做同样遍历
```

取**第一个存在的文件**。`run 名` 在外层、根目录在内层，因此「正式 run 在任何根目录下的产物」
都优先于「兜底 run」—— 这修掉了一个静默错误：原先候选表里 `visdrone_v4` 指向项目内
不存在的 `runs/train/`，于是**冒烟测试的玩具模型 `smoke_v4/weights/best.pt` 会被当作真实权重加载**
（参数量与 mAP 全错，但接口一切正常，很难察觉）。

**排查用**：`GET /api/v1/models` 会返回全部候选与各自是否存在；`GET /api/v1/status` 的
`model.run` / `model.params` 可直接核对加载的是哪份权重（v4 = `17,552,134`）。

---

## 7. 前端对接要点 / 与旧 `web/` 前端的差异（迁移须知）

> 旧的 `修正版代码/web/` 是**已被取代的简陋版前端+后端**，新前端请**忽略它**，只对接本后端。

| 旧（web/index.html 消费） | 新（本后端） |
|---------------------------|--------------|
| `/api/status`、`/api/detect`、`/api/video`、`/api/file/<name>` | 全部改为 `/api/v1/status`、`/api/v1/detect`、`/api/v1/video`、`/api/v1/jobs`、`/api/v1/files/{job_id}/{filename}` |
| 类别三字段 `classes_zh` / `classes_en` / `class_colors` | `GET /api/v1/classes` 返回 `ClassInfo[]`（每项含 `id,name_en,name_zh,color`） |
| detect 响应 `original`/`annotated` 为 URL | 现改为 **data URL（base64）**，且仅 `return_original`/`draw` 时返回；视频结果仍用 `/files/...` URL |
| 无错误统一结构 | 统一错误信封（§5） |
| 无任务轮询 | 视频改为异步任务：提交→`202`→轮询 `/jobs/{id}` |
| `/api/detect` 同时吃 multipart 与 JSON | **拆成两个端点**：`/detect`（multipart）与 `/detect/json`（JSON） |

> ⚠ **JSON 检测必须打 `/detect/json`**。把 `Content-Type: application/json` 的请求发到 `/detect`
> 会得到 `400 BAD_REQUEST`「请上传 file 或提供 image(base64)」——因为该端点的 `file`/`image`
> 是 `Form(...)`，JSON body 不会被解析。这个坑很隐蔽：请求确实到了后端、也确实返回了合法错误信封，
> 前端若只 toast 不细看就会表现为「点了推理没反应，还是旧结果」。

**最小前端启动步骤**
1. `GET /api/v1/status` → 读 `mode`、`classes`、`beidou` 初始化界面。
2. 上传图片 → `POST /api/v1/detect`（multipart，`draw=true` 可直接拿标注图，或自己用 `detections[].bbox` 画）。
3. **dataURL / 摄像头帧** → `POST /api/v1/detect/json`（JSON `{image, conf, iou, imgsz, max_det, slice}`）。
4. 置信度滑块 → 直接传 `conf` 参数（后端已返回低阈值候选，前端过滤即可）。
5. 视频 → `POST /api/v1/video` 拿到 `job_id` → 轮询 `GET /api/v1/jobs/{job_id}` → 用 `result.video_url` 播放。
6. 训练指标 → `GET /api/v1/metrics`（§4.4），配图用 `/api/v1/metrics/artifact/{name}` 直出。

---

## 8. 部署

- **健康检查**：k8s 用 `GET /healthz`（liveness）与 `GET /readyz`（readiness）。
- **就绪判断**：`/readyz` 在 lifespan 完成后即 200；是否真的 `real` 看 `/api/v1/status` 的 `mode`。
- **静态资源**：前端（`server/static/`，零依赖 SPA）已**挂载在根路径 `/`**，与 API 同源发布。
  打开 `http://localhost:8000/` 即用，不需要另起静态服务器，也不涉及 CORS。
  挂载顺序很关键：`StaticFiles` 必须在所有 `include_router` **之后**挂载，否则会抢先兜住 `/api/v1/*`。
- **生产建议**：`APP_CORS_ORIGINS` 显式写前端域名；开启 `APP_ENABLE_AUTH` 保护视频接口；
  GPU 机器用 `cu128` 版 torch（`sm_120`/RTX 50 系必需）；`APP_WORKERS` 一般保持 1（推理有状态）。
  如需前后端分离部署，把 `server/static/` 交给 Nginx/CDN，后端只留 API 即可。

---

## 9. 自检 / 测试（后端质量门）

```bash
# 快速合约测试（强制演示模式，无需模型/权重，秒级）
APP_FORCE_DEMO=true pytest server/tests -q

# 完整测试（若有 ultralytics + 权重，会额外跑真实推理集成测试）
pytest server/tests -q
```

覆盖：健康检查、`/` 归前端、`/info`、探针别名、status/classes/beidou/models、
metrics 契约与配图路径穿越防护、detect 校验、演示降级、严格 503、
超限 413、任务 404、视频 503、geo_map + 绘制 + NMS 单元测试，以及真实模式下的
「注册表→推理→解析→绘制」全链路 + 训练指标自洽校验。

视频专项（`server/tests/test_video.py`）：编码器候选/探测/回读、`_read_fourcc` 报真实编码、
绘制函数（含顶部标签翻转、空检测原样返回、ASCII 临时目录），以及**中文输出目录**下的完整
端到端（帧数守恒 + 字节不同 + 抽帧确有改动 + 结果名非 `input.*`）。

本机实测：**37 passed**（含 `test_real_pipeline` 真实模型推理、`visdrone_v4` 指标自洽、
视频端到端）。缺少 ultralytics / 权重 / VisDrone 原图时相关用例自动跳过，CI 上也能跑。

---

## 10. 目录与文件清单

| 文件 | 作用 |
|------|------|
| `server/run.py` | 启动器（注册自定义模块 + uvicorn） |
| `server/main.py` | `create_app()` 工厂 / 生命周期 / 中间件 / 路由 / **前端挂载** |
| `server/config.py` | 类型安全配置（含 `weights_candidates` 解析顺序） |
| `server/core/errors.py` | 统一异常 + 错误信封 |
| `server/services/model_registry.py` | 模型注册表（双模式核心；记录 `run_dir` / `params`） |
| `server/services/inference.py` | 推理（标准/切片/sahi/视频，含编码器探测与回读校验） |
| `server/services/drawing.py` | PIL 绘制中文标签（视频逐帧与单图共用） |
| `server/services/metrics.py` | 训练产物只读解析（results.csv / args.yaml / 对比表 / 配图） |
| `server/routers/metrics.py` | §4.4 训练指标接口 |
| `server/schemas/*` | 全部 Pydantic 契约 |
| `server/routers/*` | 全部 HTTP 接口 |
| `server/static/*` | 零依赖前端 SPA（挂在 `/`） |
| `server/tests/*` | 合约 / 集成测试 |
| `server/Dockerfile` `docker-compose.yml` `.env.example` `requirements.txt` | 部署与依赖 |
```
