#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北斗 + 改进 YOLOv8m 小目标识别辅助系统 · Web 后端
================================================
- 真实模式：加载 runs/train/{visdrone_v4,smoke_v4}/weights/best.pt（含 EMA/DySample/GSConv/VoVGSCSP），
            做真实推理；自动探测 CUDA，无卡时回落 CPU。
- 演示模式：若环境缺 ultralytics / 权重缺失，自动降级，前端仍可运行（演示数据，明确标注）。

所有重依赖（cv2 / numpy / torch / ultralytics / PIL）均懒加载，
保证「无 torch 环境也能启动 Flask 并提供页面 / 状态 / 演示流程」。
"""

# ---- Windows 中文系统编码补丁 ----
# 默认 locale 为 GBK，ultralytics 读取 settings.json 时遇到非 ASCII 字节会抛
# "'gbk' codec can't decode ..."。强制文本默认编码为 UTF-8，消除该告警。
import sys as _sys
import locale as _locale
try:
    _locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"  # noqa: E731
except Exception:
    pass
try:
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import sys
import io
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent          # web/
ROOT = HERE.parent                               # 修正版代码/
for p in (str(ROOT), str(ROOT / "custom_modules")):
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, request, jsonify, send_file

# ----------------------------- 配置 -----------------------------
WEIGHTS_CANDIDATES = [
    os.environ.get("WEIGHTS"),
    str(ROOT / "runs/train/visdrone_v4/weights/best.pt"),
    str(ROOT / "runs/train/smoke_v4/weights/best.pt"),
    str(ROOT / "yolov8m.pt"),
]
WEIGHTS_CANDIDATES = [p for p in WEIGHTS_CANDIDATES if p]

# VisDrone-2019 官方 10 类（顺序与训练一致）
CLASSES_ZH = ["行人", "人群", "自行车", "小汽车", "厢式货车",
              "卡车", "三轮车", "带棚三轮车", "公交车", "摩托车"]
CLASSES_EN = ["pedestrian", "people", "bicycle", "car", "van",
              "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# 每类固定颜色 (B, G, R)
CLASS_COLORS = [
    (50, 205, 50), (60, 20, 230), (255, 255, 0), (0, 255, 255), (255, 140, 0),
    (0, 0, 255), (220, 220, 220), (255, 0, 255), (255, 0, 0), (0, 165, 255),
]

# 北斗定位（演示坐标，可经环境变量配置真实读数）
BEIDOU = {
    "lat": float(os.environ.get("BEIDOU_LAT", "28.169")),       # 默认：长沙·岳麓（湖南大学）
    "lon": float(os.environ.get("BEIDOU_LON", "112.944")),
    "alt": float(os.environ.get("BEIDOU_ALT", "50.0")),
    "real": os.environ.get("BEIDOU_REAL", "0").lower() in ("1", "true", "yes"),
    "satellites": int(os.environ.get("BEIDOU_SATS", "12")),
}

OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(HERE / "static"))

MODEL = None
MODE = "demo"
MODEL_INFO = {}
DEVICE = "cpu"

# 推理用的「最低置信度」：低阈值拿全部候选框，前端滑块再实时过滤
DETECT_CONF_FLOOR = 0.05


# ----------------------------- 模型加载 -----------------------------
def ensure_custom_modules() -> bool:
    """确认 EMA/DySample/DualConv/GSConv/VoVGSCSP 已注册进 ultralytics。"""
    try:
        from custom_modules.register import ensure_installed
        # auto=False：仅检查；真正安装交给 start_web.py，避免进程内 SystemExit
        ensure_installed(auto=False, verbose=False)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[web] 自定义模块检查失败: {e}")
        return False


def load_model():
    global MODEL, MODE, MODEL_INFO, DEVICE
    weights = None
    for c in WEIGHTS_CANDIDATES:
        if c and Path(c).exists():
            weights = c
            break
    if not weights:
        MODE = "demo"
        MODEL_INFO = {"reason": "未找到权重文件（候选路径均不存在）"}
        print("[web] 进入演示模式：未找到权重")
        return
    try:
        ensure_custom_modules()
        import torch  # 懒加载
        DEVICE = "0" if torch.cuda.is_available() else "cpu"
        from ultralytics import YOLO  # 懒加载
        m = YOLO(weights)
        # 轻量预热，避免首帧卡顿
        try:
            import numpy as np
            m.predict(np.zeros((320, 320, 3), dtype=np.uint8),
                      device=DEVICE, verbose=False)
        except Exception:  # noqa: BLE001
            pass
        MODEL = m
        MODE = "real"
        variant = "未知"
        s = str(weights)
        if "visdrone_v4" in s or "smoke_v4" in s:
            variant = "v4（Slim-Neck + 剪 P5，3 尺度，推荐）"
        MODEL_INFO = {
            "weights": str(weights),
            "device": DEVICE,
            "cuda_name": (torch.cuda.get_device_name(0) if DEVICE == "0" else None),
            "classes": len(CLASSES_ZH),
            "variant": variant,
        }
        print(f"[web] 真实模式：加载 {weights} @ {DEVICE}")
    except Exception as e:  # noqa: BLE001
        MODE = "demo"
        MODEL_INFO = {"reason": f"模型加载失败：{e}"}
        print(f"[web] 进入演示模式：{e}")


# ----------------------------- 工具 -----------------------------
def geo_map(cx, cy, w, h):
    """示意性地理映射：图像中心对应北斗坐标，按像素偏移估算。
    注：演示用，非真实地理标定，需在部署时接入真实内参/北斗读数。"""
    deg_per_px_x = 0.00001 * (w / 640.0)
    deg_per_px_y = 0.00001 * (h / 640.0)
    dlon = (cx - w / 2) * deg_per_px_x
    dlat = -(cy - h / 2) * deg_per_px_y
    return round(BEIDOU["lon"] + dlon, 6), round(BEIDOU["lat"] + dlat, 6)


def pil_draw(img_bgr, detections, w, h):
    """用 PIL 画中文标签（自动回退字体），返回 JPEG base64。"""
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    font = None
    for cand in ("C:/Windows/Fonts/msyh.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/PingFang.ttc"):
        if Path(cand).exists():
            try:
                font = ImageFont.truetype(cand, max(14, int(h * 0.022)))
                break
            except Exception:  # noqa: BLE001
                font = None
    if font is None:
        font = ImageFont.load_default()
    scale = w / 640.0
    lw = max(1, int(2 * scale))
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        col = tuple(int(c) for c in CLASS_COLORS[d["class"]][::-1])  # RGB
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = f"{CLASSES_ZH[d['class']]} {d['conf']:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x1, y1 - 16, x1 + tw + 4, y1], fill=col)
        draw.text((x1 + 2, y1 - 15), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def read_image_request():
    """从请求中解析图像（multipart 文件 或 JSON 中的 base64）。返回 BGR ndarray。"""
    import cv2
    import numpy as np
    f = request.files.get("file")
    if f is not None:
        data = f.read()
    elif request.is_json and request.json and request.json.get("image"):
        b64 = request.json["image"].split(",", 1)[-1]
        import base64
        data = base64.b64decode(b64)
    else:
        return None
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ----------------------------- 路由 -----------------------------
@app.route("/")
def index():
    idx = HERE / "static" / "index.html"
    if idx.exists():
        return send_file(str(idx))
    return "index.html 缺失，请确认 web/static/index.html 存在。", 500


@app.route("/api/status")
def status():
    return jsonify({
        "mode": MODE,
        "device": (DEVICE if MODE == "real" else None),
        "model": MODEL_INFO,
        "classes_zh": CLASSES_ZH,
        "classes_en": CLASSES_EN,
        "class_colors": ["#%02X%02X%02X" % c[::-1] for c in CLASS_COLORS],
        "beidou": BEIDOU,
    })


@app.route("/api/beidou")
def beidou():
    return jsonify(BEIDOU)


@app.route("/api/detect", methods=["POST"])
def detect():
    # 演示模式：无模型也返回原图 + 清晰提示，不抛错
    if MODE != "real" or MODEL is None:
        try:
            import cv2
            img = read_image_request()
            if img is None:
                return jsonify({"mode": "demo", "detections": [],
                                "message": "演示模式：未上传图像"}), 200
            _, buf = cv2.imencode(".jpg", img)
            import base64
            orig = base64.b64encode(buf).decode()
            return jsonify({
                "mode": "demo",
                "original": "data:image/jpeg;base64," + orig,
                "detections": [],
                "message": "当前为演示模式（未加载真实模型）。前端可用「示例演示」查看完整交互效果。",
            })
        except Exception as e:  # noqa: BLE001
            return jsonify({"mode": "demo", "detections": [],
                            "message": f"演示模式：无法读取图像（{e}）"}), 200

    try:
        import cv2
        img = read_image_request()
        if img is None:
            return jsonify({"error": "未收到图像（file 或 image 字段）"}), 400
        h, w = img.shape[:2]
        iou = float(request.values.get("iou", 0.45))
        max_det = int(request.values.get("max_det", 300))
        imgsz = int(request.values.get("imgsz", 640))
        results = MODEL.predict(img, conf=DETECT_CONF_FLOOR, iou=iou,
                                max_det=max_det, imgsz=imgsz,
                                device=DEVICE, verbose=False)
        dets = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cls = int(b.cls[0])
                cf = float(b.conf[0])
                dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "class": cls, "conf": cf})
        # 地理映射（示意）
        for d in dets:
            cx = (d["x1"] + d["x2"]) / 2
            cy = (d["y1"] + d["y2"]) / 2
            lon, lat = geo_map(cx, cy, w, h)
            d["lon"], d["lat"] = lon, lat
        _, obuf = cv2.imencode(".jpg", img)
        import base64
        orig = base64.b64encode(obuf).decode()
        annotated = pil_draw(img.copy(), dets, w, h) if dets else None
        return jsonify({
            "mode": "real",
            "original": "data:image/jpeg;base64," + orig,
            "annotated": ("data:image/jpeg;base64," + annotated) if annotated else None,
            "detections": dets,
            "width": w,
            "height": h,
            "device": DEVICE,
            "beidou": BEIDOU,
        })
    except Exception as e:  # noqa: BLE001
        return jsonify({"mode": "real", "error": str(e)}), 500


@app.route("/api/video", methods=["POST"])
def video():
    if MODE != "real" or MODEL is None:
        return jsonify({"mode": "demo", "message": "演示模式不支持视频推理"}), 200
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "未收到视频文件"}), 400
    job = uuid.uuid4().hex[:8]
    inp = OUTPUT_DIR / f"{job}_in.mp4"
    f.save(str(inp))
    if inp.stat().st_size > 200 * 1024 * 1024:
        return jsonify({"error": "视频过大（>200MB），请先裁剪"}), 413
    try:
        conf = float(request.values.get("conf", 0.25))
        iou = float(request.values.get("iou", 0.45))
        imgsz = int(request.values.get("imgsz", 640))
        results = MODEL.predict(str(inp), conf=conf, iou=iou, imgsz=imgsz,
                                device=DEVICE, save=True,
                                project=str(OUTPUT_DIR), name=job, verbose=False)
        saved_dir = OUTPUT_DIR / job
        out_file = None
        if saved_dir.exists():
            for v in sorted(saved_dir.glob("*.mp4")):
                out_file = v
                break
        class_count = {i: 0 for i in range(len(CLASSES_ZH))}
        per_frame = []
        total = 0
        for r in results:
            n = 0
            if r.boxes is not None:
                n = len(r.boxes)
                total += n
                for b in r.boxes:
                    class_count[int(b.cls[0])] += 1
            per_frame.append(n)
        stats = {
            "total": total,
            "per_frame": per_frame,
            "class_count": [class_count[i] for i in range(len(CLASSES_ZH))],
            "frames": len(per_frame),
            "avg_per_frame": round(total / len(per_frame), 2) if per_frame else 0,
            "max_per_frame": max(per_frame) if per_frame else 0,
        }
        video_url = f"/api/file/{out_file.name}" if out_file else None
        return jsonify({"mode": "real", "video_url": video_url, "stats": stats,
                        "job": job, "device": DEVICE})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.route("/api/file/<name>")
def file(name):
    name = os.path.basename(name)  # 防目录穿越
    p = OUTPUT_DIR / name
    if not p.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(str(p), mimetype="video/mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 56)
    print("北斗 · 改进 YOLOv8m 小目标识别辅助系统 · Web 启动中 ...")
    load_model()
    print(f"模式: {MODE}   设备: {DEVICE}")
    print(f"打开浏览器访问: http://localhost:{port}")
    print("=" * 56)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
