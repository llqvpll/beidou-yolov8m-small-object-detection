"""用 PIL 绘制中文标签的检测框。

对外两个入口：
  - :func:`draw_detections_bgr` —— 返回画好框的 BGR ndarray（**视频逐帧**用）
  - :func:`draw_detections`     —— 返回 JPEG 字节（**单图**接口用，内部复用上面那个）

为什么用 PIL 而不是 cv2 的 ``putText``：OpenCV 内置字体不支持中文，
``putText("行人")`` 会画出一串问号。PIL 加载系统字体（微软雅黑 / 文泉驿 / 苹方）才能正常显示。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .classes import CLASS_COLORS_BGR, CLASSES_ZH


@lru_cache(maxsize=16)
def _find_font(size: int):
    """按可用性挑一个支持中文的字体。

    **必须缓存**：视频逐帧调用，每帧都去磁盘探测字体路径会明显拖慢整体吞吐。
    """
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    from PIL import ImageFont

    for cand in candidates:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def draw_detections_bgr(img_bgr: np.ndarray, detections: list[dict], w: int, h: int) -> np.ndarray:
    """在 BGR 图上画检测框与中文标签，返回同尺寸 BGR 图（不修改入参）。

    :param detections: ``[{class, conf, x1, y1, x2, y2}, ...]``，像素坐标。
    """
    import cv2
    from PIL import Image, ImageDraw

    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    font = _find_font(max(14, int(h * 0.022)))
    scale = w / 640.0
    lw = max(1, int(2 * scale))
    label_h = max(16, int(h * 0.026))

    for d in detections:
        cid = int(d["class"])
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        col = tuple(int(c) for c in CLASS_COLORS_BGR[cid][::-1])  # BGR -> RGB
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = f"{CLASSES_ZH[cid]} {float(d['conf']):.2f}"
        tw = draw.textlength(label, font=font)
        # 标签贴在框上沿外侧；顶到画面外就翻到框内，避免被裁掉
        ty = y1 - label_h
        if ty < 0:
            ty = min(y1, h - label_h)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + label_h], fill=col)
        draw.text((x1 + 2, ty + 1), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def draw_detections(img_bgr: np.ndarray, detections: list[dict], w: int, h: int) -> bytes:
    """单图用：返回 JPEG 字节。

    ``detections`` 为 ``[{class, conf, x1, y1, x2, y2}, ...]``（像素坐标）。
    """
    import cv2

    out = draw_detections_bgr(img_bgr, detections, w, h)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()
