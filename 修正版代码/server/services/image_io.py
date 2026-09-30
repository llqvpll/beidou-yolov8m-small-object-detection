"""图像编解码与 data URL 工具（懒加载 cv2 / numpy）。"""
from __future__ import annotations

import base64

import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """bytes -> BGR ndarray。失败抛 ValueError。"""
    import cv2

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像，请检查格式（支持 jpg/png/bmp 等）")
    return img


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("图像编码失败")
    return buf.tobytes()


def to_data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def from_data_url(s: str) -> bytes:
    """从 'data:image/jpeg;base64,....' 解析出原始字节。"""
    _, _, b64 = s.partition(",")
    if not b64:
        raise ValueError("非法的 data URL")
    return base64.b64decode(b64)
