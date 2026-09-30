"""VisDrone-2019 官方 10 类（顺序与训练、权重一致）。

优先从 ``VisDrone.yaml`` 读取 ``names``，读取失败则回退到下方硬编码
（与 yaml 顺序完全一致，且与原 web 后端保持一致）。
"""
from __future__ import annotations

from pathlib import Path

from ..config import ROOT
from ..schemas.detection import ClassInfo

# 顺序必须与训练时 VisDrone.yaml 的 names 一致
CLASSES_EN = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
CLASSES_ZH = [
    "行人", "人群", "自行车", "小汽车", "厢式货车",
    "卡车", "三轮车", "带棚三轮车", "公交车", "摩托车",
]
# 每类固定颜色 (B, G, R) —— 绘制与前端统一
CLASS_COLORS_BGR = [
    (50, 205, 50), (60, 20, 230), (255, 255, 0), (0, 255, 255), (255, 140, 0),
    (0, 0, 255), (220, 220, 220), (255, 0, 255), (255, 0, 0), (0, 165, 255),
]


def _load_from_yaml() -> list[str] | None:
    yaml_path = ROOT / "VisDrone.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, dict) and names:
            # names 形如 {0: pedestrian, ...}
            return [names[int(k)] for k in sorted(names, key=lambda x: int(x))]
    except Exception:
        return None
    return None


def classes_en() -> list[str]:
    return _load_from_yaml() or CLASSES_EN


def classes_zh() -> list[str]:
    # 中文名为人工标注，yaml 里只有英文；直接用硬编码中文表
    return CLASSES_ZH


def class_infos() -> list[ClassInfo]:
    en = classes_en()
    zh = classes_zh()
    return [
        ClassInfo(
            id=i,
            name_en=en[i],
            name_zh=zh[i],
            color="#%02X%02X%02X" % CLASS_COLORS_BGR[i][::-1],
        )
        for i in range(len(en))
    ]
