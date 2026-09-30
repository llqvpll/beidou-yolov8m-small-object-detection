"""人群密度监测与聚集预警。

设计要点
--------
1. **计数口径**：VisDrone 的 ``pedestrian``(0) 与 ``people``(1) 都算人。
   其中 ``people`` 是「密集人群」标注——一个框代表多个人，难以逐人分离。
   因此按 ``people_weight`` 折算（默认 **1.0**，即保守低估不夸大）。
   原始的两类计数会**分别上报**，便于按需自行调整口径。

2. **密度换算必须有标定**：需要画面实际覆盖面积，两种给法
   - ``area_m2``        直接给面积（m²）
   - ``meters_per_px``  给比例尺（米/像素），面积 = 宽 × 高 × 比例尺²

   **两者都没给时绝不硬算密度**，只报人数和占比。宁可不给，也不给拍脑袋的假数字。

3. **尺度无关的参考量** ``occupancy_ratio``：人群框面积 / 画面面积。
   不需要任何标定，适合"先看看挤不挤"。但它受拍摄高度影响，不能当绝对密度用。

4. **分级阈值**（人/m²）参考 Fruin 服务水平与多起踩踏事故复盘：
   ``<1 正常 | 1~2 关注 | 2~4 警戒 | >4 危险``。阈值可配置。

关键限制（必须在材料中如实说明）
------------------------------
**遮挡悖论**：人群越密，遮挡越严重，检测器越容易漏检——最需要准确的时候
精度反而最低，高密度下计数会**系统性偏低**。所以本模块更适合
「中低密度的态势感知与相对趋势判断」，不宜作为精确人数计量的依据。
"""
from __future__ import annotations

import logging

log = logging.getLogger("server.crowd")

# VisDrone 官方类别索引
CLS_PEDESTRIAN = 0
CLS_PEOPLE = 1
CROWD_CLASS_IDS = (CLS_PEDESTRIAN, CLS_PEOPLE)

# 各级：(下界, 上界, key, 中文名, 是否触发预警)
_LEVEL_KEYS = ("normal", "watch", "warn", "danger")
_LEVEL_ZH = {"normal": "正常", "watch": "关注", "warn": "警戒", "danger": "危险"}
# 只有「警戒」及以上才算预警事件
_ALERT_FROM = {"normal": False, "watch": False, "warn": True, "danger": True}


def level_of(density: float | None, watch: float = 1.0, warn: float = 2.0,
             danger: float = 4.0) -> dict | None:
    """按密度返回分级。density 为 None（未标定）时返回 None——不给假结论。"""
    if density is None:
        return None
    if density >= danger:
        key = "danger"
    elif density >= warn:
        key = "warn"
    elif density >= watch:
        key = "watch"
    else:
        key = "normal"
    return {"key": key, "zh": _LEVEL_ZH[key], "alert": _ALERT_FROM[key]}


def analyze(
    boxes,
    width: int,
    height: int,
    *,
    area_m2: float | None = None,
    meters_per_px: float | None = None,
    people_weight: float = 1.0,
    watch: float = 1.0,
    warn: float = 2.0,
    danger: float = 4.0,
) -> dict:
    """从检测结果统计人群，返回 ``{"count", ..., "density", "level", ...}``。

    :param boxes: ``[(x1, y1, x2, y2, class_id, conf), ...]``（``_parse`` 的输出格式）
    :param area_m2: 画面实际覆盖面积（m²）；与 ``meters_per_px`` 二选一，优先此项
    :param meters_per_px: 比例尺（米/像素）
    """
    n_ped = 0
    n_ppl = 0
    box_area = 0.0

    # 注意别写 `boxes or ()`：推理管线传的是 ndarray，
    # 真值判断会抛 "truth value of an array ... is ambiguous"。
    if boxes is None:
        boxes = ()
    for b in boxes:
        cid = int(b[4])
        if cid not in CROWD_CLASS_IDS:
            continue
        if cid == CLS_PEDESTRIAN:
            n_ped += 1
        else:
            n_ppl += 1
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        box_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)

    weight = float(people_weight) if people_weight and people_weight > 0 else 1.0
    total = n_ped + n_ppl * weight

    # ---- 面积与密度 ----
    area = None
    if area_m2 and area_m2 > 0:
        area = float(area_m2)
    elif meters_per_px and meters_per_px > 0:
        area = float(width) * float(height) * (float(meters_per_px) ** 2)

    density = (total / area) if (area and area > 0) else None
    level = level_of(density, watch=watch, warn=warn, danger=danger)

    # ---- 尺度无关的拥挤度参考（无需标定）----
    frame_area = float(width) * float(height)
    occupancy = min(1.0, box_area / frame_area) if frame_area > 0 else 0.0

    msg = None
    if density is None:
        msg = ("未标定画面实际面积，只给出人数与画面占比；"
               "传入 area_m2 或 meters_per_px 即可得到密度与分级。")

    return {
        "count": int(round(total)),
        "count_pedestrian": n_ped,
        "count_people": n_ppl,
        "people_weight": weight,
        "area_m2": round(area, 2) if area else None,
        "density": round(density, 3) if density is not None else None,
        "level": level,
        "occupancy_ratio": round(occupancy, 4),
        "calibrated": density is not None,
        "thresholds": {"watch": watch, "warn": warn, "danger": danger},
        "message": msg,
    }


class FrameSeries:
    """视频逐帧人群累积器：记录每帧人数/密度，最后给出汇总与抽稀后的曲线。

    曲线默认最多 ``max_points`` 个点，超长视频均匀抽稀，避免把几万个数字塞进 JSON。
    """

    def __init__(self, max_points: int = 300) -> None:
        self.max_points = max(10, int(max_points))
        self.counts: list[int] = []
        self.densities: list[float] = []
        self.peak_level = "normal"
        self.alert_frames = 0

    def add(self, count: int, density: float | None, level: dict | None) -> None:
        self.counts.append(int(count))
        if density is not None:
            self.densities.append(float(density))
        if level:
            key = level.get("key", "normal")
            if _LEVEL_KEYS.index(key) > _LEVEL_KEYS.index(self.peak_level):
                self.peak_level = key
            if level.get("alert"):
                self.alert_frames += 1

    def _thin(self, seq: list):
        """均匀抽稀到 max_points 个点（保留首尾）。"""
        n = len(seq)
        if n <= self.max_points:
            return list(seq)
        step = n / float(self.max_points)
        out = [seq[int(i * step)] for i in range(self.max_points)]
        if out and out[-1] != seq[-1]:
            out[-1] = seq[-1]
        return out

    def summary(self, watch: float = 1.0, warn: float = 2.0,
                danger: float = 4.0) -> dict:
        n = len(self.counts)
        peak_count = max(self.counts) if self.counts else 0
        avg_count = (sum(self.counts) / n) if n else 0.0
        peak_density = max(self.densities) if self.densities else None
        avg_density = (sum(self.densities) / len(self.densities)) if self.densities else None

        return {
            "frames": n,
            "peak_count": peak_count,
            "avg_count": round(avg_count, 2),
            "peak_density": round(peak_density, 3) if peak_density is not None else None,
            "avg_density": round(avg_density, 3) if avg_density is not None else None,
            "peak_level": {
                "key": self.peak_level,
                "zh": _LEVEL_ZH[self.peak_level],
                "alert": _ALERT_FROM[self.peak_level],
            },
            "alert_frames": self.alert_frames,
            "alert_ratio": round(self.alert_frames / n, 4) if n else 0.0,
            "calibrated": bool(self.densities),
            "thresholds": {"watch": watch, "warn": warn, "danger": danger},
            # 供前端画曲线（已抽稀）
            "series_count": self._thin(self.counts),
            "series_density": [round(x, 3) for x in self._thin(self.densities)],
        }
