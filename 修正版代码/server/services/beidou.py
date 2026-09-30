"""北斗定位服务：对外提供定位帧，并把像素坐标换算到地理坐标。

分工
----
* :mod:`server.services.gnss` 负责"**拿到**定位"（串口 / 回放 / 模拟）。
* 本模块负责"**用**定位"：把图像里的像素位置换算成经纬度与本地东-北坐标。

换算的唯一基准
--------------
像素→地面只有**一个**参数：地面采样距离 GSD（米/像素），来自
``Settings.geo_gsd``。历史上这里用 ``0.00001 * (w/640)`` 度/像素（≈2.94 m/px @1920），
而前端 ENU 散布用写死的 ``0.021`` m/px —— 同一张图上差约 140 倍，
表格里"经度偏移"和"东(m)"两列互相矛盾。现在两边都取同一个值。

未标定时的诚实做法
------------------
GSD 未标定（``APP_GEO_GSD_M_PER_PX=0``）时**不产出目标经纬度**，
返回 ``(None, None)``，由调用方在界面上说明"未标定"。
不编一个假坐标出来 —— 那比不给更糟。
"""
from __future__ import annotations

import logging
import math

from ..config import Settings
from ..schemas.detection import BeidouInfo
from .gnss import GnssService

log = logging.getLogger("server.beidou")

# 1 度纬度对应的米数（WGS-84 平均子午线弧长）。经度方向要再乘 cos(纬度)。
M_PER_DEG_LAT = 111_320.0


class BeidouService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gnss = GnssService(settings)

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        self.gnss.ensure_started()

    def stop(self) -> None:
        self.gnss.stop()

    def reload(self) -> BeidouInfo:
        """重新挑一次定位源，不重启服务。

        典型场景：服务已经起来了才把北斗模块插上（``auto`` 模式启动时没扫到串口），
        点一下重新扫描即可。注意配置本身来自进程启动时的快照，
        改 ``.env`` 仍需重启才生效。
        """
        self.gnss.stop()
        self.gnss = GnssService(self.settings)
        self.gnss.ensure_started()
        return self.info()

    # ------------------------------------------------------------ 定位帧
    def info(self) -> BeidouInfo:
        """当前定位帧。没有有效定位时 ``lat/lon`` 为 None，不回退到演示坐标。"""
        return self.gnss.info()

    # ------------------------------------------------------------ 换算
    def gsd(self) -> tuple[float, bool, bool]:
        """``(米/像素, 是否已标定, 是否估算值)``。"""
        return self.settings.geo_gsd

    def enu(self, cx: float, cy: float, w: int, h: int) -> dict:
        """像素 → 本地东-北坐标（米），以画面中心为原点。无需定位即可给出。

        这是尺度无关的相对量，用于"目标散布"视图；
        它受拍摄高度影响，**不能当绝对距离用**。
        """
        gsd, calibrated, estimated = self.gsd()
        if gsd <= 0:
            return {"east_m": None, "north_m": None, "calibrated": False, "estimated": False}
        return {
            "east_m": round((cx - w / 2.0) * gsd, 2),
            "north_m": round(-(cy - h / 2.0) * gsd, 2),
            "calibrated": calibrated,
            "estimated": estimated,
        }

    def geo_map(self, cx: float, cy: float, w: int, h: int,
                fix: BeidouInfo | None = None) -> tuple[float | None, float | None]:
        """像素 → ``(经度, 纬度)``。

        以图像中心对应"当前位置"为基准做平面近似。仅在**有有效定位**且
        **GSD 可用**时才给结果，否则返回 ``(None, None)``。

        :param fix: 已取好的定位帧。一张图有几百个目标时，
            每个都去取一次定位帧（带锁 + 重建卫星列表）纯属浪费，
            由调用方取一次传进来即可。
        """
        if fix is None:
            fix = self.gnss.info()
        if not fix.usable or fix.lat is None or fix.lon is None:
            return None, None
        gsd, _calibrated, _estimated = self.gsd()
        if gsd <= 0:
            return None, None

        d = self.enu(cx, cy, w, h)
        dlat = d["north_m"] / M_PER_DEG_LAT
        cos_lat = math.cos(math.radians(fix.lat))
        # 极区附近 cos→0 会把经度放大到无意义，直接不给
        if cos_lat < 1e-6:
            return None, None
        dlon = d["east_m"] / (M_PER_DEG_LAT * cos_lat)
        # 保留 7 位小数（≈1 cm）：6 位小数的量化误差约 0.1 m，
        # 而 GSD 取 0.021 m/px 时一个像素才 2 cm —— 四舍五入就把分辨率吃掉了。
        return round(fix.lon + dlon, 7), round(fix.lat + dlat, 7)
