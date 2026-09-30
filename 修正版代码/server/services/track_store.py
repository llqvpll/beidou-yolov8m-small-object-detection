"""存储卡日志的导入与查询（离线路线）。

与 :mod:`server.services.gnss` 的分工：
* ``gnss``  —— **实时**：串口/回放文件持续读，只关心最新一帧；
* 本模块 —— **离线**：一次导入整份日志，之后按任意时刻查询。

用户的实际做法是「模块接单片机 → MCU 把 NMEA + 时间写进存储卡 → 事后把日志和视频
一起导入」，所以这条离线路径才是主线，实时串口只是顺带能用的便利功能。
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings
from .gnss_track import NmeaTrack, epoch_from_utc, utc_from_epoch

log = logging.getLogger("server.track")

# 导入的日志允许的扩展名。不做白名单限制，但给个提示用的常见集合。
_COMMON_SUFFIXES = {".txt", ".log", ".nmea", ".csv", ".dat", ".bin", ".gpx"}


class TrackService:
    """进程内保存一份"当前导入的日志轨迹"。

    只保留最近一次导入 —— 界面上同一时刻也只会看一条轨迹，
    留多份反而要用户去选"现在拖的是哪条"。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.track: NmeaTrack | None = None
        self.name: str | None = None
        self.error: str | None = None
        # 抽稀序列缓存。state() 会被前端在导入后立刻调用一次，
        # 而一天 1 Hz 的日志有 8.6 万个点 —— 每次重算内联表纯属浪费。
        # 轨迹对象一换（load/clear）就作废。
        self._scrub: tuple[list[dict], int, list[list[dict]]] | None = None

    # ------------------------------------------------------------ 导入
    def load(self, path: str | Path, name: str | None = None) -> dict:
        """解析日志并返回**导入回执**（含诊断），不抛异常。"""
        p = Path(path)
        self.name = name or p.name
        try:
            track = NmeaTrack.load(str(p))
        except OSError as exc:
            self.track = None
            self._scrub = None
            self.error = f"读不到日志文件：{exc}"
            log.warning("导入轨迹失败: %s", exc)
            return {
                "ok": False, "name": self.name, "error": self.error,
                "diagnose": None, "summary": None,
            }
        self.track = track
        self._scrub = None
        self.error = None
        diag = track.diagnose()
        log.info("导入轨迹 %s：%d 个定位点，时间戳格式 %s，跳过 %d 行",
                 self.name, track.count, track.stamp_kind, track.skipped_lines)
        return {
            "ok": diag["ok"],
            "name": self.name,
            "error": None,
            "diagnose": diag,
            "summary": track.summary(),
        }

    def clear(self) -> None:
        self.track = None
        self.name = None
        self.error = None
        self._scrub = None

    # ------------------------------------------------------------ 查询
    def state(self) -> dict:
        """当前状态：摘要 + 诊断 + 抽稀折线 + **本地插值序列**。界面刷新时一次拿全。

        ``scrub`` / ``scrub_sats`` 是给前端本地插值用的（见
        :meth:`NmeaTrack.scrub_points`）。放进 state 而不是单开一个接口，
        是因为前端在"导入后"和"打开页面时"都只发这一个请求 ——
        少一次往返，也就少一处"序列还没到就拖了时间轴"的空窗。
        """
        if self.track is None:
            return {
                "loaded": False, "name": None, "error": self.error,
                "summary": None, "diagnose": None, "polyline": [],
                "scrub": [], "scrub_stride": 1, "scrub_sats": [],
            }
        if self._scrub is None:
            self._scrub = self.track.scrub_points()
        points, stride, sats = self._scrub
        return {
            "loaded": True,
            "name": self.name,
            "error": self.error,
            "summary": self.track.summary(),
            "diagnose": self.track.diagnose(),
            "polyline": self.track.polyline(),
            "scrub": points,
            "scrub_stride": stride,
            "scrub_sats": sats,
        }

    def at(self, t_video: float, *, interpolate: bool = True) -> dict | None:
        """按**视频时间轴**取定位帧。

        未对齐时 :meth:`NmeaTrack.at_video_time` 会退化为"视频 0 秒 = 日志起点"
        并把结果标上 ``aligned=False``，界面据此提示"当前按起点对齐，可能需要校正"。
        """
        if self.track is None:
            return None
        return self.track.at_video_time(t_video, interpolate=interpolate)

    def at_log_time(self, t_log: float, *, interpolate: bool = True) -> dict | None:
        if self.track is None:
            return None
        return self.track.at(t_log, interpolate=interpolate)

    # ------------------------------------------------------------ 对齐
    def align(
        self,
        *,
        origin_s: float | None = None,
        delta_s: float | None = None,
        utc: str | None = None,
    ) -> dict:
        """设定视频与日志的对齐方式。三种入参**只应给一个**。

        * ``utc``      —— 视频第 0 秒的绝对时刻（如 ``2026-09-17T04:00:00Z``）；
        * ``origin_s`` —— 直接给"视频第 0 秒对应的日志时刻"（Unix 秒或相对秒）；
        * ``delta_s``  —— 相对日志起点整体平移 N 秒（无绝对时间时的人工同步）。

        给了多个时按 ``utc`` > ``origin_s`` > ``delta_s`` 的优先级取用，
        并在返回里说明用了哪一个 —— 静默忽略用户的输入是更糟的选择。
        """
        if self.track is None:
            return {"ok": False, "error": "还没有导入日志。", "summary": None}

        used = None
        if utc:
            epoch = epoch_from_utc(utc)
            if epoch is None:
                return {"ok": False, "error": f"时间格式认不出：{utc}（需要 ISO，如 2026-09-17T04:00:00Z）",
                        "summary": self.track.summary()}
            self.track.align(epoch)
            used = "utc"
        elif origin_s is not None:
            self.track.align(origin_s)
            used = "origin_s"
        elif delta_s is not None:
            self.track.align_by_delta(delta_s)
            used = "delta_s"
        else:
            self.track.align(None)
            used = "reset"

        out = self.track.summary()
        return {"ok": True, "used": used, "summary": out,
                "origin_utc": utc_from_epoch(self.track.origin_s) if self.track.absolute else None}

    # ------------------------------------------------------------ 工具
    @staticmethod
    def known_suffix(name: str | None) -> bool:
        return Path(name or "").suffix.lower() in _COMMON_SUFFIXES
