"""带时间戳的北斗日志：解析 → 时间索引轨迹 → 按时刻查询。

适用场景（**离线对齐**）
------------------------
模块接在单片机上，MCU 把 NMEA 与时间一起写进存储卡；事后把日志和视频一起导入，
按时间戳对齐、拖动时间轴查看某一时刻的定位。这与"串口实时读数"是两种模式：

* 实时：只关心**最新一帧**（:class:`~server.services.gnss.GnssService` 负责）。
* 离线：需要**整段轨迹**，并能按任意时刻查询（本模块负责）。

日志格式容错
------------
MCU 写出来的日志极少是"纯 NMEA"，通常是"时间戳 + 逗号 + 原始语句"。
常见的几种时间戳写法都支持，**自动识别**，不需要改代码：

===========================  ==================================================
写法                          例子
===========================  ==================================================
ISO（空格或 T 分隔）           ``2026-09-17 12:00:00.123,$GNGGA,...``
ISO + Z                       ``2026-09-17T12:00:00.123Z,$GNGGA,...``
Unix 秒                        ``1758100000.123,$GNGGA,...``
Unix 毫秒（13 位）             ``1758100000123,$GNGGA,...``
当天时刻                       ``12:00:00.123,$GNGGA,...``
上电毫秒（相对时间）            ``123456,$GNGGA,...``
无时间戳                       ``$GNGGA,...``
===========================  ==================================================

分隔符支持 逗号 / 制表符 / 空格 / 竖线；时间戳在**行首**。

时间基准：必须说清的一件事
--------------------------
``Unix 秒/毫秒``、``ISO`` 是**绝对时间**，可以和视频对齐。
``上电毫秒``、``当天时刻`` 只有相对或当天信息，**缺少日期或起点**，
必须由外部给一个基准（:attr:`NmeaTrack.offset_s` 或 :meth:`NmeaTrack.rebase`），
否则只能做"相对回放"。这不是实现偷懒，而是信息本身就不够——
本模块会如实把 ``absolute`` 标成 False，界面上必须显示出来。
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree as ET

from .nmea import NmeaAccumulator, parse_line

log = logging.getLogger("server.gnss_track")

# 时间戳与语句之间的分隔符
_SEP = re.compile(r"^([^,\t|]+?)\s*[,;\t|]\s*(?=\$)")
# 行首就是时间戳、后面直接跟语句（无分隔符）
_LEAD = re.compile(r"^([0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9:.]+Z?|[0-9]{10}(?:\.[0-9]+)?|[0-9]{13})\s+(\$.*)$")

_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?(Z)?$")
_HMS_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$")

# 两种历元内语句：同一张表也用于判断"这一行是否值得建一个轨迹点"
_EPOCH_KINDS = ("GGA", "RMC")

# 注释行前缀。MCU 日志开头常写几行说明，把它们算进"没解析成定位点"
# 会吓到用户（自己写的文件头被报成错误），真正的问题反而被淹掉。
_COMMENT_PREFIX = ("#", "//", ";")


@dataclass
class StampResult:
    """一行的解析结果。"""

    epoch_s: float | None      # 绝对时间（Unix 秒）；None = 无法确定
    nmea: str                  # 原始 NMEA 语句
    kind: str                  # 时间戳写法：iso | unix | unix_ms | hms | ticks | none
    absolute: bool             # epoch_s 是否是绝对时间（可与视频对齐）
    # 设备计时器的原始读数（ticks）。虽然不知道起点，但**间隔是真实的**，
    # 丢掉它就只能按"每个点 1 秒"瞎铺，回放节奏跟现场完全对不上。
    value: float | None = None


def parse_stamped_line(line: str, day_anchor: datetime | None = None) -> StampResult | None:
    """拆出「时间戳」与「NMEA 语句」。

    :param day_anchor: 只有 ``当天时刻`` 时用来补日期；默认取今天（UTC）。
    :returns: 解析结果；``None`` = 这一行不是可用的定位记录。
    """
    if not line:
        return None
    s = line.strip()
    if not s:
        return None

    # 纯 NMEA（没有时间戳）
    if s.startswith("$"):
        sent = parse_line(s)
        if sent is None:
            return None
        return StampResult(None, s, "none", False)

    head: str | None = None
    rest: str | None = None

    m = _SEP.match(s)
    if m:
        head, rest = m.group(1).strip(), s[m.end():].strip()
    else:
        m = _LEAD.match(s)
        if m:
            head, rest = m.group(1).strip(), m.group(2).strip()

    if head is None or not rest or not rest.startswith("$"):
        return None
    if parse_line(rest) is None:
        return None

    stamp = _parse_stamp(head, day_anchor)
    if stamp is None:
        return None
    epoch_s, kind, absolute, value = stamp
    return StampResult(epoch_s, rest, kind, absolute, value)


def _parse_stamp(head: str, day_anchor: datetime | None):
    """→ ``(epoch_s, kind, absolute, value)``；``None`` = 认不出。"""
    m = _ISO_RE.match(head)
    if m:
        y, mo, d, hh, mi, ss = (int(m.group(i)) for i in range(1, 7))
        frac = m.group(7) or ""
        micro = int(frac.ljust(6, "0")) if frac else 0
        try:
            dt = datetime(y, mo, d, hh, mi, ss, micro, tzinfo=timezone.utc)
        except ValueError:
            return None
        return dt.timestamp(), "iso", True, None

    m = _HMS_RE.match(head)
    if m:
        hh, mi, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frac = m.group(4) or ""
        micro = int(frac.ljust(6, "0")) if frac else 0
        if not (0 <= hh < 24 and 0 <= mi < 60 and 0 <= ss < 60):
            return None
        base = day_anchor or datetime.now(timezone.utc)
        dt = base.replace(hour=hh, minute=mi, second=ss, microsecond=micro)
        return dt.timestamp(), "hms", False, None   # 只有当天时刻，缺日期 → 非绝对

    if head.isdigit() or _looks_float(head):
        v = float(head)
        if v >= 1e12:                             # 13 位 → 毫秒
            return v / 1000.0, "unix_ms", True, None
        if v >= 1e9:                              # 10 位 → 秒（2001 年之后）
            return v, "unix", True, None
        # 小的整数：MCU 上电以来的毫秒/秒。**起点未知但间隔真实**，
        # 把原始读数带出去，交给 NmeaTrack 还原相对节奏。
        return None, "ticks", False, v
    return None


def _looks_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _median_gap(points: list[dict]) -> float | None:
    """相邻点间隔的中位数（秒）。用于判断日志是 1 Hz 还是 10 Hz。"""
    if len(points) < 2:
        return None
    gaps = sorted(
        b["t"] - a["t"]
        for a, b in zip(points, points[1:])
        if b.get("t") is not None and a.get("t") is not None and b["t"] > a["t"]
    )
    return round(gaps[len(gaps) // 2], 3) if gaps else None


# 时间戳写法的中文说明。**必须把"是否绝对时间"说清楚** ——
# 用户看到"上电毫秒"不会知道那意味着没法跟视频精确对齐。
STAMP_KIND_ZH: dict[str, str] = {
    "iso": "ISO 日期时间（绝对时间，可与视频对齐）",
    "unix": "Unix 秒（绝对时间，可与视频对齐）",
    "unix_ms": "Unix 毫秒（绝对时间，可与视频对齐）",
    "hms": "当天时刻（缺日期，非绝对时间）",
    "ticks": "设备计时器（起点未知，只有间隔真实）",
    "gpx": "GPX 轨迹文件（绝对时间，可与视频对齐）",
    "none": "无时间戳（只能按行顺序回放）",
}

# GPX 是记录仪写卡的主流格式（SD 卡里就是 .gpx），但它是 XML，
# 一行一个历元的 NMEA 解析逻辑完全认不出来 —— 全部会落到 unrecognized。
# 与其让用户拿 GPX 转换一遍，不如在这里直接认。
_GPX_SNIFF_BYTES = 8192


def looks_like_gpx(path: str) -> bool:
    """嗅探文件头，判断是不是 GPX（只认 <gpx> / <trkpt>，不认泛泛的 <?xml）。

    刻意不把「以 <?xml 开头」当成判据：KML、TCX 也是 XML，认进来却解析不出
    轨迹点，只会多一层误导。
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_GPX_SNIFF_BYTES)
    except OSError:
        return False
    s = head.decode("utf-8", errors="ignore").lower()
    return "<gpx" in s or "<trkpt" in s


def gpx_time_to_epoch(text: str | None) -> float | None:
    """``<time>`` → Unix 秒。支持 ``Z`` / ``±HH:MM`` / 不带时区三种写法。

    GPX 1.1 规定时间是 UTC，但实测设备（含本地写卡的实现）会带本地偏移，
    所以三种都得认；认不出来就返回 ``None``，交给调用方按"无时间戳"处理，
    绝不擅自按某个时区猜一个值出来。
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    if t[-1:] in ("Z", "z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        try:                                    # "2026-09-25 09:12:30" 空格分隔
            dt = datetime.fromisoformat(t.replace(" ", "T", 1))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)    # 无时区按 UTC，与 NMEA 口径一致
    return dt.timestamp()


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点球面距离（米）。GPX 没有速度字段时用它从坐标反算。"""
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """真北方位角 0~360。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


@dataclass
class NmeaTrack:
    """按时间索引的定位轨迹，支持"任意时刻取一帧"。

    对齐用**一个**旋钮：``origin_s`` = "日志时间轴上对应视频第 0 秒的那个时刻"。
    之所以不是"偏移量"，是因为有两种真实情形，用偏移量表达会把它们混在一起：

    1. 日志是绝对时间、视频起始时刻已知 → ``origin_s`` 直接取视频起始的 Unix 秒；
    2. 日志是相对时间、靠一个人工同步点对齐 → ``origin_s`` = 日志起点 + 同步偏差。

    两种情形在数学上都是 ``t_log = origin_s + t_video``，所以一个字段就够。
    """

    points: list[dict] = field(default_factory=list)   # 每项含 t（日志时间）与完整定位帧
    t0: float | None = None
    t1: float | None = None
    origin_s: float | None = None   # 视频第 0 秒对应的日志时刻；None = 尚未对齐
    absolute: bool = True          # 日志时间是否为绝对时间
    stamp_kind: str = "none"       # 主要的时间戳写法
    total_lines: int = 0
    skipped_lines: int = 0
    source: str | None = None

    # ---- 诊断信息：导入时告诉用户"我看懂了什么、哪些没看懂" ----
    # 没有这些，"文件丢进去没反应"就只能靠猜；有了它们，格式不对会立刻暴露。
    stamp_kinds: dict[str, int] = field(default_factory=dict)
    sample_lines: list[str] = field(default_factory=list)     # 头几行原文
    skipped_samples: list[str] = field(default_factory=list)  # 认不出的行原文
    comment_lines: int = 0                                    # 注释 / 空行（不算失败）
    epoch_interval_s: float | None = None                     # 相邻历元间隔中位数

    # ------------------------------------------------------------ 装载
    @classmethod
    def load(cls, path: str) -> NmeaTrack:
        """读入日志，逐历元建点。

        GPX（XML）与 NMEA（逐行文本）走两条不同的分支——前者是记录仪写卡的格式，
        后者是串口/带时间戳日志的格式，硬用一套逻辑只会两边都认不全。
        """
        if looks_like_gpx(path):
            return cls._load_gpx(path)
        return cls._load_nmea(path)

    @classmethod
    def _load_gpx(cls, path: str) -> NmeaTrack:
        """解析 GPX 1.1：``<trkpt lat lon>`` + ``<ele>`` + ``<time>``。

        流式解析（``iterparse`` + ``clear``）——200 MB 的轨迹也不会把内存吃满。
        GPX 里没有卫星数、HDOP、定位质量这些字段，一律留 ``None``：
        **没有的数据不编造**；速度/航向是从坐标算出来的（真实可推，不算编造）。
        """
        track = cls(source=path)
        track.total_lines = _count_lines(path)
        raw: list[dict] = []
        malformed = 0

        try:
            for _event, el in ET.iterparse(path, events=("end",)):
                if el.tag.rsplit("}", 1)[-1] != "trkpt":
                    continue
                try:
                    lat = float(el.get("lat"))
                    lon = float(el.get("lon"))
                except (TypeError, ValueError):
                    malformed += 1
                    el.clear()
                    continue
                ele = None
                when = None
                for ch in el:
                    ctag = ch.tag.rsplit("}", 1)[-1]
                    if ctag == "ele":
                        try:
                            ele = float((ch.text or "").strip())
                        except ValueError:
                            ele = None
                    elif ctag == "time":
                        when = gpx_time_to_epoch(ch.text)
                raw.append({"lat": lat, "lon": lon, "alt": ele, "t": when})
                el.clear()
        except ET.ParseError as exc:
            # 解析失败不能抛出去了事：界面要的是"为什么没读出来"
            log.warning("GPX 解析失败: %s (%s)", path, exc)
            track.skipped_lines = track.total_lines
            if len(track.skipped_samples) < 5:
                track.skipped_samples.append(f"XML 解析失败：{exc}")
            track.stamp_kind = "gpx"
            track.stamp_kinds = {"gpx": 0}
            return track

        # 速度/航向：GPX 没有这两个字段，用相邻点反算（真实可推，不是编造）
        for i, p in enumerate(raw):
            spd = None
            crs = None
            if i + 1 < len(raw):
                nxt = raw[i + 1]
                dt = (nxt["t"] - p["t"]) if (p["t"] is not None and nxt["t"] is not None) else None
                if dt and dt > 0:
                    d = _haversine_m(p["lat"], p["lon"], nxt["lat"], nxt["lon"])
                    spd = round(d / dt * 3.6, 2)
                crs = round(_bearing_deg(p["lat"], p["lon"], nxt["lat"], nxt["lon"]), 1)
            elif i > 0:
                # 末点没有"下一段"，用前一段反推 —— 同样的真实可推，不是编造
                prv = raw[i - 1]
                dt = (p["t"] - prv["t"]) if (p["t"] is not None and prv["t"] is not None) else None
                if dt and dt > 0:
                    d = _haversine_m(prv["lat"], prv["lon"], p["lat"], p["lon"])
                    spd = round(d / dt * 3.6, 2)
                crs = round(_bearing_deg(prv["lat"], prv["lon"], p["lat"], p["lon"]), 1)
            track.points.append({
                "t": p["t"],
                "raw_t": None,
                "lat": p["lat"],
                "lon": p["lon"],
                "alt": p["alt"],
                "geoid_sep": None,
                "fix_quality": None,
                "fix_quality_zh": None,
                "fix_type": None,
                "fix_type_zh": None,
                "usable": True,          # 坐标有效即可用；GPX 不记录定位质量
                "satellites_used": None,
                "satellites_visible": None,
                "hdop": None, "vdop": None, "pdop": None,
                "speed_kn": round(spd / 1.852, 2) if spd is not None else None,
                "speed_kmh": spd,
                "course": crs,
                "utc": utc_from_epoch(p["t"]) if p["t"] is not None else None,
                "mode": None,
                "mode_zh": None,
                "rmc_valid": None,
                "satellites": [],
                "sentences": 1,
                "checksum_errors": 0,
                "no_checksum": 0,
                "unknown_lines": 0,
                "age_s": 0.0,
            })

        has_time = any(p["t"] is not None for p in track.points)
        track.stamp_kind = "gpx"
        track.stamp_kinds = {"gpx": len(track.points)}
        track.absolute = has_time
        track.skipped_lines = malformed
        if not track.points:
            track.sample_lines = _head_lines(path, 3)
        track._reindex()
        track.epoch_interval_s = _median_gap(track.points)
        return track

    @classmethod
    def _load_nmea(cls, path: str) -> NmeaTrack:
        """读入逐行文本日志，按历元建点。"""
        track = cls(source=path)
        acc = NmeaAccumulator()
        epoch_t: float | None = None
        raw_t: float | None = None
        seen_in_epoch: set[str] = set()
        fed_fix = False
        kinds: dict[str, int] = {}

        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                track.total_lines += 1
                line = raw.strip()
                # 注释与空行单独计数，不进 skipped_lines
                if not line or line.startswith(_COMMENT_PREFIX):
                    track.comment_lines += 1
                    continue
                # 留几行原文做诊断：格式没认出来时，用户一眼就能看出差在哪
                if len(track.sample_lines) < 3:
                    track.sample_lines.append(line[:200])
                st = parse_stamped_line(raw)
                if st is None:
                    track.skipped_lines += 1
                    if len(track.skipped_samples) < 5:
                        track.skipped_samples.append(line[:200])
                    continue
                sent = parse_line(st.nmea)
                if sent is None:
                    track.skipped_lines += 1
                    if len(track.skipped_samples) < 5:
                        track.skipped_samples.append(line[:200])
                    continue
                kinds[st.kind] = kinds.get(st.kind, 0) + 1
                if not st.absolute:
                    track.absolute = False

                k = sent.kind
                # 新历元的判据：**GGA 或 RMC 第二次出现**（与 gnss.FileSource 同一套规则）。
                # 关键是必须在 feed 之前收点 —— 否则 snapshot 里已经混进了新历元的数据，
                # 每个点都会"提前"带上下一秒的坐标。
                if k in _EPOCH_KINDS and k in seen_in_epoch:
                    track.points.append({"t": epoch_t, "raw_t": raw_t, **acc.snapshot()})
                    seen_in_epoch = set()

                acc.feed(st.nmea)
                if k in _EPOCH_KINDS:
                    fed_fix = True
                    # 时间戳以 GGA 为准（它是历元里最先出现的定位语句）
                    if k == "GGA" or epoch_t is None:
                        epoch_t, raw_t = st.epoch_s, st.value
                seen_in_epoch.add(k)

        # 收尾：最后一段历元
        if fed_fix:
            track.points.append({"t": epoch_t, "raw_t": raw_t, **acc.snapshot()})

        if kinds:
            track.stamp_kind = max(kinds, key=lambda k: kinds[k])
        track.stamp_kinds = dict(kinds)
        track._reindex()
        track.epoch_interval_s = _median_gap(track.points)
        return track

    def _reindex(self) -> None:
        """把"非绝对时间"的日志铺成一条可拖动的相对时间轴。

        优先用设备计时器的**真实间隔**还原节奏；只有连读数都没有（纯 NMEA 日志）
        才退化为"每点 1 秒"。前者能让回放速度和现场一致，后者只能保证顺序。
        """
        if self.points and any(p["t"] is None for p in self.points):
            raws = [p.get("raw_t") for p in self.points]
            if all(r is not None for r in raws) and len(raws) > 1:
                gaps = [raws[i + 1] - raws[i] for i in range(len(raws) - 1)]
                positive = sorted(g for g in gaps if g > 0)
                med = positive[len(positive) // 2] if positive else 0.0
                # 间隔中位数 ≥100 说明计时器按毫秒走（1 Hz 日志在秒制下间隔是 1）
                unit = 1000.0 if med >= 100 else 1.0
                base = raws[0]
                for p, r in zip(self.points, raws):
                    p["t"] = round((r - base) / unit, 3)
            else:
                for i, p in enumerate(self.points):
                    p["t"] = float(i)
            self.absolute = False
        for p in self.points:
            p.pop("raw_t", None)
        self.points.sort(key=lambda p: p["t"])
        if self.points:
            self.t0 = self.points[0]["t"]
            self.t1 = self.points[-1]["t"]

    # ------------------------------------------------------------ 查询
    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def duration_s(self) -> float:
        if self.t0 is None or self.t1 is None:
            return 0.0
        return round(self.t1 - self.t0, 3)

    def align(self, origin_s: float | None) -> None:
        """设定"视频第 0 秒"对应的日志时刻（Unix 秒，或相对日志的秒数）。

        绝对时间日志 + 已知视频起始时刻：传视频起始的 Unix 秒。
        """
        self.origin_s = None if origin_s is None else float(origin_s)

    def align_by_delta(self, delta_s: float) -> None:
        """相对对齐：以日志起点为基准，再平移 ``delta_s`` 秒。

        没有绝对时间可用时（MCU 写的是上电毫秒），靠一个人工同步点
        （比如画面里闪一下）把两边对上，然后用这里微调。
        """
        if self.t0 is None:
            return
        self.origin_s = self.t0 + float(delta_s)

    @property
    def aligned(self) -> bool:
        return self.origin_s is not None

    def at(self, t: float, *, interpolate: bool = True) -> dict | None:
        """取 ``t``（**日志时间轴**上的秒）对应的定位帧。

        默认在相邻两点之间做线性插值 —— 拖动时间轴时要的是连续变化，
        不是一格一格地跳。插值只对连续量（经纬度、海拔、速度、航向）做，
        离散量（定位质量、卫星数）取更近的那一端。
        """
        if not self.points:
            return None
        pts = self.points
        if t <= pts[0]["t"]:
            return dict(pts[0])
        if t >= pts[-1]["t"]:
            return dict(pts[-1])

        lo, hi = 0, len(pts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pts[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid
        a, b = pts[lo], pts[hi]

        if not interpolate or b["t"] <= a["t"]:
            return dict(a)

        k = (t - a["t"]) / (b["t"] - a["t"])
        out = dict(a)
        out["t"] = t
        for key in ("lat", "lon", "alt", "speed_kmh", "course", "hdop", "vdop", "pdop"):
            va, vb = a.get(key), b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                out[key] = round(va + (vb - va) * k, 7)
        out["interpolated"] = True
        return out

    def at_video_time(self, t_video: float, *, interpolate: bool = True) -> dict | None:
        """按**视频时间轴**取帧。

        尚未对齐时退化为"视频 0 秒 = 日志起点"，并把结果标上 ``aligned=False`` ——
        让界面能提示"当前是按起点对齐的，可能需要校正"，而不是假装已经对齐。
        """
        origin = self.origin_s if self.origin_s is not None else self.t0
        if origin is None:
            return None
        out = self.at(origin + t_video, interpolate=interpolate)
        if out is not None:
            out["aligned"] = self.aligned
        return out

    # ------------------------------------------------------------ 摘要
    def summary(self) -> dict:
        if not self.points:
            return {
                "count": 0, "duration_s": 0.0, "absolute": self.absolute,
                "aligned": self.aligned, "origin_s": self.origin_s,
                "stamp_kind": self.stamp_kind, "source": self.source,
                "total_lines": self.total_lines, "skipped_lines": self.skipped_lines,
                "message": "日志里没有可用的定位点。",
            }
        lat = [p["lat"] for p in self.points if p.get("lat") is not None]
        lon = [p["lon"] for p in self.points if p.get("lon") is not None]
        usable = sum(1 for p in self.points if p.get("usable"))
        msg = None
        if not self.absolute:
            msg = ("日志时间是相对时间（无日期/无起点），只能做相对回放；"
                   "要精确对齐视频，需要在 MCU 侧写入绝对时间（Unix 或 ISO）。")
        return {
            "count": self.count,
            "duration_s": self.duration_s,
            "t0": self.t0,
            "t1": self.t1,
            "origin_s": self.origin_s,
            "aligned": self.aligned,
            # 相对日志起点的平移量，便于界面直接显示"整体挪了 N 秒"
            "shift_s": (round(self.origin_s - self.t0, 3)
                        if self.origin_s is not None and self.t0 is not None else 0.0),
            "absolute": self.absolute,
            "stamp_kind": self.stamp_kind,
            "source": self.source,
            "total_lines": self.total_lines,
            "skipped_lines": self.skipped_lines,
            "usable_points": usable,
            "bbox": {
                "lat_min": min(lat) if lat else None,
                "lat_max": max(lat) if lat else None,
                "lon_min": min(lon) if lon else None,
                "lon_max": max(lon) if lon else None,
            },
            "message": msg,
        }

    # ------------------------------------------------------------ 诊断
    def clock_offset_s(self) -> float | None:
        """「MCU 写的时间戳」与「语句里的 UTC」之差的中位数（秒）。

        这是**对齐成败的关键量**，而且不查就一定会踩：
        单片机上的 RTC 常按本地时区走，而 NMEA 语句里的 UTC 是卫星给的。
        两者差整整 8 小时时，直接拿日志时间戳去对视频（文件名多半也是本地时间）
        反而能对上；可一旦拿语句 UTC 去对，就会整体偏 8 小时。

        取中位数而不是首值：首条语句可能还没定位、UTC 字段是空的。
        """
        if not self.absolute:
            return None
        diffs: list[float] = []
        for p in self.points:
            utc = p.get("utc")
            t = p.get("t")
            if utc is None or t is None:
                continue
            epoch = epoch_from_utc(utc)
            if epoch is None:
                continue
            diffs.append(t - epoch)
        if not diffs:
            return None
        diffs.sort()
        return round(diffs[len(diffs) // 2], 3)

    def diagnose(self) -> dict:
        """导入回执：告诉用户"这份日志我看懂了什么、哪些没看懂"。

        存在的理由：MCU 写卡的格式事先无法穷举，与其让用户反复猜"为什么没反应"，
        不如把识别结果、认不出的行、以及该怎么办一次性摆在界面上。
        """
        kind_zh = STAMP_KIND_ZH.get(self.stamp_kind, self.stamp_kind)
        hints: list[str] = []

        if self.count == 0:
            if self.stamp_kind == "gpx":
                hints.append("这份 GPX 里没有解析出任何轨迹点。请确认文件里确实有 "
                             "<trk><trkseg><trkpt lat=... lon=...> 结构，且坐标是合法数字。")
            else:
                hints.append("没有解析出任何定位点。请对照下面「认不出的行」，确认日志里"
                             "确实包含 $xxGGA / $xxRMC 这类 NMEA 语句。")
        if self.skipped_lines:
            hints.append(f"有 {self.skipped_lines} 行既不是 NMEA 语句、也认不出时间戳"
                         "（下面列出前几条）。注释和空行已单独计数，不在此列。")
        if self.comment_lines:
            hints.append(f"另有 {self.comment_lines} 行注释或空行，已跳过，属正常。")
        if self.stamp_kind == "none":
            hints.append("日志里没有时间戳。可以正常按顺序回放，但**无法与视频精确对齐**——"
                         "建议在 MCU 侧每条语句前写入时间戳。")
        elif self.stamp_kind == "gpx" and not self.absolute:
            hints.append("这份 GPX 的轨迹点没有 <time> 时间戳，已按点的先后顺序铺开回放，"
                         "间隔不保证等于真实采样间隔。要与视频精确对齐，"
                         "需要记录仪在每个轨迹点上写入时间。")
        elif not self.absolute:
            hints.append("日志时间是相对时间（缺日期或起点），间隔是真实的，但起点要靠人工同步。"
                         "要做到打开即对齐，建议 MCU 写 Unix 秒或 ISO 时间。")

        off = self.clock_offset_s()
        if off is not None and abs(off) >= 1.0:
            hours = off / 3600.0
            if abs(hours - round(hours)) < 0.01 and abs(round(hours)) <= 14:
                hints.append(
                    f"MCU 写的时间戳比语句里的 UTC 快 {hours:.0f} 小时"
                    f"（整点偏移，多半是 MCU 按本地时区在走）。"
                    "对齐时要明确以**哪一边**为基准：视频时间通常也是本地时间，"
                    "那就用日志时间戳；要用 UTC 对齐则需整体减去这个偏移。"
                )
            else:
                hints.append(
                    f"MCU 写的时间戳与语句里的 UTC 差 {off:.1f} 秒（非整点偏移），"
                    "可能是 MCU 时钟没校准或写入有延迟，对齐时需要手工微调。"
                )

        return {
            "ok": self.count > 0,
            "stamp_kind": self.stamp_kind,
            "stamp_kind_zh": kind_zh,
            "stamp_kinds": dict(self.stamp_kinds),
            "absolute": self.absolute,
            "epoch_interval_s": self.epoch_interval_s,
            "clock_offset_s": off,
            "count": self.count,
            "duration_s": self.duration_s,
            "total_lines": self.total_lines,
            "skipped_lines": self.skipped_lines,
            "comment_lines": self.comment_lines,
            "sample_lines": list(self.sample_lines),
            "skipped_samples": list(self.skipped_samples),
            "first_utc": utc_from_epoch(self.t0) if self.absolute else None,
            "last_utc": utc_from_epoch(self.t1) if self.absolute else None,
            "hints": hints,
        }

    def polyline(self, max_points: int = 400) -> list[dict]:
        """抽稀后的轨迹点，供界面画折线。

        抽稀是**等间隔取样**而不是简单切片：切片会让轨迹在开头挤成一团、
        结尾稀疏，看起来像"速度变了"。
        """
        pts = [p for p in self.points if p.get("lat") is not None and p.get("lon") is not None]
        if not pts:
            return []
        if len(pts) <= max_points:
            return [{"t": p["t"], "lat": p["lat"], "lon": p["lon"]} for p in pts]
        step = len(pts) / float(max_points)
        out = []
        i = 0.0
        while int(i) < len(pts):
            p = pts[int(i)]
            out.append({"t": p["t"], "lat": p["lat"], "lon": p["lon"]})
            i += step
        last = pts[-1]
        if out[-1]["t"] != last["t"]:
            out.append({"t": last["t"], "lat": last["lat"], "lon": last["lon"]})
        return out

    def scrub_points(self, max_points: int = 12000) -> tuple[list[dict], int, list[list[dict]]]:
        """供**前端本地插值**用的精简序列 → ``(points, stride, sats)``。

        为什么不逐帧问后端：拖动时间轴时一秒能触发 60 次查询，实测会把浏览器的
        连接池拖死（请求发不出去、页面看起来"卡住"）。把序列一次发给前端、
        在本地插值，拖动就是纯计算，既即时又不产生任何请求。

        卫星明细（GSV）**不是丢掉，而是去重后单独放一张表**，每个点只存一个下标：

        * 相邻历元的 GSV 列表几乎完全一致，逐点各存一份会让载荷膨胀几十倍
          （1 Hz 一小时 = 3600 份 × 十几颗 × 6 个字段）；
        * 去重之后同一份只留一条，实测一天 1 Hz 的日志也只有几百条；
        * 星空图因此也能完全在本地画 —— 否则要么拖动时发请求（就是上面那个坑），
          要么星空图停在几秒前的快照上（"刷新一半比不刷更糟"）。

        ``max_points`` 默认 12000：这个量级下 JSON 在本地解析是毫秒级，
        而抽稀后的间距对"直线插值"仍然足够细（20 分钟日志间距约 0.1 s）。
        """
        keys = ("t", "lat", "lon", "alt", "fix_quality", "fix_quality_zh",
                "satellites_used", "satellites_visible", "hdop", "vdop", "pdop",
                "speed_kmh", "course", "utc", "usable", "mode_zh")
        pts = self.points
        if not pts:
            return [], 1, []

        sats: list[list[dict]] = []
        seen: dict[str, int] = {}

        def _intern(arr) -> int | None:
            """卫星列表 → 表内下标；空列表记 None（"这个历元没收到 GSV"是有效信息）。"""
            if not arr:
                return None
            key = json.dumps(arr, sort_keys=True, separators=(",", ":"))
            i = seen.get(key)
            if i is None:
                i = len(sats)
                seen[key] = i
                sats.append(arr)
            return i

        def _row(p: dict) -> dict:
            row = {k: p.get(k) for k in keys}
            row["sat"] = _intern(p.get("satellites"))
            return row

        stride = max(1, -(-len(pts) // max_points))     # 向上取整
        out = [_row(pts[i]) for i in range(0, len(pts), stride)]
        # 末点必须保留：否则时间轴末尾一段没有数据，拖到底会突然没读数
        if out and out[-1].get("t") != pts[-1].get("t"):
            out.append(_row(pts[-1]))
        return out, stride, sats


def utc_from_epoch(epoch_s: float | None) -> str | None:
    """Unix 秒 → ISO(UTC)，供界面显示。"""
    if epoch_s is None:
        return None
    try:
        return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def epoch_from_utc(text: str) -> float | None:
    """ISO(UTC) → Unix 秒。用于把"视频起始时刻"这类用户输入转成时间轴基准。"""
    if not text:
        return None
    s = text.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def shift_day(anchor: datetime, seconds: float) -> datetime:
    """按秒平移一个基准时刻（测试与"当天时刻"补日期用）。"""
    return anchor + timedelta(seconds=seconds)


def _count_lines(path: str) -> int:
    """数行数。给 GPX 用——XML 里没有"历元行"的概念，只能按物理行报给用户。

    分块读，不整文件塞进内存（上传上限是 200 MB）。
    """
    n = 0
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                n += chunk.count(b"\n")
    except OSError:
        return 0
    return n + 1


def _head_lines(path: str, n: int) -> list[str]:
    """取前 n 行原文，供诊断界面展示"我读到的到底是什么"。"""
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(n):
                ln = fh.readline()
                if not ln:
                    break
                out.append(ln.strip()[:200])
    except OSError:
        return []
    return out
