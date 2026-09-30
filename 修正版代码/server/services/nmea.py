"""NMEA 0183 解析器（纯标准库，零第三方依赖）。

为什么自己写而不用 pynmea2
--------------------------
1. 只依赖标准库，没有安装失败的可能；串口模块本身（pyserial）已经是可选依赖，
   再把解析也压成可选依赖，会让"北斗"这条链路在评审环境里说不清楚。
2. 需要把 GGA / RMC / GSA / GSV **四类语句合并成一个定位帧**（含定位质量、
   DOP、卫星方位角高度角）。现成库大多只做"单句 → 对象"的翻译，合并逻辑仍要自己写，
   那还不如把解析也一起摊开，逻辑全在明面上。
3. 论文里"北斗"唯一可被验证的技术点就是这里，用黑盒库反而不好解释。

支持的语句
----------
==================  ====================================================
语句                取用的字段
==================  ====================================================
``GGA``             定位质量、已用卫星数、HDOP、经纬度、海拔、大地水准面差距、UTC
``RMC``             有效性标志、速度（节）、航向、日期、模式指示符
``GSA``             DOP（PDOP/HDOP/VDOP）、定位类型（1=未定位 / 2=二维 / 3=三维）
``GSV``             可见卫星的 PRN、方位角、高度角、信噪比
==================  ====================================================

对多星座的处理
--------------
北斗模块通常同时输出 GPS 与北斗，语句前缀会是 ``GP`` / ``GN`` / ``BD`` / ``GB``。
本模块**不区分星座**，按前缀后两位识别语句类型，把各星座的 GSV 卫星合并进同一张星空图，
但在每颗卫星上保留 ``talker`` 字段，便于前端区分来源。

校验和
------
``*`` 之后的两字节十六进制是 ``$`` 与 ``*`` 之间所有字符的异或。
**有 ``*`` 就校验，不通过直接丢弃**（这是防串口噪声产生假坐标的关键）；
没有 ``*`` 的（部分日志工具会剥掉）仍接受，但会记入 ``no_checksum`` 计数，
便于事后判断"这批数据是不是被处理过"。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("server.nmea")

# 定位质量（GGA 第 6 字段）→ 中文。0 与 8 都不是可用定位。
FIX_QUALITY_ZH: dict[int, str] = {
    0: "无定位",
    1: "单点定位",
    2: "差分定位",
    3: "PPS 定位",
    4: "RTK 固定解",
    5: "RTK 浮点解",
    6: "航位推算",
    7: "人工输入",
    8: "模拟输出",
}

# GSA 第 3 字段：定位类型
FIX_TYPE_ZH: dict[int, str] = {1: "未定位", 2: "二维定位", 3: "三维定位"}

# RMC 第 13 字段（NMEA 2.3+）模式指示符
MODE_ZH: dict[str, str] = {
    "N": "无定位",
    "A": "自主定位",
    "D": "差分定位",
    "F": "RTK 浮点解",
    "R": "RTK 固定解",
    "E": "航位推算",
    "M": "人工输入",
    "S": "模拟输出",
}

# 只有这些质量值算"可用定位"（6=航位推算、7=人工、8=模拟都不算真实卫星定位）
USABLE_FIX_QUALITY = frozenset({1, 2, 3, 4, 5})

_KINDS = ("GGA", "RMC", "GSA", "GSV")


@dataclass(frozen=True)
class Sentence:
    """一条解析成功的 NMEA 语句。"""

    talker: str          # 星座标识：GP / GN / BD / GL ...
    kind: str            # GGA / RMC / GSA / GSV
    fields: tuple[str, ...]
    raw: str


def checksum_of(payload: str) -> str:
    """``$`` 与 ``*`` 之间内容的异或校验和，返回两字节大写十六进制。"""
    c = 0
    for ch in payload:
        c ^= ord(ch)
    return f"{c:02X}"


def parse_line(line: str) -> Sentence | None:
    """解析单行 NMEA。**不做校验和判断**（由 :func:`verify` 负责），只做结构解析。

    返回 ``None`` 表示：不是本模块关心的语句（含 ``$P...`` 私有语句、空行、注释）。
    """
    if not line:
        return None
    s = line.strip()
    if not s.startswith("$"):
        return None
    body = s[1:]
    star = body.find("*")
    if star >= 0:
        body = body[:star]
    parts = body.split(",")
    # 只有地址没有字段（如 ``$GNGGA``）的句子永远带不了数据，直接不认，
    # 免得下游拿到一个"看起来解析成功、其实全空"的对象。
    if len(parts) < 2:
        return None
    addr = parts[0].upper()
    # 地址形如 GNGGA / GPGSV / BDGSA；长度不足 5 或非字母直接跳过
    if len(addr) < 5 or not addr.isalpha():
        return None
    talker, kind = addr[:2], addr[2:5]
    if kind not in _KINDS:
        return None
    return Sentence(talker=talker, kind=kind, fields=tuple(parts[1:]), raw=s)


def verify(sentence: Sentence) -> bool | None:
    """校验和检查。返回 ``True`` 通过 / ``False`` 不通过 / ``None`` 语句里没有校验和。"""
    star = sentence.raw.find("*")
    if star < 0:
        return None
    got = sentence.raw[star + 1:].strip()[:2].upper()
    if len(got) != 2:
        return None
    return got == checksum_of(sentence.raw[1:star])


def _f(v: str) -> float | None:
    """空字段/非法数字一律给 ``None`` —— 不给 0 那种会被误读成"真的是 0"的假值。"""
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(v: str) -> int | None:
    f = _f(v)
    return None if f is None else int(f)


def _dm_to_deg(v: str) -> float | None:
    """``ddmm.mmmm`` / ``dddmm.mmmm`` → 十进制度。

    不靠字段长度猜，直接按"整数部分去掉末两位即为度"来切，
    纬度（2 位度）与经度（3 位度）共用一套逻辑。
    """
    f = _f(v)
    if f is None:
        return None
    deg = int(f // 100)
    return deg + (f - deg * 100) / 60.0


def _lat_lon(value: str, hemi: str) -> float | None:
    d = _dm_to_deg(value)
    if d is None:
        return None
    h = (hemi or "").strip().upper()
    if h == "S":
        return -d
    if h == "W":
        return -d
    return d


def _hhmmss_to_iso(t: str, date_ddmmyy: str | None) -> str | None:
    """UTC 时间 ``hhmmss.sss`` (+ RMC 的 ``ddmmyy``) → ``YYYY-MM-DDTHH:MM:SSZ``。

    只有时间没有日期时给 ``HH:MM:SSZ`` 形式（GGA 单句就属于这种情况）。
    """
    f = _f(t)
    if f is None:
        return None
    try:
        hh = int(f // 10000)
        mm = int((f - hh * 10000) // 100)
        ss = f - hh * 10000 - mm * 100
        if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
            return None
        hms = f"{hh:02d}:{mm:02d}:{int(ss):02d}Z"
        if date_ddmmyy and len(date_ddmmyy) == 6 and date_ddmmyy.isdigit():
            dd, mo, yy = int(date_ddmmyy[0:2]), int(date_ddmmyy[2:4]), int(date_ddmmyy[4:6])
            if 1 <= dd <= 31 and 1 <= mo <= 12:
                # 两位年份要过世纪折点：70 以下算 20xx，其余算 19xx。
                # 直接写 ``20{yy}`` 会把公开样例里的 230394（1994-03-23）算成 2094 年。
                year = 2000 + yy if yy < 70 else 1900 + yy
                return f"{year:04d}-{mo:02d}-{dd:02d}T{hms}"
        return hms
    except Exception:  # pragma: no cover - 上面已做范围判断，这里只是兜底
        return None


@dataclass
class NmeaAccumulator:
    """把连续多句 NMEA 合并成"最新定位帧"。

    设计上**只记最新状态**，不做历史缓存——历史由视频任务自己按帧记录，
    这里再存一份只会带来两处不一致的风险。

    计数器的意义：串口噪声、接线不良、模块未定位，都会表现为"语句解析失败率升高"，
    没有计数器就只能看到"没数据"，排查时无从下手。
    """

    date: str | None = None          # RMC 给的 ddmmyy，用于补全 GGA 的时间
    sentences: int = 0               # 累计解析成功的语句数
    checksum_errors: int = 0
    no_checksum: int = 0
    unknown: int = 0                 # 结构不合法 / 非目标语句

    _gga: dict = field(default_factory=dict)
    _rmc: dict = field(default_factory=dict)
    _gsa: dict = field(default_factory=dict)
    _gsv: dict = field(default_factory=dict)     # talker -> {"total": n, "in_view": n, "sats": [...]}
    _utc: str | None = None
    _last_fix_monotonic: float | None = None     # 最近一次"含有效位置"的语句到达时刻

    # ------------------------------------------------------------ 喂数据
    def feed(self, line: str) -> Sentence | None:
        """喂入一行。返回解析成功的语句，否则 ``None``。"""
        sent = parse_line(line)
        if sent is None:
            self.unknown += 1
            return None
        ok = verify(sent)
        if ok is False:
            self.checksum_errors += 1
            return None
        if ok is None:
            self.no_checksum += 1

        self.sentences += 1
        if sent.kind == "GGA":
            self._on_gga(sent)
        elif sent.kind == "RMC":
            self._on_rmc(sent)
        elif sent.kind == "GSA":
            self._on_gsa(sent)
        else:
            self._on_gsv(sent)
        return sent

    def feed_many(self, lines) -> int:
        """批量喂入（用于文件回放），返回成功解析的语句数。"""
        n = 0
        for ln in lines:
            if self.feed(ln) is not None:
                n += 1
        return n

    # ------------------------------------------------------------ 各语句
    def _on_gga(self, s: Sentence) -> None:
        f = s.fields
        if len(f) < 10:
            return
        self._gga = {
            "utc": _hhmmss_to_iso(f[0], self.date),
            "lat": _lat_lon(f[1], f[2]),
            "lon": _lat_lon(f[3], f[4]),
            "fix_quality": _i(f[5]),
            "satellites_used": _i(f[6]),
            "hdop": _f(f[7]),
            "alt": _f(f[8]),
            "geoid_sep": _f(f[10]) if len(f) > 10 else None,
            "dgps_age": _f(f[12]) if len(f) > 12 else None,
        }
        if self._gga.get("utc"):
            self._utc = self._gga["utc"]
        if self._gga.get("lat") is not None and self._gga.get("lon") is not None:
            self._last_fix_monotonic = time.monotonic()

    def _on_rmc(self, s: Sentence) -> None:
        f = s.fields
        if len(f) < 9:
            return
        if len(f) > 8 and f[8]:
            self.date = f[8].strip() or self.date
        kn = _f(f[6])
        self._rmc = {
            "valid": (f[1].strip().upper() == "A") if f[1] is not None else None,
            "utc": _hhmmss_to_iso(f[0], self.date),
            "lat": _lat_lon(f[2], f[3]),
            "lon": _lat_lon(f[4], f[5]),
            "speed_kn": kn,
            # 1 节 = 1.852 km/h；同时给两种单位，避免前端各算一遍算错
            "speed_kmh": round(kn * 1.852, 2) if kn is not None else None,
            "course": _f(f[7]),
            "mag_var": _f(f[9]) if len(f) > 9 else None,
            "mag_var_ew": (f[10].strip().upper() or None) if len(f) > 10 else None,
            "mode": (f[11].strip().upper() or None) if len(f) > 11 else None,
        }
        if self._rmc.get("utc"):
            self._utc = self._rmc["utc"]
        if self._rmc.get("valid") and self._rmc.get("lat") is not None:
            self._last_fix_monotonic = time.monotonic()

    def _on_gsa(self, s: Sentence) -> None:
        f = s.fields
        if len(f) < 3:
            return
        used = []
        for v in f[2:14]:
            v = v.strip()
            if v:
                try:
                    used.append(int(v))
                except ValueError:
                    pass
        self._gsa = {
            "mode": (f[0].strip().upper() or None),
            "fix_type": _i(f[1]),
            "used_prns": used,
            "pdop": _f(f[14]) if len(f) > 14 else None,
            "hdop": _f(f[15]) if len(f) > 15 else None,
            "vdop": _f(f[16]) if len(f) > 16 else None,
        }

    def _on_gsv(self, s: Sentence) -> None:
        f = s.fields
        # GSV 前 3 个字段是「总句数 / 本句序号 / 可见卫星数」，卫星从**下标 3** 开始，
        # 每 4 个字段一颗（PRN / 高度角 / 方位角 / 信噪比）。
        # 这里从 4 开始遍历过一次，结果整排卫星错位一格 —— 单测锁死了这个下标。
        if len(f) < 3:
            return
        total = _i(f[0]) or 1
        num = _i(f[1]) or 1
        in_view = _i(f[2])
        slot = self._gsv.setdefault(s.talker, {"total": total, "in_view": None, "sats": []})
        slot["total"] = total
        if in_view is not None:
            slot["in_view"] = in_view
        # 第 1 句开头 = 新一轮，先清空该星座上一轮的卫星，否则会越积越多
        if num <= 1:
            slot["sats"] = []
        for i in range(3, len(f), 4):
            grp = f[i:i + 4]
            if len(grp) < 4 or not grp[0].strip():
                continue
            snr = _f(grp[3])
            slot["sats"].append({
                "prn": grp[0].strip(),
                "talker": s.talker,
                "elevation": _f(grp[1]),
                "azimuth": _f(grp[2]),
                "snr": None if snr is None or snr <= 0 else snr,
            })

    # ------------------------------------------------------------ 取结果
    def satellites(self) -> list[dict]:
        """各星座 GSV 卫星合并（同一 PRN 不同星座各自保留）。"""
        out: list[dict] = []
        for talker in sorted(self._gsv):
            out.extend(self._gsv[talker]["sats"])
        return out

    def visible_satellites(self) -> int | None:
        """可见卫星总数：优先用 GSV 报的 in_view，缺失时退化为实际收到的卫星条数。"""
        vals = [v["in_view"] for v in self._gsv.values() if v.get("in_view")]
        if vals:
            return sum(vals)
        n = len(self.satellites())
        return n or None

    def age_s(self, now: float | None = None) -> float | None:
        """距最近一次有效定位的秒数。``None`` = 从未拿到过有效定位。"""
        if self._last_fix_monotonic is None:
            return None
        now = time.monotonic() if now is None else now
        return round(max(0.0, now - self._last_fix_monotonic), 2)

    def snapshot(self, now: float | None = None) -> dict:
        """合并成一个定位帧。位置优先取 GGA（海拔与质量只在 GGA 里）。"""
        gga, rmc, gsa = self._gga, self._rmc, self._gsa

        lat = gga.get("lat")
        lon = gga.get("lon")
        if lat is None or lon is None:
            lat, lon = rmc.get("lat"), rmc.get("lon")

        fq = gga.get("fix_quality")
        if fq is None and rmc.get("mode"):
            fq = {"R": 4, "F": 5, "D": 2, "A": 1, "N": 0, "E": 6, "S": 8, "M": 7}.get(rmc["mode"])

        ft = gsa.get("fix_type")
        return {
            "lat": lat,
            "lon": lon,
            "alt": gga.get("alt"),
            "geoid_sep": gga.get("geoid_sep"),
            "fix_quality": fq,
            "fix_quality_zh": FIX_QUALITY_ZH.get(fq) if fq is not None else None,
            "fix_type": ft,
            "fix_type_zh": FIX_TYPE_ZH.get(ft) if ft is not None else None,
            "usable": bool(fq in USABLE_FIX_QUALITY and lat is not None and lon is not None),
            "satellites_used": gga.get("satellites_used"),
            "satellites_visible": self.visible_satellites(),
            "hdop": gga.get("hdop") if gga.get("hdop") is not None else gsa.get("hdop"),
            "vdop": gsa.get("vdop"),
            "pdop": gsa.get("pdop"),
            "speed_kn": rmc.get("speed_kn"),
            "speed_kmh": rmc.get("speed_kmh"),
            "course": rmc.get("course"),
            "utc": self._utc,
            "mode": rmc.get("mode"),
            "mode_zh": MODE_ZH.get(rmc["mode"]) if rmc.get("mode") else None,
            "rmc_valid": rmc.get("valid"),
            "satellites": self.satellites(),
            "sentences": self.sentences,
            "checksum_errors": self.checksum_errors,
            "no_checksum": self.no_checksum,
            "unknown_lines": self.unknown,
            "age_s": self.age_s(now),
        }

    def reset(self) -> None:
        """清空全部状态（切换定位源时用）。

        显式逐项清，而不是重跑 ``__init__``：后者能"碰巧"跑对，
        但只要将来有人给某字段加了外部引用，就会留下悬空状态。
        """
        self.date = None
        self.sentences = 0
        self.checksum_errors = 0
        self.no_checksum = 0
        self.unknown = 0
        self._gga = {}
        self._rmc = {}
        self._gsa = {}
        self._gsv = {}
        self._utc = None
        self._last_fix_monotonic = None
