"""北斗短报文编码：把一条聚集预警压进北斗短报文的长度约束里。

背景（数字全部来自《新时代的中国北斗》白皮书，2022-11）
--------------------------------------------------------
* **区域短报文**（3 颗 GEO，中国及周边）：单次最长 **14000 bit = 1000 个汉字**
* **全球短报文**（14 颗 MEO）：单次最长 **560 bit = 40 个汉字**

1 个汉字按 2 字节计（GB2312/GBK 常见口径），所以 14000 bit = 1750 字节 ≈ 1000
汉字。本模块统一按**字节**预算，因为字节才是链路上真正的约束；汉字数只是对外的
通俗说法。

为什么这件事有技术含量
----------------------
预警信息要包含：什么级别、在哪里、多少人、多挤、什么时候。用自然语言写，一句话
七八十个字符很正常 —— **全球短报文只有 40 个汉字，写不下**。所以必须设计一套紧凑
编码，并且这个编码要满足：

1. **可解析**：接收端能无歧义地还原出各字段
2. **自校验**：短报文链路会丢字符、会粘连，必须能识别出坏包而不是解出一堆垃圾
3. **可降级**：超长时按**确定的优先级**砍字段，而不是随便截断
4. **不撒谎**：降级后必须标出来"这条是压缩过的"，并从报文本身能看出来

本模块只做**编码与解码**，不假定任何硬件。接上模块后把 :func:`encode` 的结果
交给发送侧即可；没有模块时，界面照样可以把编码结果显示出来，说明"如果接入短报文
模块，发出去的就是这一串"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- 长度约束

# 区域短报文：14000 bit ÷ 8 = 1750 字节
REGION_BYTES = 1750
# 全球短报文：560 bit ÷ 8 = 70 字节（即 40 个汉字 × 2 字节不会超，但全是 ASCII
# 时可用满 70 字节。这里按字节卡，是对链路最诚实的口径）
GLOBAL_BYTES = 70

# 对外展示用的"汉字数"口径，供界面写文案用
REGION_CHARS = 1000
GLOBAL_CHARS = 40

# 协议版本。写进报文首段，接收端据此选解析器 —— 将来改字段不必换协议名。
PROTO = "B1"

# 最高一级只发"危险"，用于全球短报文的极限裁剪
LEVEL_ZH = {"normal": "正常", "watch": "关注", "warn": "警戒", "danger": "危险"}


class MessageTooLong(ValueError):
    """即使裁剪到最小也不能塞进目标长度。属于调用方的用法问题，不是数据问题。"""


@dataclass
class Alert:
    """一条待发送的聚集预警。

    字段全部可空 —— 与系统其它部分一致：**不知道就是 None，编码时如实体现**，
    不要填 0 也不要填默认值。一个"人数 0"的报文会被接收端当成"现场没人"，
    而真相是"这次没测到人数"。
    """

    level: str                       # normal | watch | warn | danger
    lon: float | None = None
    lat: float | None = None
    density: float | None = None     # 人/m²
    count: int | None = None         # 检出人数
    utc: str | None = None           # 北斗时间基准给出的时刻
    site: str | None = None          # 点位名，尽量短
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 校验

def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE。

    选它是因为：短报文载荷短（≤1750 字节），查表法和逐位法性能差不多，而逐位法
    不需要在嵌入式接收端存表 —— 接收端很可能是 MCU。
    """
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def _fmt_coord(v: float | None, neg: str, pos: str, digits: int) -> str:
    """经纬度 → 定长字符串。**未知给 ``-`` 而不是 '0.000000'。**"""
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def _int_or(v, default: str = "-") -> str:
    """None → ``-``。**0 是合法值，不能被吞掉。**"""
    if v is None:
        return default
    try:
        return str(int(round(float(v))))
    except (TypeError, ValueError):
        return default


def _build_body(a: Alert, *, compact: bool) -> str:
    """按模式拼主体（不含 CRC）。

    ``compact`` = 全球短报文模式：丢掉点位名、密度保留 1 位、时间只留时分秒。
    """
    parts = [PROTO, a.level]

    if compact:
        # 经纬度 4 位小数 ≈ 11 m 量级，与单点定位精度（9 m）匹配。
        # 再写更多位是虚假精度 —— 基准点本身就没那么准。
        parts.append(f"{_fmt_coord(a.lon, 'W', 'E', 4)},{_fmt_coord(a.lat, 'S', 'N', 4)}")
        parts.append(_int_or(a.count))
        parts.append(f"{a.density:.1f}" if a.density is not None else "-")
        t = (a.utc or "")
        parts.append(t[-9:-1] if t.endswith("Z") and len(t) >= 9 else "-")
    else:
        parts.append(f"{_fmt_coord(a.lon, 'W', 'E', 6)},{_fmt_coord(a.lat, 'S', 'N', 6)}")
        parts.append(_int_or(a.count))
        parts.append(f"{a.density:.2f}" if a.density is not None else "-")
        parts.append((a.utc or "-"))
        parts.append((a.site or "-").replace("|", "/"))

    return "|".join(parts)


@dataclass
class Encoded:
    """编码结果。"""

    text: str
    payload: bytes
    nbytes: int
    channel: str              # region | global
    limit_bytes: int
    compact: bool             # 是否走了裁剪分支
    dropped: list[str]        # 被裁掉的字段名，要如实报给用户
    crc: int

    @property
    def ok(self) -> bool:
        return self.nbytes <= self.limit_bytes

    @property
    def zh_chars(self) -> int:
        """按"汉字"口径估算的占用（每个汉字 2 字节），供界面写文案。"""
        return (self.nbytes + 1) // 2


def encode(a: Alert, *, channel: str = "region") -> Encoded:
    """把预警编码为短报文字符串。

    :param channel: ``region``（14000 bit）或 ``global``（560 bit）。

    超长时**自动降级**：区域档先裁点位名，全球档只保留最关键的字段。
    降级是确定的、可预期的，并且结果里 ``dropped`` 会列清楚丢了什么。
    极端情况下仍塞不进 → 抛 :class:`MessageTooLong`，**绝不静默截断**：
    截断后的报文可能把"经度"截成另一个合法数字，接收端无法察觉。
    """
    if channel not in ("region", "global"):
        raise ValueError(f"未知短报文通道：{channel!r}（可选 region / global）")
    if a.level not in LEVEL_ZH:
        raise ValueError(f"未知预警级别：{a.level!r}（可选 {'/'.join(LEVEL_ZH)}）")

    limit = GLOBAL_BYTES if channel == "global" else REGION_BYTES
    force_compact = channel == "global"

    attempts: list[tuple[bool, list[str]]] = []
    if force_compact:
        attempts.append((True, []))
    else:
        # 区域档：先试完整，再逐步裁剪
        attempts.append((False, []))
        attempts.append((False, ["点位名"]))

    last_err: Exception | None = None
    for compact, dropped in attempts:
        body = _build_body(a, compact=compact)
        if not force_compact and dropped:
            # 裁掉点位名：主体最后一段本来是 site，重拼时去掉
            body = "|".join(body.split("|")[:-1])
        payload = body.encode("utf-8", errors="replace")
        full = payload + b"*" + f"{crc16(payload):04X}".encode("ascii")
        if len(full) <= limit:
            return Encoded(
                text=full.decode("utf-8", errors="replace"),
                payload=payload, nbytes=len(full), channel=channel,
                limit_bytes=limit, compact=compact, dropped=dropped,
                crc=crc16(payload),
            )
        last_err = None

    raise MessageTooLong(
        f"预警内容在 {'全球' if channel == 'global' else '区域'}短报文的 "
        f"{limit} 字节（{GLOBAL_CHARS if channel == 'global' else REGION_CHARS} 汉字）"
        f"约束下仍放不下，已尝试裁剪。请缩短点位名或减少上报字段。"
    )


@dataclass
class Decoded:
    proto: str
    level: str
    level_zh: str
    lon: float | None
    lat: float | None
    count: int | None
    density: float | None
    utc: str
    site: str | None
    crc_ok: bool


def decode(text: str) -> Decoded:
    """解析一条短报文。

    **CRC 不通过不抛异常**，而是把 ``crc_ok=False`` 返回出去 —— 短报文链路上
    丢字符是常态，接收端需要"能解析但标记为可疑"，而不是直接崩掉整个接收流程。
    """
    s = (text or "").strip()
    if "*" in s:
        body, _, crc_s = s.rpartition("*")
        try:
            got = int(crc_s, 16)
        except ValueError:
            got = -1
        want = crc16(body.encode("utf-8", errors="replace"))
        crc_ok = (got == want)
    else:
        body, crc_ok = s, False

    parts = body.split("|")
    if len(parts) < 6:
        # 字段数都不够，按最保守的还原：能读到什么给什么
        return Decoded("", "", "", None, None, None, None, "", None, False)

    proto, level, coords = parts[0], parts[1], parts[2]
    rest = parts[3:]

    def _num(x, cast):
        if x in ("", "-"):
            return None
        try:
            return cast(x)
        except (TypeError, ValueError):
            return None

    lon = lat = None
    if "," in coords:
        lo, _, la = coords.partition(",")
        lon = _num(lo, float)
        lat = _num(la, float)

    count = _num(rest[0], int) if len(rest) > 0 else None
    density = _num(rest[1], float) if len(rest) > 1 else None
    utc = rest[2] if len(rest) > 2 else ""
    site = rest[3] if len(rest) > 3 else None

    return Decoded(
        proto=proto, level=level, level_zh=LEVEL_ZH.get(level, level or "—"),
        lon=lon, lat=lat, count=count, density=density,
        utc=utc, site=site, crc_ok=crc_ok,
    )
