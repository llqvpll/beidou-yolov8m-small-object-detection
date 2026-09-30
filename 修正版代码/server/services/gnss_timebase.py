"""北斗时间基准（授时）：把 NMEA 里的 UTC 从"一个字段"变成"一条可核对的时间基准"。

思路
----
北斗公开服务的授时精度优于 20 纳秒（白皮书系统能力指标）。用户在界面上看到的
"解算时刻"本来只是 GGA/RMC 里的一串字符，意义有限 —— 它既没有说明这是**哪个**
时间基准，也没有说明**与运行这台机器的时钟差多少**。

而"差多少"恰恰是我们系统最需要的东西：视频帧、检测结果、密度记录、日志，
全都挂在时间戳上。**只要这些时间戳来自本机时钟，它们就带着本机时钟的漂移**；
一旦把基准换成北斗 UTC，所有记录才有同一个可外部核对的参照。

三条诚实约束
------------
1. **没有有效定位解就不给时间基准**。模块在吐语句 ≠ 时间可信：冷启动阶段
   语句里的 UTC 可能来自模块内部的旧值。
2. **不给"本机偏差"编数字**。算不出来（没有日期、没有有效解）就是 None。
3. **区分"UTC 秒值"与"UTC 字面量"**。前者能参与运算，后者只够显示。
   ``utc`` 字段在只有 GGA 时会退化成 ``HH:MM:SSZ``（没有日期），
   这时算 epoch 会猜出一个错误的日期 —— 所以我们宁可不算。
"""

from __future__ import annotations

import calendar
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeBase:
    """一次时间基准快照。

    ``utc_epoch`` —— UTC 秒值（POSIX，无闰秒修正）。只有拿到**含日期**的
                    完整时间戳才算得出来，否则为 None。
    ``local_epoch`` —— 同一时刻本机时钟的 UTC 秒值。
    ``delta_s``   —— ``utc_epoch - local_epoch``。
                    正数表示本机时钟**慢了**（北斗时间走在前头）。
                    这个符号约定必须写死并测到，否则用户看到 -3 秒会以为
                    是"本机快了"，然后往反方向调。
    ``source_zh`` —— 时间从哪里来的。
    ``message``   —— 拿不到基准时说明原因。
    """

    utc: str | None
    utc_epoch: float | None
    local_epoch: float | None
    delta_s: float | None
    has_date: bool
    source_zh: str
    message: str | None


def parse_utc_epoch(utc: str | None) -> float | None:
    """把 ``YYYY-MM-DDTHH:MM:SSZ`` 解析为 POSIX 秒。

    **只接受带日期的完整形式。** 退化形式（``HH:MM:SSZ``，GGA 单句没有日期）
    直接返回 None —— 给它猜一个日期，算出来的偏差会错到"差一整天"这个量级，
    而这种错误在界面上看起来完全正常。
    """
    if not utc:
        return None
    s = utc.strip().rstrip("Zz")
    if "T" not in s:
        return None
    date_part, _, time_part = s.partition("T")
    if not date_part or not time_part:
        return None
    try:
        y, mo, d = (int(x) for x in date_part.split("-"))
    except (ValueError, TypeError):
        return None
    hms = time_part.split(":")
    if len(hms) < 2:
        return None
    try:
        hh = int(hms[0])
        mm = int(hms[1])
        # 秒可能有小数，也可能带闰秒的 "60"
        ss_f = float(hms[2]) if len(hms) > 2 else 0.0
        ss = int(ss_f)
        frac = ss_f - ss
    except (ValueError, TypeError):
        return None
    # GGA 在闰秒时可能给 60，calendar 会拒绝 → 归到 59 由 frac 补。宁可差 1 秒，
    # 也不要因为一个闰秒把整条时间基准断掉。
    if ss > 59:
        ss = 59
    try:
        base = calendar.timegm((y, mo, d, hh, mm, ss, 0, 0, 0))
    except (ValueError, OverflowError):
        return None
    return base + frac


def build(
    *,
    utc: str | None,
    usable: bool,
    source: str,
    now: float | None = None,
    local_epoch: float | None = None,
) -> TimeBase:
    """组装时间基准。

    :param local_epoch: 与 ``utc`` 采样同一时刻的本机时钟 UTC 秒值。
        由调用方传入，是为了让"偏差"这一对采样在时间上尽量贴近 ——
        在函数里各取一次 ``time.time()``，两次调用之间隔着解析与加锁，
        在快机器上也要几百微秒，会把小偏差淹没掉。

    **回放源（file）不算偏差。** 这是本函数最容易做错的地方：离线日志里的 UTC 是
    "当时那一刻"的，而本机时钟是"现在"，两者相差的是**回放时间跨度**（可能是几小时
    或几天），根本不是时钟精度问题。把这个数当成"本机时钟偏差"报出去，用户会去
    校准一个没有偏差的时钟 —— 而那个真正的偏差反而被一个巨大的假数字盖住了。

    所以回放源只报"日志时刻"，偏差留空并说明原因。
    """
    src_zh = {
        "serial": "北斗模块串口实时授时",
        "file": "北斗模块离线日志（回放）",
        "mock": "模拟源（不作为时间基准）",
        "unavailable": "未接入定位源",
    }.get(source, source or "未知来源")

    if source == "mock":
        return TimeBase(None, None, None, None, False, src_zh,
                        "当前是模拟数据源，其时间不作为基准使用。")
    if not usable:
        return TimeBase(None, None, None, None, False, src_zh,
                        "未取得有效定位解，暂不建立时间基准。")
    if not utc:
        return TimeBase(None, None, None, None, False, src_zh,
                        "定位语句中未包含 UTC 时间字段。")

    epoch = parse_utc_epoch(utc)
    if epoch is None:
        # 退化形式：只有时分秒。能显示，但不能当基准用。
        return TimeBase(utc, None, None, None, False, src_zh,
                        "当前 UTC 只有时分秒、没有日期（GGA 单句不带日期），"
                        "无法换算成绝对时刻，因此不计偏差。")

    # 回放源：时刻可信、可比对，但**不能与本机时钟相减**
    if source == "file":
        return TimeBase(utc, epoch, None, None, True, src_zh,
                        "这是离线日志中的时刻，与本机时钟相差的是回放时间跨度，"
                        "不是时钟偏差，故不计偏差。")

    if local_epoch is None:
        local_epoch = time.time()
    delta = epoch - local_epoch
    return TimeBase(utc, epoch, local_epoch, round(delta, 3), True, src_zh, None)
