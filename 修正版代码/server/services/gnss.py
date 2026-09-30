"""可插拔的北斗/GNSS 定位源。

三种来源，同一套下游
--------------------
======================  ==========================================================
来源                    说明
======================  ==========================================================
``serial``              串口直读北斗/GNSS 模块（NMEA 0183）。pyserial 为**可选依赖**。
``file``                回放 NMEA 日志。没有硬件时做演示、或复现现场问题。
``mock``                程序生成 NMEA 语句（带正确校验和）再喂给同一个解析器。
======================  ==========================================================

设计上的两个硬要求
------------------
1. **模拟数据也走真实解析链路**。``mock`` 不是"绕过解析器直接塞一个对象"，
   而是生成 NMEA 文本再交给 :mod:`server.services.nmea`。
   否则模拟路径和真实路径是两段代码，真实路径坏了也测不出来。
2. **拿不到定位就如实说"没有"**。绝不回退到一组写死的坐标 ——
   那会让"模块没插好"和"定位正常"在接口上长得一模一样，
   是最危险的一类静默错误（本项目已经在权重和视频上栽过两次同样的坑）。

健康度可观测
------------
串口接错线、波特率不匹配、模块未定位，现象都是"没有数据"，但原因完全不同。
因此这里持续统计 ``bytes_read`` / ``lines_read`` / ``sentences`` / ``checksum_errors``，
并据此给出**可执行**的提示，例如"收到了字节但一条语句都解析不出，多半是波特率不对"。
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..schemas.detection import (
    AccuracyTierInfo,
    BeidouInfo,
    GnssSatellite,
    SatelliteComposition,
    TimeBaseInfo,
)
from .gnss_constellation import build as build_composition
from .gnss_quality import classify, tier_level_m, tier_zh
from .gnss_timebase import build as build_timebase
from .gnss_track import parse_stamped_line
from .nmea import NmeaAccumulator, checksum_of, parse_line

log = logging.getLogger("server.gnss")

# pyserial 是可选依赖：没装也能跑（file / mock 模式不受影响）
try:  # pragma: no cover - 取决于环境
    import serial
    from serial.tools import list_ports

    SERIAL_OK = True
except Exception:  # pragma: no cover
    serial = None
    list_ports = None
    SERIAL_OK = False

# 串口描述里出现这些词就优先选它（USB-TTL 桥接芯片与常见模块厂牌）
_PORT_HINTS = ("gnss", "gps", "beidou", "bds", "ch340", "ch341", "cp210", "ftdi", "uart", "usb-serial")


def _nmea(payload: str) -> str:
    """拼一句带校验和的 NMEA。"""
    return f"${payload}*{checksum_of(payload)}\r\n"


class GnssSource:
    """定位源基类。子类只需实现 :meth:`readline`。"""

    name = "base"

    def __init__(self) -> None:
        self.detail: str | None = None
        self.error: str | None = None
        self.bytes_read = 0

    def readline(self, timeout: float) -> str | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    @property
    def ok(self) -> bool:
        return self.error is None


# --------------------------------------------------------------------- 串口

class SerialSource(GnssSource):
    """串口直读。端口留空时自动扫描，优先描述里带 GNSS/GPS/CH340 等的口。"""

    name = "serial"

    def __init__(self, port: str, baud: int, timeout: float) -> None:
        super().__init__()
        if not SERIAL_OK:
            self.error = "未安装 pyserial，无法使用串口定位源（pip install pyserial）"
            self.detail = port or None
            return
        try:
            self.port = port or self.autodetect()
            if not self.port:
                self.error = ("未发现可用串口。请插好北斗模块，"
                              "或显式指定 APP_GNSS_PORT=COM3")
                return
            self.ser = serial.Serial(self.port, baud, timeout=timeout)
            self.detail = f"{self.port} @ {baud}"
        except Exception as e:  # 端口被占用 / 权限不足 / 设备被拔
            self.error = f"打开串口失败：{e}"
            self.detail = port or None
            self.ser = None

    @staticmethod
    def autodetect() -> str | None:
        """挑一个最像 GNSS 模块的串口。"""
        if not SERIAL_OK:
            return None
        try:
            ports = list(list_ports.comports())
        except Exception:
            return None
        if not ports:
            return None
        scored: list[tuple[int, str]] = []
        for p in ports:
            text = f"{p.description or ''} {p.manufacturer or ''} {p.device}".lower()
            score = sum(1 for h in _PORT_HINTS if h in text)
            scored.append((score, p.device))
        scored.sort(key=lambda x: (-x[0], x[1]))
        best_score, best = scored[0]
        # 只有一个串口时也认；有多个但都不像 GNSS 就不猜，避免连到别的设备上乱读
        if best_score > 0 or len(scored) == 1:
            return best
        return None

    def readline(self, timeout: float) -> str | None:
        ser = getattr(self, "ser", None)
        if ser is None:
            return None
        try:
            raw = ser.readline()
        except Exception as e:
            self.error = f"串口读取失败：{e}"
            return None
        if not raw:
            return None
        self.bytes_read += len(raw)
        return raw.decode("ascii", errors="ignore").strip()

    def close(self) -> None:
        ser = getattr(self, "ser", None)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass


# --------------------------------------------------------------------- 回放

class FileSource(GnssSource):
    """回放 NMEA 日志（支持 MCU 写卡的"时间戳 + 语句"格式）。

    切成历元块，块内逐句吐出、块间按间隔等待，这样时间关系跟现场一致，
    而不是把几万行一口气灌进去。历元间隔优先取自日志里的**真实时间戳**，
    没有时间戳时才用 ``epoch_s`` 默认值。
    """

    name = "file"

    def __init__(self, path: str, loop: bool, speed: float, epoch_s: float = 1.0) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop
        self.speed = max(0.05, float(speed or 1.0))
        self.epoch_s = epoch_s
        self.chunks: list[list[str]] = []
        self._buffer: list[str] = []
        self._i = 0
        self._next_at = 0.0
        if not path:
            self.error = "未配置回放文件（APP_GNSS_FILE）"
            return
        if not self.path.exists():
            self.error = f"回放文件不存在：{self.path}"
            return
        try:
            self.chunks = self._split(self.path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            self.error = f"读取回放文件失败：{e}"
            return
        if not self.chunks:
            self.error = f"回放文件里没有可识别的 NMEA 语句：{self.path}"
            return
        self.detail = f"{self.path.name} · {sum(len(c) for c in self.chunks)} 句"

    def _split(self, text: str) -> list[list[str]]:
        """切成历元块，**只保留纯 NMEA 部分**（剥掉时间戳）。

        两个关键点：

        1. 必须用 :func:`parse_stamped_line` 先剥时间戳。MCU 写出来的日志是
           ``2026-09-17T12:00:00Z,$GNGGA,...``，直接丢给 ``parse_line`` 会因为
           行首不是 ``$`` 而**整行丢弃** —— 表现为"文件明明有内容却一句都读不到"。
        2. 切历元按 **GGA 或 RMC 第二次出现**。不能"遇到 RMC 就切"（各模块语句
           顺序不同，会把一个历元切成两块），也不能对 GSV 用重复即切（一个历元本来就有多句）。

        同时从时间戳里估出**真实历元间隔**，让回放速度与现场一致 ——
        固定按 1 秒一个历元，遇到 10 Hz 的日志会慢十倍。
        """
        chunks: list[list[str]] = []
        cur: list[str] = []
        seen: set[str] = set()
        stamps: list[float] = []

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                if cur:
                    chunks.append(cur)
                    cur, seen = [], set()
                continue
            st = parse_stamped_line(line)
            if st is None:
                continue                      # 注释头 / 调试输出 / 非法行
            s = parse_line(st.nmea)
            if s is None:
                continue
            if s.kind in ("GGA", "RMC") and s.kind in seen:
                chunks.append(cur)
                cur, seen = [], set()
            if s.kind == "GGA" and st.epoch_s is not None:
                stamps.append(st.epoch_s)
            seen.add(s.kind)
            cur.append(st.nmea)
        if cur:
            chunks.append(cur)

        if len(stamps) > 1:
            gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if b > a)
            if gaps:
                med = gaps[len(gaps) // 2]
                # 限幅：日志时间戳乱跳时不要把回放速度搞成 0.001 秒或 10 分钟
                self.epoch_s = min(10.0, max(0.05, med))
        return chunks

    def readline(self, timeout: float) -> str | None:
        """块内逐句立即返回，块间按 ``epoch_s / speed`` 等待；无数据可给时返回 None。

        注意：这里**不按 timeout 阻塞**，而是自己管节奏（回放不需要真实等待读事件）。
        """
        if self._buffer:
            return self._buffer.pop(0)
        chunk = self._take_chunk()
        if not chunk:
            return None
        # 必须 **拷贝**：直接 self._buffer = chunk 会与 chunks[i] 是同一个列表，
        # pop(0) 会把源数据就地掏空 —— 循环回放时文件越读越少，且 chunks 被改坏。
        self._buffer = list(chunk)
        return self._buffer.pop(0)

    def _take_chunk(self) -> list[str]:
        """到点则取下一个历元块，否则返回空。"""
        if not self.chunks:
            return []
        if self._i >= len(self.chunks):
            if not self.loop:
                return []
            self._i = 0
        now = time.monotonic()
        if now < self._next_at:
            return []
        chunk = self.chunks[self._i]
        self._i += 1
        self._next_at = now + self.epoch_s / self.speed
        self.bytes_read += sum(len(ln) + 1 for ln in chunk)
        return chunk


# --------------------------------------------------------------------- 模拟

class MockSource(GnssSource):
    """程序生成 NMEA。走的是和真实模块完全一样的解析链路。"""

    name = "mock"

    # 6 颗 GPS + 6 颗北斗，方位角/高度角缓慢漂移，星空图不会是一张静止的假图
    _GP = ("03", "04", "06", "13", "15", "17")
    _BD = ("C01", "C02", "C03", "C04", "C05", "C06")

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.lat = settings.BEIDOU_LAT
        self.lon = settings.BEIDOU_LON
        self.alt = settings.BEIDOU_ALT
        # 注意别写 ``getattr(...) or 1``：0 是合法取值（无定位），
        # 会被 or 判成假值而悄悄变成 1，"无定位"这个界面状态就永远测不出来。
        _q = getattr(settings, "GNSS_MOCK_FIX_QUALITY", 1)
        self.fix_quality = 1 if _q is None else int(_q)
        self.detail = "内置模拟（非真实定位）"
        self._t = 0.0
        self._next_at = 0.0
        self._queue: list[str] = []

    def readline(self, timeout: float) -> str | None:
        if self._queue:
            return self._queue.pop(0)
        now = time.monotonic()
        if now < self._next_at:
            return None
        self._next_at = now + 1.0
        self._queue = self._epoch(self._t)
        self._t += 1.0
        if not self._queue:
            return None
        return self._queue.pop(0)

    # 生成一个历元（1 秒）的语句
    def _epoch(self, t: float) -> list[str]:
        # 位置缓慢漂移，看起来像在移动（也顺便让 ENU 散布有东西可画）
        lat = self.lat + 0.00012 * math.sin(t / 40.0)
        lon = self.lon + 0.00018 * math.cos(t / 55.0)
        hh = int(t // 3600) % 24
        mm = int(t // 60) % 60
        ss = int(t % 60)
        utc = f"{hh:02d}{mm:02d}{ss:02d}.00"

        # 「已用」不能超过「可见」：模拟源一度写死 18，而 GSV 只播发 12 颗，
        # 界面上就出现「18 / 12」这种自相矛盾的读数 —— 演示数据也要自洽，
        # 否则用户第一眼就会怀疑整个定位链路。
        sats_used = (len(self._GP) + len(self._BD)) if self.fix_quality else 0
        hdop = 0.6 if self.fix_quality in (4, 5) else (0.9 if self.fix_quality else 99.9)
        # GSA 的卫星槽固定 12 个：显式补空位，别靠数逗号（数错过一次）
        slots = [f"{p:02d}" for p in (3, 4, 6, 13, 15, 17)] + [""] * 6
        lines = [
            _nmea(f"GNGGA,{utc},{_dm(lat, True)},{'N' if lat >= 0 else 'S'},"
                  f"{_dm(lon, False)},{'E' if lon >= 0 else 'W'},"
                  f"{self.fix_quality},{sats_used:02d},{hdop:.1f},{self.alt:.1f},M,10.1,M,,"),
            _nmea(f"GNRMC,{utc},A,{_dm(lat, True)},{'N' if lat >= 0 else 'S'},"
                  f"{_dm(lon, False)},{'E' if lon >= 0 else 'W'},"
                  f"0.4,{int(t * 3) % 360:03d}.0,{_dmy()},,,A"),
            _nmea(f"GNGSA,A,{3 if self.fix_quality else 1}," + ",".join(slots)
                  + f",1.1,{hdop:.1f},1.6"),
            self._gsv("GP", self._GP, t, 0.0),
            self._gsv("BD", self._BD, t, 37.0),
        ]
        return lines

    @staticmethod
    def _gsv(talker: str, prns, t: float, phase: float) -> str:
        body = f"{talker}GSV,1,1,{len(prns):02d}"
        for i, prn in enumerate(prns):
            el = 20 + 55 * abs(math.sin(t / 90.0 + i * 0.7 + phase))
            az = (i * 57 + t * 2 + phase * 3) % 360
            snr = 30 + int(12 * abs(math.cos(t / 70.0 + i)))
            body += f",{prn},{int(el):02d},{int(az):03d},{snr:02d}"
        return _nmea(body)


def _dm(deg: float, is_lat: bool) -> str:
    """十进制度 → NMEA 的 ddmm.mmmm / dddmm.mmmm。"""
    deg = abs(deg)
    d = int(deg)
    minutes = (deg - d) * 60.0
    return f"{d:02d}{minutes:07.4f}" if is_lat else f"{d:03d}{minutes:07.4f}"


def _dmy() -> str:
    """当前 UTC 日期 → NMEA 的 ddmmyy。用真实日期，演示时时间戳才对得上。"""
    now = datetime.now(timezone.utc)
    return f"{now.day:02d}{now.month:02d}{now.year % 100:02d}"


# --------------------------------------------------------------------- 服务

class GnssService:
    """后台线程持续读取定位源，对外提供**最新定位帧**。

    只保留最新一帧：历史轨迹由视频任务按帧自己记录。这里再存一份历史，
    只会带来"两处历史不一致"的风险，而排查这种问题极其费时。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._acc = NmeaAccumulator()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.source: GnssSource | None = None
        self.source_name = "unavailable"
        self.source_detail: str | None = None
        self.error: str | None = None
        self.lines_read = 0
        self._started = False

        if settings.BEIDOU_REAL:
            log.warning(
                "APP_BEIDOU_REAL 已废弃并被忽略：是否真实定位现在由定位源自动判定"
                "（serial/file + 有效定位解）。请改用 APP_GNSS_SOURCE。"
            )

    # ------------------------------------------------------------ 生命周期
    def ensure_started(self) -> None:
        if self._started:
            return
        self._started = True
        self._pick_source()
        if self.source is not None and self.source.ok:
            self._thread = threading.Thread(target=self._loop, name="gnss-reader", daemon=True)
            self._thread.start()

    def _pick_source(self) -> None:
        """按配置挑源。``auto`` 是"先试串口，再试文件"，都失败就如实报 unavailable。"""
        s = self.settings
        mode = (s.GNSS_SOURCE or "auto").strip().lower()
        errors: list[str] = []

        if mode in ("serial", "auto"):
            src = SerialSource(s.GNSS_PORT, s.GNSS_BAUD, s.GNSS_READ_TIMEOUT)
            if src.ok:
                self._adopt(src)
                return
            errors.append(src.error or "串口不可用")
            if mode == "serial":
                self._fail("serial", src.detail, src.error)
                return

        if mode in ("file", "auto") and (s.GNSS_FILE or mode == "file"):
            src = FileSource(s.GNSS_FILE, s.GNSS_REPLAY_LOOP, s.GNSS_REPLAY_SPEED)
            if src.ok:
                self._adopt(src)
                return
            errors.append(src.error or "回放文件不可用")
            if mode == "file":
                self._fail("file", s.GNSS_FILE or None, src.error)
                return

        if mode == "mock":
            self._adopt(MockSource(s))
            return

        hint = ("；".join(errors) if errors else "未配置任何定位源")
        self._fail(
            "unavailable", None,
            f"{hint}。接好北斗模块后重启即可；若只想看界面，把 APP_GNSS_SOURCE 设为 mock。",
        )

    def _adopt(self, src: GnssSource) -> None:
        self.source = src
        self.source_name = src.name
        self.source_detail = src.detail
        log.info("GNSS 定位源：%s（%s）", src.name, src.detail)

    def _fail(self, name: str, detail: str | None, error: str | None) -> None:
        self.source_name = name
        self.source_detail = detail
        self.error = error
        log.warning("GNSS 定位源不可用：%s", error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.source is not None:
            self.source.close()

    def _loop(self) -> None:
        src = self.source
        assert src is not None
        while not self._stop.is_set():
            try:
                # 所有源走同一条读路径：谁的 readline 返回 None 就表示"暂时没数据"。
                # 曾经给 FileSource 单独开了一条分支，结果两条路径的行为要各自维护，
                # 回放的节流逻辑也因此和串口不一致 —— 统一掉。
                ln = src.readline(self.settings.GNSS_READ_TIMEOUT)
                if ln is None:
                    self._stop.wait(0.02)
                elif ln:
                    with self._lock:
                        self.lines_read += 1
                        self._acc.feed(ln)

                if src.error:  # 串口中途掉了（拔线/供电不足）
                    self.error = src.error
                    break
            except Exception as e:  # pragma: no cover - 兜底，别让线程静默死掉
                log.exception("GNSS 读取线程异常：%s", e)
                self.error = f"读取线程异常：{e}"
                break

    # ------------------------------------------------------------ 对外
    def snapshot(self) -> dict:
        with self._lock:
            return self._acc.snapshot()

    def info(self) -> BeidouInfo:
        """当前定位帧。**没有定位时 lat/lon 就是 None，不会回退到演示坐标。**"""
        self.ensure_started()
        snap = self.snapshot()
        s = self.settings

        age = snap.get("age_s")
        stale = age is not None and age > s.GNSS_STALE_S
        # 数据太旧等于没有定位：模块掉线后界面必须立刻变"无定位"，
        # 而不是把最后一次坐标一直挂在那儿骗人。
        dead = age is None or age > s.GNSS_MAX_AGE_S
        usable = bool(snap.get("usable")) and not dead
        real = self.source_name in ("serial", "file") and usable

        gsd, calibrated, estimated = s.geo_gsd

        msg = self.error
        # 判定条件是「**不可用**」而不是「数据过期」：
        # 模块可以既在正常吐语句、又明确报着 fix_quality=0（没收到星）。
        # 这种情况以前会落到"没有原因"的分支 —— 界面显示无定位却什么都不说，
        # 正是要避免的那种静默状态。
        if msg is None and not usable:
            sentences = snap.get("sentences", 0)
            if self.source_name == "unavailable":
                msg = "未接入定位源。"
            elif self.lines_read == 0:
                msg = "定位源已连接但没收到任何数据，检查模块供电与接线。"
            elif sentences == 0:
                msg = (f"收到 {self.lines_read} 行数据但一条 NMEA 语句都解析不出，"
                       f"多半是波特率不匹配（当前 {s.GNSS_BAUD}）。")
            elif snap.get("fix_quality") == 0:
                msg = ("模块已正常输出语句，但当前报「无定位」（fix_quality=0），"
                       "正在收星 —— 冷启动或室内可能需 30 秒以上。")
            elif age is not None and age > s.GNSS_MAX_AGE_S:
                msg = f"最后一次有效定位已是 {age:.0f} 秒前，模块可能掉线。"
            else:
                msg = "已收到 NMEA 语句但尚未取得有效定位解。"
        elif msg is None and stale:
            msg = f"定位数据已过期 {age:.1f} 秒，可能信号丢失。"

        sat_detail = [
            GnssSatellite(
                prn=str(x.get("prn")),
                talker=x.get("talker"),
                elevation=x.get("elevation"),
                azimuth=x.get("azimuth"),
                snr=x.get("snr"),
            )
            for x in (snap.get("satellites") or [])
        ]

        # ---- 精度档位：把"有定位"细化为"哪种精度的定位" ----
        # 判定所需字段全部取自 NMEA 原生字段，不引入外部假设。
        tier_key, tier_msg = classify(
            usable=usable,
            source=self.source_name,
            fix_quality=snap.get("fix_quality"),
            mode=snap.get("mode"),
            dgps_age=snap.get("dgps_age"),
        )
        acc = None
        if tier_key is not None:
            acc = AccuracyTierInfo(
                key=tier_key,
                zh=tier_zh(tier_key) or tier_key,
                level_m=tier_level_m(tier_key),
                detail=tier_msg or "",
            )

        # ---- 时间基准：UTC 从"显示字段"升级为"可核对基准" ----
        # local_epoch 在这里取，与 utc 的采样尽量贴近（见 gnss_timebase.build 的说明）
        tb = build_timebase(
            utc=snap.get("utc"),
            usable=usable,
            source=self.source_name,
            local_epoch=time.time(),
        )

        # ---- 星座构成：说明"北斗在这一帧里出了多少力" ----
        comp = build_composition(
            snap.get("satellites"),
            snap.get("satellites_used"),
            sentences=snap.get("sentences", 0),
        )

        return BeidouInfo(
            lat=snap.get("lat") if usable else None,
            lon=snap.get("lon") if usable else None,
            alt=snap.get("alt") if usable else None,
            satellites=snap.get("satellites_used") if usable else None,
            real=real,
            source=self.source_name,
            source_detail=self.source_detail,
            usable=usable,
            fix_quality=snap.get("fix_quality"),
            fix_quality_zh=snap.get("fix_quality_zh"),
            fix_type=snap.get("fix_type"),
            fix_type_zh=snap.get("fix_type_zh"),
            mode_zh=snap.get("mode_zh"),
            hdop=snap.get("hdop"),
            vdop=snap.get("vdop"),
            pdop=snap.get("pdop"),
            satellites_used=snap.get("satellites_used"),
            satellites_visible=snap.get("satellites_visible"),
            satellites_detail=sat_detail,
            speed_kmh=snap.get("speed_kmh"),
            course=snap.get("course"),
            utc=snap.get("utc"),
            age_s=age,
            stale=stale,
            sentences=snap.get("sentences", 0),
            checksum_errors=snap.get("checksum_errors", 0),
            message=msg,
            gsd_m_per_px=gsd,
            gsd_calibrated=calibrated,
            estimated=estimated,
            accuracy_tier=acc,
            accuracy_message=tier_msg,
            timebase=TimeBaseInfo(
                utc=tb.utc,
                utc_epoch=tb.utc_epoch,
                local_epoch=tb.local_epoch,
                delta_s=tb.delta_s,
                has_date=tb.has_date,
                source_zh=tb.source_zh,
                message=tb.message,
            ),
            composition=SatelliteComposition(**comp),
        )
