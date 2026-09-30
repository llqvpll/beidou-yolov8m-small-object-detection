"""北斗定位源与坐标换算的单元测试。

两条主线：
1. **不能静默造假**。拿不到定位时必须返回"无定位"，绝不能回退到一组写死的坐标 ——
   否则"模块没插好"和"定位正常"在接口上长得一模一样。
2. **换算只有一把尺子**。像素→地面只有一个 GSD，后端 ENU 与 geo_map 必须自洽
   （历史上两边差约 140 倍，表格里两列数字互相矛盾）。
"""
import math

import pytest

from server.config import Settings
from server.services.beidou import M_PER_DEG_LAT, BeidouService
from server.services.gnss import (
    SERIAL_OK,
    FileSource,
    GnssService,
    GnssSource,
    MockSource,
    SerialSource,
)

W, H = 1920, 1080


def _settings(**kw) -> Settings:
    """构造独立配置。显式 kwargs 优先级最高，不受 .env 与进程环境干扰。"""
    base = dict(GNSS_SOURCE="mock", GNSS_PORT="", GNSS_FILE="",
                GEO_GSD_M_PER_PX=0.0, GEO_ESTIMATE_GSD=0.021, GEO_ALLOW_ESTIMATE=True)
    base.update(kw)
    return Settings(**base)


def _service(**kw) -> BeidouService:
    svc = BeidouService(_settings(**kw))
    svc.start()
    return svc


# ---------------------------------------------------------------- 模拟源

def test_mock_source_goes_through_real_parser():
    """模拟数据必须走真实 NMEA 解析链路，而不是绕过去直接塞对象。"""
    svc = _service()
    fix = svc.info()
    assert fix.source == "mock"
    assert fix.real is False          # 模拟就是模拟，绝不冒充真实
    assert fix.usable is True
    assert fix.sentences > 0          # 确实解析过语句
    assert fix.checksum_errors == 0   # 自己生成的语句校验和必须是对的
    assert round(fix.lat, 3) == 28.169
    assert round(fix.lon, 3) == 112.944


def test_mock_reports_quality_and_satellites():
    fix = _service().info()
    assert fix.fix_quality == 1
    assert fix.fix_quality_zh == "单点定位"
    assert fix.satellites_used == 12             # 与 GSV 播发的颗数一致，见下一条
    assert fix.satellites_visible == 12          # 6 颗 GPS + 6 颗北斗
    assert fix.hdop is not None
    assert fix.utc and "T" in fix.utc            # 带日期才可能是完整 ISO


def test_mock_used_satellites_never_exceeds_visible():
    """「已用」不能大于「可见」。

    模拟源曾经写死 used=18 而只播发 12 颗 GSV，界面上就出现「18 / 12」——
    演示数据自相矛盾，用户第一眼就会怀疑整个定位链路是不是假的。
    """
    fix = _service().info()
    assert fix.satellites_used is not None and fix.satellites_visible is not None
    assert fix.satellites_used <= fix.satellites_visible


def test_mock_skyplot_has_two_constellations():
    """星空图要能区分 GPS 与北斗，这是"北斗系统"最直观的展示点。"""
    detail = _service().info().satellites_detail
    talkers = {s.talker for s in detail}
    assert talkers == {"GP", "BD"}
    assert len(detail) == 12
    for s in detail:
        assert 0 <= s.elevation <= 90
        assert 0 <= s.azimuth < 360
        assert s.snr is not None


def test_mock_no_fix_state_is_representable():
    """模拟成"无定位"时，坐标必须是 None，不能是上一次的值。"""
    fix = _service(GNSS_MOCK_FIX_QUALITY=0).info()
    assert fix.fix_quality == 0
    assert fix.fix_quality_zh == "无定位"
    assert fix.usable is False
    assert fix.lat is None and fix.lon is None and fix.alt is None
    assert fix.real is False


def test_mock_rtk_fixed_state():
    fix = _service(GNSS_MOCK_FIX_QUALITY=4).info()
    assert fix.fix_quality_zh == "RTK 固定解"
    assert fix.usable is True


# ---------------------------------------------------------------- 不可用

def test_missing_replay_file_yields_no_fix_not_fake_coords():
    """回放文件不存在时必须是"无定位"，且给出可执行的原因。"""
    svc = _service(GNSS_SOURCE="file", GNSS_FILE="C:/definitely/not/here.nmea")
    fix = svc.info()
    assert fix.source == "file"
    assert fix.usable is False
    assert fix.lat is None and fix.lon is None
    assert fix.real is False
    assert fix.message and "不存在" in fix.message


def test_bad_serial_port_yields_no_fix():
    svc = _service(GNSS_SOURCE="serial", GNSS_PORT="COM_NOT_A_REAL_PORT_999")
    fix = svc.info()
    assert fix.source == "serial"
    assert fix.usable is False
    assert fix.lat is None
    assert fix.message


def test_auto_without_anything_is_unavailable_and_actionable():
    """auto 扫不到任何源时，要说清怎么启用，而不是默默给个坐标。"""
    # 显式给一个不存在的串口：本机可能真有 COM 口，测试必须与设备无关
    svc = GnssService(_settings(GNSS_SOURCE="auto", GNSS_PORT="COM_NOT_A_REAL_PORT_999"))
    svc._pick_source()                      # 直接挑源，不起线程
    assert svc.source is None
    fix = svc.info()
    assert fix.source == "unavailable"
    assert fix.lat is None and fix.lon is None
    assert fix.message and "mock" in fix.message


def test_serial_source_reports_missing_pyserial_or_bad_port():
    src = SerialSource("COM_NOT_A_REAL_PORT_999", 9600, 0.1)
    assert src.ok is False
    assert src.error
    if not SERIAL_OK:                        # pragma: no cover
        assert "pyserial" in src.error


# ---------------------------------------------------------------- 回放

def test_file_source_replays_epochs(tmp_path):
    log = tmp_path / "sample.nmea"
    # newline="" 关掉 Windows 的换行翻译：否则 \r\n 会被写成 \r\r\n，
    # 而 splitlines() 把孤立的 \r 也算一次换行 —— 文件里凭空多出一堆空行。
    log.write_text(
        "# 现场录制的 NMEA 日志（这行注释会被忽略）\n"
        "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69\n"
        "$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E\n"
        "\n"
        "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A\n"
        "$GNRMC,120001,A,2809.5100,N,11256.4100,E,0.3,091.0,170926,,,A*6F\n",
        encoding="utf-8", newline="",
    )
    # epoch_s=0 → 不节流，测试里连续把整段日志抽干
    src = FileSource(str(log), loop=False, speed=1.0, epoch_s=0.0)
    assert src.ok is True
    assert len(src.chunks) == 2              # GGA/RMC 各出现一次即一个历元
    assert [len(c) for c in src.chunks] == [2, 2]

    svc = GnssService(_settings(GNSS_SOURCE="file", GNSS_FILE=str(log),
                                GNSS_REPLAY_SPEED=1000.0))
    svc.source = src
    svc.source_name = src.name
    svc.source_detail = src.detail
    svc._started = True                      # 只喂数据，不起后台线程

    for _ in range(10):
        ln = src.readline(0.1)
        if ln is None:
            break
        svc._acc.feed(ln)

    fix = svc.info()
    assert fix.source == "file"
    assert fix.real is True                  # 回放的是真实报文，算真实数据
    assert fix.usable is True
    assert fix.satellites_used == 10         # 取到的是后一个历元
    assert fix.lat == pytest.approx(28 + 9.51 / 60, abs=1e-6)
    assert fix.utc == "2026-09-17T12:00:01Z"


def test_file_source_handles_mcu_timestamped_log(tmp_path):
    """MCU 写卡的日志是「时间戳,语句」。曾经直接丢给 parse_line，
    行首不是 $ 就整行丢弃 —— 文件明明有内容却一句都读不到。"""
    log = tmp_path / "mcu.nmea"
    log.write_text(
        "# MCU 写卡日志\n"
        "2026-09-17T12:00:00.000Z,$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69\n"
        "2026-09-17T12:00:00.000Z,$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E\n"
        "2026-09-17T12:00:00.500Z,$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69\n"
        "2026-09-17T12:00:00.500Z,$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E\n",
        encoding="utf-8", newline="",
    )
    src = FileSource(str(log), loop=False, speed=1.0, epoch_s=0.0)
    assert src.ok is True, src.error
    assert len(src.chunks) == 2
    # 回放的是**剥掉时间戳的纯 NMEA** —— 累加器只认 $ 开头
    assert all(ln.startswith("$") for c in src.chunks for ln in c)
    # 历元间隔取自真实时间戳（0.5 秒），而不是默认的 1 秒
    assert src.epoch_s == pytest.approx(0.5)


def test_replaying_does_not_consume_the_source_chunks(tmp_path):
    """回放不能把源数据掏空。

    曾经 `self._buffer = chunk` 与 `chunks[i]` 是同一个列表对象，`pop(0)`
    会就地清空它 —— 循环回放时文件越读越少，而且 chunks 被改坏后无法复原。
    """
    log = tmp_path / "keep.nmea"
    log.write_text(
        "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69\n"
        "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A\n",
        encoding="utf-8", newline="",
    )
    src = FileSource(str(log), loop=True, speed=1.0, epoch_s=0.0)
    before = [list(c) for c in src.chunks]
    for _ in range(4):                       # 连读两轮
        assert src.readline(0.1) is not None
    assert src.chunks == before              # 源数据必须完好
    assert src.readline(0.1) is not None     # 还能继续循环


def test_file_source_paces_epochs(tmp_path):
    """历元之间要按 epoch_s/speed 等待，不能把整段日志一口气灌进去 ——
    否则"回放"就退化成"瞬间跑完"，时间轴对不上。"""
    log = tmp_path / "pace.nmea"
    log.write_text(
        "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69\n"
        "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A\n",
        encoding="utf-8", newline="",
    )
    src = FileSource(str(log), loop=False, speed=1.0, epoch_s=10.0)
    assert src.readline(0.1) is not None     # 第一个历元立即可取
    assert src.readline(0.1) is None         # 第二个历元要等 10 秒，现在取不到


def test_file_source_rejects_file_without_nmea(tmp_path):
    log = tmp_path / "junk.txt"
    log.write_text("这不是 NMEA\n随便写点东西\n", encoding="utf-8")
    src = FileSource(str(log), loop=True, speed=1.0)
    assert src.ok is False
    assert "没有可识别的 NMEA" in src.error


# ---------------------------------------------------------------- 健康度提示

class _SilentSource(GnssSource):
    """连着但一条有效语句都没有 —— 模拟"波特率不对/模块没定位"。"""

    name = "serial"

    def __init__(self, line: str) -> None:
        super().__init__()
        self.detail = "COM9 @ 115200"
        self._line = line

    def readline(self, timeout: float):
        return self._line


def _service_with(source: GnssSource) -> GnssService:
    svc = GnssService(_settings())
    svc.source = source
    svc.source_name = source.name
    svc.source_detail = source.detail
    svc._started = True
    return svc


def test_hint_mentions_baud_when_bytes_arrive_but_nothing_parses():
    """收到数据却解析不出语句 —— 这是波特率不对的典型症状，提示必须点出来。"""
    svc = _service_with(_SilentSource("\x00\xff乱码不是NMEA"))
    for _ in range(5):
        ln = svc.source.readline(0.1)
        svc._lock.acquire()
        svc.lines_read += 1
        svc._acc.feed(ln)
        svc._lock.release()
    fix = svc.info()
    assert fix.usable is False
    assert fix.message and "波特率" in fix.message


def test_hint_mentions_wiring_when_no_data_at_all():
    svc = _service_with(_SilentSource(""))
    svc.lines_read = 0
    fix = svc.info()
    assert fix.usable is False
    assert fix.message and "接线" in fix.message


def test_hint_mentions_waiting_for_satellites():
    """语句解析正常但还没定位解 —— 该等收星，不是故障。"""
    from server.services.nmea import NmeaAccumulator

    svc = GnssService(_settings())
    svc._acc = NmeaAccumulator()
    svc._acc.feed("$GNGGA,120000,2809.5000,N,11256.4000,E,0,00,99.9,50.0,M,10.0,M,,*50")
    svc.source = _SilentSource("x")
    svc.source_name = "serial"
    svc.lines_read = 3
    svc._started = True
    fix = svc.info()
    assert fix.usable is False
    assert fix.message and "收星" in fix.message


# ---------------------------------------------------------------- 重新挑源

def test_reload_repicks_source():
    svc = BeidouService(_settings(GNSS_SOURCE="mock"))
    svc.start()
    assert svc.info().source == "mock"
    fix = svc.reload()
    assert fix.source == "mock"      # 重新挑一次仍然是 mock，且没崩
    assert fix.usable is True
    svc.stop()


# ---------------------------------------------------------------- 坐标换算

def test_gsd_falls_back_to_estimate_and_flags_it():
    s = _settings(GEO_GSD_M_PER_PX=0.0, GEO_ALLOW_ESTIMATE=True, GEO_ESTIMATE_GSD=0.021)
    gsd, calibrated, estimated = s.geo_gsd
    assert (gsd, calibrated, estimated) == (0.021, False, True)
    s2 = _settings(GEO_GSD_M_PER_PX=0.05)
    assert s2.geo_gsd == (0.05, True, False)
    s3 = _settings(GEO_GSD_M_PER_PX=0.0, GEO_ALLOW_ESTIMATE=False)
    assert s3.geo_gsd == (0.0, False, False)


def test_enu_is_scale_correct():
    """100 像素 × 0.021 m/px = 2.1 米。旧实现会算出约 294 米（差 140 倍）。"""
    svc = _service(GEO_GSD_M_PER_PX=0.021)
    d = svc.enu(1060, 540, W, H)
    assert d["east_m"] == pytest.approx(2.1, abs=0.01)
    assert d["north_m"] == pytest.approx(0.0, abs=0.01)
    assert d["calibrated"] is True and d["estimated"] is False
    # 中心点是原点
    c = svc.enu(W / 2, H / 2, W, H)
    assert c["east_m"] == 0.0 and c["north_m"] == 0.0
    # 向上移动 → 北为正
    up = svc.enu(W / 2, H / 2 - 100, W, H)
    assert up["north_m"] == pytest.approx(2.1, abs=0.01)


def test_enu_without_gsd_gives_none():
    svc = _service(GEO_GSD_M_PER_PX=0.0, GEO_ALLOW_ESTIMATE=False)
    d = svc.enu(1060, 540, W, H)
    assert d["east_m"] is None and d["north_m"] is None
    assert d["calibrated"] is False


def test_geo_map_center_equals_fix_position():
    svc = _service(GEO_GSD_M_PER_PX=0.021)
    fix = svc.info()
    lon, lat = svc.geo_map(W / 2, H / 2, W, H, fix=fix)
    assert lon == pytest.approx(fix.lon, abs=1e-9)
    assert lat == pytest.approx(fix.lat, abs=1e-9)


def test_geo_map_and_enu_agree_on_the_same_scale():
    """这条是 140 倍比例尺矛盾的回归锁：经纬度偏移换算回来必须等于 ENU 的米数。"""
    gsd = 0.021
    svc = _service(GEO_GSD_M_PER_PX=gsd)
    fix = svc.info()
    # 偏移取大一点：NMEA 的坐标分辨率是 1e-4 分（≈0.16 m），
    # 偏移太小的话量化误差会淹没真值，测不出比例尺对不对。
    cx, cy = W / 2 + 800, H / 2 - 600

    lon, lat = svc.geo_map(cx, cy, W, H, fix=fix)
    d = svc.enu(cx, cy, W, H)

    east_from_geo = (lon - fix.lon) * M_PER_DEG_LAT * math.cos(math.radians(fix.lat))
    north_from_geo = (lat - fix.lat) * M_PER_DEG_LAT

    # ENU 侧是精确的
    assert d["east_m"] == pytest.approx(800 * gsd, abs=0.01)
    assert d["north_m"] == pytest.approx(600 * gsd, abs=0.01)
    # 经纬度侧换算回来必须与 ENU 一致（留 3% 给坐标量化）
    assert east_from_geo == pytest.approx(d["east_m"], rel=0.03)
    assert north_from_geo == pytest.approx(d["north_m"], rel=0.03)
    # 锁死旧 bug：曾经的实现会算出约 140 倍的距离
    assert east_from_geo / d["east_m"] == pytest.approx(1.0, abs=0.05)


def test_geo_map_returns_none_without_fix():
    """没有有效定位时不给经纬度 —— 宁可空着，也不编。"""
    svc = _service(GNSS_MOCK_FIX_QUALITY=0, GEO_GSD_M_PER_PX=0.021)
    assert svc.geo_map(100, 100, W, H) == (None, None)


def test_geo_map_returns_none_without_gsd():
    svc = _service(GEO_GSD_M_PER_PX=0.0, GEO_ALLOW_ESTIMATE=False)
    assert svc.geo_map(100, 100, W, H) == (None, None)


def test_mock_source_has_no_pending_epoch_after_start():
    """MockSource 第一次调用就该立刻吐数据，不能让界面空等一个周期。"""
    src = MockSource(_settings())
    assert src.readline(0.1) is not None
