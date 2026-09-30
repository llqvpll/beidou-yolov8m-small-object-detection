"""带时间戳的北斗日志：格式识别、时间索引、按时刻查询。

这套东西是给"单片机写卡 → 事后和视频对齐"这条路线用的：
日志格式由硬件决定，软件必须容错，所以先锁住"能认几种写法"。
"""
from datetime import datetime, timezone

import pytest

from server.services.gnss_track import (
    NmeaTrack,
    epoch_from_utc,
    parse_stamped_line,
    utc_from_epoch,
)

GGA_A = "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69"
RMC_A = "$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E"
GGA_B = "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A"
RMC_B = "$GNRMC,120001,A,2809.5100,N,11256.4100,E,0.3,091.0,170926,,,A*6F"

DAY = datetime(2026, 9, 17, tzinfo=timezone.utc)


# ---------------------------------------------------------------- 格式识别

@pytest.mark.parametrize("line,kind,absolute", [
    (f"2026-09-17 12:00:00.123,{GGA_A}", "iso", True),
    (f"2026-09-17T12:00:00.123Z,{GGA_A}", "iso", True),
    (f"1758100000.123,{GGA_A}", "unix", True),
    (f"1758100000123,{GGA_A}", "unix_ms", True),
    (f"12:00:00.123,{GGA_A}", "hms", False),
    (f"123456,{GGA_A}", "ticks", False),
    (GGA_A, "none", False),
])
def test_recognises_stamp_formats(line, kind, absolute):
    st = parse_stamped_line(line, day_anchor=DAY)
    assert st is not None, line
    assert st.kind == kind
    assert st.absolute is absolute
    assert st.nmea == GGA_A


@pytest.mark.parametrize("sep", [",", "\t", " | ", ";"])
def test_stamp_separators(sep):
    st = parse_stamped_line(f"1758100000.0{sep}{GGA_A}")
    assert st is not None and st.kind == "unix"


def test_iso_timestamp_value_is_exact():
    st = parse_stamped_line(f"2026-09-17T12:00:00.500Z,{GGA_A}")
    assert st.epoch_s == pytest.approx(epoch_from_utc("2026-09-17T12:00:00.500"), abs=1e-6)


def test_unix_ms_converts_to_seconds():
    st = parse_stamped_line(f"1758100000123,{GGA_A}")
    assert st.epoch_s == pytest.approx(1758100000.123, abs=1e-6)


def test_hms_uses_day_anchor_for_date():
    st = parse_stamped_line(f"12:00:00,{GGA_A}", day_anchor=DAY)
    assert st.epoch_s == pytest.approx(DAY.replace(hour=12).timestamp(), abs=1e-6)


def test_bare_number_is_ticks_not_a_fake_date():
    """上电毫秒数不能被当成 Unix 时间 —— 那会得到 1970 年的坐标时间。"""
    st = parse_stamped_line(f"123456,{GGA_A}")
    assert st.kind == "ticks"
    assert st.epoch_s is None
    assert st.absolute is False


@pytest.mark.parametrize("line", [
    "",
    "   ",
    "2026-09-17 12:00:00.123,这不是NMEA",
    "2026-09-17 12:00:00.123",
    "随便一行日志",
    "2026-99-99 12:00:00,$GNGGA,1,2*00",
])
def test_rejects_unusable_lines(line):
    assert parse_stamped_line(line) is None


# ---------------------------------------------------------------- 轨迹装载

def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return str(path)


def test_track_builds_time_indexed_points(tmp_path):
    p = _write(tmp_path / "a.nmea", [
        f"2026-09-17T12:00:00Z,{GGA_A}",
        f"2026-09-17T12:00:00Z,{RMC_A}",
        f"2026-09-17T12:00:01Z,{GGA_B}",
        f"2026-09-17T12:00:01Z,{RMC_B}",
    ])
    tr = NmeaTrack.load(p)
    assert tr.count == 2
    assert tr.absolute is True
    assert tr.stamp_kind == "iso"
    assert tr.duration_s == pytest.approx(1.0)
    assert tr.points[0]["lat"] == pytest.approx(28 + 9.5 / 60, abs=1e-6)
    assert tr.points[1]["satellites_used"] == 10


def test_track_skips_junk_lines_without_losing_good_ones(tmp_path):
    p = _write(tmp_path / "b.nmea", [
        "# 头部注释",
        "启动完成",
        f"2026-09-17T12:00:00Z,{GGA_A}",
        f"2026-09-17T12:00:00Z,{RMC_A}",
        "",
        f"2026-09-17T12:00:01Z,{GGA_B}",
        f"2026-09-17T12:00:01Z,{RMC_B}",
    ])
    tr = NmeaTrack.load(p)
    assert tr.count == 2
    # 「启动完成」是真正认不出的行；注释与空行单独计数，
    # 否则用户自己写的文件头会被报成错误，真正的问题反而被淹掉。
    assert tr.skipped_lines == 1
    assert tr.comment_lines == 2        # 注释 / 空行
    assert tr.total_lines == 7


def test_track_without_timestamps_still_scrubbable(tmp_path):
    """没有时间戳也要能用：按顺序铺开相对时间，但必须标明"非绝对"。"""
    p = _write(tmp_path / "c.nmea", [GGA_A, RMC_A, GGA_B, RMC_B])
    tr = NmeaTrack.load(p)
    assert tr.count == 2
    assert tr.absolute is False
    assert tr.stamp_kind == "none"
    assert tr.t0 == 0.0 and tr.t1 == pytest.approx(1.0)
    # 时间轴是编出来的，必须告诉用户"这不是真实时间"，否则他拿去对齐视频会一头雾水
    assert tr.summary()["message"] is not None


def test_track_relative_time_is_flagged_with_reason(tmp_path):
    p = _write(tmp_path / "d.nmea", [
        f"1000,{GGA_A}", f"1000,{RMC_A}",
        f"2000,{GGA_B}", f"2000,{RMC_B}",
    ])
    tr = NmeaTrack.load(p)
    assert tr.count == 2
    assert tr.absolute is False
    assert tr.stamp_kind == "ticks"
    s = tr.summary()
    assert s["absolute"] is False
    assert s["message"] and "绝对时间" in s["message"]


def test_track_preserves_device_tick_intervals(tmp_path):
    """设备计时器虽然不知道起点，但**间隔是真的**，必须还原成真实节奏。

    全部按"每点 1 秒"铺开的话，回放速度和现场完全对不上。
    """
    p = _write(tmp_path / "ticks.nmea", [
        f"0,{GGA_A}", f"0,{RMC_A}",
        f"5000,{GGA_B}", f"5000,{RMC_B}",
    ])
    tr = NmeaTrack.load(p)
    assert tr.absolute is False
    assert tr.t0 == 0.0
    assert tr.duration_s == pytest.approx(5.0)      # 5000 ms → 5 秒，不是 1 秒


def test_track_handles_second_based_ticks(tmp_path):
    """按秒走的计时器不能被误当成毫秒（间隔中位数只有 1~2）。"""
    p = _write(tmp_path / "secs.nmea", [
        f"100,{GGA_A}", f"100,{RMC_A}",
        f"102,{GGA_B}", f"102,{RMC_B}",
    ])
    tr = NmeaTrack.load(p)
    assert tr.duration_s == pytest.approx(2.0)


def test_empty_log_gives_explicit_summary(tmp_path):
    p = _write(tmp_path / "e.nmea", ["什么都\n".strip(), "也不是"])
    tr = NmeaTrack.load(p)
    assert tr.count == 0
    assert tr.summary()["message"] == "日志里没有可用的定位点。"


# ---------------------------------------------------------------- 按时刻查询

def _track(tmp_path):
    p = _write(tmp_path / "t.nmea", [
        f"2026-09-17T12:00:00Z,{GGA_A}", f"2026-09-17T12:00:00Z,{RMC_A}",
        f"2026-09-17T12:00:10Z,{GGA_B}", f"2026-09-17T12:00:10Z,{RMC_B}",
    ])
    return NmeaTrack.load(p)


def test_at_interpolates_between_points(tmp_path):
    """拖动时间轴要连续变化，不能一格一格跳。"""
    tr = _track(tmp_path)
    mid = tr.at(tr.t0 + 5.0)
    a, b = tr.points[0], tr.points[1]
    assert mid["lat"] == pytest.approx((a["lat"] + b["lat"]) / 2, abs=1e-6)
    assert mid["alt"] == pytest.approx((a["alt"] + b["alt"]) / 2, abs=1e-3)
    assert mid["interpolated"] is True


def test_at_clamps_outside_range(tmp_path):
    tr = _track(tmp_path)
    assert tr.at(tr.t0 - 999)["lat"] == pytest.approx(tr.points[0]["lat"], abs=1e-9)
    assert tr.at(tr.t1 + 999)["lat"] == pytest.approx(tr.points[1]["lat"], abs=1e-9)


def test_at_keeps_discrete_fields_from_nearest_point(tmp_path):
    """定位质量、卫星数是离散量，插值出来"9.5 颗卫星"是错的。"""
    tr = _track(tmp_path)
    mid = tr.at(tr.t0 + 5.0)
    assert mid["satellites_used"] in (9, 10)
    assert mid["fix_quality"] == 1


def test_at_can_disable_interpolation(tmp_path):
    tr = _track(tmp_path)
    mid = tr.at(tr.t0 + 5.0, interpolate=False)
    assert mid["lat"] == pytest.approx(tr.points[0]["lat"], abs=1e-9)


def test_video_time_with_absolute_origin(tmp_path):
    """情形一：日志是绝对时间，视频起始时刻已知 → origin 直接取视频起始的 Unix 秒。"""
    tr = _track(tmp_path)
    video_start = tr.t0 + 3.0            # 视频从日志起点后 3 秒开始录
    tr.align(video_start)
    assert tr.aligned is True
    # 取精确点用 interpolate=False；默认是插值（拖动时间轴要连续变化）
    assert tr.at_video_time(0.0, interpolate=False)["lat"] == pytest.approx(
        tr.points[0]["lat"], abs=1e-9)
    assert tr.at_video_time(10.0, interpolate=False)["lat"] == pytest.approx(
        tr.points[1]["lat"], abs=1e-9)
    # 视频第 3 秒 = 日志第 6 秒 → 落在两个点之间（60% 处）
    mid = tr.at_video_time(3.0)
    a, b = tr.points[0]["lat"], tr.points[1]["lat"]
    assert mid["lat"] == pytest.approx(a + (b - a) * 0.6, abs=1e-6)
    assert tr.summary()["shift_s"] == pytest.approx(3.0)


def test_video_time_with_relative_sync_delta(tmp_path):
    """情形二：日志是相对时间，靠人工同步点对齐 → 以日志起点为基准微调。"""
    p = tmp_path / "rel.nmea"
    p.write_text(
        "\n".join([f"0,{GGA_A}", f"0,{RMC_A}", f"10000,{GGA_B}", f"10000,{RMC_B}"]) + "\n",
        encoding="utf-8", newline="",
    )
    tr = NmeaTrack.load(str(p))
    assert tr.aligned is False
    # 未对齐时退化为"视频 0 秒 = 日志起点"，但必须标明尚未对齐
    assert tr.at_video_time(0.0)["aligned"] is False
    tr.align_by_delta(2.5)
    assert tr.summary()["shift_s"] == pytest.approx(2.5)
    v = tr.at_video_time(0.0, interpolate=False)
    assert v["aligned"] is True
    assert v["lat"] == pytest.approx(tr.points[0]["lat"], abs=1e-9)


def test_at_on_empty_track_returns_none(tmp_path):
    tr = NmeaTrack.load(_write(tmp_path / "z.nmea", ["无内容"]))
    assert tr.at(0.0) is None


# ---------------------------------------------------------------- 摘要与工具

def test_summary_reports_bbox_and_usable(tmp_path):
    tr = _track(tmp_path)
    s = tr.summary()
    assert s["count"] == 2
    assert s["usable_points"] == 2
    assert s["bbox"]["lat_min"] < s["bbox"]["lat_max"]
    assert s["bbox"]["lon_min"] < s["bbox"]["lon_max"]


def test_epoch_utc_roundtrip():
    t = epoch_from_utc("2026-09-17T12:00:00")
    assert t is not None
    assert utc_from_epoch(t) == "2026-09-17T12:00:00Z"
    assert epoch_from_utc("") is None
    assert epoch_from_utc("不是时间") is None
    assert utc_from_epoch(None) is None


# ---------------------------------------------------------------- GPX 导入

GPX_SIMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="BeidouTracker">
<trk><name>track</name><trkseg>
<trkpt lat="29.0437780" lon="111.6702250"><ele>36.0</ele><time>2026-09-25T09:12:30+08:00</time></trkpt>
<trkpt lat="29.0437800" lon="111.6705000"><ele>36.2</ele><time>2026-09-25T09:12:35+08:00</time></trkpt>
<trkpt lat="29.0438000" lon="111.6708000"><ele>36.5</ele><time>2026-09-25T09:12:40+08:00</time></trkpt>
</trkseg></trk></gpx>
"""

# 带命名空间（真实设备/软件导出的 GPX 基本都带）
GPX_NS = """<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">
<trk><trkseg>
<trkpt lat="28.1575000" lon="112.9400000"><ele>50.0</ele><time>2026-09-17T12:00:00Z</time></trkpt>
<trkpt lat="28.1576000" lon="112.9401000"><ele>50.5</ele><time>2026-09-17T12:00:01Z</time></trkpt>
</trkseg></trk></gpx>
"""

GPX_NO_TIME = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1"><trk><trkseg>
<trkpt lat="28.1575000" lon="112.9400000"><ele>50.0</ele></trkpt>
<trkpt lat="28.1576000" lon="112.9401000"><ele>50.5</ele></trkpt>
<trkpt lat="28.1577000" lon="112.9402000"><ele>51.0</ele></trkpt>
</trkseg></trk></gpx>
"""


def _write_raw(path, text):
    path.write_text(text, encoding="utf-8", newline="")
    return str(path)


def test_gpx_imports_points_and_absolute_time(tmp_path):
    """SD 卡里就是 .gpx：能直接导入，且时间是绝对时间（可与视频对齐）。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "t.gpx", GPX_SIMPLE))
    assert tr.count == 3
    assert tr.absolute is True
    assert tr.stamp_kind == "gpx"
    assert tr.duration_s == 10.0
    assert tr.epoch_interval_s == pytest.approx(5.0)
    assert tr.points[0]["lat"] == pytest.approx(29.0437780)
    assert tr.points[0]["alt"] == pytest.approx(36.0)


def test_gpx_local_offset_time_is_converted(tmp_path):
    """带 +08:00 的本地时间要按真实时刻换算，不能当成 UTC 直接读。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "t.gpx", GPX_SIMPLE))
    s = tr.summary()
    # 2026-09-25 09:12:30+08:00 == 2026-09-25 01:12:30Z
    assert utc_from_epoch(s["t0"]) == "2026-09-25T01:12:30Z"
    assert utc_from_epoch(s["t1"]) == "2026-09-25T01:12:40Z"


def test_gpx_with_namespace(tmp_path):
    """带 xmlns 的 GPX 不能被命名空间坑掉——这是最常见的导出格式。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "ns.gpx", GPX_NS))
    assert tr.count == 2
    assert tr.points[0]["lat"] == pytest.approx(28.1575)
    assert tr.absolute is True


def test_gpx_without_time_falls_back_to_order(tmp_path):
    """没有 <time> 的 GPX：按点的先后顺序铺开，并且明确标记为非绝对时间。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "nt.gpx", GPX_NO_TIME))
    assert tr.count == 3
    assert tr.absolute is False
    assert tr.duration_s == 2.0            # 退化为每点 1 秒
    assert any("没有 <time>" in h for h in tr.diagnose()["hints"])


def test_gpx_missing_fields_stay_none(tmp_path):
    """GPX 里没有卫星数/HDOP/定位质量 —— 一律留 None，绝不编造。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "t.gpx", GPX_SIMPLE))
    p = tr.points[0]
    for k in ("satellites_used", "hdop", "vdop", "pdop", "fix_quality", "geoid_sep"):
        assert p[k] is None, k
    assert p["usable"] is True              # 坐标有效即可用
    assert p["alt"] == pytest.approx(36.0)  # 有的字段要照实给


def test_gpx_speed_is_derived_from_coordinates(tmp_path):
    """GPX 没有速度字段，但可以从相邻点坐标反算（真实可推，不是编造）。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "t.gpx", GPX_SIMPLE))
    p = tr.points[0]
    assert p["speed_kmh"] is not None
    assert 0 < p["speed_kmh"] < 60
    assert p["course"] is not None
    assert 0 <= p["course"] < 360


def test_malformed_gpx_does_not_raise(tmp_path):
    """XML 坏了要给出诊断，不能把 500 抛给用户。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "bad.gpx", "<gpx><trk><trkseg>"))
    assert tr.count == 0
    d = tr.diagnose()
    assert d["ok"] is False
    assert any("XML 解析失败" in s for s in tr.skipped_samples)


def test_kml_is_not_mistaken_for_gpx(tmp_path):
    """KML 也是 XML，但不能被当成 GPX 认进来（否则只会多一层误导）。"""
    tr = NmeaTrack.load(_write_raw(tmp_path / "a.kml",
                                   '<?xml version="1.0"?><kml><Placemark><Point>'
                                   '<coordinates>112.94,28.1575</coordinates>'
                                   '</Point></Placemark></kml>'))
    assert tr.count == 0


def test_nmea_still_goes_through_original_path(tmp_path):
    """加了 GPX 分支不能把原有的 NMEA 路径挤坏。"""
    tr = _track(tmp_path)
    assert tr.count == 2
    assert tr.stamp_kind == "iso"
