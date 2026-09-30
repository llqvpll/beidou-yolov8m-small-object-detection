"""存储卡日志的导入回执与轨迹服务。

重点在**诊断**：MCU 写卡的格式事先无法穷举，所以"文件丢进去没反应"必须能自我解释，
而不是让用户反复猜。这里锁住诊断内容的正确性。
"""
from __future__ import annotations

import pytest

from server.services.gnss_track import NmeaTrack
from server.services.track_store import TrackService

GGA_A = "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69"
RMC_A = "$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E"
GGA_B = "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A"
RMC_B = "$GNRMC,120001,A,2809.5100,N,11256.4100,E,0.3,091.0,170926,,,A*6F"
GGA_C = "$GNGGA,120002,2809.5200,N,11256.4200,E,1,10,0.6,51.0,M,10.0,M,,*6B"
RMC_C = "$GNRMC,120002,A,2809.5200,N,11256.4200,E,0.4,092.0,170926,,,A*68"

# 用户实际路线的样子：MCU 把「时间戳 + 原始语句」写进卡。
# 时间戳与语句里的 UTC 一致（12:00:00）—— 不一致的情形另有用例专测。
MCU_LOG = "\n".join([
    "# ATGM336H 开机",
    "2026-09-17T12:00:00.000Z," + GGA_A,
    "2026-09-17T12:00:00.000Z," + RMC_A,
    "2026-09-17T12:00:01.000Z," + GGA_B,
    "2026-09-17T12:00:01.000Z," + RMC_B,
    "2026-09-17T12:00:02.000Z," + GGA_C,
    "2026-09-17T12:00:02.000Z," + RMC_C,
]) + "\n"


def _write(tmp_path, text, name="card.log"):
    """写日志夹具。**必须 newline=""**：Windows 上 write_text 会把 \n 翻成 \r\n，
    夹具里再写 \r\n 就成了 \r\r\n，而 splitlines() 把孤立 \r 也算换行 → 凭空多出空行。
    """
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p


def _svc():
    from server.config import get_settings

    return TrackService(get_settings())


# ---------------------------------------------------------------- 诊断回执

def test_diagnose_reports_stamp_format(tmp_path):
    tr = NmeaTrack.load(str(_write(tmp_path, MCU_LOG)))
    d = tr.diagnose()
    assert d["ok"] is True
    assert d["stamp_kind"] == "iso"
    assert d["absolute"] is True
    assert "绝对时间" in d["stamp_kind_zh"]
    assert d["count"] == 3
    assert d["duration_s"] == 2.0
    assert d["epoch_interval_s"] == 1.0
    assert d["first_utc"] == "2026-09-17T12:00:00Z"
    assert d["last_utc"] == "2026-09-17T12:00:02Z"


def test_diagnose_keeps_sample_lines_so_format_mismatch_is_visible(tmp_path):
    """认不出的行必须原样回显 —— 这是用户唯一能自己判断"差在哪"的线索。"""
    log = MCU_LOG.replace("2026-09-17T12:00:02.000Z,", "T=2026/09/17 12:00:02,", 1)
    tr = NmeaTrack.load(str(_write(tmp_path, log)))
    d = tr.diagnose()
    assert d["skipped_lines"] >= 1
    assert any("T=2026/09/17" in s for s in d["skipped_samples"])
    assert d["sample_lines"], "头几行原文也要留着"


def test_comments_and_blank_lines_are_counted_separately(tmp_path):
    """自己写的文件头注释不该被报成"没解析成定位点"。

    混在一起报数会吓到用户（"怎么有 3 行错了"），真正的问题反而被淹掉。
    """
    log = "# 开机\n; 说明\n\n" + "\n".join([
        "2026-09-17T12:00:00.000Z," + GGA_A,
        "2026-09-17T12:00:00.000Z," + RMC_A,
        "这不是注释也不是语句",
    ]) + "\n"
    d = NmeaTrack.load(str(_write(tmp_path, log))).diagnose()
    assert d["comment_lines"] == 3          # # 行、; 行、空行
    assert d["skipped_lines"] == 1          # 只有那一行乱码
    assert d["count"] == 1
    assert any("注释" in h for h in d["hints"])
    assert not any("# 开机" in s for s in d["skipped_samples"]), "注释不该出现在未识别列表里"


def test_comment_only_log_has_no_skipped_lines(tmp_path):
    d = NmeaTrack.load(str(_write(tmp_path, "# 只有注释\n\n; 和分号注释\n"))).diagnose()
    assert d["skipped_lines"] == 0
    assert d["comment_lines"] == 3
    assert d["count"] == 0


def test_diagnose_explains_missing_timestamp(tmp_path):
    log = "\n".join([GGA_A, RMC_A, GGA_B, RMC_B]) + "\n"
    d = NmeaTrack.load(str(_write(tmp_path, log))).diagnose()
    assert d["stamp_kind"] == "none"
    assert d["absolute"] is False
    assert any("没有时间戳" in h for h in d["hints"])
    assert any("MCU" in h for h in d["hints"])


def test_diagnose_explains_relative_time(tmp_path):
    log = "\n".join([f"12345,{GGA_A}", f"12345,{RMC_A}",
                     f"13345,{GGA_B}", f"13345,{RMC_B}"]) + "\n"
    d = NmeaTrack.load(str(_write(tmp_path, log))).diagnose()
    assert d["stamp_kind"] == "ticks"
    assert d["absolute"] is False
    assert any("相对时间" in h for h in d["hints"])
    assert d["first_utc"] is None, "相对时间没有可显示的 UTC"


def test_diagnose_on_empty_log_is_actionable(tmp_path):
    d = NmeaTrack.load(str(_write(tmp_path, "hello\nworld\n"))).diagnose()
    assert d["ok"] is False
    assert d["count"] == 0
    assert any("GGA" in h for h in d["hints"])


# ---------------------------------------------------------------- 时钟偏移

def test_clock_offset_is_zero_when_stamp_matches_sentence_utc(tmp_path):
    d = NmeaTrack.load(str(_write(tmp_path, MCU_LOG))).diagnose()
    assert d["clock_offset_s"] == pytest.approx(0.0)
    assert not any("本地时区" in h for h in d["hints"])


def test_clock_offset_detects_local_time_rtc(tmp_path):
    """MCU 的 RTC 常按本地时区走，而语句里的 UTC 是卫星给的。

    差整整 8 小时时，拿日志时间戳去对本地时间的视频反而能对上，
    拿语句 UTC 去对就会整体偏 8 小时 —— 不把这个差值报出来，用户只会觉得"对不齐"。
    """
    log = "\n".join([
        "2026-09-17T20:00:00.000Z," + GGA_A,   # 本地 UTC+8，比语句 UTC 快 8 小时
        "2026-09-17T20:00:00.000Z," + RMC_A,
        "2026-09-17T20:00:01.000Z," + GGA_B,
        "2026-09-17T20:00:01.000Z," + RMC_B,
    ]) + "\n"
    d = NmeaTrack.load(str(_write(tmp_path, log))).diagnose()
    assert d["clock_offset_s"] == pytest.approx(8 * 3600.0, abs=1.0)
    assert any("本地时区" in h for h in d["hints"])
    assert any("8 小时" in h for h in d["hints"])


def test_clock_offset_reports_non_whole_hour_drift(tmp_path):
    log = "\n".join([
        "2026-09-17T12:00:37.000Z," + GGA_A,
        "2026-09-17T12:00:37.000Z," + RMC_A,
    ]) + "\n"
    d = NmeaTrack.load(str(_write(tmp_path, log))).diagnose()
    assert d["clock_offset_s"] == pytest.approx(37.0, abs=1.0)
    assert any("非整点偏移" in h for h in d["hints"])


def test_clock_offset_is_none_for_relative_time(tmp_path):
    log = "\n".join([f"12345,{GGA_A}", f"12345,{RMC_A}"]) + "\n"
    assert NmeaTrack.load(str(_write(tmp_path, log))).diagnose()["clock_offset_s"] is None


# ---------------------------------------------------------------- 抽稀折线

def test_polyline_keeps_both_ends():
    tr = NmeaTrack(points=[{"t": float(i), "lat": 28.0 + i * 1e-4, "lon": 112.0}
                           for i in range(1000)])
    tr.t0, tr.t1 = 0.0, 999.0
    poly = tr.polyline(max_points=50)
    assert len(poly) <= 51
    assert poly[0]["t"] == 0.0
    assert poly[-1]["t"] == 999.0


def test_polyline_is_evenly_spaced_not_sliced():
    """等间隔取样。简单切片会让轨迹开头密、结尾疏，看起来像"速度变了"。"""
    tr = NmeaTrack(points=[{"t": float(i), "lat": 28.0, "lon": 112.0} for i in range(900)])
    tr.t0, tr.t1 = 0.0, 899.0
    poly = tr.polyline(max_points=100)
    gaps = [b["t"] - a["t"] for a, b in zip(poly, poly[1:])]
    assert min(gaps) >= 8, f"最小间隔 {min(gaps)} 太小，说明不是等间隔取样"
    assert max(gaps) <= 11


def test_polyline_skips_points_without_coordinates():
    tr = NmeaTrack(points=[
        {"t": 0.0, "lat": None, "lon": None},
        {"t": 1.0, "lat": 28.1, "lon": 112.1},
        {"t": 2.0, "lat": 28.2, "lon": 112.2},
    ])
    poly = tr.polyline()
    assert [p["t"] for p in poly] == [1.0, 2.0]


def test_polyline_empty_track():
    assert NmeaTrack().polyline() == []


# ---------------------------------------------------------------- 服务

def test_service_import_returns_receipt(tmp_path):
    svc = _svc()
    r = svc.load(_write(tmp_path, MCU_LOG), name="card.log")
    assert r["ok"] is True
    assert r["name"] == "card.log"
    assert r["diagnose"]["count"] == 3
    assert r["summary"]["count"] == 3
    assert svc.state()["loaded"] is True


def test_service_state_is_complete_for_the_ui(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    st = svc.state()
    assert st["loaded"] is True
    assert st["summary"]["count"] == 3
    assert st["diagnose"]["stamp_kind"] == "iso"
    assert len(st["polyline"]) == 3


def test_service_empty_state_is_not_an_error():
    st = _svc().state()
    assert st["loaded"] is False
    assert st["summary"] is None
    assert st["polyline"] == []


def test_service_load_missing_file_reports_instead_of_raising(tmp_path):
    svc = _svc()
    r = svc.load(tmp_path / "nope.log")
    assert r["ok"] is False
    assert r["error"]
    assert svc.state()["loaded"] is False


def test_at_before_import_is_none():
    assert _svc().at(0.0) is None


def test_at_unaligned_falls_back_to_track_start_and_says_so(tmp_path):
    """没对齐时按"视频 0 秒 = 日志起点"给结果，但必须标 aligned=False。

    假装已经对齐会让用户以为时间轴是准的 —— 比不给结果更糟。
    """
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    fix = svc.at(0.5)
    assert fix is not None
    assert fix["aligned"] is False
    assert fix["lat"] == pytest.approx(28 + 9.505 / 60, abs=1e-6)


def test_align_by_utc_makes_it_aligned(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    r = svc.align(utc="2026-09-17T12:00:00Z")
    assert r["ok"] is True
    assert r["used"] == "utc"
    assert r["summary"]["aligned"] is True
    fix = svc.at(1.0)
    assert fix["aligned"] is True
    assert fix["utc"] == "2026-09-17T12:00:01Z"


def test_align_by_delta_shifts_the_track(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    r = svc.align(delta_s=1.0)
    assert r["used"] == "delta_s"
    assert r["summary"]["shift_s"] == pytest.approx(1.0)
    assert svc.at(0.0)["utc"] == "2026-09-17T12:00:01Z"


def test_align_rejects_unparseable_utc_without_touching_state(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    r = svc.align(utc="昨天下午")
    assert r["ok"] is False
    assert "认不出" in r["error"]
    assert svc.track.aligned is False, "失败不能留下半个对齐状态"


def test_align_without_arguments_resets_alignment(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    svc.align(utc="2026-09-17T04:00:00Z")
    r = svc.align()
    assert r["used"] == "reset"
    assert r["summary"]["aligned"] is False


def test_align_priority_is_reported_not_silently_ignored(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    r = svc.align(utc="2026-09-17T04:00:00Z", delta_s=99.0)
    assert r["used"] == "utc", "多给了参数要说明用了哪个，不能静默挑一个"


def test_align_without_track_is_refused():
    r = _svc().align(delta_s=1.0)
    assert r["ok"] is False
    assert r["error"]


def test_clear_resets_everything(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    svc.clear()
    assert svc.track is None
    assert svc.state()["loaded"] is False
    assert svc.at(0.0) is None


def test_at_carries_satellite_detail_for_the_sky_plot(tmp_path):
    """星空图要能按时刻回放，轨迹点里必须带卫星明细（含 talker）。"""
    log = "\n".join([
        "2026-09-17T04:00:00.000Z," + GGA_A,
        "2026-09-17T04:00:00.000Z," + RMC_A,
        "2026-09-17T04:00:00.000Z,$GPGSV,1,1,04,03,03,111,00,04,15,270,20,06,01,010,00,13,06,292,00*70",
    ]) + "\n"
    svc = _svc()
    svc.load(_write(tmp_path, log))
    fix = svc.at(0.0)
    sats = fix["satellites"]
    assert len(sats) == 4
    assert {s["talker"] for s in sats} == {"GP"}
    assert sats[0]["prn"] == "03"
    assert sats[1]["snr"] == 20.0


def test_known_suffix_helper():
    assert TrackService.known_suffix("card.log") is True
    assert TrackService.known_suffix("CARD.NMEA") is True
    assert TrackService.known_suffix("weird") is False
    assert TrackService.known_suffix(None) is False


# ---------------------------------------------------------------- 本地插值序列

def _synth_log(epochs: int, *, gsv: bool = True, gsv_every: int = 1,
               const_gsv: bool = False) -> str:
    """造一份 N 历元的日志，校验和用真算的（否则会被 verify 丢掉，测试就白测了）。

    ``const_gsv``：GSV 内容恒定（用来验证去重内联确实生效）。
    """
    from server.services.nmea import checksum_of

    def s(payload: str) -> str:
        return "$" + payload + "*" + checksum_of(payload)

    lines = []
    for i in range(epochs):
        hh, mm, ss = 12, 0, i % 60
        t = f"2026-09-17T{hh:02d}:{mm:02d}:{ss:02d}.{i // 60:03d}Z"
        lat = 28 + i * 0.0001
        lines.append(t + "," + s(
            f"GNGGA,{hh:02d}{mm:02d}{ss:02d},{_dm(lat)},N,{_dm(112 + i * 0.0001)},E,1,09,0.8,50.0,M,10.0,M,,"))
        lines.append(t + "," + s(
            f"GNRMC,{hh:02d}{mm:02d}{ss:02d},A,{_dm(lat)},N,{_dm(112 + i * 0.0001)},E,0.2,090.0,170926,,,A"))
        if gsv and i % gsv_every == 0:
            # 每 gsv_every 个历元换一次星座：内联表应该只多出一条，而不是每点一条
            prn = 3 if const_gsv else 3 + (i // gsv_every) % 5
            lines.append(t + "," + s(
                f"GPGSV,1,1,04,{prn:02d},03,111,20,04,15,270,25,06,01,010,00,13,06,292,00"))
    return "\n".join(lines) + "\n"


def _dm(v: float) -> str:
    """十进制度 → NMEA 的 ddmm.mmmm（这里只做北纬/东经，够用了）。"""
    d = int(v)
    return f"{d:02d}{(v - d) * 60:07.4f}"


def test_scrub_points_are_thinned_but_keep_the_last_epoch(tmp_path):
    """抽稀不能把尾巴切掉：切了时间轴末尾就有一段没读数。"""
    tr = NmeaTrack.load(str(_write(tmp_path, _synth_log(100), name="long.log")))
    pts, stride, _ = tr.scrub_points(max_points=10)
    assert stride == 10
    assert len(pts) == 11, "10 个等间隔取样 + 末点"
    assert pts[-1]["t"] == tr.points[-1]["t"]


def test_scrub_satellites_are_interned_not_duplicated(tmp_path):
    """卫星明细必须去重成一张表 —— 逐点各存一份会让载荷膨胀几十倍。

    这里是同一份星座连续 40 个历元：内联后应该**只有 1 条**。
    """
    tr = NmeaTrack.load(str(_write(tmp_path, _synth_log(40, const_gsv=True), name="same.log")))
    pts, _, sats = tr.scrub_points()
    assert len(sats) == 1, f"同样的 GSV 列表应只留一条，实际 {len(sats)} 条"
    assert all(p["sat"] == 0 for p in pts if p["sat"] is not None)
    assert sats[0][0]["prn"] == "03"


def test_scrub_payload_does_not_grow_with_epoch_count(tmp_path):
    """真正的目的：日志变长时载荷**只按点数线性增长**，不按"点数 × 卫星数"增长。"""
    import json

    a = NmeaTrack.load(str(_write(tmp_path, _synth_log(20, const_gsv=True), name="a.log")))
    b = NmeaTrack.load(str(_write(tmp_path, _synth_log(200, const_gsv=True), name="b.log")))
    pa, _, sa = a.scrub_points()
    pb, _, sb = b.scrub_points()
    la = len(json.dumps({"p": pa, "s": sa}))
    lb = len(json.dumps({"p": pb, "s": sb}))
    # 历元数 ×10，载荷应远小于 ×10（内联表恒定，只有点数组在长）
    assert len(sb) == len(sa) == 1
    assert lb < la * 11


def test_scrub_satellite_table_grows_with_distinct_constellations(tmp_path):
    tr = NmeaTrack.load(str(_write(tmp_path, _synth_log(20, gsv_every=1), name="var.log")))
    _, _, sats = tr.scrub_points()
    # PRN 按 3,4,5,6,7 循环 → 最多 5 条不同的列表
    assert 1 < len(sats) <= 5


def test_scrub_points_do_not_carry_the_satellite_array_inline(tmp_path):
    """点里只放下标。整段数组内联的话，一天 1 Hz 的日志会到几十 MB。"""
    tr = NmeaTrack.load(str(_write(tmp_path, _synth_log(20), name="idx.log")))
    pts, _, _ = tr.scrub_points()
    assert "satellites" not in pts[0]
    assert "sat" in pts[0]


def test_scrub_points_without_gsv_use_none_index(tmp_path):
    """没收到 GSV 是**有效信息**（不是 0 号卫星），必须与"表里第 0 条"区分开。"""
    tr = NmeaTrack.load(str(_write(tmp_path, _synth_log(5, gsv=False), name="nogsv.log")))
    pts, _, sats = tr.scrub_points()
    assert sats == []
    assert all(p["sat"] is None for p in pts)


def test_scrub_points_on_empty_track_is_empty():
    pts, stride, sats = NmeaTrack().scrub_points()
    assert pts == [] and stride == 1 and sats == []


def test_service_state_includes_scrub_series_and_table(tmp_path):
    """前端只发这一个请求就要能拖动时间轴 —— 序列必须在 state 里。"""
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    st = svc.state()
    assert len(st["scrub"]) == 3
    assert st["scrub_stride"] == 1
    assert isinstance(st["scrub_sats"], list)
    # 序列里的 t 是日志时间轴上的值，配合 summary.origin_s 才能换算成视频时刻
    assert st["scrub"][0]["t"] == st["summary"]["t0"]
    assert st["summary"]["origin_s"] is not None or st["summary"]["t0"] is not None


def test_service_empty_state_has_empty_scrub():
    st = _svc().state()
    assert st["scrub"] == [] and st["scrub_sats"] == [] and st["scrub_stride"] == 1


def test_scrub_cache_is_invalidated_on_reimport(tmp_path):
    """缓存没作废的话，换一份日志后时间轴还在拖上一份的数据。"""
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG, name="a.log"))
    assert len(svc.state()["scrub"]) == 3
    svc.load(_write(tmp_path, _synth_log(7), name="b.log"))
    assert len(svc.state()["scrub"]) == 7


def test_scrub_cache_is_invalidated_on_clear(tmp_path):
    svc = _svc()
    svc.load(_write(tmp_path, MCU_LOG))
    svc.state()
    svc.clear()
    assert svc.state()["scrub"] == []
