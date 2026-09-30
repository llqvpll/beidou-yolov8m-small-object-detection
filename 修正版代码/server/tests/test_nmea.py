"""NMEA 0183 解析器的单元测试。

测试用的语句都是**带正确校验和的字面量**（第一句 GGA 与公开文档里的经典样例
逐字节一致），这样测的是"能不能读真实模块吐出来的东西"，
而不是"能不能读懂我自己拼的字符串"。

另一条主线是**防噪声**：串口接错线、波特率不对、供电不足，都会产生随机字节。
要求是"宁可丢一句，也不能产出一个假坐标"——所以校验和不通过必须整句丢弃。
"""
import time

import pytest

from server.services.nmea import (
    FIX_QUALITY_ZH,
    NmeaAccumulator,
    checksum_of,
    parse_line,
    verify,
)

# ---------------------------------------------------------------- 样例数据

GGA = "$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*59"
RMC = "$GNRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W,A*19"
GSA = "$GNGSA,A,3,04,05,09,12,24,25,29,31,,,,,1.8,1.0,1.5*23"
GSV1 = "$GPGSV,3,1,11,03,03,111,00,04,15,270,00,06,01,010,00,13,06,292,00*74"
GSV2 = "$GPGSV,3,2,11,15,20,045,31,17,08,190,22,19,35,260,39,23,55,330,44*7A"
GSV_BD = "$BDGSV,1,1,04,01,45,120,42,02,30,200,38,03,60,300,45,04,15,050,25*64"
GGA_RTK = "$GNGGA,235959,2809.5000,N,11256.4000,E,4,21,0.6,52.3,M,10.1,M,,*6A"


def _acc(*lines) -> NmeaAccumulator:
    a = NmeaAccumulator()
    for ln in lines:
        a.feed(ln)
    return a


# ---------------------------------------------------------------- 校验和

def test_checksum_known_value():
    """经典样例的校验和是 59，用来锁住异或实现。"""
    assert checksum_of("GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,") == "59"


def test_verify_accepts_good_and_rejects_corrupt():
    good = parse_line(GGA)
    assert verify(good) is True
    # 把纬度 4807.038 改成 4907.038，但校验和不动 —— 必须被拒
    bad = parse_line(GGA.replace("4807.038", "4907.038"))
    assert verify(bad) is False


def test_verify_none_when_no_checksum():
    s = parse_line("$GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,")
    assert verify(s) is None


def test_corrupt_sentence_is_dropped_and_counted():
    """校验和不通过的句子必须整句丢弃，且不能污染已有定位。"""
    a = _acc(GGA)
    assert a.snapshot()["lat"] is not None
    a.feed(GGA.replace("4807.038", "4907.038"))  # 改字段但不动校验和
    assert a.checksum_errors == 1
    assert a.sentences == 1
    # 位置仍是第一句的真实值，没被噪声改掉
    assert round(a.snapshot()["lat"], 4) == 48.1173


# ---------------------------------------------------------------- 结构解析

@pytest.mark.parametrize("line", [
    "",
    "   ",
    "GNGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,",
    "$",
    "$GNGGA",
    "$PGRME,15.0,M,45.0,M,25.0,M*22",
    "$GNXXX,1,2,3*00",
    "$123456,1,2*00",
])
def test_parse_line_rejects_non_target(line):
    assert parse_line(line) is None


def test_parse_line_extracts_talker_and_kind():
    s = parse_line(GGA)
    assert (s.talker, s.kind) == ("GN", "GGA")
    assert parse_line(GSV_BD).talker == "BD"
    assert parse_line(RMC).kind == "RMC"


def test_unknown_counter_tracks_junk():
    a = _acc("", "$PGRME,1,2*00", "garbage", GGA)
    assert a.unknown == 3
    assert a.sentences == 1


def test_crlf_is_tolerated():
    """串口读出来常带 \r\n，不能因为结尾符号解析失败。"""
    a = _acc(GGA + "\r\n")
    assert a.sentences == 1
    assert round(a.snapshot()["lat"], 4) == 48.1173


# ---------------------------------------------------------------- GGA

def test_gga_position_and_quality():
    f = _acc(GGA).snapshot()
    assert round(f["lat"], 4) == 48.1173
    assert round(f["lon"], 4) == 11.5167
    assert f["alt"] == 545.4
    assert f["geoid_sep"] == 46.9
    assert f["hdop"] == 0.9
    assert f["satellites_used"] == 8
    assert f["fix_quality"] == 1
    assert f["fix_quality_zh"] == "单点定位"
    assert f["usable"] is True


def test_rtk_fixed_is_usable():
    f = _acc(GGA_RTK).snapshot()
    assert f["fix_quality"] == 4
    assert f["fix_quality_zh"] == "RTK 固定解"
    assert f["usable"] is True
    assert f["satellites_used"] == 21


def test_no_fix_is_not_usable():
    """fix_quality=0（无定位）不能算"可用定位"，坐标字段即使是数字也不行。"""
    line = GGA.replace(",E,1,08,", ",E,0,08,")
    line = line[:line.find("*")] + "*" + checksum_of(line[1:line.find("*")])
    f = _acc(line).snapshot()
    assert f["fix_quality"] == 0
    assert f["fix_quality_zh"] == "无定位"
    assert f["usable"] is False


def test_hemisphere_south_west_is_negative():
    line = GGA.replace(",N,", ",S,").replace(",E,", ",W,")
    line = line[:line.find("*")] + "*" + checksum_of(line[1:line.find("*")])
    f = _acc(line).snapshot()
    assert f["lat"] < 0 and f["lon"] < 0
    assert round(f["lat"], 4) == -48.1173


def test_empty_fields_give_none_not_zero():
    """空字段必须给 None。给 0 会被下游误读成"海拔真的是 0 米"。"""
    line = "$GNGGA,123519,4807.038,N,01131.000,E,1,,,,,,M,,M,,"
    line += "*" + checksum_of(line[1:])
    f = _acc(line).snapshot()
    assert f["alt"] is None
    assert f["hdop"] is None
    assert f["satellites_used"] is None
    # 定位质量有值就照常给，不能因为别的字段空着就一起丢掉
    assert f["fix_quality"] == 1


def test_short_gga_is_ignored_without_crash():
    a = _acc("$GNGGA,123519,4807.038*00")
    assert a._gga == {}
    assert a.snapshot()["lat"] is None


# ---------------------------------------------------------------- RMC

def test_rmc_speed_course_and_date():
    f = _acc(RMC).snapshot()
    assert f["rmc_valid"] is True
    assert f["speed_kn"] == 22.4
    assert f["speed_kmh"] == 41.48
    assert f["course"] == 84.4
    assert f["mode"] == "A"
    assert f["mode_zh"] == "自主定位"
    # 230394 + 123519 → 1994-03-23T12:35:19Z（两位年份 94 → 1994，不是 2094）
    assert f["utc"] == "1994-03-23T12:35:19Z"


def test_rmc_void_status_is_not_usable():
    line = RMC.replace(",A,4807", ",V,4807")
    line = line[:line.find("*")] + "*" + checksum_of(line[1:line.find("*")])
    f = _acc(line).snapshot()
    assert f["rmc_valid"] is False


def test_rmc_mode_r_implies_rtk_fixed_without_gga():
    """只有 RMC 时也要能推出定位质量，不能因为缺 GGA 就什么都不说。"""
    line = RMC.replace(",A*", ",R*")
    line = line[:line.find("*")] + "*" + checksum_of(line[1:line.find("*")])
    f = _acc(line).snapshot()
    assert f["mode"] == "R"
    assert f["fix_quality"] == 4
    assert f["fix_quality_zh"] == "RTK 固定解"
    assert f["usable"] is True


def test_gga_time_without_date_is_still_valid_iso():
    f = _acc(GGA).snapshot()
    assert f["utc"] == "12:35:19Z"


# ---------------------------------------------------------------- GSA

def test_gsa_dop_and_fix_type():
    f = _acc(GSA).snapshot()
    assert f["fix_type"] == 3
    assert f["fix_type_zh"] == "三维定位"
    assert f["pdop"] == 1.8
    assert f["hdop"] == 1.0
    assert f["vdop"] == 1.5


def test_gga_hdop_wins_over_gsa():
    """GGA 的 HDOP 与 GSA 不一致时以 GGA 为准（同一时刻它更贴近定位解）。"""
    f = _acc(GGA, GSA).snapshot()
    assert f["hdop"] == 0.9


# ---------------------------------------------------------------- GSV

def test_gsv_multi_message_merges():
    f = _acc(GSV1, GSV2).snapshot()
    assert len(f["satellites"]) == 8
    assert f["satellites_visible"] == 11
    first = f["satellites"][0]
    assert (first["prn"], first["talker"]) == ("03", "GP")
    assert (first["elevation"], first["azimuth"]) == (3.0, 111.0)
    # snr=00 表示未跟踪，给 None 而不是 0
    assert first["snr"] is None
    assert f["satellites"][4]["snr"] == 31.0


def test_gsv_new_round_replaces_old_satellites():
    """第 1 句到来时必须清空上一轮，否则卫星会越积越多、星空图变成一团。"""
    a = _acc(GSV1, GSV2)
    assert len(a.satellites()) == 8
    a.feed(GSV1)                       # 新一轮
    assert len(a.satellites()) == 4


def test_multi_constellation_satellites_are_kept_separately():
    f = _acc(GSV1, GSV2, GSV_BD).snapshot()
    talkers = {s["talker"] for s in f["satellites"]}
    assert talkers == {"GP", "BD"}
    assert len(f["satellites"]) == 12
    assert f["satellites_visible"] == 15      # 11 (GPS) + 4 (北斗)


def test_visible_falls_back_to_count_when_in_view_missing():
    line = "$GPGSV,1,1,,03,03,111,10,04,15,270,20"
    line += "*" + checksum_of(line[1:])
    f = _acc(line).snapshot()
    assert f["satellites_visible"] == 2


# ---------------------------------------------------------------- 数据龄期

def test_age_none_before_any_fix():
    assert NmeaAccumulator().age_s() is None
    assert NmeaAccumulator().snapshot()["age_s"] is None


def test_age_grows_with_time():
    a = _acc(GGA)
    now = time.monotonic()
    assert 0 <= a.age_s(now) < 1.0
    assert a.age_s(now + 7.5) == pytest.approx(7.5, abs=0.05)


def test_rmc_without_position_does_not_refresh_age():
    """没有位置的语句不能刷新"数据新鲜度"，否则断线了也显示数据很新。"""
    a = _acc(GGA)
    before = a._last_fix_monotonic
    a.feed("$GNRMC,123519,V,,,,,,,230394,,,N*00")
    assert a._last_fix_monotonic == before


# ---------------------------------------------------------------- 复位

def test_reset_clears_everything():
    a = _acc(GGA, RMC, GSA, GSV1, GSV2)
    assert a.snapshot()["lat"] is not None
    a.reset()
    f = a.snapshot()
    assert f["lat"] is None and f["lon"] is None
    assert f["satellites"] == []
    assert a.sentences == 0 and a.checksum_errors == 0
    assert a.age_s() is None


# ---------------------------------------------------------------- 契约

def test_fix_quality_table_covers_all_codes():
    """0~8 每个码都要有中文，界面上不能出现"未知"这种半成品。"""
    assert set(FIX_QUALITY_ZH) == set(range(9))


def test_snapshot_keys_are_stable():
    """快照的键集合是前后端契约，缺一个前端就会显示空白。"""
    need = {
        "lat", "lon", "alt", "geoid_sep", "fix_quality", "fix_quality_zh",
        "fix_type", "fix_type_zh", "usable", "satellites_used", "satellites_visible",
        "hdop", "vdop", "pdop", "speed_kn", "speed_kmh", "course", "utc",
        "mode", "mode_zh", "rmc_valid", "satellites", "sentences",
        "checksum_errors", "no_checksum", "unknown_lines", "age_s",
    }
    assert need <= set(_acc(GGA, RMC, GSA, GSV1, GSV2).snapshot())
