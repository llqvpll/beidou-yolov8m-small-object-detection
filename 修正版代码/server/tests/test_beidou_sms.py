"""北斗短报文编码测试。

测试重点在**约束下的行为**，而不是"随便编一条能不能解回来"：
长度上限、超长降级、CRC 检出、以及"空值不许被编成 0"。

长度口径来源：《新时代的中国北斗》白皮书（2022-11）
* 区域短报文 14000 bit（1000 个汉字）
* 全球短报文 560 bit（40 个汉字）
"""
from __future__ import annotations

import pytest

from server.services import beidou_sms as sms


def _alert(**kw) -> sms.Alert:
    base = dict(level="warn", lon=112.938812, lat=28.228234, density=2.63,
                count=137, utc="2026-09-25T11:22:33Z", site="常德万达广场东门")
    base.update(kw)
    return sms.Alert(**base)


class TestLengthConstraint:
    def test_region_channel_fits(self):
        e = sms.encode(_alert(), channel="region")
        assert e.ok
        assert e.limit_bytes == sms.REGION_BYTES == 1750
        assert not e.compact

    def test_global_channel_goes_compact(self):
        e = sms.encode(_alert(), channel="global")
        assert e.ok
        assert e.limit_bytes == sms.GLOBAL_BYTES == 70
        assert e.compact

    def test_global_drops_site(self):
        """全球档只有 40 个汉字，点位名是第一个该砍的。"""
        e = sms.encode(_alert(), channel="global")
        assert "常德万达广场东门" not in e.text
        assert sms.decode(e.text).site is None

    def test_global_reduces_coord_precision_to_match_fix_accuracy(self):
        """全球档经纬度只保留 4 位小数（≈11 m），与单点定位精度匹配。
        写更多位是虚假精度 —— 基准点本身就没那么准。"""
        e = sms.encode(_alert(), channel="global")
        assert "112.9388" in e.text
        assert "112.938812" not in e.text

    def test_global_keeps_essential_fields(self):
        d = sms.decode(sms.encode(_alert(), channel="global").text)
        assert d.level == "warn"
        assert d.count == 137
        assert d.lon == pytest.approx(112.9388)
        assert d.lat == pytest.approx(28.2282)

    def test_overlong_site_triggers_drop_then_fits(self):
        """区域档：点位名过长时先裁点位名，而不是报错。"""
        e = sms.encode(_alert(site="非常长的点位名称" * 200), channel="region")
        assert e.ok
        assert e.dropped == ["点位名"]

    @pytest.mark.parametrize("channel,limit", [("region", 1750), ("global", 70)])
    def test_never_exceeds_limit(self, channel, limit):
        e = sms.encode(_alert(), channel=channel)
        assert e.nbytes <= limit

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="未知短报文通道"):
            sms.encode(_alert(), channel="mars")

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="未知预警级别"):
            sms.encode(_alert(level="catastrophic"), channel="region")


class TestNullHandling:
    """本项目的底线：不知道就是不知道，不许编成 0。"""

    def test_missing_coords_encode_as_dash(self):
        e = sms.encode(sms.Alert(level="danger"), channel="global")
        assert "-,-" in e.text
        d = sms.decode(e.text)
        assert d.lon is None and d.lat is None

    def test_missing_count_is_not_zero(self):
        """count=None 编成 0 的话，接收端会读成"现场没人"，而真相是"没测到"。"""
        e = sms.encode(_alert(count=None), channel="global")
        assert sms.decode(e.text).count is None

    def test_real_zero_count_survives(self):
        """0 是合法值，不能被当成 None 吞掉。"""
        e = sms.encode(_alert(count=0), channel="global")
        assert sms.decode(e.text).count == 0

    def test_missing_density_is_not_zero(self):
        e = sms.encode(_alert(density=None), channel="global")
        assert sms.decode(e.text).density is None

    def test_missing_utc_becomes_dash(self):
        e = sms.encode(_alert(utc=None), channel="global")
        assert sms.decode(e.text).utc == "-"


class TestCrc:
    def test_crc_roundtrip(self):
        e = sms.encode(_alert(), channel="region")
        assert sms.decode(e.text).crc_ok is True

    def test_tampered_last_byte_detected(self):
        e = sms.encode(_alert(), channel="region")
        assert sms.decode(e.text[:-1] + "X").crc_ok is False

    def test_tampered_body_detected(self):
        e = sms.encode(_alert(), channel="region")
        assert sms.decode(e.text.replace("warn", "WXRN")).crc_ok is False

    def test_missing_crc_section_is_flagged(self):
        e = sms.encode(_alert(), channel="region")
        assert sms.decode(e.text.split("*")[0]).crc_ok is False

    def test_truncated_message_is_flagged_not_crashed(self):
        """短报文链路上丢字符是常态：要能标记可疑，而不是抛异常。"""
        e = sms.encode(_alert(), channel="region")
        d = sms.decode(e.text[:20])
        assert d.crc_ok is False

    def test_crc16_known_vector(self):
        # CRC-16/CCITT-FALSE 的标准测试向量
        assert sms.crc16(b"123456789") == 0x29B1

    def test_crc16_empty(self):
        assert sms.crc16(b"") == 0xFFFF


class TestRoundTrip:
    @pytest.mark.parametrize("channel", ["region", "global"])
    @pytest.mark.parametrize("level", ["normal", "watch", "warn", "danger"])
    def test_all_levels_roundtrip(self, channel, level):
        d = sms.decode(sms.encode(_alert(level=level), channel=channel).text)
        assert d.level == level
        assert d.level_zh == sms.LEVEL_ZH[level]
        assert d.crc_ok

    def test_protocol_version_present(self):
        d = sms.decode(sms.encode(_alert(), channel="region").text)
        assert d.proto == sms.PROTO

    def test_region_keeps_full_precision(self):
        d = sms.decode(sms.encode(_alert(), channel="region").text)
        assert d.lon == pytest.approx(112.938812, abs=1e-6)

    def test_pipe_in_site_is_sanitized(self):
        """点位名里的 | 会破坏字段分隔，必须替换掉，否则解出来字段全错位。"""
        e = sms.encode(_alert(site="东门|西门"), channel="region")
        d = sms.decode(e.text)
        assert d.crc_ok
        assert "|" not in (d.site or "")

    def test_encoded_reports_zh_char_estimate(self):
        e = sms.encode(_alert(), channel="global")
        assert e.zh_chars >= 1
