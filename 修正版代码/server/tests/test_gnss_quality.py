"""定位精度档位 / 时间基准 / 星座构成 的单元测试。

这三块是本项目"与北斗结合"的抓手，共同遵循同一条原则：
**判不出来就是 None + 说明原因，绝不回退到看起来正常的默认值。**
所以测试的重点不是"happy path 对不对"，而是"**判不出来的时候有没有说出来**"。
"""
from __future__ import annotations

import pytest

from server.services.gnss_constellation import build as build_composition
from server.services.gnss_quality import DGPS_AGE_LIMIT_S, classify, tier_level_m, tier_zh
from server.services.gnss_timebase import build as build_timebase
from server.services.gnss_timebase import parse_utc_epoch


# ============================================================ 精度档位

class TestAccuracyTier:
    def test_rtk_fixed_is_the_finest(self):
        key, msg = classify(usable=True, source="serial", fix_quality=4, mode=None)
        assert key == "fixed"
        assert "固定解" in msg

    def test_rtk_float(self):
        key, _ = classify(usable=True, source="serial", fix_quality=5, mode=None)
        assert key == "float"

    def test_differential_is_sbas(self):
        key, _ = classify(usable=True, source="serial", fix_quality=2, mode=None)
        assert key == "sbas"

    def test_single_point(self):
        key, msg = classify(usable=True, source="serial", fix_quality=1, mode=None)
        assert key == "single"
        # 说明里要点出这是**系统能力**，不能说成本机实测
        assert "系统能力" in msg

    def test_mode_r_beats_missing_fix_quality(self):
        """只有 RMC 时靠 mode 也能判：R/F 只有载波相位解才有。"""
        key, _ = classify(usable=True, source="serial", fix_quality=None, mode="R")
        assert key == "fixed"

    def test_unknown_fix_quality_is_not_silently_downgraded(self):
        """fix_quality=9 不在已知档位 → 必须返回 None，不能当作单点定位。"""
        key, msg = classify(usable=True, source="serial", fix_quality=9, mode=None)
        assert key is None
        assert "无法判定" in msg

    def test_no_fix_gives_none_with_reason(self):
        key, msg = classify(usable=False, source="serial", fix_quality=0, mode=None)
        assert key is None
        assert msg                        # 必须有原因，不能空着

    def test_mock_source_never_claims_a_tier(self):
        """模拟源即使字段看起来完美，也不许给档位。"""
        key, msg = classify(usable=True, source="mock", fix_quality=4, mode="R")
        assert key is None
        assert "模拟" in msg

    def test_stale_dgps_age_downgrades_to_single(self):
        """模块仍报 RTK 固定解、但改正数已过期 —— 最危险的一类"看起来正常"。"""
        key, msg = classify(usable=True, source="serial", fix_quality=4, mode=None,
                            dgps_age=DGPS_AGE_LIMIT_S + 1)
        assert key == "single"
        assert "过期" in msg

    def test_fresh_dgps_age_keeps_tier(self):
        key, msg = classify(usable=True, source="serial", fix_quality=4, mode=None, dgps_age=3)
        assert key == "fixed"
        assert "3" in msg

    def test_age_boundary_is_inclusive(self):
        key, _ = classify(usable=True, source="serial", fix_quality=4, mode=None,
                          dgps_age=DGPS_AGE_LIMIT_S)
        assert key == "fixed"

    def test_conflicting_quality_and_mode_takes_the_coarser(self):
        """宁可低估精度，不可高估。fix_quality 说固定解、mode 说单点 → 取单点。"""
        key, _ = classify(usable=True, source="serial", fix_quality=4, mode="A")
        assert key == "single"

    @pytest.mark.parametrize("key", ["single", "sbas", "ppp", "float", "fixed"])
    def test_every_tier_has_name_and_level(self, key):
        assert tier_zh(key)
        assert tier_level_m(key) is not None

    def test_unknown_key_maps_to_none(self):
        assert tier_zh(None) is None
        assert tier_level_m(None) is None
        assert tier_zh("nonsense") is None


# ============================================================ 时间基准

class TestTimeBase:
    def test_parse_full_timestamp(self):
        assert parse_utc_epoch("2026-09-25T11:22:33Z") == 1790335353.0

    def test_parse_with_fraction(self):
        assert parse_utc_epoch("2026-09-25T11:22:33.500Z") == 1790335353.5

    def test_time_only_returns_none(self):
        """只有时分秒时**必须**返回 None：猜个日期会让偏差错到"差一天"。"""
        assert parse_utc_epoch("11:22:33Z") is None
        assert parse_utc_epoch("11:22:33") is None

    def test_empty_and_garbage(self):
        assert parse_utc_epoch(None) is None
        assert parse_utc_epoch("") is None
        assert parse_utc_epoch("not-a-time") is None
        assert parse_utc_epoch("2026-13-45T99:99:99Z") is None

    def test_delta_sign_means_local_is_slow(self):
        """delta = 北斗 − 本机。为正 = 本机慢了。符号约定必须锁死。"""
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=True, source="serial",
                            local_epoch=1790335350.0)
        assert tb.delta_s == 3.0
        assert tb.has_date is True

    def test_delta_negative_means_local_is_fast(self):
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=True, source="serial",
                            local_epoch=1790335358.0)
        assert tb.delta_s == -5.0

    def test_no_fix_means_no_timebase(self):
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=False, source="serial")
        assert tb.utc_epoch is None and tb.delta_s is None
        assert "有效定位解" in tb.message

    def test_mock_source_is_not_a_time_reference(self):
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=True, source="mock")
        assert tb.delta_s is None
        assert "模拟" in tb.message

    def test_time_only_still_displayed_but_no_delta(self):
        tb = build_timebase(utc="11:22:33Z", usable=True, source="serial")
        assert tb.utc == "11:22:33Z"      # 能显示
        assert tb.delta_s is None          # 但不参与计算
        assert tb.has_date is False
        assert "没有日期" in tb.message

    def test_source_label_is_human_readable(self):
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=True,
                            source="file", local_epoch=1790335353.0)
        assert "北斗" in tb.source_zh

    def test_replay_source_does_not_report_a_fake_clock_delta(self):
        """回放日志的 UTC 是"当时"，与本机相差的是**回放跨度**，不是时钟偏差。

        这是最容易做错的一处：拿一个几小时的跨度当"本机时钟偏差"报出去，
        用户会去校准一个本来正常的时钟，而真正的偏差被这个假数字盖住了。
        """
        tb = build_timebase(utc="2026-09-25T01:12:30Z", usable=True, source="file",
                            local_epoch=1790311984.0)
        assert tb.utc_epoch is not None      # 时刻本身可信、可显示
        assert tb.utc == "2026-09-25T01:12:30Z"
        assert tb.delta_s is None            # 但不计偏差
        assert tb.local_epoch is None
        assert "回放时间跨度" in tb.message

    def test_live_serial_source_does_report_delta(self):
        """实时串口才是真正需要核对时钟偏差的场景。"""
        tb = build_timebase(utc="2026-09-25T11:22:33Z", usable=True, source="serial",
                            local_epoch=1790335353.2)
        assert tb.delta_s == -0.2
        assert tb.message is None


# ============================================================ 星座构成

class TestConstellation:
    def test_counts_by_constellation(self):
        out = build_composition(
            [{"talker": "BD"}, {"talker": "BD"}, {"talker": "GP"}, {"talker": "GL"}], 12)
        assert out["available"] is True
        assert out["visible_total"] == 4
        assert out["beidou_visible"] == 2
        assert out["beidou_share"] == 0.5

    def test_gb_and_gq_count_as_beidou(self):
        out = build_composition([{"talker": "GB"}, {"talker": "GQ"}], 2)
        assert out["beidou_visible"] == 2

    def test_used_by_constellation_is_never_fabricated(self):
        """合并语句里拆不出"参与解算"的星座归属 → 必须为 None，不按比例摊派。"""
        out = build_composition([{"talker": "BD"}, {"talker": "GP"}], 9)
        assert out["used_total"] == 9
        assert out["used_by_constellation"] is None

    def test_all_gn_cannot_split(self):
        out = build_composition([{"talker": "GN"}, {"talker": "GN"}], 10)
        assert out["available"] is True
        assert out["beidou_share"] is None      # 拆不出就是拆不出
        assert "合并" in out["message"]

    def test_no_gsv_is_reported_honestly(self):
        out = build_composition([], None, sentences=0)
        assert out["available"] is False
        assert out["beidou_visible"] is None
        assert "未收到 GSV" in out["message"]

    def test_sentences_but_no_gsv_gives_distinct_message(self):
        out = build_composition([], None, sentences=50)
        assert out["available"] is False
        assert "没有 GSV" in out["message"]

    def test_zero_beidou_is_flagged_not_hidden(self):
        out = build_composition([{"talker": "GP"}, {"talker": "GL"}], 8)
        assert out["beidou_visible"] == 0
        assert "没有北斗" in out["message"]

    def test_ordering_is_stable(self):
        """同一份数据每次输出顺序必须一致，便于比对与留证。"""
        a = build_composition([{"talker": "GP"}, {"talker": "BD"}], 5)
        b = build_composition([{"talker": "BD"}, {"talker": "GP"}], 5)
        assert list(a["visible_by_constellation"]) == list(b["visible_by_constellation"])

    def test_entries_without_talker_are_skipped(self):
        out = build_composition([{"talker": "BD"}, {"talker": None}, {}], 3)
        assert out["visible_total"] == 1
