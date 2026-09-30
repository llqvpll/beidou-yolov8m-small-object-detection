"""人群密度监测与聚集预警的单元测试。

这里刻意把「不给标定就不出密度」当成正面行为来测——
宁可返回 None，也不能拍脑袋编一个密度出来。
"""
import numpy as np
import pytest

from server.services.crowd import CLS_PEDESTRIAN, CLS_PEOPLE, FrameSeries, analyze, level_of


def _box(x1, y1, x2, y2, cid, conf=0.9):
    return (x1, y1, x2, y2, cid, conf)


# ---------------------------------------------------------------- 分级

@pytest.mark.parametrize("density,key", [
    (0.0, "normal"),
    (0.99, "normal"),
    (1.0, "watch"),
    (1.99, "watch"),
    (2.0, "warn"),
    (3.99, "warn"),
    (4.0, "danger"),
    (9.5, "danger"),
])
def test_level_boundaries(density, key):
    lv = level_of(density)
    assert lv["key"] == key
    # 只有「警戒」及以上是预警事件
    assert lv["alert"] is (key in ("warn", "danger"))


def test_level_none_when_uncalibrated():
    """未标定（density=None）时返回 None，绝不给假结论。"""
    assert level_of(None) is None


# ---------------------------------------------------------------- 单帧统计

def test_count_only_people_classes():
    """行人 + 人群算人，车不算。"""
    boxes = [
        _box(0, 0, 10, 10, CLS_PEDESTRIAN),
        _box(0, 0, 10, 10, CLS_PEDESTRIAN),
        _box(0, 0, 10, 10, CLS_PEOPLE),
        _box(0, 0, 10, 10, 3),          # car
        _box(0, 0, 10, 10, 3),          # car
    ]
    r = analyze(boxes, 100, 100)
    assert r["count"] == 3
    assert r["count_pedestrian"] == 2
    assert r["count_people"] == 1


def test_people_weight_applied():
    """people 是「密集人群」框，按 weight 折算；原始计数仍分别上报。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEOPLE)] * 5
    r = analyze(boxes, 100, 100, people_weight=3.0)
    assert r["count_people"] == 5
    assert r["count"] == 15


def test_people_weight_invalid_falls_back_to_one():
    boxes = [_box(0, 0, 10, 10, CLS_PEOPLE)] * 4
    for bad in (0, -1, None):
        r = analyze(boxes, 100, 100, people_weight=bad)
        assert r["count"] == 4


def test_uncalibrated_gives_no_density():
    """没给面积 → density/level 都是 None，但人数和占比照常给。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEDESTRIAN)] * 10
    r = analyze(boxes, 100, 100)
    assert r["calibrated"] is False
    assert r["density"] is None
    assert r["level"] is None
    assert r["count"] == 10
    assert r["area_m2"] is None
    assert r["message"]
    assert 0 < r["occupancy_ratio"] < 1


def test_area_m2_calibration():
    """10 人 / 100 m² = 0.1 人/m² → 正常。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEDESTRIAN)] * 10
    r = analyze(boxes, 100, 100, area_m2=100.0)
    assert r["calibrated"] is True
    assert r["density"] == pytest.approx(0.1, abs=1e-3)
    assert r["level"]["key"] == "normal"
    assert r["area_m2"] == pytest.approx(100.0)
    assert r["message"] is None


def test_meters_per_px_calibration():
    """100×100 px、1 px = 0.1 m → 画面 10×10 = 100 m²。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEDESTRIAN)] * 10
    r = analyze(boxes, 100, 100, meters_per_px=0.1)
    assert r["area_m2"] == pytest.approx(100.0, abs=1e-2)
    assert r["density"] == pytest.approx(0.1, abs=1e-3)


def test_area_m2_wins_over_meters_per_px():
    """两个都给时以 area_m2 为准，避免口径含糊。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEDESTRIAN)] * 10
    r = analyze(boxes, 100, 100, area_m2=50.0, meters_per_px=0.1)
    assert r["area_m2"] == pytest.approx(50.0)
    assert r["density"] == pytest.approx(0.2, abs=1e-3)


def test_custom_thresholds():
    """阈值可配置：把 watch 调到 0.05，同样的 0.1 人/m² 就该升级。"""
    boxes = [_box(0, 0, 10, 10, CLS_PEDESTRIAN)] * 10
    r = analyze(boxes, 100, 100, area_m2=100.0, watch=0.05, warn=0.5, danger=1.0)
    assert r["density"] == pytest.approx(0.1, abs=1e-3)
    assert r["level"]["key"] == "watch"


def test_occupancy_ratio_is_scale_invariant():
    """占比只跟框面积有关，跟画面分辨率无关。"""
    boxes = [_box(0, 0, 50, 50, CLS_PEDESTRIAN)] * 4   # 4 × 2500 = 10000 px²
    a = analyze(boxes, 200, 200)      # 10000 / 40000 = 0.25
    b = analyze(boxes, 400, 400)      # 框不变，画面变大 → 占比变小
    assert a["occupancy_ratio"] == pytest.approx(0.25, abs=1e-3)
    assert b["occupancy_ratio"] == pytest.approx(10000 / 160000, abs=1e-3)


def test_empty_boxes():
    r = analyze([], 640, 640, area_m2=100.0)
    assert r["count"] == 0
    assert r["density"] == 0.0
    assert r["level"]["key"] == "normal"
    assert r["occupancy_ratio"] == 0.0


def test_accepts_numpy_boxes():
    """推理管线给的是 ndarray，必须能吃。"""
    arr = np.array([
        [0, 0, 10, 10, CLS_PEDESTRIAN, 0.9],
        [0, 0, 10, 10, CLS_PEOPLE, 0.8],
    ], dtype=float)
    r = analyze(arr, 100, 100, area_m2=10.0)
    assert r["count"] == 2
    assert r["density"] == pytest.approx(0.2, abs=1e-3)


# ---------------------------------------------------------------- 视频序列

def test_series_empty():
    s = FrameSeries(max_points=10)
    r = s.summary()
    assert r["frames"] == 0
    assert r["peak_count"] == 0
    assert r["peak_density"] is None
    assert r["calibrated"] is False
    assert r["series_count"] == []


def test_series_peak_and_alert_ratio():
    s = FrameSeries(max_points=300)
    # 10 帧：3 帧正常、2 帧关注、3 帧警戒、2 帧危险 → 预警 5/10
    plan = [
        (5, 0.5, "normal"), (5, 0.5, "normal"), (5, 0.5, "normal"),
        (10, 1.5, "watch"), (10, 1.5, "watch"),
        (20, 3.0, "warn"), (20, 3.0, "warn"), (20, 3.0, "warn"),
        (40, 6.0, "danger"), (40, 6.0, "danger"),
    ]
    for n, d, k in plan:
        s.add(n, d, level_of(d))
    r = s.summary()
    assert r["frames"] == 10
    assert r["peak_count"] == 40
    assert r["peak_density"] == pytest.approx(6.0, abs=1e-3)
    assert r["peak_level"]["key"] == "danger"
    assert r["peak_level"]["alert"] is True
    assert r["alert_frames"] == 5
    assert r["alert_ratio"] == pytest.approx(0.5, abs=1e-3)
    assert r["calibrated"] is True


def test_series_peak_level_is_highest_not_last():
    """峰值等级取历史最高，不是最后一帧——最后一帧人散了也不能把预警抹掉。"""
    s = FrameSeries()
    s.add(40, 6.0, level_of(6.0))
    s.add(2, 0.2, level_of(0.2))
    assert s.summary()["peak_level"]["key"] == "danger"


def test_series_uncalibrated_has_no_density_curve():
    s = FrameSeries()
    for n in (1, 2, 3):
        s.add(n, None, None)
    r = s.summary()
    assert r["frames"] == 3
    assert r["peak_count"] == 3
    assert r["calibrated"] is False
    assert r["series_density"] == []
    assert r["series_count"] == [1, 2, 3]


def test_series_thinning_keeps_bounds():
    """超长视频要抽稀，但首尾必须保留，否则曲线会"缺头少尾"。"""
    s = FrameSeries(max_points=10)
    for i in range(1000):
        s.add(i, float(i) / 100.0, level_of(float(i) / 100.0))
    r = s.summary()
    assert len(r["series_count"]) == 10
    assert len(r["series_density"]) == 10
    assert r["series_count"][0] == 0
    assert r["series_count"][-1] == 999
    assert r["peak_count"] == 999


def test_series_max_points_floor():
    """max_points 给得太小会被抬到 10，防止除零/退化。"""
    s = FrameSeries(max_points=1)
    for i in range(50):
        s.add(1, 1.0, level_of(1.0))
    assert len(s.summary()["series_count"]) == 10
