"""轨迹接口：导入 / 状态 / 按时刻取帧 / 对齐 / 清除。

服务是进程内单例，所以每个用例先清空，避免互相影响。
"""
from __future__ import annotations

import io

import pytest

GGA_A = "$GNGGA,120000,2809.5000,N,11256.4000,E,1,09,0.8,50.0,M,10.0,M,,*69"
RMC_A = "$GNRMC,120000,A,2809.5000,N,11256.4000,E,0.2,090.0,170926,,,A*6E"
GGA_B = "$GNGGA,120001,2809.5100,N,11256.4100,E,1,10,0.7,50.5,M,10.0,M,,*6A"
RMC_B = "$GNRMC,120001,A,2809.5100,N,11256.4100,E,0.3,091.0,170926,,,A*6F"

LOG = "\n".join([
    "# 开机",
    "2026-09-17T12:00:00.000Z," + GGA_A,
    "2026-09-17T12:00:00.000Z," + RMC_A,
    "2026-09-17T12:00:01.000Z," + GGA_B,
    "2026-09-17T12:00:01.000Z," + RMC_B,
]) + "\n"

API = "/api/v1"


@pytest.fixture(autouse=True)
def _clean_track(client):
    client.delete(f"{API}/track")
    yield
    client.delete(f"{API}/track")


def _import(client, text=LOG, name="card.log"):
    return client.post(
        f"{API}/track/import",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/plain")},
    )


def test_import_returns_diagnose_and_summary(client):
    r = _import(client)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["ok"] is True
    assert j["name"] == "card.log"
    assert j["diagnose"]["stamp_kind"] == "iso"
    assert j["diagnose"]["count"] == 2
    assert j["summary"]["count"] == 2
    assert j["summary"]["duration_s"] == 1.0


def test_state_is_empty_before_import(client):
    j = client.get(f"{API}/track").json()
    assert j["loaded"] is False
    assert j["summary"] is None
    assert j["polyline"] == []


def test_state_after_import_has_polyline(client):
    _import(client)
    j = client.get(f"{API}/track").json()
    assert j["loaded"] is True
    assert j["name"] == "card.log"
    assert len(j["polyline"]) == 2
    assert j["polyline"][0]["lat"] == pytest.approx(28 + 9.5 / 60, abs=1e-6)


def test_at_returns_fix_with_satellite_detail(client):
    _import(client)
    j = client.get(f"{API}/track/at", params={"t": 0.0}).json()
    assert j["found"] is True
    assert j["fix"]["lat"] is not None
    assert j["fix"]["aligned"] is False, "没对齐就必须如实标出来"
    assert isinstance(j["fix"]["satellites"], list)


def test_at_without_track_is_not_found_not_an_error(client):
    j = client.get(f"{API}/track/at", params={"t": 3.0}).json()
    assert j["found"] is False
    assert j["fix"] is None
    assert j["message"]


def test_at_interpolates_between_epochs(client):
    _import(client)
    a = client.get(f"{API}/track/at", params={"t": 0.0}).json()["fix"]
    b = client.get(f"{API}/track/at", params={"t": 1.0}).json()["fix"]
    mid = client.get(f"{API}/track/at", params={"t": 0.5}).json()["fix"]
    assert mid["lat"] == pytest.approx((a["lat"] + b["lat"]) / 2, abs=1e-7)
    assert mid.get("interpolated") is True


def test_at_can_disable_interpolation(client):
    _import(client)
    mid = client.get(f"{API}/track/at", params={"t": 0.5, "interpolate": False}).json()["fix"]
    a = client.get(f"{API}/track/at", params={"t": 0.0, "interpolate": False}).json()["fix"]
    assert mid["lat"] == a["lat"]


def test_align_by_utc_marks_aligned(client):
    _import(client)
    j = client.post(f"{API}/track/align", json={"utc": "2026-09-17T12:00:00Z"}).json()
    assert j["ok"] is True
    assert j["used"] == "utc"
    assert j["summary"]["aligned"] is True
    fix = client.get(f"{API}/track/at", params={"t": 1.0}).json()["fix"]
    assert fix["aligned"] is True
    assert fix["utc"] == "2026-09-17T12:00:01Z"


def test_align_by_delta(client):
    _import(client)
    j = client.post(f"{API}/track/align", json={"delta_s": 1.0}).json()
    assert j["used"] == "delta_s"
    assert j["summary"]["shift_s"] == pytest.approx(1.0)


def test_align_rejects_bad_utc(client):
    _import(client)
    j = client.post(f"{API}/track/align", json={"utc": "昨天"}).json()
    assert j["ok"] is False
    assert "认不出" in j["error"]
    assert client.get(f"{API}/track").json()["summary"]["aligned"] is False


def test_align_without_track_is_refused(client):
    j = client.post(f"{API}/track/align", json={"delta_s": 1.0}).json()
    assert j["ok"] is False
    assert j["error"]


def test_align_empty_body_resets(client):
    _import(client)
    client.post(f"{API}/track/align", json={"utc": "2026-09-17T12:00:00Z"})
    j = client.post(f"{API}/track/align", json={}).json()
    assert j["used"] == "reset"
    assert j["summary"]["aligned"] is False


def test_delete_clears(client):
    _import(client)
    assert client.get(f"{API}/track").json()["loaded"] is True
    assert client.delete(f"{API}/track").json()["ok"] is True
    assert client.get(f"{API}/track").json()["loaded"] is False


def test_import_unparseable_log_reports_hints(client):
    r = _import(client, text="这不是日志\n随便写的\n", name="bad.txt")
    j = r.json()
    assert j["ok"] is False
    assert j["diagnose"]["count"] == 0
    assert j["diagnose"]["hints"]
    # 文件本身没问题，是内容认不出 —— 不该返回 4xx/5xx
    assert r.status_code == 200


def test_import_gbk_log_does_not_crash(client):
    """现场日志常是 GBK/ANSI。读取用 errors='ignore'，认不出就当注释跳过。"""
    data = "开机注释\n".encode("gbk") + LOG.encode("utf-8")
    r = client.post(
        f"{API}/track/import",
        files={"file": ("gbk.log", io.BytesIO(data), "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["diagnose"]["count"] == 2


def test_import_strips_path_from_filename(client):
    """文件名只用于显示，不能让它带路径进来。"""
    r = _import(client, name="../../evil.log")
    assert r.status_code == 200
    assert "/" not in r.json()["name"]
    assert "\\" not in r.json()["name"]


# ---------------------------------------------------------------- 本地插值序列

def test_state_carries_the_scrub_series_for_local_interpolation(client):
    """前端拖时间轴**一次请求都不发**，全靠这一份序列 —— 它必须随状态一起下发。

    这是被实测事故逼出来的契约：原先拖动时逐帧调 ``/track/at``，一秒 60 次，
    把浏览器的连接池拖死（请求全部挂到 120 s 客户端超时）。
    """
    _import(client)
    j = client.get(f"{API}/track").json()
    assert j["loaded"] is True
    assert len(j["scrub"]) == 2
    assert j["scrub_stride"] == 1
    assert isinstance(j["scrub_sats"], list)
    row = j["scrub"][0]
    # 序列里必须是**日志时间轴**上的 t，前端配合 summary.origin_s 换算视频时刻
    assert row["t"] == j["summary"]["t0"]
    for key in ("lat", "lon", "alt", "hdop", "fix_quality", "satellites_used", "usable", "utc"):
        assert key in row, f"本地插值要用 {key}"
    # 卫星明细只放下标，整段数组放表里
    assert "satellites" not in row and "sat" in row


def test_empty_state_has_empty_scrub(client):
    j = client.get(f"{API}/track").json()
    assert j["scrub"] == [] and j["scrub_sats"] == [] and j["scrub_stride"] == 1
