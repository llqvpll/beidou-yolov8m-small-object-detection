"""后端 API 契约测试（演示模式，确定且快速）。

覆盖：健康检查、元信息、检测（含验证/降级/strict）、任务查询、错误信封。
真实推理（requires 权重）单独以一次性脚本验证，不纳入本套件以保持 CI 轻量。
"""
import base64
import io

from PIL import Image


def _png(color=(255, 0, 0), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _b64url(color=(0, 255, 0), size=(64, 64)) -> str:
    return "data:image/png;base64," + base64.b64encode(_png(color, size)).decode()


# ----------------------------- 健康检查 ----------------------------- #
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_readyz(client):
    assert client.get("/readyz").status_code == 200


def test_root_serves_frontend(client):
    """`/` 归前端（静态 SPA）；服务信息已移到 `/info`。

    若把 `/` 留给 JSON 路由，页面会被抢先命中、返回 application/json，
    所以这里同时锁住两件事：`/` 是 HTML、`/info` 才是 JSON。
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<!DOCTYPE html>" in r.text


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    assert "/docs" in r.json()["docs"]


def test_health_alias(client):
    """前端探测会先试带前缀的 /api/v1/health。"""
    for path in ("/health", "/healthz", "/readyz", "/api/v1/health"):
        assert client.get(path).status_code == 200, path


def test_metrics_contract(client):
    """演示模式下也要有稳定契约（available=False + 可读的 reason）。"""
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "available" in d and "runs" in d
    assert client.get("/api/v1/metrics/runs").status_code == 200
    # 配图必须做白名单校验：目录穿越与不存在文件都应是 404
    assert client.get("/api/v1/metrics/artifact/../../config.py").status_code == 404
    assert client.get("/api/v1/metrics/artifact/nope.png").status_code == 404


def test_openapi(client):
    assert client.get("/openapi.json").status_code == 200


# ----------------------------- 元信息 ----------------------------- #
def test_status_demo(client):
    r = client.get("/api/v1/status")
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "demo"
    assert j["engine_available"] is False
    assert len(j["classes"]) == 10
    # 定位来自模拟源（conftest 固定 APP_GNSS_SOURCE=mock），
    # 所以必须显式标成"非真实"，不能让它看起来像真模块读数
    b = j["beidou"]
    assert b["source"] == "mock"
    assert b["real"] is False
    assert b["usable"] is True
    assert b["fix_quality"] == 1
    assert b["satellites"] == b["satellites_used"]


def test_classes(client):
    r = client.get("/api/v1/classes")
    assert r.status_code == 200
    classes = r.json()
    assert len(classes) == 10
    assert classes[3]["name_en"] == "car" and classes[3]["name_zh"] == "小汽车"
    assert classes[0]["color"].startswith("#")


def test_beidou(client):
    r = client.get("/api/v1/beidou")
    assert r.status_code == 200
    assert "lat" in r.json() and "lon" in r.json()


def test_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    assert r.json()["mode"] == "demo"


# ----------------------------- 检测 ----------------------------- #
def test_detect_missing_input(client):
    r = client.post("/api/v1/detect")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BAD_REQUEST"


def test_detect_demo_multipart(client):
    data = _png()
    r = client.post("/api/v1/detect", files={"file": ("t.png", data, "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "demo"
    assert j["detections"] == []


def test_detect_demo_json(client):
    r = client.post("/api/v1/detect/json", json={"image": _b64url()})
    assert r.status_code == 200
    assert r.json()["mode"] == "demo"


def test_detect_strict_returns_503(client):
    r = client.post("/api/v1/detect/json", json={"image": _b64url(), "strict": True})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_READY"
    assert "request_id" in r.json()


def test_detect_payload_too_large(client):
    big = _png(size=(4, 4))
    # 伪装超大：直接塞很多字节
    huge = big + b"x" * (60 * 1024 * 1024)
    r = client.post("/api/v1/detect", files={"file": ("big.png", huge, "image/png")})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


# ----------------------------- 任务 / 视频 ----------------------------- #
def test_job_not_found(client):
    r = client.get("/api/v1/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_video_unsupported_in_demo(client):
    r = client.post(
        "/api/v1/video",
        files={"file": ("v.mp4", b"", "video/mp4")},
        data={"conf": 0.25},
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_READY"
