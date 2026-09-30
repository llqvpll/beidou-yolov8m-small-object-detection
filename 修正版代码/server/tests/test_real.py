"""真实推理链路的单元测试 + 集成测试。

- 纯单元测试（不依赖 ultralytics / 权重）：直接喂合成框，验证
  ``_to_detections``（含北斗 geo_map）、``draw_detections``（中文标签绘制）、
  ``_nms``（按类非极大抑制）这些“只有在出现真实框时才会触发”的分支。
- 集成测试（``importorskip("ultralytics")``）：若当前环境装了 ultralytics 且有可用
  权重，则真正加载模型并跑一遍 detect，验证「注册表→推理→解析→绘制」全链路；
  否则优雅跳过。这意味着在本机 / CI / 用户训练机上都能一键验证 real 模式。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # 修正版代码/
sys.path.insert(0, str(ROOT))

from server.config import Settings
from server.services.beidou import BeidouService
from server.services.inference import InferenceService, _nms
from server.services.drawing import draw_detections


class _FakeRegistry:
    """最小桩：仅满足 InferenceService 单测所需字段。"""

    mode = "demo"
    device = "cpu"
    error = None

    def is_real(self) -> bool:
        return False


@pytest.fixture
def svc():
    s = Settings(FORCE_DEMO=True)
    beidou = BeidouService(s)
    return InferenceService(_FakeRegistry(), beidou, s)


def test_to_detections_runs_geomap(svc):
    """合成框 → Detection 列表，且 geo_map 产出经纬度。"""
    dets = svc._to_detections([(10.0, 20.0, 30.0, 40.0, 2, 0.9)], w=640, h=480)
    assert len(dets) == 1
    d = dets[0]
    assert d.class_id == 2
    assert d.name_zh  # 中文类别名非空
    assert isinstance(d.lon, float) and isinstance(d.lat, float)
    assert d.bbox.x1 == 10.0 and d.bbox.x2 == 30.0


def test_draw_detections_emits_jpeg(svc):
    """合成框 → 返回合法 JPEG 字节（中文标签绘制分支）。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_detections(
        img, [{"class": 2, "conf": 0.91, "x1": 10, "y1": 20, "x2": 30, "y2": 40}], 640, 480
    )
    assert isinstance(out, bytes)
    assert out[:2] == b"\xff\xd8"  # JPEG SOI 标记


def test_nms_suppresses_overlapping_same_class():
    """同类高重叠框应被抑制到 1 个。"""
    boxes = [
        (0.0, 0.0, 10.0, 10.0, 1, 0.9),
        (0.0, 0.0, 10.0, 10.0, 1, 0.8),
    ]
    out = _nms(boxes, iou=0.5)
    assert len(out) == 1
    assert out[0][5] == pytest.approx(0.9, rel=1e-3)  # 保留高置信度（float32→float 容差）


def test_nms_keeps_different_classes():
    """不同类别即便重叠也各自保留。"""
    boxes = [
        (0.0, 0.0, 10.0, 10.0, 1, 0.9),
        (0.0, 0.0, 10.0, 10.0, 2, 0.8),
    ]
    out = _nms(boxes, iou=0.5)
    assert len(out) == 2


def test_real_pipeline():
    ultralytics = pytest.importorskip("ultralytics")
    from custom_modules.installer import is_installed, install

    if not is_installed():
        install(verbose=False)

    s = Settings(FORCE_DEMO=False, PRELOAD_MODEL=False, DEVICE="cpu", INFERENCE_CONCURRENCY=1)
    from server.services.model_registry import ModelRegistry

    reg = ModelRegistry(s)
    reg.ensure_ready()
    if not reg.is_real():
        pytest.skip(f"real 模式不可用（{reg.error}），跳过真实推理断言")

    beidou = BeidouService(s)
    inf = InferenceService(reg, beidou, s)

    # 合成黑图：模型大概率 0 框，但「加载→predict→解析→绘制→nms」全链路跑通
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    import cv2

    _, buf = cv2.imencode(".jpg", img)
    res = asyncio.run(
        inf.detect(image_bytes=buf.tobytes(), conf=0.25, draw=True, return_original=True)
    )
    assert res["mode"] == "real"
    assert isinstance(res["detections"], list)
    assert res["annotated"].startswith("data:image/jpeg;base64,")

    # 切片分支也能跑通
    res2 = asyncio.run(inf.detect(image_bytes=buf.tobytes(), conf=0.25, slice_=True))
    assert res2["mode"] == "real"
    assert res2["engine"] in ("slice", "standard", "sahi")

    # 关键回归：自动选中的必须是「正式 run」的权重，而不是冒烟/自检产物。
    # 曾经的 bug：候选表把 visdrone_v4 指向不存在的路径，于是 smoke_v4 的玩具模型
    # 被静默当作真实权重加载（接口全绿，但参数量/mAP 全错）。
    picked = Path(reg.loaded_variant or "")
    assert picked.exists(), f"权重路径不存在：{picked}"
    assert not any(tag in str(picked) for tag in ("smoke_", "gpu_check")), \
        f"误加载了冒烟/自检权重：{picked}"
    assert "visdrone_" in str(picked), f"未加载正式 VisDrone 权重：{picked}"
    # run_dir 只对「训练产物」有意义（<run>/weights/best.pt）。
    # 正式训练权重：必须能回溯到含 results.csv 的 run 目录；
    # 外部预训练权重（APP_WEIGHTS 直接指向某个 best.pt）：没有训练产物，
    # 此时必须**不认领** run_dir，否则 /metrics 会把无关目录当训练产物去读。
    if picked.parent.name == "weights" and (picked.parent.parent / "results.csv").exists():
        assert reg.run_dir is not None and (reg.run_dir / "results.csv").exists(), \
            f"run_dir 未指向训练产物目录：{reg.run_dir}"
    else:
        assert reg.run_dir is None, \
            f"非训练产物权重不应伪造 run_dir：{reg.run_dir}"


def test_metrics_matches_training_artifacts():
    """若本机有正式训练产物，校验指标解析与对比表**自洽**（不写死数值，换机可跑）。"""
    run_dir = Path("C:/yolo_runs/train/visdrone_v4")
    if not (run_dir / "results.csv").exists():
        pytest.skip("本机无 visdrone_v4 训练产物，跳过指标自洽校验")

    from server.services.metrics import MetricsService

    class _Reg:
        loaded_variant = str(run_dir / "weights" / "best.pt")
        model_info: dict = {}

        def __init__(self) -> None:
            self.run_dir = run_dir

    snap = MetricsService(_Reg(), Settings()).snapshot()

    assert snap["available"] is True, snap.get("reason")
    assert snap["run"] == "visdrone_v4"
    assert snap["epochs"] > 0
    assert snap["best"] is not None
    # best 必须等于逐 epoch mAP50 的最大值（与 ultralytics 选 best.pt 的口径一致）
    assert snap["best"]["mAP50"] == max(snap["curves"]["mAP50"])
    # 曲线长度一致，前端按索引画图才不会错位
    lens = {len(v) for v in snap["curves"].values()}
    assert lens == {snap["epochs"]}, lens

    cmp_ = snap["comparison"]
    assert cmp_ and cmp_["items"], "应解析出对比表"
    by_name = {("v4" if "v4" in i["name"] else "v1"): i for i in cmp_["items"]}
    assert "v1" in by_name and "v4" in by_name
    # 轻量化路线的核心结论：v4 参数量显著小于 v1（相对误差 10% 以上）
    assert by_name["v4"]["params"] < by_name["v1"]["params"] * 0.9
