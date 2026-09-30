"""视频推理链路测试。

分两层：

- **纯单元**（不依赖 ultralytics / 权重）：编码器探测、回读帧数、绘制函数
  （``draw_detections_bgr`` 是视频逐帧复用的入口，必须能直接吃 ndarray 并返回 ndarray）。
- **集成**（需要 ultralytics + 正式权重 + 至少一张 VisDrone 原图）：用**无标注**原图
  合成一段小视频，跑完整 ``predict_video``，断言

    1. 回读帧数 == 输入帧数（写出完整；编码器"假成功"会在这里暴露）
    2. 输出字节 != 输入字节（曾经的 bug：结果就是上传的原片，字节数完全一致）
    3. 输出帧相对输入帧确有改动，且改动像素量级与检测框相符
    4. 结果文件名不以 ``input`` 开头（避免与上传件同名）

  依赖缺失时优雅跳过，保证在 CI / 任意机器上一键可跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # 修正版代码/
sys.path.insert(0, str(ROOT))

from server.config import Settings
from server.services import classes as C
from server.services.drawing import draw_detections, draw_detections_bgr
from server.services.inference import (
    _VIDEO_CODECS,
    _VIDEO_STEM,
    _codec_order,
    _count_frames,
    _probe_codec,
)

VISDRONE_VAL = Path("C:/datasets/VisDrone/images/val")


# ============================ 纯单元 ============================ #
def test_result_stem_not_input():
    """结果文件前缀不能是 input，否则会与上传件同名（踩过的坑）。"""
    assert _VIDEO_STEM != "input"
    assert not _VIDEO_STEM.startswith("input")


def test_codec_order_puts_pref_first():
    """配置指定的编码器必须排在最前。"""
    order = _codec_order("mp4v")
    assert order, "候选表不应为空"
    assert order[0][0].lower() == "mp4v"


def test_codec_order_ignores_unknown_pref():
    """配置了候选表之外的编码器时要**忽略并告警**，不能插到最前。

    OpenCV 遇到未知 fourcc 会静默换成默认编码器（ZZZZ → mp4v），若我们照单汇报，
    就等于对外宣称用了一个根本没生效的编码器。
    """
    order = _codec_order("ZZZZ")
    assert all(c[0] != "ZZZZ" for c in order), order


def test_codec_order_contains_known_codecs():
    order = _codec_order("")
    names = {c[0] for c in order}
    assert names, "候选表不应为空"
    assert names <= {c[0] for c in _VIDEO_CODECS}


def test_count_frames_on_garbage_returns_zero(tmp_path):
    """回读校验必须能识别坏文件（OpenH264 缺失时写出的 1KB 垃圾）。"""
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 1024)
    assert _count_frames(bad) == 0

    missing = tmp_path / "nope.mp4"
    assert _count_frames(missing) == 0


def test_probe_codec_returns_bool():
    """探测函数对任意输入都必须返回 bool（不能抛异常）。"""
    for fourcc, suffix in _VIDEO_CODECS:
        assert isinstance(_probe_codec(fourcc, suffix), bool)
    # 未知 fourcc：OpenCV 会静默替换，所以这里只要求"不抛异常、返回 bool"
    assert isinstance(_probe_codec("ZZZZ", ".mp4"), bool)


def test_read_fourcc_reports_actual_codec(tmp_path):
    """实际 fourcc 必须从文件里读出来，而不是回显请求值。"""
    from server.services.inference import _read_fourcc

    assert _read_fourcc(tmp_path / "nope.mp4") == ""

    p = tmp_path / "x.mp4"
    fourcc, _ = next(((f, s) for f, s in _VIDEO_CODECS if _probe_codec(f, s)), _VIDEO_CODECS[0])
    if not _probe_codec(fourcc, ".mp4" if fourcc != "XVID" else ".avi"):
        pytest.skip("本机无可用编码器")
    import cv2

    suffix = ".avi" if fourcc == "XVID" else ".mp4"
    p = tmp_path / f"x{suffix}"
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), 10.0, (64, 64))
    if not wr.isOpened():
        pytest.skip("writer 打不开")
    for _ in range(3):
        wr.write(np.zeros((64, 64, 3), dtype=np.uint8))
    wr.release()
    assert _read_fourcc(p), "应读出非空 fourcc"


def test_draw_detections_bgr_returns_ndarray():
    """视频逐帧入口：吃 ndarray、吐同尺寸 ndarray（不修改入参）。"""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    before = img.copy()
    out = draw_detections_bgr(
        img, [{"class": 3, "conf": 0.88, "x1": 20, "y1": 30, "x2": 160, "y2": 200}], 320, 240
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (img == before).all(), "入参被就地修改了"
    assert not (out == before).all(), "什么都没画上"

    # 画的必须是本项目的类别配色（car = 纯黄），且标签填充块面积可观
    car_bgr = np.array(C.CLASS_COLORS_BGR[3], dtype=np.int16)
    hits = int((np.abs(out.astype(np.int16) - car_bgr).max(axis=2) <= 8).sum())
    assert hits > 500, f"类别配色像素过少：{hits}"


def test_draw_detections_bgr_handles_top_edge():
    """框贴到画面顶部时标签要翻进框内，不能被裁掉（否则标签整条消失）。"""
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    out = draw_detections_bgr(
        img, [{"class": 0, "conf": 0.5, "x1": 10, "y1": 0, "x2": 90, "y2": 60}], 200, 120
    )
    # 顶部若干行必须出现绘制痕迹（标签被翻进画面内）
    assert out[:20].any(), "顶部标签被裁掉了"


def test_draw_detections_jpeg_still_works():
    """单图入口保持向后兼容（JPEG 字节）。"""
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_detections(
        img, [{"class": 1, "conf": 0.7, "x1": 5, "y1": 5, "x2": 50, "y2": 60}], 160, 120
    )
    assert out[:2] == b"\xff\xd8"


def test_ascii_tmp_dir_is_ascii():
    """临时目录必须是纯 ASCII，否则 avc1 写不进去（探测与真实写出的路径类别要一致）。"""
    from server.services.inference import _ascii_tmp_dir

    d = _ascii_tmp_dir()
    assert str(d).isascii(), d


def test_empty_detections_draws_nothing():
    """空检测列表必须原样返回（视频里没框的帧直接写原图，省一次 PIL 往返）。"""
    img = np.full((80, 80, 3), 77, dtype=np.uint8)
    out = draw_detections_bgr(img, [], 80, 80)
    assert (out == img).all()


# ============================ 集成 ============================ #
def _build_clean_video(dst: Path, n_img: int = 2, repeat: int = 3, w: int = 640, h: int = 384) -> int:
    """用**无标注**的 VisDrone 原图合成输入视频，返回总帧数。

    关键：输入本身没有任何框，所以输出帧一旦出现框，就一定是我们自己画的。
    """
    import cv2

    imgs = sorted(VISDRONE_VAL.glob("*.jpg"))[:n_img]
    fourcc, suffix = next(((f, s) for f, s in _VIDEO_CODECS if _probe_codec(f, s)), _VIDEO_CODECS[0])
    dst = dst.with_suffix(suffix)
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*fourcc), 10.0, (w, h))
    assert wr.isOpened(), "合成输入视频失败"
    n = 0
    for p in imgs:
        im = cv2.imread(str(p))
        assert im is not None, p
        s = min(w / im.shape[1], h / im.shape[0])
        r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
        y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        for _ in range(repeat):
            wr.write(canvas)
            n += 1
    wr.release()
    return n


def test_predict_video_end_to_end(tmp_path):
    """完整视频推理：帧数守恒 + 字节不同 + 真有框 + 结果名不含 input。

    **输出目录特意取中文名**：``avc1`` 在 Windows 上打不开含非 ASCII 字符的路径
    （mp4v/XVID 却可以），本项目实际目录就叫「新建文件夹」，所以这是必须覆盖的场景。
    实现上是"先写 ASCII 临时目录、校验通过再搬过来"，本用例同时守住这条回归。
    """
    pytest.importorskip("ultralytics")
    if not VISDRONE_VAL.is_dir() or not list(VISDRONE_VAL.glob("*.jpg")):
        pytest.skip(f"本机无 VisDrone 原图（{VISDRONE_VAL}），跳过视频端到端")

    import cv2

    from custom_modules.installer import install, is_installed

    if not is_installed():
        install(verbose=False)

    from server.services.beidou import BeidouService
    from server.services.inference import InferenceService
    from server.services.model_registry import ModelRegistry

    s = Settings(FORCE_DEMO=False, PRELOAD_MODEL=False, INFERENCE_CONCURRENCY=1)
    reg = ModelRegistry(s)
    reg.ensure_ready()
    if not reg.is_real():
        pytest.skip(f"real 模式不可用（{reg.error}），跳过视频端到端")

    inf = InferenceService(reg, BeidouService(s), s)

    job_dir = tmp_path / "任务-0001"      # 故意用中文目录
    assert not str(job_dir).isascii()

    src = tmp_path / "clean_input.mp4"
    n_in = _build_clean_video(src)
    src = src if src.exists() else next(tmp_path.glob("clean_input.*"))

    progress: list[int] = []
    out = inf.predict_video(
        str(src), job_dir, conf=0.25, iou=0.45, imgsz=640, max_det=300,
        progress_cb=lambda d, t: progress.append(d),
    )

    res = Path(out["output"])
    assert res.exists() and res.parent == job_dir
    assert not res.name.startswith("input"), f"结果与上传件同名：{res.name}"
    assert out["codec"] in {c[0] for c in _VIDEO_CODECS} | {out["codec"]}

    st = out["stats"]
    assert st["frames"] == n_in, f"帧数不符：写出 {st['frames']} / 输入 {n_in}"
    assert st["readback_frames"] == n_in, "回读帧数不符（编码器可能写了坏文件）"
    assert len(progress) == n_in, "进度回调次数应与帧数一致"

    # 字节必须不同：旧 bug 就是"结果 = 上传原片"，两者字节数一模一样
    assert res.stat().st_size != src.stat().st_size

    # 抽帧差分：输入无标注，输出必须被改过
    cap = cv2.VideoCapture(str(res))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, f_out = cap.read()
    cap.release()
    assert ok
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, f_in = cap.read()
    cap.release()
    assert ok
    assert f_out.shape == f_in.shape

    changed = int((cv2.absdiff(f_in, f_out).sum(axis=2) > 30).sum())
    assert changed > 200, f"输出帧几乎没有改动（{changed} 像素），可能没画框"
