"""推理服务：图像检测、内置切片推理（小目标友好）、视频批量检测。

设计要点（对标 roboflow/inference 的 infer→preprocess→predict→postprocess）：
  - 同步的 torch 推理放在线程池里跑（``asyncio.to_thread``），不阻塞事件循环；
  - 用信号量 ``INFERENCE_CONCURRENCY`` 串行化 GPU 推理，避免显存被打满；
  - 双模式：未加载模型时返回 ``mode=demo`` 的优雅降级结果，接口契约不变；
  - 切片推理自带实现（零依赖），并可选回退到 sahi。
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from pathlib import Path

from ..config import Settings
from ..core.errors import BadRequestError, ModelNotReadyError
from ..schemas.detection import BeidouInfo, BoundingBox, Detection
from . import classes as C
from .beidou import BeidouService
from .crowd import FrameSeries, analyze as analyze_crowd
from .drawing import draw_detections, draw_detections_bgr
from .image_io import (
    decode_image,
    encode_jpeg,
    from_data_url,
    to_data_url,
)

log = logging.getLogger("server.inference")


class InferenceService:
    def __init__(self, registry, beidou: BeidouService, settings: Settings) -> None:
        self.registry = registry
        self.beidou = beidou
        self.settings = settings
        self._sem = asyncio.Semaphore(max(1, settings.INFERENCE_CONCURRENCY))

    # ============================ 图像检测 ============================ #
    async def detect(
        self,
        *,
        image_bytes: bytes | None = None,
        image_data_url: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        slice_: bool = False,
        draw: bool = False,
        strict: bool = False,
        return_original: bool = False,
        area_m2: float | None = None,
        meters_per_px: float | None = None,
    ) -> dict:
        """图像检测。

        :param area_m2: 画面实际覆盖面积（m²），用于把人数换算成密度
        :param meters_per_px: 比例尺（米/像素），与 area_m2 二选一
        """
        # demo 模式：优雅降级
        if not self.registry.is_real():
            if strict:
                raise ModelNotReadyError(
                    self.registry.error or "推理引擎未就绪（演示模式）",
                    details={"mode": "demo"},
                )
            orig = None
            if return_original and image_bytes:
                orig = to_data_url(encode_jpeg(decode_image(image_bytes)))
            return {
                "mode": "demo", "width": 0, "height": 0, "detections": [],
                "original": orig, "annotated": None, "device": None, "engine": None,
                # 显式给 None 而不是让字段缺失：契约一致，前端不必区分「没这字段」和「没标」
                "crowd": None,
                "beidou": self.beidou.info(),
                "message": "当前为演示模式（未加载真实模型）。前端可展示演示数据；"
                           "在装有 ultralytics + 权重的机器上启动即自动切真实推理。",
            }

        if image_bytes is None and image_data_url:
            image_bytes = from_data_url(image_data_url)
        if image_bytes is None:
            raise BadRequestError("未收到图像（file 或 image 字段）")

        async with self._sem:
            result = await asyncio.to_thread(
                self._detect_sync, image_bytes, conf, iou, imgsz, max_det, slice_, draw,
                return_original, area_m2, meters_per_px,
            )
        return result

    def _detect_sync(self, image_bytes, conf, iou, imgsz, max_det, slice_, draw, return_original,
                     area_m2: float | None = None, meters_per_px: float | None = None) -> dict:
        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        h, w = img.shape[:2]
        model = self.registry.get_model()

        engine = "standard"
        if slice_ and self.settings.SLICE_ENABLED:
            # 优先按配置选择 sahi；任何失败都回退到自带切片，再回退标准推理
            if self.settings.SLICE_ENGINE == "sahi":
                try:
                    boxes = self._predict_sahi(model, img, conf, iou, imgsz)
                    engine = "sahi"
                except Exception as e:  # noqa: BLE001
                    log.warning("sahi 切片失败，回退内置切片: %s", e)
                    boxes = None
            else:
                boxes = None
            if engine == "standard" or boxes is None:
                try:
                    boxes = self._predict_slice(model, img, conf, iou, imgsz)
                    engine = "slice"
                except Exception as e:  # noqa: BLE001
                    log.warning("内置切片失败，回退标准推理: %s", e)
                    boxes = self._predict_standard(model, img, conf, iou, imgsz, max_det)
                    engine = "standard"
        else:
            boxes = self._predict_standard(model, img, conf, iou, imgsz, max_det)

        dets = self._to_detections(boxes, w, h)
        annotated = None
        if draw:
            dicts = [
                {"class": d.class_id, "conf": d.conf,
                 "x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2}
                for d in dets
            ]
            annotated = to_data_url(draw_detections(img, dicts, w, h))
        original = to_data_url(encode_jpeg(img)) if return_original else None
        elapsed = (time.perf_counter() - t0) * 1000

        # 人群密度分析（无论是否标定都给人数；密度需标定才有）
        crowd = None
        if self.settings.CROWD_ENABLED:
            crowd = analyze_crowd(
                boxes, w, h,
                area_m2=area_m2, meters_per_px=meters_per_px,
                people_weight=self.settings.CROWD_PEOPLE_WEIGHT,
                watch=self.settings.CROWD_WATCH,
                warn=self.settings.CROWD_WARN,
                danger=self.settings.CROWD_DANGER,
            )

        return {
            "mode": "real", "width": w, "height": h, "detections": dets,
            "crowd": crowd,
            "original": original, "annotated": annotated,
            "device": self.registry.device, "engine": engine,
            "beidou": self.beidou.info(), "message": None, "elapsed_ms": round(elapsed, 1),
        }

    # ----------------------------- 标准推理 ----------------------------- #
    def _predict_standard(self, model, img, conf, iou, imgsz, max_det):
        results = model.predict(
            img, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
            device=self.registry.device, verbose=False,
        )
        return self._parse(results)

    # ----------------------------- 内置切片推理 ----------------------------- #
    def _predict_slice(self, model, img, conf, iou, imgsz):
        """把大图切成重叠小块分别检测再合并（NMS）。零依赖，专为小目标设计。

        切片坐标天然对应原图，可进一步与北斗定位联动（见参考清单 A 方向）。
        """
        import numpy as np

        h, w = img.shape[:2]
        sh = sw = self.settings.SLICE_SIZE
        ov = self.settings.SLICE_OVERLAP
        step_x = max(1, int(sw * (1 - ov)))
        step_y = max(1, int(sh * (1 - ov)))
        device = self.registry.device

        ys = list(range(0, max(1, h - sh + 1), step_y)) or [0]
        xs = list(range(0, max(1, w - sw + 1), step_x)) or [0]

        collected = []
        for y in ys:
            for x in xs:
                y2 = min(y + sh, h)
                x2 = min(x + sw, w)
                crop = img[y:y2, x:x2]
                if crop.shape[0] < sh or crop.shape[1] < sw:
                    tile = np.zeros((sh, sw, 3), dtype=img.dtype)
                    tile[: crop.shape[0], : crop.shape[1]] = crop
                else:
                    tile = crop
                res = model.predict(tile, conf=conf, iou=iou, imgsz=imgsz,
                                    device=device, verbose=False)
                for r in res:
                    for b in r.boxes:
                        bx = [float(v) for v in b.xyxy[0].tolist()]
                        collected.append((bx[0] + x, bx[1] + y, bx[2] + x, bx[3] + y,
                                          int(b.cls[0]), float(b.conf[0])))
        return _nms(collected, iou=iou)

    # ----------------------------- sahi（可选） ----------------------------- #
    def _predict_sahi(self, model, img, conf, iou, imgsz):  # pragma: no cover - 需 sahi
        from sahi import get_sliced_prediction
        from sahi.model import YOLOv8DetectionModel

        device = "cuda:0" if self.registry.device != "cpu" else "cpu"
        detection_model = YOLOv8DetectionModel(
            model_path=self.registry.loaded_variant,
            confidence_threshold=conf,
            device=device,
        )
        sh = sw = self.settings.SLICE_SIZE
        ov = self.settings.SLICE_OVERLAP
        result = get_sliced_prediction(
            img,
            detection_model,
            slice_height=sh, slice_width=sw,
            overlap_height_ratio=ov, overlap_width_ratio=ov,
            perform_standard_pred=False,
        )
        out = []
        for obj in result.object_prediction_list:
            if obj.score.value < conf:
                continue
            b = obj.bbox
            out.append((b.minx, b.miny, b.maxx, b.maxy, int(obj.category.id), float(obj.score.value)))
        return _nms(out, iou=iou)

    # ----------------------------- 解析 / 组装 ----------------------------- #
    @staticmethod
    def _parse(results):
        out = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                out.append((x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])))
        return out

    def _to_detections(self, boxes, w, h) -> list[Detection]:
        en = C.classes_en()
        zh = C.classes_zh()
        dets: list[Detection] = []
        # 定位帧只取一次：一张图可能有几百个目标，逐个去取会重复加锁并重建卫星列表。
        # 同一张图上的所有目标本来就该共享同一个定位解。
        fix = self.beidou.info()
        for (x1, y1, x2, y2, cls, cf) in boxes:
            cid = int(cls)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # 未标定 GSD 或没有有效定位时给 (None, None)，前端会显示"—"并说明原因
            lon, lat = self.beidou.geo_map(cx, cy, w, h, fix=fix)
            dets.append(Detection(
                class_id=cid,
                name_en=en[cid] if cid < len(en) else f"class_{cid}",
                name_zh=zh[cid] if cid < len(zh) else f"类别{cid}",
                conf=float(cf),
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                lon=lon, lat=lat,
            ))
        return dets

    # ============================ 视频检测 ============================ #
    def predict_video(
        self,
        input_path: str,
        output_dir: Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        max_det: int = 300,
        progress_cb=None,
        area_m2: float | None = None,
        meters_per_px: float | None = None,
    ) -> dict:
        """逐帧检测并写出标注视频。

        :param area_m2: 画面实际覆盖面积（m²），用于逐帧人群密度换算
        :param meters_per_px: 比例尺（米/像素），与 area_m2 二选一


        与早期实现的根本区别：**读写路径全由自己控制**，不再把 ultralytics 的
        ``save=True`` 交给它。原因有两个，都是踩过的坑：

        1. ``save_dir`` 指向输入文件所在目录时，ultralytics 见目录已存在会**自增**
           成 ``<job_id>-2/``，于是任务目录里只剩上传的原片，glob 一抓就把它当结果
           返回 —— 表现为"结果视频和输入字节数一模一样"，静默且难查。
        2. 结果文件名必须与上传件（``input.<ext>``）**不同名**，否则同目录覆盖风险。

        编码器也不轻信：``VideoWriter.isOpened()`` 返回 True 并不代表真能写，
        所以先探测、写完再回读校验（见 :func:`_probe_codec` / :func:`_count_frames`）。
        """
        if not self.registry.is_real():
            raise ModelNotReadyError("演示模式不支持视频推理")

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.registry.get_model()

        errors: list[str] = []
        stem = _ascii_slug(output_dir.name)
        for fourcc, suffix in _codec_order(self.settings.VIDEO_CODEC):
            out_path = output_dir / f"{_VIDEO_STEM}_{stem}{suffix}"
            try:
                stats = self._video_pass(
                    model, input_path, out_path, fourcc,
                    conf, iou, imgsz, max_det, progress_cb,
                    area_m2=area_m2, meters_per_px=meters_per_px,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("视频编码器 %s 失败：%s", fourcc, e)
                errors.append(f"{fourcc}: {e}")
                continue
            if stats is None:
                errors.append(f"{fourcc}: 写出后回读校验不通过（编码器不可用）")
                continue
            log.info("视频写出成功：%s（请求 %s / 实际 %s / %s 帧 / %.1f KB）",
                     out_path.name, fourcc, stats["codec"],
                     stats["frames"], stats["bytes"] / 1024)
            return {
                "output": str(out_path),
                "stats": stats,
                # 对外汇报实际编码器（可能与请求的不同，见 _read_fourcc）
                "codec": stats["codec"],
                "requested_codec": fourcc,
                "container": suffix,
            }

        raise RuntimeError("视频写出失败，已尝试所有编码器：" + "；".join(errors))

    def _video_pass(
        self,
        model,
        input_path: Path,
        out_path: Path,
        fourcc: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        progress_cb,
        area_m2: float | None = None,
        meters_per_px: float | None = None,
    ) -> dict | None:
        """跑一遍逐帧推理并写出，返回 stats；回读校验不通过返回 ``None``。"""
        import cv2

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise BadRequestError(f"无法打开视频：{input_path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if w <= 0 or h <= 0:
            cap.release()
            raise BadRequestError("视频尺寸无效（无法解码首帧）")

        n_classes = len(C.classes_en())
        class_count = [0] * n_classes
        total = 0
        frames = 0
        per_frame: list[int] = []

        # 人群密度：逐帧累积，最后汇总（含抽稀后的密度曲线）
        crowd_on = bool(self.settings.CROWD_ENABLED)
        series = FrameSeries(max_points=self.settings.CROWD_CURVE_POINTS) if crowd_on else None

        # ---- 编码器与输出路径 ----
        # 坑：``avc1``（libopenh264 分支）在 Windows 上**打不开含非 ASCII 字符的路径**，
        # ``isOpened()`` 直接 False；而 mp4v/XVID 走另一条 FFmpeg 分支，中文路径正常。
        # 本项目目录名常带中文（``新建文件夹``），于是 avc1 会被静默降级成 mp4v ——
        # 结果是 FMP4，浏览器 <video> 播不了。
        #
        # 修法：目标路径含非 ASCII 时，先写到**全 ASCII 临时路径**（目录名与文件名都要
        # ASCII），校验通过后再搬到最终位置。
        # 注意文件名也必须 ASCII：OpenCV 在 Windows 上把路径按 ANSI 处理，
        # 中文文件名会被"乱码化"落盘 —— 此时 ``Path.exists()`` 为 False，
        # 而 cv2 自己却能读到（读写两侧同样乱码），极易误判成"写出失败"。
        write_path = out_path
        if not str(out_path).isascii():
            import uuid

            write_path = _ascii_tmp_dir() / f"bds_{uuid.uuid4().hex}{out_path.suffix}"
            assert str(write_path).isascii(), write_path

        writer = cv2.VideoWriter(str(write_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            _unlink_quiet(write_path)
            raise RuntimeError(f"VideoWriter 打开失败（{fourcc}）")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1

                boxes = self._parse(model.predict(
                    frame, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                    device=self.registry.device, verbose=False,
                ))
                n = len(boxes)
                total += n
                per_frame.append(n)
                for (_x1, _y1, _x2, _y2, cid, _cf) in boxes:
                    if 0 <= cid < n_classes:
                        class_count[cid] += 1

                if series is not None:
                    c = analyze_crowd(
                        boxes, w, h,
                        area_m2=area_m2, meters_per_px=meters_per_px,
                        people_weight=self.settings.CROWD_PEOPLE_WEIGHT,
                        watch=self.settings.CROWD_WATCH,
                        warn=self.settings.CROWD_WARN,
                        danger=self.settings.CROWD_DANGER,
                    )
                    series.add(c["count"], c["density"], c["level"])

                if n:
                    dicts = [
                        {"class": cid, "conf": cf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                        for (x1, y1, x2, y2, cid, cf) in boxes
                    ]
                    frame = draw_detections_bgr(frame, dicts, w, h)
                writer.write(frame)

                if progress_cb:
                    progress_cb(frames, n_total or frames)
        finally:
            cap.release()
            writer.release()

        # 写完必须回读：OpenH264 缺失时 isOpened() 为 True 却写出 1KB 坏文件
        readback = _count_frames(write_path)
        size = write_path.stat().st_size if write_path.exists() else -1
        if frames == 0 or readback == 0 or size < 1024:
            log.warning("视频回读校验失败：%s 写出 %s 帧 / 回读 %s 帧 / %s B",
                        write_path.name, frames, readback, size)
            _unlink_quiet(write_path)
            return None

        codec = _read_fourcc(write_path) or fourcc

        # 校验通过后再搬到最终位置（跨盘也无妨，shutil.move 会退化成复制）
        if write_path != out_path:
            import shutil

            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(write_path), str(out_path))

        return {
            "total": total,
            "frames": frames,
            "readback_frames": readback,
            "avg_per_frame": round(total / frames, 2) if frames else 0.0,
            "max_per_frame": max(per_frame) if per_frame else 0,
            "class_count": [class_count[i] for i in range(n_classes)],
            "fps": round(fps, 2),
            "size": [w, h],
            # 报回读出来的真实 fourcc，而不是我们请求的那个（OpenCV 可能静默替换）
            "codec": codec,
            "requested_codec": fourcc,
            "bytes": size,
            "crowd": series.summary(
                watch=self.settings.CROWD_WATCH,
                warn=self.settings.CROWD_WARN,
                danger=self.settings.CROWD_DANGER,
            ) if series is not None else None,
        }


# ============================ 视频编码器 ============================ #
# (fourcc, 容器后缀) 候选表，按优先级排列：
#   avc1 -> 真 H.264，浏览器 <video> 可直接播放（首选）
#   mp4v -> FMP4，能解码但部分浏览器黑屏
#   XVID -> AVI，兼容性最广的老式兜底
_VIDEO_CODECS: tuple[tuple[str, str], ...] = (
    ("avc1", ".mp4"),
    ("mp4v", ".mp4"),
    ("XVID", ".avi"),
)

# 结果文件前缀。**绝不能与上传件同名**（上传件固定叫 ``input.<ext>``），
# 否则会出现"glob 抓回原片当结果"的静默错误。
_VIDEO_STEM = "result"


def _ascii_slug(raw: str, limit: int = 32) -> str:
    """把任意字符串压成 ASCII 文件名片段（只留字母数字与 ``-``/``_``）。

    结果文件名必须保持 ASCII：OpenCV 在 Windows 上按 ANSI 处理路径，
    含中文的文件名会被"乱码化"落盘，导致 ``Path.exists()`` 找不到文件
    （而 cv2 自己还能读到），排查起来非常费劲。
    """
    import uuid

    keep = [c for c in (raw or "") if c.isascii() and (c.isalnum() or c in "-_")]
    slug = "".join(keep)[:limit]
    return slug or uuid.uuid4().hex[:8]


def _unlink_quiet(path: Path) -> None:
    """删文件，失败也不抛（沙箱/回收站受限环境下删除常被拦截）。"""
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


@lru_cache(maxsize=1)
def _ascii_tmp_dir() -> Path:
    """返回一个**纯 ASCII 路径**的可写临时目录。

    为什么需要它：``avc1`` 在 Windows 上打不开含非 ASCII 字符的输出路径
    （``isOpened()`` 直接 False），而本项目目录名常带中文。所以含中文的
    目标路径要"先写 ASCII 临时目录，再搬过去"。

    候选依次尝试；若系统 TEMP 本身含中文（例如用户名是中文），退到盘根目录。
    """
    import os
    import tempfile

    cands = [
        Path(tempfile.gettempdir()) / "bds_video",
        Path(os.environ.get("SystemDrive", "C:") + "/bds_video_tmp"),
        Path("C:/bds_video_tmp"),
    ]
    for c in cands:
        if not str(c).isascii():
            continue
        try:
            c.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:  # noqa: BLE001
            continue
    # 兜底：即便含中文也只能用（总比不写强）
    return Path(tempfile.gettempdir())


@lru_cache(maxsize=8)
def _probe_codec(fourcc: str, suffix: str) -> bool:
    """试写一个极小的合成视频并回读，确认编码器**真的**可用。

    ``VideoWriter.isOpened()`` 返回 True 并不可信：OpenH264 缺失时它会写出
    1KB 级别的坏文件（编码器"假成功"）。只有回读帧数 > 0 才算通过。

    探测必须写在**纯 ASCII 路径**下，与真实写出的路径类别保持一致 ——
    否则会出现"探测通过、实际打开失败"这种自相矛盾的降级（``avc1`` 就是如此）。

    探测代价约几十毫秒，结果按 (fourcc, suffix) 在进程内缓存，每个任务不会重复探测。
    """
    import shutil
    import tempfile

    import cv2
    import numpy as np

    d = Path(tempfile.mkdtemp(prefix="probe_", dir=str(_ascii_tmp_dir())))
    p = d / f"probe{suffix}"
    try:
        wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), 10.0, (128, 128))
        if not wr.isOpened():
            return False
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(3):
            frame[:] = (i * 40, 80, 160)
            wr.write(frame)
        wr.release()
        if not p.exists() or p.stat().st_size < 256:
            return False
        return _count_frames(p) > 0
    except Exception:  # noqa: BLE001
        return False
    finally:
        # 沙箱/回收站受限环境可能拦截删除，失败可忽略（只是临时目录残留）
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


def _count_frames(path: Path) -> int:
    """回读视频，统计**真实可解码**帧数。

    这是判断"编码器是否真的写出了可播文件"的唯一可信判据。
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
    finally:
        cap.release()
    return n


def _read_fourcc(path: Path) -> str:
    """读出视频文件**实际**使用的 fourcc。

    为什么不直接相信请求的 fourcc：OpenCV 遇到不认识的 fourcc 会**静默替换**成
    默认编码器（例如 ``ZZZZ`` → mp4v），此时 ``isOpened()`` 仍为 True、文件也正常，
    但对外汇报的编码器名就是假的。所以只报回读出来的真实值。
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ""
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).replace("\x00", "").strip()


@lru_cache(maxsize=8)
def _codec_order(pref: str) -> tuple[tuple[str, str], ...]:
    """编码器尝试顺序：配置指定项优先，其余按 ``_VIDEO_CODECS`` 顺延。

    只保留探测通过者；若全部不通过，仍返回最后一个候选（让主流程报出真实错误，
    而不是在这里静默失败）。

    ``pref`` 不在候选表里时会被**忽略并告警**，而不是插到最前 —— 因为 OpenCV 对
    未知 fourcc 会静默换成别的编码器，那样我们就会对外汇报一个根本没生效的名字。
    """
    cands: list[tuple[str, str]] = list(_VIDEO_CODECS)
    pref = (pref or "").strip()
    if pref:
        hit = [c for c in cands if c[0].lower() == pref.lower()]
        if hit:
            cands = hit + [c for c in cands if c[0].lower() != pref.lower()]
        else:
            log.warning("APP_VIDEO_CODEC=%s 不在候选表 %s 中，已忽略",
                        pref, [c[0] for c in cands])

    ok = [c for c in cands if _probe_codec(*c)]
    return tuple(ok or cands[-1:])


def _nms(boxes, iou: float = 0.5):
    """纯 numpy 的按类 NMS 合并。boxes: [(x1,y1,x2,y2,cls,conf), ...]"""
    if not boxes:
        return []
    import numpy as np

    arr = np.array([[b[0], b[1], b[2], b[3], b[4], b[5]] for b in boxes], dtype=np.float32)
    out = []
    for cls in np.unique(arr[:, 4]):
        idx = np.where(arr[:, 4] == cls)[0]
        bb = arr[idx]
        order = bb[:, 5].argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(bb[i, 0], bb[order[1:], 0])
            yy1 = np.maximum(bb[i, 1], bb[order[1:], 1])
            xx2 = np.minimum(bb[i, 2], bb[order[1:], 2])
            yy2 = np.minimum(bb[i, 3], bb[order[1:], 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_i = (bb[i, 2] - bb[i, 0]) * (bb[i, 3] - bb[i, 1])
            area_j = (bb[order[1:], 2] - bb[order[1:], 0]) * (bb[order[1:], 3] - bb[order[1:], 1])
            union = area_i + area_j - inter
            ovr = np.where(union > 0, inter / union, 0.0)
            order = order[1:][ovr <= iou]
        for i in keep:
            r = bb[i]
            out.append((float(r[0]), float(r[1]), float(r[2]), float(r[3]), int(r[4]), float(r[5])))
    return out
