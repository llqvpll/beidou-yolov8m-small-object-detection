from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..config import Settings, get_settings
from ..core.deps import get_inference, get_job_store
from ..core.errors import ModelNotReadyError, PayloadTooLargeError
from ..core.security import require_api_key
from ..schemas.video import CreateVideoResponse, JobStatus
from ..services.inference import InferenceService
from ..services.jobs import JobStore, run_video_job

router = APIRouter(tags=["视频"])


@router.post("/video", status_code=202, response_model=CreateVideoResponse,
             summary="视频目标检测（异步任务）")
async def create_video(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    max_det: int = Form(300),
    slice: bool = Form(False),
    area_m2: float | None = Form(None),       # 画面实际覆盖面积（m²），用于逐帧人群密度
    meters_per_px: float | None = Form(None),  # 或比例尺（米/像素）
    inference: InferenceService = Depends(get_inference),
    store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_api_key),
) -> CreateVideoResponse:
    data = await file.read()
    if len(data) > settings.max_video_bytes:
        raise PayloadTooLargeError(f"视频过大（>{settings.MAX_VIDEO_MB}MB）")
    if not inference.registry.is_real():
        raise ModelNotReadyError(
            "演示模式不支持视频推理；请在装有 ultralytics + 权重的机器上运行。",
            details={"mode": "demo"},
        )

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    inp = store.dir / job_id / ("input" + suffix)
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_bytes(data)

    params = {
        "conf": conf, "iou": iou, "imgsz": imgsz, "max_det": max_det, "slice": slice,
        "area_m2": area_m2, "meters_per_px": meters_per_px,
    }
    store.create(job_id, params)

    task = asyncio.create_task(run_video_job(job_id, str(inp), params, inference, store))
    store.register_task(job_id, task)

    return CreateVideoResponse(
        job_id=job_id,
        status=JobStatus.queued,
        status_url=f"{settings.API_V1_PREFIX}/jobs/{job_id}",
        message="已加入队列，轮询 status_url 获取进度与结果",
    )
