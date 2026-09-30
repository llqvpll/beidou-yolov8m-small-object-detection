from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..core.deps import get_inference
from ..core.errors import BadRequestError, PayloadTooLargeError
from ..schemas.detection import DetectResponse
from ..services.inference import InferenceService

router = APIRouter(tags=["检测"])


# --------------------------- multipart（浏览器/表单） --------------------------- #
@router.post("/detect", response_model=DetectResponse, summary="图像目标检测（multipart）")
async def detect_multipart(
    file: UploadFile | None = File(None),
    image: str | None = Form(None),            # 备用：base64 data URL
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    max_det: int = Form(300),
    slice: bool = Form(False),
    draw: bool = Form(False),
    strict: bool = Form(False),
    return_original: bool = Form(False),
    area_m2: float | None = Form(None),       # 画面实际覆盖面积（m²），用于算人群密度
    meters_per_px: float | None = Form(None),  # 或比例尺（米/像素）
    inference: InferenceService = Depends(get_inference),
    settings: Settings = Depends(get_settings),
) -> dict:
    if file is not None:
        data = await file.read()
        if not data:
            raise BadRequestError("空文件")
        if len(data) > settings.max_upload_bytes:
            raise PayloadTooLargeError(f"图片过大（>{settings.MAX_UPLOAD_MB}MB）")
    elif image:
        data = None
    else:
        raise BadRequestError("请上传 file 或提供 image(base64)")
    return await inference.detect(
        image_bytes=data, image_data_url=image, conf=conf, iou=iou, imgsz=imgsz,
        max_det=max_det, slice_=slice, draw=draw, strict=strict,
        return_original=return_original, area_m2=area_m2, meters_per_px=meters_per_px,
    )


# --------------------------- JSON（程序化 / 摄像头帧） --------------------------- #
class DetectJSON(BaseModel):
    image: str                                    # base64 data URL
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 300
    slice: bool = False
    draw: bool = False
    strict: bool = False
    return_original: bool = False
    area_m2: float | None = None        # 画面实际覆盖面积（m²）
    meters_per_px: float | None = None  # 或比例尺（米/像素）


@router.post("/detect/json", response_model=DetectResponse, summary="图像目标检测（JSON base64）")
async def detect_json(
    body: DetectJSON,
    inference: InferenceService = Depends(get_inference),
) -> dict:
    return await inference.detect(
        image_data_url=body.image, conf=body.conf, iou=body.iou, imgsz=body.imgsz,
        max_det=body.max_det, slice_=body.slice, draw=body.draw, strict=body.strict,
        return_original=body.return_original,
        area_m2=body.area_m2, meters_per_px=body.meters_per_px,
    )


# --------------------------- 批量（多图） --------------------------- #
@router.post("/detect/batch", response_model=dict, summary="批量图像检测")
async def detect_batch(
    files: list[UploadFile] = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    max_det: int = Form(300),
    slice: bool = Form(False),
    area_m2: float | None = Form(None),
    meters_per_px: float | None = Form(None),
    inference: InferenceService = Depends(get_inference),
    settings: Settings = Depends(get_settings),
) -> dict:
    if len(files) > 16:
        raise BadRequestError("单次批量最多 16 张")
    results = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_bytes:
            raise PayloadTooLargeError(f"图片 {f.filename} 过大")
        r = await inference.detect(
            image_bytes=data, conf=conf, iou=iou, imgsz=imgsz,
            max_det=max_det, slice_=slice, draw=False, return_original=False,
            area_m2=area_m2, meters_per_px=meters_per_px,
        )
        results.append({"filename": f.filename, **r})
    return {"count": len(results), "results": results}
