"""存储卡日志轨迹：导入 / 查询 / 对齐。

对应用户的实际路线：模块接单片机 → MCU 把 NMEA + 时间写进存储卡 →
事后把「视频」和「存储卡内容」一起导入，拖动时间轴查看任意时刻的定位。
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..core.deps import get_track
from ..core.errors import PayloadTooLargeError
from ..core.security import require_api_key
from ..services.track_store import TrackService

router = APIRouter(tags=["轨迹"])


class AlignRequest(BaseModel):
    """三者只应给一个；给多个时按 utc > origin_s > delta_s 取用，并在响应里说明。"""

    utc: str | None = None        # 视频第 0 秒的绝对时刻（ISO，UTC）
    origin_s: float | None = None  # 视频第 0 秒对应的日志时刻
    delta_s: float | None = None   # 相对日志起点整体平移 N 秒


@router.post("/track/import", summary="导入存储卡日志（自动识别时间戳格式，返回诊断回执）")
async def import_track(
    file: UploadFile = File(...),
    track: TrackService = Depends(get_track),
    settings: Settings = Depends(get_settings),
    _: None = Depends(require_api_key),
) -> dict:
    """把日志解析成可拖动的时间索引轨迹。

    返回的 ``diagnose`` 是**给用户看的**：识别出的时间戳格式、历元数、时长、
    认不出的行、以及下一步建议。格式对不上时用户能立刻看到差在哪，
    不必反复猜"为什么没反应"。
    """
    data = await file.read()
    if len(data) > settings.max_track_bytes:
        raise PayloadTooLargeError(f"日志过大（>{settings.MAX_TRACK_MB}MB）")

    # 文件名只用来显示；落盘一律用随机名，避免用户的路径穿越进来
    name = Path(file.filename).name if file.filename else "track.log"
    dest = settings.track_dir / (uuid.uuid4().hex[:12] + "_" + name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    return track.load(dest, name=name)


@router.get("/track", summary="当前轨迹状态（摘要 + 诊断 + 抽稀折线）")
def track_state(track: TrackService = Depends(get_track)) -> dict:
    return track.state()


@router.get("/track/at", summary="按视频时刻取定位帧（拖动时间轴用）")
def track_at(
    t: float = Query(..., description="视频时间轴上的秒数"),
    interpolate: bool = Query(True, description="是否在相邻历元间插值"),
    track: TrackService = Depends(get_track),
) -> dict:
    """返回该时刻的定位帧（含卫星明细，可直接驱动星空图）。

    未对齐时结果里带 ``aligned=false``，界面必须据此提示"当前按起点对齐"。
    """
    fix = track.at(t, interpolate=interpolate)
    if fix is None:
        return {"found": False, "t": t, "fix": None,
                "message": "还没有导入日志，或日志里没有可用的定位点。"}
    # 回显 t_video：fix["t"] 是**日志时间轴**上的值（绝对时间日志下就是个 Unix 秒），
    # 调用方真正要的是视频时刻，别让它自己去记。
    return {"found": True, "t": t, "t_video": t, "fix": fix}


@router.post("/track/align", summary="设置视频与日志的对齐锚点")
def track_align(
    req: AlignRequest,
    track: TrackService = Depends(get_track),
) -> dict:
    return track.align(origin_s=req.origin_s, delta_s=req.delta_s, utc=req.utc)


@router.delete("/track", summary="清除当前导入的轨迹")
def track_clear(track: TrackService = Depends(get_track)) -> dict:
    track.clear()
    return {"ok": True, "message": "已清除。"}
