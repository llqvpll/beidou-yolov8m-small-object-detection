from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..core.deps import get_beidou, get_registry
from ..schemas.detection import BeidouInfo, ClassInfo, ModelInfo, StatusResponse
from ..services.beidou import BeidouService
from ..services.classes import class_infos
from ..services.model_registry import ModelRegistry

router = APIRouter(tags=["元信息"])


class SwitchRequest(BaseModel):
    weight: str


@router.get("/status", response_model=StatusResponse, summary="运行状态（模式/设备/模型/类别/北斗）")
def status(
    registry: ModelRegistry = Depends(get_registry),
    beidou: BeidouService = Depends(get_beidou),
    settings: Settings = Depends(get_settings),
) -> StatusResponse:
    return StatusResponse(
        mode=registry.mode,
        device=registry.device if registry.is_real() else None,
        model=ModelInfo(**registry.model_info),
        classes=class_infos(),
        beidou=beidou.info(),
        engine_available=registry.ultralytics_ok,
        slicing_available=settings.SLICE_ENABLED,
        message=registry.error,
    )


@router.get("/classes", response_model=list[ClassInfo], summary="检测类别（中英文 + 颜色）")
def classes() -> list[ClassInfo]:
    return class_infos()


@router.get("/beidou", response_model=BeidouInfo, summary="北斗定位信息（含定位质量/卫星/HDOP）")
def beidou(beidou: BeidouService = Depends(get_beidou)) -> BeidouInfo:
    return beidou.info()


@router.post("/beidou/reload", response_model=BeidouInfo, summary="重新扫描定位源（插上模块后无需重启）")
def beidou_reload(beidou: BeidouService = Depends(get_beidou)) -> BeidouInfo:
    """重新挑一次定位源：串口插拔、回放文件替换后用它，不必重启后端。

    只重新扫描设备，不重读 ``.env``（配置是进程启动时的快照）。
    """
    return beidou.reload()


@router.get("/models", summary="模型权重候选与加载情况")
def models(registry: ModelRegistry = Depends(get_registry)) -> dict:
    return {
        "mode": registry.mode,
        "loaded": registry.is_real(),
        "loaded_variant": registry.loaded_variant,
        "device": registry.device,
        "error": registry.error,
        "candidates": registry.settings.weights_candidates,
        "candidates_exist": registry.candidates_exist(),
    }


@router.post("/models/switch", summary="切换当前加载的权重（进程内原地替换，不重启）")
def switch_model(
    req: SwitchRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> dict:
    """在已运行的后端内切换权重，返回切换后的模型信息。

    仅接受 ``GET /models`` 候选列表中的路径，避免任意路径注入。
    失败（含回退失败）以 400 返回，原模型尽量保持可用。
    """
    result = registry.switch_model(req.weight)
    if not result.get("ok"):
        raise HTTPException(
            status_code=400,
            detail={"error": result.get("error", "切换失败"), "reverted_to": result.get("reverted_to")},
        )
    return result
