"""训练指标路由：把训练产物（results.csv / args.yaml / 对比表 / 论文配图）暴露给前端。

前端「模型 / 分析」视图据此渲染真实收敛曲线与消融对比，避免写死假数据。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..config import Settings, get_settings
from ..core.deps import get_metrics, get_registry
from ..core.errors import NotFoundError
from ..services.metrics import MetricsService, artifact_path
from ..services.model_registry import ModelRegistry

router = APIRouter(tags=["训练指标"])


@router.get("/metrics", summary="已加载权重对应 run 的训练指标与消融对比")
def metrics(svc: MetricsService = Depends(get_metrics)) -> dict:
    return svc.snapshot()


@router.get("/metrics/runs", summary="列出训练根目录下所有可用 run")
def runs(svc: MetricsService = Depends(get_metrics)) -> dict:
    items = svc.list_runs()
    return {"count": len(items), "items": items}


@router.get("/metrics/artifact/{name}", summary="训练配图（results.png / PR 曲线 / 混淆矩阵…）")
def artifact(
    name: str,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    p = artifact_path(registry, settings, name)
    if p is None:
        raise NotFoundError(f"未找到训练配图 {name}（或当前 run 目录无此文件）")
    return FileResponse(p)
