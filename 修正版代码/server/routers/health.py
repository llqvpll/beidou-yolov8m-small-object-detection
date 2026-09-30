from __future__ import annotations

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter(tags=["健康检查"])


@router.get("/healthz", summary="存活探针 (liveness)")
def healthz() -> dict:
    """k8s liveness：进程在跑即返回 200。"""
    return {"status": "ok"}


@router.get("/health", summary="健康检查（前端探测用别名）")
def health() -> dict:
    return {"status": "ok", "service": "bds-yolov8m-api"}


@router.get("/readyz", summary="就绪探针 (readiness)")
def readyz() -> dict:
    """k8s readiness：应用已完成启动（lifespan）即返回 200。"""
    return {"status": "ok"}


@router.get("/info", summary="服务信息")
def info() -> dict:
    """服务元信息。

    注意：这里**不能**用 ``/``，否则会挡住挂在根路径的前端页面。
    """
    s = get_settings()
    return {
        "project": s.PROJECT_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "api_prefix": s.API_V1_PREFIX,
        "frontend": "/",
        "note": "前端页面在 /；接口在 " + s.API_V1_PREFIX + " 下",
    }
