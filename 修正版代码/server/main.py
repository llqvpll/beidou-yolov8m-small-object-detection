"""应用入口：app 工厂 + 生命周期 + 中间件 + 路由装配 + 异常处理器。

遵循 high-star 实践（tiangolo / full-stack-fastapi-template）：
  - ``create_app()`` 工厂，便于测试时独立构建；
  - ``lifespan`` 启动期预热模型 / 加载任务库，关闭期取消在途任务；
  - 中间件：CORS + 关联 ID（X-Request-ID，串联日志与错误信封）；
  - 统一异常处理器（见 core/errors）。
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .core.deps import get_beidou, get_job_store, get_registry
from .core.errors import register_exception_handlers
from .logging_setup import configure_logging
from .routers import detect, files, health, jobs, meta, metrics, track, video

log = logging.getLogger("server")

# 前端静态资源目录（零依赖 SPA，随后端一起发布）
STATIC_DIR = Path(__file__).resolve().parent / "static"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """为每个请求分配关联 ID（支持客户端透传 X-Request-ID）。"""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 预热模型（失败则降级 demo，不阻塞启动）
    if settings.PRELOAD_MODEL:
        try:
            get_registry().ensure_ready()
        except Exception as e:  # noqa: BLE001
            log.warning("模型预热失败（已降级 demo）：%s", e)

    get_job_store().load()

    # 启动定位源（串口/回放/模拟）。失败**不影响启动**：
    # 定位不可用时接口如实返回"无定位"，推理与人机界面照常可用。
    try:
        get_beidou().start()
    except Exception as e:  # noqa: BLE001
        log.warning("北斗定位源启动失败（不影响推理）：%s", e)

    log.info("启动完成 | mode=%s device=%s", get_registry().mode, get_registry().device)
    yield

    # 关闭：取消在途视频任务，并停掉定位读取线程
    store = get_job_store()
    for task in list(store._tasks.values()):
        task.cancel()
    try:
        get_beidou().stop()
    except Exception:  # noqa: BLE001
        pass
    log.info("服务关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description=(
            "北斗 + 改进 YOLOv8m 小目标识别辅助系统 · 生产级推理后端（FastAPI）。\n\n"
            "功能：图像/批量/视频检测、内置切片推理（小目标友好）、北斗坐标映射、"
            "双模式（真实/演示）优雅降级、任务轮询、健康检查与 OpenAPI 文档。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationIdMiddleware)
    register_exception_handlers(app)

    # / 与探针不加前缀
    app.include_router(health.router)
    # 前端探测会先试 /api/v1/health，这里补一份别名（不进 OpenAPI，避免重复 operationId）
    app.include_router(health.router, prefix=settings.API_V1_PREFIX, include_in_schema=False)
    # 业务接口统一前缀 /api/v1
    app.include_router(meta.router, prefix=settings.API_V1_PREFIX)
    app.include_router(detect.router, prefix=settings.API_V1_PREFIX)
    app.include_router(video.router, prefix=settings.API_V1_PREFIX)
    app.include_router(jobs.router, prefix=settings.API_V1_PREFIX)
    app.include_router(files.router, prefix=settings.API_V1_PREFIX)
    app.include_router(metrics.router, prefix=settings.API_V1_PREFIX)
    app.include_router(track.router, prefix=settings.API_V1_PREFIX)

    # 前端静态资源（必须最后挂载："/" 会兜住所有未匹配路径）。
    # 与 API 同源发布，前端无需 CORS，也无需另起静态服务器。
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
        log.info("前端已挂载：http://localhost:%s/  ← %s", settings.PORT, STATIC_DIR)
    else:
        log.warning("未找到前端目录 %s，仅提供 API", STATIC_DIR)

    return app


app = create_app()
