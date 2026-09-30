"""统一异常体系 + 全局异常处理器。

所有业务异常都继承 :class:`AppError`，返回一致的错误信封：

    {
      "error": {
        "code": "MODEL_NOT_READY",
        "message": "推理引擎不可用：ultralytics 未安装",
        "details": null
      },
      "request_id": "a1b2c3..."
    }

同时接管 FastAPI 的 ``RequestValidationError`` / ``StarletteHTTPException`` /
未捕获异常，保证任何错误都走同一信封格式。
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("server.errors")


class AppError(Exception):
    """业务异常的基类。

    :param code: 机器可读错误码（全大写蛇形），前端可据此分支。
    :param message: 人类可读信息（可含中文）。
    :param status_code: HTTP 状态码。
    :param details: 可选的结构化细节（如字段级错误）。
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


class BadRequestError(AppError):
    def __init__(self, message: str, *, code: str = "BAD_REQUEST", details=None):
        super().__init__(message, code=code, status_code=400, details=details)


class UnauthorizedError(AppError):
    def __init__(self, message: str = "缺少或无效的 API Key", *, code: str = "UNAUTHORIZED"):
        super().__init__(message, code=code, status_code=401)


class NotFoundError(AppError):
    def __init__(self, message: str, *, code: str = "NOT_FOUND", details=None):
        super().__init__(message, code=code, status_code=404, details=details)


class UnsupportedMediaError(AppError):
    def __init__(self, message: str, *, code: str = "UNSUPPORTED_MEDIA", details=None):
        super().__init__(message, code=code, status_code=415, details=details)


class PayloadTooLargeError(AppError):
    def __init__(self, message: str, *, code: str = "PAYLOAD_TOO_LARGE", details=None):
        super().__init__(message, code=code, status_code=413, details=details)


class ModelNotReadyError(AppError):
    def __init__(self, message: str, *, code: str = "MODEL_NOT_READY", details=None):
        super().__init__(message, code=code, status_code=503, details=details)


def _envelope(code: str, message: str, details: Any | None, request: Request | None) -> dict:
    rid = getattr(request.state, "request_id", "-") if request is not None else "-"
    return {
        "error": {"code": code, "message": message, "details": details},
        "request_id": rid,
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError):  # noqa: ANN001
        log.warning("AppError[%s] %s | rid=%s", exc.code, exc.message,
                    getattr(request.state, "request_id", "-"))
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details, request),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):  # noqa: ANN001
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "VALIDATION_ERROR",
                "请求参数校验失败",
                exc.errors(),
                request,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_handler(request: Request, exc: StarletteHTTPException):  # noqa: ANN001
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                f"HTTP_{exc.status_code}",
                str(exc.detail),
                None,
                request,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception):  # noqa: ANN001
        log.exception("未捕获异常 rid=%s", getattr(request.state, "request_id", "-"))
        return JSONResponse(
            status_code=500,
            content=_envelope(
                "INTERNAL_ERROR",
                "服务器内部错误，请稍后重试或联系管理员",
                None,
                request,
            ),
        )
