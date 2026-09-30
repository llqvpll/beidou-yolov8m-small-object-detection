"""可选的 API Key 鉴权（默认关闭）。

开启方式：环境变量 ``APP_ENABLE_AUTH=true`` 且 ``APP_API_KEY=xxx``。
受保护的写操作（视频提交等）通过依赖 ``require_api_key`` 接入。
"""
from __future__ import annotations

from fastapi import Header, Request

from ..config import get_settings
from .errors import UnauthorizedError


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.ENABLE_AUTH:
        return
    if not settings.API_KEY:
        # 配置了开启却没设 key，属于部署错误：拒绝一切写请求，避免“假安全”
        raise UnauthorizedError("服务端未配置 APP_API_KEY，拒绝写操作")
    if x_api_key != settings.API_KEY:
        raise UnauthorizedError()
