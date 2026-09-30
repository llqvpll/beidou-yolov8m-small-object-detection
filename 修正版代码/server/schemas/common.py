from __future__ import annotations

from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str | None = None


class MessageResponse(BaseModel):
    message: str
    detail: str | None = None
