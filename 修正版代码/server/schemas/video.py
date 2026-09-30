from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class VideoStats(BaseModel):
    total: int = 0
    frames: int = 0
    avg_per_frame: float = 0.0
    max_per_frame: int = 0
    class_count: list[int] = []


class VideoJob(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: dict = {}
    progress: dict = {"done": 0, "total": 0}
    result: dict = {}      # {"video_url": ..., "stats": {...}}
    error: str | None = None


class CreateVideoResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    message: str
