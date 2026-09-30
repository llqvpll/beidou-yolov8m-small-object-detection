from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..core.deps import get_job_store
from ..core.errors import NotFoundError
from ..schemas.video import VideoJob
from ..services.jobs import JobStore

router = APIRouter(tags=["任务"])


@router.get("/jobs", response_model=list[VideoJob], summary="列出近期视频任务")
def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    store: JobStore = Depends(get_job_store),
) -> list[VideoJob]:
    return store.list(limit)


@router.get("/jobs/{job_id}", response_model=VideoJob, summary="查询任务状态/结果")
def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> VideoJob:
    job = store.get(job_id)
    if job is None:
        raise NotFoundError(f"任务不存在：{job_id}")
    return job
