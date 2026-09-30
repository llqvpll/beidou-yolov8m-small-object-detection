from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..core.deps import get_job_store
from ..core.errors import NotFoundError
from ..services.jobs import JobStore

router = APIRouter(tags=["文件"])


@router.get("/files/{job_id}/{filename}", summary="获取视频推理结果文件")
def serve_file(
    job_id: str,
    filename: str,
    store: JobStore = Depends(get_job_store),
) -> FileResponse:
    base = (store.dir / job_id).resolve()
    target = (base / filename).resolve()  # 解析 ../ 等，防目录穿越

    # 目标文件必须位于任务目录内，且确实存在
    if base not in target.parents or not target.exists():
        raise NotFoundError("文件不存在")

    mt = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    return FileResponse(str(target), media_type=mt, filename=target.name)
