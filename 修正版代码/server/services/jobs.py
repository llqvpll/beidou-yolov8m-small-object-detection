"""视频任务管理：内存 + JSON 持久化的任务库，及异步处理协程。

视频推理是长耗时任务，采用「提交即返回 job_id + 轮询」模式：
  POST /video -> 202 + job_id
  GET  /jobs/{id} -> 进度 / 结果
这避免了 HTTP 长连接超时，也便于水平扩展（任务可交给独立 worker）。
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..schemas.video import JobStatus, VideoJob
from .inference import InferenceService

log = logging.getLogger("server.jobs")


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.dir = settings.OUTPUT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "jobs.json"
        self._jobs: dict[str, VideoJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for j in data:
                self._jobs[j["job_id"]] = VideoJob(**j)
        except Exception as e:  # noqa: BLE001
            log.warning("任务库加载失败（忽略）：%s", e)

    def _persist(self) -> None:
        try:
            self._path.write_text(
                json.dumps([j.model_dump() for j in self._jobs.values()], default=str),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("任务库持久化失败：%s", e)

    def create(self, job_id: str, params: dict) -> VideoJob:
        now = datetime.now(timezone.utc)
        job = VideoJob(
            job_id=job_id, status=JobStatus.queued, created_at=now, updated_at=now,
            params=params, progress={"done": 0, "total": 0}, result={}, error=None,
        )
        self._jobs[job_id] = job
        self._persist()
        return job

    def get(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> VideoJob | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = datetime.now(timezone.utc)
        self._persist()
        return job

    def list(self, limit: int = 20) -> list[VideoJob]:
        items = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return items[:limit]

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks[job_id] = task

    def unregister_task(self, job_id: str) -> None:
        self._tasks.pop(job_id, None)


async def run_video_job(
    job_id: str,
    input_path: str,
    params: dict,
    inference: InferenceService,
    store: JobStore,
) -> None:
    """后台协程：跑视频推理并更新任务状态。"""
    try:
        store.update(job_id, status=JobStatus.processing)

        def progress(done: int, total: int = 0) -> None:
            # 总帧数由推理侧从视频容器读出（比路由层的预估更准）
            store.update(job_id, progress={"done": done, "total": total or params.get("frames") or 0})

        out = await asyncio.to_thread(
            inference.predict_video,
            input_path,
            store.dir / job_id,
            conf=params.get("conf", 0.25),
            iou=params.get("iou", 0.45),
            imgsz=params.get("imgsz", 640),
            max_det=params.get("max_det", 300),
            progress_cb=progress,
            area_m2=params.get("area_m2"),
            meters_per_px=params.get("meters_per_px"),
        )
        video_url = None
        if out["output"]:
            fname = Path(out["output"]).name
            video_url = f"/api/v1/files/{job_id}/{fname}"
        store.update(
            job_id,
            status=JobStatus.completed,
            progress={"done": out["stats"]["frames"], "total": out["stats"]["frames"]},
            result={
                "video_url": video_url,
                "stats": out["stats"],
                "codec": out.get("codec"),
                "container": out.get("container"),
            },
        )
        log.info("视频任务完成 %s：%s 帧 / %s 目标 / %s",
                 job_id, out["stats"]["frames"], out["stats"]["total"], out.get("codec"))
    except Exception as e:  # noqa: BLE001
        log.exception("视频任务失败 %s", job_id)
        store.update(job_id, status=JobStatus.failed, error=str(e))
    finally:
        store.unregister_task(job_id)
