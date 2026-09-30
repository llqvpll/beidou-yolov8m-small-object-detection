"""依赖注入：把配置 / 单例服务（模型注册表、推理、北斗、任务库）注入路由。

采用模块级懒单例：第一次使用时初始化，进程内复用。
模型注册表在初始化时会 import ultralytics（较重），因此仅在
``PRELOAD_MODEL`` 或首次请求时触发。
"""
from __future__ import annotations

from ..config import Settings, get_settings
from ..services.beidou import BeidouService
from ..services.inference import InferenceService
from ..services.jobs import JobStore
from ..services.metrics import MetricsService
from ..services.model_registry import ModelRegistry
from ..services.track_store import TrackService

_registry: ModelRegistry | None = None
_beidou: BeidouService | None = None
_inference: InferenceService | None = None
_store: JobStore | None = None
_metrics: MetricsService | None = None
_track: TrackService | None = None


def get_settings_dep() -> Settings:
    return get_settings()


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(get_settings())
        _registry.ensure_ready()
    return _registry


def get_beidou() -> BeidouService:
    global _beidou
    if _beidou is None:
        _beidou = BeidouService(get_settings())
    return _beidou


def get_inference() -> InferenceService:
    global _inference
    if _inference is None:
        _inference = InferenceService(get_registry(), get_beidou(), get_settings())
    return _inference


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(get_settings())
        _store.load()
    return _store


def get_metrics() -> MetricsService:
    """训练指标服务（只读解析已加载权重所属 run 的训练产物）。"""
    global _metrics
    if _metrics is None:
        _metrics = MetricsService(get_registry(), get_settings())
    return _metrics


def get_track() -> TrackService:
    """存储卡日志轨迹服务（离线：一次导入，按时刻查询）。"""
    global _track
    if _track is None:
        _track = TrackService(get_settings())
    return _track
