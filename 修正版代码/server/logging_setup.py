"""统一日志配置：控制台格式日志 + 可选 JSON 行格式。

关联 ID（request id）由中间件写入 ``logging.LogRecord`` 的 ``request_id``
属性，便于在日志中串联一次请求的全部轨迹。
"""
from __future__ import annotations

import logging
import sys

REQUIRED_KEYS = ("request_id",)

_OLD_FACTORY = logging.getLogRecordFactory()


class _Record(logging.LogRecord):
    request_id: str = "-"  # type: ignore[assignment]


def _record_factory(*args, **kwargs):  # noqa: ANN002, ANN003
    record = _OLD_FACTORY(*args, **kwargs)
    # 把 request_id 透传到每条记录；中间件会写入 LogRecord 的属性
    return record


def configure_logging(level: str = "info", json_logs: bool = False) -> None:
    logging.setLogRecordFactory(_record_factory)
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore

            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
                )
            )
        except Exception:  # pragma: no cover - json logger 可选
            json_logs = False
    if not json_logs:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # 第三方库降噪
    for noisy in ("uvicorn.access", "uvicorn.error", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 把 request_id 字段补到 LogRecord，避免格式化时报缺失
    logging.LoggerAdapter  # 触发导入，保持引用

    logging.getLogger("server").info("日志系统已初始化 (level=%s, json=%s)", level, json_logs)
