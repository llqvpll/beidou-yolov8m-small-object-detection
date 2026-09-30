#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后端启动器（对标原 web/start_web.py，但面向 FastAPI）。

职责：
  1) 把自定义模块注册进 ultralytics（通过子进程跑 install_custom_modules.py，
     避免在已导入 ultralytics 的本进程里打补丁导致不生效）；
  2) 启动 uvicorn（自动预热模型 / 进入演示模式）。

用法：
    python server/run.py                 # 默认 8000
    APP_PORT=8080 python server/run.py   # 自定义端口
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 修正版代码/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_custom_modules() -> None:
    try:
        print("[run] 检查自定义模块注册（EMA/DySample/GSConv/VoVGSCSP）...")
        subprocess.run(
            [sys.executable, str(ROOT / "install_custom_modules.py")],
            check=False, cwd=str(ROOT),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[run][警告] 自定义模块注册步骤异常：{e}")


def main() -> None:
    _ensure_custom_modules()

    import uvicorn

    from server.config import get_settings

    s = get_settings()
    print("=" * 60)
    print("  北斗 · 改进 YOLOv8m 小目标识别辅助系统 · 推理后端")
    print("=" * 60)
    print(f"  文档:    http://localhost:{s.PORT}/docs")
    print(f"  OpenAPI: http://localhost:{s.PORT}/openapi.json")
    print(f"  API:     {s.API_V1_PREFIX}/...")
    print("=" * 60)
    uvicorn.run(
        "server.main:app",
        host=s.HOST,
        port=s.PORT,
        workers=s.WORKERS,
        log_level=s.LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
