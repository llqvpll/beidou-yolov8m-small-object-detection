"""pytest 配置：强制演示模式，保证测试确定且快速（不加载模型/不依赖 GPU）。"""
import os
import sys
from pathlib import Path

# 必须在 import server 之前设置，使 get_settings() 读取到
os.environ["APP_FORCE_DEMO"] = "true"
os.environ["APP_PRELOAD_MODEL"] = "false"
# 定位源固定为模拟：auto 会去扫本机串口，有真模块时测试结果随设备变化，
# 测试必须是确定性的。串口/回放的行为由 test_gnss.py 单独构造源来测。
os.environ["APP_GNSS_SOURCE"] = "mock"

ROOT = Path(__file__).resolve().parent.parent.parent  # 修正版代码/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from server.main import app

    with TestClient(app) as c:
        yield c
