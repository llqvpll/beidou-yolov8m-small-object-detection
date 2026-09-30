"""北斗 · 改进 YOLOv8m 小目标识别辅助系统 —— 生产级推理后端。

架构借鉴高收藏开源设计：
  - FastAPI / tiangolo 全家桶（分层 api / core / services / schemas）
  - roboflow/inference：模型注册表（ModelManager）+ 标准化 CV API + 关联 ID 中间件
  - 分层 + Pydantic Settings + 依赖注入 + 自定义异常处理器 + 健康检查探针

本包使用相对导入，运行时要求 ``修正版代码/`` 在 ``sys.path`` 上
（``run.py`` / ``uvicorn server.main:app`` 已保证），以便同时导入
``custom_modules``（自定义模块注册）与 ``ultralytics``。
"""

__version__ = "1.0.0"
