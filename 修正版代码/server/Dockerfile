# 北斗·改进YOLOv8m 推理后端 镜像
# 构建上下文为「修正版代码/」（即 compose 的 context: ..）
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PRELOAD_MODEL=true

WORKDIR /app

# opencv 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# torch：默认 CPU 版。GPU 部署请把这一行替换为 CUDA 版：
#   RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 拷贝整个项目（修正版代码/）
COPY . /app

EXPOSE 8000
CMD ["python", "server/run.py"]
