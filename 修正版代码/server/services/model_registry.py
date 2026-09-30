"""模型注册表（借鉴 roboflow/inference 的 ModelManager 思路）。

职责：
  - 进程内单例，懒加载并缓存 YOLO 模型；
  - 启动期确保自定义模块已注册进 ultralytics（写补丁，零侵入）；
  - 自动探测 CUDA / 选择可用权重；
  - 双模式：可用则 ``real``，否则优雅降级为 ``demo``（接口永不崩）。
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import Settings
from .classes import CLASSES_ZH

log = logging.getLogger("server.model_registry")


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode: str = "demo"
        self.models: dict[str, object] = {}
        self.loaded_variant: str | None = None
        self.run_dir: Path | None = None       # 权重所属训练产物目录（供 /metrics 读取）
        self.model_info: dict = {"classes": len(CLASSES_ZH), "loaded": False}
        self.device: str = "cpu"
        self.ultralytics_ok: bool = False
        self.error: str | None = None

    # ------------------------------------------------------------------ #
    def ensure_ready(self) -> None:
        """探测运行环境并（尝试）加载权重。任何失败都降级为 demo，不抛异常。"""
        if self.settings.FORCE_DEMO:
            self.mode = "demo"
            self.error = "已通过 APP_FORCE_DEMO 强制演示模式"
            self.ultralytics_ok = False
            log.info("强制演示模式（APP_FORCE_DEMO=true）")
            return
        try:
            import importlib.util

            if importlib.util.find_spec("ultralytics") is None:
                raise ImportError("ultralytics 未安装")

            # 自定义模块注册（installer 不 import ultralytics，安全）
            from custom_modules.installer import is_installed, install

            if not is_installed():
                log.warning("自定义模块尚未注册，正在写入 ultralytics 源码 ...")
                install(verbose=False)

            self.ultralytics_ok = True
        except Exception as e:  # noqa: BLE001
            self.ultralytics_ok = False
            self.mode = "demo"
            self.error = f"推理引擎不可用：{e}"
            log.warning("进入演示模式：%s", self.error)
            return

        # 设备
        self.device = self._resolve_device()

        # 权重
        weights = self._find_weights()
        if weights is None:
            self.mode = "demo"
            self.error = "未找到可用的 VisDrone 训练权重（候选路径均不存在）"
            log.warning("进入演示模式：%s", self.error)
            return

        try:
            self._load(weights)
            self.mode = "real"
            log.info("真实模式：已加载 %s @ %s", weights, self.device)
        except Exception as e:  # noqa: BLE001
            self.mode = "demo"
            self.error = f"模型加载失败：{e}"
            log.warning("进入演示模式：%s", self.error)

    # ------------------------------------------------------------------ #
    def _resolve_device(self) -> str:
        dev = self.settings.DEVICE
        if dev and dev != "auto":
            return dev
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _find_weights(self) -> str | None:
        for c in self.settings.weights_candidates:
            if c and Path(c).exists():
                return c
        return None

    def _load(self, weights: str) -> None:
        from ultralytics import YOLO

        m = YOLO(weights)
        # 轻量预热，避免首帧卡顿
        try:
            import numpy as np

            m.predict(np.zeros((320, 320, 3), dtype=np.uint8),
                      device=self.device, verbose=False)
        except Exception:  # noqa: BLE001
            pass

        self.models["default"] = m
        self.loaded_variant = weights
        # 训练产物布局是 <run>/weights/best.pt，再上一级才是 run 目录。
        # 必须确认目录里真有训练产物才认领，否则外部预训练权重（如直接放在某个
        # 文件夹下的 best.pt）会把 C:/ 之类无关目录当成 run 目录，导致 /metrics 读空。
        try:
            p = Path(weights).resolve()
            cand = p.parent.parent
            if cand.is_dir() and ((cand / "results.csv").exists() or (cand / "args.yaml").exists()):
                self.run_dir = cand
            else:
                self.run_dir = None
        except Exception:  # noqa: BLE001
            self.run_dir = None

        cuda_name = None
        try:
            import torch

            if self.device != "cpu" and torch.cuda.is_available():
                cuda_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            cuda_name = None

        # 类别数优先取模型自带的（比硬编码更可信）
        n_classes = len(CLASSES_ZH)
        try:
            names = getattr(m, "names", None)
            if isinstance(names, dict) and names:
                n_classes = len(names)
        except Exception:  # noqa: BLE001
            pass

        # 参数量：直接数已加载权重的张量，避免依赖训练侧记录
        n_params = None
        n_layers = None
        try:
            n_params = sum(int(p.numel()) for p in m.model.parameters())
            n_layers = len(list(m.model.modules()))
        except Exception:  # noqa: BLE001
            pass

        self.model_info = {
            "weights": weights,
            "device": self.device,
            "cuda_name": cuda_name,
            "classes": n_classes,
            "loaded": True,
            "run": self.run_dir.name if self.run_dir else None,
            "params": n_params,
            "layers": n_layers,
        }

    # ------------------------------------------------------------------ #
    def get_model(self):
        return self.models.get("default")

    def is_real(self) -> bool:
        return self.mode == "real"

    def candidates_exist(self) -> list[str]:
        return [c for c in self.settings.weights_candidates if c and Path(c).exists()]

    # ------------------------------------------------------------------ #
    def _ensure_engine(self) -> bool:
        """确保推理引擎（ultralytics + 自定义模块）已就绪。

        切换模型前调用：引擎可能尚未初始化（例如 ``FORCE_DEMO`` 之外的
        懒加载路径），此处补齐注册而不重跑整条 ``ensure_ready`` 的权重探测。
        """
        try:
            import importlib.util

            if importlib.util.find_spec("ultralytics") is None:
                raise ImportError("ultralytics 未安装")

            from custom_modules.installer import is_installed, install

            if not is_installed():
                log.warning("自定义模块尚未注册，正在写入 ultralytics 源码 ...")
                install(verbose=False)

            self.ultralytics_ok = True
            self.device = self._resolve_device()
            return True
        except Exception as e:  # noqa: BLE001
            self.ultralytics_ok = False
            self.error = f"推理引擎不可用：{e}"
            log.warning("模型切换失败（引擎不可用）：%s", self.error)
            return False

    def switch_model(self, weight_path: str) -> dict:
        """在进程内切换当前加载的权重（单例原地替换，不重启进程）。

        返回 ``{"ok": True, ...}`` 或 ``{"ok": False, "error": ...}``。
        失败时会尽量回退到切换前的权重，保证服务不被破坏。
        """
        candidates = self.settings.weights_candidates
        if weight_path not in candidates:
            return {"ok": False, "error": "该权重不在可选列表内，拒绝加载（防止任意路径注入）"}
        if not weight_path or not Path(weight_path).exists():
            return {"ok": False, "error": f"权重文件不存在：{weight_path}"}

        if not self.ultralytics_ok and not self._ensure_engine():
            return {"ok": False, "error": self.error or "推理引擎不可用"}

        prev = self.loaded_variant
        try:
            self._load(weight_path)
            self.mode = "real"
            self.error = None
            log.info("已切换模型 → %s @ %s", weight_path, self.device)
            return {
                "ok": True,
                "loaded_variant": self.loaded_variant,
                "model_info": self.model_info,
                "candidates": candidates,
                "candidates_exist": self.candidates_exist(),
                "message": f"已切换至 {Path(weight_path).name}",
            }
        except Exception as e:  # noqa: BLE001
            self.error = f"模型切换失败：{e}"
            log.warning("模型切换失败，尝试回退到 %s：%s", prev, e)
            # 尽量回退，保持服务可用
            if prev and prev != weight_path and Path(prev).exists():
                try:
                    self._load(prev)
                    self.mode = "real"
                    self.error = None
                except Exception as e2:  # noqa: BLE001
                    self.mode = "demo"
                    self.error = f"切换失败且回退失败：{e2}"
            return {"ok": False, "error": str(e), "reverted_to": prev}
