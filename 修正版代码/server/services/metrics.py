"""训练指标服务：只读解析 YOLO 训练产物，供前端「模型 / 分析」视图展示真实数据。

数据来源（**全部只读**，绝不改写训练产物）：
  - ``<run>/results.csv``       逐 epoch 指标（ultralytics 官方列名，UTF-8）
  - ``<run>/args.yaml``         本次训练超参
  - ``<runs_root>/对比_*.txt``  多版本消融对比表（可选，自动探测）
  - ``<figures_dir>/*.png``     项目内重绘图（MATLAB 重绘曲线 + 重跑验证的 PR/混淆矩阵）

设计原则：
  - 单例 + 惰性解析 + 结果缓存（训练结束后产物不再变化，缓存即可）；
  - 任何字段缺失都返回 ``None`` 而不是抛异常，接口永不 500；
  - 曲线全量返回（VisDrone 训练仅百余 epoch，体积可忽略，前端无需再降采样）；
  - **配图优先用重绘图目录**：run 目录里的图是训练结束时画的，与页面 KPI 不同源
    （best.pt 按 fitness 选轮，KPI 按 mAP50 峰值选轮），同屏会出现 44.6% / 44.2%
    这类自相矛盾的读数。重绘图目录为空时才回落到 run 目录原图。
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from ..config import Settings

log = logging.getLogger("server.metrics")

# run 目录里可直接展示的训练原图（回退用）
RUN_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("results.png", "训练过程总览"),
    ("BoxPR_curve.png", "PR 曲线"),
    ("BoxF1_curve.png", "F1 曲线"),
    ("BoxP_curve.png", "P 曲线"),
    ("BoxR_curve.png", "R 曲线"),
    ("confusion_matrix.png", "混淆矩阵"),
    ("confusion_matrix_normalized.png", "归一化混淆矩阵"),
    ("labels.jpg", "标注分布"),
    ("val_batch0_labels.jpg", "验证集标注样例"),
)

# 重绘图目录（settings.figures_dir）里优先展示的顺序与中文说明。
# 不在此表内的图片按文件名排在后面 —— 换一批图不用改代码。
FIGURE_CAPTIONS: dict[str, str] = {
    "matlab_map.png": "mAP 收敛曲线（MATLAB 重绘）",
    "matlab_pr.png": "精确率 / 召回率（MATLAB 重绘）",
    "matlab_loss.png": "训练 / 验证损失（MATLAB 重绘）",
    "matlab_lr.png": "学习率调度（MATLAB 重绘）",
    "matlab_panel.png": "全指标面板 2×5（MATLAB 重绘）",
    "val_pr_curve.png": "PR 曲线（当前权重重跑验证）",
    "val_f1_curve.png": "F1 曲线（当前权重重跑验证）",
    "val_p_curve.png": "P 曲线（当前权重重跑验证）",
    "val_r_curve.png": "R 曲线（当前权重重跑验证）",
    "val_confusion_matrix.png": "混淆矩阵（当前权重重跑验证）",
    "val_confusion_matrix_norm.png": "归一化混淆矩阵（当前权重重跑验证）",
}

# 明确**不展示**的配图。
#
# 为什么需要它：FIGURE_CAPTIONS 只决定"优先顺序与中文说明"，
# 不在表内的图片会走 `extra` 分支照样列出来（这是为了"换一批图不用改代码"）。
# 所以只想删掉某一张时，光删 FIGURE_CAPTIONS 里的条目**没用**，必须在这里显式排除。
#
# val_batch0_pred.jpg（验证集预测样例）：按用户要求不展示。
HIDDEN_ARTIFACTS: frozenset[str] = frozenset({
    "val_batch0_pred.jpg",
})

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")

# results.csv 列名 -> 对外字段名
COLUMN_MAP: dict[str, str] = {
    "epoch": "epoch",
    "time": "time",
    "train/box_loss": "box_loss",
    "train/cls_loss": "cls_loss",
    "train/dfl_loss": "dfl_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "val/box_loss": "val_box_loss",
    "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr0",
}

# 对外返回的超参白名单（args.yaml 里其余字段对前端无意义）
ARG_KEYS = (
    "model", "data", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
    "momentum", "weight_decay", "warmup_epochs", "patience", "seed", "device",
    "workers", "cache", "pretrained", "amp", "cos_lr", "close_mosaic",
    "mosaic", "mixup", "copy_paste", "erasing", "auto_augment",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr",
    "save_period", "project", "name",
)

# 对比表里的一行：名称 + 5 个数值列（最后一列参数量带千分位逗号）
_COMPARE_ROW = re.compile(
    r"^(?P<name>.+?)\s+"
    r"(?P<mAP50>[\d.]+)\s+"
    r"(?P<mAP50_95>[\d.]+)\s+"
    r"(?P<precision>[\d.]+)\s+"
    r"(?P<recall>[\d.]+)\s+"
    r"(?P<params>[\d,]+)\s*$"
)
_BEST_EPOCH = re.compile(r"^(?P<name>.+?)[：:]\s*best mAP50 出现在第\s*(?P<epoch>\d+)\s*epoch")
_DELTA = re.compile(r"^(?P<key>mAP50|mAP50-95|参数量|训练轮数)\s+(?P<value>.+?)\s*$")


class MetricsService:
    """训练指标只读服务。"""

    def __init__(self, registry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings
        self._cache: dict | None = None

    # ------------------------------------------------------------------ #
    def snapshot(self, force: bool = False) -> dict:
        """返回已加载权重对应 run 的完整指标快照。"""
        if self._cache is not None and not force:
            return self._cache

        run_dir: Path | None = getattr(self.registry, "run_dir", None)
        weights = getattr(self.registry, "loaded_variant", None)

        if run_dir is None or not run_dir.is_dir():
            self._cache = {
                "available": False,
                "reason": "未加载真实权重，或权重不在标准训练产物目录下"
                          "（期望 <runs_root>/<run>/weights/best.pt）",
                "weights": weights,
                "run": None,
                "runs": self.list_runs(),
            }
            return self._cache

        curves, rows = self._parse_results(run_dir / "results.csv")
        best = self._pick_best(rows)
        last = rows[-1] if rows else None

        # 配图优先用重绘图目录（与页面曲线/KPI 同源），没有才回落到 run 目录原图。
        figures = self._list_figures()
        artifacts = figures or self._list_artifacts(run_dir)

        self._cache = {
            "available": bool(rows),
            "reason": None if rows else "未找到 results.csv（该目录不是 ultralytics 训练产物？）",
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "weights": weights,
            "weights_mb": self._size_mb(weights),
            "epochs": len(rows),
            "params": (getattr(self.registry, "model_info", {}) or {}).get("params"),
            "layers": (getattr(self.registry, "model_info", {}) or {}).get("layers"),
            "best": best,
            "best_fitness": self._pick_best_fitness(rows),
            "last": last,
            "curves": curves,
            "args": self._parse_args(run_dir / "args.yaml"),
            "artifacts": artifacts,
            "artifacts_source": "figures" if figures else "run",
            "artifacts_dir": str(self.settings.figures_dir) if figures else str(run_dir),
            "comparison": self._parse_comparison(run_dir),
            "runs": self.list_runs(),
        }
        return self._cache

    # ------------------------------------------------------------------ #
    def list_runs(self) -> list[dict]:
        """扫描配置的训练根目录，列出所有可用的 run（含 best mAP50）。"""
        out: list[dict] = []
        seen: set[str] = set()
        for root in self.settings.runs_dirs:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                csv_path = child / "results.csv"
                if not csv_path.exists() or str(child) in seen:
                    continue
                seen.add(str(child))
                rows = self._read_rows(csv_path)
                best = self._pick_best(rows)
                out.append({
                    "name": child.name,
                    "dir": str(child),
                    "epochs": len(rows),
                    "best_mAP50": best["mAP50"] if best else None,
                    "best_mAP50_95": best["mAP50_95"] if best else None,
                    "best_epoch": best["epoch"] if best else None,
                    "has_weights": (child / "weights" / "best.pt").exists(),
                })
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict]:
        if not csv_path.exists():
            return []
        rows: list[dict] = []
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for raw in reader:
                    row: dict = {}
                    for col, key in COLUMN_MAP.items():
                        v = raw.get(col)
                        if v is None or v == "":
                            continue
                        try:
                            row[key] = float(v)
                        except (TypeError, ValueError):
                            continue
                    if row:
                        rows.append(row)
        except Exception as e:  # noqa: BLE001
            log.warning("解析 results.csv 失败 %s: %s", csv_path, e)
            return []
        return rows

    def _parse_results(self, csv_path: Path) -> tuple[dict, list[dict]]:
        rows = self._read_rows(csv_path)
        curves: dict[str, list] = {}
        for key in set(COLUMN_MAP.values()):
            curves[key] = [r[key] for r in rows if key in r]
        return curves, rows

    @staticmethod
    def _pick_best(rows: list[dict]) -> dict | None:
        """按 mAP50 取最优 epoch（**页面上「最佳 mAP50」这个 KPI 用的口径**）。"""
        cand = [r for r in rows if "mAP50" in r]
        if not cand:
            return None
        b = max(cand, key=lambda r: r["mAP50"])
        return {
            "epoch": int(b.get("epoch", 0)),
            "mAP50": b.get("mAP50"),
            "mAP50_95": b.get("mAP50_95"),
            "precision": b.get("precision"),
            "recall": b.get("recall"),
        }

    @staticmethod
    def _pick_best_fitness(rows: list[dict]) -> dict | None:
        """按 ultralytics 的 fitness 取最优 epoch —— **这才是 best.pt 存的那一轮**。

        ``fitness = 0.1 * mAP50 + 0.9 * mAP50-95``（ultralytics 源码口径）。
        它和 ``_pick_best``（按 mAP50）常常不是同一轮，实测本项目是第 66 轮 vs 第 74 轮，
        于是 PR 曲线图例的 0.4438 与 KPI 的 0.44573 会对不上。把两个都报出来，
        界面就能把差异讲清楚，而不是让人以为数据是拼凑的。
        """
        cand = [r for r in rows if "mAP50" in r and "mAP50_95" in r]
        if not cand:
            return None
        b = max(cand, key=lambda r: 0.1 * r["mAP50"] + 0.9 * r["mAP50_95"])
        return {
            "epoch": int(b.get("epoch", 0)),
            "mAP50": b.get("mAP50"),
            "mAP50_95": b.get("mAP50_95"),
            "fitness": round(0.1 * b["mAP50"] + 0.9 * b["mAP50_95"], 6),
        }

    @staticmethod
    def _parse_args(yaml_path: Path) -> dict:
        if not yaml_path.exists():
            return {}
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("解析 args.yaml 失败 %s: %s", yaml_path, e)
            return {}
        return {k: data.get(k) for k in ARG_KEYS if k in data}

    @staticmethod
    def _list_artifacts(run_dir: Path) -> list[dict]:
        """回退用：列出 run 目录里可直接展示的训练原图。

        注意这些图是**训练结束时** ultralytics 画的，与页面 KPI 不是同一次测量
        （best.pt 按 fitness 选轮，KPI 按 mAP50 峰值选轮），所以只在重绘图目录
        为空时才用它们，并在 ``source`` 里标明来源供界面提示。
        """
        out: list[dict] = []
        for name, caption in RUN_ARTIFACTS:
            p = run_dir / name
            if p.exists():
                out.append({
                    "name": name,
                    "url": f"/api/v1/metrics/artifact/{name}",
                    "caption": caption,
                    "source": "run",
                })
        return out

    def _list_figures(self) -> list[dict]:
        """列出项目内重绘图目录里的图（优先展示）。

        顺序先按 ``FIGURE_CAPTIONS`` 声明（决定展示优先级与中文说明），
        目录里其余图片按文件名追加 —— 换一批图不用改代码。
        ``HIDDEN_ARTIFACTS`` 里的文件名一律跳过（连 extra 分支也不列）。
        """
        d = self.settings.figures_dir
        if not d.is_dir():
            return []
        try:
            present = {
                p.name for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXT
            }
        except Exception as e:  # noqa: BLE001
            log.warning("扫描重绘图目录失败 %s: %s", d, e)
            return []

        present -= HIDDEN_ARTIFACTS
        ordered = [n for n in FIGURE_CAPTIONS if n in present]
        extra = sorted(present - set(ordered))
        out: list[dict] = []
        for name in ordered + extra:
            out.append({
                "name": name,
                "url": f"/api/v1/metrics/artifact/{name}",
                "caption": FIGURE_CAPTIONS.get(name, name),
                "source": "figures",
            })
        return out

    def _parse_comparison(self, run_dir: Path) -> dict | None:
        """读取 run 上级目录下的对比汇总 txt，拆成结构化表格。"""
        pattern = self.settings.METRICS_COMPARE_GLOB or "对比_*.txt"
        found: Path | None = None
        for base in (run_dir.parent, run_dir):
            try:
                hits = sorted(base.glob(pattern))
            except Exception:  # noqa: BLE001
                hits = []
            if hits:
                found = hits[0]
                break
        if found is None:
            return None

        try:
            text = found.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("读取对比文件失败 %s: %s", found, e)
            return None

        items: list[dict] = []
        notes: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s or set(s) <= {"=", "-"}:
                continue
            m = _COMPARE_ROW.match(s)
            if m:
                items.append({
                    "name": m.group("name").strip(),
                    "mAP50": float(m.group("mAP50")),
                    "mAP50_95": float(m.group("mAP50_95")),
                    "precision": float(m.group("precision")),
                    "recall": float(m.group("recall")),
                    "params": int(m.group("params").replace(",", "")),
                })
                continue
            mb = _BEST_EPOCH.match(s)
            if mb:
                notes.append(f"{mb.group('name').strip()}：best mAP50 第 {mb.group('epoch')} epoch")
                continue
            md = _DELTA.match(s)
            if md:
                notes.append(f"{md.group('key')} {md.group('value').strip()}")
                continue
            if s.startswith("v1(") or "消融对比" in s or s.startswith("变体"):
                continue
            notes.append(s)

        return {
            "file": str(found),
            "name": found.name,
            "text": text,
            "items": items,
            "notes": notes,
        }

    @staticmethod
    def _size_mb(path: str | None) -> float | None:
        if not path:
            return None
        try:
            return round(Path(path).stat().st_size / 1024 / 1024, 1)
        except Exception:  # noqa: BLE001
            return None


def artifact_path(registry, settings: Settings, name: str) -> Path | None:
    """把配图名解析成真实文件路径（做白名单校验，防目录穿越）。

    查找顺序与 ``_list_artifacts`` 的优先级一致：**先重绘图目录，再 run 目录**。
    两处都只允许单层文件名，且解析后必须仍在各自目录内。
    """
    if "/" in name or "\\" in name or name.startswith("."):
        return None

    for base in (settings.figures_dir, getattr(registry, "run_dir", None)):
        if base is None:
            continue
        base = Path(base)
        if not base.is_dir():
            continue
        try:
            p = (base / name).resolve()
            p.relative_to(base.resolve())
        except (ValueError, OSError):
            continue
        if p.is_file():
            return p
    return None
