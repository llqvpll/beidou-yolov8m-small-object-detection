"""`生成论文配图.py` 的 run 选择逻辑。

为什么值得单独测：这个脚本会**覆盖** <项目>/figures/ 里的配图。
如果挑错了 run，页面上的真图会被玩具模型的图替换掉，而且不报任何错 ——
本项目历史上已经吃过一次同款亏（"冒烟模型静默顶替真实权重"）。

实测踩到的具体坑：按 `results.csv` 的 mtime 挑"最新"，会选中
`smoke_visinit`（3 epoch，mAP50 0.073），因为它的 csv 比 `visdrone_v4` 新。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent.parent   # 修正版代码/
TOOL = HERE / "生成论文配图.py"

CSV_HEADER = [
    "epoch", "time",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]


@pytest.fixture(scope="module")
def tool():
    """按路径加载中文文件名的脚本（普通 import 语句写不出来）。"""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("gen_paper_figures", TOOL)
    assert spec and spec.loader, f"加载不了 {TOOL}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_run(root: Path, name: str, epochs: int, best_map50: float, with_weights=True):
    d = root / name
    (d / "weights").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, epochs + 1):
        v = best_map50 if i == epochs else best_map50 * i / epochs
        rows.append([i, float(i * 100), 0.5, 1.0, 1.2, v, v * 0.8, v, v * 0.5,
                     0.4, 0.9, 1.1, 5e-4])
    with open(d / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    if with_weights:
        (d / "weights" / "best.pt").write_bytes(b"x" * 1024)
    return d


# ------------------------------------------------------------ 冒烟识别

@pytest.mark.parametrize("name,smoke", [
    ("smoke_visinit", True),
    ("smoke_v4", True),
    ("gpu_check", True),
    ("_figs_val", True),
    ("debug_run", True),
    ("visdrone_v4", False),
    ("visdrone_v1", False),
    ("visdrone_v5_slim", False),
])
def test_looks_like_smoke(tool, name, smoke):
    """`visdrone_v4` 里也有下划线，别把 `_ in name` 当成判据。"""
    assert tool.looks_like_smoke(name) is smoke


# ------------------------------------------------------------ 选择逻辑

def test_pick_run_skips_smoke_dirs_and_takes_newest_real(tmp_path, tool):
    """冒烟目录的 csv 更新，也不能被选中 —— 这正是实测踩到的坑。"""
    import os
    import time

    _mk_run(tmp_path, "visdrone_v4", 117, 0.4457)
    time.sleep(0.02)
    smoke = _mk_run(tmp_path, "smoke_visinit", 3, 0.0729)
    # 明确把冒烟目录的 mtime 推到最后，模拟"它是最新的"
    os.utime(smoke / "results.csv", None)

    got, why = tool.pick_run(tmp_path, None, use_backend=False)
    assert got is not None
    assert got.name == "visdrone_v4", f"选中了 {got.name}（{why}）"
    assert "最新" in why


def test_pick_run_falls_back_to_smoke_when_nothing_else(tmp_path, tool):
    """只剩冒烟目录时仍要能选中（并如实说明），不能直接报错。"""
    _mk_run(tmp_path, "smoke_visinit", 3, 0.0729)
    got, why = tool.pick_run(tmp_path, None, use_backend=False)
    assert got is not None and got.name == "smoke_visinit"
    assert "冒烟" in why


def test_pick_run_honours_explicit_name(tmp_path, tool):
    _mk_run(tmp_path, "visdrone_v4", 117, 0.4457)
    got, why = tool.pick_run(tmp_path, "visdrone_v4", use_backend=False)
    assert got is not None and got.name == "visdrone_v4"
    assert why == "命令行指定"


def test_pick_run_rejects_missing_explicit_name(tmp_path, tool):
    got, why = tool.pick_run(tmp_path, "nope", use_backend=False)
    assert got is None
    assert "nope" in why


def test_pick_run_ignores_dirs_without_results_csv(tmp_path, tool):
    """只有 weights/ 没有 results.csv 的目录不是可用的 run。"""
    (tmp_path / "half_done" / "weights").mkdir(parents=True)
    (tmp_path / "half_done" / "weights" / "best.pt").write_bytes(b"x")
    _mk_run(tmp_path, "visdrone_v4", 5, 0.4)
    got, _ = tool.pick_run(tmp_path, None, use_backend=False)
    assert got is not None and got.name == "visdrone_v4"


# ------------------------------------------------------------ 摘要

def test_describe_reports_epochs_and_best(tmp_path, tool):
    d = _mk_run(tmp_path, "visdrone_v4", 10, 0.4457)
    s = tool.describe(d)
    assert "10 epoch" in s
    assert "0.4457" in s
    assert "MB" in s          # 有权重
    assert "无权重" not in s


def test_describe_handles_missing_weights(tmp_path, tool):
    d = _mk_run(tmp_path, "visdrone_v4", 4, 0.3, with_weights=False)
    assert "无权重" in tool.describe(d)
