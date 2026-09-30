"""训练配图的来源与优先级。

背景（真实发生过）：页面 KPI 写「最佳 mAP50 44.6%」，而同一屏的 PR 曲线图例写
`all classes 0.442`。两边都没错，但没人说明它们是**两次不同的测量** ——
KPI 取 mAP50 峰值那一轮，best.pt 是按 fitness 选的另一轮。用户看到两个数不一致，
合理地怀疑"这些图是从别处随便找来的"。

这里的用例锁住三件事：
  1. 配图**优先**取项目内重绘图目录，为空才回落到 run 目录原图，并如实上报来源；
  2. 配图名的解析不会越出目录（防目录穿越）；
  3. 两个「最优」口径都算出来并分别上报，界面才有机会把差异讲清楚。
"""
from __future__ import annotations

import csv

import pytest

from server.services.metrics import MetricsService, artifact_path

CSV_HEADER = [
    "epoch", "time",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "lr/pg0", "lr/pg1", "lr/pg2",
]


class _Registry:
    """最小可用的权重注册表替身：MetricsService 只读这几个属性。"""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.loaded_variant = str(run_dir / "weights" / "best.pt")
        self.model_info = {"params": 17552134, "layers": 540}


def _settings(figures_dir):
    from server.config import Settings

    return Settings(FIGURES_DIR=str(figures_dir))


def _run_dir(tmp_path, rows=None):
    """造一个像样的 run 目录：results.csv + args.yaml + 训练原图。"""
    d = tmp_path / "run" / "visdrone_v4"
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "weights" / "best.pt").write_bytes(b"not-a-real-checkpoint")

    rows = rows or [
        # epoch 1：mAP50 更高 → KPI 口径的「最优」
        [1, 100.0, 0.5, 1.0, 1.2, 0.50, 0.40, 0.50, 0.10, 0.4, 0.9, 1.1, 5e-4],
        # epoch 2：mAP50-95 更高 → fitness 口径的「最优」（= best.pt 存的那一轮）
        [2, 200.0, 0.4, 0.9, 1.1, 0.48, 0.42, 0.45, 0.25, 0.35, 0.85, 1.0, 4e-4],
    ]
    with open(d / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)

    for name in ("results.png", "BoxPR_curve.png", "confusion_matrix.png"):
        (d / name).write_bytes(b"fake-png")
    return d


def _figures_dir(tmp_path, names=("matlab_map.png", "val_pr_curve.png")):
    d = tmp_path / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"fake-png")
    return d


# ------------------------------------------------------ 来源优先级

def test_figures_dir_takes_precedence_over_run_dir(tmp_path):
    """重绘图目录有内容时，展示它，而不是 run 目录里的训练原图。"""
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path)
    svc = MetricsService(_Registry(run), _settings(figs))

    snap = svc.snapshot()
    names = [a["name"] for a in snap["artifacts"]]
    assert snap["artifacts_source"] == "figures"
    assert "matlab_map.png" in names
    # run 目录里的原图不该再出现 —— 它们与 KPI 不同源，同屏会显得自相矛盾
    assert "results.png" not in names
    assert all(a["source"] == "figures" for a in snap["artifacts"])


def test_run_dir_is_used_as_fallback_when_figures_empty(tmp_path):
    """重绘图目录为空/不存在 → 回落到 run 目录，并如实标明来源为 run。"""
    run = _run_dir(tmp_path)
    svc = MetricsService(_Registry(run), _settings(tmp_path / "nope"))

    snap = svc.snapshot()
    names = [a["name"] for a in snap["artifacts"]]
    assert snap["artifacts_source"] == "run"
    assert "results.png" in names
    assert all(a["source"] == "run" for a in snap["artifacts"])


def test_declared_order_wins_and_unknown_names_are_appended(tmp_path):
    """声明顺序决定展示优先级；目录里多出来的图按文件名排在后面，不用改代码。"""
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path, names=("zzz_extra.png", "matlab_map.png", "aaa_extra.png"))
    svc = MetricsService(_Registry(run), _settings(figs))

    names = [a["name"] for a in svc.snapshot()["artifacts"]]
    assert names[0] == "matlab_map.png"          # 声明过 → 排最前
    assert names[1:] == ["aaa_extra.png", "zzz_extra.png"]


def test_non_image_files_in_figures_dir_are_ignored(tmp_path):
    """目录里混进 txt/m 之类不该被当成配图列出来。"""
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path, names=("matlab_map.png",))
    (figs / "make_figures.m").write_text("% script", encoding="utf-8")
    (figs / "README.txt").write_text("x", encoding="utf-8")
    svc = MetricsService(_Registry(run), _settings(figs))

    names = [a["name"] for a in svc.snapshot()["artifacts"]]
    assert names == ["matlab_map.png"]


# ------------------------------------------------------ 路径解析

def test_artifact_path_prefers_figures_then_run(tmp_path):
    """同名文件以重绘图目录为准；只在 run 目录里的仍能取到。"""
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path, names=("results.png", "matlab_map.png"))
    st = _settings(figs)
    reg = _Registry(run)

    assert artifact_path(reg, st, "results.png") == (figs / "results.png").resolve()
    assert artifact_path(reg, st, "matlab_map.png") == (figs / "matlab_map.png").resolve()
    # 只在 run 目录里的（BoxPR_curve.png）应回落到 run
    assert artifact_path(reg, st, "BoxPR_curve.png") == (run / "BoxPR_curve.png").resolve()


@pytest.mark.parametrize("bad", ["../secret.png", "sub/x.png", "..\\x.png", ".hidden.png"])
def test_artifact_path_rejects_traversal(tmp_path, bad):
    """只允许单层文件名，且解析后必须仍在目录内。"""
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path)
    assert artifact_path(_Registry(run), _settings(figs), bad) is None


def test_artifact_path_missing_file_returns_none(tmp_path):
    run = _run_dir(tmp_path)
    figs = _figures_dir(tmp_path)
    assert artifact_path(_Registry(run), _settings(figs), "nope.png") is None


# ------------------------------------------------------ 两个「最优」

def test_best_and_best_fitness_are_reported_separately(tmp_path):
    """mAP50 峰值与 fitness 最优不是同一轮时，两个都要报出来。

    夹具里 epoch 1 的 mAP50 更高（0.50），epoch 2 的 mAP50-95 更高（0.25），
    fitness = 0.1*mAP50 + 0.9*mAP50-95 → epoch 2 胜出。这正是本项目
    「KPI 44.6% / PR 曲线 44.2%」那个现象的成因。
    """
    run = _run_dir(tmp_path)
    svc = MetricsService(_Registry(run), _settings(tmp_path / "nope"))
    snap = svc.snapshot()

    assert snap["best"]["epoch"] == 1
    assert snap["best"]["mAP50"] == pytest.approx(0.50)

    bf = snap["best_fitness"]
    assert bf["epoch"] == 2
    assert bf["mAP50"] == pytest.approx(0.45)
    assert bf["fitness"] == pytest.approx(0.1 * 0.45 + 0.9 * 0.25)


def test_best_fitness_equals_best_when_same_epoch(tmp_path):
    """两个口径落在同一轮时也不能报错，界面据此显示"一致"。"""
    rows = [
        [1, 100.0, 0.5, 1.0, 1.2, 0.50, 0.40, 0.30, 0.20, 0.4, 0.9, 1.1, 5e-4],
        [2, 200.0, 0.4, 0.9, 1.1, 0.60, 0.50, 0.45, 0.25, 0.35, 0.85, 1.0, 4e-4],
    ]
    run = _run_dir(tmp_path, rows=rows)
    snap = MetricsService(_Registry(run), _settings(tmp_path / "nope")).snapshot()

    assert snap["best"]["epoch"] == 2
    assert snap["best_fitness"]["epoch"] == 2


def test_curves_and_best_come_from_the_same_results_csv(tmp_path):
    """曲线长度、epoch 数与 best 必须同源 —— 否则页面上的图与数字又会打架。"""
    run = _run_dir(tmp_path)
    snap = MetricsService(_Registry(run), _settings(tmp_path / "nope")).snapshot()

    assert snap["epochs"] == 2
    assert len(snap["curves"]["mAP50"]) == 2
    assert snap["curves"]["mAP50"][0] == pytest.approx(0.50)
    assert snap["curves"]["mAP50_95"][1] == pytest.approx(0.25)
    assert snap["available"] is True
