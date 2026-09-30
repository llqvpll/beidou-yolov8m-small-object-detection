"""类型安全的配置（pydantic-settings）。

所有配置项均通过环境变量注入，前缀 ``APP_``，例如 ``APP_PORT=8080``。
未设置时回退到合理默认值；缺失**必填项**会在启动时立即报错（fail-fast）。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 修正版代码/  (server/ 的上一级)
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- 服务 ----
    PROJECT_NAME: str = "北斗·改进YOLOv8m 小目标识别辅助系统"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    FORCE_DEMO: bool = False            # True=强制演示模式（不加载模型，纯 UI 演示部署用）
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "info"

    # CORS：逗号分隔的来源列表；["*"] 表示全部（开发期）。生产请显式指定前端域名。
    CORS_ORIGINS: list[str] = ["*"]

    # ---- 模型 / 推理 ----
    PRELOAD_MODEL: bool = True          # 启动时即加载权重（false=首次请求时懒加载）
    DEFAULT_VARIANT: str = "auto"       # 仅占位，权重按 candidates 顺序自动选择
    DEVICE: str = "auto"                # auto | cpu | cuda | 0 | 0,1 ...
    INFERENCE_CONCURRENCY: int = 1      # GPU 建议 1（串行化避免 OOM）；CPU 可调大
    DETECT_CONF_FLOOR: float = 0.05     # 后端一次性返回的低阈值候选框，前端再按滑块过滤

    # ---- 北斗 / GNSS 定位 ----
    # 定位源：auto | serial | file | mock
    #   auto   先试串口，再试回放文件；都没有则进入 unavailable（**不静默编假坐标**）
    #   serial 串口直读北斗/GNSS 模块的 NMEA 0183
    #   file   回放 NMEA 日志（没有硬件时做演示、或复现问题用）
    #   mock   显式使用模拟坐标（界面上会明确标成"模拟"）
    GNSS_SOURCE: str = "auto"
    GNSS_PORT: str = ""                 # 如 COM3 / /dev/ttyUSB0；留空则自动扫描
    GNSS_BAUD: int = 9600               # 多数北斗模块 9600，部分 115200
    GNSS_FILE: str = ""                 # NMEA 日志路径（GNSS_SOURCE=file 时必填）
    GNSS_REPLAY_LOOP: bool = True       # 回放是否循环
    GNSS_REPLAY_SPEED: float = 1.0      # 回放倍速（按日志内的时间戳推进）
    GNSS_STALE_S: float = 5.0           # 超过该秒数没有新定位即视为"数据过期"
    GNSS_READ_TIMEOUT: float = 1.0      # 串口单次读取超时（秒）
    GNSS_MAX_AGE_S: float = 30.0        # 超过该秒数彻底判为"无定位"

    # 模拟源的定位质量（GNSS_SOURCE=mock 时生效），用来演示界面上的不同状态：
    #   0 无定位 | 1 单点定位 | 2 差分定位 | 4 RTK 固定解 | 5 RTK 浮点解
    GNSS_MOCK_FIX_QUALITY: int = 1

    # 无硬件时的模拟坐标（GNSS_SOURCE=mock 时使用）
    BEIDOU_LAT: float = 28.169          # 默认：长沙·岳麓（湖南大学）
    BEIDOU_LON: float = 112.944
    BEIDOU_ALT: float = 50.0
    BEIDOU_SATELLITES: int = 12

    # 已废弃：真实与否改由定位源自动判定（serial/file + 有效定位解 = real）。
    # 保留字段只为兼容旧 .env，设为 True 也**不会**被采纳，启动时会打一条警告。
    BEIDOU_REAL: bool = False

    # ---- 像素 → 地面距离（目标地理坐标换算）----
    # 地面采样距离 GSD（米/像素）。0 = 未标定，此时**不产出目标地理坐标**。
    # 正解应由 飞行高度 × 像元角，或用地面控制点 标定得到。
    GEO_GSD_M_PER_PX: float = 0.0
    # 未标定时用于"示意"的 GSD。仅当 GEO_ALLOW_ESTIMATE=True 时生效，
    # 产出的坐标会带 estimated=true，前端也会标明是估算值。
    GEO_ESTIMATE_GSD: float = 0.021
    GEO_ALLOW_ESTIMATE: bool = True

    # ---- 上传限制 ----
    MAX_UPLOAD_MB: int = 50             # 单张图片
    MAX_VIDEO_MB: int = 500             # 单个视频
    MAX_TRACK_MB: int = 200             # 存储卡日志（1 Hz 的 NMEA 一天约 20MB，留足余量）

    # ---- 切片推理（小目标利器，不改模型）----
    SLICE_ENABLED: bool = True          # 是否暴露 /detect?slice=true
    SLICE_SIZE: int = 640              # 切片边长（像素）
    SLICE_OVERLAP: float = 0.2         # 切片重叠比例
    SLICE_ENGINE: str = "builtin"      # builtin（自带，零依赖）| sahi（若已安装）
    SAHI_AUTO: bool = True             # 内置切片失败时是否回退 sahi

    # ---- 人群密度监测与聚集预警 ----
    CROWD_ENABLED: bool = True
    # VisDrone 的 people 类是「密集人群」标注（一框代表多人），单个框折算成几人。
    # 默认 1.0 = 保守低估不夸大；实测后可按场景调大。
    CROWD_PEOPLE_WEIGHT: float = 1.0
    # 分级阈值，单位 人/m²。参考 Fruin 服务水平与踩踏事故复盘：>4 为危险临界。
    CROWD_WATCH: float = 1.0
    CROWD_WARN: float = 2.0
    CROWD_DANGER: float = 4.0
    CROWD_CURVE_POINTS: int = 300   # 视频密度曲线最多点数（超长视频均匀抽稀）

    # ---- 鉴权（可选，默认关闭）----
    ENABLE_AUTH: bool = False
    API_KEY: str = ""

    # ---- 输出目录（视频结果 / 任务落盘）----
    OUTPUT_DIR: Path = ROOT / "server" / "output"

    # ---- 视频输出 ----
    # 首选编码器：avc1(H.264，浏览器 <video> 可直接播放) | mp4v | XVID。
    # 后端会先试写小样回读校验，不可用则自动降级到候选表里的下一个。
    VIDEO_CODEC: str = "avc1"

    # ---- 权重搜索（默认按 candidates 顺序自动选择，见 weights_candidates）----
    WEIGHTS: str = ""                   # 显式指定权重路径，决定自动加载哪一个（最高优先级）

    # 始终进入候选列表的额外权重（与 APP_WEIGHTS 解耦）：
    # APP_WEIGHTS 只决定「一启动默认加载谁」；这里列出的路径则保证始终出现在 UI 选择器里，
    # 即使 APP_WEIGHTS 未设置。多个用 ";" 分隔；不存在的路径会被前端标为「缺失」并禁用。
    # 默认带一份「现成 MIT 标准 VisDrone 预训练权重」作为基线对照选项。
    EXTRA_WEIGHTS: str = "C:/visdrone_weights_eval/best(yolov8m-visdrone).pt"

    # 训练产物根目录，多个用 "," 或 ";" 分隔。
    # 会按 ``<根>/<run名>/weights/best.pt`` 依次探测（先按 run 名优先级，再按根目录顺序）。
    # 默认同时扫描「外部训练目录」与「项目内 runs/」，两者都兼容。
    RUNS_DIRS: str = "C:/yolo_runs/train;" + str(ROOT / "runs" / "train")

    # 正式训练 run 名（优先级从高到低）
    RUN_NAMES: str = "visdrone_v4,visdrone_v3,visdrone_v2,visdrone_v1"

    # 兜底 run 名：冒烟/自检产物，仅当正式 run 全部缺失时才使用，
    # 避免「玩具模型顶掉真实权重」这类静默错误。
    FALLBACK_RUN_NAMES: str = "smoke_v4,smoke_v4-2,gpu_check"

    # ---- 训练指标展示（/api/v1/metrics）----
    METRICS_ENABLED: bool = True        # 是否暴露训练指标接口
    METRICS_COMPARE_GLOB: str = "对比_*.txt"   # 对比汇总文件名（相对 run 目录的上一级）

    # 项目内「重绘图」目录。有内容时**优先**展示这里的图，为空才回落到 run 目录里的训练原图。
    #
    # 为什么需要这一层：run 目录里的 results.png / PR 曲线 / 混淆矩阵是 ultralytics
    # 在**训练结束时**画的，与页面 KPI 不是同一次测量 —— KPI 取「mAP50 峰值」那一轮，
    # 而 best.pt 是按 fitness（0.1*mAP50 + 0.9*mAP50-95）选的另一轮。两者同屏出现
    # 会显示成 44.6% 与 44.2%，看着像数据是拼凑的。这里放的是：
    #   - 由 MATLAB 从**当前 run 的 results.csv** 重绘的收敛曲线（与页面曲线同源）；
    #   - 用**当前权重**重跑一次验证生成的 PR/F1/P/R 曲线与混淆矩阵。
    FIGURES_DIR: str = str(ROOT / "figures")

    @staticmethod
    def _split_multi(raw: str) -> list[str]:
        """把 ``a,b;c`` 这类多值字符串切成列表（同时兼容逗号与分号）。"""
        return [p.strip() for p in re.split(r"[,;]", raw or "") if p.strip()]

    # ---- 运行时派生：权重候选路径（第一个存在者生效）----
    @property
    def weights_candidates(self) -> list[str]:
        cands: list[str] = []
        if self.WEIGHTS:
            cands.append(self.WEIGHTS)

        roots = self._split_multi(self.RUNS_DIRS)
        names = self._split_multi(self.RUN_NAMES) + self._split_multi(self.FALLBACK_RUN_NAMES)
        # run 名在外层循环：保证「正式 run 在任何根目录下的产物」都优先于「兜底 run」
        for name in names:
            for root in roots:
                cands.append(str(Path(root) / name / "weights" / "best.pt"))

        # 始终可选的额外权重（与 APP_WEIGHTS 解耦，只控制「默认加载哪一个」）。
        # 重复路径会被去重，保持原顺序。
        seen: set[str] = set(cands)
        for p in self._split_multi(self.EXTRA_WEIGHTS or ""):
            if p not in seen:
                seen.add(p)
                cands.append(p)
        return cands

    @property
    def runs_dirs(self) -> list[Path]:
        return [Path(p) for p in self._split_multi(self.RUNS_DIRS)]

    @property
    def figures_dir(self) -> Path:
        """重绘图目录（不存在也返回路径，调用方自行判断 is_dir）。"""
        raw = (self.FIGURES_DIR or "").strip()
        return Path(raw) if raw else (ROOT / "figures")

    @property
    def geo_gsd(self) -> tuple[float, bool, bool]:
        """像素→地面的换算基准，返回 ``(米/像素, 是否已标定, 是否为估算值)``。

        这是**唯一**的换算来源：后端 ``geo_map`` 与前端 ENU 散布都用它，
        避免两边各写一个常数（历史上后端约 2.94 m/px、前端写死 0.021 m/px，
        同一张图上相差约 140 倍，表格里两列数字互相矛盾）。

        未标定且不允许估算时返回 ``(0, False, False)`` —— 调用方应据此拒绝出坐标。
        """
        if self.GEO_GSD_M_PER_PX and self.GEO_GSD_M_PER_PX > 0:
            return float(self.GEO_GSD_M_PER_PX), True, False
        if self.GEO_ALLOW_ESTIMATE and self.GEO_ESTIMATE_GSD > 0:
            return float(self.GEO_ESTIMATE_GSD), False, True
        return 0.0, False, False

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def max_video_bytes(self) -> int:
        return self.MAX_VIDEO_MB * 1024 * 1024

    @property
    def max_track_bytes(self) -> int:
        return self.MAX_TRACK_MB * 1024 * 1024

    @property
    def track_dir(self) -> Path:
        """导入的存储卡日志落盘位置（与视频任务目录分开，便于单独清理）。"""
        return self.OUTPUT_DIR / "tracks"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例配置。"""
    return Settings()
