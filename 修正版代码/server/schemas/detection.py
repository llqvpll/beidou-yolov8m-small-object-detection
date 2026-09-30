from __future__ import annotations

from pydantic import BaseModel


class ClassInfo(BaseModel):
    id: int
    name_en: str
    name_zh: str
    color: str  # 十六进制颜色，如 #FF0000


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    name_en: str
    name_zh: str
    conf: float
    bbox: BoundingBox
    lon: float | None = None  # 目标经纬度（按 GSD 从像素偏移换算，见 BeidouInfo.estimated）
    lat: float | None = None


class GnssSatellite(BaseModel):
    """单颗可见卫星（来自 GSV 语句），供星空图使用。"""

    prn: str
    talker: str | None = None        # 星座：GP / BD / GL / GA ...
    elevation: float | None = None   # 高度角（度，0~90）
    azimuth: float | None = None     # 方位角（度，0~360）
    snr: float | None = None         # 信噪比 dB-Hz；None = 未跟踪（原报文为 00）


class AccuracyTierInfo(BaseModel):
    """定位精度档位。

    为什么要有这一层：界面上"实时定位"这四个字把单点解（≈9 m 量级）和 RTK 固定解
    （厘米级）说成了同一件事，而下游的坐标换算却按同一个 GSD 在推。用户看到
    "实时定位"就以为坐标可信 —— 档位就是用来打破这个默认假设的。

    ``key`` 判不出来时整体为 None，**不退化为"单点定位"**。
    """

    key: str                          # single | sbas | ppp | float | fixed
    zh: str                           # 单点定位 / 星基增强 / 精密单点定位 / ...
    level_m: float | None = None      # 该档位的**系统能力量级**（米），非本机实测
    detail: str = ""
    # 重要：level_m 是"北斗该服务的能力指标"，不是"本机实测精度"。
    # 本机实测需要静态观测比对已知点，属后续工作。
    is_measured: bool = False


class TimeBaseInfo(BaseModel):
    """北斗时间基准（授时）。

    把 NMEA 的 UTC 从"一个显示字段"升级为"一条可核对的时间基准"：
    既说明时间从哪里来，也给出它与本机时钟的偏差。

    ``delta_s`` 为正表示**本机时钟慢了**（北斗时间走在前头）。
    """

    utc: str | None = None            # 北斗 UTC 字面量
    utc_epoch: float | None = None    # UTC 秒值；无日期时为 None
    local_epoch: float | None = None  # 同一时刻本机时钟的 UTC 秒值
    delta_s: float | None = None      # utc_epoch − local_epoch，正=本机慢
    has_date: bool = False
    source_zh: str = ""
    message: str | None = None        # 建立不起来时说明原因


class SatelliteComposition(BaseModel):
    """可见卫星的星座构成。

    ``used_by_constellation`` 恒为 None：合并语句（GN）里拆不出"参与解算"的
    星座归属。可见数能拆（GSV 分星座播发），参与解算数拆不了 —— 不按比例摊派。
    """

    available: bool = False
    visible_total: int | None = None
    visible_by_constellation: dict[str, int] = {}
    beidou_visible: int | None = None
    beidou_share: float | None = None
    used_total: int | None = None
    used_by_constellation: dict[str, int] | None = None
    message: str | None = None


class BeidouInfo(BaseModel):
    """北斗/GNSS 定位帧。

    字段全部可空：**"没有定位"必须是可表达的状态**。
    历史上这里是非空 float 并回退到一组写死的演示坐标，结果是"没插模块"和
    "模块正常定位"在接口上长得一模一样 —— 正是最危险的那种静默错误。
    """

    # ---- 基础（向后兼容，但改为可空）----
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    satellites: int | None = None    # = satellites_used，保留旧名
    real: bool = False               # 是否来自真实设备且定位有效

    # ---- 数据来源 ----
    source: str = "unavailable"      # serial | file | mock | unavailable
    source_detail: str | None = None # 如 "COM3 @ 9600" / 日志文件名

    # ---- 定位质量 ----
    usable: bool = False             # fix_quality 属于 1~5 且坐标有效
    fix_quality: int | None = None
    fix_quality_zh: str | None = None
    fix_type: int | None = None
    fix_type_zh: str | None = None
    mode_zh: str | None = None       # RMC 模式指示符的中文

    # ---- 精度因子与卫星 ----
    hdop: float | None = None
    vdop: float | None = None
    pdop: float | None = None
    satellites_used: int | None = None
    satellites_visible: int | None = None
    satellites_detail: list[GnssSatellite] = []

    # ---- 运动与时间 ----
    speed_kmh: float | None = None
    course: float | None = None
    utc: str | None = None           # 定位解算时刻（UTC，ISO 形式）
    age_s: float | None = None       # 距最近一次有效定位的秒数
    stale: bool = False              # age_s 超过 GNSS_STALE_S

    # ---- 健康度（排查串口/接线问题用）----
    sentences: int = 0
    checksum_errors: int = 0
    message: str | None = None

    # ---- 像素→地理换算的元信息 ----
    gsd_m_per_px: float = 0.0        # 当前使用的地面采样距离
    gsd_calibrated: bool = False     # True=实测标定；False=未标定
    estimated: bool = False          # True=目标坐标是按估算 GSD 推出的示意值

    # ---- 精度档位 / 授时 / 星座构成（2026-09 新增，用于强化北斗主题结合）----
    # 三者都遵循同一条原则：**判不出来就是 None + 说明原因**。
    accuracy_tier: AccuracyTierInfo | None = None
    accuracy_message: str | None = None
    timebase: TimeBaseInfo | None = None
    composition: SatelliteComposition | None = None


class ModelInfo(BaseModel):
    weights: str | None = None
    device: str | None = None
    cuda_name: str | None = None
    classes: int
    loaded: bool
    run: str | None = None       # 权重所属训练产物目录名，如 visdrone_v4
    params: int | None = None    # 参数量（直接统计已加载权重）
    layers: int | None = None    # 模块数


class StatusResponse(BaseModel):
    mode: str  # real | demo
    device: str | None = None
    model: ModelInfo
    classes: list[ClassInfo]
    beidou: BeidouInfo
    engine_available: bool
    slicing_available: bool
    message: str | None = None


class CrowdLevel(BaseModel):
    """人群密度分级。"""
    key: str    # normal | watch | warn | danger
    zh: str
    alert: bool


class CrowdInfo(BaseModel):
    """单帧（图像）人群统计结果。

    ``density`` / ``level`` 仅在已标定画面面积（area_m2 或 meters_per_px）时才有值；
    未标定时只给 ``count`` 与尺度无关的 ``occupancy_ratio``，不给假密度。
    """
    count: int
    count_pedestrian: int
    count_people: int
    people_weight: float
    area_m2: float | None = None
    density: float | None = None        # 人/m²
    level: CrowdLevel | None = None
    occupancy_ratio: float = 0.0        # 人群框面积 / 画面面积
    calibrated: bool = False
    thresholds: dict | None = None
    message: str | None = None


class CrowdSeries(BaseModel):
    """视频逐帧人群汇总（含抽稀后的密度曲线）。"""
    frames: int = 0
    peak_count: int = 0
    avg_count: float = 0.0
    peak_density: float | None = None
    avg_density: float | None = None
    peak_level: CrowdLevel | None = None
    alert_frames: int = 0
    alert_ratio: float = 0.0
    calibrated: bool = False
    thresholds: dict | None = None
    series_count: list[int] = []
    series_density: list[float] = []


class DetectResponse(BaseModel):
    mode: str
    width: int
    height: int
    detections: list[Detection]
    crowd: CrowdInfo | None = None     # 人群密度分析（见 services/crowd.py）
    original: str | None = None    # 可选回显原图（data URL）
    annotated: str | None = None   # 可选服务端绘制结果（data URL）
    device: str | None = None
    engine: str | None = None      # standard | slice
    beidou: BeidouInfo | None = None
    message: str | None = None
    elapsed_ms: float | None = None
