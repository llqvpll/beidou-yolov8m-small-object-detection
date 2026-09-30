"""定位精度档位与时间基准：把"有定位"细化为"**哪种精度**的定位"。

为什么需要这个模块
------------------
界面上原来只有一个"实时定位"的 chip。但同样是"实时定位"，单点解（水平优于 9 m
量级）和 RTK 固定解（厘米级）差了三个数量级 —— 而我们下游的坐标换算却按同一个
GSD 在推。用户看到"实时定位"就以为坐标可信，实际上它可能只到"哪个路口"这一档。

所以档位不是装饰，它是**坐标可信度的前提条件**：GSD 标定得再准，基准点本身精度
不够，最后算出来的经纬度也就那样。

判定依据（全部来自NMEA原生字段，不引入外部假设）
------------------------------------------------
* ``fix_quality``（GGA 第 6 字段）—— 定位质量，1=单点 2=差分 4=RTK固定 5=RTK浮点
* ``mode``（RMC 第 12 字段）—— N=无解 A=单点 D=差分 **R=RTK固定 F=RTK浮点**
* ``dgps_age``（GGA 第 13 字段）—— 差分龄期（秒）。龄期越大，改正数越旧，精度越差

档位判不出来时就返回 ``None``
-----------------------------
这是本项目的**第一原则**。北斗白皮书给的"水平优于 9 m / 优于 30 cm"是**系统服务
能力**，不是"我们这台设备这一刻的实际精度"。拿系统指标冒充本机实测精度，比不显示
更糟。所以：

* 没有有效定位 → 档位为 None，理由是"未取得有效定位解"
* 是差分类解但没有龄期 → 给档位，但**标注龄期未知**
* 龄期大到不能支撑该档 → **降档**，并说明降档原因
* 模拟源 → 档位为 None，理由写清"模拟源不评价精度"
"""

from __future__ import annotations

from dataclasses import dataclass

# 差分龄期（秒）超过这个值就不再认为是"实时差分"。
# 30 秒是 RTK/差分服务的常见龄期上限：星基增强（SBAS）的改正数有效期通常按
# 十几秒到几十秒计，超过之后改正数已经不反映当前的电离层/对流层状态。
DGPS_AGE_LIMIT_S = 30.0

# 档位由粗到细。数值本身无意义，只用于比较优劣，避免前端按字符串比较。
TIER_ORDER = ("none", "single", "sbas", "ppp", "float", "fixed")


@dataclass(frozen=True)
class AccuracyTier:
    """一个精度档位的判定结果。

    ``key`` 稳定可比较；``zh`` 是界面文案；``level_m`` 是**该档位的典型水平精度
    量级（米）**，用于下游判断"这个坐标能支撑什么样的决策"。

    ``level_m`` 的取值口径必须写清楚：它描述的是**该定位模式的系统能力量级**，
    不是本机实测值。本机实测需要静态观测比对已知点才能给出，属于后续工作。
    """

    key: str
    zh: str
    level_m: float | None
    detail: str


# 各档位的系统能力量级。数字来源：《新时代的中国北斗》白皮书（2022-11）与
# NMEA 定位质量的标准含义。**这里是"系统能力"，不是"本机实测精度"。**
_SINGLE = AccuracyTier(
    "single", "单点定位", 9.0,
    "北斗公开服务全球水平定位精度优于 9 m（系统能力）；单点解无外部改正",
)
_SBAS = AccuracyTier(
    "sbas", "星基增强", 1.0,
    "北斗星基增强服务面向中国及周边提供 Ⅰ 类精密进近；米级",
)
_PPP = AccuracyTier(
    "ppp", "精密单点定位", 0.3,
    "北斗精密单点定位服务水平优于 30 cm（系统能力）；收敛需数十分钟",
)
_FLOAT = AccuracyTier(
    "float", "RTK 浮点解", 0.5,
    "载波相位差分为浮点解，模糊度未固定，通常分米级",
)
_FIXED = AccuracyTier(
    "fixed", "RTK 固定解", 0.02,
    "载波相位差分为固定解；实时厘米级，定位精度最高的可用档位",
)


def _tier_by_key(key: str) -> AccuracyTier | None:
    for t in (_SINGLE, _SBAS, _PPP, _FLOAT, _FIXED):
        if t.key == key:
            return t
    return None


def classify(
    *,
    usable: bool,
    source: str,
    fix_quality: int | None,
    mode: str | None,
    dgps_age: float | None = None,
) -> tuple[str | None, str | None]:
    """判定精度档位，返回 ``(档位 key, 说明)``。

    **判不出来就返回 ``(None, 原因)``。** 说明字符串是给用户看的，必须能回答
    "为什么我这儿显示—"这个问题 —— 一个不说话的空值，用户只会当成 bug。

    优先用 GGA 的 ``fix_quality``（比 RMC 的 ``mode`` 更细），两者都有时以更保守
    （更粗）的那个为准：定位质量"看起来更好"而实际更差，是最危险的误判方向。
    """
    if source == "mock":
        return None, "当前为模拟定位源，不评价精度档位。"
    if not usable:
        return None, "未取得有效定位解，无法评价精度档位。"

    # fix_quality → 档位候选
    fq_tier: AccuracyTier | None = None
    if fix_quality == 4:
        fq_tier = _FIXED
    elif fix_quality == 5:
        fq_tier = _FLOAT
    elif fix_quality == 2:
        fq_tier = _SBAS
    elif fix_quality in (1, 3):
        fq_tier = _SINGLE

    # RMC mode → 档位候选（R/F 只有载波相位解才会出现，比 fix_quality 更可信）
    mode_tier: AccuracyTier | None = None
    m = (mode or "").strip().upper()
    if m == "R":
        mode_tier = _FIXED
    elif m == "F":
        mode_tier = _FLOAT
    elif m == "D":
        mode_tier = _SBAS
    elif m == "A":
        mode_tier = _SINGLE
    elif m in ("N", ""):
        mode_tier = None

    if fq_tier is None and mode_tier is None:
        return None, (f"定位质量字段为 {fix_quality!r}，不在已知档位内，"
                      "无法判定精度，请核对模块输出格式。")

    # 取两者中**更粗**的：宁可低估精度，不可高估
    if fq_tier is None:
        tier = mode_tier
    elif mode_tier is None:
        tier = fq_tier
    else:
        tier = fq_tier if TIER_ORDER.index(fq_tier.key) <= TIER_ORDER.index(mode_tier.key) else mode_tier

    assert tier is not None

    # 差分类解：龄期太大就降档。这条是"看起来正常、其实是错的"最典型的一种：
    # 模块仍然报 RTK 固定解，但改正数已经是几分钟前的了。
    if tier.key in ("sbas", "ppp", "float", "fixed"):
        if dgps_age is None:
            return tier.key, (f"{tier.zh}（差分龄期未知）。"
                              f"{tier.detail}")
        if dgps_age > DGPS_AGE_LIMIT_S:
            downgraded = _SINGLE
            return downgraded.key, (
                f"模块报 {tier.zh}，但差分龄期已达 {dgps_age:.0f} 秒"
                f"（超过 {DGPS_AGE_LIMIT_S:.0f} 秒）—— 改正数已过期，"
                f"按 {downgraded.zh} 计。请检查差分数据链路。"
            )
        return tier.key, f"{tier.zh}（差分龄期 {dgps_age:.0f} 秒）。{tier.detail}"

    return tier.key, tier.detail


def tier_zh(key: str | None) -> str | None:
    """档位 key → 中文名。None 原样返回 None（不给默认档位）。"""
    if key is None:
        return None
    t = _tier_by_key(key)
    return t.zh if t else None


def tier_level_m(key: str | None) -> float | None:
    """档位 key → 该档位的**典型水平精度量级（米）**。未知档位返回 None。

    调用方用它来判断"这个坐标能不能支撑某个决策"，例如
    "要不要把目标坐标标进到人行道级别"。
    """
    if key is None:
        return None
    t = _tier_by_key(key)
    return t.level_m if t else None
