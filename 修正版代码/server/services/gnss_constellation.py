"""星座构成统计：回答"这一次定位里，北斗出了多少力"。

为什么值得单独做
----------------
北斗是**多星座兼容**的系统（白皮书明确写了"实现与其他卫星导航系统的兼容共用"）。
所以一台接收机同时收到 GPS / 北斗 / GLONASS / Galileo 是常态 —— 界面显示
"用了 12 颗卫星"并不能说明北斗的贡献，而"北斗主题"的项目恰恰需要说清楚这一点。

**但必须说清一个不能反推的边界**：如果模块用的是合并语句（``$GNGGA`` /
``$GNRMC``），那一帧里的"参与解算卫星数"是所有星座之和，**从语句里无法拆出北斗
占了几颗**。GSV 语句是分星座发的（``$BDGSV`` / ``$GPGSV``），所以"**可见**"能拆，
"**参与解算**"拆不了。

于是本模块只统计**可见卫星的星座构成**，并且在只有合并语句时如实标注
"参与解算的星座构成不可判定" —— 不按比例摊派，不假设。这是本项目一贯的做法：
分不清就是分不清，不要让用户以为这个数字是实测拆出来的。
"""

from __future__ import annotations

# NMEA talker ID → 星座。与前端 visual.js 的 TALKER 表保持同一口径，
# 避免同一颗卫星在北斗视图和星空图上被认成不同星座。
TALKER_ZH: dict[str, str] = {
    "BD": "北斗",
    "GB": "北斗",
    "GQ": "北斗",
    "GP": "GPS",
    "GL": "GLONASS",
    "GA": "Galileo",
    "GN": "多星座合并",
}

# 北斗的 talker 前缀集合（BD = 北斗二号/三号；GB/GQ 为兼容写法）
_BDS_PREFIXES = frozenset({"BD", "GB", "GQ"})


def build(satellites_detail: list[dict] | None,
          satellites_used: int | None,
          sentences: int = 0) -> dict:
    """统计星座构成。

    :param satellites_detail: 来自 GSV 的逐星列表，每项含 ``talker``。
    :param satellites_used: GGA 的参与解算卫星数（**不区分星座**）。

    返回结构与 :class:`server.schemas.detection.SatelliteComposition` 对应。
    """
    detail = satellites_detail or []

    visible: dict[str, int] = {}
    for s in detail:
        tk = str((s or {}).get("talker") or "").upper()
        if not tk:
            continue
        name = TALKER_ZH.get(tk, tk)
        visible[name] = visible.get(name, 0) + 1

    # talker 是否出现过分星座形式（BD/GP/GL/GA 而不是 GN）。
    # 全是 GN 的话，"北斗几颗"这个问题在当前数据下无解。
    split_known = any(
        str((s or {}).get("talker") or "").upper() not in ("", "GN")
        for s in detail
    )

    if not detail:
        # 一条 GSV 都没收到。此时连"可见"都不知道，更不用说构成。
        msg = ("未收到 GSV 卫星语句，" if sentences == 0
               else "已收到语句但其中没有 GSV，")
        msg += "无法统计星座构成。"
        return {
            "available": False,
            "visible_total": None,
            "visible_by_constellation": {},
            "beidou_visible": None,
            "beidou_share": None,
            "used_total": satellites_used,
            "used_by_constellation": None,
            "message": msg,
        }

    total = sum(visible.values())

    # 合并语句（GN）场景下，visible 里会有一个"多星座合并"桶，北斗单独为 0。
    # 这个 0 **不是"没有北斗"**，而是"拆不出来"。两者在界面上必须长得不一样：
    # 显示 0% 会被读成"这颗卫星都没搜到北斗"，是一个假陈述。
    if not split_known:
        return {
            "available": True,
            "visible_total": total,
            "visible_by_constellation": dict(sorted(visible.items(), key=lambda kv: (-kv[1], kv[0]))),
            "beidou_visible": None,          # 拆不出，不报 0
            "beidou_share": None,            # 同上：不给 0.0
            "used_total": satellites_used,
            "used_by_constellation": None,
            "message": ("卫星语句为多星座合并形式（GN），可见卫星里无法区分星座来源，"
                        "因此不给出北斗数量与占比。"),
        }

    beidou_visible = visible.get("北斗", 0)
    share = round(beidou_visible / total, 3) if total > 0 else None

    if beidou_visible == 0:
        msg = ("本次可见卫星中没有北斗卫星 —— 若确认处于有北斗信号的环境，"
               "请检查模块是否启用了北斗星座。")
    else:
        msg = None

    return {
        "available": True,
        "visible_total": total,
        # 排序保证同一份数据每次输出顺序一致，便于比对与截图留证
        "visible_by_constellation": dict(sorted(visible.items(), key=lambda kv: (-kv[1], kv[0]))),
        "beidou_visible": beidou_visible,
        "beidou_share": share,
        "used_total": satellites_used,
        # 参与解算数**不按星座拆**：合并语句里没有这个信息
        "used_by_constellation": None,
        "message": msg,
    }
