"""
Slim-Neck 模块：GSConv / GSBottleneck / VoVGSCSP

来源：论文《Slim-neck by GSConv: A lightweight-design for real-time detector architectures》
      官方实现 https://github.com/AlanLi1997/slim-neck-by-gsconv
      本文件为按论文结构整理的可直接接入 ultralytics 的版本（请按该仓库 LICENSE 使用并引用）。

为什么用它替代 DualConv：
  - 有正式论文背书，写进盲审报告比自造模块更有说服力；
  - GSConv 在 Neck 阶段"轻量化 + 保精度"，论文在 20+ 组对比实验中验证；
  - 与 C2f 同位置可替换（VoVGSCSP 是 C2f 的轻量替代），消融改动干净。

yaml 用法：
  GSConv    [-1, 1, GSConv,    [<nominal_out>, <k>, <stride>]]   # 非重复模块
  VoVGSCSP  [-1, 3, VoVGSCSP,  [<nominal_out>]]                  # 重复模块（n 由框架插入）
"""
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

__all__ = ("GSConv", "GSBottleneck", "VoVGSCSP")


class GSConv(nn.Module):
    """GSConv：一半标准卷积 + 一半深度可分离卷积，再 channel shuffle 混合。"""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        assert c_ > 0, f"GSConv 要求输出通道为偶数（当前 c2={c2}）"
        self.cv1 = Conv(c1, c_, k, s, None, g, 1, act)      # 标准卷积支路
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1, act)     # 深度可分离卷积支路

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)               # (B, c2, H, W)
        b, n, h, w = x2.size()
        y = x2.reshape(b, 2, n // 2, h, w).permute(1, 0, 2, 3, 4)   # shuffle
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """GSBottleneck：GSConv(1x1) -> GSConv(3x3) + 1x1 shortcut。"""

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False),
        )
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """VoVGSCSP：C2f 的轻量替代（单次聚合 + GS 瓶颈）。"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))
