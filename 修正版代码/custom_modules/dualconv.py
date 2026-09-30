"""
DualConv —— 双路轻量卷积（对齐论文结构）

修复要点：
  1. 原实现是 conv3x3(groups) -> conv1x1 的“串联”，等于两层线性叠加，
     中间缺非线性；且与 DualConv 论文的“并联”结构不符。
     本实现改为 **并联**：3x3 分组卷积 + 1x1 点卷积后相加。
  2. 补上 channel shuffle（分组卷积后组间信息不流通，是精度损失的主因）。
  3. 补上 BatchNorm + SiLU。

参数量对比（g=4）： 9·in·out/g + in·out = 3.25·in·out，约为标准 3x3 卷积的 36%。

yaml 用法：[-1, 1, DualConv, [<nominal_out_channels>, <k>, <stride>]]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, k: int = 3, s: int = 1, g: int = 4):
        super().__init__()
        # 分组数必须同时整除 in / out
        g = max(1, math.gcd(int(g), math.gcd(int(in_channels), int(out_channels))))
        self.groups = g
        p = k // 2

        # 支路 1：3x3 分组卷积（局部感受野，参数少）
        self.conv_g = nn.Conv2d(in_channels, out_channels, k, s, p, groups=g, bias=False)
        # 支路 2：1x1 点卷积（跨通道信息）
        self.conv_p = nn.Conv2d(in_channels, out_channels, 1, s, 0, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_g(x) + self.conv_p(x)     # 并联相加
        y = F.channel_shuffle(y, self.groups)   # 组间信息交换
        return self.act(self.bn(y))
