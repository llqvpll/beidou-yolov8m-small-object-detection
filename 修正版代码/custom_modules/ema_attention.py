"""
EMA 注意力模块（轻量化实现）

设计要点：
  - 沿 H / W 两个方向做全局池化，得到坐标位置先验；
  - 1x1 卷积 + BN + SiLU 做通道融合，3x3 分组卷积做局部上下文建模；
  - Sigmoid 门控回乘到输入特征。

注意：本实现是“坐标注意力(Coordinate Attention)”思路的轻量化变体，
并非原论文 EMA(Efficient Multi-Scale Attention) 的完整三支路结构。
论文中请如实命名为“注意力引导模块 / EMA-lite”，避免“挂名”。

yaml 用法：[-1, 1, EMAAttention, [<nominal_out_channels>, <groups>]]
其中 nominal_out_channels 填“该处输入通道的原始数值”，
框架会按 width 乘子缩放，最终 c2 会自动等于输入通道 c1。
"""
import torch
import torch.nn as nn


class EMAAttention(nn.Module):
    def __init__(self, in_channels, out_channels, groups: int = 32):
        super().__init__()
        assert in_channels == out_channels, (
            f"EMAAttention 不改变通道数，但收到 in={in_channels}, out={out_channels}；"
            f"请检查 yaml 中 nominal 通道值是否与该层输入一致。"
        )
        # groups 必须整除通道数，这里做一次“向下取整”的容错
        g = min(groups, in_channels)
        while in_channels % g != 0 and g > 1:
            g //= 2
        self.groups = g

        # 方向池化后的通道融合（BN + SiLU 补上原实现缺失的归一化与非线性）
        self.pool_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        # 3x3 分组卷积 + 通道映射
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_w = x.mean(dim=2, keepdim=True)   # (B, C, 1, W)
        avg_h = x.mean(dim=3, keepdim=True)   # (B, C, H, 1)
        y = self.pool_conv(avg_h) + self.pool_conv(avg_w)  # 广播相加 -> (B, C, H, W)
        y = self.conv3(y)
        return x * y.sigmoid()                # 门控（sigmoid ∈ (0,1)，此处是可选的抑制项）
