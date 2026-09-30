"""
DySample —— 可学习偏移的动态上采样（轻量化实现）

修复要点：
  1. 原实现是 `x_up = interpolate(x) * sigmoid(mask)`，门控恒 <1，只能衰减激活；
     本实现改为“学习采样偏移 + grid_sample”，采样位置可学习，不再压缩幅度。
  2. 原实现把 1x1 卷积放在 2x 分辨率之后（FLOPs 为低分辨率的 4 倍）；
     本实现把 pointwise 放回低分辨率上执行，理论上省 ~75% 计算量。

说明：这是 DySample 思想的轻量化实现（单组、无 pixel_unshuffle 分支）。
若需要与原论文完全对齐的多组 / 'pl' 风格，可在此基础上扩展 groups 与 offset 初始化。

yaml 用法：[-1, 1, DySample, [<nominal_out_channels>, <scale>]]
scale 默认 2；in/out 通道必须一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    def __init__(self, in_channels, out_channels, scale: int = 2):
        super().__init__()
        assert in_channels == out_channels, (
            f"DySample 不改变通道数，但收到 in={in_channels}, out={out_channels}"
        )
        self.scale = int(scale)
        s = self.scale
        # 预测每个输出像素的采样偏移，通道顺序固定为 [dx * s*s, dy * s*s]
        self.offset = nn.Conv2d(in_channels, 2 * s * s, kernel_size=1)
        # 零初始化 -> 初始等价于标准半像素对齐上采样，训练中再学习偏移
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        # 通道融合放在低分辨率上执行（关键性能优化）
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.scale

        x = self.pointwise(x)                                  # 低分辨率 1x1

        # 1) 预测偏移并重排到 (B, 2, sH, sW)
        offset = self.offset(x)                                # (B, 2*s*s, H, W)
        offset = F.pixel_shuffle(offset, s)                    # (B, 2, sH, sW)

        # 2) 构造采样网格：输出像素 (i,j) 对应输入坐标 (i+0.5)/s-0.5
        dtype, device = x.dtype, x.device
        yy, xx = torch.meshgrid(
            torch.arange(s * H, device=device, dtype=dtype),
            torch.arange(s * W, device=device, dtype=dtype),
            indexing="ij",
        )
        px = (xx + 0.5) / s - 0.5 + offset[:, 0]               # (B, sH, sW)
        py = (yy + 0.5) / s - 0.5 + offset[:, 1]

        # 3) 归一化到 [-1, 1]（align_corners=False 的映射关系）
        gx = (2.0 * px + 1.0) / W - 1.0
        gy = (2.0 * py + 1.0) / H - 1.0
        grid = torch.stack((gx, gy), dim=-1)                   # (B, sH, sW, 2)

        # 4) 采样
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
