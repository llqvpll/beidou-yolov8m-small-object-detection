"""
自定义损失函数（修正版）

相对原实现的改动：
  [P1-1] NWD 公式对齐论文：宽高项补 /4，并对 W2 取 sqrt 后再指数衰减；
         增加 eps 避免 sqrt 在零点梯度爆炸。
  [P1-2] DFLoss 去掉 in-place 的 clamp_()，避免静默修改外部张量。
  [P1-3] BboxLoss 对 target_scores_sum 做下限保护，并处理无正样本批次，
         消除除零导致的 NaN / inf。
  [兼容] BboxLoss.forward 增加 *args/**kwargs，兼容新版 ultralytics 传入的
         imgsz / stride 等额外参数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import bbox2dist


class NWDLoss(nn.Module):
    """Normalized Gaussian Wasserstein Distance loss（对齐论文形式）。

    W2^2 = (Δcx^2 + Δcy^2) + ((Δw^2 + Δh^2) / 4)
    NWD  = exp(-sqrt(W2^2) / C)
    loss = 1 - NWD

    提示：C 是“像素/格点尺度”超参，需按数据集目标尺度调参
    （ultralytics 内部 bbox 以 stride 归一化后的格点为单位，建议扫描 C ∈ {6, 8, 12, 16, 20}）。
    """

    def __init__(self, C: float = 12.0):
        super().__init__()
        self.C = float(C)

    def forward(self, pred_xywh: torch.Tensor, gt_xywh: torch.Tensor) -> torch.Tensor:
        # pred_xywh, gt_xywh: (N, 4) 以 (cx, cy, w, h) 表示
        diff = torch.sum((pred_xywh[:, :2] - gt_xywh[:, :2]) ** 2, dim=1)
        wh_diff = torch.sum((pred_xywh[:, 2:] - gt_xywh[:, 2:]) ** 2, dim=1) / 4.0
        w2 = diff + wh_diff
        return 1.0 - torch.exp(-torch.sqrt(w2 + 1e-7) / self.C)


class DFLoss(nn.Module):
    """Distribution Focal Loss（去掉了 in-place clamp）。"""

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred_dist: (N*4, reg_max)（已按 fg_mask 展平）；target: (N, 4)。

        返回 (N, 1)，与 ultralytics 官方 `BboxLoss._df_loss` 的形状/语义完全一致。
        """
        # target ∈ [0, reg_max-1]；注意不要用 clamp_ 就地改输入
        target = target.clamp(0, self.reg_max - 1 - 1e-6)
        tl = target.long()                  # 左格点，最大 = reg_max-2
        tr = tl + 1                         # 右格点，最大 = reg_max-1（索引安全）
        wl = tr - target                    # 左权重
        wr = 1.0 - wl                       # 右权重（= target - tl）
        loss = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view_as(tl) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view_as(tr) * wr
        )
        return loss.mean(-1, keepdim=True)  # (N, 4) -> (N, 1)


class BboxLoss(nn.Module):
    """NWD + DFL 组合的边界框回归损失（可直接替换 ultralytics 的 BboxLoss）。"""

    def __init__(self, reg_max: int = 16, use_dfl: bool = True, nwd_C: float = 12.0):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.dfl = DFLoss(reg_max) if use_dfl else None
        self.nwd = NWDLoss(C=nwd_C)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: float,
        fg_mask: torch.Tensor,
        *args,
        **kwargs,
    ):
        # 无正样本批次：返回与计算图相连的 0，避免 NaN（同时保住梯度图）
        if fg_mask.sum() == 0:
            zero = pred_bboxes.sum() * 0.0
            return zero, zero

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        target_scores_sum = max(float(target_scores_sum), 1.0)   # 除零保护

        # --- NWD 回归损失 ---
        pred_xywh = xyxy2xywh(pred_bboxes[fg_mask])
        gt_xywh = xyxy2xywh(target_bboxes[fg_mask])
        nwd_vals = self.nwd(pred_xywh, gt_xywh).unsqueeze(-1)
        loss_iou = (nwd_vals * weight).sum() / target_scores_sum

        # --- DFL 损失 ---
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            pred_dist_pos = pred_dist[fg_mask].view(-1, self.reg_max)   # (N*4, reg_max)
            dfl_vals = self.dfl(pred_dist_pos, target_ltrb[fg_mask])    # (N, 1)
            loss_dfl = (dfl_vals * weight).sum() / target_scores_sum
        else:
            loss_dfl = pred_bboxes.sum() * 0.0

        return loss_iou, loss_dfl
