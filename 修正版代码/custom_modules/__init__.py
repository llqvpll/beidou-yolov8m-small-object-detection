"""自定义模块包。

注意：这里刻意不在包级导入具体模块（它们依赖 torch），
以免在未安装 torch 的环境下（例如只跑 installer）导入失败。
请按需显式导入，例如：`from custom_modules.ema_attention import EMAAttention`
"""

__all__ = ["EMAAttention", "DySample", "DualConv", "GSConv", "GSBottleneck", "VoVGSCSP"]
