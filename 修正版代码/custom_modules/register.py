"""
运行时挂接：NWD 损失 + 自定义模块安装检查。

- ensure_installed()：检查 3 个自定义模块是否已装进 ultralytics；
  没有则自动调用 installer 打补丁（需要重启进程生效）。
- use_nwd_loss()：把 ultralytics 的 BboxLoss 换成 NWD 版本。
  这一项 **不需要改源码**：v8DetectionLoss.__init__ 在调用时才从模块全局解析
  BboxLoss，因此直接替换 ultralytics.utils.loss.BboxLoss 即可生效。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from custom_loss import BboxLoss  # noqa: E402


def ensure_installed(auto: bool = True, verbose: bool = True) -> bool:
    """确认 EMAAttention / DySample / DualConv 已装进 ultralytics。"""
    from . import installer

    if installer.is_installed():
        return True

    if not auto:
        raise RuntimeError(
            "自定义模块尚未安装到 ultralytics。请先运行：\n    python install_custom_modules.py"
        )

    if verbose:
        print("[register] 检测到自定义模块未安装，正在自动安装 ...")
    installer.install(verbose=verbose)
    print(
        "\n[register] 已写入 ultralytics 源码，需要**重启 Python 进程**后生效。\n"
        "           请重新运行本脚本，第二次运行即可正常开始训练。\n"
    )
    raise SystemExit(0)


def use_nwd_loss(verbose: bool = True) -> None:
    """用 NWD 版 BboxLoss 替换检测默认的 IoU 版 BboxLoss。"""
    import ultralytics.utils.loss as loss_mod

    loss_mod.BboxLoss = BboxLoss
    if verbose:
        print("[register] 已启用 NWD BboxLoss")


def setup(verbose: bool = True) -> None:
    ensure_installed(verbose=verbose)
    use_nwd_loss(verbose)
