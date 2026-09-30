"""
把自定义模块安装进 Ultralytics 源码（幂等、带备份、可检查）。

为什么必须改源码？
  ultralytics/nn/tasks.py 的 parse_model() 里有一个关键分支：

      if m in base_modules:                      # 只有“内置模块”走这里
          c1, c2 = ch[f], args[0]
          c2 = make_divisible(min(c2, max_channels) * width, 8)
          args = [c1, c2, *args[1:]]             # 会把通道数自动补成 (c1, c2, ...)
      ...
      else:
          c2 = ch[f]                             # 未知模块：不补通道，args 原样传入

  也就是说，未注册的自定义模块会把 yaml 里的参数“原样”丢给构造函数，
  既拿不到正确的 c1，也不会做 width / max_channels 缩放。
  所以必须把自己的类加入 base_modules（官方的自定义模块做法）。

安装动作（全部幂等，重复执行不会重复插入）：
  1. 复制 3 个模块文件到 ultralytics/nn/modules/
  2. 在模块包 __init__.py 末尾追加导入，并扩展 __all__
  3. 在 tasks.py 的 base_modules 集合中插入 3 个类名
  4. 在 tasks.py 末尾追加导入语句（让类进入 tasks 的全局命名空间）
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

MODULE_FILES = ("ema_attention.py", "dysample.py", "dualconv.py", "gsconv.py")
CLASS_NAMES = ("EMAAttention", "DySample", "DualConv", "GSConv", "VoVGSCSP")
# 这些模块 yaml 里的 "repeats" 需要被框架插入到构造函数（对应 parse_model 的 repeat_modules）
REPEAT_NAMES = ("VoVGSCSP",)
MARKER = "# === injected by install_custom_modules.py ==="

# 各模块文件对外暴露的类（用于写入 modules/__init__.py）
EXPORTS = {
    "ema_attention": ("EMAAttention",),
    "dysample": ("DySample",),
    "dualconv": ("DualConv",),
    "gsconv": ("GSConv", "GSBottleneck", "VoVGSCSP"),
}

_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE.parent          # 修正版代码/


def find_ultralytics_dir() -> Path:
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise ImportError("未找到 ultralytics，请先 `pip install ultralytics`。")
    return Path(spec.origin).resolve().parent


def is_installed() -> bool:
    try:
        ult = find_ultralytics_dir()
    except ImportError:
        return False
    tasks = ult / "nn" / "tasks.py"
    init = ult / "nn" / "modules" / "__init__.py"
    if not tasks.exists() or not init.exists():
        return False
    t = tasks.read_text(encoding="utf-8")
    i = init.read_text(encoding="utf-8")
    return all(name in t for name in CLASS_NAMES) and all(name in i for name in CLASS_NAMES)


def _backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)


def _copy_module_files(ult: Path):
    dst = ult / "nn" / "modules"
    for name in MODULE_FILES:
        src = _SRC_DIR / "custom_modules" / name
        target = dst / name
        if not src.exists():
            raise FileNotFoundError(f"缺少源文件：{src}")
        if target.exists() and target.read_text(encoding="utf-8") == src.read_text(encoding="utf-8"):
            continue
        shutil.copy2(src, target)
        print(f"  [copy] {name} -> {target}")


def _patch_modules_init(ult: Path):
    init = ult / "nn" / "modules" / "__init__.py"
    src = init.read_text(encoding="utf-8")
    if MARKER in src:
        print("  [skip] modules/__init__.py 已打过补丁")
        return
    _backup(init)
    lines = [f"\n\n{MARKER}"]
    for mod, names in EXPORTS.items():
        lines.append(f"from .{mod} import {', '.join(names)}  # noqa: E402,F401")
    names_literal = ", ".join(f'"{n}"' for n in CLASS_NAMES)
    lines.append(f"__all__ = tuple(__all__) + ({names_literal})\n")
    init.write_text(src.rstrip() + "\n".join(lines), encoding="utf-8")
    print(f"  [patch] {init.name}")


def _insert_into_frozenset(src: str, anchor: str, names, count=1):
    """在 `anchor = frozenset({ ...` 的左花括号后插入若干类名。

    注意：不同 ultralytics 版本的写法不一样，两种都要兼容——
        base_modules = frozenset(
            { ... }                                  # 8.4.x
        repeat_modules = frozenset(  # modules with 'repeat' arguments
            { ... }                                  # ← frozenset( 和 { 之间夹了注释
    """
    payload = "".join(f"\n            {n}," for n in names)
    pattern = rf"({re.escape(anchor)}\s*=\s*frozenset\(\s*(?:\#[^\n]*\s*)*\{{)"
    new_src, n = re.subn(pattern, lambda m: m.group(1) + payload, src, count=count)
    if n == 0:
        raise RuntimeError(
            f"未在 tasks.py 中定位到 {anchor} 集合，无法自动打补丁。\n"
            f"        请打开 ultralytics/nn/tasks.py，搜索 `{anchor}`，\n"
            f"        手动把 {list(names)} 加到该 frozenset 里。"
        )
    return new_src


def _patch_tasks(ult: Path):
    tasks = ult / "nn" / "tasks.py"
    src = tasks.read_text(encoding="utf-8")
    if MARKER in src:
        print("  [skip] tasks.py 已打过补丁")
        return
    _backup(tasks)

    # (a) 进 base_modules —— 让框架自动补 c1/c2 并按 width 缩放
    new_src = _insert_into_frozenset(src, "base_modules", CLASS_NAMES)
    # (b) 进 repeat_modules —— yaml 的 repeats 会被插入构造函数（VoVGSCSP 需要）
    if REPEAT_NAMES:
        new_src = _insert_into_frozenset(new_src, "repeat_modules", REPEAT_NAMES)

    # (c) 追加导入，使类进入 tasks 的全局命名空间（parse_model 用 globals()[name] 查找）
    new_src = (
        new_src.rstrip()
        + f"\n\n\n{MARKER}\n"
        + f"from ultralytics.nn.modules import {', '.join(CLASS_NAMES)}  # noqa: E402,F401\n"
    )
    tasks.write_text(new_src, encoding="utf-8")
    print(f"  [patch] {tasks.name}")


def install(verbose: bool = True) -> bool:
    ult = find_ultralytics_dir()
    if verbose:
        print(f"[install] ultralytics 路径: {ult}")
    _copy_module_files(ult)
    _patch_modules_init(ult)
    _patch_tasks(ult)
    if verbose:
        print("[install] 完成。请重启 Python 进程后再 import ultralytics。")
    return True


def main():
    try:
        install()
    except Exception as e:  # pragma: no cover
        print(f"[install][ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
