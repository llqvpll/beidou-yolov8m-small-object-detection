"""
一键安装：把自定义模块写入 ultralytics 源码。

用法：
    python install_custom_modules.py

幂等：重复运行不会重复插入；原文件会备份为 *.bak。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_modules.installer import install, is_installed  # noqa: E402


if __name__ == "__main__":
    if is_installed():
        print("[install] 已安装，无需重复操作。")
        sys.exit(0)
    install()
