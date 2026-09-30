"""环境自检与修复 —— 解决"pip list 里装了，import 却 ModuleNotFoundError"。

典型症状：
    ModuleNotFoundError: No module named 'cv2'
    但 `pip list | findstr opencv` 明明显示 opencv-python 已安装。

原因：site-packages 里的包目录被部分或整体删掉，只剩 `*.dist-info` 元数据。
     常见诱因：磁盘清理工具、杀软隔离、被中断的 pip 卸载（pip 卸载会先改名成 `~xxx` 再删）。

用法：
    python 环境自检.py            # 只体检，不改动任何文件
    python 环境自检.py --fix      # 对体检出的受损包执行覆盖式重装
    python 环境自检.py --fix 包名1 包名2   # 只修指定的包

修复策略：`pip install --ignore-installed --no-deps <包>`
    --ignore-installed 跳过卸载、直接覆盖写入（避开"卸载删一半"的老路）
    --no-deps          不顺手升级别的包，避免把环境越修越乱
"""
from __future__ import annotations

import glob
import importlib
import os
import subprocess
import sys

SP = os.path.join(sys.prefix, "Lib", "site-packages")
FIX = "--fix" in sys.argv
EXPLICIT = [a for a in sys.argv[1:] if not a.startswith("-")]

# 本机 cpuid 探测失败会让 polars 的 CPU 自检误报 "unknown feature flag: 'sse3'"，用官方开关跳过
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")

# 关键模块的导入自检（ultralytics 训练链路 + 常用库）
KEY_IMPORTS = [
    "numpy", "cv2", "torch", "torchvision", "filelock", "yaml", "PIL",
    "psutil", "matplotlib", "polars", "requests", "certifi", "kiwisolver",
    "mpmath", "ultralytics",
]


def _meta_name(dist_dir: str) -> str | None:
    md = os.path.join(dist_dir, "METADATA")
    if not os.path.isfile(md):
        return None
    for line in open(md, encoding="utf-8", errors="ignore"):
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1].strip()
    return None


def _record_entries(dist_dir: str) -> list[str] | None:
    rec = os.path.join(dist_dir, "RECORD")
    if not os.path.isfile(rec):
        return None
    out = []
    for line in open(rec, encoding="utf-8", errors="ignore"):
        p = line.split(",")[0].strip()
        if not p or p.endswith(".pyc") or "/.." in p or "\\.." in p:
            continue
        out.append(p)
    return out


def diagnose() -> list[tuple[str, str, str, list[str]]]:
    """返回 [(dist-info目录名, 包名, 原因, 需删除的顶层目录)]"""
    broken = []
    for dist_dir in sorted(glob.glob(os.path.join(SP, "*.dist-info"))):
        base = os.path.basename(dist_dir)
        name = _meta_name(dist_dir)
        entries = _record_entries(dist_dir)
        if name is None or entries is None:
            broken.append((base, base.split("-")[0].replace("_", "-"), "METADATA/RECORD 缺失", []))
            continue
        miss, total, tops = 0, 0, set()
        for p in entries:
            total += 1
            rel = p.replace("/", os.sep)
            if os.path.exists(os.path.join(SP, rel)):
                continue
            miss += 1
            head = rel.split(os.sep)[0]
            if head and not head.endswith(".dist-info"):
                tops.add(head)
        if miss:
            broken.append((base, name, f"{miss}/{total} 文件缺失", sorted(tops)))
    return broken


def import_check() -> list[str]:
    failed = []
    for m in KEY_IMPORTS:
        try:
            mod = importlib.import_module(m)
            ver = getattr(mod, "__version__", "")
            print(f"  [OK]   {m:<14} {ver}")
        except Exception as e:
            failed.append(m)
            print(f"  [FAIL] {m:<14} {type(e).__name__}: {e}")
    return failed


def repair(targets: list[tuple[str, str, list[str]]]) -> None:
    ok, fail = [], []
    for base, name, tops in targets:
        print(f"  修复 {name} ...", flush=True)
        for top in tops:
            p = os.path.join(SP, top)
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p, ignore_errors=True)
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--ignore-installed",
             "--no-deps", "--disable-pip-version-check", name],
            capture_output=True, text=True,
        )
        (ok if r.returncode == 0 else fail).append(name)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-2:]
            print(f"    FAIL :: {' | '.join(tail)}")
    print(f"\n  修复结果：成功 {len(ok)}，失败 {len(fail)}")
    if fail:
        print("  失败清单:", ", ".join(fail))


def main() -> None:
    print("=" * 62)
    print("运行环境自检")
    print(f"  python      : {sys.executable}")
    print(f"  site-packages: {SP}")
    print("=" * 62)

    print("\n[1/2] 关键模块导入检查")
    import_check()

    print("\n[2/2] 包完整性扫描（比对 *.dist-info/RECORD）")
    broken = diagnose()
    if not broken:
        print("  未发现受损包。")
    else:
        print(f"  发现 {len(broken)} 个受损包：")
        for base, name, why, tops in broken:
            print(f"    {name:<28} {why:<22} 顶层目录={tops or '(无)'}")

    if not FIX:
        print("\n体检模式结束。加 --fix 可执行修复（覆盖式重装，不会卸载其他包）。")
        return

    targets = []
    if EXPLICIT:
        for n in EXPLICIT:
            targets.append((f"{n}.dist-info", n, []))
    else:
        targets = [(b, n, t) for b, n, w, t in broken]

    if not targets:
        print("\n没有需要修复的包。")
        return

    print(f"\n开始修复 {len(targets)} 个包 …")
    repair(targets)

    print("\n=== 修复后复检 ===")
    import_check()
    left = diagnose()
    print(f"  剩余受损包：{len(left)} 个")
    for base, name, why, tops in left:
        print(f"    {name:<28} {why}")


if __name__ == "__main__":
    main()
