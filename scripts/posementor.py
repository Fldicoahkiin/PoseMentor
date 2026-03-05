#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys


def _which(executable):
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for directory in paths:
        candidate = os.path.join(directory, executable)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _reexec_with_python3():
    script_path = os.path.abspath(__file__)
    args = sys.argv[1:]
    candidates = []
    if os.name == "nt":
        candidates.extend([["py", "-3"], ["python"]])
    else:
        candidates.extend([["python3"], ["python"]])

    for prefix in candidates:
        exe = prefix[0]
        if _which(exe) is None:
            continue
        command = prefix + [script_path] + args
        try:
            raise SystemExit(subprocess.call(command))
        except OSError:
            continue
    raise SystemExit("未找到 Python 3.11+，请先安装后重试。")


def main():
    if sys.version_info < (3, 11):
        _reexec_with_python3()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_dir = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.join(project_root, "src")
    cleaned_path = []
    for item in sys.path:
        current = os.path.abspath(item or ".")
        if current == script_dir:
            continue
        cleaned_path.append(item)
    if src_dir not in cleaned_path:
        cleaned_path.insert(0, src_dir)
    sys.path[:] = cleaned_path

    from posementor.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
