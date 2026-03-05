#!/usr/bin/env python
import os
import sys


def main():
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
