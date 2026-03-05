#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Deepchecks 数据完整性扫描")
    parser.add_argument("--input", type=Path, required=True, help="首选输入 CSV")
    parser.add_argument("--fallback-input", type=Path, default=Path(""), help="备选输入 CSV")
    parser.add_argument("--output", type=Path, required=True, help="输出 HTML 报告路径")
    parser.add_argument("--min-rows", type=int, default=20, help="最小数据行数")
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="当输入缺失或行数不足时跳过 Deepchecks（返回成功）",
    )
    return parser.parse_args()


def resolve_input(primary: Path, fallback: Path) -> Path | None:
    if primary.exists():
        return primary
    if fallback and str(fallback) and fallback.exists():
        return fallback
    return None


def main() -> None:
    args = parse_args()

    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity
    except ImportError as exc:
        raise RuntimeError(
            "未安装 Deepchecks 运行依赖，请先执行 `uv sync --group dev` 或使用 uv run --with deepchecks"
        ) from exc

    input_path = resolve_input(args.input, args.fallback_input)
    if input_path is None:
        if args.allow_skip:
            print(f"[SKIP] 找不到可用输入文件: {args.input} / {args.fallback_input}")
            return
        raise FileNotFoundError(f"找不到可用输入文件: {args.input} / {args.fallback_input}")

    df = pd.read_csv(input_path)
    if len(df) < args.min_rows:
        if args.allow_skip:
            print(f"[SKIP] 数据行数不足，跳过 Deepchecks: {len(df)} < {args.min_rows}")
            return
        raise ValueError(f"数据行数过少: {len(df)} < {args.min_rows}")

    dataset = Dataset(df)
    suite = data_integrity()
    result = suite.run(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "save_as_html"):
        result.save_as_html(str(args.output))

    failed_count = 0
    if hasattr(result, "get_not_passed_checks"):
        failed_checks = result.get_not_passed_checks()
        failed_count = len(failed_checks)
    print(
        "[DONE] Deepchecks 扫描完成:"
        f" input={input_path}"
        f" rows={len(df)}"
        f" failed_checks={failed_count}"
        f" report={args.output}"
    )
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
