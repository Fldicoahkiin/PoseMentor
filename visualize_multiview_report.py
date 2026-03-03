#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import ast
from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from posementor.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成四机位数据对齐报告可视化")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/multiview/multiview_manifest.csv"),
        help="多机位清单 CSV 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/visualization/multiview/multiview_report.html"),
        help="输出 HTML 路径",
    )
    return parser.parse_args()


def _parse_offsets(value: str) -> dict[str, int]:
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    return {str(k): int(v) for k, v in parsed.items()}


def _build_offset_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        session = str(row["session"])
        offsets = _parse_offsets(str(row["offsets"]))
        for cam, offset in offsets.items():
            rows.append({"session": session, "camera": cam, "offset": offset})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(f"找不到 manifest: {args.manifest}")

    df = pd.read_csv(args.manifest)
    if df.empty:
        raise RuntimeError("manifest 为空，无法生成可视化报告。")

    offset_df = _build_offset_frame(df)
    frame_fig = px.bar(
        df,
        x="session",
        y="frames",
        title="每个 Session 的可用同步帧数",
        labels={"session": "Session", "frames": "Frames"},
        template="plotly_white",
    )
    offset_fig = px.bar(
        offset_df,
        x="session",
        y="offset",
        color="camera",
        barmode="group",
        title="每个 Session 的四机位起始偏移",
        labels={"session": "Session", "offset": "Offset (frames)", "camera": "Camera"},
        template="plotly_white",
    )

    fig = make_subplots(rows=2, cols=1, subplot_titles=("同步帧数", "相机偏移"), vertical_spacing=0.14)
    for trace in frame_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in offset_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=980,
        title="PoseMentor 多机位对齐报告",
        margin=dict(l=40, r=20, t=90, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(tickangle=-30)

    out_path = args.output
    ensure_dir(out_path.parent)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[DONE] 多机位可视化报告已生成: {out_path}")


if __name__ == "__main__":
    main()
