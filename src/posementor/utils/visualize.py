from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import plotly.graph_objects as go

from posementor.utils.joints import JOINT_NAMES, SKELETON_EDGES


def draw_pose_2d(
    image: np.ndarray,
    keypoints: np.ndarray,
    bad_joint_names: Iterable[str] | None = None,
    conf_thres: float = 0.15,
) -> np.ndarray:
    """在图像上绘制骨架，错误关节高亮红色，其余为绿色。"""
    bad_joint_set = set(bad_joint_names or [])
    canvas = image.copy()

    for a, b in SKELETON_EDGES:
        if keypoints[a, 2] < conf_thres or keypoints[b, 2] < conf_thres:
            continue
        pa = tuple(np.round(keypoints[a, :2]).astype(int).tolist())
        pb = tuple(np.round(keypoints[b, :2]).astype(int).tolist())
        color = (0, 220, 0)
        if JOINT_NAMES[a] in bad_joint_set or JOINT_NAMES[b] in bad_joint_set:
            color = (0, 0, 255)
        cv2.line(canvas, pa, pb, color, 2, lineType=cv2.LINE_AA)

    for idx, name in enumerate(JOINT_NAMES):
        if keypoints[idx, 2] < conf_thres:
            continue
        color = (0, 255, 0)
        if name in bad_joint_set:
            color = (0, 0, 255)
        p = tuple(np.round(keypoints[idx, :2]).astype(int).tolist())
        cv2.circle(canvas, p, 4, color, -1, lineType=cv2.LINE_AA)

    return canvas


def draw_metrics_panel(
    image: np.ndarray,
    score: float,
    mpjpe_mm: float,
    angle_error: float,
    advice: str,
) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (8, 8), (520, 140), (20, 20, 20), -1)
    cv2.putText(panel, f"Score: {score:5.1f}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    cv2.putText(
        panel,
        f"MPJPE: {mpjpe_mm:5.1f} mm | Angle: {angle_error:4.1f} deg",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
    )
    cv2.putText(panel, advice[:34], (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 1)
    cv2.putText(panel, advice[34:68], (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 120), 1)
    return panel


def build_3d_skeleton_figure(pred3d: np.ndarray, ref3d: np.ndarray | None = None) -> go.Figure:
    """返回 Plotly Figure，方便直接嵌入 Gradio Plot。"""
    fig = go.Figure()

    def _add_pose(points3d: np.ndarray, color: str, name: str) -> None:
        for a, b in SKELETON_EDGES:
            fig.add_trace(
                go.Scatter3d(
                    x=[points3d[a, 0], points3d[b, 0]],
                    y=[points3d[a, 1], points3d[b, 1]],
                    z=[points3d[a, 2], points3d[b, 2]],
                    mode="lines",
                    line={"width": 6, "color": color},
                    name=name,
                    showlegend=False,
                )
            )

    _add_pose(pred3d, "#00cc66", "user")
    if ref3d is not None:
        _add_pose(ref3d, "#ff6b6b", "template")

    fig.update_layout(
        scene={
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
            "zaxis": {"title": "Z"},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        title="3D骨骼对比（绿=用户，红=模板）",
    )
    return fig
