from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from lightning.pytorch.callbacks import Callback
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import torch

from posementor.utils.io import ensure_dir
from posementor.utils.joints import JOINT_NAMES
from posementor.utils.visualize import build_3d_skeleton_figure, draw_pose_2d


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float | None
    train_pos: float | None
    train_vel: float | None
    val_loss: float | None
    val_pos: float | None
    val_vel: float | None
    val_mpjpe_mm: float | None
    lr: float | None


class TrainingVisualizationCallback(Callback):
    """在训练过程中持续输出曲线，便于本地直接查看训练状态。"""

    def __init__(
        self,
        output_dir: Path,
        mean_2d: np.ndarray | None = None,
        std_2d: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.output_dir = ensure_dir(output_dir)
        self.sample_dir = ensure_dir(self.output_dir / "samples")
        self.history: list[EpochMetrics] = []
        self.mean_2d = mean_2d.astype(np.float32) if mean_2d is not None else None
        self.std_2d = std_2d.astype(np.float32) if std_2d is not None else None
        self.cached_batch: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            # Lightning 的 metric 可能是 Tensor，这里统一转 float。
            return float(value.item() if hasattr(value, "item") else value)
        except Exception:  # noqa: BLE001
            return None

    def _write_csv(self) -> None:
        out_path = self.output_dir / "training_history.csv"
        rows = [
            {
                "epoch": item.epoch,
                "train_loss": item.train_loss,
                "train_pos": item.train_pos,
                "train_vel": item.train_vel,
                "val_loss": item.val_loss,
                "val_pos": item.val_pos,
                "val_vel": item.val_vel,
                "val_mpjpe_mm": item.val_mpjpe_mm,
                "lr": item.lr,
            }
            for item in self.history
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_pos",
                    "train_vel",
                    "val_loss",
                    "val_pos",
                    "val_vel",
                    "val_mpjpe_mm",
                    "lr",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_html(self) -> None:
        if not self.history:
            return

        epochs = [item.epoch for item in self.history]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Train Loss", "Val Loss", "Val MPJPE (mm)", "Learning Rate"),
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[item.train_loss for item in self.history],
                mode="lines+markers",
                name="train/loss",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[item.val_loss for item in self.history],
                mode="lines+markers",
                name="val/loss",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[item.val_mpjpe_mm for item in self.history],
                mode="lines+markers",
                name="val/mpjpe_mm",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=[item.lr for item in self.history],
                mode="lines+markers",
                name="lr",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="PoseMentor Training Curves",
            template="plotly_white",
            height=720,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            margin=dict(l=40, r=20, t=80, b=40),
        )
        fig.write_html(str(self.output_dir / "training_curves.html"), include_plotlyjs="cdn")

    def _dump(self) -> None:
        self._write_csv()
        self._write_html()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return
        self.cached_batch = None

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0) -> None:  # type: ignore[override]
        if trainer.sanity_checking or batch_idx != 0:
            return
        if self.cached_batch is not None:
            return
        self.cached_batch = {
            "kp2d": batch["kp2d"][:1].detach().cpu(),
            "conf": batch["conf"][:1].detach().cpu(),
            "gt3d": batch["gt3d"][:1].detach().cpu(),
        }

    def _denorm_kp2d(self, kp2d: np.ndarray) -> np.ndarray:
        if self.mean_2d is None or self.std_2d is None:
            return kp2d
        mean = self.mean_2d.reshape(-1, 2)[0]
        std = self.std_2d.reshape(-1, 2)[0]
        return kp2d * std + mean

    @staticmethod
    def _fit_points_to_canvas(points_xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
        canvas_w, canvas_h = 960, 540
        margin = 48.0
        fitted = points_xy.copy().astype(np.float32)
        valid = conf[:, 0] > 0.1
        if np.sum(valid) < 3:
            return fitted

        min_xy = fitted[valid].min(axis=0)
        max_xy = fitted[valid].max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        scale_x = (canvas_w - margin * 2.0) / span[0]
        scale_y = (canvas_h - margin * 2.0) / span[1]
        scale = float(min(scale_x, scale_y))
        fitted = (fitted - min_xy) * scale + margin
        return fitted

    def _write_sample_visualization(self, trainer, pl_module) -> None:
        if self.cached_batch is None:
            return

        epoch = int(trainer.current_epoch)
        kp2d_t = self.cached_batch["kp2d"]
        conf_t = self.cached_batch["conf"]
        gt3d_t = self.cached_batch["gt3d"]

        with torch.no_grad():
            pred3d_t = pl_module(kp2d_t.to(pl_module.device)).detach().cpu()

        kp2d = kp2d_t.numpy()[0]
        conf = conf_t.numpy()[0]
        gt3d = gt3d_t.numpy()[0]
        pred3d = pred3d_t.numpy()[0]

        frame_idx = int(kp2d.shape[0] // 2)
        kp2d_frame = self._denorm_kp2d(kp2d[frame_idx])
        conf_frame = conf[frame_idx]

        kp2d_draw = np.concatenate([kp2d_frame, conf_frame], axis=-1).astype(np.float32)
        kp2d_draw[:, :2] = self._fit_points_to_canvas(kp2d_draw[:, :2], kp2d_draw[:, 2:3])
        canvas = np.full((540, 960, 3), 248, dtype=np.uint8)
        canvas = draw_pose_2d(canvas, kp2d_draw, conf_thres=0.05)
        cv2.putText(
            canvas,
            f"Train Sample 2D (epoch={epoch}, frame={frame_idx})",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            lineType=cv2.LINE_AA,
        )

        sample2d_file = self.sample_dir / f"sample_2d_epoch_{epoch:03d}.png"
        cv2.imwrite(str(sample2d_file), canvas)
        cv2.imwrite(str(self.sample_dir / "sample_2d_latest.png"), canvas)

        pred_frame = pred3d[frame_idx]
        gt_frame = gt3d[frame_idx]
        fig = build_3d_skeleton_figure(pred_frame, gt_frame)
        fig.update_layout(title=f"3D Skeleton Sample (epoch={epoch}, frame={frame_idx})")
        sample3d_file = self.sample_dir / f"sample_3d_epoch_{epoch:03d}.html"
        fig.write_html(str(sample3d_file), include_plotlyjs="cdn")
        fig.write_html(str(self.sample_dir / "sample_3d_latest.html"), include_plotlyjs="cdn")

        error = np.linalg.norm(pred_frame - gt_frame, axis=-1)
        rank = np.argsort(error)[::-1][:5]
        summary_path = self.sample_dir / "sample_summary_latest.txt"
        lines = [
            f"epoch={epoch}",
            f"frame={frame_idx}",
            "top_joint_error_mm_or_meter_mixed:",
        ]
        lines.extend([f"- {JOINT_NAMES[i]}: {float(error[i]):.4f}" for i in rank])
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        item = EpochMetrics(
            epoch=int(trainer.current_epoch),
            train_loss=self._to_float(metrics.get("train/loss")),
            train_pos=self._to_float(metrics.get("train/pos")),
            train_vel=self._to_float(metrics.get("train/vel")),
            val_loss=self._to_float(metrics.get("val/loss")),
            val_pos=self._to_float(metrics.get("val/pos")),
            val_vel=self._to_float(metrics.get("val/vel")),
            val_mpjpe_mm=self._to_float(metrics.get("val/mpjpe_mm")),
            lr=self._to_float(metrics.get("lr-AdamW")),
        )
        self.history.append(item)
        self._dump()
        self._write_sample_visualization(trainer=trainer, pl_module=pl_module)
