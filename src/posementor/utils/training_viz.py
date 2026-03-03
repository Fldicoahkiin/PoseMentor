from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from posementor.utils.io import ensure_dir


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

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = ensure_dir(output_dir)
        self.history: list[EpochMetrics] = []

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

