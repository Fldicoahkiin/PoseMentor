from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from lightning.pytorch.callbacks import Callback
from plotly.subplots import make_subplots

from posementor.utils.io import ensure_dir
from posementor.utils.joints import JOINT_NAMES, SKELETON_EDGES
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
        self.cached_batch: dict[str, Any] | None = None

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

        seq_ids = batch.get("seq_id", [])
        start_idx = batch.get("start_idx")
        video_paths = batch.get("video_path", [])
        if isinstance(start_idx, torch.Tensor):
            first_start_idx = int(start_idx[0].item())
        else:
            first_start_idx = int(start_idx[0]) if start_idx else 0

        self.cached_batch = {
            "kp2d": batch["kp2d"][:1].detach().cpu(),
            "conf": batch["conf"][:1].detach().cpu(),
            "gt3d": batch["gt3d"][:1].detach().cpu(),
            "seq_id": str(seq_ids[0]) if seq_ids else "",
            "start_idx": first_start_idx,
            "video_path": str(video_paths[0]) if video_paths else "",
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

    @staticmethod
    def _project_3d(points3d: np.ndarray) -> np.ndarray:
        yaw = np.deg2rad(35.0)
        pitch = np.deg2rad(18.0)
        rot_y = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        root = (points3d[11] + points3d[12]) * 0.5
        centered = points3d - root[None, :]
        rotated = centered @ rot_y.T @ rot_x.T
        return rotated[:, :2].astype(np.float32)

    @staticmethod
    def _normalize_2d(
        points2d: np.ndarray,
        min_xy: np.ndarray,
        max_xy: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        margin = 42.0
        span = np.maximum(max_xy - min_xy, 1.0)
        scale_x = (width - margin * 2.0) / span[0]
        scale_y = (height - margin * 2.0) / span[1]
        scale = float(min(scale_x, scale_y))
        normalized = (points2d - min_xy[None, :]) * scale + margin
        normalized[:, 1] = float(height) - normalized[:, 1]
        return normalized

    @staticmethod
    def _draw_pose_3d_panel(
        pred_points: np.ndarray,
        ref_points: np.ndarray,
        min_xy: np.ndarray,
        max_xy: np.ndarray,
        width: int,
        height: int,
        epoch: int,
        frame_idx: int,
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), 246, dtype=np.uint8)
        pred_xy = TrainingVisualizationCallback._normalize_2d(pred_points, min_xy, max_xy, width, height)
        ref_xy = TrainingVisualizationCallback._normalize_2d(ref_points, min_xy, max_xy, width, height)

        for a, b in SKELETON_EDGES:
            pa_ref = tuple(np.round(ref_xy[a]).astype(int).tolist())
            pb_ref = tuple(np.round(ref_xy[b]).astype(int).tolist())
            cv2.line(canvas, pa_ref, pb_ref, (64, 83, 199), 3, lineType=cv2.LINE_AA)

        for a, b in SKELETON_EDGES:
            pa = tuple(np.round(pred_xy[a]).astype(int).tolist())
            pb = tuple(np.round(pred_xy[b]).astype(int).tolist())
            cv2.line(canvas, pa, pb, (24, 133, 86), 3, lineType=cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"3D Pred (green) vs GT (blue) | epoch={epoch} frame={frame_idx}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (24, 24, 24),
            2,
            lineType=cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _read_video_clip(video_path: str, start_idx: int, frame_count: int) -> tuple[list[np.ndarray], float]:
        if not video_path:
            return [], 25.0
        path = Path(video_path)
        if not path.exists():
            return [], 25.0

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return [], 25.0

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(start_idx, 0))

        frames: list[np.ndarray] = []
        for _ in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames, fps

    @staticmethod
    def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
        if not frames:
            return
        height, width = frames[0].shape[:2]
        frame_fps = max(1.0, fps)

        with tempfile.TemporaryDirectory(prefix="posementor_viz_", dir=str(path.parent)) as tmp_dir:
            raw_path = Path(tmp_dir) / "raw.mp4"
            writer = cv2.VideoWriter(
                str(raw_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_fps,
                (width, height),
            )
            for frame in frames:
                writer.write(frame)
            writer.release()

            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                command = [
                    ffmpeg_bin,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(raw_path),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-profile:v",
                    "baseline",
                    "-level",
                    "3.1",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(path),
                ]
                try:
                    subprocess.run(command, check=True)
                    return
                except Exception:  # noqa: BLE001
                    pass

            shutil.copy2(raw_path, path)

    def _write_sync_videos(
        self,
        epoch: int,
        seq_id: str,
        start_idx: int,
        video_path: str,
        kp2d: np.ndarray,
        conf: np.ndarray,
        pred3d: np.ndarray,
        gt3d: np.ndarray,
    ) -> None:
        frame_count = int(kp2d.shape[0])
        source_frames, source_fps = self._read_video_clip(video_path=video_path, start_idx=start_idx, frame_count=frame_count)
        has_source_video = len(source_frames) > 0

        if has_source_video:
            height, width = source_frames[0].shape[:2]
        else:
            width, height = 960, 540
            source_frames = [np.full((height, width, 3), 244, dtype=np.uint8) for _ in range(frame_count)]

        while len(source_frames) < frame_count:
            source_frames.append(source_frames[-1].copy())
        source_frames = source_frames[:frame_count]

        source_frames_draw: list[np.ndarray] = []
        pose2d_frames: list[np.ndarray] = []

        projected_pred = np.stack([self._project_3d(pred3d[i]) for i in range(frame_count)], axis=0)
        projected_gt = np.stack([self._project_3d(gt3d[i]) for i in range(frame_count)], axis=0)
        all_projected = np.concatenate([projected_pred.reshape(-1, 2), projected_gt.reshape(-1, 2)], axis=0)
        min_xy = np.nanmin(all_projected, axis=0)
        max_xy = np.nanmax(all_projected, axis=0)
        pose3d_frames: list[np.ndarray] = []

        for frame_idx in range(frame_count):
            base_source = source_frames[frame_idx].copy()
            cv2.putText(
                base_source,
                f"{seq_id} frame={start_idx + frame_idx}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (235, 235, 235),
                2,
                lineType=cv2.LINE_AA,
            )
            source_frames_draw.append(base_source)

            kp2d_frame = self._denorm_kp2d(kp2d[frame_idx])
            conf_frame = conf[frame_idx]
            kp2d_draw = np.concatenate([kp2d_frame, conf_frame], axis=-1).astype(np.float32)
            if not has_source_video:
                kp2d_draw[:, :2] = self._fit_points_to_canvas(kp2d_draw[:, :2], kp2d_draw[:, 2:3])
            pose2d_canvas = draw_pose_2d(source_frames[frame_idx].copy(), kp2d_draw, conf_thres=0.05)
            cv2.putText(
                pose2d_canvas,
                f"2D Skeleton | epoch={epoch} frame={start_idx + frame_idx}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (24, 24, 24),
                2,
                lineType=cv2.LINE_AA,
            )
            pose2d_frames.append(pose2d_canvas)

            pose3d_canvas = self._draw_pose_3d_panel(
                pred_points=projected_pred[frame_idx],
                ref_points=projected_gt[frame_idx],
                min_xy=min_xy,
                max_xy=max_xy,
                width=width,
                height=height,
                epoch=epoch,
                frame_idx=start_idx + frame_idx,
            )
            pose3d_frames.append(pose3d_canvas)

        source_latest = self.sample_dir / "sample_video_latest.mp4"
        source_epoch = self.sample_dir / f"sample_video_epoch_{epoch:03d}.mp4"
        self._write_mp4(source_latest, source_frames_draw, source_fps)
        self._write_mp4(source_epoch, source_frames_draw, source_fps)

        pose2d_latest = self.sample_dir / "sample_2d_latest.mp4"
        pose2d_epoch = self.sample_dir / f"sample_2d_epoch_{epoch:03d}.mp4"
        self._write_mp4(pose2d_latest, pose2d_frames, source_fps)
        self._write_mp4(pose2d_epoch, pose2d_frames, source_fps)

        pose3d_latest = self.sample_dir / "sample_3d_latest.mp4"
        pose3d_epoch = self.sample_dir / f"sample_3d_epoch_{epoch:03d}.mp4"
        self._write_mp4(pose3d_latest, pose3d_frames, source_fps)
        self._write_mp4(pose3d_epoch, pose3d_frames, source_fps)

        sync_meta = {
            "seq_id": seq_id,
            "start_frame": start_idx,
            "frame_count": frame_count,
            "fps": source_fps,
            "has_source_video": has_source_video,
            "video_path": video_path,
        }
        (self.sample_dir / "sample_sync_meta_latest.json").write_text(
            json.dumps(sync_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_sample_visualization(self, trainer, pl_module) -> None:
        if self.cached_batch is None:
            return

        epoch = int(trainer.current_epoch)
        kp2d_t = self.cached_batch["kp2d"]
        conf_t = self.cached_batch["conf"]
        gt3d_t = self.cached_batch["gt3d"]
        seq_id = str(self.cached_batch.get("seq_id", ""))
        start_idx = int(self.cached_batch.get("start_idx", 0))
        video_path = str(self.cached_batch.get("video_path", ""))

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

        self._write_sync_videos(
            epoch=epoch,
            seq_id=seq_id,
            start_idx=start_idx,
            video_path=video_path,
            kp2d=kp2d,
            conf=conf,
            pred3d=pred3d,
            gt3d=gt3d,
        )

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
        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        if max_epochs > 0:
            print(f"[PROGRESS] epoch={item.epoch + 1}/{max_epochs}")
        else:
            print(f"[PROGRESS] epoch={item.epoch + 1}")
