#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from posementor.data.aist_dataset import (
    AISTLiftDataset,
    compute_2d_norm_stats,
    load_sequence_pairs,
)
from posementor.models.lift_net import PoseLiftTransformer, temporal_velocity_loss
from posementor.utils.io import ensure_dir
from posementor.utils.training_viz import TrainingVisualizationCallback


class LiftLightningModule(L.LightningModule):
    def __init__(
        self,
        lr: float,
        weight_decay: float,
        velocity_loss_weight: float,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = PoseLiftTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

    def forward(self, kp2d: torch.Tensor) -> torch.Tensor:
        return self.model(kp2d)

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp2d = batch["kp2d"]
        gt3d = batch["gt3d"]
        conf = batch["conf"]

        pred3d = self(kp2d)
        conf_w = torch.clamp(conf, min=0.05)
        pos_loss = ((pred3d - gt3d).abs() * conf_w).mean()
        vel_loss = temporal_velocity_loss(pred3d, gt3d)
        total_loss = pos_loss + self.hparams.velocity_loss_weight * vel_loss
        return total_loss, pos_loss, vel_loss

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        loss, pos_loss, vel_loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos", pos_loss)
        self.log("train/vel", vel_loss)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> None:
        loss, pos_loss, vel_loss = self._compute_loss(batch)
        pred3d = self(batch["kp2d"])
        gt3d = batch["gt3d"]

        mpjpe_m = torch.linalg.norm(pred3d - gt3d, dim=-1).mean()
        mpjpe_mm = torch.where(mpjpe_m < 10.0, mpjpe_m * 1000.0, mpjpe_m)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos", pos_loss)
        self.log("val/vel", vel_loss)
        self.log("val/mpjpe_mm", mpjpe_mm, prog_bar=True)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class TrainStepProgressCallback(Callback):
    def __init__(self, log_every_n_steps: int = 25) -> None:
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking:
            return
        total_batches = int(getattr(trainer, "num_training_batches", 0) or 0)
        if total_batches <= 0:
            return
        step_now = int(batch_idx) + 1
        if step_now not in {1, total_batches} and step_now % self.log_every_n_steps != 0:
            return
        epoch_now = int(getattr(trainer, "current_epoch", 0) or 0) + 1
        epoch_total = int(getattr(trainer, "max_epochs", 0) or 0)
        if epoch_total > 0:
            print(f"[TRAIN_STEP] epoch={epoch_now}/{epoch_total} step={step_now}/{total_batches}")
        else:
            print(f"[TRAIN_STEP] epoch={epoch_now} step={step_now}/{total_batches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIST++ 单视角 3D lift 训练脚本（Demo）")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--yolo2d-dir", type=Path, default=None, help="训练2D目录，默认取 config.data.yolo2d_dir")
    parser.add_argument("--gt3d-dir", type=Path, default=None, help="训练3D目录，默认取 config.data.gt3d_dir")
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help="原始视频目录，用于输出同步可视化视频（默认 data/raw/aistpp/videos）",
    )
    parser.add_argument("--artifact-dir", type=Path, default=None, help="输出目录，默认取 config.train.artifact_dir")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--epochs", type=int, default=0, help="覆盖配置中的训练轮数，0 表示使用配置值")
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=0,
        help="仅使用前 N 个训练序列（0 表示全部）",
    )
    parser.add_argument(
        "--max-val-pairs",
        type=int,
        default=0,
        help="仅使用前 N 个验证序列（0 表示全部）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="覆盖 dataloader 的 num_workers，-1 表示使用配置值",
    )
    parser.add_argument("--sample-stride", type=int, default=0, help="覆盖配置中的 sample_stride，0 表示使用配置值")
    parser.add_argument("--seq-len", type=int, default=0, help="覆盖配置中的 seq_len，0 表示使用配置值")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_onnx(model: PoseLiftTransformer, seq_len: int, out_path: Path) -> None:
    model.eval().cpu()
    dummy = torch.randn(1, seq_len, 17, 2, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["keypoints2d"],
        output_names=["joints3d"],
        dynamic_axes={
            "keypoints2d": {0: "batch", 1: "frames"},
            "joints3d": {0: "batch", 1: "frames"},
        },
        opset_version=17,
    )
    print(f"[DONE] ONNX 已导出: {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    yolo_dir = args.yolo2d_dir if args.yolo2d_dir is not None else Path(data_cfg["yolo2d_dir"])
    gt_dir = args.gt3d_dir if args.gt3d_dir is not None else Path(data_cfg["gt3d_dir"])
    videos_root_cfg = data_cfg.get("videos_root", "data/raw/aistpp/videos")
    videos_root = args.videos_root if args.videos_root is not None else Path(videos_root_cfg)

    train_pairs = load_sequence_pairs(
        yolo_dir=yolo_dir,
        gt_dir=gt_dir,
        val_ratio=float(data_cfg["val_ratio"]),
        split="train",
        videos_root=videos_root,
    )
    val_pairs = load_sequence_pairs(
        yolo_dir=yolo_dir,
        gt_dir=gt_dir,
        val_ratio=float(data_cfg["val_ratio"]),
        split="val",
        videos_root=videos_root,
    )

    if args.max_train_pairs > 0:
        train_pairs = train_pairs[: args.max_train_pairs]
    if args.max_val_pairs > 0:
        val_pairs = val_pairs[: args.max_val_pairs]

    if not train_pairs:
        raise RuntimeError(
            "训练集为空，请先执行 extract_pose_yolo11.py 或 extract_pose_aist2d.py，并完成 download_and_prepare_aist.py"
        )

    mean_2d, std_2d = compute_2d_norm_stats(train_pairs)

    seq_len = int(data_cfg["seq_len"]) if args.seq_len <= 0 else args.seq_len
    sample_stride = int(data_cfg["sample_stride"]) if args.sample_stride <= 0 else args.sample_stride

    train_ds = AISTLiftDataset(
        pairs=train_pairs,
        seq_len=seq_len,
        sample_stride=sample_stride,
        mean_2d=mean_2d,
        std_2d=std_2d,
    )
    val_ds = AISTLiftDataset(
        pairs=val_pairs,
        seq_len=seq_len,
        sample_stride=sample_stride,
        mean_2d=mean_2d,
        std_2d=std_2d,
    )

    is_macos = platform.system() == "Darwin"
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    num_workers = int(train_cfg["num_workers"]) if args.num_workers < 0 else args.num_workers
    if is_macos:
        # macOS 下多进程 DataLoader 在长序列视频训练时更容易卡在首个 batch，优先稳定性。
        num_workers = 0
    pin_memory = torch.cuda.is_available()
    batch_size = int(train_cfg["batch_size"])
    if is_macos and has_mps:
        batch_size = min(batch_size, 8)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    model_module = LiftLightningModule(
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        velocity_loss_weight=float(train_cfg["velocity_loss_weight"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        max_seq_len=seq_len,
    )

    artifact_dir = ensure_dir(args.artifact_dir if args.artifact_dir is not None else Path(train_cfg["artifact_dir"]))
    ckpt_cb = ModelCheckpoint(
        dirpath=artifact_dir,
        filename="lift-demo-{epoch:02d}-{val_mpjpe_mm:.2f}",
        monitor="val/mpjpe_mm",
        mode="min",
        save_top_k=1,
    )

    logger = CSVLogger(save_dir=str(artifact_dir / "logs"), name="lift_demo")
    viz_cb = TrainingVisualizationCallback(
        output_dir=artifact_dir / "visualizations",
        mean_2d=mean_2d,
        std_2d=std_2d,
    )
    progress_cb = TrainStepProgressCallback(log_every_n_steps=max(5, len(train_loader) // 20))

    max_epochs = int(train_cfg["epochs"]) if args.epochs <= 0 else args.epochs
    trainer_accelerator = train_cfg.get("accelerator", "auto")
    trainer_precision = train_cfg.get("precision", "16-mixed")
    if is_macos and has_mps and trainer_precision == "16-mixed":
        trainer_precision = "32-true"

    print(
        "[INFO] 训练参数:"
        f" batch_size={batch_size} num_workers={num_workers}"
        f" accelerator={trainer_accelerator} precision={trainer_precision}"
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=trainer_accelerator,
        devices=train_cfg.get("devices", "auto"),
        precision=trainer_precision,
        log_every_n_steps=10,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch"), progress_cb, viz_cb],
        logger=logger,
    )

    trainer.fit(model=model_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 统一输出一份固定名称权重，便于推理脚本读取。
    final_ckpt = artifact_dir / "lift_demo.ckpt"
    if ckpt_cb.best_model_path:
        best_ckpt = Path(ckpt_cb.best_model_path)
        state = torch.load(best_ckpt, map_location="cpu")
        torch.save(state, final_ckpt)
        print(f"[DONE] 最优权重已复制到: {final_ckpt}")
    else:
        trainer.save_checkpoint(final_ckpt)
        print(f"[WARN] 未找到最优权重，已保存最后一次训练权重: {final_ckpt}")

    norm_file = artifact_dir / "lift_demo_norm.npz"
    np.savez_compressed(norm_file, mean_2d=mean_2d, std_2d=std_2d)
    print(f"[DONE] 归一化参数保存: {norm_file}")
    print(f"[DONE] 训练曲线可视化: {artifact_dir / 'visualizations' / 'training_curves.html'}")
    print(f"[DONE] 训练样例可视化: {artifact_dir / 'visualizations' / 'samples' / 'sample_3d_latest.html'}")

    if args.export_onnx:
        export_model = model_module.model
        if ckpt_cb.best_model_path:
            state = torch.load(ckpt_cb.best_model_path, map_location="cpu")
            sd = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
            export_model.load_state_dict(sd, strict=False)
        export_onnx(export_model, seq_len, artifact_dir / "lift_demo.onnx")


if __name__ == "__main__":
    main()
