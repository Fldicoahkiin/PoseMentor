#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from posementor.data.aist_dataset import (
    AISTLiftDataset,
    compute_2d_norm_stats,
    load_sequence_pairs,
)
from posementor.models.lift_net import PoseLiftTransformer, temporal_velocity_loss
from posementor.utils.io import ensure_dir


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIST++ 单视角 3D lift 训练脚本（Demo）")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--export-onnx", action="store_true")
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

    yolo_dir = Path(data_cfg["yolo2d_dir"])
    gt_dir = Path(data_cfg["gt3d_dir"])

    train_pairs = load_sequence_pairs(
        yolo_dir=yolo_dir,
        gt_dir=gt_dir,
        val_ratio=float(data_cfg["val_ratio"]),
        split="train",
    )
    val_pairs = load_sequence_pairs(
        yolo_dir=yolo_dir,
        gt_dir=gt_dir,
        val_ratio=float(data_cfg["val_ratio"]),
        split="val",
    )

    if not train_pairs:
        raise RuntimeError("训练集为空，请先执行 extract_pose_yolo11.py 与 download_and_prepare_aist.py")

    mean_2d, std_2d = compute_2d_norm_stats(train_pairs)

    train_ds = AISTLiftDataset(
        pairs=train_pairs,
        seq_len=int(data_cfg["seq_len"]),
        sample_stride=int(data_cfg["sample_stride"]),
        mean_2d=mean_2d,
        std_2d=std_2d,
    )
    val_ds = AISTLiftDataset(
        pairs=val_pairs,
        seq_len=int(data_cfg["seq_len"]),
        sample_stride=int(data_cfg["sample_stride"]),
        mean_2d=mean_2d,
        std_2d=std_2d,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        shuffle=False,
        pin_memory=True,
    )

    model_module = LiftLightningModule(
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        velocity_loss_weight=float(train_cfg["velocity_loss_weight"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        max_seq_len=int(data_cfg["seq_len"]),
    )

    artifact_dir = ensure_dir(Path(train_cfg["artifact_dir"]))
    ckpt_cb = ModelCheckpoint(
        dirpath=artifact_dir,
        filename="lift-demo-{epoch:02d}-{val_mpjpe_mm:.2f}",
        monitor="val/mpjpe_mm",
        mode="min",
        save_top_k=1,
    )

    logger = CSVLogger(save_dir=str(artifact_dir / "logs"), name="lift_demo")

    trainer = L.Trainer(
        max_epochs=int(train_cfg["epochs"]),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        precision=train_cfg.get("precision", "16-mixed"),
        log_every_n_steps=10,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
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

    if args.export_onnx:
        export_model = model_module.model
        if ckpt_cb.best_model_path:
            state = torch.load(ckpt_cb.best_model_path, map_location="cpu")
            sd = {k.replace("model.", "", 1): v for k, v in state["state_dict"].items()}
            export_model.load_state_dict(sd, strict=False)
        export_onnx(export_model, int(data_cfg["seq_len"]), artifact_dir / "lift_demo.onnx")


if __name__ == "__main__":
    main()
