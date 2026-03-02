from __future__ import annotations

import torch
import torch.nn as nn


class PoseLiftTransformer(nn.Module):
    """PoseFormerV2 思路的轻量版本：时间维 Transformer + 关节回归头。"""

    def __init__(
        self,
        num_joints: int = 17,
        in_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 243,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_dim = in_dim

        self.input_proj = nn.Linear(num_joints * in_dim, hidden_dim)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_joints * 3)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, kp2d: torch.Tensor) -> torch.Tensor:
        """kp2d: [B, T, J, 2] -> pred3d: [B, T, J, 3]"""
        b, t, j, c = kp2d.shape
        assert j == self.num_joints and c == self.in_dim

        x = kp2d.reshape(b, t, j * c)
        x = self.input_proj(x)
        x = x + self.time_pos_embed[:, :t, :]
        x = self.encoder(x)
        x = self.norm(x)
        out = self.head(x)
        out = out.reshape(b, t, self.num_joints, 3)
        return out


def temporal_velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """相邻帧速度约束，降低时序抖动。"""
    if pred.shape[1] < 2:
        return torch.zeros((), device=pred.device)
    pred_v = pred[:, 1:] - pred[:, :-1]
    tgt_v = target[:, 1:] - target[:, :-1]
    return (pred_v - tgt_v).abs().mean()
