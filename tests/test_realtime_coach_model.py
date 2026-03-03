from __future__ import annotations

import torch

from posementor.pipeline.realtime_coach import build_lift_model_from_state


def test_build_lift_model_from_state_respects_checkpoint_shapes() -> None:
    state = {
        "time_pos_embed": torch.zeros(1, 81, 256),
        "input_proj.weight": torch.zeros(256, 34),
        "input_proj.bias": torch.zeros(256),
        "head.weight": torch.zeros(51, 256),
        "head.bias": torch.zeros(51),
    }

    model, seq_len = build_lift_model_from_state(state)
    assert seq_len == 81
    assert tuple(model.time_pos_embed.shape) == (1, 81, 256)
    assert model.num_joints == 17
    assert model.in_dim == 2
