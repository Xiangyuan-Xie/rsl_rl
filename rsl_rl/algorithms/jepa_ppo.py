# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an action-conditioned self-predictive representation loss."""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import RNNModel
from rsl_rl.storage import RolloutStorage


class JEPAPPO(PPO):
    """Jointly optimize PPO and a multi-step latent dynamics objective."""

    def __init__(
        self,
        *args,
        jepa_loss_coef: float = 0.1,
        jepa_target_momentum: float = 0.996,
        jepa_prediction_times_ms: tuple[int, ...] | list[int] = (10, 20, 40, 80),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self._raw_actor, RNNModel) or self._raw_actor.structure != "split_cnn_jepa_gru_mlp":
            raise ValueError("JEPAPPO requires an RNNModel with split_cnn_jepa_gru_mlp structure.")
        if jepa_loss_coef < 0.0:
            raise ValueError("jepa_loss_coef must be non-negative.")
        if not 0.0 <= jepa_target_momentum < 1.0:
            raise ValueError("jepa_target_momentum must be in [0, 1).")
        prediction_times = tuple(int(value) for value in jepa_prediction_times_ms)
        if len(prediction_times) != len(self._raw_actor.jepa_prediction_steps) or any(
            value <= 0 for value in prediction_times
        ):
            raise ValueError("jepa_prediction_times_ms must match the actor prediction steps.")
        self.jepa_loss_coef = jepa_loss_coef
        self.jepa_target_momentum = jepa_target_momentum
        self.jepa_prediction_times_ms = prediction_times

    def _compute_auxiliary_loss(
        self, batch: RolloutStorage.Batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if batch.trajectory_actions is None:
            raise RuntimeError("JEPAPPO requires padded action trajectories from recurrent rollout storage.")
        loss, horizon_losses, latent_std = self._raw_actor.compute_jepa_loss(batch.trajectory_actions)
        metrics = {
            "jepa_loss": loss.detach(),
            "jepa_prediction_cosine": (1.0 - loss).detach(),
            "jepa_latent_std": latent_std.detach(),
        }
        for step, time_ms in zip(
            self._raw_actor.jepa_prediction_steps, self.jepa_prediction_times_ms, strict=True
        ):
            metrics[f"jepa_loss_{time_ms}ms"] = horizon_losses[step].detach()
        return self.jepa_loss_coef * loss, metrics

    def _after_optimizer_step(self) -> None:
        self._raw_actor.update_jepa_target(self.jepa_target_momentum)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        actor_state = loaded_dict.get("actor_state_dict", {})
        if not any(key.startswith("jepa_online_encoder.") for key in actor_state):
            raise ValueError("A pre-JEPA actor checkpoint cannot be resumed with JEPAPPO.")
        return super().load(loaded_dict, load_cfg, strict)
