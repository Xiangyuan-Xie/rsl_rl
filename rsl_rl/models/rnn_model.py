# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Literal

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class RNNModel(MLPModel):
    """RNN-based neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a distribution module is
    used to sample the outputs.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        structure: Literal["gru_mlp", "mlp_gru", "split_cnn_gru_mlp", "split_cnn_jepa_gru_mlp"] = "gru_mlp",
        recurrent_obs_range: tuple[int, int] | list[int] | None = None,
        cnn_obs_range: tuple[int, int] | list[int] | None = None,
        cnn_obs_shape: tuple[int, int] | list[int] | None = None,
        cnn_channels: tuple[int, ...] | list[int] | None = None,
        cnn_kernel_size: int | None = None,
        cnn_output_dim: int | None = None,
        shortcut_obs_range: tuple[int, int] | list[int] | None = None,
        jepa_state_obs_range: tuple[int, int] | list[int] | None = None,
        jepa_previous_action_obs_range: tuple[int, int] | list[int] | None = None,
        jepa_latent_dim: int = 32,
        jepa_prediction_steps: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            rnn_type: Type of RNN to use ("lstm" or "gru").
            rnn_hidden_dim: Dimension of the RNN hidden state.
            rnn_num_layers: Number of RNN layers.
            structure: Network ordering. ``"gru_mlp"`` keeps the upstream ``obs -> RNN -> MLP`` behavior;
                ``"mlp_gru"`` uses ``obs -> encoder MLP -> RNN -> linear head``; ``"split_cnn_gru_mlp"``
                fuses recurrent state, temporal trajectory features, and a direct state shortcut.
            recurrent_obs_range: Half-open observation range passed through the GRU by ``"split_cnn_gru_mlp"``.
            cnn_obs_range: Half-open trajectory range processed by ``"split_cnn_gru_mlp"``.
            cnn_obs_shape: Trajectory shape as ``(time, channels)`` before the temporal convolutions.
            cnn_channels: Output channel count for each valid Conv1d layer.
            cnn_kernel_size: Shared kernel size of the valid Conv1d layers.
            cnn_output_dim: Dimension of the linear projection after the flattened convolutions.
            shortcut_obs_range: Half-open direct-feedback range concatenated at the fusion layer.
            jepa_state_obs_range: Half-open physical-state range encoded by the online and target encoders.
            jepa_previous_action_obs_range: Half-open previous-action range concatenated with the online latent.
            jepa_latent_dim: Dimension of the JEPA physical-state latent.
            jepa_prediction_steps: Positive rollout step offsets used by the latent predictor.
        """
        split_structures = {"split_cnn_gru_mlp", "split_cnn_jepa_gru_mlp"}
        if structure not in {"gru_mlp", "mlp_gru", *split_structures}:
            raise ValueError(f"Unsupported structure: {structure}")
        if structure in split_structures:
            if rnn_type.lower() != "gru":
                raise ValueError("Split CNN recurrent structures only support GRU modules.")
            recurrent_obs_range = self._validate_obs_range("recurrent_obs_range", recurrent_obs_range)
            cnn_obs_range = self._validate_obs_range("cnn_obs_range", cnn_obs_range)
            shortcut_obs_range = self._validate_obs_range("shortcut_obs_range", shortcut_obs_range)
            if cnn_obs_shape is None or len(cnn_obs_shape) != 2:
                raise ValueError("cnn_obs_shape must contain (time, channels) for split_cnn_gru_mlp.")
            cnn_obs_shape = tuple(int(value) for value in cnn_obs_shape)
            if cnn_obs_shape[0] <= 0 or cnn_obs_shape[1] <= 0:
                raise ValueError(f"cnn_obs_shape dimensions must be positive, got {cnn_obs_shape}.")
            if cnn_obs_range[1] - cnn_obs_range[0] != cnn_obs_shape[0] * cnn_obs_shape[1]:
                raise ValueError("cnn_obs_range length must equal the product of cnn_obs_shape.")
            if cnn_channels is None or not cnn_channels or any(int(value) <= 0 for value in cnn_channels):
                raise ValueError("cnn_channels must contain positive channel counts.")
            cnn_channels = tuple(int(value) for value in cnn_channels)
            if cnn_kernel_size is None or cnn_kernel_size <= 0:
                raise ValueError("cnn_kernel_size must be positive.")
            cnn_length = cnn_obs_shape[0] - len(cnn_channels) * (cnn_kernel_size - 1)
            if cnn_length <= 0:
                raise ValueError("CNN kernel stack leaves no temporal output samples.")
            if cnn_output_dim is None or cnn_output_dim <= 0:
                raise ValueError("cnn_output_dim must be positive.")
            if structure == "split_cnn_jepa_gru_mlp":
                jepa_state_obs_range = self._validate_obs_range("jepa_state_obs_range", jepa_state_obs_range)
                jepa_previous_action_obs_range = self._validate_obs_range(
                    "jepa_previous_action_obs_range", jepa_previous_action_obs_range
                )
                if jepa_latent_dim <= 0:
                    raise ValueError("jepa_latent_dim must be positive.")
                if not jepa_prediction_steps or any(int(step) <= 0 for step in jepa_prediction_steps):
                    raise ValueError("jepa_prediction_steps must contain positive step offsets.")
                jepa_prediction_steps = tuple(sorted({int(step) for step in jepa_prediction_steps}))
                if jepa_previous_action_obs_range[1] - jepa_previous_action_obs_range[0] != output_dim:
                    raise ValueError("The previous-action observation range must match the policy action dimension.")
        elif any(
            value is not None
            for value in (
                recurrent_obs_range,
                cnn_obs_range,
                cnn_obs_shape,
                cnn_channels,
                cnn_kernel_size,
                cnn_output_dim,
                shortcut_obs_range,
                jepa_state_obs_range,
                jepa_previous_action_obs_range,
                jepa_prediction_steps,
            )
        ):
            raise ValueError("Observation branch settings are only valid with a split network structure.")
        elif jepa_latent_dim != 32:
            raise ValueError("jepa_latent_dim is only valid with split_cnn_jepa_gru_mlp.")
        self.structure = structure
        self.recurrent_obs_range = recurrent_obs_range
        self.cnn_obs_range = cnn_obs_range
        self.cnn_obs_shape = cnn_obs_shape
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_output_dim = cnn_output_dim
        self.shortcut_obs_range = shortcut_obs_range
        self.jepa_state_obs_range = jepa_state_obs_range
        self.jepa_previous_action_obs_range = jepa_previous_action_obs_range
        self.jepa_latent_dim = jepa_latent_dim
        self.jepa_prediction_steps = jepa_prediction_steps
        if structure in split_structures:
            assert shortcut_obs_range is not None
            assert cnn_output_dim is not None
            self.latent_dim = rnn_hidden_dim + cnn_output_dim + shortcut_obs_range[1] - shortcut_obs_range[0]
            if structure == "split_cnn_jepa_gru_mlp":
                self.latent_dim += jepa_latent_dim
        else:
            self.latent_dim = rnn_hidden_dim

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.output_dim = output_dim

        if self.structure in split_structures:
            assert recurrent_obs_range is not None
            assert cnn_obs_range is not None
            assert shortcut_obs_range is not None
            ranges = [recurrent_obs_range, cnn_obs_range, shortcut_obs_range]
            if self.structure == "split_cnn_jepa_gru_mlp":
                assert jepa_state_obs_range is not None
                assert jepa_previous_action_obs_range is not None
                ranges.extend((jepa_state_obs_range, jepa_previous_action_obs_range))
            if any(obs_range[1] > self.obs_dim for obs_range in ranges):
                raise ValueError(
                    "split_cnn_gru_mlp observation ranges must fit within the concatenated observation dimension "
                    f"{self.obs_dim}, got recurrent={recurrent_obs_range}, cnn={cnn_obs_range}, "
                    f"and shortcut={shortcut_obs_range}."
                )

        # RNN
        if self.structure == "gru_mlp":
            self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        elif self.structure == "mlp_gru":
            mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
            self.encoder = MLP(self.obs_dim, rnn_hidden_dim, hidden_dims, activation)
            self.rnn = RNN(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
            if isinstance(mlp_output_dim, int):
                self.mlp = nn.Sequential(nn.Linear(self._get_latent_dim(), mlp_output_dim))
            else:
                total_output_dim = 1
                for dim in mlp_output_dim:
                    total_output_dim *= dim
                self.mlp = nn.Sequential(
                    nn.Linear(self._get_latent_dim(), total_output_dim),
                    nn.Unflatten(dim=-1, unflattened_size=mlp_output_dim),
                )
            if self.distribution is not None:
                self.distribution.init_mlp_weights(self.mlp)
        else:
            assert recurrent_obs_range is not None
            rnn_input_dim = recurrent_obs_range[1] - recurrent_obs_range[0]
            if self.structure == "split_cnn_jepa_gru_mlp":
                assert jepa_previous_action_obs_range is not None
                rnn_input_dim = jepa_latent_dim + jepa_previous_action_obs_range[1] - jepa_previous_action_obs_range[0]
            self.rnn = RNN(
                rnn_input_dim,
                rnn_hidden_dim,
                rnn_num_layers,
                rnn_type,
            )
            if self.structure in split_structures:
                assert cnn_obs_shape is not None
                assert cnn_channels is not None
                assert cnn_kernel_size is not None
                assert cnn_output_dim is not None
                layers: list[nn.Module] = []
                input_channels = cnn_obs_shape[1]
                for index, output_channels in enumerate(cnn_channels):
                    layers.append(nn.Conv1d(input_channels, output_channels, cnn_kernel_size))
                    if index < len(cnn_channels) - 1:
                        layers.append(copy.deepcopy(resolve_nn_activation(activation)))
                    input_channels = output_channels
                final_length = cnn_obs_shape[0] - len(cnn_channels) * (cnn_kernel_size - 1)
                layers.extend((nn.Flatten(), nn.Linear(cnn_channels[-1] * final_length, cnn_output_dim)))
                self.cnn_encoder = nn.Sequential(*layers)
            if self.structure == "split_cnn_jepa_gru_mlp":
                assert jepa_state_obs_range is not None
                state_dim = jepa_state_obs_range[1] - jepa_state_obs_range[0]
                self.jepa_online_encoder = nn.Sequential(
                    nn.Linear(state_dim, jepa_latent_dim),
                    copy.deepcopy(resolve_nn_activation(activation)),
                    nn.Linear(jepa_latent_dim, jepa_latent_dim),
                    nn.LayerNorm(jepa_latent_dim),
                )
                self.jepa_target_encoder = copy.deepcopy(self.jepa_online_encoder)
                self.jepa_target_encoder.requires_grad_(False)
                self.jepa_context_projection = nn.Linear(rnn_hidden_dim, jepa_latent_dim)
                self.jepa_transition_predictor = nn.Sequential(
                    nn.Linear(jepa_latent_dim + output_dim, 64),
                    copy.deepcopy(resolve_nn_activation(activation)),
                    nn.Linear(64, jepa_latent_dim),
                )
                self._jepa_training_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the RNN."""
        # Extract and concatenate observation groups and normalize
        latent = super().get_latent(obs)
        if self.structure == "mlp_gru":
            latent = self.encoder(latent)
        elif self.structure == "split_cnn_gru_mlp" or self.structure == "split_cnn_jepa_gru_mlp":
            assert self.recurrent_obs_range is not None
            assert self.cnn_obs_range is not None
            assert self.cnn_obs_shape is not None
            assert self.shortcut_obs_range is not None
            cnn_start, cnn_end = self.cnn_obs_range
            shortcut_start, shortcut_end = self.shortcut_obs_range
            if self.structure == "split_cnn_jepa_gru_mlp":
                assert self.jepa_state_obs_range is not None
                assert self.jepa_previous_action_obs_range is not None
                state_start, state_end = self.jepa_state_obs_range
                action_start, action_end = self.jepa_previous_action_obs_range
                physical_state = latent[..., state_start:state_end]
                online_latent = self.jepa_online_encoder(physical_state)
                recurrent_input = torch.cat((online_latent, latent[..., action_start:action_end]), dim=-1)
                if masks is None:
                    recurrent_latent = self.rnn(recurrent_input, None, hidden_state).squeeze(0)
                else:
                    if hidden_state is None:
                        raise ValueError("Hidden states are required for recurrent JEPA updates.")
                    padded_hidden, _ = self.rnn.rnn(recurrent_input, hidden_state)
                    recurrent_latent = unpad_trajectories(padded_hidden, masks)
                    self._jepa_training_cache = (physical_state, online_latent, padded_hidden, masks)
            else:
                recurrent_start, recurrent_end = self.recurrent_obs_range
                recurrent_latent = self.rnn(latent[..., recurrent_start:recurrent_end], masks, hidden_state).squeeze(0)
            cnn_observations = latent[..., cnn_start:cnn_end]
            leading_shape = cnn_observations.shape[:-1]
            cnn_latent = self.cnn_encoder(
                cnn_observations.reshape(-1, self.cnn_obs_shape[0], self.cnn_obs_shape[1]).transpose(1, 2)
            ).reshape(*leading_shape, -1)
            shortcut_latent = latent[..., shortcut_start:shortcut_end]
            if masks is not None:
                cnn_latent = unpad_trajectories(cnn_latent, masks)
                shortcut_latent = unpad_trajectories(shortcut_latent, masks)
                if self.structure == "split_cnn_jepa_gru_mlp":
                    online_latent = unpad_trajectories(online_latent, masks)
            if self.structure == "split_cnn_jepa_gru_mlp":
                return torch.cat((online_latent, recurrent_latent, cnn_latent, shortcut_latent), dim=-1)
            return torch.cat((recurrent_latent, cnn_latent, shortcut_latent), dim=-1)
        # Pass through the RNN
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    def compute_jepa_loss(
        self, trajectory_actions: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
        """Predict future physical-state latents from recurrent context and sampled actions."""
        if self.structure != "split_cnn_jepa_gru_mlp" or self._jepa_training_cache is None:
            raise RuntimeError("JEPA loss requires a preceding padded forward pass through a JEPA actor.")
        physical_state, online_latent, hidden, masks = self._jepa_training_cache
        self._jepa_training_cache = None
        if trajectory_actions.shape[:2] != masks.shape:
            raise ValueError("Padded action trajectories must align with the recurrent trajectory masks.")

        target_latent = self.jepa_target_encoder(physical_state).detach()
        prediction = self.jepa_context_projection(hidden)
        losses: dict[int, torch.Tensor] = {}
        for step in range(1, max(self.jepa_prediction_steps) + 1):
            prediction = self.jepa_transition_predictor(
                torch.cat((prediction[:-1], trajectory_actions[step - 1 : -1]), dim=-1)
            )
            if step not in self.jepa_prediction_steps:
                continue
            valid = masks[:-step] & masks[step:]
            if valid.any():
                cosine = torch.nn.functional.cosine_similarity(prediction[valid], target_latent[step:][valid], dim=-1)
                losses[step] = 1.0 - cosine.mean()
            else:
                losses[step] = online_latent.sum() * 0.0

        valid_online_latent = online_latent[masks]
        latent_std = valid_online_latent.std(dim=0, unbiased=False).mean()
        return torch.stack(tuple(losses.values())).mean(), losses, latent_std

    @torch.no_grad()
    def update_jepa_target(self, momentum: float) -> None:
        """Update the target encoder with an exponential moving average."""
        if self.structure != "split_cnn_jepa_gru_mlp":
            raise RuntimeError("Only JEPA actors have a target encoder.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"JEPA target momentum must be in [0, 1), got {momentum}.")
        for target, online in zip(
            self.jepa_target_encoder.parameters(), self.jepa_online_encoder.parameters(), strict=True
        ):
            target.lerp_(online, 1.0 - momentum)

    @staticmethod
    def _validate_obs_range(name: str, obs_range: tuple[int, int] | list[int] | None) -> tuple[int, int]:
        """Validate and normalize one half-open observation range."""
        if obs_range is None or len(obs_range) != 2:
            raise ValueError(f"{name} must contain exactly two indices for a split network structure.")
        start, end = (int(value) for value in obs_range)
        if start < 0 or end <= start:
            raise ValueError(f"{name} must satisfy 0 <= start < end, got {(start, end)}.")
        return (start, end)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_jit_with_explicit_state(self) -> nn.Module:
        """Return a batch-safe Torch JIT export with caller-owned GRU state."""
        if not isinstance(self.rnn.rnn, nn.GRU):
            raise NotImplementedError("Explicit-state JIT export currently only supports GRU models.")
        return _TorchGRUModelWithExplicitState(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.latent_dim


class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of a GRU-based RNNModel."""
        super().__init__()
        self.structure = model.structure
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder) if model.structure == "mlp_gru" else nn.Identity()
        self.recurrent_obs_start, self.recurrent_obs_end = model.recurrent_obs_range or (0, model.obs_dim)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.cnn_channels = list(model.cnn_channels or ())
        self.cnn_kernel_size = int(model.cnn_kernel_size or 0)
        self.cnn_output_dim = int(model.cnn_output_dim or 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.jepa_state_obs_start, self.jepa_state_obs_end = model.jepa_state_obs_range or (0, 0)
        self.jepa_previous_action_obs_start, self.jepa_previous_action_obs_end = (
            model.jepa_previous_action_obs_range or (0, 0)
        )
        self.jepa_online_encoder = (
            copy.deepcopy(model.jepa_online_encoder) if model.structure == "split_cnn_jepa_gru_mlp" else nn.Identity()
        )
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder)
            if model.structure in {"split_cnn_gru_mlp", "split_cnn_jepa_gru_mlp"}
            else nn.Identity()
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU inference step and update hidden states."""
        x = self.obs_normalizer(x)
        cnn = x[..., :0]
        online_latent = x[..., :0]
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_cnn_gru_mlp" or self.structure == "split_cnn_jepa_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            if self.structure == "split_cnn_jepa_gru_mlp":
                online_latent = self.jepa_online_encoder(x[..., self.jepa_state_obs_start : self.jepa_state_obs_end])
                x = torch.cat(
                    (
                        online_latent,
                        x[..., self.jepa_previous_action_obs_start : self.jepa_previous_action_obs_end],
                    ),
                    dim=-1,
                )
            else:
                x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        if self.structure == "split_cnn_gru_mlp":
            x = torch.cat((x, cnn, bypass), dim=-1)
        elif self.structure == "split_cnn_jepa_gru_mlp":
            x = torch.cat((online_latent, x, cnn, bypass), dim=-1)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported GRU hidden states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchGRUModelWithExplicitState(nn.Module):
    """Exportable GRU model whose recurrent state is owned by the caller."""

    def __init__(self, model: RNNModel) -> None:
        super().__init__()
        self.structure = model.structure
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder) if model.structure == "mlp_gru" else nn.Identity()
        self.recurrent_obs_start, self.recurrent_obs_end = model.recurrent_obs_range or (0, model.obs_dim)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.cnn_channels = list(model.cnn_channels or ())
        self.cnn_kernel_size = int(model.cnn_kernel_size or 0)
        self.cnn_output_dim = int(model.cnn_output_dim or 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.jepa_state_obs_start, self.jepa_state_obs_end = model.jepa_state_obs_range or (0, 0)
        self.jepa_previous_action_obs_start, self.jepa_previous_action_obs_end = (
            model.jepa_previous_action_obs_range or (0, 0)
        )
        self.jepa_online_encoder = (
            copy.deepcopy(model.jepa_online_encoder) if model.structure == "split_cnn_jepa_gru_mlp" else nn.Identity()
        )
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder)
            if model.structure in {"split_cnn_gru_mlp", "split_cnn_jepa_gru_mlp"}
            else nn.Identity()
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        if model.structure == "split_cnn_jepa_gru_mlp":
            self.format_version = 4
            self.fusion_order = "online_latent,gru_hidden,cnn_latent,shortcut"
        elif model.structure == "split_cnn_gru_mlp":
            self.format_version = 3
            self.fusion_order = "gru_hidden,cnn_latent,shortcut"
        else:
            self.format_version = 1
            self.fusion_order = ""
        self.observation_dim = model.obs_dim
        self.action_dim = model.output_dim
        self.num_layers = self.rnn.num_layers
        self.hidden_size = self.rnn.hidden_size

    def forward(self, observations: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one deterministic inference step with explicit GRU state."""
        x = self.obs_normalizer(observations)
        cnn = x[..., :0]
        online_latent = x[..., :0]
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_cnn_gru_mlp" or self.structure == "split_cnn_jepa_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            if self.structure == "split_cnn_jepa_gru_mlp":
                online_latent = self.jepa_online_encoder(x[..., self.jepa_state_obs_start : self.jepa_state_obs_end])
                x = torch.cat(
                    (
                        online_latent,
                        x[..., self.jepa_previous_action_obs_start : self.jepa_previous_action_obs_end],
                    ),
                    dim=-1,
                )
            else:
                x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]
        x, next_hidden_state = self.rnn(x.unsqueeze(0), hidden_state)
        x = x.squeeze(0)
        if self.structure == "split_cnn_gru_mlp":
            x = torch.cat((x, cnn, bypass), dim=-1)
        elif self.structure == "split_cnn_jepa_gru_mlp":
            x = torch.cat((online_latent, x, cnn, bypass), dim=-1)
        actions = self.deterministic_output(self.mlp(x))
        return actions, next_hidden_state


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based RNNModel."""
        super().__init__()
        self.structure = model.structure
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder) if model.structure == "mlp_gru" else nn.Identity()
        self.recurrent_obs_start, self.recurrent_obs_end = model.recurrent_obs_range or (0, model.obs_dim)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LSTM inference step and update hidden and cell states."""
        x = self.obs_normalizer(x)
        if self.structure == "mlp_gru":
            x = self.encoder(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNModel."""
        super().__init__()
        self.verbose = verbose
        self.structure = model.structure
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder) if model.structure == "mlp_gru" else nn.Identity()
        self.recurrent_obs_start, self.recurrent_obs_end = model.recurrent_obs_range or (0, model.obs_dim)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.jepa_state_obs_start, self.jepa_state_obs_end = model.jepa_state_obs_range or (0, 0)
        self.jepa_previous_action_obs_start, self.jepa_previous_action_obs_end = (
            model.jepa_previous_action_obs_range or (0, 0)
        )
        self.jepa_online_encoder = (
            copy.deepcopy(model.jepa_online_encoder) if model.structure == "split_cnn_jepa_gru_mlp" else nn.Identity()
        )
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder)
            if model.structure in {"split_cnn_gru_mlp", "split_cnn_jepa_gru_mlp"}
            else nn.Identity()
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(obs)
        cnn = x[..., :0]
        online_latent = x[..., :0]
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_cnn_gru_mlp" or self.structure == "split_cnn_jepa_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            if self.structure == "split_cnn_jepa_gru_mlp":
                online_latent = self.jepa_online_encoder(x[..., self.jepa_state_obs_start : self.jepa_state_obs_end])
                x = torch.cat(
                    (
                        online_latent,
                        x[..., self.jepa_previous_action_obs_start : self.jepa_previous_action_obs_end],
                    ),
                    dim=-1,
                )
            else:
                x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            if self.structure == "split_cnn_gru_mlp":
                x = torch.cat((x, cnn, bypass), dim=-1)
            elif self.structure == "split_cnn_jepa_gru_mlp":
                x = torch.cat((online_latent, x, cnn, bypass), dim=-1)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            if self.structure == "split_cnn_gru_mlp":
                x = torch.cat((x, cnn, bypass), dim=-1)
            elif self.structure == "split_cnn_jepa_gru_mlp":
                x = torch.cat((online_latent, x, cnn, bypass), dim=-1)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
