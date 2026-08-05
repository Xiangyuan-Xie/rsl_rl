# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

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
        structure: Literal["gru_mlp", "mlp_gru", "split_gru_mlp", "split_cnn_gru_mlp"] = "gru_mlp",
        recurrent_obs_range: tuple[int, int] | list[int] | None = None,
        bypass_obs_range: tuple[int, int] | list[int] | None = None,
        cnn_obs_range: tuple[int, int] | list[int] | None = None,
        cnn_obs_shape: tuple[int, int] | list[int] | None = None,
        cnn_channels: tuple[int, ...] | list[int] | None = None,
        cnn_kernel_size: int | None = None,
        cnn_output_dim: int | None = None,
        shortcut_obs_range: tuple[int, int] | list[int] | None = None,
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
                ``"mlp_gru"`` uses ``obs -> encoder MLP -> RNN -> linear head``; ``"split_gru_mlp"`` sends one
                observation slice through a GRU and concatenates its hidden state with a direct observation slice.
            recurrent_obs_range: Half-open observation range passed through the GRU by ``"split_gru_mlp"``.
            bypass_obs_range: Half-open observation range concatenated after the GRU by ``"split_gru_mlp"``.
            cnn_obs_range: Half-open trajectory range processed by ``"split_cnn_gru_mlp"``.
            cnn_obs_shape: Trajectory shape as ``(time, channels)`` before the temporal convolutions.
            cnn_channels: Output channel count for each valid Conv1d layer.
            cnn_kernel_size: Shared kernel size of the valid Conv1d layers.
            cnn_output_dim: Dimension of the linear projection after the flattened convolutions.
            shortcut_obs_range: Half-open direct-feedback range concatenated at the fusion layer.
        """
        if structure not in {"gru_mlp", "mlp_gru", "split_gru_mlp", "split_cnn_gru_mlp"}:
            raise ValueError(f"Unsupported structure: {structure}")
        if structure == "split_gru_mlp":
            if rnn_type.lower() != "gru":
                raise ValueError("split_gru_mlp only supports GRU recurrent modules.")
            if any(
                value is not None
                for value in (
                    cnn_obs_range,
                    cnn_obs_shape,
                    cnn_channels,
                    cnn_kernel_size,
                    cnn_output_dim,
                    shortcut_obs_range,
                )
            ):
                raise ValueError("CNN branch settings are not valid with structure='split_gru_mlp'.")
            recurrent_obs_range = self._validate_obs_range("recurrent_obs_range", recurrent_obs_range)
            bypass_obs_range = self._validate_obs_range("bypass_obs_range", bypass_obs_range)
        elif structure == "split_cnn_gru_mlp":
            if rnn_type.lower() != "gru":
                raise ValueError("split_cnn_gru_mlp only supports GRU recurrent modules.")
            if bypass_obs_range is not None:
                raise ValueError("bypass_obs_range is not valid with structure='split_cnn_gru_mlp'.")
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
        elif any(
            value is not None
            for value in (
                recurrent_obs_range,
                bypass_obs_range,
                cnn_obs_range,
                cnn_obs_shape,
                cnn_channels,
                cnn_kernel_size,
                cnn_output_dim,
                shortcut_obs_range,
            )
        ):
            raise ValueError("Observation branch settings are only valid with a split network structure.")
        self.structure = structure
        self.recurrent_obs_range = recurrent_obs_range
        self.bypass_obs_range = bypass_obs_range
        self.cnn_obs_range = cnn_obs_range
        self.cnn_obs_shape = cnn_obs_shape
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_output_dim = cnn_output_dim
        self.shortcut_obs_range = shortcut_obs_range
        if structure == "split_cnn_gru_mlp":
            assert shortcut_obs_range is not None
            assert cnn_output_dim is not None
            self.latent_dim = rnn_hidden_dim + cnn_output_dim + shortcut_obs_range[1] - shortcut_obs_range[0]
        elif bypass_obs_range is not None:
            self.latent_dim = rnn_hidden_dim + bypass_obs_range[1] - bypass_obs_range[0]
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

        if self.structure == "split_gru_mlp":
            assert recurrent_obs_range is not None
            assert bypass_obs_range is not None
            if recurrent_obs_range[1] > self.obs_dim or bypass_obs_range[1] > self.obs_dim:
                raise ValueError(
                    "split_gru_mlp observation ranges must fit within the concatenated observation dimension "
                    f"{self.obs_dim}, got recurrent={recurrent_obs_range} and bypass={bypass_obs_range}."
                )
        elif self.structure == "split_cnn_gru_mlp":
            assert recurrent_obs_range is not None
            assert cnn_obs_range is not None
            assert shortcut_obs_range is not None
            ranges = (recurrent_obs_range, cnn_obs_range, shortcut_obs_range)
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
            self.rnn = RNN(
                recurrent_obs_range[1] - recurrent_obs_range[0],
                rnn_hidden_dim,
                rnn_num_layers,
                rnn_type,
            )
            if self.structure == "split_cnn_gru_mlp":
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

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the RNN."""
        # Extract and concatenate observation groups and normalize
        latent = super().get_latent(obs)
        if self.structure == "mlp_gru":
            latent = self.encoder(latent)
        elif self.structure == "split_gru_mlp":
            assert self.recurrent_obs_range is not None
            assert self.bypass_obs_range is not None
            recurrent_start, recurrent_end = self.recurrent_obs_range
            bypass_start, bypass_end = self.bypass_obs_range
            recurrent_latent = self.rnn(latent[..., recurrent_start:recurrent_end], masks, hidden_state).squeeze(0)
            bypass_latent = latent[..., bypass_start:bypass_end]
            if masks is not None:
                bypass_latent = unpad_trajectories(bypass_latent, masks)
            return torch.cat((recurrent_latent, bypass_latent), dim=-1)
        elif self.structure == "split_cnn_gru_mlp":
            assert self.recurrent_obs_range is not None
            assert self.cnn_obs_range is not None
            assert self.cnn_obs_shape is not None
            assert self.shortcut_obs_range is not None
            recurrent_start, recurrent_end = self.recurrent_obs_range
            cnn_start, cnn_end = self.cnn_obs_range
            shortcut_start, shortcut_end = self.shortcut_obs_range
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
            return torch.cat((recurrent_latent, cnn_latent, shortcut_latent), dim=-1)
        # Pass through the RNN
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    @staticmethod
    def _validate_obs_range(
        name: str, obs_range: tuple[int, int] | list[int] | None
    ) -> tuple[int, int]:
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
        self.bypass_obs_start, self.bypass_obs_end = model.bypass_obs_range or (0, 0)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.cnn_channels = list(model.cnn_channels or ())
        self.cnn_kernel_size = int(model.cnn_kernel_size or 0)
        self.cnn_output_dim = int(model.cnn_output_dim or 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder) if model.structure == "split_cnn_gru_mlp" else nn.Identity()
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
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_gru_mlp":
            bypass = x[..., self.bypass_obs_start : self.bypass_obs_end]
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        elif self.structure == "split_cnn_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        if self.structure == "split_gru_mlp":
            x = torch.cat((x, bypass), dim=-1)
        elif self.structure == "split_cnn_gru_mlp":
            x = torch.cat((x, cnn, bypass), dim=-1)
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
        self.bypass_obs_start, self.bypass_obs_end = model.bypass_obs_range or (0, 0)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.cnn_channels = list(model.cnn_channels or ())
        self.cnn_kernel_size = int(model.cnn_kernel_size or 0)
        self.cnn_output_dim = int(model.cnn_output_dim or 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder) if model.structure == "split_cnn_gru_mlp" else nn.Identity()
        )
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        if model.structure == "split_cnn_gru_mlp":
            self.format_version = 3
            self.fusion_order = "gru_hidden,cnn_latent,shortcut"
        elif model.structure == "split_gru_mlp":
            self.format_version = 2
            self.fusion_order = "gru_hidden,bypass"
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
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_gru_mlp":
            bypass = x[..., self.bypass_obs_start : self.bypass_obs_end]
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        elif self.structure == "split_cnn_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]
        x, next_hidden_state = self.rnn(x.unsqueeze(0), hidden_state)
        x = x.squeeze(0)
        if self.structure == "split_gru_mlp":
            x = torch.cat((x, bypass), dim=-1)
        elif self.structure == "split_cnn_gru_mlp":
            x = torch.cat((x, cnn, bypass), dim=-1)
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
        self.bypass_obs_start, self.bypass_obs_end = model.bypass_obs_range or (0, 0)
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
            bypass = x[..., :0]
        elif self.structure == "split_gru_mlp":
            bypass = x[..., self.bypass_obs_start : self.bypass_obs_end]
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        if self.structure == "split_gru_mlp":
            x = torch.cat((x, bypass), dim=-1)
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
        self.bypass_obs_start, self.bypass_obs_end = model.bypass_obs_range or (0, 0)
        self.cnn_obs_start, self.cnn_obs_end = model.cnn_obs_range or (0, 0)
        self.cnn_time, self.cnn_features = model.cnn_obs_shape or (0, 0)
        self.shortcut_obs_start, self.shortcut_obs_end = model.shortcut_obs_range or (0, 0)
        self.cnn_encoder = (
            copy.deepcopy(model.cnn_encoder) if model.structure == "split_cnn_gru_mlp" else nn.Identity()
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
        if self.structure == "mlp_gru":
            x = self.encoder(x)
            bypass = x[..., :0]
        elif self.structure == "split_gru_mlp":
            bypass = x[..., self.bypass_obs_start : self.bypass_obs_end]
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        elif self.structure == "split_cnn_gru_mlp":
            bypass = x[..., self.shortcut_obs_start : self.shortcut_obs_end]
            cnn = x[..., self.cnn_obs_start : self.cnn_obs_end]
            cnn = self.cnn_encoder(cnn.reshape(x.shape[0], self.cnn_time, self.cnn_features).transpose(1, 2))
            x = x[..., self.recurrent_obs_start : self.recurrent_obs_end]
        else:
            bypass = x[..., :0]

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            if self.structure == "split_gru_mlp":
                x = torch.cat((x, bypass), dim=-1)
            elif self.structure == "split_cnn_gru_mlp":
                x = torch.cat((x, cnn, bypass), dim=-1)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            if self.structure == "split_gru_mlp":
                x = torch.cat((x, bypass), dim=-1)
            elif self.structure == "split_cnn_gru_mlp":
                x = torch.cat((x, cnn, bypass), dim=-1)
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
