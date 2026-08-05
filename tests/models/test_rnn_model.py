# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RNNModel."""

from __future__ import annotations

import tempfile

import onnx
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import RNNModel
from rsl_rl.utils import split_and_pad_trajectories
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4


def _make_rnn_model(rnn_type: str = "gru", **kwargs: object) -> tuple[RNNModel, TensorDict]:
    """Create an RNNModel and matching observations."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    obs_groups = {"actor": ["policy"]}
    defaults = dict(
        hidden_dims=[32, 32],
        activation="elu",
        rnn_type=rnn_type,
        rnn_hidden_dim=16,
        rnn_num_layers=1,
    )
    defaults.update(kwargs)
    model = RNNModel(obs, obs_groups, "actor", NUM_ACTIONS, **defaults)
    return model, obs


def _make_cnn_gru_model(observation_dim: int = 72) -> tuple[RNNModel, TensorDict]:
    obs = make_obs(NUM_ENVS, observation_dim)
    model = RNNModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        NUM_ACTIONS,
        hidden_dims=[48],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=32,
        rnn_num_layers=1,
        structure="split_cnn_gru_mlp",
        recurrent_obs_range=(55, observation_dim),
        cnn_obs_range=(0, 55),
        cnn_obs_shape=(11, 5),
        cnn_channels=(32, 32, 8),
        cnn_kernel_size=3,
        cnn_output_dim=32,
        shortcut_obs_range=(55, 64),
        distribution_cfg={"class_name": "BetaDistribution", "action_range": (0.0, 1.0)},
    )
    return model, obs


class TestHiddenStateReset:
    """Tests for hidden state reset behavior on done environments."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_reset_done_env_zeros_its_state(self, rnn_type: str) -> None:
        """reset(dones) should zero the hidden state of done envs, preserving others."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Run a step to initialize hidden state
        model(obs)
        assert model.get_hidden_state() is not None

        # Save state of env 1 before reset
        if rnn_type == "lstm":
            h_before, _c_before = model.get_hidden_state()
            h1_before = h_before[:, 1, :].clone()
        else:
            h_before = model.get_hidden_state()
            h1_before = h_before[:, 1, :].clone()

        # Mark env 0 as done, env 1 as not done
        dones = torch.zeros(NUM_ENVS)
        dones[0] = 1.0
        model.reset(dones)

        if rnn_type == "lstm":
            h_after, c_after = model.get_hidden_state()
            # Env 0 should be zeroed
            assert torch.all(h_after[:, 0, :] == 0)
            assert torch.all(c_after[:, 0, :] == 0)
            # Env 1 should be preserved
            assert torch.allclose(h_after[:, 1, :], h1_before)
        else:
            h_after = model.get_hidden_state()
            assert torch.all(h_after[:, 0, :] == 0)
            assert torch.allclose(h_after[:, 1, :], h1_before)

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_full_reset_clears_all_state(self, rnn_type: str) -> None:
        """reset() without arguments should clear the hidden state entirely."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)
        assert model.get_hidden_state() is not None

        model.reset()
        assert model.get_hidden_state() is None


class TestHiddenStateDetach:
    """Tests for hidden state detach (TBPTT support)."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_detach_breaks_gradient(self, rnn_type: str) -> None:
        """After detach_hidden_state(), the hidden state should not require grad."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)

        model.detach_hidden_state()

        state = model.get_hidden_state()
        if rnn_type == "lstm":
            assert not state[0].requires_grad
            assert not state[1].requires_grad
        else:
            assert not state.requires_grad

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_detach_preserves_values(self, rnn_type: str) -> None:
        """Detaching should not change the hidden state values."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)
        model(obs)

        state = model.get_hidden_state()
        if rnn_type == "lstm":
            h_before = state[0].clone()
            c_before = state[1].clone()
        else:
            h_before = state.clone()

        model.detach_hidden_state()

        state_after = model.get_hidden_state()
        if rnn_type == "lstm":
            assert torch.allclose(state_after[0], h_before)
            assert torch.allclose(state_after[1], c_before)
        else:
            assert torch.allclose(state_after, h_before)


class TestSequentialConsistency:
    """Tests that hidden state accumulates context over sequential steps."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_hidden_state_accumulates_context(self, rnn_type: str) -> None:
        """Running multiple steps should change the output compared to a fresh model."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Fresh output
        output_fresh = model(obs).detach().clone()
        model.reset()

        # Run 3 steps to accumulate hidden state
        for _ in range(3):
            model(obs)
        output_with_context = model(obs).detach()

        assert not torch.allclose(output_fresh, output_with_context, atol=1e-6), (
            "Output should differ after accumulating hidden state context"
        )

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_reset_restores_fresh_behavior(self, rnn_type: str) -> None:
        """After reset(), output should match a fresh start."""
        model, obs = _make_rnn_model(rnn_type=rnn_type)

        # Fresh output
        output_fresh = model(obs).detach().clone()

        # Accumulate context
        for _ in range(5):
            model(obs)

        # Reset and check
        model.reset()
        output_after_reset = model(obs).detach()

        assert torch.allclose(output_fresh, output_after_reset, atol=1e-6), (
            "Output after reset should match fresh output"
        )


class TestRNNModelStructure:
    """Tests for RNN/MLP ordering options."""

    def test_mlp_gru_structure_places_encoder_before_rnn(self) -> None:
        """The mlp_gru structure should encode observations before recurrence."""
        model, obs = _make_rnn_model(structure="mlp_gru", hidden_dims=[32, 16])

        assert model.structure == "mlp_gru"
        assert hasattr(model, "encoder")
        assert model.encoder[0].in_features == OBS_DIM
        assert model.rnn.rnn.input_size == 16
        assert model.mlp[0].in_features == 16

        output = model(obs)
        assert output.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_default_structure_keeps_upstream_gru_mlp_behavior(self) -> None:
        """The default structure should preserve the upstream GRU-first layout."""
        model, _obs = _make_rnn_model(hidden_dims=[32, 16])

        assert model.structure == "gru_mlp"
        assert not hasattr(model, "encoder")
        assert model.rnn.rnn.input_size == OBS_DIM
        assert model.mlp[0].in_features == 16

    def test_split_gru_mlp_fuses_recurrent_and_overlapping_bypass_slices(self) -> None:
        """The split structure should preserve the direct slice while recurrently encoding its overlap."""
        model, obs = _make_rnn_model(
            structure="split_gru_mlp",
            recurrent_obs_range=(2, 8),
            bypass_obs_range=(0, 4),
            hidden_dims=[32, 16],
        )

        assert model.rnn.rnn.input_size == 6
        assert model.mlp[0].in_features == 20

        observations = obs["policy"]
        hidden = torch.zeros(1, NUM_ENVS, 16)
        model.reset(hidden_state=hidden.clone())
        with torch.no_grad():
            recurrent, expected_hidden = model.rnn.rnn(observations[:, 2:8].unsqueeze(0), hidden)
            expected = model.mlp(torch.cat((recurrent.squeeze(0), observations[:, 0:4]), dim=-1))
            actual = model(obs)

        assert torch.allclose(actual, expected)
        assert torch.allclose(model.get_hidden_state(), expected_hidden)

    def test_split_cnn_gru_mlp_matches_am_pose_contract(self) -> None:
        """The AM Pose temporal encoder must preserve all three branch widths and parameter count."""
        model, obs = _make_cnn_gru_model()

        assert model.rnn.rnn.input_size == 17
        assert model.mlp[0].in_features == 73
        assert model.cnn_encoder[0].in_channels == 5
        assert model.cnn_encoder[0].out_channels == 32
        assert model.cnn_encoder[2].out_channels == 32
        assert model.cnn_encoder[4].out_channels == 8
        assert model.cnn_encoder[-1].in_features == 40
        assert model.cnn_encoder[-1].out_features == 32
        assert sum(parameter.numel() for parameter in model.parameters()) == 14_544
        assert model(obs).shape == (NUM_ENVS, NUM_ACTIONS)

        dual_model, dual_obs = _make_cnn_gru_model(observation_dim=78)
        assert dual_model.rnn.rnn.input_size == 23
        assert sum(parameter.numel() for parameter in dual_model.parameters()) == 15_120
        assert dual_model(dual_obs).shape == (NUM_ENVS, NUM_ACTIONS)

    def test_split_cnn_gru_mlp_uses_current_and_future_trajectory_points(self) -> None:
        """Both endpoints of the 11-point sequence must influence the CNN latent."""
        model, _obs = _make_cnn_gru_model()
        baseline = torch.zeros(NUM_ENVS, 72)
        first_changed = baseline.clone()
        first_changed[:, 0] = 1.0
        last_changed = baseline.clone()
        last_changed[:, 54] = 1.0

        def encode(observations: torch.Tensor) -> torch.Tensor:
            trajectory = observations[:, :55].reshape(-1, 11, 5).transpose(1, 2)
            return model.cnn_encoder(trajectory)

        assert not torch.allclose(encode(baseline), encode(first_changed))
        assert not torch.allclose(encode(baseline), encode(last_changed))

    def test_split_cnn_gru_mlp_unpads_all_branches_with_recurrent_masks(self) -> None:
        """CNN and shortcut samples must follow the same valid-step mask as GRU outputs."""
        model, _obs = _make_cnn_gru_model()
        observations = TensorDict(
            {"policy": torch.randn(5, 2, 72)},
            batch_size=[5, 2],
        )
        dones = torch.zeros(5, 2, dtype=torch.bool)
        dones[1, 0] = True
        padded, masks = split_and_pad_trajectories(observations, dones)
        hidden = torch.zeros(1, masks.shape[1], 32)

        output = model(padded, masks=masks, hidden_state=hidden)

        assert output.shape == (5, 2, NUM_ACTIONS)
        assert torch.all(torch.isfinite(output))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"recurrent_obs_range": (2, 8)}, "bypass_obs_range must contain"),
            (
                {"recurrent_obs_range": (2, 9), "bypass_obs_range": (0, 4)},
                "must fit within",
            ),
            (
                {"recurrent_obs_range": (2, 8), "bypass_obs_range": (0, 4), "rnn_type": "lstm"},
                "only supports GRU",
            ),
        ],
    )
    def test_invalid_split_structure_configuration_is_rejected(self, kwargs: dict, message: str) -> None:
        """Incomplete, out-of-bounds, and non-GRU split configurations should fail early."""
        defaults = {
            "structure": "split_gru_mlp",
            "recurrent_obs_range": (2, 8),
            "bypass_obs_range": (0, 4),
        }
        defaults.update(kwargs)
        if "bypass_obs_range" not in kwargs and "recurrent_obs_range" in kwargs:
            defaults.pop("bypass_obs_range")
        with pytest.raises(ValueError, match=message):
            _make_rnn_model(**defaults)

    def test_invalid_structure_is_rejected(self) -> None:
        """Unknown RNN/MLP structures should fail during construction."""
        with pytest.raises(ValueError, match="Unsupported structure"):
            _make_rnn_model(structure="cnn_gru")


class TestRNNModelJITExport:
    """Tests for JIT export fidelity of RNN models."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_jit_export_model(self, rnn_type: str) -> None:
        """JIT export of an RNN model should produce deterministic outputs matching the original."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            activation="elu",
            rnn_type=rnn_type,
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        original_output = model(obs, stochastic_output=False).detach()

        jit_model = torch.jit.script(model.as_jit())
        jit_model.reset()
        obs_concat = torch.cat([obs[g] for g in model.obs_groups], dim=-1)
        jit_output = jit_model(obs_concat)

        assert torch.allclose(original_output, jit_output, atol=1e-5), f"JIT stochastic export mismatch for {rnn_type}"

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_jit_export_sequential_consistency(self, rnn_type: str) -> None:
        """JIT model should accumulate hidden state across steps, matching original multi-step behavior."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            activation="elu",
            rnn_type=rnn_type,
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        jit_model = torch.jit.script(model.as_jit())
        jit_model.reset()
        obs_concat = torch.cat([obs[g] for g in model.obs_groups], dim=-1)

        for _ in range(3):
            original_out = model(obs, stochastic_output=False).detach()
            jit_out = jit_model(obs_concat)

        assert torch.allclose(original_out, jit_out, atol=1e-5), f"JIT sequential mismatch for {rnn_type}"


class TestRNNModelExplicitStateJITExport:
    """Tests for batch-safe GRU exports with caller-owned hidden state."""

    def _export(
        self, num_envs: int = NUM_ENVS, distribution_cfg: dict | None = None
    ) -> tuple[RNNModel, TensorDict, torch.jit.ScriptModule]:
        obs = make_obs(num_envs, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 16],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_dim=16,
            rnn_num_layers=2,
            structure="mlp_gru",
            distribution_cfg=distribution_cfg,
        ).eval()
        return model, obs, torch.jit.script(model.as_jit_with_explicit_state())

    def test_export_matches_model_and_exposes_contract_metadata(self) -> None:
        """Explicit-state export should match the model and publish its contract."""
        distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}
        model, obs, exported = self._export(distribution_cfg=distribution_cfg)
        hidden = torch.zeros(2, NUM_ENVS, 16)
        model.reset(hidden_state=hidden.clone())

        expected = model(obs, stochastic_output=False).detach()
        obs_concat = torch.cat([obs[group] for group in model.obs_groups], dim=-1)
        actions, next_hidden = exported(obs_concat, hidden)

        assert torch.allclose(actions, expected, atol=1e-5)
        assert torch.allclose(next_hidden, model.get_hidden_state(), atol=1e-5)
        assert exported.format_version == 1
        assert exported.observation_dim == OBS_DIM
        assert exported.action_dim == NUM_ACTIONS
        assert exported.num_layers == 2
        assert exported.hidden_size == 16

    def test_caller_owned_hidden_state_preserves_sequence_context(self) -> None:
        """Feeding returned state into the next call should preserve context."""
        model, obs, exported = self._export()
        obs_concat = torch.cat([obs[group] for group in model.obs_groups], dim=-1)
        hidden = torch.zeros(2, NUM_ENVS, 16)

        first_actions, hidden = exported(obs_concat, hidden)
        second_actions, second_hidden = exported(obs_concat, hidden)

        assert not torch.allclose(first_actions, second_actions)
        assert not torch.allclose(hidden, second_hidden)

    def test_export_accepts_different_batch_sizes(self) -> None:
        """An exported policy should not bind recurrent state to one batch size."""
        _model, _obs, exported = self._export()

        for num_envs in (1, 7):
            actions, hidden = exported(torch.randn(num_envs, OBS_DIM), torch.zeros(2, num_envs, 16))
            assert actions.shape == (num_envs, NUM_ACTIONS)
            assert hidden.shape == (2, num_envs, 16)

    def test_beta_export_is_deterministic(self) -> None:
        """Beta policies should export their deterministic distribution mean."""
        model, obs, exported = self._export(
            distribution_cfg={"class_name": "BetaDistribution", "action_range": (-1.0, 1.0)}
        )
        hidden = torch.zeros(2, NUM_ENVS, 16)
        model.reset(hidden_state=hidden.clone())
        expected = model(obs, stochastic_output=False).detach()
        obs_concat = torch.cat([obs[group] for group in model.obs_groups], dim=-1)

        first, _ = exported(obs_concat, hidden)
        second, _ = exported(obs_concat, hidden)

        assert torch.allclose(first, expected, atol=1e-5)
        assert torch.equal(first, second)

    def test_explicit_state_export_rejects_lstm(self) -> None:
        """The v1 explicit-state contract should reject unsupported LSTM models."""
        model, _obs = _make_rnn_model(rnn_type="lstm")

        with pytest.raises(NotImplementedError, match="only supports GRU"):
            model.as_jit_with_explicit_state()

    def test_split_export_matches_model_and_records_v2_contract(self) -> None:
        """Explicit-state split exports should preserve the fusion and publish its observation slices."""
        model, obs = _make_rnn_model(
            structure="split_gru_mlp",
            recurrent_obs_range=(2, 8),
            bypass_obs_range=(0, 4),
            hidden_dims=[32, 16],
        )
        exported = torch.jit.script(model.as_jit_with_explicit_state())
        hidden = torch.zeros(1, NUM_ENVS, 16)
        model.reset(hidden_state=hidden.clone())
        expected = model(obs).detach()

        actions, next_hidden = exported(obs["policy"], hidden)

        assert torch.allclose(actions, expected, atol=1e-5)
        assert torch.allclose(next_hidden, model.get_hidden_state(), atol=1e-5)
        assert exported.format_version == 2
        assert exported.recurrent_obs_start == 2
        assert exported.recurrent_obs_end == 8
        assert exported.bypass_obs_start == 0
        assert exported.bypass_obs_end == 4
        assert exported.fusion_order == "gru_hidden,bypass"

    def test_cnn_split_export_matches_model_and_records_v3_contract(self) -> None:
        """Explicit-state v3 export must retain CNN and shortcut metadata and numerics."""
        model, obs = _make_cnn_gru_model()
        exported = torch.jit.script(model.as_jit_with_explicit_state())
        hidden = torch.zeros(1, NUM_ENVS, 32)
        model.reset(hidden_state=hidden.clone())
        expected = model(obs).detach()

        actions, next_hidden = exported(obs["policy"], hidden)

        assert torch.allclose(actions, expected, atol=1.0e-5)
        assert torch.allclose(next_hidden, model.get_hidden_state(), atol=1.0e-5)
        assert exported.format_version == 3
        assert (exported.cnn_obs_start, exported.cnn_obs_end) == (0, 55)
        assert (exported.cnn_time, exported.cnn_features) == (11, 5)
        assert list(exported.cnn_channels) == [32, 32, 8]
        assert (exported.shortcut_obs_start, exported.shortcut_obs_end) == (55, 64)
        assert exported.fusion_order == "gru_hidden,cnn_latent,shortcut"


@pytest.mark.filterwarnings("ignore:.*legacy TorchScript.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:.*will be removed.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings("ignore::UserWarning:torch.onnx")
class TestRNNModelONNXExport:
    """Tests for ONNX export fidelity of RNN models."""

    @pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
    def test_onnx_export_model(self, rnn_type: str) -> None:
        """ONNX-exported RNN model should be a valid ONNX graph with correct I/O names."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 32],
            activation="elu",
            rnn_type=rnn_type,
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        )
        model.eval()

        onnx_model = model.as_onnx(verbose=False)
        onnx_model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            torch.onnx.export(
                onnx_model,
                onnx_model.get_dummy_inputs(),
                f.name,
                export_params=True,
                opset_version=18,
                input_names=onnx_model.input_names,
                output_names=onnx_model.output_names,
            )
            loaded = onnx.load(f.name)
            onnx.checker.check_model(loaded)

            assert [i.name for i in loaded.graph.input] == onnx_model.input_names
            assert [o.name for o in loaded.graph.output] == onnx_model.output_names

    def test_split_gru_mlp_onnx_export_preserves_direct_bypass(self) -> None:
        """The ONNX wrapper must reproduce the split actor before graph serialization."""
        obs = make_obs(1, OBS_DIM)
        model = RNNModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 16],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_dim=16,
            rnn_num_layers=1,
            structure="split_gru_mlp",
            recurrent_obs_range=(2, 8),
            bypass_obs_range=(0, 4),
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ).eval()
        exported = model.as_onnx(verbose=False).eval()
        inputs = exported.get_dummy_inputs()

        model.reset(hidden_state=inputs[1].clone())
        expected = model(obs, stochastic_output=False).detach()
        actual, next_hidden, _ = exported(obs["policy"], inputs[1])

        assert torch.allclose(actual, expected, atol=1.0e-5)
        assert torch.allclose(next_hidden, model.get_hidden_state(), atol=1.0e-5)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            torch.onnx.export(
                exported,
                inputs,
                file.name,
                export_params=True,
                opset_version=18,
                input_names=exported.input_names,
                output_names=exported.output_names,
            )
            onnx.checker.check_model(onnx.load(file.name))

    def test_split_cnn_gru_mlp_onnx_export_matches_model(self) -> None:
        """The ONNX wrapper must preserve CNN, GRU, and shortcut fusion."""
        model, obs = _make_cnn_gru_model()
        model = model.eval()
        exported = model.as_onnx(verbose=False).eval()
        hidden = torch.zeros(1, NUM_ENVS, 32)
        model.reset(hidden_state=hidden.clone())

        expected = model(obs, stochastic_output=False).detach()
        actual, next_hidden, _ = exported(obs["policy"], hidden)

        assert torch.allclose(actual, expected, atol=1.0e-5)
        assert torch.allclose(next_hidden, model.get_hidden_state(), atol=1.0e-5)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as file:
            torch.onnx.export(
                exported,
                exported.get_dummy_inputs(),
                file.name,
                export_params=True,
                opset_version=18,
                input_names=exported.input_names,
                output_names=exported.output_names,
            )
            onnx.checker.check_model(onnx.load(file.name))
