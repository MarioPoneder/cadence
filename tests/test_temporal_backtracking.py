"""Parameter-step acceptance against independently evaluated causal behavior."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import TemporalPatchNet


def rollout(parameters, inputs, boundary):
    state = boundary.copy()
    outputs = []
    for column in inputs.transpose(1, 0, 2):
        state = np.tanh(state) @ parameters["A"].T + column @ parameters["B"].T
        outputs.append(np.tanh(state) @ parameters["C"].T)
    return np.stack(outputs, axis=1)


@pytest.mark.parametrize("precision", [np.ones(2), np.array([0.3, 2.0])])
def test_overshoot_backtracks_from_original_boundary_and_carries_only_free(precision):
    rng = np.random.default_rng(1031)
    net = TemporalPatchNet(2, 3, 2, seed=1033, output_precision=precision)
    net.advance(rng.normal(size=(2, 3, 2)))
    boundary = net.state
    inputs = rng.normal(size=(2, 5, 2)) * 0.2
    target = rng.normal(size=(2, 5, 2)) * 0.1
    old = net.parameters()
    before = net.readback()
    free = net.imagine(inputs)
    old_loss = 0.5 * np.mean(precision * (rollout(old, inputs, boundary) - target) ** 2)
    result = net.observe(inputs, target, rate=128.0, backtrack=True)
    assert result.updated and result.reason == "updated"
    assert 0 < result.accepted_rate < 128
    assert result.replay_calls == len(result.replay_losses) > 1
    assert_allclose(result.initial_loss, old_loss, rtol=1e-14)
    new_loss = 0.5 * np.mean(
        precision * (rollout(net.parameters(), inputs, boundary) - target) ** 2
    )
    assert_allclose(result.final_loss, new_loss, rtol=1e-14)
    assert new_loss < old_loss
    norm = sum(np.sum(g * g) for g in result.delta.values())
    assert new_loss <= old_loss - 1e-4 * result.accepted_rate * norm
    for index, measured in enumerate(result.replay_losses):
        step = 128.0 * 0.5**index
        proposed = {k: old[k] - step * result.delta[k] for k in old}
        expected = 0.5 * np.mean(precision * (rollout(proposed, inputs, boundary) - target) ** 2)
        assert_allclose(measured, expected, rtol=1e-14)
    assert_array_equal(net.state, free.final_state)
    after = net.readback()
    assert after.updates == before.updates + 1
    assert after.parameter_revision == before.parameter_revision + 1
    assert after.state_parameter_revision == before.parameter_revision


def test_small_accepted_step_matches_fixed_rate_path_exactly():
    rng = np.random.default_rng(1049)
    net = TemporalPatchNet(2, 3, 1, seed=1051)
    net.advance(rng.normal(size=(2, 2, 2)) * 0.1)
    fixed = TemporalPatchNet.restore(net.snapshot())
    inputs = rng.normal(size=(2, 4, 2)) * 0.1
    target = rng.normal(size=(2, 4, 1)) * 0.1
    guarded = net.observe(inputs, target, rate=0.1, backtrack=True)
    original = fixed.observe(inputs, target, rate=0.1)
    assert guarded.updated and original.updated and guarded.accepted_rate == 0.1
    for key, value in fixed.snapshot().items():
        assert_array_equal(net.snapshot()[key], value)


@pytest.mark.parametrize("zero_rate", [False, True])
def test_rejection_preserves_weights_revisions_counts_but_advances_valid_free(zero_rate):
    net = TemporalPatchNet(1, 2, 1, seed=1061)
    inputs = np.zeros((1, 3, 1))
    target = np.ones((1, 3, 1)) if zero_rate else np.zeros((1, 3, 1))
    before = net.snapshot()
    result = net.observe(inputs, target, rate=0.0 if zero_rate else 1.0, backtrack=True)
    assert not result.updated and result.reason == "no_decreasing_parameter_step"
    assert result.accepted_rate == 0 and result.replay_calls == 0
    assert result.initial_loss == result.final_loss
    for key in ("A", "B", "C"):
        assert_array_equal(before[key], net.parameters()[key])
    assert net.updates == 0 and net.readback().parameter_revision == 0
    assert_array_equal(net.state, result.free.final_state)


def test_failed_phases_are_not_relabelled_as_parameter_rejection():
    net = TemporalPatchNet(1, 2, 1, seed=1063, max_iterations=0)
    before = net.parameters()
    result = net.observe(np.ones((1, 3, 1)), np.ones((1, 3, 1)), backtrack=True)
    assert not result.updated and result.reason == "phase_failed"
    assert result.replay_calls == 0 and result.initial_loss is None
    assert net.updates == 0
    for key in before:
        assert_array_equal(before[key], net.parameters()[key])
    assert_array_equal(net.state, result.free.final_state)


def test_exhausted_search_cannot_commit_an_unusable_finite_rate():
    net = TemporalPatchNet(1, 2, 1, seed=1087)
    before = net.parameters()
    result = net.observe(
        np.full((1, 3, 1), 0.2),
        np.full((1, 3, 1), 0.1),
        rate=1e100,
        backtrack=True,
    )
    assert not result.updated and result.reason == "no_decreasing_parameter_step"
    assert len(result.replay_losses) == result.replay_calls == 16
    assert result.accepted_rate == 0 and result.final_loss == result.initial_loss
    assert net.updates == net.readback().parameter_revision == 0
    for key in before:
        assert_array_equal(before[key], net.parameters()[key])
    assert_array_equal(net.state, result.free.final_state)


@pytest.mark.parametrize("invalid", [1, None, "yes"])
def test_invalid_option_leaves_complete_state_unchanged(invalid):
    net = TemporalPatchNet(1, 2, 1, seed=1069)
    before = net.snapshot()
    with pytest.raises(ValueError, match="backtrack"):
        net.observe(np.ones((1, 3, 1)), np.ones((1, 3, 1)), backtrack=invalid)
    for key in before:
        assert_array_equal(before[key], net.snapshot()[key])
