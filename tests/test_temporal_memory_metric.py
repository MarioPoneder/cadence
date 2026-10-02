"""Constrained EP metric parity, causal acceptance, and transaction boundaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cadence import TemporalMemory, TemporalPatchNet

FIXTURES = Path(__file__).parent / "fixtures" / "temporal_metric"


def same(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def fixture(name):
    with np.load(FIXTURES / name, allow_pickle=False) as data:
        net = TemporalPatchNet.restore({k[4:]: data[k] for k in data.files if k.startswith("net_")})
        memory = TemporalMemory.restore({k[7:]: data[k] for k in data.files if k.startswith("memory_")})
        values = {k: data[k].copy() for k in data.files if not k.startswith(("net_", "memory_"))}
    net.reset()
    return net, memory, values


def test_opt_in_matches_frozen_accepted_metric_update_without_old_paths():
    net, memory, data = fixture("accepted.npz")
    updates = net.updates
    result = memory.observe(net, data["inputs"], data["target"], beta=.001, rate=1., readout_damping=1e-4)
    assert result.updated and net.updates == updates + 1
    for key, value in net.parameters().items():
        np.testing.assert_allclose(value, data["expected_" + key], rtol=2e-11, atol=2e-11)
    assert result.delta is not None
    np.testing.assert_array_equal(net.state, result.free.final_state)
    assert memory.snapshot().keys() == {"basis_A", "basis_B", "basis_C", "relative_tolerance", "binding"}


def test_real_far_negative_branch_is_rejected_and_only_free_state_is_carried():
    net, memory, data = fixture("branch.npz")
    expected = TemporalPatchNet.restore(net.snapshot())
    expected.advance(data["inputs"])
    before = memory.snapshot()
    result = memory.observe(
        net, data["inputs"], data["target"], beta=.1, rate=1., readout_damping=1e-4,
        symmetry_tolerance=np.inf,
    )
    assert not result.updated and result.reason == "metric_step_rejected"
    assert result.plus.converged and result.minus.converged
    assert np.linalg.norm(result.minus.hidden - result.free.hidden) > 1.
    same(net.snapshot(), expected.snapshot())
    same(memory.snapshot(), before)
    # The symmetry check halves beta away from the far branch before any candidate is built.
    net, memory, data = fixture("branch.npz")
    centered = memory.observe(net, data["inputs"], data["target"], beta=.1, rate=1., readout_damping=1e-4)
    assert centered.updated and centered.contrast_halvings == 2 and centered.beta == .025
    assert np.linalg.norm(centered.minus.hidden - centered.free.hidden) < 1.


@pytest.mark.parametrize("damping", [0., -1., np.inf, -np.inf, np.nan])
def test_invalid_metric_setting_changes_neither_object(damping):
    net = TemporalPatchNet(1, 3, 1)
    memory = TemporalMemory()
    before, protection = net.snapshot(), memory.snapshot()
    with pytest.raises(ValueError, match="damping"):
        memory.observe(net, np.ones((1, 1, 1)), np.ones((1, 1, 1)), readout_damping=damping)
    same(net.snapshot(), before)
    same(memory.snapshot(), protection)


def test_default_none_is_exact_legacy_projected_update():
    net = TemporalPatchNet(2, 6, 2, seed=113)
    memory = TemporalMemory()
    inputs = np.array([[[.2, -.1], [.1, .3]]])
    target = np.full((1, 2, 2), .1)
    expected = TemporalPatchNet.restore(net.snapshot())
    binding = TemporalMemory.restore(memory.snapshot())
    before = net.parameters()
    result = expected.observe(inputs, target)
    expected.set_parameters(binding.project(before, expected.parameters()))
    actual = memory.observe(net, inputs, target, readout_damping=None)
    assert result.updated and actual.updated
    same(net.snapshot(), expected.snapshot())
    same(memory.snapshot(), binding.snapshot())


def test_weighted_nonzero_boundary_replay_and_cold_pair_continuation(tmp_path):
    net = TemporalPatchNet(2, 5, 3, seed=131, output_precision=np.array([.2, 2., 5.]))
    memory = TemporalMemory()
    net.advance(np.array([[[.3, -.2]]]))
    boundary = net.state.copy()
    inputs = np.array([[[.1, .2], [.0, .1]]])
    target = np.array([[[.1, -.2, .2], [.2, .1, -.1]]])
    before = net.imagine(inputs, state=boundary).output
    result = memory.observe(net, inputs, target, beta=.001, rate=.01, readout_damping=.01)
    assert result.updated
    after = net.imagine(inputs, state=boundary).output
    assert np.mean(net.output_precision * (after - target)**2) < np.mean(net.output_precision * (before - target)**2)
    np.testing.assert_array_equal(net.state, result.free.final_state)
    net.save(tmp_path / "net.npz")
    np.savez(tmp_path / "memory.npz", **memory.snapshot())
    other = TemporalPatchNet.load(tmp_path / "net.npz")
    with np.load(tmp_path / "memory.npz", allow_pickle=False) as saved:
        restored = TemporalMemory.restore(dict(saved))
    a = memory.observe(net, inputs, target, beta=.001, rate=.01, readout_damping=.01)
    b = restored.observe(other, inputs, target, beta=.001, rate=.01, readout_damping=.01)
    assert a.updated == b.updated and a.reason == b.reason
    same(net.snapshot(), other.snapshot())
    same(memory.snapshot(), restored.snapshot())


def test_zero_metric_step_is_rejection_not_phase_failure_or_update():
    net = TemporalPatchNet(1, 2, 1)
    net.set_parameters({k: np.zeros_like(v) for k, v in net.parameters().items()})
    memory = TemporalMemory()
    snapshot = memory.snapshot()
    result = memory.observe(net, np.ones((1, 1, 1)), np.ones((1, 1, 1)), readout_damping=.01)
    assert not result.updated and result.reason == "metric_step_rejected"
    assert all(p.converged for p in (result.free, result.plus, result.minus))
    assert net.updates == 0
    same(memory.snapshot(), snapshot)


def test_injected_projection_error_keeps_transaction_unchanged(monkeypatch):
    net = TemporalPatchNet(1, 3, 1)
    memory = TemporalMemory()
    before, protection = net.snapshot(), memory.snapshot()

    def fail(*args, **kwargs):
        raise ArithmeticError("deliberate projection failure")

    monkeypatch.setattr(TemporalMemory, "project", fail)
    with pytest.raises(ArithmeticError, match="deliberate"):
        memory.observe(net, np.ones((1, 1, 1)), np.ones((1, 1, 1)), readout_damping=.01)
    same(net.snapshot(), before)
    same(memory.snapshot(), protection)


@pytest.mark.parametrize("new_output,accepted", [([1., -2.], True), ([-.1, 1.], False)])
def test_acceptance_uses_weighted_loss_when_unweighted_loss_disagrees(monkeypatch, new_output, accepted):
    net = TemporalPatchNet(1, 3, 2, output_precision=np.array([100., .01]))
    weights = net.parameters()
    weights["C"][:] = 0.
    net.set_parameters(weights)
    memory = TemporalMemory()
    original_imagine = TemporalPatchNet.imagine

    def controlled_replay(self, inputs, *, state=None):
        phase = original_imagine(self, inputs, state=state)
        # A synthetic readback isolates the acceptance calculation. Real
        # weighted recurrence/learning and nonzero boundaries are tested above.
        return replace(phase, output=np.asarray(new_output)[None, None])

    monkeypatch.setattr(TemporalPatchNet, "imagine", controlled_replay)
    result = memory.observe(net, np.ones((1, 1, 1)), np.ones((1, 1, 2)),
                            beta=.001, rate=.01, readout_damping=.01)
    assert result.updated == accepted
    weighted = np.mean(net.output_precision * (np.asarray(new_output) - 1.)**2)
    assert (weighted < np.mean(net.output_precision)) == accepted
    assert (np.mean((np.asarray(new_output) - 1.)**2) < 1.) != accepted


def test_projection_residual_violation_rejects_before_replay(monkeypatch):
    net = TemporalPatchNet(1, 3, 1)
    memory = TemporalMemory()
    inputs = np.ones((1, 1, 1))
    memory.protect(net, inputs)
    expected = TemporalPatchNet.restore(net.snapshot())
    expected.advance(inputs)
    before = memory.snapshot()
    original_project = TemporalMemory.project

    def corrupt(self, old, proposed):
        result = original_project(self, old, proposed)
        result["B"] += 1e-4  # This input direction is fully protected.
        return result

    monkeypatch.setattr(TemporalMemory, "project", corrupt)
    result = memory.observe(net, inputs, np.ones((1, 1, 1)), readout_damping=.01)
    assert not result.updated and result.reason == "metric_step_rejected"
    same(net.snapshot(), expected.snapshot())
    same(memory.snapshot(), before)
