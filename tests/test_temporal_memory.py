"""Protected-path invariance and remaining plasticity are separate obligations."""

import numpy as np
import pytest

from cadence import TemporalPatchNet
from cadence.temporal_memory import TemporalMemory


def test_projection_preserves_path_but_allows_a_different_input():
    rng = np.random.default_rng(113)
    net = TemporalPatchNet(4, 12, 3, seed=127)
    inputs = rng.normal(size=(1, 5, 4)) * 0.2
    cold = np.zeros((1, 12))
    old = net.imagine(inputs, state=cold).output
    memory = TemporalMemory()
    report = memory.protect(net, inputs, state=cold)
    assert report.ranks["A"] <= 4 and report.ranks["C"] <= 5
    assert report.maximum_residual < 1e-14
    different = rng.normal(size=(1, 5, 4)) * 0.2
    before_other = net.imagine(different, state=cold).output
    for _ in range(20):
        before = net.parameters()
        proposal = {k: v + rng.normal(size=v.shape) * 0.02 for k, v in before.items()}
        net.set_parameters(memory.project(before, proposal))
    np.testing.assert_allclose(net.imagine(inputs, state=cold).output, old, atol=3e-15)
    assert np.max(np.abs(net.imagine(different, state=cold).output - before_other)) > 1e-4
    resumed = TemporalMemory.restore(memory.snapshot())
    assert resumed.report().ranks == report.ranks
    before = net.parameters()
    proposal = {k: v + rng.normal(size=v.shape) * 0.02 for k, v in before.items()}
    restored = resumed.project(before, proposal)
    for k, expected in memory.project(before, proposal).items():
        np.testing.assert_array_equal(restored[k], expected)


def test_projection_does_not_protect_unselected_queries_or_override_other_updates():
    net = TemporalPatchNet(2, 4, 1, seed=139)
    inputs = np.array([[[1.0, 0.0]]])
    memory = TemporalMemory()
    memory.protect(net, inputs)
    before = net.parameters()
    changed = {k: v + 0.01 for k, v in before.items()}
    with pytest.raises(ValueError, match="outside"):
        memory.project(changed, before)
    np.testing.assert_array_equal(memory.snapshot()["binding"], TemporalMemory._fingerprint(before))
    bad = {**changed, "C": np.full_like(before["C"], np.nan)}
    with pytest.raises(ValueError, match="finite"):
        memory.project(before, bad)
    np.testing.assert_array_equal(memory.snapshot()["binding"], TemporalMemory._fingerprint(before))


def test_protected_projection_keeps_actual_old_skill_while_new_ep_loss_falls():
    net = TemporalPatchNet(2, 8, 1, seed=151)
    old_input, new_input = np.array([[[1.0, 0.0]]]), np.array([[[0.0, 1.0]]])
    memory = TemporalMemory()
    old_output = net.imagine(old_input).output
    memory.protect(net, old_input)
    new_target = np.full((1, 1, 1), 0.25)
    before_loss = float(np.square(net.imagine(new_input).output - new_target).mean())
    for _ in range(80):
        net.reset()
        result = memory.observe(net, new_input, new_target, beta=0.01, rate=0.1)
        assert result.updated
    cold = np.zeros((1, 8))
    np.testing.assert_allclose(net.imagine(old_input, state=cold).output, old_output, atol=1e-14)
    after_loss = float(np.square(net.imagine(new_input, state=cold).output - new_target).mean())
    assert after_loss < before_loss * 0.01


def same_snapshot(left, right):
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_atomic_projection_error_rolls_back_network_and_memory(monkeypatch):
    net = TemporalPatchNet(2, 4, 1)
    memory = TemporalMemory()
    inputs = np.array([[[1.0, 0.0]]])
    memory.protect(net, inputs)
    before_net, before_memory = net.snapshot(), memory.snapshot()

    def rejected(*args):
        raise ArithmeticError("injected constraint failure")

    monkeypatch.setattr(TemporalMemory, "project", rejected)
    with pytest.raises(ArithmeticError):
        memory.observe(net, inputs, np.ones((1, 1, 1)))
    same_snapshot(before_net, net.snapshot())
    same_snapshot(before_memory, memory.snapshot())


def test_failed_phase_preserves_binding_and_only_carries_free_state():
    net = TemporalPatchNet(2, 4, 1, max_iterations=0)
    memory = TemporalMemory()
    inputs = np.array([[[1.0, 0.0]]])
    memory.protect(net, inputs)
    binding = memory.snapshot()
    before = net.parameters()
    expected = net.imagine(inputs).final_state
    result = memory.observe(net, inputs, np.ones((1, 1, 1)))
    assert not result.updated and net.updates == 0
    same_snapshot(binding, memory.snapshot())
    same_snapshot(before, net.parameters())
    np.testing.assert_array_equal(net.state, expected)


def test_new_large_scale_protection_never_discards_earlier_directions():
    net = TemporalPatchNet(2, 3, 1)
    memory = TemporalMemory()
    memory.protect(net, np.array([[[1.0, 0.0]]]))
    memory.protect(net, np.array([[[0.0, 1e14]]]))
    assert memory.report().ranks["B"] == 2
    before = net.parameters()
    proposal = {key: value + 0.1 for key, value in before.items()}
    result = memory.project(before, proposal)
    # Full-rank B has no remaining plastic directions.
    np.testing.assert_allclose(result["B"], before["B"], atol=1e-15)


def test_cold_network_and_memory_restore_preserves_atomic_learning():
    net = TemporalPatchNet(2, 4, 1)
    memory = TemporalMemory()
    protected = np.array([[[1.0, 0.0]]])
    new = np.array([[[0.0, 1.0]]])
    memory.protect(net, protected)
    assert memory.observe(net, new, np.ones((1, 1, 1))).updated
    other = TemporalPatchNet.restore(net.snapshot())
    recalled = TemporalMemory.restore(memory.snapshot())
    net.reset()
    other.reset()
    assert memory.observe(net, new, np.ones((1, 1, 1))).updated
    assert recalled.observe(other, new, np.ones((1, 1, 1))).updated
    same_snapshot(net.snapshot(), other.snapshot())
    same_snapshot(memory.snapshot(), recalled.snapshot())
    expected_bytes = (
        8 + 32 + sum(v.nbytes for k, v in memory.snapshot().items() if k.startswith("basis_"))
    )
    assert memory.report().bytes == expected_bytes


def test_unprotected_atomic_learning_can_be_snapshotted_and_invalid_input_is_atomic():
    net = TemporalPatchNet(1, 3, 1)
    memory = TemporalMemory()
    path = np.ones((1, 2, 1))
    assert memory.observe(net, path, path).updated
    restored = TemporalMemory.restore(memory.snapshot())
    assert restored.report().ranks == {"A": 0, "B": 0, "C": 0}
    before_net, before_memory = net.snapshot(), memory.snapshot()
    with pytest.raises(ValueError):
        memory.observe(net, path, path * np.nan)
    same_snapshot(net.snapshot(), before_net)
    same_snapshot(memory.snapshot(), before_memory)
