"""Focused controls for a fixed parameter subspace of the same temporal energy."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cadence import TemporalMemory, TemporalPatchNet
from cadence.experimental import PartitionedTemporalPatchNet, two_group_masks


def same_snapshot(left, right, *, ignore_masks=False):
    assert left.keys() == right.keys()
    for key in left:
        if key == "meta" and ignore_masks:
            a, b = json.loads(str(left[key])), json.loads(str(right[key]))
            a.pop("partitioned_temporal", None)
            b.pop("partitioned_temporal", None)
            assert a == b
        else:
            np.testing.assert_array_equal(left[key], right[key])


def fixture():
    masks = two_group_masks(2, 2, 3, 1, context_inputs=[0, 1], motor_inputs=[2])
    return PartitionedTemporalPatchNet(
        3, 4, 1, masks=masks, seed=41, initial_radius=0.35, tolerance=1e-12
    )


def data():
    rng = np.random.default_rng(919)
    return rng.normal(size=(2, 4, 3)) * 0.2, rng.normal(size=(2, 4, 1)) * 0.15


@pytest.mark.parametrize("backtrack", [False, True])
def test_dense_all_true_exact_numerical_parity(backtrack):
    dense = TemporalPatchNet(3, 4, 1, seed=41, initial_radius=0.35, tolerance=1e-12)
    partition = PartitionedTemporalPatchNet(3, 4, 1, seed=41, initial_radius=0.35, tolerance=1e-12)
    u, y = data()
    dense.advance(u[:, :1])
    partition.advance(u[:, :1])
    left = dense.observe(u, y, beta=0.001, rate=3.0, backtrack=backtrack)
    right = partition.observe(u, y, beta=0.001, rate=3.0, backtrack=backtrack)
    assert left.updated == right.updated and left.reason == right.reason
    assert left.accepted_rate == right.accepted_rate and left.replay_losses == right.replay_losses
    for a, b in zip(
        (left.free, left.plus, left.minus), (right.free, right.plus, right.minus), strict=True
    ):
        np.testing.assert_array_equal(a.hidden, b.hidden)
        np.testing.assert_array_equal(a.output, b.output)
        assert a.energy_history == b.energy_history and a.residual == b.residual
    for key in left.delta:
        np.testing.assert_array_equal(left.delta[key], right.delta[key])
    same_snapshot(dense.snapshot(), partition.snapshot(), ignore_masks=True)


def test_projected_EP_matches_independent_finite_difference():
    net = fixture()
    u, y = data()
    net.advance(u[:, :1])
    before, boundary = net.parameters(), net.state.copy()
    beta = 0.0002
    observation = net.observe(u, y, beta=beta, rate=0.0)
    assert observation.updated
    rng = np.random.default_rng(929)
    direction = {
        key: rng.normal(size=value.shape) * net.masks[key] for key, value in before.items()
    }
    norm = np.sqrt(sum(np.sum(value * value) for value in direction.values()))
    direction = {key: value / norm for key, value in direction.items()}
    predicted = sum(float(np.sum(observation.delta[key] * direction[key])) for key in before)

    def cost(sign):
        parameters = {key: before[key] + sign * 1e-5 * direction[key] for key in before}
        h = boundary.copy()
        outputs = []
        for t in range(u.shape[1]):
            h = np.tanh(h) @ parameters["A"].T + u[:, t] @ parameters["B"].T
            outputs.append(np.tanh(h) @ parameters["C"].T)
        return 0.5 * np.mean((np.stack(outputs, axis=1) - y) ** 2)

    derivative = (cost(1) - cost(-1)) / 2e-5
    np.testing.assert_allclose(predicted, derivative, rtol=3e-5, atol=2e-7)
    for key, mask in net.masks.items():
        assert np.all(observation.delta[key][~mask] == 0)
        assert np.all(net.parameters()[key][~mask] == 0)


def test_mask_precedes_acceptance_and_only_legal_step_is_committed():
    net = fixture()
    u, y = data()
    net.advance(u[:, :1])
    before, boundary = net.parameters(), net.state.copy()
    result = net.observe(u, y, beta=0.001, rate=128.0, backtrack=True)
    assert result.updated and result.final_loss < result.initial_loss
    for key in before:
        np.testing.assert_array_equal(
            net.parameters()[key], before[key] - result.accepted_rate * result.delta[key]
        )
        assert np.all(net.parameters()[key][~net.masks[key]] == 0)
    actual = net.imagine(u, state=boundary)
    assert result.final_loss == 0.5 * np.mean((actual.output - y) ** 2)
    np.testing.assert_array_equal(net.state, result.free.final_state)


def test_checkpoint_restores_masks_and_exact_continuation(tmp_path):
    net = fixture()
    u, y = data()
    assert net.observe(u, y, beta=0.001, rate=1.0, backtrack=True).updated
    net.save(tmp_path / "net.npz")
    restored = PartitionedTemporalPatchNet.load(tmp_path / "net.npz")
    same_snapshot(net.snapshot(), restored.snapshot())
    np.testing.assert_array_equal(net.imagine(u).output, restored.imagine(u).output)
    assert restored.trainable_parameter_count == sum(
        np.count_nonzero(m) for m in net.masks.values()
    )
    for key in net.masks:
        np.testing.assert_array_equal(net.masks[key], restored.masks[key])
    exposed = restored.masks
    exposed["B"][:] = True
    assert not restored.masks["B"].all()
    before = restored.snapshot()
    params = restored.parameters()
    params["B"][0, 2] = 1.0
    with pytest.raises(ValueError):
        restored.set_parameters(params)
    same_snapshot(before, restored.snapshot())


def test_invalid_masks_or_snapshots_are_rejected():
    net = fixture()
    for bad in (
        {"A": np.ones((4, 4), dtype=bool)},
        {**net.masks, "B": np.ones((4, 3))},
        {**net.masks, "C": np.ones((1, 3), dtype=bool)},
    ):
        with pytest.raises(ValueError):
            PartitionedTemporalPatchNet(3, 4, 1, masks=bad)
    snapshot = net.snapshot()
    snapshot["B"][0, 2] = 1.0
    with pytest.raises(ValueError):
        PartitionedTemporalPatchNet.restore(snapshot)
    with pytest.raises(ValueError):
        PartitionedTemporalPatchNet.restore(TemporalPatchNet(3, 4, 1).snapshot())
    with pytest.raises(ValueError):
        two_group_masks(2, 2, 3, 1, context_inputs=[3], motor_inputs=[2])


def test_failed_detuning_keeps_weights_masks_and_revision(monkeypatch):
    net = fixture()
    u, y = data()
    before = net.parameters()
    readback = net.readback()
    original = TemporalPatchNet._solve

    def fail_positive(self, *args, **kwargs):
        phase = original(self, *args, **kwargs)
        beta = args[3] if len(args) > 3 else kwargs.get("beta", 0.0)
        return replace(phase, converged=False, reason="injected_failure") if beta > 0 else phase

    monkeypatch.setattr(TemporalPatchNet, "_solve", fail_positive)
    result = net.observe(u, y, beta=0.001, rate=1.0, backtrack=True)
    assert not result.updated and result.reason == "phase_failed"
    for key in before:
        np.testing.assert_array_equal(before[key], net.parameters()[key])
    assert net.readback().parameter_revision == readback.parameter_revision
    assert net.updates == readback.updates
    np.testing.assert_array_equal(net.state, result.free.final_state)
    same_snapshot(net.snapshot(), PartitionedTemporalPatchNet.restore(net.snapshot()).snapshot())


def test_rejected_parameter_trials_do_not_commit(monkeypatch):
    net = fixture()
    u, y = data()
    before = net.parameters()
    rb = net.readback()
    original = TemporalPatchNet.imagine

    def reject_private_replay(self, *args, **kwargs):
        phase = original(self, *args, **kwargs)
        return replace(phase, converged=False, reason="injected_replay_failure")

    monkeypatch.setattr(TemporalPatchNet, "imagine", reject_private_replay)
    result = net.observe(u, y, beta=0.001, rate=1.0, backtrack=True)
    assert not result.updated and result.reason == "no_decreasing_parameter_step"
    assert result.replay_calls == 16
    for key in before:
        np.testing.assert_array_equal(before[key], net.parameters()[key])
    assert net.updates == rb.updates and net.readback().parameter_revision == rb.parameter_revision
    np.testing.assert_array_equal(net.state, result.free.final_state)


def test_delayed_cross_group_coupling_carries_cue_and_credit():
    masks = two_group_masks(1, 1, 2, 1, context_inputs=[0], motor_inputs=[1])
    net = PartitionedTemporalPatchNet(2, 2, 1, masks=masks, seed=41, tolerance=1e-12)
    parameters = dict(
        A=np.array([[0.2, 0.1], [0.8, 0.1]]),
        B=np.array([[0.7, 0.0], [0.0, 0.3]]),
        C=np.array([[0.0, 0.9]]),
    )
    net.set_parameters(parameters)
    u = np.array([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    prediction = net.imagine(u).output
    disconnected = {key: value.copy() for key, value in parameters.items()}
    disconnected["A"][1, 0] = 0.0
    no_link = PartitionedTemporalPatchNet(2, 2, 1, masks=masks)
    no_link.set_parameters(disconnected)
    assert prediction[0, 1, 0] > 0.3
    np.testing.assert_array_equal(no_link.imagine(u).output, np.zeros((1, 3, 1)))
    observed = net.observe(u, np.zeros_like(prediction), beta=0.0002, rate=0.0)
    assert observed.updated and abs(observed.delta["A"][1, 0]) > 1e-3
    assert abs(observed.delta["B"][0, 0]) > 1e-3
    assert observed.delta["B"][1, 0] == 0 and observed.delta["C"][0, 0] == 0


def test_joint_free_path_equals_explicit_two_component_recurrence():
    net = fixture()
    u, _ = data()
    parameters = net.parameters()
    boundary = np.random.default_rng(937).normal(size=(2, 4)) * 0.2
    context, motor = boundary[:, :2].copy(), boundary[:, 2:].copy()
    hidden, output = [], []
    a, b, c = [parameters[key] for key in ("A", "B", "C")]
    for t in range(u.shape[1]):
        # Both components read the preceding joint state, not a newly written
        # same-time neighbor. This is exactly the delayed block factorization.
        old_context, old_motor = np.tanh(context), np.tanh(motor)
        context = old_context @ a[:2, :2].T + old_motor @ a[:2, 2:].T + u[:, t] @ b[:2].T
        motor = old_context @ a[2:, :2].T + old_motor @ a[2:, 2:].T + u[:, t] @ b[2:].T
        hidden.append(np.concatenate((context, motor), axis=1))
        output.append(np.tanh(motor) @ c[:, 2:].T)
    phase = net.imagine(u, state=boundary)
    np.testing.assert_allclose(phase.hidden, np.stack(hidden, axis=1), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(phase.output, np.stack(output, axis=1), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("damping", [None, 1e-4])
def test_protected_learning_rejects_subclass_without_mutating_either_object(damping):
    net = fixture()
    inputs, target = data()
    net.advance(inputs[:, :1])
    memory = TemporalMemory()
    # Collecting bases is permitted, but is not a promise of a compatible
    # protected parameter transaction.
    memory.protect(net, inputs)
    net_before, memory_before = net.snapshot(), memory.snapshot()
    with pytest.raises(TypeError, match="requires TemporalPatchNet"):
        memory.observe(net, inputs, target, readout_damping=damping)
    same_snapshot(net_before, net.snapshot())
    same_snapshot(memory_before, memory.snapshot())


def test_private_planning_after_checkpoint_keeps_routing_and_live_state(tmp_path):
    net = fixture()
    inputs, _ = data()
    net.advance(inputs[:, :1])
    net.save(tmp_path / "partitioned.npz")
    restored = PartitionedTemporalPatchNet.load(tmp_path / "partitioned.npz")
    before = restored.snapshot()
    goal = restored.imagine(inputs).output + 0.025
    mask = np.zeros_like(inputs, dtype=bool)
    mask[:, :, 2] = True
    result = restored.plan(
        inputs, goal=goal, controls=mask, bounds=(-1.0, 1.0), beta=0.001, max_steps=4
    )
    assert result.improved
    np.testing.assert_array_equal(result.inputs[~mask], inputs[~mask])
    assert np.all(np.abs(result.inputs[mask]) <= 1)
    np.testing.assert_array_equal(result.prediction.output, restored.imagine(result.inputs).output)
    same_snapshot(before, restored.snapshot())
    same_snapshot(net.snapshot(), restored.snapshot())
    for key, allowed in restored.masks.items():
        assert np.all(restored.parameters()[key][~allowed] == 0)


def test_real_pilot_step_matches_frozen_helper_endpoints_and_checkpoint():
    directory = Path(__file__).parent / "fixtures/partitioned_phrase"
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, expected in provenance["artifacts"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    with np.load(directory / "1103_partitioned_initial.npz", allow_pickle=False) as archive:
        initial = {key: archive[key].copy() for key in archive.files}
    with np.load(directory / "1103_partitioned_step0.npz", allow_pickle=False) as archive:
        witness = {key: archive[key].copy() for key in archive.files}
    ledger = json.loads((directory / "step0.json").read_text())
    inputs = np.zeros((4, 33, 11))
    inputs[:, 0, :3] = [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]
    target = np.zeros((4, 33, 36))
    for row in range(4):
        for index in range(32):
            inputs[row, index + 1, 3 + index % 8] = 1
            target[row, index + 1, (index + 8 * row) % 32] = 1
            target[row, index + 1, [32, 34, 35]] = 1
    net = PartitionedTemporalPatchNet.restore(initial)
    result = net.observe(inputs, target, beta=0.01, rate=8.0, backtrack=True)
    assert result.updated == ledger["updated"] and result.reason == ledger["reason"]
    assert result.accepted_rate == ledger["accepted_rate"]
    assert result.replay_calls == ledger["replay_calls"]
    np.testing.assert_allclose(result.replay_losses, ledger["replay_losses"], atol=1e-11, rtol=1e-9)
    assert result.delta is not None
    for key in "ABC":
        np.testing.assert_allclose(
            result.delta[key], witness["delta_" + key], atol=1e-11, rtol=1e-9
        )
        np.testing.assert_allclose(
            net.parameters()[key], witness["after_" + key], atol=1e-11, rtol=1e-9
        )
    for name in ("free", "plus", "minus"):
        phase = getattr(result, name)
        assert phase is not None and phase.converged
        np.testing.assert_allclose(phase.hidden, witness[name + "_hidden"], atol=1e-11, rtol=1e-9)
        np.testing.assert_allclose(phase.output, witness[name + "_output"], atol=1e-11, rtol=1e-9)
    actual = net.snapshot()
    for key in ("state", "output_precision"):
        np.testing.assert_allclose(actual[key], witness["after_" + key], atol=1e-11, rtol=1e-9)
    expected_meta = json.loads(str(witness["after_meta"]))
    actual_meta = json.loads(str(actual["meta"]))
    for key in (
        "updates",
        "parameter_revision",
        "state_parameter_revision",
        "partitioned_temporal",
    ):
        assert actual_meta[key] == expected_meta[key]
