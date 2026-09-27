"""Joint record-port continuation and failure atomicity, without equilibrium claims."""

import json

import numpy as np
import pytest

from cadence import JointRecordPatches, Port
from cadence.record_ports import build


def zero_joint(count=2, **kwargs):
    joint = build([1] * count, [1] * count, [1] * count, [], seed=0,
                  cells=2, active=1, **kwargs)
    for net in joint.cortices:
        params = net.parameters()
        for value in params.values():
            value.fill(0)
        net.set_parameters(params)
    return joint


def assert_snapshot(joint, expected):
    actual = joint.snapshot()
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_cut_snapshot_save_and_restore_preserve_exact_training_continuation(tmp_path):
    joint = build([2, 2], [1, 1], [1, 1], [Port(0, 1, 0, 2)],
                  seed=3, rounds=3, damping=.5, cells=8, active=2)
    joint.cut = True
    inputs = [np.ones((2, 3, 1)), np.full((2, 3, 1), .4)]
    targets = [np.full((2, 3, 1), .2), np.full((2, 3, 1), .1)]
    joint.observe(inputs, targets, rate=.01)
    saved = joint.snapshot()
    path = joint.save(tmp_path / "joint")
    for twin in (JointRecordPatches.restore(saved), JointRecordPatches.load(path), joint.clone()):
        assert twin.cut is True
        assert_snapshot(twin, saved)
        twin.observe(inputs, targets, rate=.01)
        expected = JointRecordPatches.restore(saved)
        expected.cut = True  # Explicit intended ablation, independent of restore default.
        expected.observe(inputs, targets, rate=.01)
        assert_snapshot(twin, expected.snapshot())


def test_legacy_snapshot_without_cut_remains_connected():
    saved = zero_joint().snapshot()
    meta = json.loads(str(saved["meta"]))
    del meta["cut"]
    saved["meta"] = np.array(json.dumps(meta))
    assert JointRecordPatches.restore(saved).cut is False


@pytest.mark.parametrize("field,value", [
    ("cut", "false"), ("rounds", 1.5), ("rounds", True),
    ("cortices", 2.5), ("damping", np.nan), ("cross_adjoint", "false"),
    ("own", [-1, 1]), ("own", [1.2, 1]),
])
def test_restore_rejects_malformed_metadata(field, value):
    saved = zero_joint().snapshot()
    meta = json.loads(str(saved["meta"]))
    meta[field] = value
    saved["meta"] = np.array(json.dumps(meta))
    with pytest.raises(ValueError):
        JointRecordPatches.restore(saved)


@pytest.mark.parametrize("args", [(0.5, 1, 0, 1), (0, True, 0, 1), (0, 1, -1, 1), (0, 1, 0, 1.5)])
def test_port_indices_are_not_silently_truncated(args):
    with pytest.raises(ValueError):
        Port(*args)


@pytest.mark.parametrize("size", [1, 3])
def test_target_and_state_counts_are_exact(size):
    joint = zero_joint()
    before = joint.snapshot()
    inputs = [np.ones((1, 1, 1))] * 2
    with pytest.raises(ValueError, match="target path"):
        joint.observe(inputs, [np.ones((1, 1, 1))] * size)
    with pytest.raises(ValueError, match="boundary state"):
        joint.imagine(inputs, states=[np.ones((1, 1))] * size)
    assert_snapshot(joint, before)


@pytest.mark.parametrize("options", [{"write": "false"}, {"backtrack": "false"}, {"rate": np.nan}])
def test_invalid_options_do_not_advance_or_write(options):
    joint = zero_joint()
    before = joint.snapshot()
    with pytest.raises(ValueError):
        joint.observe([np.ones((1, 1, 1))] * 2, [np.ones((1, 1, 1))] * 2, **options)
    assert_snapshot(joint, before)


def test_later_nonfinite_prediction_cannot_advance_any_cortex():
    joint = zero_joint()
    net = joint.cortices[1]
    params = net.parameters()
    params["c"].fill(1e308)
    net.set_parameters(params)
    net.records.projection.fill(0)
    net.records.offset.fill(1)
    net.records.tables["y"].fill(1e308)
    before = joint.snapshot()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="joint path must be finite"):
            joint.advance([np.ones((1, 1, 1))] * 2)
    assert_snapshot(joint, before)


def test_unrepresentable_later_input_norm_rejects_before_any_write():
    joint = build([1, 1], [1, 2], [1, 1], [], seed=0, cells=2, active=1)
    for net in joint.cortices:
        params = net.parameters()
        for value in params.values():
            value.fill(0)
        net.set_parameters(params)
    joint.cortices[1]._input_norm = 1e308
    before = joint.snapshot()
    with pytest.raises(ValueError, match="input norms"):
        joint.observe([np.ones((1, 1, 1)), np.full((1, 1, 2), 1.3e308)],
                      [np.ones((1, 1, 1))] * 2)
    assert_snapshot(joint, before)


def test_sum_of_finite_losses_cannot_admit_overflow():
    joint = zero_joint(3)
    for net in joint.cortices:
        net.set_output_precision(np.array([1.7]))
    before = joint.snapshot()
    result = joint.observe([np.ones((1, 1, 1))] * 3, [np.full((1, 1, 1), 1e154)] * 3)
    assert all(np.isfinite(path.loss) for path in result.settled.paths)
    assert result.settled.loss is None
    assert not result.updated and result.reason == "nonfinite_prediction"
    assert_snapshot(joint, before)


def test_later_record_overflow_rolls_back_every_write_and_context():
    joint = zero_joint(record_rate=2, habituation=1)
    # Witnessing changes the winner from cell 0 to cell 1. The latter is unused
    # by the finite initial prediction but holds an extreme finite old record.
    for net in joint.cortices:
        net.records.projection.fill(0)
        net.records.projection[0] = [1, -1]
        net.records.offset.fill(1)
    joint.cortices[1].records.tables["y"][:, 0] = [0, 1e308]
    before = joint.snapshot()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="record write would produce nonfinite state"):
            joint.observe([np.ones((1, 1, 1))] * 2,
                          [np.ones((1, 1, 1)), np.zeros((1, 1, 1))])
    assert_snapshot(joint, before)


def test_backtracking_rejects_finite_parameters_with_overflowing_trial_drives():
    joint = zero_joint(1)
    net = joint.cortices[0]
    params = net.parameters()
    params["C"].fill(1)
    net.set_parameters(params)
    net._input_norm = 1e200
    original = net.parameters()
    # The finite gradient has norm about 5e99. Each finite proposed B still
    # overflows B*x, so no trial is a valid decreasing computation.
    with np.errstate(over="ignore", invalid="ignore"):
        result = joint.observe([np.full((1, 1, 1), 1e200)],
                               [np.full((1, 1, 1), 1e-100)],
                               rate=1e150, backtrack=True, write=False)
    assert not result.updated
    assert result.replay_calls == 16
    for key, value in net.parameters().items():
        np.testing.assert_array_equal(value, original[key])
    assert np.isfinite(net.state).all()


def test_zero_rate_retains_activity_and_records_without_a_parameter_update():
    joint = zero_joint()
    for net in joint.cortices:
        params = net.parameters()
        params["b"].fill(.5)
        net.set_parameters(params)
        net.records.projection.fill(0)
        net.records.offset.fill(1)
    original = joint.parameters()
    revisions = [net._revision for net in joint.cortices]
    updates = [net.updates for net in joint.cortices]
    result = joint.observe([np.ones((1, 1, 1))] * 2,
                           [np.ones((1, 1, 1))] * 2, rate=0)
    assert not result.updated and result.reason == "no_step"
    assert result.accepted_rate == 0 and result.replay_calls == 0
    assert result.initial_loss == result.final_loss
    assert result.writes == 2 and result.delta is not None
    for i, net in enumerate(joint.cortices):
        for key, value in net.parameters().items():
            np.testing.assert_array_equal(value, original[i][key])
        assert net.updates == updates[i] and net._revision == revisions[i]
        np.testing.assert_array_equal(net.state, result.settled.paths[i].final_state)
        assert np.any(net.state != 0)
        assert net.records.writes == 1 and np.any(net.records.tables["y"] != 0)


def test_failed_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    joint = zero_joint()
    path = joint.save(tmp_path / "joint")
    original = path.read_bytes()

    def fail(handle, **arrays):
        handle.write(b"partial replacement")
        raise OSError("injected storage failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="storage failure"):
        joint.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
