"""The record patch: an exact adjoint, its detuning identity, one-shot records and isolation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import RecordPatchNet


def same_snapshot(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert_array_equal(left[key], right[key])


def independent_path(parameters, inputs, boundary, net=None, read=None):
    """A per-row loop of the documented equations.

    The record read is taken from ``net`` (its records and reading scale), from a
    fixed ``read`` array, or is zero."""
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    batch, horizon, _ = inputs.shape
    hidden = np.zeros((batch, horizon, parameters["B"].shape[0]))
    outputs = []
    for row in range(batch):
        h = boundary[row].copy()
        for t in range(horizon):
            u = inputs[row, t]
            gate = sigmoid(parameters["g"] + parameters["G"] @ u)
            z = np.tanh(parameters["B"] @ u + parameters["b"])
            h = gate * h + (1.0 - gate) * z
            hidden[row, t] = h
            m = np.zeros(parameters["c"].shape)
            if net is not None:
                code = net.records.code(
                    np.concatenate((u * (np.sqrt(net.inputs) / net._input_norm), h * net._scale))
                )[0, 0]
                m = code @ net.records.tables["y"]
            elif read is not None:
                m = read[row, t]
            outputs.append(parameters["C"] @ h + parameters["c"] + m)
    return hidden, np.array(outputs).reshape(batch, horizon, -1)


def differences(function, point, step=1e-6):
    result = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        result[index] = (function(plus) - function(minus)) / (2 * step)
    return result


def test_free_path_matches_the_documented_recurrence_and_is_bounded():
    rng = np.random.default_rng(3)
    net = RecordPatchNet(3, 5, 2, seed=11, cells=200, active=8)
    inputs = rng.normal(size=(2, 7, 3)) * 3.0
    boundary = rng.uniform(-1, 1, size=(2, 5))
    path = net.imagine(inputs, state=boundary)
    hidden, output = independent_path(net.parameters(), inputs, boundary)
    assert_allclose(path.hidden, hidden, atol=1e-12)
    assert_allclose(path.output, output, atol=1e-12)  # empty records read zero
    assert np.all(np.abs(path.hidden) < 1.0)
    assert np.all((path.gate > 0) & (path.gate < 1))


def test_records_change_the_output_through_the_read_port_only():
    rng = np.random.default_rng(4)
    net = RecordPatchNet(3, 5, 2, seed=11, cells=300, active=8, record_rate=1.0)
    inputs, target = rng.normal(size=(1, 4, 3)), rng.normal(size=(1, 4, 2))
    net.observe(inputs, target, rate=0.0)
    boundary = np.zeros((1, 5))
    path = net.imagine(inputs, state=boundary)
    hidden, output = independent_path(net.parameters(), inputs, boundary, net=net)
    assert_allclose(path.hidden, hidden, atol=1e-12)
    assert_allclose(path.output, output, atol=1e-12)
    assert np.abs(path.read).max() > 0.0


@pytest.mark.parametrize("precision", [np.ones(2), np.array([0.3, 2.5])])
def test_adjoint_gradient_matches_central_differences(precision):
    rng = np.random.default_rng(5)
    net = RecordPatchNet(2, 3, 2, seed=8, cells=64, active=4, output_precision=precision)
    inputs, target = rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 2))
    net.observe(inputs, target, rate=0.0)  # fill a few records so the read port is live
    net.reset()
    parameters = net.parameters()
    result = net.observe(inputs, target, rate=0.0, write=False)
    assert not result.updated and result.reason == "no_step" and result.delta is not None
    assert np.abs(result.prediction.read).max() > 0.0  # the read is live, and outside the loss

    def loss(name, value):
        _, output = independent_path({**parameters, name: value}, inputs, np.zeros((2, 3)))
        return 0.5 * np.mean(precision * (output - target) ** 2)

    for name, value in parameters.items():
        numerical = differences(lambda v, name=name: loss(name, v), value)
        assert_allclose(result.delta[name], numerical, atol=1e-8, rtol=1e-6)


def test_detuned_contrast_equals_the_adjoint_gradient():
    rng = np.random.default_rng(6)
    net = RecordPatchNet(3, 6, 2, seed=9, cells=128, active=6, output_precision=[1.0, 2.0])
    inputs, target = rng.normal(size=(2, 9, 3)), rng.normal(size=(2, 9, 2))
    net.observe(inputs, target, rate=0.0)
    net.reset()
    adjoint = net.observe(inputs, target, rate=0.0, write=False).delta
    net.reset()
    before = net.snapshot()
    contrast = net.detune(inputs, target, beta=1e-4)
    assert contrast.converged and contrast.plus_energy > 0 and contrast.minus_energy < 0
    for name, value in adjoint.items():
        error = np.linalg.norm(contrast.contrast[name] - value) / np.linalg.norm(value)
        assert error < 1e-6, (name, error)
    coarse = net.detune(inputs, target, beta=1e-2)
    fine_error = max(
        np.linalg.norm(contrast.contrast[k] - adjoint[k]) / np.linalg.norm(adjoint[k])
        for k in adjoint
    )
    coarse_error = max(
        np.linalg.norm(coarse.contrast[k] - adjoint[k]) / np.linalg.norm(adjoint[k])
        for k in adjoint
    )
    assert coarse_error > fine_error  # the contrast error shrinks with beta
    same_snapshot(before, net.snapshot())  # detuning changed nothing


def test_one_observation_is_recalled_from_the_records():
    rng = np.random.default_rng(7)
    net = RecordPatchNet(17, 8, 6, seed=2, cells=4096, active=32, record_rate=1.0)
    inputs = np.zeros((1, 16, 17))
    inputs[0, 0, 0] = 1.0  # one cue
    inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0  # sixteen distinct heard events
    identities = rng.integers(6, size=16)
    target = np.eye(6)[identities][None]
    first = net.observe(inputs, target, rate=0.0)
    assert first.writes == 16 and first.initial_loss is not None
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
    # Keys of neighbouring moments overlap through the shared context, so one
    # write is not exact; the delta rule converges over repeated observation.
    assert net._loss(recalled.output, target) < 0.3 * first.initial_loss
    second = net.observe(inputs, target, rate=0.0)
    third = net.observe(inputs, target, rate=0.0)
    fourth = net.observe(inputs, target, rate=0.0)
    assert second.prediction.loss < 0.3 * first.prediction.loss
    assert third.prediction.loss < second.prediction.loss
    assert fourth.prediction.loss < 0.05 * first.prediction.loss
    for name, value in net.parameters().items():  # the slow readout did not move at rate 0
        assert_array_equal(value, RecordPatchNet(17, 8, 6, seed=2).parameters()[name])


def test_readings_that_differ_only_by_context_are_separated_only_partly():
    """Same clock phase, different bar: the leaky context alone separates the keys
    partly. This records the measured limit; a heard stream supplies the rest."""
    rng = np.random.default_rng(7)
    net = RecordPatchNet(4, 8, 6, seed=2, cells=4096, active=32, record_rate=1.0)
    inputs = np.zeros((1, 16, 4))
    inputs[0, 0, 0] = 1.0
    inputs[0, np.arange(16), 1 + np.arange(16) % 3] = 1.0  # a three-position clock
    target = np.eye(6)[rng.integers(6, size=16)][None]
    identities = target[0].argmax(axis=1)
    for _ in range(3):
        net.observe(inputs, target, rate=0.0)
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    correct = int((recalled.output[0].argmax(axis=1) == identities).sum())
    assert 5 <= correct <= 14


def test_the_slow_readout_learns_what_the_records_already_patch():
    """A second observation of the same path finds the record patching the error,
    yet the slow parameters keep learning it: their loss falls while the
    prediction's loss is already small."""
    rng = np.random.default_rng(12)
    net = RecordPatchNet(3, 6, 2, seed=3, cells=512, active=8, record_rate=1.0)
    inputs, target = rng.normal(size=(2, 8, 3)), rng.normal(size=(2, 8, 2))
    first = net.observe(inputs, target, rate=2.0, backtrack=True)
    net.reset()
    second = net.observe(inputs, target, rate=2.0, backtrack=True)
    net.reset()
    third = net.observe(inputs, target, rate=2.0, backtrack=True)
    assert second.prediction.loss < 0.7 * first.prediction.loss  # the record patched it
    assert third.prediction.loss < 0.5 * first.prediction.loss
    assert second.initial_loss < first.initial_loss  # and the slow readout still learned
    assert third.initial_loss < second.initial_loss
    assert second.updated and second.final_loss < second.initial_loss


def test_writes_can_be_withheld_and_learning_without_writes_still_updates():
    rng = np.random.default_rng(8)
    net = RecordPatchNet(2, 4, 1, seed=1, cells=64, active=4)
    inputs, target = rng.normal(size=(1, 5, 2)), rng.normal(size=(1, 5, 1))
    result = net.observe(inputs, target, rate=0.5, write=False)
    assert result.updated and result.writes == 0 and net.records.writes == 0
    assert net.updates == 1 and net.readback().writes == 0


def test_imagine_is_pure_and_observe_predicts_what_imagine_predicts():
    rng = np.random.default_rng(9)
    net = RecordPatchNet(3, 4, 2, seed=5, cells=128, active=8)
    inputs, target = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 6, 2))
    net.observe(inputs, target)
    net.reset()
    before = net.snapshot()
    branch = net.imagine(inputs)
    same_snapshot(before, net.snapshot())
    result = net.observe(inputs, target, rate=1.0, backtrack=True)
    assert_array_equal(result.prediction.output, branch.output)
    assert result.updated and result.accepted_rate is not None
    assert result.final_loss < result.initial_loss
    assert result.replay_calls >= 1 and net.state is not None
    assert_array_equal(net.state, result.prediction.final_state)
    # The carried context was computed before the accepted step, as in TemporalPatchNet.
    assert net.readback().state_parameter_revision == net.readback().parameter_revision - 1
    net.advance(inputs)
    assert net.readback().state_parameter_revision == net.readback().parameter_revision


def test_backtracking_rejects_a_step_that_cannot_lower_the_loss():
    net = RecordPatchNet(1, 2, 1, seed=3, cells=32, active=2)
    inputs, target = np.zeros((1, 3, 1)), np.zeros((1, 3, 1))
    net.set_parameters({**net.parameters(), "c": np.zeros(1)})
    result = net.observe(inputs, target, rate=1.0, backtrack=True)
    assert not result.updated and result.reason == "no_decreasing_parameter_step"
    assert result.replay_calls == 0 and net.updates == 0


def test_checkpoints_restore_records_state_and_counts_exactly(tmp_path):
    rng = np.random.default_rng(10)
    net = RecordPatchNet(3, 4, 2, seed=6, cells=128, active=8, output_precision=[2.0, 1.0])
    inputs, target = rng.normal(size=(2, 6, 3)), rng.normal(size=(2, 6, 2))
    net.observe(inputs, target, rate=0.3)
    net.advance(inputs)
    saved = net.save(tmp_path / "patch")
    restored = RecordPatchNet.load(saved)
    same_snapshot(net.snapshot(), restored.snapshot())
    assert restored.updates == 1 and restored.records.writes == 12
    assert_array_equal(restored.imagine(inputs).output, net.imagine(inputs).output)
    cold = RecordPatchNet.load(saved)
    cold.reset()
    assert cold.state is None and cold.readback().state_parameter_revision is None
    bad = net.snapshot()
    bad["meta"] = np.array('{"format": "cadence-temporal/1"}')
    with pytest.raises(ValueError):
        RecordPatchNet.restore(bad)
    missing = {k: v for k, v in net.snapshot().items() if k != "records_table_y"}
    with pytest.raises(ValueError):
        RecordPatchNet.restore(missing)


def test_invalid_data_changes_nothing():
    net = RecordPatchNet(2, 3, 1, seed=4, cells=32, active=2)
    before = net.snapshot()
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 3)), np.zeros((1, 2, 1)))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.full((1, 2, 1), np.nan))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.zeros((1, 3, 1)))
    with pytest.raises(ValueError):
        net.observe(np.zeros((1, 2, 2)), np.zeros((1, 2, 1)), rate=-1.0)
    with pytest.raises(ValueError):
        net.imagine(np.zeros((1, 2, 2)), state=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        net.set_parameters({**net.parameters(), "g": np.zeros(2)})
    with pytest.raises(ValueError):
        net.set_output_precision(np.array([0.0]))
    same_snapshot(before, net.snapshot())


def test_records_state_round_trip_and_witness_moves_the_mean():
    from cadence import Records

    rng = np.random.default_rng(11)
    records = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    readings = rng.normal(size=(3, 5))
    twin = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    twin.code(readings, adapt=True)
    records.witness(readings)
    assert_array_equal(records.mean, twin.mean) and records.seen == 3
    records.write(records.code(readings[0])[:, 0], {"y": np.array([1.0, -1.0])})
    other = Records(5, {"y": 2}, cells=64, active=4, seed=1)
    other.load_state(records.state())
    assert other.writes == 1 and other.seen == 3
    assert_array_equal(other.tables["y"], records.tables["y"])
    with pytest.raises(ValueError):
        other.load_state({k: v for k, v in records.state().items() if k != "mean"})


def test_snapshot_round_trip_carries_the_input_norm_and_record_state():
    rng = np.random.default_rng(12)
    net = RecordPatchNet(
        3,
        5,
        2,
        seed=8,
        cells=120,
        active=6,
        record_rate=0.05,
        record_averaging=True,
        record_homeostasis=0.05,
    )
    inputs, target = rng.normal(size=(2, 9, 3)) * 2.0, rng.normal(size=(2, 9, 2))
    net.observe(inputs, target, rate=0.0)
    assert net._input_norm != 1.0
    assert net.records.count.sum() > 0 and np.abs(net.records.boost).max() > 0
    twin = RecordPatchNet.restore(net.snapshot())
    assert twin._input_norm == net._input_norm
    assert np.array_equal(twin.records.count, net.records.count)
    probe = rng.normal(size=(1, 4, 3))
    assert_allclose(
        twin.imagine(probe, state=np.zeros((1, 5))).output,
        net.imagine(probe, state=np.zeros((1, 5))).output,
        atol=1e-12,
    )


# ------------------------------------------------------------------ categorical ports, batch writes


def _symbols(rng, batch, horizon, inputs, groups):
    x = np.eye(inputs)[rng.integers(inputs, size=(batch, horizon))]
    target = np.concatenate(
        [np.eye(g)[rng.integers(g, size=(batch, horizon))] for g in groups], axis=-1
    )
    return x, target


def test_categorical_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    net = RecordPatchNet(5, 6, 7, seed=4, cells=128, active=6, groups=(3, 4))
    x, target = _symbols(rng, 2, 5, 5, (3, 4))
    delta = net.observe(x, target, rate=0.0, write=False).delta
    base = net.parameters()
    for name in ("G", "g", "B", "b", "C", "c"):
        index = tuple(rng.integers(n) for n in base[name].shape)
        losses = []
        for sign in (1.0, -1.0):
            moved = {k: v.copy() for k, v in base.items()}
            moved[name][index] += sign * 1e-6
            net.set_parameters(moved)
            net.reset()
            losses.append(net.observe(x, target, rate=0.0, write=False).prediction.slow_loss)
        numeric = (losses[0] - losses[1]) / 2e-6
        assert abs(numeric - delta[name][index]) < 1e-7 * max(1.0, abs(numeric))
    net.set_parameters(base)


def test_categorical_ports_are_distributions_and_recall_in_one_shot():
    rng = np.random.default_rng(7)
    net = RecordPatchNet(17, 8, 6, seed=2, cells=4096, active=32, record_rate=1.0, groups=(6,))
    inputs = np.zeros((1, 16, 17))
    inputs[0, 0, 0] = 1.0
    inputs[0, np.arange(16), 1 + np.arange(16)] = 1.0
    identities = rng.integers(6, size=16)
    target = np.eye(6)[identities][None]
    first = net.observe(inputs, target, rate=0.0)
    assert_allclose(first.prediction.slow_output.sum(axis=-1), 1.0)
    assert first.writes == 16
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    assert np.array_equal(recalled.output[0].argmax(axis=1), identities)


def test_categorical_ports_validate_targets_and_refuse_detune():
    net = RecordPatchNet(3, 4, 4, seed=1, cells=64, active=4, groups=(2, 2))
    x = np.ones((1, 2, 3))
    with pytest.raises(ValueError, match="distribution"):
        net.observe(x, np.ones((1, 2, 4)))
    with pytest.raises(ValueError, match="partition"):
        RecordPatchNet(3, 4, 4, groups=(2, 3))
    good = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (1, 2, 1))
    with pytest.raises(ValueError, match="quadratic"):
        net.detune(x, good)


def test_default_checkpoints_keep_their_format_and_port_checkpoints_round_trip():
    plain = RecordPatchNet(3, 4, 2, seed=1, cells=64, active=4)
    assert '"cadence-record-patch/2"' in str(plain.snapshot()["meta"])
    net = RecordPatchNet(3, 4, 4, seed=1, cells=64, active=4, groups=(2, 2), record_writes="batch")
    x = np.eye(3)[None, [0, 1, 2]]
    target = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (1, 3, 1))
    net.observe(x, target)
    again = RecordPatchNet.restore(net.snapshot())
    assert again.groups == (2, 2) and again.record_writes == "batch"
    assert_array_equal(
        again.imagine(x, state=np.zeros((1, 4))).output,
        net.imagine(x, state=np.zeros((1, 4))).output,
    )


def test_one_writer_in_a_batch_is_the_sequential_write():
    from cadence import Records

    rng = np.random.default_rng(5)
    a = Records(6, {"y": 3}, cells=256, active=8, rate=0.5, seed=3)
    b = Records(6, {"y": 3}, cells=256, active=8, rate=0.5, seed=3)
    reading, outcome = rng.normal(size=(1, 6)), rng.normal(size=3)
    code = a.code(reading)
    a.write(code[:, 0], {"y": outcome})
    b.write_batch(code, {"y": outcome[None]})
    assert_allclose(a.tables["y"], b.tables["y"], atol=1e-15)


def test_agreeing_writers_in_a_batch_do_not_overshoot():
    from cadence import Records

    store = Records(6, {"y": 2}, cells=256, active=8, rate=1.0, seed=3)
    reading = np.tile(np.random.default_rng(1).normal(size=(1, 6)), (12, 1))
    codes = store.code(reading)
    store.write_batch(codes, {"y": np.tile([1.0, -2.0], (12, 1))})
    assert_allclose(store.read(codes[:, 0])["y"], [1.0, -2.0], atol=1e-12)


def test_batch_writes_converge_no_slower_than_sequential_writes():
    wrong = {}
    for writes in ("sequential", "batch"):
        net = RecordPatchNet(
            12,
            16,
            5,
            seed=6,
            cells=4096,
            active=24,
            record_rate=1.0,
            groups=(5,),
            record_writes=writes,
        )
        x, target = _symbols(np.random.default_rng(11), 4, 10, 12, (5,))
        wrong[writes] = []
        for _ in range(12):
            net.reset()
            seen = net.observe(x, target, rate=0.0).prediction.output
            wrong[writes].append(int(np.sum(seen.argmax(-1) != target.argmax(-1))))
    # Two rows open with the same symbol and different outcomes: one moment is undecidable.
    assert wrong["batch"][-1] <= 1 and wrong["sequential"][-1] <= 1
    assert wrong["batch"][0] == wrong["sequential"][0] > 20
    assert sum(wrong["batch"]) <= sum(wrong["sequential"])


def test_a_coded_store_holds_the_coded_residual_and_recalls_a_wide_port():
    rng = np.random.default_rng(13)
    net = RecordPatchNet(
        24, 8, 60, seed=3, cells=4096, active=32, record_rate=1.0, groups=(60,), record_width=24
    )
    assert net.records.tables["y"].shape == (4096, 24)
    inputs = np.eye(24)[None, :20]
    identities = rng.integers(60, size=20)
    target = np.eye(60)[identities][None]
    first = net.observe(inputs, target, rate=0.0)
    # One write at rate one: the cells of the first reading hold the code of its residual.
    residual = target[0, 0] - first.prediction.slow_output[0, 0]
    reading = net._readings(inputs, first.prediction.hidden)[0, :1]
    held = net.records.read(net.records.code(reading, valued=False)[:, 0])["y"]
    overlap = 1.0  # the first reading's own write; later writes add their overlap with it
    assert np.linalg.norm(held - overlap * residual @ net._output_code) < 0.5 * np.linalg.norm(held)
    for _ in range(5):
        net.reset()
        net.observe(inputs, target, rate=0.0)
    recalled = net.imagine(inputs, state=np.zeros((1, 8)))
    assert np.array_equal(recalled.output[0].argmax(axis=1), identities)
    again = RecordPatchNet.restore(net.snapshot())
    assert again.record_width == 24
    assert_array_equal(again.imagine(inputs, state=np.zeros((1, 8))).output, recalled.output)


def test_sleep_moves_what_the_store_holds_into_the_slow_weights():
    rng = np.random.default_rng(21)
    net = RecordPatchNet(
        12, 12, 5, seed=4, cells=2048, active=16, record_rate=1.0, groups=(5,), slowest=8.0
    )
    x = np.eye(12)[rng.integers(12, size=(3, 8))]
    symbol = x.argmax(-1)
    target = np.eye(5)[(symbol + np.roll(symbol, 1, axis=1)) % 5]
    # The day: the store takes the pairs; the slow weights do not move (rate zero).
    for _ in range(8):
        net.reset()
        net.observe(x, target, rate=0.0)
    net.reset()
    awake = net.imagine(x, state=np.zeros((3, 12)))
    assert np.array_equal(awake.output.argmax(-1), target.argmax(-1))
    assert not np.array_equal(awake.slow_output.argmax(-1), target.argmax(-1))
    # The night: fixed dreams, the slow weights learn them, the store is rewritten at dawn.
    report = net.sleep([x], passes=240, rate=8.0, backtrack=True)
    assert report["updates"] > 200
    assert report["dream_loss_after"] < 0.25 * report["dream_loss_before"]
    assert report["dawn_writes"] == 48
    alone = RecordPatchNet.restore(net.snapshot())
    alone.records.tables["y"][:] = 0.0
    slept = alone.imagine(x, state=np.zeros((3, 12)))
    assert np.array_equal(slept.output.argmax(-1), target.argmax(-1))  # the slow weights alone
    with_store = net.imagine(x, state=np.zeros((3, 12)))
    assert np.array_equal(with_store.output.argmax(-1), target.argmax(-1))


def test_sleep_and_dream_change_nothing_but_the_slow_weights_and_the_store_reference():
    net = RecordPatchNet(4, 6, 3, seed=2, cells=128, active=8, groups=(3,))
    x = np.eye(4)[None, [0, 1, 2, 3, 0]]
    dream = net.dream(x)
    assert dream.shape == (1, 5, 3) and np.array_equal(dream.sum(-1), np.ones((1, 5)))
    assert net.records.writes == 0 and net.updates == 0
    with pytest.raises(ValueError, match="at least one cue"):
        net.sleep([])
    report = net.sleep([x], passes=0, dawn_passes=1)
    assert report["updates"] == 0 and report["dawn_writes"] == 5 and net.updates == 0
