"""Adversarial continuation, custody and measurement checks for composition."""

import numpy as np
import pytest

from cadence import BeliefPatch, DenseBlock, Gaze, Softmax, Steered, StructuredPort
from cadence.instruments import dishabituation, orienting
from cadence.life import Life, LifeConfig, NeverWakes, PatchGovernor, Signals, ThresholdGovernor
from cadence.ports import MapBlock, block_from_dict


def patch(inputs=2, outputs=2, seed=3):
    brain = BeliefPatch(
        StructuredPort(inputs, [DenseBlock(i, 1, 2) for i in range(inputs)]),
        1,
        3,
        outputs,
        cells=8,
        active=2,
        record_width=2,
        seed=seed,
    )
    p = brain.parameters()
    p["C"][:] = np.random.default_rng(seed).normal(size=p["C"].shape)
    brain.set_parameters(p)
    return brain


def composition(gaze=False):
    cortex = patch()
    steering = patch(inputs=9 if gaze else 5, outputs=1 if gaze else 2, seed=4)
    return Steered(
        cortex,
        steering,
        Gaze(2, sigma=1, start=0.3) if gaze else Softmax(2),
        lagged=gaze,
        relative=gaze,
        reads_age=gaze,
        reads_output=gaze,
    )


def life(brain=None, **kwargs):
    return Life(
        patch() if brain is None else brain,
        NeverWakes(),
        habit=lambda r: np.ones(1),
        propose=lambda r: np.ones((1, 1)),
        advance=lambda r, y: r + y,
        cost=lambda r, a: np.zeros(len(r)),
        target=lambda r, next_r: next_r - r,
        config=LifeConfig(window=2, min_window=1, passes=1, **kwargs),
    )


def lived(brain=None, **kwargs):
    loop = life(brain, **kwargs)
    for i in range(2):
        loop.decide(np.array([i, 0.0]))
        loop.outcome(np.array([i + 1.0, 1.0]))
    return loop


def equal_snapshot(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_rejected_learning_restores_records_state_step_and_updates_but_counts_work():
    loop = lived(validity=0, write=True)
    before = loop.patch.snapshot()
    cost = dict(loop.patch.cost)
    result = loop._learn()
    assert not result["kept"]
    equal_snapshot(loop.patch.snapshot(), before)
    assert loop.patch.cost["moments"] > cost["moments"]


def test_learning_replays_original_zero_boundary_and_keeps_present_state():
    loop = lived(validity=0)
    before = loop.patch.state
    o, a, y = np.array(loop.o)[None], np.array(loop.a)[None], np.array(loop.y)[None]
    expected = loop.patch.observe(o, a, y, rate=0, state=np.zeros((1, 3)), keep_live=True)
    result = loop._learn()
    assert result["loss_before"] == expected.initial_loss
    np.testing.assert_array_equal(loop.patch.state, before)


def test_learning_exception_rolls_back_whole_transaction(monkeypatch):
    loop = lived(write=True)
    before = loop.patch.snapshot()

    def broken(*args):
        raise RuntimeError("failed validation replay")

    monkeypatch.setattr(loop.brain, "loss", broken)
    with pytest.raises(RuntimeError, match="validation replay"):
        loop._learn()
    equal_snapshot(loop.patch.snapshot(), before)


def test_belief_partial_record_write_failure_rolls_back_entire_observation(monkeypatch):
    brain = patch()
    before = brain.snapshot()
    original = brain.records.write
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(brain.records, "write", broken)
    with pytest.raises(RuntimeError, match="second write"):
        brain.observe(np.ones((1, 2, 2)), np.ones((1, 2, 1)), np.ones((1, 2, 2)), write=True)
    assert calls == 2
    equal_snapshot(brain.snapshot(), before)


def test_nonfinite_prediction_is_not_committed_as_live_belief():
    brain = patch()
    params = brain.parameters()
    params["c"][:] = 1e308
    brain.set_parameters(params)
    before = brain.snapshot()
    with np.errstate(over="ignore", invalid="ignore"):
        result = brain.observe(np.ones((1, 1, 2)), np.ones((1, 1, 1)), np.zeros((1, 1, 2)))
    assert result.reason == "nonfinite_prediction"
    equal_snapshot(brain.snapshot(), before)


def test_overflowing_forward_output_cannot_poison_live_continuation():
    brain = patch()
    params = brain.parameters()
    params["T"][:] = 0
    params["F"][:] = 0
    params["f_b"][:] = 20
    params["C"][:] = 1e308
    brain.set_parameters(params)
    before = brain.snapshot()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="nonfinite"):
            brain.assimilate(np.ones((1, 1, 2)), np.zeros((1, 1, 1)))
    equal_snapshot(brain.snapshot(), before)


def test_sleeping_cortex_is_respected_by_life_and_live_boundary_is_preserved():
    loop = lived(composition(), learn_cortex=False, rollback=False)
    before = loop.patch.parameters()
    boundary = loop.patch.boundary()
    loop._learn()
    for key, value in before["cortex"].items():
        np.testing.assert_array_equal(loop.patch.cortex.parameters()[key], value)
    assert loop.patch.steering.updates > 0
    np.testing.assert_array_equal(loop.patch.boundary().cortex, boundary.cortex)


def test_history_does_not_alias_readings_actions_or_returned_decisions():
    loop = life()
    reading = np.array([1.0, 2.0])
    action = loop.decide(reading)
    reading[:] = 9
    action[:] = 9
    np.testing.assert_array_equal(loop.o[-1], [1, 2])
    np.testing.assert_array_equal(loop.a[-1], [1])
    decision = loop.outcome(np.array([2.0, 3.0]))
    decision.target[:] = 9
    decision.action[:] = 9
    np.testing.assert_array_equal(loop.y[-1], [1, 1])
    np.testing.assert_array_equal(loop.records[-1].target, [1, 1])
    loop.compute()["decisions"]["habit"] = 100
    assert loop.totals["decisions"]["habit"] == 1


def test_reset_discards_only_unclosed_moment_and_invalid_outcome_is_retryable():
    loop = lived()
    loop.decide(np.zeros(2))
    with pytest.raises(ValueError, match="next_reading"):
        loop.outcome(np.array([np.nan, 0]))
    assert loop.pending is not None and len(loop.y) == 2
    loop.reset()
    assert loop.pending is None
    assert [len(getattr(loop, k)) for k in ("o", "a", "y", "boundaries")] == [2] * 4
    loop.decide(np.zeros(2))
    loop.outcome(np.ones(2))
    assert [len(getattr(loop, k)) for k in ("o", "a", "y", "boundaries")] == [3] * 4
    loop._learn()


def test_steered_invalid_target_and_parameter_replacement_are_atomic():
    brain = composition()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="target"):
        brain.run(np.ones((1, 1, 2)), np.ones((1, 1, 1)), np.zeros((1, 1, 3)))
    equal_snapshot(brain.snapshot(), before)
    parameters = brain.parameters()
    parameters["cortex"]["c"][:] = 12
    parameters["steering"]["c"][:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        brain.set_parameters(parameters)
    equal_snapshot(brain.snapshot(), before)


def test_joint_replay_exception_cannot_leave_trial_parameters_or_live_state(monkeypatch):
    brain = composition()
    before = brain.snapshot()

    def broken(*args):
        raise RuntimeError("trial failed")

    monkeypatch.setattr(brain, "_objective", broken)
    with pytest.raises(RuntimeError, match="trial failed"):
        brain.run(np.ones((1, 2, 2)), np.ones((1, 2, 1)), np.ones((1, 2, 2)), rate=1)
    equal_snapshot(brain.snapshot(), before)


def test_preconditioned_armijo_uses_directional_slope_not_squared_scale():
    brain = composition()
    brain.rate_scale = 1e6
    # Unit effective steering step: the old squared scale makes its required
    # Armijo improvement exceed even the infinitesimal directional decrease.
    result = brain.run(
        np.ones((1, 2, 2)), np.ones((1, 2, 1)), np.ones((1, 2, 2)), rate=1e-6, learn_cortex=False
    )
    assert result.steering_updated and result.reason == "updated"


def test_mac_estimate_counts_kernel_reuse_and_executed_readback():
    dense = BeliefPatch(
        StructuredPort(4, [DenseBlock(0, 4, 4)]), 1, 3, 2, cells=8, active=2, record_width=2
    )
    mapped = BeliefPatch(
        StructuredPort(4, [MapBlock(0, 1, 2, 2, 1, 1)]), 1, 3, 2, cells=8, active=2, record_width=2
    )
    # Dense costs 16 MACs; its tied 1x1 alternative performs four, not one.
    assert dense.macs_per_moment() - mapped.macs_per_moment() == 12
    brain = composition()
    brain.run(np.ones((2, 3, 2)), np.ones((2, 3, 1)))
    assert brain.cost["macs"] == 6 * brain.macs_per_moment()
    assert brain.cost["macs"] > brain.cortex.cost["macs"] + brain.steering.cost["macs"]


@pytest.mark.parametrize("gaze", [False, True])
def test_checkpoint_continues_gains_readback_and_output_without_reset(gaze):
    brain = composition(gaze)
    rng = np.random.default_rng(51)
    o, a = rng.normal(size=(1, 6, 2)), rng.normal(size=(1, 6, 1))
    brain.run(o[:, :3], a[:, :3])
    brain.deaf = np.ones(brain.channels)
    brain.deaf[0] = 0
    saved = brain.snapshot()
    twin = Steered.restore(saved)
    # Returned boundaries and paths must not mutate the next continuation.
    boundary = brain.boundary()
    boundary.cortex[:] = 100
    boundary.residual[:] = 100
    if boundary.weighing is not None:
        boundary.weighing[:] = 100
    first, second = brain.run(o[:, 3:], a[:, 3:]), twin.run(o[:, 3:], a[:, 3:])
    for key in ("gains", "readback", "output"):
        np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
    first.last.belief[:] = 99
    np.testing.assert_array_equal(brain.boundary().cortex, twin.boundary().cortex)


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_norm", np.array(0.0)),
        ("output_code", np.zeros((1, 1))),
        ("step_size", np.array([-1.0])),
        ("step_size", np.array([1.0, 2.0])),
        ("state", np.full((1, 3), np.nan)),
    ],
)
def test_malformed_belief_checkpoint_is_rejected(key, value):
    saved = patch().snapshot()
    saved[key] = value
    with pytest.raises(ValueError):
        BeliefPatch.restore(saved)


def test_governor_capped_attempt_does_not_authorize_provisional_mode():
    governor = PatchGovernor(budget=2, chunk=1, tolerance=1e-12)
    signals = Signals(
        np.array([10.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]), 0, 1, 0, 0, "habit", [], False
    )
    mode, steps = governor.settle(signals)
    assert mode == "habit" and steps == 2 and not governor.converged
    assert governor.residual > governor.tolerance and governor.state is None


def test_threshold_cooldown_gene_changes_when_learning_can_recur():
    signals = Signals(np.ones(7), 2, 1, 0, 0, "habit", [2], True)
    slow = ThresholdGovernor({"persist": 1, "k_learn": 1, "cooldown": 3})
    fast = ThresholdGovernor({"persist": 1, "k_learn": 1, "cooldown": 0})
    assert slow.settle(signals)[0] == fast.settle(signals)[0] == "learn"
    slow.after(signals)
    assert slow.settle(signals)[0] != "learn" and fast.settle(signals)[0] == "learn"
    slow.after(signals)
    assert slow.settle(signals)[0] != "learn"
    slow.after(signals)
    assert slow.settle(signals)[0] == "learn"
    slow.reset()
    assert slow.cooldown_left == 0


def test_instrument_does_not_invent_return_for_small_or_truncated_response():
    g = np.zeros(10)
    g[3:] = 0.01
    row = orienting(g, [("event", 3, 1)])["rows"][0]
    assert row["capture"] == 0.01 and row["return"] is None and row["return_censored"]
    g[6:] = 0
    row = orienting(g, [("event", 3, 1)])["rows"][0]
    assert row["return"] == 2 and not row["return_censored"]
    with pytest.raises(ValueError, match="finite"):
        orienting(np.array([np.nan]), [])
    with pytest.raises(ValueError, match="count"):
        dishabituation([], consequential="a", kind="b", count=0)


def test_invalid_port_configuration_cannot_change_meaning_on_restore():
    with pytest.raises(ValueError, match="kind"):
        block_from_dict({"kind": "typo", "start": 0, "inputs": 2, "outputs": 1})
    with pytest.raises(ValueError, match="mask"):
        StructuredPort(2, [DenseBlock(0, 2, 1)], mask=[0, np.nan])
    with pytest.raises(ValueError, match="stride"):
        MapBlock(0, 1, 4, 4, 1, 2, stride=0)
    with pytest.raises(ValueError, match="integer"):
        StructuredPort(2.5, [DenseBlock(0, 2, 1)])


@pytest.mark.parametrize(
    "field,value", [("passes", 0), ("recent", 0), ("horizon", 0), ("slow_rate", 2)]
)
def test_invalid_life_configuration_fails_before_running(field, value):
    with pytest.raises(ValueError, match=field):
        LifeConfig(**{field: value})
