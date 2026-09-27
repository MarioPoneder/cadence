"""Release re-audit: factual input, action-selection and child-state custody."""

import numpy as np
import pytest

from cadence import (
    BeliefPatch,
    DenseBlock,
    JointRecordPatches,
    RecordPatchNet,
    Softmax,
    Steered,
    StructuredPort,
)
from cadence.life import AlwaysAwake, Life, LifeConfig, NeverWakes


def patch():
    return BeliefPatch(
        StructuredPort(1, [DenseBlock(0, 1, 1)]), 1, 2, 1,
        cells=4, active=1, record_width=1,
    )


def life(*, governor=None, baseline=1.0, config=None):
    return Life(
        patch(), governor or NeverWakes(),
        habit=lambda reading: np.zeros(1),
        propose=lambda reading: np.array([[0.0], [1.0]]),
        advance=lambda reading, output: reading + output,
        cost=lambda readings, actions: np.zeros(len(readings)),
        target=lambda reading, next_reading: next_reading,
        baseline=baseline, config=config,
    )


def test_perfect_outcomes_with_zero_requested_floor_close_once_and_keep_finite_signals():
    loop = life(config=LifeConfig(floor=0, baseline_rate=1))
    for _ in range(3):
        loop.decide(np.zeros(1))
        loop.outcome(np.zeros(1))
        assert loop.pending is None
        assert loop.baseline > 0 and np.isfinite(loop.readback()).all()
    assert loop.t == len(loop.y) == len(loop.records) == 3


def test_log_surprise_stays_finite_when_the_raw_ratio_overflows():
    loop = life(baseline=1e-320, config=LifeConfig(floor=0, baseline_rate=0))
    loop.decide(np.zeros(1))
    result = loop.outcome(np.array([10.0]))
    assert result.surprise == 100.0
    assert np.isfinite(loop.readback()).all()
    expected = np.log(100.0) - np.log(loop.baseline)
    assert loop.readback()[0] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.0]), np.array([0.0, np.inf]), 0.0])
def test_imagination_rejects_nonfinite_or_broadcast_task_cost_before_action(bad):
    loop = life(governor=AlwaysAwake())
    loop.task_cost = lambda readings, actions: bad
    with pytest.raises(ValueError, match="cost"):
        loop.decide(np.zeros(1))
    assert loop.pending is None and not loop.a and not loop.o
    assert loop.patch.state is None


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.zeros(2)])
def test_imagination_rejects_invalid_predicted_readings(bad):
    loop = life(governor=AlwaysAwake())
    loop.advance = lambda reading, output: bad
    with pytest.raises(ValueError, match="advance"):
        loop.decide(np.zeros(1))
    assert loop.pending is None and loop.patch.state is None


def test_imagination_rejects_overflow_of_otherwise_finite_costs():
    loop = life(governor=AlwaysAwake(), config=LifeConfig(horizon=2))
    loop.task_cost = lambda readings, actions: np.full(len(readings), 1e308)
    with pytest.raises(ValueError, match="cost"):
        loop.decide(np.zeros(1))
    assert loop.pending is None


def test_imagination_still_selects_the_smallest_valid_full_horizon_cost():
    loop = life(governor=AlwaysAwake(), config=LifeConfig(horizon=3))
    loop.task_cost = lambda readings, actions: -actions[:, 0]
    np.testing.assert_array_equal(loop.decide(np.zeros(1)), [1.0])
    assert loop.pending.imagined == 6


def test_joint_children_cannot_alias_mutable_context_and_parameters():
    child = RecordPatchNet(1, 1, 1, cells=2, active=1)
    before = child.snapshot()
    with pytest.raises(ValueError, match="distinct"):
        JointRecordPatches([child, child], [1, 1], [])
    for name, value in before.items():
        np.testing.assert_array_equal(child.snapshot()[name], value)
    twin = RecordPatchNet.restore(before)
    joint = JointRecordPatches([child, twin], [1, 1], [])
    result = joint.observe(
        [np.zeros((1, 1, 1))] * 2,
        [np.ones((1, 1, 1)), -np.ones((1, 1, 1))], rate=0.1, write=False,
    )
    assert result.updated
    assert child.parameters()["c"][0] > 0 > twin.parameters()["c"][0]


def test_steering_and_cortex_must_have_distinct_mutable_state():
    child = BeliefPatch(
        StructuredPort(5, [DenseBlock(0, 2, 1), DenseBlock(2, 3, 1)]), 1, 2, 2,
        cells=4, active=1, record_width=1,
    )
    with pytest.raises(ValueError, match="distinct"):
        Steered(child, child, Softmax(2))


@pytest.mark.parametrize("bad", [np.array([np.nan]), np.array([2]), np.array(["false"])])
def test_observation_mask_cannot_promote_invalid_values_into_factual_records(bad):
    brain = patch()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="observed"):
        brain.observe(
            np.ones((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)),
            observed=bad, rate=0, write=True,
        )
    for name, value in before.items():
        np.testing.assert_array_equal(brain.snapshot()[name], value)


def test_binary_integer_observation_masks_keep_existing_behavior():
    brain = patch()
    result = brain.observe(
        np.ones((1, 2, 1)), np.zeros((1, 2, 1)), np.ones((1, 2, 1)),
        observed=np.array([0, 1]), rate=0, write=True,
    )
    assert result.writes == 1


def test_reset_retains_history_but_replay_starts_at_the_new_continuation():
    loop = life(config=LifeConfig(window=6, min_window=2, passes=1, validity=0))
    for _ in range(3):
        loop.decide(np.ones(1))
        loop.outcome(np.array([100.0]))
    loop.reset()
    assert len(loop.y) == 3 and not loop.signals().can_learn
    with pytest.raises(RuntimeError, match="current continuation"):
        loop._learn()
    for index, target in enumerate((0.2, 0.4)):
        loop.decide(np.zeros(1))
        loop.outcome(np.array([target]))
        assert loop.signals().can_learn == (index == 1)
    state = loop.patch.state
    result = loop._learn()
    assert result["window"] == 2 and result["loss_before"] == pytest.approx(0.05)
    assert len(loop.y) == 5  # Historical evidence is retained, not spliced into this replay.
    np.testing.assert_array_equal(loop.patch.state, state)


def test_continuation_window_still_works_after_its_first_boundary_is_pruned():
    loop = life(config=LifeConfig(keep=3, window=3, min_window=2, passes=1, validity=0))
    loop.decide(np.zeros(1))
    loop.outcome(np.array([100.0]))
    loop.reset()
    for target in (0.1, 0.2, 0.3, 0.4):
        loop.decide(np.zeros(1))
        loop.outcome(np.array([target]))
    assert all(boundary is not None for boundary in loop.boundaries)
    result = loop._learn()
    assert result["window"] == 3
    assert result["loss_before"] == pytest.approx(0.5 * np.mean(np.square([0.2, 0.3, 0.4])))
