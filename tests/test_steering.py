"""The steered cortex: the readback layout and the moments against the belief patch's own,
the weighing's pull against finite differences, the joint admitted step, the sleeping cortex,
the online life against the batch run, the rule and fixed arms, the ablations, custody, cost."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import BeliefPatch, DenseBlock, StructuredPort
from cadence.steering import Boundary, Gaze, Rule, Softmax, Steered


def _implied(y):
    r = np.full((len(y), 8), np.nan)
    r[:, :2] = y
    return r


def _cortex(seed=1, readout=True):
    eye, ear = DenseBlock(0, 6, 8), DenseBlock(6, 2, 4)
    cortex = BeliefPatch(StructuredPort(8, [eye, ear]), actions=1, belief=8, outputs=2, cells=64, active=4, record_width=8, seed=seed)
    cortex.set_implied_reading(_implied, units=[0.1, 0.1])
    if readout:
        rng = np.random.default_rng(seed)
        params = cortex.parameters()
        params["C"] = rng.normal(size=params["C"].shape) * 0.3
        cortex.set_parameters(params)
    return cortex


def _steering(channels, seed=7, mask=None, outputs=2):
    port = StructuredPort(channels, [DenseBlock(0, channels, 5)], mask=mask)
    patch = BeliefPatch(port, actions=1, belief=6, outputs=outputs, cells=16, active=2, record_width=4, seed=seed)
    rng = np.random.default_rng(seed)
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * 0.5  # a readout, so the gains move
    patch.set_parameters(params)
    return patch


def _brain(seed=1, mask=None, **kw):
    cortex = _cortex(seed)
    steering = _steering(9, seed + 6, mask=mask)  # 2 probes, 2 surprises, residual, 4 ear units
    return Steered(cortex, steering, Softmax(2, span=2.0), evidence=[1], **kw)


def _data(rng, n=3, t=5):
    return rng.random((n, t, 8)), np.zeros((n, t, 1)), rng.normal(size=(n, t, 2))


def test_the_readback_layout_and_the_moments_match_the_belief_patch():
    rng = np.random.default_rng(10)
    brain = _brain(10)
    o, a, y = _data(rng)
    path = brain.run(o, a)
    assert brain.channels == 9 and path.readback.shape == (3, 5, 9)
    assert brain.readback_names[:5] == ["probe:0", "probe:1", "surprise:0", "surprise:1", "residual"]
    assert brain.readback_names[5:] == [f"evidence:1:{i}" for i in range(4)]
    # the moments, replayed by hand under the recorded gains, are the cortex's own moments
    cortex = brain.cortex
    z = np.zeros((3, cortex.belief))
    for k in range(5):
        moment = cortex.readback(o[:, k], a[:, k], state=z)
        expected = np.concatenate([moment.residual_alone, moment.surprise, (np.zeros(3) if k == 0 else prev)[:, None], moment.evidence[:, 8:]], axis=-1)
        np.testing.assert_allclose(path.readback[:, k], expected, atol=1e-12)
        step = cortex.assimilate(o[:, k][:, None], a[:, k][:, None], gains=path.gains[:, k][:, None], state=z, keep_live=True)
        np.testing.assert_allclose(path.output[:, k], step.output[:, 0], atol=1e-12)
        z, prev = step.final_state, step.residual[:, 0]
    np.testing.assert_allclose(path.gains.sum(axis=-1), 2.0)  # the softmax weighing sums to the blocks
    assert path.loss is None and path.last is not None and path.last_steering is not None
    live = brain.boundary()
    assert isinstance(live, Boundary) and live.moments == 5
    np.testing.assert_allclose(live.cortex, z)


def test_the_pull_matches_finite_differences_of_the_objective_in_the_steering_outputs():
    rng = np.random.default_rng(11)
    brain = _brain(11)
    o, a, y = _data(rng, n=2, t=4)
    path = brain.run(o, a, y)
    start = brain._fresh(2)
    grad = brain.cortex.observe(o, a, y, gains=path.gains, state=start.cortex, rate=0.0, write=False, keep_live=True)
    dy = brain.weighing.pull(path.steering_output, path.gains, grad.gain_gradient, [None] * 4)

    def objective(ys):
        gains = brain._weigh_chunk(ys, None)
        replay = brain.cortex.assimilate(o, a, gains=gains, state=start.cortex, keep_live=True)
        return brain.cortex._loss(replay.slow_output, y)

    eps = 1e-6
    worst = 0.0
    for _ in range(8):
        i, k, j = rng.integers(0, 2), rng.integers(0, 4), rng.integers(0, 2)
        up, dn = path.steering_output.copy(), path.steering_output.copy()
        up[i, k, j] += eps
        dn[i, k, j] -= eps
        fd = (objective(up) - objective(dn)) / (2 * eps)
        worst = max(worst, abs(fd - dy[i, k, j]) / max(1e-3, abs(fd)))
    assert worst < 1e-4


def test_the_joint_step_lowers_the_objective_and_the_cortex_can_sleep():
    rng = np.random.default_rng(12)
    brain = _brain(12, rate_scale=0.7)
    o, a, y = _data(rng)
    before = brain.parameters()
    taught = brain.run(o, a, y, rate=8.0)
    assert taught.updated and taught.steering_updated and taught.reason == "updated"
    assert taught.replays >= 1 and 0 < taught.step <= 8.0 and brain.step_size == taught.step
    after = brain.parameters()
    assert any(not np.array_equal(before["cortex"][k], after["cortex"][k]) for k in after["cortex"])
    assert any(not np.array_equal(before["steering"][k], after["steering"][k]) for k in after["steering"])
    replayed = brain.run(o, a, y, state=brain._fresh(3))
    assert replayed.objective < taught.objective  # the admitted step lowered the chunk's objective
    # the same step is verified by the replay the admission ran: a replay from the same boundary
    again = brain.run(o, a, y, rate=8.0, state=brain._fresh(3))
    assert again.step <= 2.0 * taught.step + 1e-12  # the start follows the last admitted step
    # the cortex sleeps: its parameters stay, the steering patch still learns from its gradient
    asleep = _brain(12)
    kept = asleep.parameters()["cortex"]
    night = asleep.run(o, a, y, rate=8.0, learn_cortex=False)
    assert night.steering_updated and not night.updated
    assert all(np.array_equal(kept[k], asleep.cortex.parameters()[k]) for k in kept)
    assert asleep.cortex.updates == 0 and asleep.steering.updates == 1
    assert brain.cost["replays"] == taught.replays + again.replays and brain.cost["moments"] > 0


def test_the_online_life_from_a_kept_boundary_equals_the_batch_run():
    rng = np.random.default_rng(13)
    brain = _brain(13)
    o, a, y = _data(rng, n=1, t=6)
    batch = brain.run(o, a, y, state=brain._fresh(1))
    # the page's loop: one moment at a time, the boundary kept before the chunk
    brain.reset()
    boundary = brain._fresh(1)
    brain._live = boundary
    seen = []
    for k in range(6):
        seen.append(brain.run(o[:, k : k + 1], a[:, k : k + 1]).output[:, 0])
    np.testing.assert_allclose(np.stack(seen, axis=1), batch.output, atol=1e-12)
    after_steps = brain.boundary()
    np.testing.assert_allclose(after_steps.cortex, brain.run(o, a, state=boundary, keep_live=True).last.final_state, atol=1e-12)
    # learning from the kept boundary with keep_live leaves the place; without it, the live boundary is the replay's end
    learned = brain.run(o, a, y, rate=4.0, state=boundary, keep_live=True)
    assert learned.loss == pytest.approx(batch.loss)
    np.testing.assert_allclose(brain.boundary().cortex, after_steps.cortex)
    brain.run(o, a, y, state=boundary)
    assert brain.boundary().moments == 6
    with pytest.raises(ValueError, match="boundary"):
        brain.run(o, a, state=Boundary(np.zeros((2, 8)), np.zeros((2, 6)), None, np.zeros(2), None, 0))


def test_a_rule_and_fixed_gains_are_the_arms_below_the_rung():
    rng = np.random.default_rng(14)
    o, a, y = _data(rng)
    rule = Steered(_cortex(14), weighing=Rule(lambda r: np.tile([1.6, 0.4], (len(r), 1)), macs=6))
    path = rule.run(o, a, y, rate=4.0)
    np.testing.assert_allclose(path.gains, 1.6 * np.ones((3, 5, 2)) * [1.0, 0.25])
    assert path.updated and path.steering_output is None and rule.macs_per_moment() == rule.cortex.macs_per_moment() + 6
    assert rule.moments_per_decision() == 1 and rule.channels == 5 and not rule.probes_on
    fixed = Steered(_cortex(14))
    ones = fixed.run(o, a, y, rate=4.0)
    np.testing.assert_array_equal(ones.gains, 1.0)
    assert ones.updated and ones.step is not None and fixed.step_size == ones.step
    asleep = Steered(_cortex(14)).run(o, a, y, rate=4.0, learn_cortex=False)
    assert not asleep.updated and asleep.reason == "no_step"
    with pytest.raises(ValueError, match="rule"):
        Steered(_cortex(14), _steering(5), Rule(lambda r: r[:, :2]))
    with pytest.raises(ValueError, match="steering patch"):
        Steered(_cortex(14), weighing=Softmax(2))
    with pytest.raises(ValueError, match="readback channels"):
        Steered(_cortex(14), _steering(7), Softmax(2), evidence=[1])


def test_the_ablations_cut_the_gains_and_deafen_channels():
    rng = np.random.default_rng(15)
    brain = _brain(15)
    o, a, y = _data(rng)
    brain.run(o, a, y, rate=8.0)
    brain.ablation = "cut"
    cut = brain.run(o, a, y, state=brain._fresh(3))
    np.testing.assert_array_equal(cut.gains, 1.0)
    assert cut.steering_output is not None  # the steering patch still runs and is counted
    taught = brain.run(o, a, y, rate=8.0, state=brain._fresh(3))
    assert not taught.steering_updated and taught.reason in ("updated", "no_decreasing_parameter_step")
    brain.ablation = lambda g: g[:, ::-1]  # the ventriloquist's shuffle
    shuffled = brain.run(o, a, state=brain._fresh(3))
    brain.ablation = None
    on = brain.run(o, a, state=brain._fresh(3))
    np.testing.assert_allclose(shuffled.gains[:, 0], on.gains[:, 0, ::-1])  # the first moment reads the same readback
    with pytest.raises(ValueError, match="same shape"):
        brain.ablation = lambda g: g[:, :1]
        brain.run(o, a, state=brain._fresh(3))
    brain.ablation = None
    deaf = np.ones(9, dtype=bool)
    deaf[5:] = False
    brain.deaf = deaf
    quiet = brain.run(o, a, state=brain._fresh(3))
    assert np.all(quiet.readback[:, :, 5:] == 0.0) and np.any(quiet.readback[:, :, :5] != 0.0)


def test_the_gaze_turns_a_window_and_its_pull_is_exact_within_a_moment():
    rng = np.random.default_rng(16)
    b = 8
    cortex = BeliefPatch(StructuredPort(b, [DenseBlock(i, 1, 1) for i in range(b)]), actions=1, belief=6, outputs=b, cells=16, active=2, record_width=4, seed=3)
    params = cortex.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * 0.3
    cortex.set_parameters(params)
    steering = _steering(2 * b + 1, seed=4, outputs=1)
    gaze = Gaze(b, sigma=0.5, cut=2.0, lamp=0.3, span=0.4, price=0.05, start=0.3)
    brain = Steered(cortex, steering, gaze)
    o, a, y = rng.random((2, 1, b)), np.zeros((2, 1, 1)), rng.normal(size=(2, 1, b))
    path = brain.run(o, a, y)
    centre = brain.boundary().weighing
    np.testing.assert_allclose(centre, ((0.3 + 0.4 * np.tanh(path.steering_output[:, 0, 0])) + np.pi) % (2 * np.pi) - np.pi)
    gains, jac = gaze.profile(centre)
    np.testing.assert_allclose(path.gains[:, 0], gains)
    assert gains.max() <= min(1.0, 0.3 / 0.5) and np.any(gains == 0.0)  # a cut window
    assert path.price == pytest.approx(0.5 * 0.05 * np.mean((0.4 * np.tanh(path.steering_output[:, 0, 0])) ** 2))
    start = brain._fresh(2)
    grad = cortex.observe(o, a, y, gains=path.gains, state=start.cortex, rate=0.0, write=False, keep_live=True)
    dy = gaze.pull(path.steering_output, path.gains, grad.gain_gradient, [start.weighing])

    def objective(ys):
        g = brain._weigh_chunk(ys, start.weighing)
        replay = cortex.assimilate(o, a, gains=g, state=start.cortex, keep_live=True)
        return cortex._loss(replay.slow_output, y) + gaze.price(ys, [start.weighing])

    eps = 1e-6
    for i in range(2):
        up, dn = path.steering_output.copy(), path.steering_output.copy()
        up[i, 0, 0] += eps
        dn[i, 0, 0] -= eps
        fd = (objective(up) - objective(dn)) / (2 * eps)
        assert abs(fd - dy[i, 0, 0]) <= 1e-5 * max(1.0, abs(fd))
    taught = brain.run(o, a, y, rate=2.0, state=start)
    assert taught.reason in ("updated", "no_decreasing_parameter_step")


def test_custody_keeps_both_patches_the_weighing_and_the_step():
    rng = np.random.default_rng(17)
    brain = _brain(17)
    o, a, y = _data(rng)
    brain.run(o, a, y, rate=8.0)
    twin = Steered.restore(brain.snapshot())
    twin.cortex.set_implied_reading(_implied, units=[0.1, 0.1])  # a declaration, not part of a snapshot
    assert twin.step_size == brain.step_size and twin.evidence == (1,) and twin.channels == 9
    assert twin.weighing.to_dict() == brain.weighing.to_dict()
    np.testing.assert_allclose(twin.run(o, a, state=twin._fresh(3)).output, brain.run(o, a, state=brain._fresh(3)).output)
    assert np.array_equal(twin.steering.port.mask, brain.steering.port.mask) if brain.steering.port.mask is not None else twin.steering.port.mask is None
    rule = Steered(_cortex(17), weighing=Rule(lambda r: np.ones((len(r), 2)), macs=3))
    with pytest.raises(ValueError, match="rule="):
        Steered.restore(rule.snapshot())
    back = Steered.restore(rule.snapshot(), rule=lambda r: np.ones((len(r), 2)))
    assert back.rule is not None and back.rule.macs == 3 and back.steering is None
    gaze = Steered(_cortex(17), None)
    assert Steered.restore(gaze.snapshot()).weighing is None
