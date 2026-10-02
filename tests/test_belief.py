"""The belief patch: the adjoint against finite differences, imagination's contract, the store,
custody, scene and action meeting inside the repair, the gain per block, the readback, the
admitted step, the masks, the external output gradient and the torch twin."""

from __future__ import annotations

import numpy as np
import pytest

from cadence.belief import BeliefPatch, BeliefReadback
from cadence.ports import DenseBlock, MapBlock, StructuredPort


def _patch(seed: int = 0, iterations: int = 2) -> BeliefPatch:
    port = StructuredPort(
        1 * 5 * 5 + 3, [MapBlock(0, 1, 5, 5, 2, 3, 2), DenseBlock(25, 3, 4)], broadcast=(25, 3)
    )
    return BeliefPatch(
        port,
        actions=3,
        belief=6,
        outputs=4,
        iterations=iterations,
        damping=0.5,
        cells=64,
        active=4,
        record_width=8,
        seed=seed,
    )


def _data(rng, n=2, t=4, patch=None):
    patch = patch or _patch()
    o = rng.random((n, t, patch.inputs))
    a = rng.random((n, t, patch.actions))
    y = rng.normal(size=(n, t, patch.outputs))
    return o, a, y


def _readout(patch, rng, scale=0.3):
    """A nonzero readout so every path carries gradient."""
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * scale
    patch.set_parameters(params)


def _finite_differences(patch, o, a, y, boundary, rng, per_key=4, eps=1e-6, **kw):
    """The worst relative error of the adjoint against central differences on every parameter
    group, and on the gains when ``gains`` is given, for the loss ``observe`` reports."""
    result = patch.observe(o, a, y, rate=0.0, write=False, state=boundary, **kw)
    scratch = BeliefPatch.restore(patch.snapshot())
    base = patch.parameters()

    def loss(params, **override):
        scratch.set_parameters(params)
        return scratch.observe(
            o, a, y, rate=0.0, write=False, state=boundary, **{**kw, **override}
        ).initial_loss

    worst = 0.0
    for key in base:
        for _ in range(per_key):
            idx = tuple(rng.integers(0, s) for s in base[key].shape)
            up = {k: v.copy() for k, v in base.items()}
            dn = {k: v.copy() for k, v in base.items()}
            up[key][idx] += eps
            dn[key][idx] -= eps
            fd = (loss(up) - loss(dn)) / (2 * eps)
            worst = max(worst, abs(fd - result.delta[key][idx]) / max(1.0, abs(fd)))
    worst_gain = 0.0
    if kw.get("gains") is not None:
        gains = np.asarray(kw["gains"], dtype=float)
        for _ in range(3 * per_key):
            idx = tuple(rng.integers(0, s) for s in gains.shape)
            up, dn = gains.copy(), gains.copy()
            up[idx] += eps
            dn[idx] -= eps
            fd = (loss(base, gains=up) - loss(base, gains=dn)) / (2 * eps)
            worst_gain = max(worst_gain, abs(fd - result.gain_gradient[idx]) / max(1.0, abs(fd)))
    return worst, worst_gain, result


def test_the_adjoint_matches_finite_differences_through_iterations_and_the_transition():
    rng = np.random.default_rng(0)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    _readout(patch, rng)  # a nonzero readout and a live boundary so every path carries gradient
    patch.reset()
    patch.assimilate(o[:, :1], a[:, :1])
    boundary = patch.state
    worst, _, _ = _finite_differences(patch, o, a, y, boundary, rng)
    assert worst < 1e-5


def test_imagination_consumes_no_observation_and_changes_nothing():
    rng = np.random.default_rng(1)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    patch.observe(o, a, y, rate=0.1)
    before = patch.snapshot()
    path = patch.imagine(rng.random((2, 5, patch.actions)))
    after = patch.snapshot()
    assert path.belief.shape == (2, 5, patch.belief) and np.isfinite(path.output).all()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    assert np.allclose(path.residual, 0.0)  # nothing to repair without evidence


def test_one_write_moves_the_read_toward_the_outcome():
    rng = np.random.default_rng(2)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    first = patch.observe(o, a, y, rate=0.0, write=True)
    patch.reset()
    again = patch.assimilate(o, a)
    assert first.writes == 8
    assert np.abs(again.output - y).mean() < np.abs(first.path.output - y).mean()


def test_custody_and_state():
    rng = np.random.default_rng(3)
    patch = _patch()
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    patch.observe(o, a, y, rate=0.05)
    copy = BeliefPatch.restore(patch.snapshot())
    assert np.allclose(copy.state, patch.state)
    more = rng.random((2, 3, patch.actions))
    assert np.allclose(copy.imagine(more).output, patch.imagine(more).output)


def test_scene_and_action_meet_inside_the_repair():
    """The mixed difference of the belief over (observation, action) is nonzero: the repair map
    reads the encoded evidence and the transition's expectation together."""
    rng = np.random.default_rng(4)
    patch = _patch()
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape)
    patch.set_parameters(params)
    o1, o2 = rng.random((1, 1, patch.inputs)), rng.random((1, 1, patch.inputs))
    a1, a2 = rng.random((1, 1, patch.actions)), rng.random((1, 1, patch.actions))

    def y(o, a):
        patch.reset()
        return patch.assimilate(o, a).slow_output[0, 0]

    mixed = np.abs(y(o1, a1) - y(o1, a2) - y(o2, a1) + y(o2, a2)).max()
    assert mixed > 1e-6


def test_the_torch_backend_matches_the_library_forward_and_gradient():
    torch = pytest.importorskip("torch")
    from cadence.belief_torch import TorchBelief

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(5)
    patch = _patch()
    params = patch.parameters()
    params["C"] = rng.normal(size=params["C"].shape) * 0.3
    patch.set_parameters(params)
    twin = TorchBelief(
        patch.port,
        patch.actions,
        patch.belief,
        patch.outputs,
        iterations=patch.iterations,
        damping=patch.damping,
        record_width=patch.record_width,
    )
    twin.load(patch.parameters())
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    result = patch.observe(o, a, y, rate=0.0, write=False)
    beliefs, outputs = twin(torch.as_tensor(o), torch.as_tensor(a))
    assert np.allclose(beliefs.detach().numpy(), result.path.belief, atol=1e-10)
    assert np.allclose(outputs.detach().numpy(), result.path.slow_output, atol=1e-10)
    loss = 0.5 * ((outputs - torch.as_tensor(y)) ** 2).mean()
    loss.backward()
    assert np.isclose(loss.item(), result.path.loss)
    assert np.allclose(twin.C.grad.numpy(), result.delta["C"], atol=1e-9)
    assert np.allclose(twin.F.grad.numpy(), result.delta["F"], atol=1e-9)
    assert np.allclose(twin.T.grad.numpy(), result.delta["T"], atol=1e-9)
    assert np.allclose(
        np.concatenate([w.grad.numpy().ravel() for w in twin.E.weights]),
        result.delta["E"],
        atol=1e-9,
    )
    again = BeliefPatch.restore(patch.snapshot())
    again.set_parameters(twin.export())
    again.reset()
    assert np.allclose(again.assimilate(o, a).slow_output, result.path.slow_output)
    imagined = (
        twin(None, torch.as_tensor(a), state=torch.as_tensor(result.path.final_state))[1]
        .detach()
        .numpy()
    )
    patch.reset()
    assert np.allclose(
        imagined, patch.imagine(a, state=result.path.final_state).slow_output, atol=1e-10
    )
    torch.set_default_dtype(torch.float32)


def test_a_boundary_state_starts_the_moments_and_the_final_belief_goes_live():
    rng = np.random.default_rng(21)
    patch = _patch(seed=21)
    o, a, y = _data(rng, patch=patch)
    patch.reset()
    patch.assimilate(o[:, :2], a[:, :2])
    live = patch.state.copy()
    other = rng.normal(size=live.shape)
    from_live = patch.assimilate(o[:, 2:], a[:, 2:])
    patch._state = live.copy()
    from_other = patch.assimilate(o[:, 2:], a[:, 2:], state=other)
    assert not np.allclose(from_live.belief, from_other.belief)
    np.testing.assert_allclose(patch.state, from_other.final_state)
    patch._state = live.copy()
    seen = patch.observe(o[:, 2:], a[:, 2:], y[:, 2:], rate=0.0, write=False, state=other)
    np.testing.assert_allclose(seen.path.belief, from_other.belief, atol=1e-12)
    np.testing.assert_allclose(patch.state, from_other.final_state)


def test_the_step_is_the_last_repair_move_per_unit_and_its_norm_is_the_residual():
    rng = np.random.default_rng(22)
    patch = _patch(seed=22)
    o, a, _ = _data(rng, patch=patch)
    patch.reset()
    path = patch.assimilate(o, a)
    assert path.step.shape == path.belief.shape
    np.testing.assert_allclose(np.linalg.norm(path.step, axis=-1), path.residual, atol=1e-12)
    imagined = patch.imagine(a)
    assert np.all(imagined.step == 0.0) and np.all(imagined.residual == 0.0)


# ---------------------------------------------------------------- the gain per block
def test_a_gain_per_block_enters_the_repair_and_its_gradient_matches_finite_differences():
    rng = np.random.default_rng(30)
    patch = _patch(seed=30)
    n, t = 3, 5
    o, a, y = _data(rng, n=n, t=t, patch=patch)
    _readout(patch, rng)
    patch.reset()
    patch.assimilate(o[:, :1], a[:, :1])
    boundary = patch.state
    gains = np.exp(rng.normal(size=(n, t, patch.block_count)) * 0.5)
    # the gain acts inside the repair: the belief and the store's code change with it
    plain = patch.assimilate(o, a, state=boundary)
    gained = patch.assimilate(o, a, state=boundary, gains=gains)
    assert not np.allclose(plain.belief, gained.belief)
    assert np.array_equal(plain.gains, np.ones((n, t, patch.block_count)))
    assert np.array_equal(gained.gains, gains)
    raw = np.tanh(patch.port.apply(o[:, 0], patch._blocks()) + patch.parameters()["e_b"])
    np.testing.assert_allclose(plain.evidence[:, 0], raw, atol=1e-12)
    np.testing.assert_allclose(gained.evidence[:, 0], raw * gains[:, 0][:, patch._unit_block], atol=1e-12)
    # one gain per block, or one per stream and block, is the same gain at every moment
    shared = patch.assimilate(o, a, state=boundary, gains=gains[0, 0])
    expanded = patch.assimilate(o, a, state=boundary, gains=np.broadcast_to(gains[0, 0], (n, t, 2)))
    np.testing.assert_array_equal(shared.output, expanded.output)
    per_stream = patch.assimilate(o, a, state=boundary, gains=gains[:, 0])
    expanded = patch.assimilate(o, a, state=boundary, gains=np.broadcast_to(gains[:, 0][:, None], (n, t, 2)))
    np.testing.assert_array_equal(per_stream.output, expanded.output)
    # the parameter gradient under the gains and the gradient into the gains
    worst, worst_gain, result = _finite_differences(patch, o, a, y, boundary, rng, gains=gains)
    assert worst < 1e-5 and worst_gain < 1e-5
    assert result.gain_gradient.shape == (n, t, patch.block_count)
    assert np.abs(result.gain_gradient).max() > 0
    with pytest.raises(ValueError, match="gains"):
        patch.assimilate(o, a, gains=np.ones(3))
    with pytest.raises(ValueError, match="gains"):
        patch.observe(o, a, y, rate=0.0, gains=np.ones((n, t + 1, 2)))


# ---------------------------------------------------------------- the readback
def test_the_path_carries_the_readback_of_every_moment():
    rng = np.random.default_rng(31)
    patch = _patch(seed=31)
    n, t = 3, 4
    o, a, _ = _data(rng, n=n, t=t, patch=patch)
    _readout(patch, rng)
    patch.reset()
    path = patch.assimilate(o, a, probe=True)
    assert path.surprise is None  # no implied reading declared
    # the store's code at the final reading
    reading = patch._reading(path.evidence[:, 1], path.belief[:, 1])
    np.testing.assert_array_equal(path.code[:, 1], patch.records.code(reading, valued=False)[0])
    # the residual-alone probe: one evaluation of the repair map with one block heard
    params = patch.parameters()
    raw = np.tanh(patch.port.apply(o[:, 0], patch._blocks()) + params["e_b"])
    p = path.expectation[:, 0]
    for b in range(patch.block_count):
        alone = np.zeros_like(raw)
        sel = patch._block_slices[b]
        alone[:, sel] = raw[:, sel]
        u = np.concatenate([p, alone, p, np.zeros((n, patch.record_width)), np.ones((n, 1))], axis=-1)
        h = np.tanh(u @ params["F"].T + params["f_b"])
        np.testing.assert_allclose(path.residual_alone[:, 0, b], np.linalg.norm(h - p, axis=-1), atol=1e-12)
    # the surprise against the reading the previous belief's slow readout implies, on the
    # channels the map implies (the map leaves the dense block NaN), in persistence units
    mixing = rng.normal(size=(patch.outputs, patch.inputs))

    def implied(outputs):
        reading = outputs @ mixing
        reading[:, 25:] = np.nan
        return reading

    patch.set_implied_reading(implied, units=[0.5, 2.0])
    patch.reset()
    path = patch.assimilate(o, a)
    previous = np.concatenate([np.zeros((n, 1, patch.belief)), path.belief[:, :-1]], axis=1)
    for k in range(t):
        predicted = implied(previous[:, k] @ params["C"].T + params["c"])
        surprise = np.sqrt(np.mean((predicted[:, :25] - o[:, k, :25]) ** 2, axis=-1) / 0.5)
        np.testing.assert_allclose(path.surprise[:, k, 0], surprise, atol=1e-12)
    assert np.all(path.surprise[:, :, 1] == 0.0)
    # one moment's readback before its repair, from a given state, equals the path's
    before = patch.snapshot()
    readback = patch.readback(o[:, 2], a[:, 2], state=path.belief[:, 1])
    assert isinstance(readback, BeliefReadback)
    np.testing.assert_allclose(readback.expectation, path.expectation[:, 2], atol=1e-12)
    np.testing.assert_allclose(readback.surprise, path.surprise[:, 2], atol=1e-12)
    patch.reset()
    probed = patch.assimilate(o, a, probe=True)
    np.testing.assert_allclose(readback.residual_alone, probed.residual_alone[:, 2], atol=1e-12)
    after = patch.snapshot()
    assert all(np.array_equal(before[k], after[k]) for k in before if k != "state")
    # a row that observes nothing is not surprised; imagination carries no readback
    masked = patch.assimilate(o, a, observed=np.array([[True, False, True, True]] * n), probe=True)
    assert np.all(masked.surprise[:, 1] == 0.0) and np.all(masked.residual_alone[:, 1] == 0.0)
    imagined = patch.imagine(a, gains=np.array([2.0, 0.5]))
    assert imagined.residual_alone is None and imagined.surprise is None
    assert np.all(imagined.evidence == 0.0) and np.all(imagined.gains == [2.0, 0.5])
    patch.set_implied_reading(None)
    assert patch.assimilate(o, a).surprise is None
    with pytest.raises(ValueError, match="units"):
        patch.set_implied_reading(implied, units=[1.0, 0.0])


# ---------------------------------------------------------------- the admitted step
def test_backtracking_admits_a_rate_that_diverges_without_it():
    rng = np.random.default_rng(32)
    plain = _patch(seed=32)
    _readout(plain, rng)
    admitted = BeliefPatch.restore(plain.snapshot())
    o, a, y = _data(rng, n=2, t=6, patch=plain)
    rate = 100.0
    losses = []
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(4):
            plain.reset()
            losses.append(
                plain.observe(o, a, y, rate=rate, write=False, backtrack=False).initial_loss
            )
    assert losses[-1] is None or losses[-1] > 100 * losses[0]  # the plain step diverges
    previous = None
    accepted = []
    for _ in range(6):
        admitted.reset()
        result = admitted.observe(o, a, y, rate=rate, write=False, backtrack=True)
        assert result.updated and result.reason == "updated"
        assert result.replay_calls >= 1
        assert result.final_loss < result.initial_loss  # the loss never rises on an accepted step
        if previous is not None:
            assert result.initial_loss == previous  # the replayed loss is the next chunk's loss
        previous = result.final_loss
        accepted.append(result.accepted_rate)
    assert all(0 < r < rate for r in accepted)
    assert admitted.updates == 6
    # the admission starts from twice the last admitted step: after the first chunk the start
    # is below the rate, so the replays fall and every accepted step stays within the ceiling
    assert all(later <= 2.0 * earlier + 1e-12 for earlier, later in zip(accepted[:-1], accepted[1:], strict=True))
    assert admitted.step_size == accepted[-1]
    # no step reports none
    admitted.reset()
    still = admitted.observe(o, a, y, rate=0.0, write=False, backtrack=True)
    assert not still.updated and still.reason == "no_step" and still.accepted_rate is None
    assert still.replay_calls == 0
    with pytest.raises(ValueError, match="backtrack"):
        admitted.observe(o, a, y, rate=1.0, backtrack="yes")


# ---------------------------------------------------------------- the masks
def test_a_loss_weight_and_a_per_row_mask_match_finite_differences():
    rng = np.random.default_rng(33)
    patch = _patch(seed=33)
    n, t = 3, 5
    o, a, y = _data(rng, n=n, t=t, patch=patch)
    _readout(patch, rng)
    patch.reset()
    patch.assimilate(o[:, :1], a[:, :1])
    boundary = patch.state
    weight = rng.random((n, t))
    weight[0, 1] = weight[2, 3] = 0.0
    observed = rng.random((n, t)) > 0.3
    observed[:, 0] = True
    observed[1, 2] = False
    gains = np.exp(rng.normal(size=(n, t, 2)) * 0.5)
    worst, worst_gain, result = _finite_differences(
        patch, o, a, y, boundary, rng, gains=gains, loss_weight=weight, observed=observed
    )
    assert worst < 1e-5 and worst_gain < 1e-5
    # the loss is the weighted error normalized by the weight's sum
    slow = result.path.slow_output
    direct = 0.5 * np.sum(weight[:, :, None] * (slow - y) ** 2) / (patch.outputs * weight.sum())
    assert np.isclose(result.initial_loss, direct)
    # a weight per moment is the same weight on every stream; ones are the plain loss
    per_moment = patch.observe(o, a, y, rate=0.0, write=False, state=boundary, loss_weight=weight[0])
    rows = patch.observe(o, a, y, rate=0.0, write=False, state=boundary, loss_weight=np.tile(weight[0], (n, 1)))
    assert per_moment.initial_loss == rows.initial_loss
    ones = patch.observe(o, a, y, rate=0.0, write=False, state=boundary, loss_weight=np.ones(t))
    none = patch.observe(o, a, y, rate=0.0, write=False, state=boundary)
    assert np.isclose(ones.initial_loss, none.initial_loss)
    assert all(np.allclose(ones.delta[k], none.delta[k]) for k in none.delta)
    # a row that observes nothing keeps its expectation while the other rows repair
    path = result.path
    assert np.allclose(path.belief[1, 2], path.expectation[1, 2]) and path.residual[1, 2] == 0.0
    assert np.all(path.evidence[1, 2] == 0.0)
    assert path.residual[0, 2] > 0.0 and path.residual[2, 2] > 0.0
    time_mask = np.array([True, False, True, True, False])
    as_rows = patch.assimilate(o, a, observed=np.tile(time_mask, (n, 1)), state=boundary)
    as_time = patch.assimilate(o, a, observed=time_mask, state=boundary)
    np.testing.assert_array_equal(as_rows.belief, as_time.belief)
    # a moment of weight zero is not written; a row that observes nothing is not written
    fresh = BeliefPatch.restore(patch.snapshot())
    fresh.reset()
    written = fresh.observe(o, a, y, rate=0.0, write=True, loss_weight=weight, observed=observed)
    assert written.writes == int(np.sum(observed & (weight > 0)))
    with pytest.raises(ValueError, match="loss_weight"):
        patch.observe(o, a, y, rate=0.0, loss_weight=np.zeros(t))
    with pytest.raises(ValueError, match="observed"):
        patch.assimilate(o, a, observed=np.ones((n + 1, t), dtype=bool))


# ---------------------------------------------------------------- the external output gradient
def test_an_external_output_gradient_matches_the_target_form():
    rng = np.random.default_rng(34)
    patch = _patch(seed=34)
    n, t = 3, 4
    o, a, y = _data(rng, n=n, t=t, patch=patch)
    _readout(patch, rng)
    boundary = np.zeros((n, patch.belief))
    gains = np.exp(rng.normal(size=(n, t, 2)) * 0.5)
    weight = rng.random((n, t))
    taught = patch.observe(o, a, y, rate=0.0, write=False, gains=gains, loss_weight=weight, state=boundary)
    precision = np.ones(patch.outputs)
    dy = precision * (taught.path.slow_output - y) * (weight[:, :, None] / (patch.outputs * weight.sum()))
    seam = patch.observe(o, a, output_gradient=dy, rate=0.0, gains=gains, state=boundary)
    assert seam.path.loss is None and seam.initial_loss is None and seam.writes == 0
    assert all(np.allclose(seam.delta[k], taught.delta[k], atol=1e-14) for k in taught.delta)
    np.testing.assert_allclose(seam.gain_gradient, taught.gain_gradient, atol=1e-14)
    # the adjoint of the linear functional sum(dy * output) on every parameter group
    scratch = BeliefPatch.restore(patch.snapshot())
    base = patch.parameters()

    def functional(params):
        scratch.set_parameters(params)
        return float(np.sum(dy * scratch.observe(o, a, output_gradient=dy, rate=0.0, gains=gains, state=boundary).path.slow_output))

    worst = 0.0
    for key in base:
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in base[key].shape)
            up = {k: v.copy() for k, v in base.items()}
            dn = {k: v.copy() for k, v in base.items()}
            up[key][idx] += 1e-6
            dn[key][idx] -= 1e-6
            fd = (functional(up) - functional(dn)) / 2e-6
            worst = max(worst, abs(fd - seam.delta[key][idx]) / max(1.0, abs(fd)))
    assert worst < 1e-5
    # a step under the seam moves the parameters; the admission needs a target
    stepped = patch.observe(o, a, output_gradient=dy, rate=0.5, gains=gains, state=boundary)
    assert stepped.updated and stepped.accepted_rate == 0.5
    with pytest.raises(ValueError, match="backtrack"):
        patch.observe(o, a, output_gradient=dy, rate=0.5, backtrack=True)
    with pytest.raises(ValueError, match="loss_weight"):
        patch.observe(o, a, output_gradient=dy, rate=0.0, loss_weight=weight)
    with pytest.raises(ValueError, match="one of the two"):
        patch.observe(o, a, y, output_gradient=dy, rate=0.0)
    with pytest.raises(ValueError, match="one of the two"):
        patch.observe(o, a, rate=0.0)


# ---------------------------------------------------------------- the torch twin
def test_the_torch_twin_takes_gains_and_a_row_mask():
    torch = pytest.importorskip("torch")
    from cadence.belief_torch import TorchBelief

    torch.set_default_dtype(torch.float64)
    try:
        rng = np.random.default_rng(35)
        patch = _patch(seed=35)
        _readout(patch, rng)
        n, t = 3, 5
        o, a, y = _data(rng, n=n, t=t, patch=patch)
        gains = np.exp(rng.normal(size=(n, t, 2)) * 0.5)
        observed = rng.random((n, t)) > 0.3
        observed[1, 2] = False
        weight = rng.random((n, t))
        twin = TorchBelief(
            patch.port, patch.actions, patch.belief, patch.outputs,
            iterations=patch.iterations, damping=patch.damping, record_width=patch.record_width,
        )
        twin.load(patch.parameters())
        patch.reset()
        result = patch.observe(o, a, y, rate=0.0, write=False, gains=gains, observed=observed, loss_weight=weight)
        g = torch.as_tensor(gains).requires_grad_()
        beliefs, outputs = twin(torch.as_tensor(o), torch.as_tensor(a), gains=g, observed=torch.as_tensor(observed))
        assert np.allclose(beliefs.detach().numpy(), result.path.belief, atol=1e-10)
        assert np.allclose(outputs.detach().numpy(), result.path.slow_output, atol=1e-10)
        w = torch.as_tensor(weight)
        loss = 0.5 * (w[:, :, None] * (outputs - torch.as_tensor(y)) ** 2).sum() / (patch.outputs * w.sum())
        loss.backward()
        assert np.isclose(loss.item(), result.initial_loss)
        for key in ("C", "F", "T", "G"):
            assert np.allclose(getattr(twin, key).grad.numpy(), result.delta[key], atol=1e-9)
        assert np.allclose(
            np.concatenate([v.grad.numpy().ravel() for v in twin.E.weights]), result.delta["E"], atol=1e-9
        )
        assert np.allclose(g.grad.numpy(), result.gain_gradient, atol=1e-9)
        # one gain per block receives the summed gradient
        shared = torch.as_tensor(gains[0, 0]).requires_grad_()
        _, outputs = twin(torch.as_tensor(o), torch.as_tensor(a), gains=shared)
        outputs.sum().backward()
        patch.reset()
        summed = patch.observe(o, a, output_gradient=np.ones((n, t, patch.outputs)), rate=0.0, gains=gains[0, 0])
        assert np.allclose(shared.grad.numpy(), summed.gain_gradient.sum(axis=(0, 1)), atol=1e-9)
        with pytest.raises(ValueError, match="gains"):
            twin(torch.as_tensor(o), torch.as_tensor(a), gains=torch.ones(3))
    finally:
        torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------- the composition release
def test_the_readback_carries_the_raw_evidence_and_encode_is_public():
    rng = np.random.default_rng(41)
    patch = _patch(seed=41)
    o, a, _ = _data(rng, n=2, t=3, patch=patch)
    params = patch.parameters()
    raw = np.tanh(patch.port.apply(o[:, 0], patch._blocks()) + params["e_b"])
    moment = patch.readback(o[:, 0], a[:, 0], state=np.zeros((2, patch.belief)))
    np.testing.assert_allclose(moment.evidence, raw, atol=1e-12)
    encoded = patch.encode(o)  # (batch, time, encoded)
    np.testing.assert_allclose(
        encoded, np.tanh(patch.port.apply(o, patch._blocks()) + params["e_b"]), atol=1e-12
    )
    gains = np.tile([1.5, 0.5], (2, 3, 1))
    patch.reset()
    path = patch.assimilate(o, a, gains=gains)  # the path's evidence is the raw evidence gained
    np.testing.assert_allclose(path.evidence[:, 0], raw * gains[:, 0][:, patch._unit_block], atol=1e-12)
    with pytest.raises(ValueError, match="observations"):
        patch.encode(np.zeros((2, patch.inputs + 1)))


def test_a_step_on_a_target_is_admitted_by_default_and_a_step_on_a_gradient_is_plain():
    rng = np.random.default_rng(42)
    patch = _patch(seed=42)
    _readout(patch, rng)
    o, a, y = _data(rng, n=2, t=4, patch=patch)
    patch.reset()
    taught = patch.observe(o, a, y, rate=100.0, write=False)
    assert taught.updated and taught.replay_calls >= 1
    assert taught.accepted_rate < 100.0 and taught.final_loss < taught.initial_loss
    dy = rng.normal(size=(2, 4, patch.outputs))
    patch.reset()
    seam = patch.observe(o, a, output_gradient=dy, rate=0.5, write=False)  # plain, no replay
    assert seam.updated and seam.accepted_rate == 0.5 and seam.replay_calls == 0
    assert seam.final_loss is None and seam.writes == 0
    with pytest.raises(ValueError, match="backtrack needs a target"):
        patch.observe(o, a, output_gradient=dy, rate=0.5, backtrack=True)
    patch.reset()
    plain = patch.observe(o, a, y, rate=0.01, write=False, backtrack=False)
    assert plain.updated and plain.accepted_rate == 0.01 and plain.replay_calls == 0
    with pytest.raises(ValueError, match="backtrack"):
        patch.observe(o, a, y, rate=1.0, backtrack="yes")


def test_the_admitted_step_size_is_kept_across_chunks_and_snapshots():
    rng = np.random.default_rng(43)
    patch = _patch(seed=43)
    _readout(patch, rng)
    o, a, y = _data(rng, n=2, t=4, patch=patch)
    assert patch.step_size is None
    patch.reset()
    first = patch.observe(o, a, y, rate=100.0, write=False)
    assert first.updated and patch.step_size == first.accepted_rate
    patch.reset()
    second = patch.observe(o, a, y, rate=100.0, write=False)
    assert second.updated and second.accepted_rate <= 2.0 * first.accepted_rate + 1e-12
    restored = BeliefPatch.restore(patch.snapshot())
    assert restored.step_size == patch.step_size
    old = patch.snapshot()
    del old["step_size"]  # a snapshot from before the step size travelled
    assert BeliefPatch.restore(old).step_size is None
    patch.reset()  # the live state goes, the step size stays
    assert patch.step_size == second.accepted_rate
    patch.reset_step()
    assert patch.step_size is None
    assert BeliefPatch.restore(patch.snapshot()).step_size is None


def test_keep_live_leaves_the_live_state_alone():
    rng = np.random.default_rng(44)
    patch = _patch(seed=44)
    o, a, y = _data(rng, n=2, t=4, patch=patch)
    patch.reset()
    patch.assimilate(o[:, :2], a[:, :2])
    live = patch.state
    boundary = np.zeros((2, patch.belief))
    replay = patch.assimilate(o, a, state=boundary, keep_live=True)
    assert np.array_equal(patch.state, live)
    assert not np.array_equal(replay.final_state, live)
    patch.observe(o, a, y, rate=0.0, write=False, state=boundary, keep_live=True)
    assert np.array_equal(patch.state, live)
    patch.assimilate(o, a)
    assert not np.array_equal(patch.state, live)


def test_the_cost_counters_count_every_moment_and_every_replay():
    rng = np.random.default_rng(45)
    patch = _patch(seed=45)
    _readout(patch, rng)
    o, a, y = _data(rng, n=2, t=4, patch=patch)
    one = patch.macs_per_moment()
    fi = patch.belief + patch.encoded + patch.belief + patch.record_width + 1
    assert one > 0 and patch.macs_per_moment(probes=True) == one + patch.block_count * patch.belief * fi
    patch.reset_cost()
    patch.reset()
    patch.assimilate(o, a)
    assert patch.cost == {"moments": 8, "macs": 8 * one, "replays": 0}
    patch.imagine(a[:, :3])
    assert patch.cost["moments"] == 14 and patch.cost["macs"] == 14 * one
    patch.reset_cost()
    patch.reset()
    patch.assimilate(o, a, probe=True)
    assert patch.cost["macs"] == 8 * patch.macs_per_moment(probes=True)
    patch.reset_cost()
    patch.reset()
    taught = patch.observe(o, a, y, rate=100.0, write=False)
    assert taught.replay_calls >= 1
    assert patch.cost["replays"] == taught.replay_calls
    assert patch.cost["moments"] == 8 * (1 + taught.replay_calls)
    patch.reset_cost()
    patch.reset()
    patch.observe(o, a, y, rate=0.01, write=False, backtrack=False)
    assert patch.cost == {"moments": 8, "macs": 8 * one, "replays": 0}
