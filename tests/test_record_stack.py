from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import RecordPatchStack


def _paths(seed, batch=2, horizon=6, inputs=5, groups=(4,)):
    rng = np.random.default_rng(seed)
    x = np.eye(inputs)[rng.integers(inputs, size=(batch, horizon))]
    target = np.concatenate(
        [np.eye(g)[rng.integers(g, size=(batch, horizon))] for g in groups], axis=-1
    )
    return x, target


@pytest.mark.parametrize("groups", [None, (4,)])
def test_stack_gradient_matches_finite_differences_in_both_patches(groups):
    x, target = _paths(1)
    net = RecordPatchStack(5, 6, 4, lower=7, seed=3, cells=128, active=6, groups=groups)
    rng = np.random.default_rng(2)
    moved = {k: v + 0.3 * rng.normal(size=v.shape) for k, v in net.parameters().items()}
    net.set_parameters(moved)
    delta = net.observe(x, target, rate=0.0, write=False).delta
    base = net.parameters()
    for name in base:
        index = tuple(rng.integers(n) for n in base[name].shape)
        losses = []
        for sign in (1.0, -1.0):
            trial = {k: v.copy() for k, v in base.items()}
            trial[name][index] += sign * 1e-6
            net.set_parameters(trial)
            net.reset()
            losses.append(net.observe(x, target, rate=0.0, write=False).prediction.slow_loss)
        numeric = (losses[0] - losses[1]) / 2e-6
        assert abs(numeric - delta[name][index]) < 2e-7 * max(1.0, abs(numeric)), name


def test_stack_learns_writes_and_continues_from_a_checkpoint():
    x, _ = _paths(4, batch=3, horizon=8)
    symbol = x.argmax(-1)
    # A decidable outcome: the sum of this symbol and the one before it, modulo four.
    target = np.eye(4)[(symbol + np.roll(symbol, 1, axis=1)) % 4]
    net = RecordPatchStack(5, 8, 4, seed=5, cells=2048, active=16, groups=(4,), record_rate=1.0)
    first = net.observe(x, target, rate=2.0, backtrack=True)
    assert first.updated and first.writes == 24
    assert first.final_loss < first.initial_loss
    slow, full = [], []
    for _ in range(20):
        net.reset()
        seen = net.observe(x, target, rate=2.0, backtrack=True).prediction
        slow.append(seen.slow_loss)
        full.append(seen.loss)
    assert slow[-1] < slow[0]  # the slow weights of both patches learn
    # The records hold what they have not learned yet. One moment whose read pushes the true
    # port below zero costs the clipped cross-entropy 27 nats, so the median is compared.
    assert np.median(full[3:]) < slow[-1]
    # The slow weights still move at every step, so a record lags them by one presentation.
    assert np.sum(seen.output.argmax(-1) != target.argmax(-1)) <= 3 < first.writes
    again = RecordPatchStack.restore(net.snapshot())
    zero = (np.zeros((3, 8)), np.zeros((3, 8)))
    assert_array_equal(again.imagine(x, state=zero).output, net.imagine(x, state=zero).output)
    for key, value in net.parameters().items():
        assert_array_equal(again.parameters()[key], value)


def test_stack_state_is_carried_by_advance_and_cleared_by_reset():
    x, _ = _paths(6, batch=1, horizon=4)
    net = RecordPatchStack(5, 6, 4, seed=1, cells=64, active=4)
    whole = net.imagine(np.concatenate((x, x), axis=1)).output[:, 4:]
    net.advance(x)
    assert_allclose(net.advance(x).output, whole, atol=1e-12)
    net.reset()
    zero = (np.zeros((1, 6)), np.zeros((1, 6)))
    assert_allclose(net.advance(x).output, net.imagine(x, state=zero).output)


def test_a_structured_lower_port_learns_and_survives_custody():
    from cadence.ports import DenseBlock, MapBlock, StructuredPort

    rng = np.random.default_rng(9)
    lower = StructuredPort(
        2 * 6 * 6 + 4, [MapBlock(0, 2, 6, 6, 3, 3, 1), DenseBlock(72, 4, 5)], broadcast=(72, 4)
    )
    stack = RecordPatchStack(lower.inputs, 6, 2, lower_port=lower, seed=2, cells=32, active=3)
    assert stack.lower == lower.outputs
    x = rng.normal(size=(1, 5, lower.inputs))
    t = rng.normal(size=(1, 5, 2))
    stack.reset()
    r = stack.observe(x, t, rate=0.0, write=False)
    base = stack.parameters()

    def loss(params):
        m = RecordPatchStack.restore(stack.snapshot())
        m.set_parameters(params)
        m.reset()
        return m._slow_loss(x, t, np.zeros((1, m.lower)), np.zeros((1, m.upper.hidden)))

    for key in ("B1", "G1", "b1", "B", "C"):
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in base[key].shape)
            up = {k: v.copy() for k, v in base.items()}
            dn = {k: v.copy() for k, v in base.items()}
            up[key][idx] += 1e-6
            dn[key][idx] -= 1e-6
            fd = (loss(up) - loss(dn)) / 2e-6
            assert abs(fd - r.delta[key][idx]) < 1e-6 * max(1.0, abs(fd)), key
    again = RecordPatchStack.restore(stack.snapshot())
    stack.reset()
    again.reset()
    assert np.allclose(stack.imagine(x).output, again.imagine(x).output)
    stack.reset()
    assert stack.observe(x, t, rate=0.1).updated
