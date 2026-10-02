"""Record patches joined by ports: parity with a single record patch, finite-difference parity of the
joint adjoint across the seams, a hub that learns only through its ports, custody and isolation."""
import numpy as np
import pytest

from cadence import JointRecordPatches, Port, RecordPatchNet
from cadence.record_ports import build


def _streams(seed, batch=3, horizon=7, v=5):
    rng = np.random.default_rng(seed)
    xa = np.eye(v)[rng.integers(v, size=(batch, horizon))]
    xb = np.eye(v)[rng.integers(v, size=(batch, horizon))]
    ta = np.eye(4)[rng.integers(4, size=(batch, horizon))]
    tb = np.eye(4)[rng.integers(4, size=(batch, horizon))]
    return [xa, xb], [ta, tb]


def _moved(brain, seed, scale=0.4):
    rng = np.random.default_rng(seed)
    for net in brain.cortices:
        net.set_parameters({k: v + scale * rng.normal(size=v.shape) for k, v in net.parameters().items()})


def test_independent_cortices_equal_the_library():
    xs, ts = _streams(1)
    brain = build([6, 8], [5, 5], [4, 4], [], seed=3, cells=256, active=8, groups=[(4,), (4,)])
    _moved(brain, 2)
    twins = [RecordPatchNet.restore(net.snapshot()) for net in brain.cortices]
    ours = brain.observe(xs, ts, rate=0.0, write=True)
    for i, twin in enumerate(twins):
        theirs = twin.observe(xs[i], ts[i], rate=0.0, write=True)
        np.testing.assert_allclose(ours.settled.paths[i].output, theirs.prediction.output, atol=1e-12)
        assert abs(ours.settled.paths[i].slow_loss - theirs.prediction.slow_loss) < 1e-12
        for k in theirs.delta:
            np.testing.assert_allclose(ours.delta[i][k], theirs.delta[k], atol=1e-12, rtol=1e-9)
        np.testing.assert_allclose(brain.cortices[i].records.tables["y"], twin.records.tables["y"], atol=1e-12)
        assert brain.cortices[i]._input_norm == twin._input_norm


def test_delayed_plain_port_equals_the_library_given_the_port_values():
    xs, ts = _streams(2)
    ports = [Port(0, 1, 1, 3), Port(1, 0, 2, 2)]
    brain = build([6, 8], [5, 5], [4, 4], ports, seed=5, rounds=1, cross_adjoint=False, cells=256, active=8, groups=[(4,), (4,)])
    _moved(brain, 4)
    twins = [RecordPatchNet.restore(net.snapshot()) for net in brain.cortices]
    ours = brain.observe(xs, ts, rate=0.0, write=False)
    for i, twin in enumerate(twins):
        theirs = twin.observe(ours.settled.inputs[i], ts[i], rate=0.0, write=False)
        np.testing.assert_allclose(ours.settled.paths[i].output, theirs.prediction.output, atol=1e-12)
        for k in theirs.delta:
            np.testing.assert_allclose(ours.delta[i][k], theirs.delta[k], atol=1e-12, rtol=1e-9)
    # round one carries the context of the moment before
    for port in ports:
        held = (ours.settled.hidden[port.source] * brain.cortices[port.source]._scale)[:, :-1, port.start:port.start + port.width]
        offset = next(o for q, o in brain.incoming[port.target] if q is port)
        carried = ours.settled.p[port.target][:, 1:, 0, offset:offset + port.width]
        np.testing.assert_allclose(carried, held, atol=1e-12)


@pytest.mark.parametrize("rounds,damping,groups", [(1, 1.0, True), (3, 1.0, True), (3, 0.5, True), (2, 1.0, False), (4, 1.0, True)])
def test_joint_adjoint_matches_finite_differences(rounds, damping, groups):
    xs, ts = _streams(3)
    if not groups:
        ts = [t + 0.3 * np.random.default_rng(9).normal(size=t.shape) for t in ts]
    ports = [Port(0, 1, 1, 3), Port(1, 0, 2, 2)]
    brain = build([6, 8], [5, 5], [4, 4], ports, seed=7, rounds=rounds, damping=damping, cross_adjoint=True,
                  cells=256, active=8, groups=[(4,), (4,)] if groups else None)
    _moved(brain, 8)
    brain.reset()
    delta = brain.observe(xs, ts, rate=0.0, write=False).delta
    base = brain.parameters()
    rng = np.random.default_rng(11)
    worst = 0.0
    for i in range(2):
        for name in base[i]:
            for _ in range(3):
                index = tuple(rng.integers(n) for n in base[i][name].shape)
                losses = []
                for sign in (1.0, -1.0):
                    trial = [{k: v.copy() for k, v in p.items()} for p in base]
                    trial[i][name][index] += sign * 1e-6
                    for net, p in zip(brain.cortices, trial, strict=True):
                        net.set_parameters(p)
                    brain.reset()
                    losses.append(brain.observe(xs, ts, rate=0.0, write=False).settled.slow_loss)
                numeric = (losses[0] - losses[1]) / 2e-6
                err = abs(numeric - delta[i][name][index]) / max(1.0, abs(numeric))
                worst = max(worst, err)
                assert err < 1e-6, (i, name, numeric, delta[i][name][index])
    print("worst relative error", worst)


def test_hub_learns_only_through_the_seams():
    """A third cortex with no observation and a zero target gets a gradient only through the ports."""
    xs, ts = _streams(4)
    hub_x = np.zeros((xs[0].shape[0], xs[0].shape[1], 1))
    ports = [Port(0, 2, 0, 3), Port(1, 2, 0, 3), Port(2, 0, 0, 4), Port(2, 1, 0, 4)]
    brain = build([6, 6, 5], [5, 5, 1], [4, 4, 1], ports, seed=9, rounds=3, cells=256, active=8, groups=[(4,), (4,), None])
    _moved(brain, 10)
    hub = brain.cortices[2]
    hub.set_parameters({**hub.parameters(), "C": np.zeros_like(hub._C), "c": np.zeros_like(hub._c)})  # no own loss: the hub's output equals its zero target
    brain.reset()
    result = brain.observe(xs + [hub_x], ts + [np.zeros((xs[0].shape[0], xs[0].shape[1], 1))], rate=0.0, write=False)
    assert np.abs(result.delta[2]["B"]).sum() > 0
    brain.cross_adjoint = False
    brain.reset()
    plain = brain.observe(xs + [hub_x], ts + [np.zeros((xs[0].shape[0], xs[0].shape[1], 1))], rate=0.0, write=False)
    assert np.abs(plain.delta[2]["B"]).sum() == 0


def test_custody_and_isolation():
    xs, ts = _streams(5)
    brain = build([6, 8], [5, 5], [4, 4], [Port(0, 1, 0, 4)], seed=12, rounds=2, cells=256, active=8, groups=[(4,), (4,)])
    brain.observe(xs, ts, rate=2.0, backtrack=True)
    before = {k: v.copy() for k, v in brain.snapshot().items()}
    brain.imagine(xs)
    after = brain.snapshot()
    assert all(np.array_equal(before[k], after[k]) for k in before)
    twin = JointRecordPatches.restore(after)
    a = twin.imagine(xs).paths[1].output
    b = brain.imagine(xs).paths[1].output
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert brain.cortices[0].updates == 1 and brain.cortices[1].updates == 1
