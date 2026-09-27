"""The structured port: dense equivalence, the adjoint, translation, custody."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import RecordPatchNet
from cadence.ports import DenseBlock, MapBlock, StructuredPort


def _port():
    grid = MapBlock(start=0, channels_in=2, height=6, width=7, channels_out=3, kernel=3, stride=2)
    dense = DenseBlock(start=grid.inputs, inputs=5, outputs=4)
    return grid, StructuredPort(grid.inputs + 5, [grid, dense])


def test_apply_transpose_and_gradient_match_the_dense_matrix():
    rng = np.random.default_rng(0)
    grid, port = _port()
    w = port.initial(rng)
    u = rng.normal(size=(2, 3, port.inputs))
    m = port.dense_matrix(w)
    y = port.apply(u, w)
    assert np.allclose(y, u @ m.T)
    v = rng.normal(size=y.shape)
    assert np.allclose(port.transpose(v, w), v @ m)
    g = port.gradient(v, u)

    def f(ws):
        return float(np.sum(v * port.apply(u, ws)))

    for k, gw in enumerate(g):
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in gw.shape)
            up = [x.copy() for x in w]
            dn = [x.copy() for x in w]
            up[k][idx] += 1e-6
            dn[k][idx] -= 1e-6
            fd = (f(up) - f(dn)) / 2e-6
            assert abs(fd - gw[idx]) < 1e-6 * max(1.0, abs(fd))


def test_a_map_block_answers_the_same_thing_wherever_it_is():
    rng = np.random.default_rng(1)
    grid, port = _port()
    w = port.initial(rng)
    a = np.zeros((1, 1, port.inputs))
    b = np.zeros((1, 1, port.inputs))
    x = np.zeros((2, 6, 7))
    x[0, 1, 1] = 1.0
    a[0, 0, : grid.inputs] = x.ravel()
    x[:] = 0.0
    x[0, 3, 3] = 1.0
    b[0, 0, : grid.inputs] = x.ravel()
    ma = port.apply(a, w)[0, 0, : grid.outputs].reshape(3, grid.out_height, grid.out_width)
    mb = port.apply(b, w)[0, 0, : grid.outputs].reshape(3, grid.out_height, grid.out_width)
    assert np.abs(ma).max() > 0
    assert np.allclose(ma[:, 0, 0], mb[:, 1, 1])


def test_a_port_of_dense_blocks_is_the_plain_record_patch():
    rng = np.random.default_rng(2)
    inputs, hidden, outputs = 9, 6, 2
    plain = RecordPatchNet(inputs, hidden, outputs, seed=3, cells=64, active=4)
    ported = RecordPatchNet(
        inputs,
        hidden,
        outputs,
        seed=3,
        cells=64,
        active=4,
        port=StructuredPort(inputs, [DenseBlock(0, inputs, hidden)]),
    )
    pp = plain.parameters()
    ported.set_parameters({**pp, "B": pp["B"].ravel(), "G": pp["G"].ravel()})
    x = rng.normal(size=(2, 7, inputs))
    t = rng.normal(size=(2, 7, outputs))
    assert np.allclose(plain.imagine(x).output, ported.imagine(x).output)
    ra = plain.observe(x, t, rate=0.1)
    rb = ported.observe(x, t, rate=0.1)
    assert np.isclose(ra.initial_loss, rb.initial_loss)
    assert np.allclose(ra.delta["B"].ravel(), rb.delta["B"])
    assert np.allclose(ra.delta["G"].ravel(), rb.delta["G"])
    assert np.allclose(plain.parameters()["B"].ravel(), ported.parameters()["B"])


def test_the_adjoint_through_a_map_block_matches_finite_differences_and_survives_custody():
    rng = np.random.default_rng(5)
    port = StructuredPort(28, [MapBlock(0, 1, 5, 5, 2, 3, 1), DenseBlock(25, 3, 2)])
    net = RecordPatchNet(28, port.outputs, 2, seed=5, cells=32, active=3, port=port)
    x = rng.normal(size=(1, 6, 28))
    t = rng.normal(size=(1, 6, 2))
    net.reset()
    r = net.observe(x, t, rate=0.0, write=False)

    def loss(params):
        m = RecordPatchNet.restore(net.snapshot())
        m.set_parameters(params)
        m.reset()
        return m._free(x, np.zeros((1, m.hidden)), t)[0].slow_loss

    base = net.parameters()
    for key in ("B", "G", "b", "g", "C"):
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in base[key].shape)
            up = {k: v.copy() for k, v in base.items()}
            dn = {k: v.copy() for k, v in base.items()}
            up[key][idx] += 1e-6
            dn[key][idx] -= 1e-6
            fd = (loss(up) - loss(dn)) / 2e-6
            assert abs(fd - r.delta[key][idx]) < 1e-6 * max(1.0, abs(fd))
    m = RecordPatchNet.restore(net.snapshot())
    m.reset()
    net.reset()
    assert np.allclose(m.imagine(x).output, net.imagine(x).output)
    assert m.hidden == port.outputs


def test_broadcast_channels_reach_every_position_and_keep_the_three_maps_consistent():
    """With a broadcast slice, a map block's kernel reads the tiled units; the dense
    equivalent, the transpose and the gradient stay consistent, and the mixed difference
    over grid and broadcast is nonzero under a nonlinearity."""
    rng = np.random.default_rng(3)
    grid = MapBlock(start=0, channels_in=1, height=5, width=5, channels_out=2, kernel=3, stride=2)
    port = StructuredPort(25 + 3, [grid, DenseBlock(25, 3, 2)], broadcast=(25, 3))
    w = port.initial(rng)
    assert w[0].shape == (2, 4, 3, 3)
    u = rng.normal(size=(2, 3, port.inputs))
    m = port.dense_matrix(w)
    assert np.allclose(port.apply(u, w), u @ m.T)
    v = rng.normal(size=(2, 3, port.outputs))
    assert np.allclose(port.transpose(v, w), v @ m)
    g = port.gradient(v, u)

    def f(ws):
        return float(np.sum(v * port.apply(u, ws)))

    for k, gw in enumerate(g):
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in gw.shape)
            up = [x.copy() for x in w]
            dn = [x.copy() for x in w]
            up[k][idx] += 1e-6
            dn[k][idx] -= 1e-6
            assert abs((f(up) - f(dn)) / 2e-6 - gw[idx]) < 1e-6
    net = RecordPatchNet(port.inputs, port.outputs, 2, seed=1, cells=32, active=3, port=port)
    again = RecordPatchNet.restore(net.snapshot())
    x = rng.normal(size=(1, 4, port.inputs))
    net.reset()
    again.reset()
    assert np.allclose(net.imagine(x).output, again.imagine(x).output)

    def y(scene, action):
        z = np.zeros((1, 1, port.inputs))
        z[0, 0, :25] = scene
        z[0, 0, 25:] = action
        net.reset()
        return np.tanh(port.apply(z, net._unpack(net._B))[0, 0])  # the port under its nonlinearity

    s1, s2 = rng.normal(size=25), rng.normal(size=25)
    a1, a2 = rng.normal(size=3), rng.normal(size=3)
    mixed = np.abs(y(s1, a1) - y(s1, a2) - y(s2, a1) + y(s2, a2)).max()
    assert mixed > 1e-6


def test_a_mask_silences_inputs_in_every_map_and_travels_with_custody():
    rng = np.random.default_rng(7)
    blocks = [MapBlock(0, 1, 4, 4, 2, 3, 1), DenseBlock(16, 3, 4)]
    mask = np.ones(19, dtype=bool)
    mask[[2, 5, 17]] = False
    heard = StructuredPort(19, blocks, broadcast=(16, 3), mask=mask)
    plain = StructuredPort(19, blocks, broadcast=(16, 3))
    weights = heard.initial(rng)
    u = rng.random((3, 19))
    v = rng.random((3, heard.outputs))
    np.testing.assert_allclose(heard.apply(u, weights), plain.apply(u * mask, weights))
    back = heard.transpose(v, weights)
    assert np.all(back[:, ~mask] == 0.0)
    np.testing.assert_allclose(back, plain.transpose(v, weights) * mask)
    for g, h in zip(heard.gradient(v, u), plain.gradient(v, u * mask), strict=True):
        np.testing.assert_allclose(g, h)
    loud = u.copy()
    loud[:, ~mask] += 5.0  # a masked input changes nothing
    np.testing.assert_allclose(heard.apply(loud, weights), heard.apply(u, weights))
    again = StructuredPort.from_dict(heard.to_dict())
    assert again.mask is not None and np.array_equal(again.mask, mask)
    assert StructuredPort.from_dict(plain.to_dict()).mask is None
    with pytest.raises(ValueError, match="mask"):
        StructuredPort(19, blocks, mask=np.ones(18, dtype=bool))
