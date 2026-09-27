"""Direct tests for cadence.fused: every compiled kernel against the NumPy arithmetic it replaces."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

pytest.importorskip("numba")

from cadence import fused  # noqa: E402


def dense_brain(seed: int, adaptation: cd.Adaptation | None = None) -> cd.Brain:
    """Input, hidden and output ranges with reciprocal hidden/output synapses: a blocked layout."""
    rng = np.random.default_rng(seed)
    inputs, hidden, outputs = 5, 7, 3
    pre, post = [], []
    for i in range(inputs):
        for h in range(hidden):
            pre.append(i)
            post.append(inputs + h)
    for h in range(hidden):
        for o in range(outputs):
            pre += [inputs + h, inputs + hidden + o]
            post += [inputs + hidden + o, inputs + h]
    n = inputs + hidden + outputs
    connectome = cd.Connectome.from_synapses(
        n,
        pre=np.array(pre),
        post=np.array(post),
        sign=rng.choice([-1.0, 1.0], size=len(pre)) * rng.uniform(0.2, 1.0, size=len(pre)),
        populations={
            "input": range(0, inputs),
            "hidden": range(inputs, inputs + hidden),
            "output": range(inputs + hidden, n),
        },
    )
    model = cd.learning_neuron_model(dt=1.0).replace(adaptation=adaptation)
    return cd.Brain(connectome, model, bias=rng.normal(0.0, 0.3, n))


def numpy_settle(brain: cd.Brain, drive: np.ndarray, **kw: object) -> cd.BrainState:
    original = brain_module._FUSED
    brain_module._FUSED = False
    try:
        return brain.settle_batch(drive, **kw)  # type: ignore[arg-type]
    finally:
        brain_module._FUSED = original


def test_available_reports_numba() -> None:
    assert fused.available()
    assert set(fused.__all__) == {"available", "fused_residual", "fused_settle"}


@pytest.mark.parametrize("adaptation", [None, cd.Adaptation(tau_steps=15.0, strength=0.3)])
@pytest.mark.parametrize("tolerance", [None, 1e-6])
def test_fused_settle_matches_the_numpy_loop(
    adaptation: cd.Adaptation | None, tolerance: float | None
) -> None:
    brain = dense_brain(2, adaptation)
    assert brain._blocks is not None and brain_module._FUSED
    n = brain.connectome.n
    drive = np.random.default_rng(0).normal(0.0, 1.5, (4, n))
    out = np.array(brain.connectome.populations["output"])
    mask = np.zeros(n)
    mask[out] = 1.0
    target = np.zeros((4, n))
    target[:, out[0]] = 1.0
    keep = np.ones(n)
    keep[out[1]] = 0.0
    for kw in (
        dict(),
        dict(nudge=cd.Nudge(target, mask, 0.4)),
        dict(mask=keep),
        dict(mask=keep, nudge=cd.Nudge(target, mask, 0.2, softmax_temperature=0.3)),
    ):
        compiled = brain.settle_batch(drive, steps=25, tolerance=tolerance, **kw)
        expected = numpy_settle(brain, drive, steps=25, tolerance=tolerance, **kw)
        assert compiled.steps == expected.steps
        np.testing.assert_allclose(compiled.v, expected.v, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(compiled.activation, expected.activation, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(compiled.adaptation, expected.adaptation, rtol=1e-12, atol=1e-12)


def test_fused_settle_from_rest_and_warm_agree_with_numpy() -> None:
    brain = dense_brain(4)
    n = brain.connectome.n
    drive = np.random.default_rng(1).normal(0.0, 1.0, (2, n))
    warm = brain.settle_batch(drive, steps=5)
    compiled = brain.settle_batch(drive, steps=10, state=warm)
    expected = numpy_settle(brain, drive, steps=10, state=warm)
    np.testing.assert_allclose(compiled.v, expected.v, rtol=1e-12, atol=1e-12)


def phases(seed: int, batch: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_minus = rng.uniform(0.0, 1.0, (batch, n))
    return s_minus + rng.normal(0.0, 0.05, (batch, n)), s_minus


def edge_contrast(s_plus: np.ndarray, s_minus: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    a_plus, a_minus = s_plus[:, pre], s_minus[:, pre]
    b_plus, b_minus = s_plus[:, post], s_minus[:, post]
    return a_plus * (b_plus - b_minus) + (a_plus - a_minus) * b_minus


def test_contrast_mean_matches_numpy() -> None:
    n, batch, span = 9, 6, 0.4
    rng = np.random.default_rng(3)
    pre, post = rng.integers(0, n, 20), rng.integers(0, n, 20)
    s_plus, s_minus = phases(3, batch, n)
    edges, neurons = fused.contrast_mean(s_plus, s_minus, pre, post, span)
    np.testing.assert_allclose(edges, edge_contrast(s_plus, s_minus, pre, post).mean(0) / span)
    np.testing.assert_allclose(neurons, (s_plus - s_minus).mean(0) / span)


def test_trace_step_decays_accumulates_and_weights_by_dopamine() -> None:
    n, batch, edges, span, decay = 7, 3, 12, 0.5, 0.8
    rng = np.random.default_rng(5)
    pre, post = rng.integers(0, n, edges), rng.integers(0, n, edges)
    s_plus, s_minus = phases(5, batch, n)
    trace = rng.normal(0.0, 1.0, (batch, edges))
    trace_bias = rng.normal(0.0, 1.0, (batch, n))
    delta = rng.normal(0.0, 1.0, batch)
    expected_trace = decay * trace + edge_contrast(s_plus, s_minus, pre, post) / span
    expected_bias = decay * trace_bias + (s_plus - s_minus) / span
    step_scale, step_bias = fused.trace_step(
        trace, trace_bias, decay, s_plus, s_minus, pre, post, span, delta
    )
    np.testing.assert_allclose(trace, expected_trace)  # updated in place
    np.testing.assert_allclose(trace_bias, expected_bias)
    np.testing.assert_allclose(step_scale, (delta[:, None] * expected_trace).mean(0))
    np.testing.assert_allclose(step_bias, (delta[:, None] * expected_bias).mean(0))


def test_fused_residual_is_zero_only_at_equilibrium() -> None:
    brain = dense_brain(6)
    n = brain.connectome.n
    drive = np.random.default_rng(2).normal(0.0, 1.0, (2, n))
    eq = brain.equilibrate(drive, budget=400, chunk=10, tolerance=1e-9)
    assert eq.converged.all()
    early = brain.settle_batch(drive, steps=1)
    assert (brain.residual(drive, eq.state) < 1e-8).all()
    assert (brain.residual(drive, early) > 1e-6).all()
