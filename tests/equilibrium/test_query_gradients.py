"""Frozen parameters need no derivative; active repair remains fully checked."""

import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

from cadence.experimental.equilibrium import Cortex, _repair
from cadence.experimental.equilibrium._validation import number


def test_query_qualifies_when_only_an_inactive_derivative_overflows():
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    # E(x) = 1.01*x*x/2: its state optimum is exactly zero. The derivative
    # with respect to the frozen zero weight overflows at the starting state.
    with pytest.raises(ValueError, match="finite numeric range"):
        _repair.evaluate(graph, [1e308], [4.0], [0.0], [0.0])
    result = _repair.settle(
        graph,
        [1e308],
        [4.0],
        [0.0],
        [0.0],
        state_bound=4.0,
    )
    assert result["qualified"]
    assert abs(result["state"][0]) <= 1e-6
    assert result["weights"] == (0.0,)
    # Learning makes that derivative eligible, so it must still be rejected.
    with pytest.raises(ValueError, match="finite numeric range"):
        _repair.settle(
            graph,
            [1e308],
            [4.0],
            [0.0],
            [0.0],
            state_bound=4.0,
            learn=True,
        )


def test_public_query_preserves_continuation_with_large_finite_input():
    cortex = Cortex(seed=1, initial_scale=3.5e-309, parameter_prior=10)
    sensor = cortex.input("sensor", shape=1)
    sensing = cortex.column(patches=1, inputs=sensor)
    patch = cortex.column(patches=1, inputs=sensing)
    cortex.output("answer", shape=1, reads=patch)
    brain = cortex.build()
    assert brain.observe({"sensor": [0]}, {"answer": [1]})["accepted"]
    # The sensing patch's relation is weights[0]/biases[0]; drive it to -atanh(1/3).
    value = (-math.atanh(1 / 3) - brain.biases[0]) / brain.weights[0]
    assert math.isfinite(value)
    snapshot = brain.snapshot()
    result = brain.settle({"sensor": [value]})
    assert result["qualified"]
    assert result["state"][0] == pytest.approx(-1 / 3.03, abs=1e-6)
    assert brain.snapshot() == snapshot


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("learn", (False, True))
def test_eligible_derivatives_preserve_full_evaluation_repair(monkeypatch, seed, learn):
    rng = random.Random(seed)
    # Recurrent states, transitive error readback and a clamp share one solve.
    graph = _repair.Graph(
        2,
        4,
        (
            ("input", 0, 0),
            ("input", 1, 1),
            ("state", 0, 1),
            ("state", 1, 0),
            ("residual", 0, 2),
            ("residual", 1, 2),
            ("state", 2, 3),
            ("residual", 2, 3),
            ("state", 3, 0),
        ),
    )
    args = (
        graph,
        [0.4, -0.3],
        [rng.uniform(-0.4, 0.4) for _ in range(4)],
        [rng.uniform(-0.4, 0.4) for _ in graph.edges],
        [0.1, -0.2, 0.3, -0.1],
    )
    options = dict(clamps={3: 0.5}, learn=learn, parameter_prior=1, step=20)
    optimized = _repair.settle(*args, **options)
    assert optimized["qualified"]
    original = _repair._evaluate

    def full_derivatives(*args, **kwargs):
        kwargs["parameter_gradients"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", full_derivatives)
    assert _repair.settle(*args, **options) == optimized


@pytest.mark.parametrize("value", (0, -2, 0.5, Decimal("0.25"), Fraction(1, 8)))
def test_numeric_fast_path_preserves_supported_real_values(value):
    assert number(value, "sample") == float(value)


@pytest.mark.parametrize(
    "value",
    (
        True,
        False,
        "1",
        None,
        1j,
        math.inf,
        -math.inf,
        math.nan,
        Decimal("sNaN"),
        10**1000,
    ),
)
def test_numeric_fast_path_does_not_bypass_validation(value):
    with pytest.raises(ValueError):
        number(value, "sample")


def test_float_subclasses_retain_their_conversion_contract():
    class Converted(float):
        def __float__(self):
            return 0.125

    assert number(Converted(3.0), "sample") == 0.125
    for value in (0.0, -1, Decimal("0")):
        with pytest.raises(ValueError, match="positive"):
            number(value, "sample", positive=True)
