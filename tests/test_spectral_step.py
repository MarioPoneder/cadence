"""Independent solution and safety checks for the adaptive scalar repair step."""

import math
from itertools import pairwise

import pytest

from cadence import Cortex
from cadence._repair import Graph, _next_step, evaluate, settle


def secant(old, new, before, after, *, step=0.25, backtracks=32):
    """Supply explicit coordinate/gradient measurements to the private helper."""
    return _next_step(
        ((old, new, "gradient_state"),),
        {"gradient_state": before},
        {"gradient_state": after},
        step,
        backtracks,
    )


def test_tiny_initial_step_reaches_independently_known_quadratic_minimum():
    # With a fixed prediction p, E(x) = (x-p)^2/2 + alpha*x^2/2.
    # Its unique minimum is p/(1+alpha), regardless of the start or step.
    graph = Graph(0, 1, ())
    bias, alpha = 0.3, 0.25
    expected = math.tanh(bias) / (1 + alpha)
    result = settle(
        graph,
        (),
        (0.9,),
        (),
        (bias,),
        state_prior=alpha,
        step=1e-5,
        budget=8,
        tolerance=1e-10,
    )
    assert result["qualified"]
    assert result["sweeps"] <= 3
    assert result["state"] == pytest.approx((expected,), abs=1e-10)
    assert result["weights"] == ()
    assert result["biases"] == (bias,)
    # Qualification is not a demand that the irreducible compromise vanish.
    assert result["prediction_residual"] > 0.05
    checked = evaluate(graph, (), result["state"], (), (bias,), state_prior=alpha)
    assert max(map(abs, checked["gradient_state"])) <= 1e-10
    assert all(new < old for old, new in pairwise(result["energy_history"]))


@pytest.mark.parametrize("observer", [False, True])
def test_five_patch_witness_qualifies_and_admission_remains_atomic(observer):
    # This fixed generic input/witness exhausted all 512 default sweeps with
    # the constant-step engine. It tests admission, not an XOR learning claim.
    cortex = Cortex(seed=0)
    sensors = cortex.input("cue", shape=2)
    lower = cortex.column("base", patches=4, inputs=sensors)
    if observer:
        output = cortex.observer("answer", patches=1, observes=lower)
    else:
        output = cortex.column("answer", patches=1, inputs=lower)
    cortex.output("choice", shape=1, reads=output)
    brain = cortex.build()
    inputs, targets = {"cue": [-1, -1]}, {"choice": [-0.6]}
    original = brain.snapshot()
    original_weights = brain.weights
    refused = brain.observe(inputs, targets, event_id=0, budget=1)
    assert not refused["accepted"]
    assert brain.snapshot() == original
    assert brain.inspect()["admissions"] == 0

    admitted = brain.observe(inputs, targets, event_id=0)
    assert admitted["accepted"] and admitted["qualified"]
    assert admitted["stationarity"] <= cortex.config["tolerance"]
    assert admitted["sweeps"] < cortex.config["settle_budget"]
    assert brain.inspect()["admissions"] == 1
    assert brain.weights != original_weights
    assert all(new < old for old, new in pairwise(admitted["energy_history"]))
    checkpoint = brain.snapshot()
    duplicate = brain.observe(inputs, targets, event_id=0, budget=0)
    assert duplicate["duplicate"]
    assert brain.snapshot() == checkpoint
    query = brain.settle(inputs)
    assert query["qualified"]
    assert brain.snapshot() == checkpoint


def test_secant_step_matches_known_positive_curvature():
    # Two displacements (1,2), gradient changes (4,8): curvature is four.
    assert secant((0, 0), (1, 2), (0, 0), (4, 8), step=1) == 0.25


@pytest.mark.parametrize(
    ("old", "new", "before", "after"),
    [
        ((0,), (1,), (0,), (-1,)),  # negative curvature
        ((0,), (1,), (0,), (0,)),  # zero curvature
        ((0,), (0,), (0,), (1,)),  # no displacement
        ((0,), (1e200,), (0,), (1e200,)),  # squared displacement overflows
        ((0, 0), (1e154, 1e154), (0, 0), (1e154, 1e154)),  # fsum overflow
        ((0, 0), (1e200, 1e200), (0, 0), (1e200, -1e200)),  # inf cancellation
        ((0,), (1,), (-1e308,), (1e308,)),  # gradient difference overflows
        ((0,), (1e-200,), (0,), (1e100,)),  # squared distance underflows
        ((0,), (1e-200,), (0,), (1e-200,)),  # curvature underflows
        ((0,), (1e150,), (0,), (1e-308,)),  # quotient overflows
    ],
)
def test_unsafe_secant_falls_back_without_raising(old, new, before, after):
    assert secant(old, new, before, after) == 0.25


def test_unmoved_coordinates_do_not_pollute_a_valid_secant():
    # Frozen query parameters can have different gradients without moving.
    # The second gradient difference overflows, but it has no displacement.
    assert secant((0, 2), (1, 2), (1, -1e308), (3, 1e308)) == 0.5


@pytest.mark.parametrize("backtracks", [1, 2, 8, 32, 1024, 2048])
def test_growth_cap_keeps_configured_step_reachable(backtracks):
    step = math.ldexp(1.0, -1000)
    proposal = secant((0,), (1,), (0,), (1e-100,), step=step, backtracks=backtracks)
    assert math.isfinite(proposal) and proposal > 0
    # Each failed Armijo proposal halves the trial. There are backtracks
    # opportunities including the initial trial, not backtracks+1.
    last_allowed = proposal
    for _ in range(backtracks - 1):
        last_allowed *= 0.5
    assert last_allowed <= step
    if backtracks == 1:
        assert proposal == step


def test_overflowing_growth_ceiling_preserves_configured_step():
    assert secant((0,), (1,), (0,), (2,), step=1e308) == 1e308
