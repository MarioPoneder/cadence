"""Independent derivatives and qualification of shared-parameter batch repair."""

import math

import pytest

from cadence import _repair

EDGES = (
    ("input", 0, 0),
    ("input", 1, 1),
    ("state", 2, 0),
    ("state", 0, 1),
    ("residual", 0, 1),
    ("state", 1, 2),
    ("residual", 0, 2),
    ("residual", 1, 2),
)


def independent_energy(inputs, state, weights, biases, alpha, anchors, prior):
    """Direct definition: three patches, nested errors and recurrent state."""
    rows = len(state) // 3
    energy = 0.0
    for row in range(rows):
        values, sensors = state[3 * row : 3 * row + 3], inputs[2 * row : 2 * row + 2]
        errors = []
        for target in range(3):
            drive = biases[target]
            for (kind, source, destination), weight in zip(EDGES, weights, strict=True):
                if destination == target:
                    signals = (
                        sensors
                        if kind == "input"
                        else values
                        if kind == "state"
                        else errors
                    )
                    drive += weight * signals[source]
            errors.append(values[target] - math.tanh(drive))
        energy += (sum(e * e for e in errors) + alpha * sum(x * x for x in values)) / (
            2 * rows
        )
    if anchors is not None:
        energy += (
            prior
            * sum(
                (value - anchor) ** 2
                for value, anchor in zip(
                    weights + biases, anchors[0] + anchors[1], strict=True
                )
            )
            / 2
        )
    return energy


@pytest.mark.parametrize("rows", [1, 2, 3])
@pytest.mark.parametrize("anchored", [False, True])
def test_batch_gradients_match_independent_nested_observer_energy(rows, anchored):
    graph = _repair.Graph(2, 3, EDGES)
    inputs = [0.7, -0.4, -0.2, 0.8, 0.1, 0.3][: 2 * rows]
    groups = [
        [0.12, -0.33, 0.41, -0.08, 0.22, -0.35, 0.3, 0.1, -0.2][: 3 * rows],
        [0.4, -0.2, 0.3, -0.1, 0.5, 0.6, -0.7, 0.8],
        [0.1, -0.2, 0.3],
    ]
    anchors = ([0.02] * len(EDGES), [-0.03] * 3) if anchored else None
    alpha, prior = 0.17, 0.23
    result = _repair._evaluate_batch(
        graph,
        inputs,
        *groups,
        alpha,
        anchors[0] if anchored else None,
        anchors[1] if anchored else None,
        prior,
        batch_size=rows,
    )
    assert result["energy"] == pytest.approx(
        independent_energy(inputs, *groups, alpha, anchors, prior), abs=1e-14
    )
    for index, key in enumerate(
        ("gradient_state", "gradient_weights", "gradient_biases")
    ):
        for coordinate, analytic in enumerate(result[key]):
            plus, minus = [list(g) for g in groups], [list(g) for g in groups]
            h = 2e-6
            plus[index][coordinate] += h
            minus[index][coordinate] -= h
            numeric = (
                independent_energy(inputs, *plus, alpha, anchors, prior)
                - independent_energy(inputs, *minus, alpha, anchors, prior)
            ) / (2 * h)
            assert analytic == pytest.approx(numeric, rel=2e-7, abs=2e-9)


def test_parameter_anchor_added_once_and_work_counts_every_row():
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    visits = {"edge_visits": 0, "patch_visits": 0}
    values = (graph, [0.5, -0.5, 0.2], [0.2, -0.4, 0.3], [0.4], [-0.1], 0.07)
    plain = _repair._evaluate_batch(*values, None, None, 0.3)
    anchored = _repair._evaluate_batch(*values, [0.2], [0.1], 0.3, visits)
    assert anchored["energy"] - plain["energy"] == pytest.approx(
        0.3 * (0.2**2 + (-0.2) ** 2) / 2
    )
    assert anchored["gradient_weights"][0] - plain["gradient_weights"][
        0
    ] == pytest.approx(0.06)
    assert anchored["gradient_biases"][0] - plain["gradient_biases"][
        0
    ] == pytest.approx(-0.06)
    assert visits == {"edge_visits": 6, "patch_visits": 6}


@pytest.mark.parametrize("copies", [2, 3, 8])
def test_duplicate_rows_preserve_objective_solution_and_activity_scale(copies):
    graph = _repair.Graph(1, 2, (("input", 0, 0), ("state", 0, 1), ("residual", 0, 1)))
    args = (graph, [0.4], [0.1, -0.2], [0.2, 0.3, -0.1], [0.02, -0.03])
    options = dict(learn=True, clamps={1: 0.5}, tolerance=1e-8)
    single = _repair.settle(*args, **options)
    repeated = _repair.settle(
        graph,
        args[1] * copies,
        args[2] * copies,
        args[3],
        args[4],
        **{**options, "clamps": {2 * row + 1: 0.5 for row in range(copies)}},
        _batch_size=copies,
    )
    assert single["qualified"] and repeated["qualified"]
    for key in ("weights", "biases", "energy"):
        assert repeated[key] == pytest.approx(single[key], abs=3e-8)
    assert repeated["state"] == pytest.approx(single["state"] * copies, abs=3e-8)
    assert repeated["sweeps"] <= single["sweeps"] + 2


def test_one_bad_private_state_cannot_hide_in_a_large_batch():
    graph = _repair.Graph(0, 1, ())
    # An incorrect mean-gradient check would see 4e-5 / 64 < tolerance.
    states = [4e-5] + [0.0] * 63
    result = _repair.settle(
        graph, [], states, [], [0.0], _batch_size=64, budget=0, tolerance=1e-6
    )
    assert not result["qualified"] and result["reason"] == "budget"
    assert result["stationarity"] == pytest.approx(1.01 * states[0])
    fixed = _repair.settle(
        graph, [], states, [], [0.0], _batch_size=64, budget=4, tolerance=1e-10
    )
    assert fixed["qualified"]
    assert max(map(abs, fixed["state"])) <= 1e-10


@pytest.mark.parametrize("units,threshold", [(2, 1), (9, 8), (25, 24)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_subnormal_private_gradients_cannot_qualify_by_rounding_the_mean(
    units, threshold, sign
):
    smallest = math.ulp(0.0)
    state, tolerance = sign * units * smallest, threshold * smallest
    graph = _repair.Graph(0, 1, ())
    single = _repair.settle(
        graph, (), (state,), (), (0.0,), budget=0, tolerance=tolerance
    )
    batch = _repair.settle(
        graph,
        (),
        (state,) * 8,
        (),
        (0.0,),
        budget=0,
        tolerance=tolerance,
        _batch_size=8,
    )
    assert not single["qualified"] and not batch["qualified"]
    assert batch["stationarity"] == single["stationarity"] == abs(state)
    evaluated = _repair._evaluate_batch(
        graph, (), (state,) * 8, (), (0.0,), 0.01, None, None, 0.1
    )
    assert evaluated["gradient_state_unscaled"] == (state,) * 8
    assert abs(evaluated["gradient_state"][0] * 8) <= tolerance


def test_subnormal_state_move_uses_original_gradient_then_fails_closed_on_zero_slope(
    monkeypatch,
):
    ordinary = _repair._evaluate_batch
    states = []

    def measured(graph, inputs, state, *args, **kwargs):
        states.append(state)
        return ordinary(graph, inputs, state, *args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate_batch", measured)
    initial = (9 * math.ulp(0.0),) * 8
    result = _repair.settle(
        _repair.Graph(0, 1, ()),
        (),
        initial,
        (),
        (0.0,),
        budget=1,
        backtracks=1,
        tolerance=math.ulp(0.0),
        _batch_size=8,
    )
    assert states == [initial, (0.0,) * 8, initial]
    assert not result["qualified"] and result["reason"] == "line_search"
    assert result["state"] == initial


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("units", [2, 9, 25])
def test_shared_parameter_subnormal_mean_cannot_falsely_qualify(sign, units):
    graph = _repair.Graph(0, 1, ())
    target = sign * units * math.ulp(0.0)
    residuals = []
    for rows in (1, 8):
        result = _repair.settle(
            graph,
            (),
            (0.0,) * rows,
            (),
            (0.0,),
            learn=True,
            clamps=dict.fromkeys(range(rows), target),
            budget=0,
            tolerance=math.ulp(0.0),
            _batch_size=rows,
        )
        assert not result["qualified"]
        residuals.append(result["stationarity"])
    assert residuals == [abs(target), abs(target)]


@pytest.mark.parametrize("sign", [-1, 1])
def test_mean_preserves_huge_finite_values_and_tiny_cancellation_remainders(sign):
    huge, tiny = sign * 1.2e308, sign * 1e-321
    assert _repair._mean((huge, huge)) == huge
    assert _repair._mean((tiny,) * 8) == tiny
    # The first two terms overflow a binary64 sum. Exact cancellation must
    # preserve the tiny remainder rather than losing it by prescaling terms.
    assert _repair._mean((huge, huge, -huge, -huge, tiny)) == tiny / 5
    assert _repair._mean((huge, -huge, huge, -huge, tiny)) == tiny / 5
    assert _repair._mean((huge, -huge)) == 0.0


def test_state_preconditioner_and_secant_metric_ignore_duplicate_count():
    graph = _repair.Graph(0, 1, ())
    options = dict(state_prior=0.25, step=1e-5, budget=8, tolerance=1e-10)
    single = _repair.settle(graph, [], [0.9], [], [0.3], **options)
    repeated = _repair.settle(
        graph, [], [0.9] * 32, [], [0.3], _batch_size=32, **options
    )
    assert single["qualified"] and repeated["qualified"]
    assert repeated["sweeps"] == single["sweeps"] <= 3
    assert repeated["state"] == single["state"] * 32
    assert repeated["energy_history"] == single["energy_history"]


def test_shared_parameter_gradients_are_mean_then_prior_not_rescaled_for_stationarity():
    graph = _repair.Graph(0, 1, ())
    result = _repair.settle(
        graph,
        [],
        [0.0] * 4,
        [],
        [0.0],
        _batch_size=4,
        learn=True,
        clamps={0: 0.8, 1: -0.2, 2: -0.2, 3: -0.2},
        budget=0,
    )
    assert result["stationarity"] == pytest.approx(0.05)
    assert not result["qualified"]


def test_batch_solves_one_shared_relation_not_average_independently_fitted_relations():
    graph = _repair.Graph(0, 1, ())
    targets, prior = (0.9, -0.3), 0.1
    result = _repair.settle(
        graph,
        [],
        [0.0, 0.0],
        [],
        [0.0],
        learn=True,
        clamps=dict(enumerate(targets)),
        _batch_size=2,
        parameter_prior=prior,
        tolerance=1e-10,
    )
    assert result["qualified"]
    # Independent one-dimensional optimality equation for their mean target.
    lower, upper = 0.0, math.atanh(sum(targets) / 2)
    for _ in range(80):
        middle = (lower + upper) / 2
        p = math.tanh(middle)
        gradient = (p - sum(targets) / 2) * (1 - p * p) + prior * middle
        if gradient < 0:
            lower = middle
        else:
            upper = middle
    assert result["biases"] == pytest.approx([(lower + upper) / 2], abs=1e-8)
    independent = [
        _repair.settle(
            graph,
            [],
            [0.0],
            [],
            [0.0],
            learn=True,
            clamps={0: target},
            parameter_prior=prior,
            tolerance=1e-10,
        )
        for target in targets
    ]
    assert all(row["qualified"] for row in independent)
    assert (
        abs(result["biases"][0] - sum(row["biases"][0] for row in independent) / 2)
        > 0.01
    )


@pytest.mark.parametrize("rows", [1, 3])
def test_engine_dispatch_preserves_flat_batch_clamps_and_shared_anchors(rows):
    class Capture:
        def settle(self, inputs, state, weights, biases, **options):
            assert inputs == (0.2,) * rows
            assert state == (0.6,) * rows
            assert weights == options["anchor_weights"] == (0.4,)
            assert biases == options["anchor_biases"] == (0.1,)
            assert options.get("_batch_size", 1) == rows
            assert ("_batch_size" in options) == (rows > 1)
            assert options["clamps"] == dict.fromkeys(range(rows), 0.6)
            return {"dispatched": True}

    result = _repair.settle(
        _repair.Graph(1, 1, (("input", 0, 0),)),
        [0.2] * rows,
        [0.0] * rows,
        [0.4],
        [0.1],
        learn=True,
        clamps=dict.fromkeys(range(rows), 0.6),
        _batch_size=rows,
        _engine=Capture(),
    )
    assert result == {"dispatched": True}


def test_single_row_settlement_keeps_original_evaluator(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("single-row settlement must not use a batch evaluator")

    monkeypatch.setattr(_repair, "_evaluate_batch", unexpected)
    args = (_repair.Graph(1, 1, (("input", 0, 0),)), [0.4], [0.0], [0.2], [0.1])
    for learn in (False, True):
        options = dict(learn=learn, clamps={0: 0.5} if learn else None)
        assert _repair.settle(*args, **options) == _repair.settle(
            *args, **options, _batch_size=1
        )


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_invalid_batch_size_fails_before_engine_dispatch(size):
    with pytest.raises(ValueError, match="batch_size"):
        _repair.settle(_repair.Graph(0, 1, ()), [], [0.0], [], [0.0], _batch_size=size)


def test_batch_shape_clamps_and_anchors_are_validated():
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    args = (graph, [0.1, 0.2], [0.0, 0.0], [0.3], [0.0])
    for change in (
        {"clamps": {2: 0.4}},
        {"clamps": {1: 2.0}},
        {"anchor_weights": [0.3]},
    ):
        with pytest.raises(ValueError):
            _repair.settle(*args, _batch_size=2, learn=True, **change)
    with pytest.raises(ValueError, match="inputs"):
        _repair.settle(graph, [0.1], args[2], args[3], args[4], _batch_size=2)
    with pytest.raises(ValueError, match="state"):
        _repair.settle(graph, args[1], [0.0], args[3], args[4], _batch_size=2)


def test_query_batch_omits_overflowing_frozen_parameter_derivatives():
    graph = _repair.Graph(1, 1, (("input", 0, 0),))
    result = _repair.settle(
        graph,
        [1e308, 1e308],
        [0.0, 0.0],
        [0.0],
        [0.0],
        clamps={0: 2.0, 1: 2.0},
        state_bound=3.0,
        _batch_size=2,
        budget=0,
    )
    assert result["qualified"]
    with pytest.raises(ValueError, match="numeric range"):
        _repair.settle(
            graph,
            [1e308, 1e308],
            [0.0, 0.0],
            [0.0],
            [0.0],
            clamps={0: 2.0, 1: 2.0},
            state_bound=3.0,
            _batch_size=2,
            budget=0,
            learn=True,
        )
