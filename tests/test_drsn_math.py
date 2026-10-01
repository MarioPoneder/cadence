"""Independent scalar-energy checks for the population repair kernel."""

import math
import random

import pytest

from cadence._repair import Graph, evaluate, settle

EDGES = (
    ("input", 0, 2),
    ("input", 1, 0),
    ("state", 2, 0),
    ("state", 0, 2),
    ("state", 1, 3),
    ("residual", 2, 1),
    ("residual", 0, 1),
    ("residual", 1, 3),
    ("residual", 2, 3),
)


def independent_energy(inputs, state, weights, biases, alpha, anchors=None):
    """Scalar oracle with an explicit order, independent of kernel traversal."""
    errors = {}
    predictions = {}
    for target in (0, 2, 1, 3):
        drive = biases[target]
        for (kind, source, destination), weight in zip(EDGES, weights, strict=True):
            if destination != target:
                continue
            if kind == "input":
                signal = inputs[source]
            elif kind == "state":
                signal = state[source]
            else:
                signal = errors[source]
            drive += weight * signal
        predictions[target] = math.tanh(drive)
        errors[target] = state[target] - predictions[target]
    energy = 0.5 * sum(e * e for e in errors.values())
    energy += 0.5 * alpha * sum(x * x for x in state)
    if anchors is not None:
        old_weights, old_biases, strength = anchors
        energy += (
            0.5
            * strength
            * sum(
                (x - y) ** 2
                for x, y in zip(weights + biases, old_weights + old_biases, strict=True)
            )
        )
    return energy, [predictions[i] for i in range(4)], [errors[i] for i in range(4)]


@pytest.mark.parametrize("anchored", [False, True])
def test_all_gradients_include_transitive_residual_feedback(anchored):
    graph = Graph(2, 4, EDGES)
    inputs = [0.7, -0.4]
    state = [0.12, -0.33, 0.41, -0.08]
    weights = [0.4, -0.2, 0.3, -0.1, 0.5, 0.6, -0.7, 0.8, -0.4]
    biases = [0.1, -0.2, 0.3, -0.4]
    old_weights = [0.02] * len(weights)
    old_biases = [-0.03] * len(biases)
    alpha, strength = 0.17, 0.23
    options = dict(state_prior=alpha)
    anchors = None
    if anchored:
        options.update(
            anchor_weights=old_weights,
            anchor_biases=old_biases,
            parameter_prior=strength,
        )
        anchors = old_weights, old_biases, strength
    result = evaluate(graph, inputs, state, weights, biases, **options)
    energy, predictions, errors = independent_energy(
        inputs, state, weights, biases, alpha, anchors
    )
    assert result["energy"] == pytest.approx(energy, abs=1e-14)
    assert result["predictions"] == pytest.approx(predictions, abs=1e-14)
    assert result["errors"] == pytest.approx(errors, abs=1e-14)
    groups = [state, weights, biases]
    for group_index, field in enumerate(
        ("gradient_state", "gradient_weights", "gradient_biases")
    ):
        numerical = []
        for coordinate in range(len(groups[group_index])):
            plus, minus = [list(g) for g in groups], [list(g) for g in groups]
            h = 2e-6
            plus[group_index][coordinate] += h
            minus[group_index][coordinate] -= h
            upper = independent_energy(inputs, *plus, alpha, anchors)[0]
            lower = independent_energy(inputs, *minus, alpha, anchors)[0]
            numerical.append((upper - lower) / (2 * h))
        assert result[field] == pytest.approx(numerical, rel=2e-7, abs=2e-9)


@pytest.mark.parametrize("seed", range(12))
def test_arbitrary_recurrent_graph_derivatives_match_independent_energy(seed):
    """Vary traversal order, self connections and multiple error-readback paths."""
    rng = random.Random(seed)
    count = 5
    order = rng.sample(range(count), count)
    edges = [("input", target % 2, target) for target in range(count)]
    # A full recurrent state cycle and a self edge are valid. Only dependencies
    # between derived errors, generated in the independent order, form a DAG.
    edges += [("state", source, (source + 1) % count) for source in range(count)]
    edges += [("state", order[0], order[0])]
    edges += [
        ("residual", source, target)
        for rank, source in enumerate(order)
        for target in order[rank + 1 :]
        if rng.random() < 0.65
    ]
    rng.shuffle(edges)
    graph = Graph(2, count, tuple(edges))
    inputs = [0.6, -0.7]
    groups = [
        [rng.uniform(-0.8, 0.8) for _ in range(length)]
        for length in (count, len(edges), count)
    ]
    anchors = [[rng.uniform(-0.8, 0.8) for _ in group] for group in groups[1:]]
    alpha, strength = 0.13, 0.27

    def energy(state, weights, biases):
        errors = {}
        for target in order:
            drive = biases[target]
            for (kind, source, destination), weight in zip(edges, weights, strict=True):
                if destination == target:
                    values = (
                        inputs
                        if kind == "input"
                        else state
                        if kind == "state"
                        else errors
                    )
                    drive += weight * values[source]
            errors[target] = state[target] - math.tanh(drive)
        result = sum(error**2 for error in errors.values()) / 2
        result += alpha * sum(value**2 for value in state) / 2
        for values, reference in zip((weights, biases), anchors, strict=True):
            result += (
                strength
                * sum(
                    (value - anchor) ** 2
                    for value, anchor in zip(values, reference, strict=True)
                )
                / 2
            )
        return result

    result = evaluate(
        graph,
        inputs,
        *groups,
        state_prior=alpha,
        parameter_prior=strength,
        anchor_weights=anchors[0],
        anchor_biases=anchors[1],
    )
    assert result["energy"] == pytest.approx(energy(*groups), abs=2e-15)
    for group_index, key in enumerate(
        ("gradient_state", "gradient_weights", "gradient_biases")
    ):
        for coordinate, analytic in enumerate(result[key]):
            plus, minus = [list(g) for g in groups], [list(g) for g in groups]
            h = 1e-5
            plus[group_index][coordinate] += h
            minus[group_index][coordinate] -= h
            numerical = (energy(*plus) - energy(*minus)) / (2 * h)
            assert analytic == pytest.approx(numerical, rel=2e-7, abs=2e-9)


@pytest.mark.parametrize("prior", [0.03, 0.1, 0.7])
def test_learning_matches_independently_solved_scalar_optimum(prior):
    """Check retained relation values, not the trivially clamped training output."""
    target = 0.6
    graph = Graph(1, 1, (("input", 0, 0),))
    learned = settle(
        graph,
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        learn=True,
        clamps={0: target},
        parameter_prior=prior,
        tolerance=1e-10,
        budget=2048,
    )
    assert learned["qualified"]

    # Symmetry gives w=b=t. The scalar derivative changes sign once between
    # zero and atanh(target)/2; bisection is independent of projected descent.
    low, high = 0.0, math.atanh(target) / 2
    for _ in range(80):
        middle = (low + high) / 2
        prediction = math.tanh(2 * middle)
        gradient = (prediction - target) * (1 - prediction**2) + prior * middle
        if gradient < 0:
            low = middle
        else:
            high = middle
    expected_parameter = (low + high) / 2
    assert learned["weights"] == pytest.approx([expected_parameter], abs=1e-8)
    assert learned["biases"] == pytest.approx([expected_parameter], abs=1e-8)

    # No teacher clamp and a reset live state: only retained parameters supply
    # the prediction. The activity prior gives x = tanh(w+b)/(1+state_prior).
    recalled = settle(
        graph,
        [1.0],
        [0.0],
        learned["weights"],
        learned["biases"],
        tolerance=1e-10,
    )
    assert recalled["qualified"]
    expected_state = math.tanh(2 * expected_parameter) / 1.01
    assert recalled["state"] == pytest.approx([expected_state], abs=1e-8)


def test_explicit_parameter_anchor_stays_fixed_across_all_repairs():
    graph = Graph(0, 1, ())
    anchor, start, target, prior = -0.3, 0.4, 0.6, 0.2
    learned = settle(
        graph,
        [],
        [0.0],
        [],
        [start],
        learn=True,
        clamps={0: target},
        anchor_weights=[],
        anchor_biases=[anchor],
        parameter_prior=prior,
        tolerance=1e-10,
        budget=1024,
    )
    assert learned["qualified"] and learned["sweeps"] > 1
    bias = learned["biases"][0]
    prediction = math.tanh(bias)
    final_derivative = (prediction - target) * (1 - prediction**2)
    assert abs(final_derivative + prior * (bias - anchor)) <= 1e-10
    # Replacing the explicit anchor with the starting/current parameters would
    # optimize a different objective; neither may qualify this solve.
    assert abs(final_derivative + prior * (bias - start)) > 0.1
    assert abs(final_derivative) > 0.05
    expected_energy = (
        (target - prediction) ** 2 / 2
        + 0.01 * target**2 / 2
        + prior * (bias - anchor) ** 2 / 2
    )
    assert learned["energy"] == pytest.approx(expected_energy, abs=1e-15)


def test_stationarity_is_not_zero_prediction_residual():
    graph = Graph(0, 1, ())
    bias = math.atanh(0.6)
    result = settle(graph, [], [0.0], [], [bias], state_prior=0.2, tolerance=1e-10)
    assert result["qualified"]
    assert result["state"] == pytest.approx([0.5], abs=1e-9)
    assert result["prediction_residual"] == pytest.approx(0.1, abs=1e-9)
    assert result["stationarity"] <= 1e-10
    assert result["biases"] == (bias,)


def test_hard_clamped_coordinates_are_not_free_stationarity_conditions():
    graph = Graph(0, 1, ())
    result = settle(graph, [], [0.0], [], [0.0], clamps={0: 0.8}, budget=0)
    assert result["qualified"]
    assert result["state"] == (0.8,)
    assert result["prediction_residual"] == pytest.approx(0.8)
    assert result["stationarity"] == 0


def test_active_state_bound_uses_projected_stationarity():
    graph = Graph(0, 1, ())
    result = settle(graph, [], [0.0], [], [2.0], state_bound=0.2, tolerance=1e-10)
    assert result["qualified"]
    assert result["state"] == (0.2,)
    raw = evaluate(graph, [], result["state"], [], [2.0])["gradient_state"][0]
    assert raw < -0.7
    assert result["stationarity"] <= 1e-10


def test_active_parameter_bounds_use_the_declared_projected_gradient():
    graph = Graph(1, 1, (("input", 0, 0),))
    result = settle(
        graph,
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        clamps={0: 0.8},
        learn=True,
        parameter_bound=0.1,
        parameter_prior=0.01,
        tolerance=1e-10,
    )
    assert result["qualified"]
    assert result["weights"] == result["biases"] == (0.1,)
    assert result["prediction_residual"] > 0.6
    assert result["stationarity"] == 0


def test_joint_repair_learns_without_mutating_arguments_or_query_parameters():
    graph = Graph(1, 1, (("input", 0, 0),))
    inputs, state, weights, biases = [0.7], [0.0], [0.0], [0.0]
    saved = [list(v) for v in (inputs, state, weights, biases)]
    learned = settle(
        graph,
        inputs,
        state,
        weights,
        biases,
        learn=True,
        clamps={0: 0.5},
        tolerance=1e-8,
        budget=1024,
    )
    assert learned["qualified"]
    assert learned["state"] == (0.5,)
    assert learned["weights"][0] > 0 and learned["biases"][0] > 0
    assert [inputs, state, weights, biases] == saved
    history = learned["energy_history"]
    assert len(history) > 1
    assert all(math.isfinite(e) and e >= 0 for e in history)
    assert all(b <= a + 1e-14 for a, b in zip(history, history[1:], strict=False))
    recalled = settle(
        graph,
        inputs,
        [0.0],
        learned["weights"],
        learned["biases"],
        tolerance=1e-8,
    )
    assert recalled["qualified"] and recalled["state"][0] > 0.3
    assert recalled["weights"] == learned["weights"]
    assert recalled["biases"] == learned["biases"]


def test_nonstationary_zero_budget_is_a_retained_refusal():
    graph = Graph(0, 1, ())
    result = settle(graph, [], [0.0], [], [0.4], budget=0)
    assert not result["qualified"]
    assert result["sweeps"] == 0
    assert result["stationarity"] > 0.3


def test_final_learning_stationarity_includes_original_parameter_anchor():
    graph = Graph(2, 4, EDGES)
    inputs = [0.4, -0.2]
    state = [0.0] * 4
    weights = [0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1]
    biases = [0.01, 0.0, -0.02, 0.03]
    result = settle(
        graph,
        inputs,
        state,
        weights,
        biases,
        learn=True,
        clamps={3: 0.4},
        state_prior=0.17,
        parameter_prior=0.23,
        tolerance=1e-7,
        budget=2048,
    )
    assert result["qualified"]
    final = [list(result[key]) for key in ("state", "weights", "biases")]
    # All final coordinates are interior in this fixture. Only the witnessed
    # state coordinate is excluded; parameter priors stay at event-start values.
    gradients = []
    h = 2e-6
    for group_index, group in enumerate(final):
        for coordinate in range(len(group)):
            if group_index == 0 and coordinate == 3:
                continue
            plus, minus = [list(g) for g in final], [list(g) for g in final]
            plus[group_index][coordinate] += h
            minus[group_index][coordinate] -= h
            anchors = weights, biases, 0.23
            upper = independent_energy(inputs, *plus, 0.17, anchors)[0]
            lower = independent_energy(inputs, *minus, 0.17, anchors)[0]
            gradients.append(abs((upper - lower) / (2 * h)))
    assert max(gradients) <= 1.01e-7
    assert result["stationarity"] == pytest.approx(max(gradients), abs=1e-9)
    assert result["energy"] == pytest.approx(
        independent_energy(inputs, *final, 0.17, (weights, biases, 0.23))[0],
        abs=1e-14,
    )


def test_work_charges_rejected_proposals_and_final_equation_recomputation(monkeypatch):
    from cadence import _repair

    calls = []
    original = _repair._evaluate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", counted)
    graph = Graph(1, 1, (("input", 0, 0),))
    result = settle(graph, [0.5], [0.0], [0.4], [0.1], step=50, budget=256)
    assert result["qualified"]
    work = result["work"]
    assert work["backtracks"] > 0
    assert work["evaluations"] == len(calls) == work["proposals"] + 2
    assert work["proposals"] == result["sweeps"] + work["backtracks"]
    assert len(result["energy_history"]) == result["sweeps"] + 1
    # Every proposal reuses input-only predictions computed on the initial
    # traversal. Reverse derivatives and final qualification still visit all.
    assert work["edge_visits"] == len(graph.edges) * (
        2 * len(calls) - work["proposals"]
    )
    assert work["patch_visits"] == 2 * graph.n_patches * len(calls)


def test_exhausted_line_search_does_not_admit_a_stale_state():
    graph = Graph(0, 1, ())
    result = settle(graph, [], [0.0], [], [0.5], step=1000, backtracks=1)
    assert not result["qualified"]
    assert result["reason"] == "line_search"
    assert result["state"] == (0.0,)
    assert result["sweeps"] == 0
    assert result["work"]["backtracks"] == 1


def test_nonrepresentable_learning_proposals_are_counted_and_refused():
    graph = Graph(1, 1, (("input", 0, 0),))
    result = settle(
        graph,
        [1e308],
        [0.0],
        [0.0],
        [0.0],
        learn=True,
        clamps={0: 0.5},
        backtracks=3,
    )
    assert not result["qualified"] and result["reason"] == "line_search"
    assert result["weights"] == result["biases"] == (0.0,)
    assert result["work"]["backtracks"] == 3
    assert result["work"]["evaluations"] == 5
    with pytest.raises(ValueError):
        evaluate(graph, [1e308], [0.0], [4.0], [0.0])


@pytest.mark.parametrize(
    "state,weights,biases,options",
    [
        ([1.1], [0.0], [0.0], {}),
        ([0.0], [4.1], [0.0], {}),
        ([0.0], [0.0], [4.1], {}),
        ([0.0], [0.0], [0.0], {"learn": 1}),
        ([0.0], [0.0], [0.0], {"clamps": [0.1]}),
        ([0.0], [0.0], [0.0], {"clamps": {1: 0.1}}),
        (
            [0.0],
            [0.0],
            [0.0],
            {"learn": True, "anchor_weights": [4.1], "anchor_biases": [0.0]},
        ),
        ([0.0], [0.0], [0.0], {"learn": True, "anchor_weights": [0.0]}),
    ],
)
def test_invalid_starts_and_explicit_anchors_raise_before_repair(
    state, weights, biases, options
):
    graph = Graph(1, 1, (("input", 0, 0),))
    with pytest.raises(ValueError):
        settle(graph, [0.1], state, weights, biases, **options)


@pytest.mark.parametrize(
    "counts,edges",
    [
        ((True, 1), ()),
        ((0, 0), ()),
        ((0, 1), (("unknown", 0, 0),)),
        ((0, 1), (("input", 0, 0),)),
        ((1, 1), (("input", True, 0),)),
        ((1, 1), (("state", 1, 0),)),
        ((1, 1), (("state", 0, -1),)),
        ((0, 1), (("residual", 0, 0),)),
        ((0, 2), (("residual", 0, 1), ("residual", 1, 0))),
    ],
)
def test_invalid_graphs_and_residual_cycles_fail(counts, edges):
    with pytest.raises(ValueError):
        Graph(*counts, edges)


@pytest.mark.parametrize(
    "options",
    [
        {"budget": True},
        {"budget": -1},
        {"tolerance": math.nan},
        {"tolerance": 0},
        {"state_prior": -1},
        {"parameter_prior": -1},
        {"state_bound": math.inf},
        {"parameter_bound": 0},
        {"step": 0},
        {"backtracks": True},
        {"clamps": {True: 0.1}},
        {"clamps": {0: math.nan}},
        {"clamps": {0: 1.01}},
        {"anchor_weights": [], "anchor_biases": [0.0]},
    ],
)
def test_invalid_solver_configuration_is_not_a_numerical_refusal(options):
    with pytest.raises(ValueError):
        settle(Graph(0, 1, ()), [], [0.0], [], [0.0], **options)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, "0.2"])
def test_non_numeric_or_nonfinite_model_values_rejected(value):
    graph = Graph(1, 1, (("input", 0, 0),))
    for index in range(4):
        arguments = [[0.1], [0.2], [0.3], [0.4]]
        arguments[index] = [value]
        with pytest.raises(ValueError):
            evaluate(graph, *arguments)
