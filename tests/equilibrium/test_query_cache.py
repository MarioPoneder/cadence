"""Invariant-query reuse must preserve the complete reference trajectory."""

import math
import random

import pytest

from cadence.experimental.equilibrium import Cortex, _repair
from cadence.experimental.equilibrium._repair import Graph


def _without_work(result):
    return {key: value for key, value in result.items() if key != "work"}


def _compare(monkeypatch, graph, inputs, state, weights, biases, **options):
    cached = _repair.settle(graph, inputs, state, weights, biases, **options)
    original = _repair._evaluate

    def uncached(*args, **kwargs):
        kwargs.pop("_query_cache", None)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_repair, "_evaluate", uncached)
        reference = _repair.settle(graph, inputs, state, weights, biases, **options)
    # Exact equality includes every accepted energy, state, prediction, error,
    # stationary residual, refusal reason and parameter. Only actual work differs.
    assert _without_work(cached) == _without_work(reference)
    for key in ("evaluations", "proposals", "backtracks", "patch_visits"):
        assert cached["work"][key] == reference["work"][key]
    skipped = sum(
        len(incoming)
        for incoming in graph.incoming
        if all(graph.edges[edge][0] == "input" for edge in incoming)
    )
    if options.get("learn") or options.get("_batch_size", 1) != 1:
        skipped = 0
    assert cached["work"]["edge_visits"] == (
        reference["work"]["edge_visits"] - skipped * cached["work"]["proposals"]
    )
    return cached


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kind", ["flat", "recursive", "recurrent"])
def test_query_cache_preserves_complete_random_graph_trajectory(monkeypatch, seed, kind):
    rng = random.Random(seed)
    edges = [("input", source, target) for target in range(6) for source in range(3)]
    if kind != "flat":
        # Patch 0 stays input-only but its changing error is observed through
        # several paths. There is also a bias-only invariant prediction at 5.
        edges = [edge for edge in edges if edge[2] != 5]
        edges += [("state", source, source + 1) for source in range(4)]
        edges += [("residual", source, source + 1) for source in range(4)]
        edges += [("residual", 0, 3), ("residual", 1, 4)]
    if kind == "recurrent":
        edges += [("state", 4, 1), ("state", 3, 3)]
    rng.shuffle(edges)
    graph = Graph(3, 6, tuple(edges))
    result = _compare(
        monkeypatch,
        graph,
        [rng.uniform(-1, 1) for _ in range(3)],
        [rng.uniform(-0.7, 0.7) for _ in range(6)],
        [rng.uniform(-0.6, 0.6) for _ in edges],
        [rng.uniform(-0.3, 0.3) for _ in range(6)],
        clamps={4: 0.3} if seed % 2 else {},
        state_prior=0.2,
        state_bound=0.8,
        tolerance=1e-8,
        step=8,
        budget=512,
    )
    assert result["qualified"]
    assert result["work"]["proposals"] > 0


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"budget": 0}, "budget"),
        ({"budget": 1, "step": 0.01}, "budget"),
        ({"step": 1000, "backtracks": 1}, "line_search"),
    ],
)
def test_refusals_and_capped_trajectories_are_unchanged(monkeypatch, options, reason):
    graph = Graph(1, 2, (("input", 0, 0), ("residual", 0, 1)))
    result = _compare(monkeypatch, graph, [0.9], [0, 0], [0.6, 0.4], [0.1, 0.2], **options)
    assert not result["qualified"]
    assert result["reason"] == reason


def test_cancellation_uses_original_fsum_prediction(monkeypatch):
    graph = Graph(3, 1, tuple(("input", source, 0) for source in range(3)))
    result = _compare(monkeypatch, graph, [1e308, 1, -1e308], [0], [1, 1, 1], [0])
    assert result["qualified"]
    assert result["predictions"] == (math.tanh(1),)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_are_rejected_before_any_cache(monkeypatch, value):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input reached an evaluator")

    monkeypatch.setattr(_repair, "_evaluate", unexpected)
    with pytest.raises(ValueError):
        _repair.settle(Graph(1, 1, (("input", 0, 0),)), [value], [0], [1], [0])


def test_nonfinite_initial_prediction_is_not_hidden():
    graph = Graph(2, 1, (("input", 0, 0), ("input", 1, 0)))
    with pytest.raises(ValueError, match="finite numeric range"):
        _repair.settle(graph, [1e308, 1e308], [0], [1, 1], [0])


def test_full_final_recompute_rejects_corrupted_reuse(monkeypatch):
    original = _repair._evaluate
    caches = []

    def corrupt_reuse(*args, **kwargs):
        cache = kwargs.get("_query_cache")
        caches.append(cache)
        if cache is not None:
            predictions, signals = cache
            kwargs["_query_cache"] = (
                tuple(None if p is None else p + 0.25 for p in predictions),
                signals,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", corrupt_reuse)
    cortex = Cortex(seed=2)
    source = cortex.input("in", shape=(1,))
    features = cortex.column(patches=2, inputs=source)
    population = cortex.column(patches=1, inputs=features)
    cortex.output("out", shape=(1,), reads=population)
    brain = cortex.build()
    before = brain.snapshot()
    result = brain.step({"in": [0.9]})
    assert caches[0] is caches[-1] is None
    assert any(cache is not None for cache in caches)
    assert not result["qualified"] and not result["accepted"]
    assert brain.snapshot() == before


@pytest.mark.parametrize("mode", ["learning", "batch", "tensor"])
def test_cache_scope_excludes_learning_batch_and_tensor_dispatch(monkeypatch, mode):
    original = _repair._evaluate

    def inspect(*args, **kwargs):
        assert kwargs.get("_query_cache") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", inspect)
    graph = Graph(1, 1, (("input", 0, 0),))
    if mode == "tensor":

        class Engine:
            def settle(self, *args, **kwargs):
                assert "_query_cache" not in kwargs
                return {"delegated": True}

        result = _repair.settle(graph, [0.5], [0], [0.4], [0.1], _engine=Engine())
        assert result == {"delegated": True}
    elif mode == "batch":
        result = _compare(monkeypatch, graph, [0.5, -0.5], [0, 0], [0.4], [0.1], _batch_size=2)
        assert result["qualified"]
    else:
        result = _compare(monkeypatch, graph, [0.5], [0], [0.4], [0.1], learn=True)
        assert result["qualified"]


def test_public_queries_rebuild_cache_after_inputs_parameters_and_state_change(
    monkeypatch,
):
    cortex = Cortex(seed=7)
    source = cortex.input("in", shape=(1,))
    base = cortex.column(patches=2, inputs=source)
    observer = cortex.observer(patches=2, observes=base)
    cortex.output("out", shape=(1,), reads=observer)
    brain = cortex.build()
    initial = brain.snapshot()
    original = _repair._evaluate

    def sequence():
        answers = []
        for value in (0.4, -0.7):
            before = brain.snapshot()
            answers.append(_without_work(brain.settle({"in": [value]})))
            assert brain.snapshot() == before
            answers.append(_without_work(brain.step({"in": [value]})))
            answers.append(_without_work(brain.observe({"in": [value]}, {"out": [0.9 * value]})))
            answers.append(_without_work(brain.settle({"in": [value]})))
        return answers, brain.snapshot()

    cached = sequence()
    brain.restore(initial)

    def uncached(*args, **kwargs):
        kwargs.pop("_query_cache", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", uncached)
    assert cached == sequence()


@pytest.mark.parametrize("learn,batch_size", [(False, 1), (True, 1), (True, 2)])
def test_zero_proposals_use_one_full_same_call_evaluation(monkeypatch, learn, batch_size):
    graph = Graph(1, 2, (("input", 0, 0), ("state", 0, 1)))
    calls = []
    original = _repair._evaluate

    def record(*args, **kwargs):
        assert kwargs.get("_query_cache") is None
        calls.append((args[2], args[3], args[4]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", record)
    result = _repair.settle(
        graph,
        [0.0] * batch_size,
        [0.0] * (2 * batch_size),
        [0.3, 0.7],
        [0.0, 0.0],
        learn=learn,
        _batch_size=batch_size,
        budget=0,
    )
    assert result["qualified"] and result["stationarity"] == 0
    assert result["sweeps"] == result["work"]["proposals"] == 0
    assert result["work"]["evaluations"] == 1
    assert len(calls) == batch_size
    assert result["work"]["patch_visits"] == 2 * graph.n_patches * batch_size
    assert result["work"]["edge_visits"] == 2 * len(graph.edges) * batch_size


@pytest.mark.parametrize("changed", ["inputs", "weights", "biases", "clamps"])
def test_zero_budget_rechecks_changed_problem_from_actual_post_clamp_state(monkeypatch, changed):
    graph = Graph(1, 2, (("input", 0, 0), ("state", 0, 1)))
    inputs, weights, biases = [0.4], [0.3, 0.7], [0.1, -0.2]
    prior = 0.2
    initial = _repair.settle(graph, inputs, [0.0, 0.0], weights, biases, state_prior=prior)
    assert initial["qualified"]
    state = initial["state"]
    clamps = {}
    if changed == "inputs":
        inputs = [0.9]
    elif changed == "weights":
        weights = [0.9, 0.7]
    elif changed == "biases":
        biases = [-0.5, 0.2]
    else:
        clamps = {0: 0.8}
    calls = []
    original = _repair._evaluate

    def record(*args, **kwargs):
        assert kwargs.get("_query_cache") is None
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", record)
    result = _repair.settle(
        graph,
        inputs,
        state,
        weights,
        biases,
        clamps=clamps,
        state_prior=prior,
        budget=0,
    )
    x0, x1 = (clamps.get(i, x) for i, x in enumerate(state))
    p0 = math.tanh(weights[0] * inputs[0] + biases[0])
    p1 = math.tanh(weights[1] * x0 + biases[1])
    e0, e1 = x0 - p0, x1 - p1
    gradients = (
        e0 + prior * x0 - weights[1] * (1 - p1 * p1) * e1,
        e1 + prior * x1,
    )
    residual = max(
        abs(x - max(-1, min(1, x - gradient)))
        for i, (x, gradient) in enumerate(zip((x0, x1), gradients, strict=True))
        if i not in clamps
    )
    assert calls == [(x0, x1)]
    assert result["predictions"] == pytest.approx((p0, p1), abs=1e-15)
    assert result["errors"] == pytest.approx((e0, e1), abs=1e-15)
    assert result["stationarity"] == pytest.approx(residual, abs=1e-15)
    assert not result["qualified"] and result["reason"] == "budget"
    assert result["work"]["evaluations"] == 1


def test_rejected_proposal_with_zero_sweeps_still_gets_fresh_final_check(monkeypatch):
    graph = Graph(1, 2, (("input", 0, 0), ("state", 0, 1)))
    caches = []
    original = _repair._evaluate

    def record(*args, **kwargs):
        caches.append(kwargs.get("_query_cache"))
        return original(*args, **kwargs)

    monkeypatch.setattr(_repair, "_evaluate", record)
    result = _repair.settle(
        graph,
        [0.9],
        [0.0, 0.0],
        [0.6, 0.4],
        [0.1, 0.2],
        step=1000,
        backtracks=1,
    )
    assert not result["qualified"] and result["reason"] == "line_search"
    assert result["sweeps"] == 0
    assert result["work"]["proposals"] == result["work"]["backtracks"] == 1
    assert result["work"]["evaluations"] == len(caches) == 3
    assert caches[0] is caches[-1] is None
    assert caches[1] is not None
