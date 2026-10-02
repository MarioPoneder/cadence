"""Real CPU tensor queries may reuse invariants during reference refinement."""

import pytest

from cadence.experimental.equilibrium import _repair
from cadence.experimental.equilibrium._tensor import TensorEngine

pytest.importorskip("torch")


@pytest.mark.parametrize("case", ("device_hint", "device_proposals"))
def test_tensor_reference_refinement_preserves_uncached_trajectory(monkeypatch, case):
    graph = _repair.Graph(
        1,
        3,
        (
            ("input", 0, 0),
            ("state", 0, 1),
            ("residual", 0, 1),
            ("state", 1, 2),
            ("residual", 1, 2),
            ("residual", 0, 2),
        ),
    )
    weights = (0.4, 0.3, 0.2, -0.25, 0.15, 0.1)
    if case == "device_hint":
        # This is already inside the binary32 stopping hint but outside the
        # requested tolerance. The real device phase must hand off to repair.
        inputs, state, biases = (0.0,), (1e-7, -1e-7, 1e-7), (0.0, 0.0, 0.0)
    else:
        # This also exercises genuine tensor proposals before strict float64
        # reference refinement. No device objective or gradient is mocked.
        inputs, state, biases = (0.4,), (0.0, 0.0, 0.0), (0.1, -0.1, 0.05)
    original = _repair._evaluate

    def run(*, reuse):
        calls = []

        def traced(*args, **kwargs):
            cache = kwargs.get("_query_cache")
            if not reuse:
                kwargs.pop("_query_cache", None)
            result = original(*args, **kwargs)
            calls.append((cache is not None, args[2], result))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(_repair, "_evaluate", traced)
            result = _repair.settle(
                graph,
                inputs,
                state,
                weights,
                biases,
                tolerance=1e-12,
                _engine=TensorEngine(graph, "cpu", "float32"),
            )
        return result, calls

    cached, calls = run(reuse=True)
    uncached, reference_calls = run(reuse=False)
    assert cached["qualified"] and cached["stationarity"] <= 1e-12
    assert cached["execution"]["reference_sweeps"] > 0
    if case == "device_hint":
        assert cached["execution"]["tensor_sweeps"] == 0
    else:
        assert cached["execution"]["tensor_sweeps"] > 0
    reused = sum(has_cache for has_cache, _, _ in calls)
    assert reused > 0
    assert not calls[0][0] and not calls[-1][0]
    assert calls[-1][1] == cached["state"]
    assert len(calls) == len(reference_calls)
    assert len(calls) == cached["execution"]["reference_evaluations"]
    assert {key: value for key, value in cached.items() if key != "work"} == {
        key: value for key, value in uncached.items() if key != "work"
    }
    for key in ("evaluations", "patch_visits", "proposals", "backtracks"):
        assert cached["work"][key] == uncached["work"][key]
    # Exactly one incoming input edge can be skipped per reused evaluation.
    # Tensor work and all reverse and final traversals remain charged.
    assert cached["work"]["edge_visits"] == uncached["work"]["edge_visits"] - reused

    final = _repair.evaluate(graph, inputs, cached["state"], weights, biases)
    for key in ("energy", "predictions", "errors"):
        assert final[key] == cached[key] == calls[-1][2][key]
    assert (
        _repair._stationarity(cached["state"], weights, biases, final, {}, False, 1.0, 4.0)
        == cached["stationarity"]
    )
