"""Dense defaults preserve each declared input path; sparse wiring is explicit."""

import json
import random

import pytest

import cadence.cortex as cortex_module
from cadence import Brain, Cortex, _repair


def wide_brain(**options):
    cortex = Cortex(seed=7, **options)
    sensors = cortex.input("sensors", shape=16)
    processing = cortex.column("processing", patches=8, inputs=sensors)
    cortex.column("reader", patches=8, inputs=processing)
    cortex.output("answer", shape=1, reads=processing)
    return cortex.build()


def output_sources(brain):
    return {
        source
        for kind, source, target in brain.graph.edges
        if kind == "input" and target == 0
    }


def test_default_output_reaches_every_declared_sensor_coordinate():
    brain = wide_brain()
    assert brain.config["fan_in"] is None
    assert len(brain.graph.edges) == 128 + 64
    for patch in range(8):
        assert {
            source
            for kind, source, target in brain.graph.edges
            if kind == "input" and target == patch
        } == set(range(16))


def test_sparse_aggregate_coverage_does_not_imply_output_dependency():
    sparse = wide_brain(fan_in=8, max_connections=128)
    dense = wide_brain()
    assert sparse.inspect()["sensor_coverage"] == 16
    assert len(sparse.graph.edges) == 128
    assert len(output_sources(sparse)) == 8
    missing = min(set(range(16)) - output_sources(sparse))
    # The output patch reads eight coordinates directly. Through the reader
    # population that settles with every processing patch it is coupled to the
    # other eight as well, and structural coverage reports that coupling.
    assert sparse.inspect()["outputs"][0]["sensor_coverage_by_coordinate"] == (16,)

    def predictions(brain):
        answers = []
        for value in (-1, 1):
            sensors = [0.0] * 16
            sensors[missing] = value
            result = brain.settle({"sensors": sensors})
            assert result["qualified"]
            answers.append(result["outputs"]["answer"][0])
        return answers

    low, high = predictions(dense)
    assert abs(low - high) > 0.01
    assert low == pytest.approx(-high, abs=1e-6)
    with pytest.raises(ValueError, match="Connection budget"):
        wide_brain(max_connections=128)


def multimodal_brain(fan_in):
    cortex = Cortex(seed=11, fan_in=fan_in)
    image = cortex.input("image", shape=5)
    sound = cortex.input("sound", shape=2)
    base = cortex.column("base", patches=3, inputs=(image, sound))
    observer = cortex.observer(
        "observer", patches=2, inputs=(image, sound, base), observes=base
    )
    cortex.output("answer", shape=1, reads=observer)
    return cortex.build()


def test_none_and_sufficient_integer_have_identical_graph_and_initial_relations():
    dense, explicit = multimodal_brain(None), multimodal_brain(64)
    assert dense.graph == explicit.graph
    assert dense.weights == explicit.weights
    assert dense.biases == explicit.biases
    assert dense.state == explicit.state
    # State readback shared by inputs/observes is deduplicated; error readback
    # remains a separate signal. All modes keep exactly the same patch rule.
    assert len(dense.graph.edges) == 3 * 7 + 2 * (7 + 3 + 3)
    inputs = {"image": [0.2] * 5, "sound": [0.3, -0.4]}
    assert dense.settle(inputs) == explicit.settle(inputs)


def test_dense_connection_budget_preflights_before_materializing_sources(monkeypatch):
    cortex = Cortex(max_connections=1024)
    sensor = cortex.input("large", shape=100_000)
    population = cortex.column("processing", patches=64, inputs=sensor)
    cortex.column("reader", patches=1, inputs=population)
    cortex.output("answer", shape=1, reads=population)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Over-budget source was materialized")

    monkeypatch.setattr(cortex_module, "list", unexpected_allocation, raising=False)
    monkeypatch.setattr(random.Random, "shuffle", unexpected_allocation)
    monkeypatch.setattr(_repair, "Graph", unexpected_allocation)
    with pytest.raises(ValueError, match="Connection budget"):
        cortex.build()


def test_dense_budget_is_inclusive():
    cortex = Cortex(max_connections=36)
    sensor = cortex.input("sensors", shape=8)
    population = cortex.column("processing", patches=4, inputs=sensor)
    cortex.column("reader", patches=1, inputs=population)
    cortex.output("answer", shape=1, reads=population)
    assert len(cortex.build().graph.edges) == 36


def test_null_configuration_roundtrips_after_admitted_experience():
    brain = wide_brain()
    inputs = {"sensors": [0.0] * 16}
    result = brain.observe(inputs, {"answer": [0.25]}, event_id=0)
    assert result["accepted"]
    checkpoint = brain.snapshot()
    assert json.loads(checkpoint)["config"]["fan_in"] is None
    restored = Brain.from_snapshot(checkpoint)
    assert restored.config["fan_in"] is None
    assert restored.snapshot() == checkpoint
    assert restored.predict(inputs) == brain.predict(inputs)
    assert restored.observe(inputs, {"answer": [0.25]}, event_id=0)["duplicate"]
    assert restored.snapshot() == checkpoint


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "8", "dense"])
def test_dense_sentinel_does_not_relax_integer_validation(invalid):
    with pytest.raises(ValueError, match="fan_in"):
        Cortex(fan_in=invalid)
