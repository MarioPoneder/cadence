"""Population construction, causal coupling and transactional learning contracts."""

import hashlib
import json
import math
from decimal import Decimal
from fractions import Fraction
from importlib.resources import files

import pytest

from cadence.experimental.equilibrium import Brain, Cortex, SettlementError
from cadence.experimental.equilibrium._repair import Graph, settle
from cadence.experimental.equilibrium.brain import IMPLEMENTATION


def small_brain(**options):
    config = dict(seed=11, fan_in=2, settle_budget=2048, tolerance=1e-7)
    config.update(options)
    cortex = Cortex(**config)
    sensor = cortex.input("sensor", shape=(2,))
    base = cortex.column("base", patches=3, inputs=(sensor,))
    observer = cortex.observer("observer", patches=2, observes=(base,))
    cortex.output("answer", shape=(2,), reads=base, indices=(2, 0))
    cortex.output("readback", shape=(2,), reads=observer)
    return cortex.build()


def test_width_parallel_sources_and_depth_are_separate_resources():
    cortex = Cortex(seed=5, fan_in=1)
    eye = cortex.input("eye", shape=(2, 3))
    ear = cortex.input("ear", shape=(2,))
    left = cortex.column("left", patches=3, inputs=(eye,))
    right = cortex.column("right", patches=4, inputs=(ear,))
    first = cortex.observer("first", patches=2, observes=(left, right))
    second = cortex.observer("second", patches=5, inputs=(ear,), observes=(left, first))
    cortex.output("motor", shape=(2,), reads=second)
    brain = cortex.build()
    assert brain.graph.n_inputs == 8
    assert brain.graph.n_patches == 14
    assert len(brain.state) == len(brain.biases) == 14
    assert len(brain.weights) == len(brain.graph.edges)
    # Output ports are views of existing patch state, not additional neurons.
    assert brain.graph.n_patches == 3 + 4 + 2 + 5
    connected_inputs = {s for kind, s, _ in brain.graph.edges if kind == "input"}
    assert connected_inputs == set(range(8))
    for source_group, target_group in [({0, 1, 2}, {7, 8}), ({3, 4, 5, 6}, {7, 8})]:
        for kind in ("state", "residual"):
            covered = {
                source
                for edge_kind, source, target in brain.graph.edges
                if edge_kind == kind and target in target_group
            }
            assert source_group <= covered


def test_initialization_is_seeded_without_global_random_state():
    import random

    random.seed(987)
    before = random.getstate()
    first, second = small_brain(), small_brain()
    assert random.getstate() == before
    assert first.snapshot() == second.snapshot()
    other = small_brain(seed=12)
    assert first.weights != other.weights or first.biases != other.biases


@pytest.mark.parametrize("fan_in,connections", [(1, 18), (2, 24), (10, 37)])
def test_reading_and_observing_the_same_population_keeps_one_state_port(fan_in, connections):
    def layout(limit):
        cortex = Cortex(fan_in=fan_in, max_connections=limit)
        sensor = cortex.input("sensor", shape=5)
        base = cortex.column("base", patches=2, inputs=sensor)
        observer = cortex.observer("observer", patches=3, inputs=(base, sensor), observes=base)
        cortex.output("answer", shape=1, reads=observer)
        return cortex

    brain = layout(connections).build()
    assert len(brain.graph.edges) == len(set(brain.graph.edges)) == connections
    assert brain.inspect()["sensor_coverage"] == 5
    for kind in ("state", "residual"):
        assert {s for k, s, _ in brain.graph.edges if k == kind} == {0, 1}
    with pytest.raises(ValueError, match="Connection budget"):
        layout(connections - 1).build()


@pytest.mark.parametrize("value", [1, 0.5, Decimal("0.25"), Fraction(1, 8)])
def test_builder_and_repair_accept_real_numeric_types(value):
    cortex = Cortex(tolerance=value)
    assert cortex.config["tolerance"] == float(value)
    assert settle(Graph(0, 1, ()), [], [0], [], [0], tolerance=value)["qualified"]


@pytest.mark.parametrize(
    "value",
    [True, None, "0.2", 0.2j, math.inf, math.nan, -1, 0, Decimal("sNaN"), 10**1000],
)
def test_builder_and_repair_reject_invalid_positive_options(value):
    with pytest.raises(ValueError, match="tolerance"):
        Cortex(tolerance=value)
    with pytest.raises(ValueError, match="tolerance"):
        settle(Graph(0, 1, ()), [], [0], [], [0], tolerance=value)


def test_flat_and_nested_sensor_values_are_identical_and_outputs_are_state_views():
    cortex = Cortex(seed=1, settle_budget=2048)
    sensor = cortex.input("image", shape=(2, 2))
    base = cortex.column("base", patches=3, inputs=(sensor,))
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("answer", shape=(2,), reads=base, indices=(2, 0))
    brain = cortex.build()
    before = brain.snapshot()
    nested = brain.settle({sensor: [[0.2, -0.4], [0.1, 0.7]]})
    flat = brain.settle({"image": [0.2, -0.4, 0.1, 0.7]})
    assert nested["qualified"] and flat["qualified"]
    assert nested == flat
    assert flat["outputs"]["answer"] == (flat["state"][2], flat["state"][0])
    assert brain.snapshot() == before
    assert brain.predict({"image": [0.2, -0.4, 0.1, 0.7]}) == flat["outputs"]
    assert brain.snapshot() == before


def test_observer_feedback_is_two_way_and_disappears_with_contacts_cut():
    brain = small_brain()
    inputs = {"sensor": [0.3, -0.2]}
    low = brain.settle(inputs, interventions={"observer": [-0.8, -0.8]})
    high = brain.settle(inputs, interventions={"observer": [0.8, 0.8]})
    assert low["qualified"] and high["qualified"]
    assert max(abs(a - b) for a, b in zip(low["state"][:3], high["state"][:3], strict=True)) > 1e-5
    base_low = brain.settle(inputs, interventions={"base": [-0.6] * 3})
    base_high = brain.settle(inputs, interventions={"base": [0.6] * 3})
    assert base_low["qualified"] and base_high["qualified"]
    assert (
        max(abs(a - b) for a, b in zip(base_low["state"][3:], base_high["state"][3:], strict=True))
        > 1e-5
    )
    retained = [
        i
        for i, (kind, source, target) in enumerate(brain.graph.edges)
        if kind == "input" or ((source < 3) == (target < 3))
    ]
    graph = Graph(2, 5, tuple(brain.graph.edges[i] for i in retained))
    weights = [brain.weights[i] for i in retained]
    cuts = [
        settle(
            graph,
            [0.3, -0.2],
            brain.state,
            weights,
            brain.biases,
            clamps={3: value, 4: value},
            budget=2048,
            tolerance=1e-7,
        )
        for value in (-0.8, 0.8)
    ]
    assert all(result["qualified"] for result in cuts)
    assert cuts[0]["state"][:3] == pytest.approx(cuts[1]["state"][:3], abs=1e-9)


def test_observed_target_changes_later_target_free_prediction():
    cortex = Cortex(seed=2, fan_in=1, settle_budget=4096, tolerance=1e-7)
    sensor = cortex.input("sensor", shape=(1,))
    base = cortex.column("base", patches=1, inputs=(sensor,))
    cortex.column("reader", patches=1, inputs=(base,))
    output = cortex.output("answer", shape=(1,), reads=base)
    brain = cortex.build()
    inputs = {"sensor": [0.4]}
    before = brain.predict(inputs)["answer"][0]
    weights, biases = brain.weights, brain.biases
    result = brain.observe(inputs, {output: [0.6]}, event_id=1)
    assert result["accepted"] and result["qualified"] and not result["duplicate"]
    assert brain.weights != weights or brain.biases != biases
    after = brain.predict(inputs)["answer"][0]
    assert abs(after - 0.6) < abs(before - 0.6) * 0.6
    assert abs(after - 0.6) > 1e-4  # This is a new free solve, not the target clamp.


def test_alternating_witnesses_learn_a_sensor_dependent_free_response():
    cortex = Cortex(seed=2, settle_budget=1200, tolerance=1e-5)
    sensor = cortex.input("sensor", shape=1)
    base = cortex.column("base", patches=4, inputs=(sensor,))
    observer = cortex.observer("observer", patches=2, inputs=(sensor,), observes=(base,))
    cortex.output("answer", shape=1, reads=observer)
    brain = cortex.build()
    before = tuple(brain.predict({"sensor": [x]})["answer"][0] for x in (-0.8, 0.8))
    for event in range(40):
        x = -0.8 if event % 2 == 0 else 0.8
        assert brain.observe({"sensor": [x]}, {"answer": [x]})["accepted"]
    after = tuple(brain.predict({"sensor": [x]})["answer"][0] for x in (-0.8, 0.8))
    assert sum((y - x) ** 2 for y, x in zip(after, (-0.8, 0.8), strict=True)) < 0.05
    assert (
        sum((y - x) ** 2 for y, x in zip(after, (-0.8, 0.8), strict=True))
        < sum((y - x) ** 2 for y, x in zip(before, (-0.8, 0.8), strict=True)) / 10
    )
    # These amplitudes were never taught. No target is supplied to any probe.
    heldout = tuple(brain.predict({"sensor": [x]})["answer"][0] for x in (-0.4, 0.4))
    assert -0.7 < heldout[0] < -0.2 and 0.2 < heldout[1] < 0.7
    # Same learned biases and recurrent relations; sever only sensor influence.
    cut_weights = tuple(
        0.0 if kind == "input" else weight
        for (kind, _, _), weight in zip(brain.graph.edges, brain.weights, strict=True)
    )
    cut = [
        settle(
            brain.graph,
            [x],
            brain.state,
            cut_weights,
            brain.biases,
            budget=1200,
            tolerance=1e-5,
        )
        for x in (-0.4, 0.4)
    ]
    assert all(r["qualified"] for r in cut)
    assert cut[0]["state"] == cut[1]["state"]


def test_refused_observe_is_atomic_and_does_not_consume_event():
    brain = small_brain()
    before = brain.snapshot()
    values, targets = {"sensor": [0.5, 0.1]}, {"answer": [0.6, -0.4]}
    refused = brain.observe(values, targets, event_id=1, budget=0)
    assert not refused["accepted"] and not refused["qualified"]
    assert brain.snapshot() == before
    accepted = brain.observe(values, targets, event_id=1)
    assert accepted["accepted"] and accepted["qualified"]


def test_only_exact_latest_event_retries_are_duplicates():
    brain = small_brain()
    values, targets = {"sensor": [0.2, -0.3]}, {"answer": [0.4, -0.2]}
    assert brain.observe(values, targets, event_id=1)["accepted"]
    before = brain.snapshot()
    retry = brain.observe(values, targets, event_id=1, budget=0)
    assert retry["duplicate"] and not retry["accepted"]
    assert brain.snapshot() == before
    with pytest.raises(ValueError):
        brain.observe(values, {"answer": [0.5, -0.2]}, event_id=1)
    assert brain.snapshot() == before
    assert brain.observe(values, targets, event_id=2)["accepted"]
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.observe(values, targets, event_id=1)
    assert brain.snapshot() == before


def test_checkpoint_restores_learning_fast_state_and_retry_custody():
    original = small_brain()
    values = {"sensor": [0.2, -0.1]}
    assert original.observe(values, {"answer": [0.4, -0.3]}, event_id=1)["accepted"]
    assert original.step({"sensor": [-0.4, 0.6]})["qualified"]
    clone = Brain.from_snapshot(original.snapshot())
    assert clone.snapshot() == original.snapshot()
    assert clone.predict(values) == original.predict(values)
    for model in (original, clone):
        assert model.observe(values, {"answer": [-0.2, 0.4]}, event_id=2)["accepted"]
    assert clone.snapshot() == original.snapshot()


def test_conflicting_output_aliases_are_rejected_atomically():
    cortex = Cortex(seed=1)
    sensor = cortex.input("sensor", shape=(1,))
    base = cortex.column("base", patches=1, inputs=(sensor,))
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("one", shape=(1,), reads=base)
    cortex.output("two", shape=(1,), reads=base)
    brain = cortex.build()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.observe({"sensor": [0.1]}, {"one": [0.2], "two": [-0.2]})
    assert brain.snapshot() == before


@pytest.mark.parametrize("value", [True, math.nan, math.inf, "0.2"])
def test_invalid_runtime_data_cannot_mutate_learning(value):
    brain = small_brain()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.observe({"sensor": [value, 0.1]}, {"answer": [0.2, 0.3]})
    assert brain.snapshot() == before
    with pytest.raises(ValueError):
        brain.observe({"sensor": [0.2, 0.1]}, {"answer": [value, 0.3]})
    assert brain.snapshot() == before


def test_input_and_population_ownership_is_not_name_equality():
    first, second = Cortex(), Cortex()
    a, b = first.input("sensor", shape=(1,)), second.input("sensor", shape=(1,))
    base = first.column("base", patches=1, inputs=(a,))
    with pytest.raises(ValueError):
        first.column("foreign", patches=1, inputs=(b,))
    with pytest.raises(ValueError):
        second.observer("foreign", patches=1, observes=(base,))
    with pytest.raises(ValueError):
        second.output("foreign", shape=(1,), reads=base)


@pytest.mark.parametrize("patches", [True, 0, -1, 1.5])
def test_population_width_is_a_positive_integer(patches):
    cortex = Cortex()
    sensor = cortex.input("sensor", shape=(1,))
    with pytest.raises(ValueError):
        cortex.column("base", patches=patches, inputs=(sensor,))


@pytest.mark.parametrize("shape", [(0,), (True,), (1.5,), (-1, 2)])
def test_invalid_sensor_shapes_fail_before_layout_allocation(shape):
    with pytest.raises(ValueError):
        Cortex().input("sensor", shape=shape)


def test_resource_limits_refuse_before_building_oversize_graph():
    cortex = Cortex(max_inputs=3, max_patches=3, max_connections=3)
    with pytest.raises(ValueError):
        cortex.input("large", shape=(4,))
    sensor = cortex.input("sensor", shape=(2,))
    with pytest.raises(ValueError):
        cortex.column("large", patches=4, inputs=(sensor,))


def test_checkpoint_rejects_nonfinite_state_without_partial_restore():
    brain = small_brain()
    before = brain.snapshot()
    checkpoint = json.loads(before)
    assert "state" in checkpoint
    checkpoint["state"][0] = math.nan
    with pytest.raises(ValueError):
        brain.restore(json.dumps(checkpoint))
    assert brain.snapshot() == before


@pytest.mark.parametrize("field", ["state", "weights", "biases"])
def test_checkpoint_wrong_vector_sizes_are_rejected_atomically(field):
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    data[field].append(0.0)
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


@pytest.mark.parametrize("part", ["inputs", "populations", "outputs"])
def test_checkpoint_unknown_layout_fields_are_not_silently_discarded(part):
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    data["layout"][part][0]["unknown_semantics"] = True
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


def test_checkpoint_layout_configuration_and_fingerprint_are_bound():
    brain = small_brain()
    before = brain.snapshot()
    for field, value in [("seed", 12), ("state_prior", 0.02)]:
        data = json.loads(before)
        data["config"][field] = value
        with pytest.raises(ValueError):
            brain.restore(json.dumps(data))
        assert brain.snapshot() == before
    data = json.loads(before)
    data["fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


def test_checkpoint_implementation_identity_is_required_and_checked():
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    assert isinstance(data["implementation"], dict) and data["implementation"]
    key = next(iter(data["implementation"]))
    data["implementation"][key] = "0" * 64
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before
    data = json.loads(before)
    del data["implementation"]
    with pytest.raises(ValueError):
        Brain.from_snapshot(json.dumps(data))


def test_checkpoint_impossible_declared_width_is_rejected_before_graph_build(
    monkeypatch,
):
    brain = small_brain()
    data = json.loads(brain.snapshot())
    data["layout"]["populations"][0]["patches"] = 1000

    def forbidden_build(*args, **kwargs):
        pytest.fail("Malformed vector counts reached graph allocation")

    monkeypatch.setattr(Cortex, "_compile", forbidden_build)
    with pytest.raises(ValueError):
        Brain.from_snapshot(json.dumps(data))


def test_inspection_and_result_mutation_cannot_change_owned_state():
    brain = small_brain()
    before = brain.snapshot()
    inspected = brain.inspect()
    inspected["config"]["state_prior"] = 99
    inspected["populations"][0]["name"] = "tampered"
    result = brain.settle({"sensor": [0.2, 0.3]})
    result["state"] = (0.9,) * len(brain.state)
    result["outputs"]["answer"] = (0.9, 0.9)
    assert brain.snapshot() == before
    with pytest.raises(TypeError):
        brain.config["state_prior"] = 99
    with pytest.raises(AttributeError):
        brain.state = (0.9,) * len(brain.state)
    with pytest.raises(AttributeError):
        brain.graph = Graph(0, 1, ())
    with pytest.raises(AttributeError):
        brain.config = {}


def test_pure_target_clamp_is_not_a_learning_event_and_failed_step_is_atomic():
    brain = small_brain()
    before = brain.snapshot()
    result = brain.settle({"sensor": [0.2, 0.1]}, targets={"answer": [0.3, 0.4]})
    assert result["qualified"]
    assert result["outputs"]["answer"] == (0.3, 0.4)
    assert brain.snapshot() == before
    result = brain.step({"sensor": [0.9, -0.7]}, budget=0)
    assert not result["qualified"] and not result["accepted"]
    assert brain.snapshot() == before


def test_shape_keys_and_duplicate_runtime_references_are_validated():
    cortex = Cortex()
    sensor = cortex.input("image", shape=(2, 2))
    base = cortex.column("base", patches=2, inputs=(sensor,))
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("answer", shape=1, reads=base)
    brain = cortex.build()
    before = brain.snapshot()
    for inputs in (
        {},
        {"unknown": [0.0] * 4},
        {"image": [[0.0, 0.1, 0.2], [0.3]]},
        {"image": [0.0] * 3},
        {"image": [0.0] * 4, sensor: [0.0] * 4},
    ):
        with pytest.raises(ValueError):
            brain.settle(inputs)
        assert brain.snapshot() == before


def test_built_layout_is_frozen_and_connection_budget_refusal_is_retryable():
    cortex = Cortex(max_connections=1)
    sensor = cortex.input("sensor", shape=2)
    base = cortex.column("base", patches=1, inputs=(sensor,))
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("answer", shape=1, reads=base)
    with pytest.raises(ValueError, match="Connection"):
        cortex.build()
    with pytest.raises(ValueError, match="Connection"):
        cortex.build()
    cortex = Cortex()
    base = cortex.column(patches=1)
    cortex.column(patches=1, inputs=base)
    cortex.output("answer", shape=(), reads=base)
    brain = cortex.build()
    assert brain.predict({})["answer"] == (0.0,)
    with pytest.raises(ValueError):
        cortex.column(patches=1)
    with pytest.raises(ValueError):
        cortex.build()


def test_requested_three_population_layout_has_exactly_768_processing_patches():
    cortex = Cortex(seed=7, fan_in=2)
    eyes = cortex.input("eyes", shape=(4, 4))
    ears = cortex.input("ears", shape=8)
    body = cortex.input("body", shape=4)
    sources = (eyes, ears, body)
    first = cortex.column("c1", patches=256, inputs=sources)
    second = cortex.observer("c2", patches=256, inputs=sources, observes=(first,))
    third = cortex.observer("c3", patches=256, inputs=sources, observes=(first, second))
    cortex.output("motor", shape=8, reads=third)
    brain = cortex.build()
    info = brain.inspect()
    assert info["patches"] == len(brain.state) == len(brain.biases) == 768
    assert info["input_samples"] == info["sensor_coverage"] == 28
    assert [p["patches"] for p in info["populations"]] == [256, 256, 256]
    assert [p["role"] for p in info["populations"]] == [
        "processing",
        "observer",
        "observer",
    ]
    assert info["populations"][2]["observes"] == ["c1", "c2"]
    assert len(info["outputs"]) == 1
    assert Brain.from_snapshot(brain.snapshot()).snapshot() == brain.snapshot()


@pytest.mark.parametrize("event", [True, -1, 1.5, math.nan])
def test_invalid_event_identity_never_changes_state(event):
    brain = small_brain()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.observe({"sensor": [0.0, 0.0]}, {"answer": [0.2, 0.3]}, event_id=event)
    assert brain.snapshot() == before


def test_empty_targets_and_unobserving_observer_are_rejected():
    brain = small_brain()
    before = brain.snapshot()
    with pytest.raises(ValueError):
        brain.observe({"sensor": [0.0, 0.0]}, {})
    assert brain.snapshot() == before
    with pytest.raises(ValueError):
        Cortex().observer(patches=2, observes=())


@pytest.mark.parametrize("field", ["weights", "biases"])
def test_pristine_checkpoint_cannot_claim_unadmitted_parameter_changes(field):
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    data[field][0] += 0.01
    with pytest.raises(ValueError, match="without an admitted event"):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


@pytest.mark.parametrize("key", ["state_prior", "seed", "max_connections"])
def test_checkpoint_requires_complete_configuration_even_for_defaults(key):
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    del data["config"][key]
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


def test_large_finite_initial_scale_does_not_overflow_random_interval():
    cortex = Cortex(initial_scale=1e308, parameter_bound=1e308)
    source = cortex.input("source", shape=2)
    population = cortex.column(patches=2, inputs=(source,))
    cortex.column(patches=1, inputs=(population,))
    cortex.output("answer", shape=1, reads=population)
    brain = cortex.build()
    assert all(math.isfinite(w) and abs(w) <= 1e308 for w in brain.weights)


def test_observed_population_declared_as_input_does_not_duplicate_a_state_port():
    cortex = Cortex(fan_in=1, max_connections=3)
    sensor = cortex.input("sensor", shape=1)
    base = cortex.column("base", patches=1, inputs=(sensor,))
    observer = cortex.observer("observer", patches=1, inputs=(base,), observes=(base,))
    cortex.output("answer", shape=1, reads=observer)
    brain = cortex.build()
    assert brain.graph.edges == (
        ("input", 0, 0),
        ("state", 0, 1),
        ("residual", 0, 1),
    )
    assert brain.inspect()["connections"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", -1),
        ("event_id", True),
        ("admissions", 0),
        ("admissions", 999),
        ("event_digest", "0"),
        ("event_digest", "G" * 64),
        ("event_digest", None),
        ("weights", None),
    ],
)
def test_checkpoint_malformed_event_custody_is_refused_atomically(field, value):
    brain = small_brain()
    assert brain.observe({"sensor": [0.4, 0.1]}, {"answer": [0.3, 0.2]})["accepted"]
    before = brain.snapshot()
    data = json.loads(before)
    data[field] = value
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


@pytest.mark.parametrize("field", ["state", "weights", "biases"])
def test_checkpoint_finite_but_out_of_bound_state_is_refused(field):
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    data[field][0] = 99.0
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


def test_restoring_valid_but_different_model_is_atomic():
    brain = small_brain()
    before = brain.snapshot()
    other = small_brain(seed=12)
    with pytest.raises(ValueError):
        brain.restore(other.snapshot())
    assert brain.snapshot() == before


def test_unknown_checkpoint_schema_and_missing_reference_are_not_reinterpreted():
    brain = small_brain()
    before = brain.snapshot()
    data = json.loads(before)
    data["schema"] = "different-brain/1"
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    data = json.loads(before)
    data["layout"]["outputs"][0]["reads"] = "missing-population"
    with pytest.raises(ValueError):
        brain.restore(json.dumps(data))
    assert brain.snapshot() == before


def test_prediction_refuses_a_nonstationary_zero_budget_model():
    brain = small_brain(settle_budget=0)
    result = brain.settle({"sensor": [0.9, -0.7]})
    assert not result["qualified"]
    with pytest.raises(SettlementError):
        brain.predict({"sensor": [0.9, -0.7]})


def test_layout_handles_are_immutable_owned_references():
    from dataclasses import FrozenInstanceError, replace

    cortex = Cortex()
    sensor = cortex.input("sensor", shape=1)
    population = cortex.column("population", patches=1, inputs=(sensor,))
    output = cortex.output("answer", shape=1, reads=population)
    for node in (sensor, population, output):
        assert not hasattr(node, "__dict__")
        with pytest.raises(FrozenInstanceError):
            node.name = "changed"
        copied = replace(node)
        assert copied is not node and copied != node
    with pytest.raises(ValueError, match="existing nodes"):
        cortex.column("forged", patches=1, inputs=(replace(sensor),))
    cortex.column("reader", patches=1, inputs=(population,))
    brain = cortex.build()
    with pytest.raises(ValueError, match="foreign"):
        brain.predict({replace(sensor): [0.2]})


def test_recursive_handle_hash_and_repr_do_not_walk_ancestry(monkeypatch):
    from cadence.experimental.equilibrium import Input

    cortex = Cortex(fan_in=1)
    sensor = cortex.input("sensor", shape=1)
    first = cortex.column("base", patches=1, inputs=(sensor,))

    def forbidden_hash(_):
        pytest.fail("A population reference recursively hashed sensor ancestry")

    monkeypatch.setattr(Input, "__hash__", forbidden_hash)
    populations = [first]
    for index in range(32):
        populations.append(
            cortex.observer(f"observer{index}", patches=1, observes=tuple(populations))
        )
    rendered = repr(populations[-1])
    assert len(rendered) < 1024
    assert rendered.count("Population(") == 1
    cortex.output("answer", shape=1, reads=populations[-1])
    brain = cortex.build()
    assert brain.graph.n_patches == 33
    assert brain.inspect()["connections"] == 1 + 32 * 33


def test_deep_observer_handle_can_be_used_without_python_recursion():
    cortex = Cortex(fan_in=1)
    population = cortex.column("base", patches=1)
    for index in range(1100):
        population = cortex.observer(f"observer{index}", patches=1, observes=(population,))
    cortex.output("answer", shape=1, reads=population)
    assert cortex.build().graph.n_patches == 1101


def test_source_provenance_includes_validation_and_is_read_only():
    assert set(IMPLEMENTATION) == {
        "brain.py",
        "cortex.py",
        "column.py",
        "ports.py",
        "_repair.py",
        "_validation.py",
        "_tensor.py",
    }
    for name, digest in IMPLEMENTATION.items():
        assert (
            digest
            == hashlib.sha256(
                files("cadence.experimental.equilibrium").joinpath(name).read_bytes()
            ).hexdigest()
        )
    with pytest.raises(TypeError):
        IMPLEMENTATION["_validation.py"] = "0" * 64
    info = small_brain().inspect()
    info["implementation"]["_validation.py"] = "0" * 64
    assert info["implementation"] != IMPLEMENTATION


@pytest.mark.parametrize("kind", ["input", "column", "observer", "output"])
def test_invalid_unicode_names_are_refused_before_layout_changes(kind):
    cortex = Cortex()
    source = cortex.input("source", shape=1)
    base = cortex.column("base", patches=1, inputs=(source,))
    options = {
        "input": {"shape": 1},
        "column": {"patches": 1},
        "observer": {"patches": 1, "observes": (base,)},
        "output": {"shape": 1, "reads": base},
    }
    with pytest.raises(ValueError, match="UTF-8"):
        getattr(cortex, kind)("\ud800", **options[kind])
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("answer", shape=1, reads=base)
    brain = cortex.build()
    assert brain.graph.n_inputs == 1 and brain.graph.n_patches == 2
    assert Brain.from_snapshot(brain.snapshot()).snapshot() == brain.snapshot()


def test_connection_and_output_iterators_are_consumed_with_declared_limits():
    cortex = Cortex()
    sensor = cortex.input("sensor", shape=1)

    def too_many_sources():
        yield sensor
        yield sensor
        pytest.fail("Source validation consumed beyond the existing node count")

    with pytest.raises(ValueError):
        cortex.column("invalid", patches=1, inputs=too_many_sources())
    base = cortex.column("base", patches=1, inputs=(sensor,))

    def too_many_indices():
        yield 0
        yield 0
        pytest.fail("Index validation consumed beyond the declared output size")

    with pytest.raises(ValueError):
        cortex.output("invalid", shape=1, reads=base, indices=too_many_indices())
    cortex.column("reader", patches=1, inputs=(base,))
    cortex.output("answer", shape=1, reads=base)
    assert cortex.build().graph.n_patches == 2


@pytest.mark.parametrize("shape", [(1000000,), (1, 2), 2])
def test_wrong_array_shape_is_rejected_before_copying_samples(shape):
    class WrongArray:
        def tolist(self):
            pytest.fail("A wrong-sized array was materialized before validation")

    value = WrongArray()
    value.shape = shape
    brain = small_brain()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="shape"):
        brain.step({"sensor": value})
    assert brain.snapshot() == before


def test_array_protocol_accepts_nested_and_flat_values_without_a_dependency():
    class Array:
        def __init__(self, shape, data):
            self.shape, self.data = shape, data

        def tolist(self):
            return self.data

    cortex = Cortex()
    source = cortex.input("image", shape=(2, 2))
    population = cortex.column("base", patches=1, inputs=(source,))
    cortex.column("reader", patches=1, inputs=(population,))
    cortex.output("answer", shape=1, reads=population)
    brain = cortex.build()
    nested = Array((2, 2), [[0.1, 0.2], [0.3, 0.4]])
    flat = Array((4,), [0.1, 0.2, 0.3, 0.4])
    assert brain.predict({source: nested}) == brain.predict({source: flat})


def test_malformed_foreign_reference_names_raise_validation_errors():
    from cadence.experimental.equilibrium import Input

    class Foreign:
        name = []

    brain = small_brain()
    before = brain.snapshot()
    with pytest.raises(ValueError, match="foreign"):
        brain.predict({Foreign(): [0.1, 0.2]})
    assert brain.snapshot() == before
    cortex = Cortex()
    cortex.input("sensor", shape=1)
    forged = Input([], (1,))
    with pytest.raises(ValueError, match="existing nodes"):
        cortex.column("invalid", patches=1, inputs=(forged,))


@pytest.mark.parametrize("module", sorted(IMPLEMENTATION))
@pytest.mark.parametrize("operation", ["change", "remove"])
def test_checkpoint_binds_every_semantic_module_before_restoring(module, operation):
    brain = small_brain()
    assert brain.observe({"sensor": [0.3, -0.2]}, {"answer": [0.4, -0.1]})["accepted"]
    before = brain.snapshot()
    tampered = json.loads(before)
    if operation == "change":
        tampered["implementation"][module] = "0" * 64
    else:
        del tampered["implementation"][module]
    text = json.dumps(tampered)
    with pytest.raises(ValueError, match="implementation mismatch"):
        brain.restore(text)
    assert brain.snapshot() == before
    with pytest.raises(ValueError, match="implementation mismatch"):
        Brain.from_snapshot(text)
