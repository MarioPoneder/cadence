"""Expose structural input gaps without forbidding intentional constant branches."""

import pytest

from cadence.experimental.equilibrium import Brain, Cortex


def output_coverage(brain, name="answer"):
    return next(
        output["sensor_coverage_by_coordinate"]
        for output in brain.inspect()["outputs"]
        if output["name"] == name
    )


def test_output_on_a_sensorless_population_is_coupled_to_the_sensors_by_its_reader():
    cortex = Cortex(seed=7, initial_scale=1.5, tolerance=1e-9, settle_budget=2048)
    sensor = cortex.input("sensor", shape=2)
    perception = cortex.column("perception", patches=3, inputs=sensor)
    constant = cortex.column("constant", patches=1)
    cortex.column("joint", patches=1, inputs=(perception, constant))
    cortex.output("answer", shape=1, reads=constant)
    brain = cortex.build()
    assert brain.inspect()["sensor_coverage"] == 2
    # The constant patch reads no sensor. The population that reads it also
    # reads perception, so the joint settlement couples the answer to both
    # sensor coordinates and to all five patches.
    assert output_coverage(brain) == (2,)
    assert brain.inspect()["output_connected_patches"] == 5
    first = brain.predict({"sensor": [-0.8, 0.2]})["answer"][0]
    second = brain.predict({"sensor": [0.9, -0.6]})["answer"][0]
    assert abs(first - second) > 1e-5


def test_sensorless_population_that_nothing_reads_is_refused():
    cortex = Cortex()
    sensor = cortex.input("sensor", shape=2)
    perception = cortex.column("perception", patches=3, inputs=sensor)
    cortex.column("integration", patches=1, inputs=perception)
    constant = cortex.column("constant", patches=1)
    cortex.output("answer", shape=1, reads=constant)
    with pytest.raises(ValueError, match="'constant' settles with no other"):
        cortex.build()


def test_shared_fixed_input_does_not_bridge_independent_patch_components():
    cortex = Cortex(seed=7, fan_in=1)
    sensor = cortex.input("sensor", shape=3)
    processing = cortex.column("processing", patches=2, inputs=sensor)
    cortex.column("readers", patches=2, inputs=processing)
    cortex.output("answer", shape=1, reads=processing)
    # The population-level read joins two disjoint processing/reader pairs.
    # Their shared fixed sensory coordinates cannot merge the equilibria.
    with pytest.raises(ValueError, match="disconnected settlement components"):
        cortex.build()


@pytest.mark.parametrize("join", ["column", "observer"])
def test_shared_downstream_population_returns_influence_to_upstream_output(join):
    cortex = Cortex(seed=7, initial_scale=1.5, tolerance=1e-9, settle_budget=2048)
    first = cortex.input("first", shape=1)
    second = cortex.input("second", shape=1)
    left = cortex.column("left", patches=1, inputs=first)
    right = cortex.column("right", patches=1, inputs=second)
    if join == "column":
        cortex.column("joint", patches=1, inputs=(left, right))
    else:
        cortex.observer("joint", patches=1, observes=(left, right))
    cortex.output("answer", shape=1, reads=left)
    brain = cortex.build()
    before = brain.snapshot()
    answers = [
        brain.predict({"first": [0.0], "second": [value]})["answer"][0] for value in (-0.8, 0.8)
    ]
    # No forward read path runs from second to left. The joint energy
    # nevertheless couples them through the shared downstream population.
    assert output_coverage(brain) == (2,)
    assert brain.inspect()["output_connected_patches"] == 3
    assert abs(answers[0] - answers[1]) > 1e-5
    assert brain.snapshot() == before


def test_output_aliases_and_selected_indices_report_coverage_and_coupled_patches():
    cortex = Cortex()
    sensor = cortex.input("sensor", shape=2)
    processing = cortex.column("processing", patches=4, inputs=sensor)
    cortex.column("reader", patches=1, inputs=processing)
    cortex.output("pair", shape=(1, 2), reads=processing, indices=(3, 1))
    cortex.output("alias", shape=(), reads=processing, indices=(1,))
    brain = cortex.build()
    assert output_coverage(brain, "pair") == (2, 2)
    assert output_coverage(brain, "alias") == (2,)
    # The reader settles with all four processing patches, so every patch of
    # the brain is coupled to the outputs.
    assert brain.inspect()["output_connected_patches"] == 5


@pytest.mark.parametrize(
    "indices,coverage",
    [((0,), (3,)), ((1, 0, 2), (3, 3, 3))],
)
def test_sparse_error_contacts_extend_coverage_beyond_state_contacts(indices, coverage):
    cortex = Cortex(seed=3, fan_in=1)
    sensor = cortex.input("sensor", shape=3)
    base = cortex.column("base", patches=3, inputs=sensor)
    cortex.observer("observer", patches=3, observes=base)
    cortex.output("answer", shape=len(indices), reads=base, indices=indices)
    brain = cortex.build()
    # State contacts alone form three separate pairs. Error readback joins
    # them into one six-patch component, carrying every sensor to every output.
    assert {e for e in brain.graph.edges if e[0] == "state"} == {
        ("state", 0, 3),
        ("state", 2, 4),
        ("state", 1, 5),
    }
    assert {e for e in brain.graph.edges if e[0] == "residual"} == {
        ("residual", 1, 3),
        ("residual", 0, 4),
        ("residual", 2, 5),
    }
    assert output_coverage(brain) == coverage
    assert brain.inspect()["output_connected_patches"] == 6


def test_partial_error_readback_cannot_leave_a_separate_patch_component():
    cortex = Cortex(seed=0, fan_in=1)
    sensor = cortex.input("sensor", shape=3)
    base = cortex.column("base", patches=3, inputs=sensor)
    cortex.observer("observer", patches=3, observes=base)
    cortex.output("answer", shape=1, reads=base)
    # In this sampled graph, error contacts join the first and third state
    # pairs but leave the middle pair separate. An observer label is not proof
    # of one connected compiled equilibrium.
    with pytest.raises(ValueError, match="disconnected settlement components"):
        cortex.build()


def test_constant_brain_has_valid_zero_coverage_and_inspection_is_owned():
    cortex = Cortex()
    constant = cortex.column("constant", patches=2)
    cortex.column("echo", patches=1, inputs=constant)
    cortex.output("answer", shape=1, reads=constant)
    brain = cortex.build()
    original = brain.inspect()
    assert output_coverage(brain) == (0,)
    assert original["sensor_coverage"] == 0
    assert original["output_connected_patches"] == 3
    changed = brain.inspect()
    changed["outputs"][0]["sensor_coverage_by_coordinate"] = (999,)
    changed["output_connected_patches"] = 999
    assert brain.inspect() == original
    assert Brain.from_snapshot(brain.snapshot()).inspect() == original


def test_nested_observers_include_all_unique_sensor_coordinates():
    cortex = Cortex(seed=2)
    eyes = cortex.input("eyes", shape=3)
    ears = cortex.input("ears", shape=2)
    visual = cortex.column("visual", patches=2, inputs=eyes)
    auditory = cortex.column("auditory", patches=2, inputs=ears)
    observer = cortex.observer("observer", patches=2, observes=(visual, auditory))
    cortex.observer("meta", patches=1, observes=observer, inputs=eyes)
    cortex.output("answer", shape=2, reads=visual)
    brain = cortex.build()
    assert output_coverage(brain) == (5, 5)
    assert brain.inspect()["output_connected_patches"] == 7
