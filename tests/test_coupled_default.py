"""A brain settles between populations: the builder refuses uncoupled layouts."""

import pytest

from cadence import Cortex


def test_a_lone_population_reading_only_sensors_is_refused():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    response = layout.column("response", patches=1, inputs=signal)
    layout.output("answer", shape=1, reads=response)
    with pytest.raises(ValueError, match="settles with no other population"):
        layout.build()


def test_an_isolated_population_beside_a_coupled_pair_is_refused():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=4, inputs=signal)
    response = layout.column("response", patches=1, inputs=features)
    layout.column("aside", patches=1, inputs=signal)
    layout.output("answer", shape=1, reads=response)
    with pytest.raises(ValueError, match="'aside' settles with no other population"):
        layout.build()


def test_a_population_read_by_another_is_coupled_even_if_it_reads_only_sensors():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=4, inputs=signal)
    layout.column("response", patches=1, inputs=(signal, features))
    layout.output("answer", shape=1, reads=features)
    brain = layout.build()
    graph = brain.graph
    kinds = {kind for kind, _, _ in graph.edges}
    assert kinds == {"input", "state"}
    assert brain.settle({"signal": [0.4]})["qualified"]


def test_observation_also_couples():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=2, inputs=signal)
    monitor = layout.observer("monitor", patches=1, observes=features)
    layout.output("answer", shape=1, reads=monitor)
    brain = layout.build()
    assert {kind for kind, _, _ in brain.graph.edges} == {"input", "state", "residual"}


def test_refusal_leaves_the_layout_unbuilt_and_usable():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    response = layout.column("response", patches=1, inputs=signal)
    layout.output("answer", shape=1, reads=response)
    with pytest.raises(ValueError):
        layout.build()
    reader = layout.column("reader", patches=1, inputs=response)
    layout.output("echo", shape=1, reads=reader)
    assert layout.build().inspect()["patches"] == 2


def test_two_clusters_that_settle_apart_are_refused():
    layout = Cortex(seed=2)
    signal = layout.input("signal", shape=1)
    a = layout.column("a", patches=2, inputs=signal)
    b = layout.column("b", patches=1, inputs=a)
    c = layout.column("c", patches=2, inputs=signal)
    d = layout.column("d", patches=1, inputs=c)
    layout.output("answer", shape=1, reads=b)
    layout.output("other", shape=1, reads=d)
    with pytest.raises(ValueError, match=r"Populations \['c', 'd'\] settle apart"):
        layout.build()
    layout.column("bridge", patches=1, inputs=(b, d))
    assert layout.build().inspect()["patches"] == 7
