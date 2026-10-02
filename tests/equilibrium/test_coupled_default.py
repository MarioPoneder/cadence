"""A brain settles between populations: the builder refuses uncoupled layouts."""

import pytest

from cadence.experimental.equilibrium import Cortex


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


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("depth", [2, 3])
def test_sparse_population_chain_cannot_hide_separate_patch_components(seed, depth):
    layout = Cortex(seed=seed, fan_in=1)
    signal = layout.input("signal", shape=1)
    previous = layout.column("features", patches=2, inputs=signal)
    for _ in range(depth - 1):
        previous = layout.column(patches=2, inputs=previous)
    layout.output("answer", shape=1, reads=previous)
    # Each population reads the previous one, but one-to-one contacts form
    # two independent chains. Their common fixed sensor cannot join them.
    with pytest.raises(ValueError, match="2 disconnected settlement components"):
        layout.build()


def test_recursive_error_contacts_must_also_connect_the_actual_patches():
    layout = Cortex(seed=0, fan_in=1)
    signal = layout.input("signal", shape=1)
    features = layout.column("features", patches=2, inputs=signal)
    monitor = layout.observer("monitor", patches=2, observes=features)
    layout.output("answer", shape=1, reads=monitor)
    # For this seed the state and residual matchings coincide: exact error
    # readback alone does not join the two pairs.
    with pytest.raises(ValueError, match="2 disconnected settlement components"):
        layout.build()


def test_patch_connectivity_refusal_is_retryable_with_an_explicit_bridge():
    layout = Cortex(seed=0, fan_in=1)
    signal = layout.input("signal", shape=2)
    features = layout.column("features", patches=2, inputs=signal)
    response = layout.column("response", patches=2, inputs=features)
    layout.output("answer", shape=1, reads=response)
    with pytest.raises(ValueError, match="disconnected settlement components"):
        layout.build()
    # Source coverage raises the single bridge patch's fan-in to two. The
    # explicit new contacts join the pairs; no hidden edges were inserted.
    layout.column("bridge", patches=1, inputs=response)
    brain = layout.build()
    assert brain.inspect()["output_connected_patches"] == 5
    assert len(brain.graph.edges) == 6
    assert brain.settle({"signal": [0.2, -0.3]})["qualified"]


@pytest.mark.parametrize("fan_in", [None, 2])
def test_dense_state_contacts_join_all_compiled_patches(fan_in):
    layout = Cortex(fan_in=fan_in)
    signal = layout.input("signal", shape=2)
    features = layout.column("features", patches=2, inputs=signal)
    response = layout.column("response", patches=2, inputs=features)
    layout.output("answer", shape=1, reads=response)
    brain = layout.build()
    assert brain.inspect()["output_connected_patches"] == 4


def test_sparse_modules_can_share_one_equilibrium_through_a_narrow_bridge():
    layout = Cortex(seed=5, fan_in=1)
    sensor = layout.input("sensor", shape=1)
    modules = []
    for name in ("left", "right"):
        local = layout.column(name, patches=4, inputs=sensor)
        modules.append(layout.column(f"{name}_readout", patches=4, inputs=local))
    bridge = layout.column("bridge", patches=1, inputs=tuple(modules))
    layout.output("answer", shape=1, reads=bridge)
    brain = layout.build()
    state_contacts = [edge for edge in brain.graph.edges if edge[0] == "state"]
    # Seventeen patches and only sixteen inter-patch contacts form a tree.
    # Neither all-to-all wiring nor a contact between every pair is required.
    assert brain.graph.n_patches == 17
    assert len(state_contacts) == 16
    assert brain.inspect()["output_connected_patches"] == 17
    assert brain.step({"sensor": [0.4]})["qualified"]
