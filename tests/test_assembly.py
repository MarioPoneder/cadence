import numpy as np
import pytest

import cadence as cd
from cadence.circuits import assemble, reflex_arc


def test_regions_exchange_feedback_in_one_equilibrium():
    eye = cd.Connectome.from_synapses(1, pre=[], post=[])
    regions = {"vision": eye, "movement": reflex_arc(1)}
    synapses = [("vision", 0, "movement", 0, 0.5), ("movement", 1, "vision", 0, -0.2)]
    drive = np.array([0.6, 0.0, 0.0, 0.0])
    neuron_model = cd.NeuronModel(
        gain=1, slope=2, threshold=0, leak=1, dt=0.25, stimulus_amplitude=1
    )
    results = []
    for links in [synapses, synapses[:1], []]:
        connectome = assemble(regions, links)
        assert connectome.populations["movement"] == (1, 2, 3)
        assert connectome.populations["movement/motor"] == (2, 3)
        brain = cd.Brain(connectome, neuron_model)
        state = brain.settle(drive, steps=400, tolerance=0)
        assert brain.residual(drive, state)[0] < 1e-10
        results.append(state.activation)
    joint, forward, disconnected = results
    assert 0 < joint[2] < forward[2]
    assert joint[0] < forward[0]  # Motor readback changes the sensory equilibrium.
    np.testing.assert_array_equal(disconnected[1:], [0, 0, 0])
    # Equation residual is independent of a small integration step.
    v = np.arctanh(joint)
    assert abs(v[0] - (0.6 - 0.2 * joint[2])) < 1e-10
    assert abs(v[1] - 0.5 * joint[0]) < 1e-10


def test_assemble_preserves_contact_counts_signs_and_named_ports():
    region = cd.Connectome.from_synapses(
        2, pre=[0], post=[1], count=[3], sign=[-0.4], populations={"out": [1]}
    )
    connectome = assemble({"left": region, "right": region}, [("left", 1, "right", 0, 0.2)])
    edges = dict(
        zip(
            zip(connectome.pre, connectome.post, strict=True),
            connectome.count * connectome.sign,
            strict=True,
        )
    )
    assert edges == {(0, 1): -1.2000000000000002, (2, 3): -1.2000000000000002, (1, 2): 0.2}
    assert connectome.populations["right/out"] == (3,)
    assert region.synapses == 1 and region.populations["out"] == (1,)


@pytest.mark.parametrize(
    "synapse",
    [
        ("absent", 0, "a", 1, 1),
        ("a", -1, "a", 1, 1),
        ("a", 2, "a", 0, 1),
        ("a", True, "a", 0, 1),
        ("a", 0, "a", 0, 1),
        ("a", 0, "a", 1, float("nan")),
    ],
)
def test_assemble_rejects_invalid_synapses(synapse):
    region = cd.Connectome.from_synapses(2, pre=[], post=[])
    with pytest.raises(ValueError):
        assemble({"a": region}, [synapse])


def test_assemble_requires_unambiguous_region_names():
    with pytest.raises(ValueError):
        assemble({})
    with pytest.raises(ValueError):
        assemble({"a/b": cd.Connectome.from_synapses(1, pre=[], post=[])})
