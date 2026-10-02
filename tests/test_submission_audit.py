"""Adversarial regressions for the submission's numerical and store boundaries."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.temporal import contrast_asymmetry


def test_parameter_replacement_cannot_overflow_effective_weights():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1e100])
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model())
    before = brain.weights.copy()
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.with_parameters(efficacy=np.array([1e300]))
    with pytest.raises(ValueError, match="effective synaptic weights"):
        brain.efficacy = np.array([1e300])
    np.testing.assert_array_equal(brain.weights, before)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("invalid", ["target", "code", "overflow"])
def test_failed_record_write_is_atomic_across_fields(batched, invalid):
    records = cd.Records(1, {"first": 1, "second": 1}, cells=2, active=1, averaging=True)
    codes = np.array([[1.0, 0.0], [1.0, 0.0]])
    targets = {"first": np.array([1.0]), "second": np.array([2.0])}
    if invalid == "target":
        targets["second"][0] = np.nan
    elif invalid == "code":
        records.valued = frozenset({"second"})
        codes[1, 0] = np.nan
    else:
        records.valued = frozenset({"second"})
        codes[1, 0] = 1e200
        targets["second"][0] = 1e200
    before = records.state()
    with pytest.raises(ValueError, match="finite"):
        if batched:
            records.write_batch(codes[:, None], {key: value[None] for key, value in targets.items()})
        else:
            records.write(codes, targets)
    for key, value in before.items():
        np.testing.assert_array_equal(records.state()[key], value)


def test_record_write_validates_known_mask_without_broadcasting():
    records = cd.Records(1, {"first": 1, "second": 2}, cells=2, active=1, averaging=True)
    code = np.array([[1.0, 0.0], [1.0, 0.0]])
    before = records.state()
    with pytest.raises(ValueError, match="known mask"):
        records.write(code, {"first": np.ones(1), "second": np.ones(2)}, {"second": [True]})
    for key, value in before.items():
        np.testing.assert_array_equal(records.state()[key], value)


@pytest.mark.parametrize("batched", [False, True])
def test_unset_valued_code_can_write_only_consequence_fields(batched):
    records = cd.Records(1, {"y": 1, "value": 1}, cells=2, active=1, valued=("value",), rate=1)
    code = np.array([[1.0, 0.0], [np.nan, np.nan]])
    if batched:
        records.write_batch(code[:, None], {"y": np.ones((1, 1))})
    else:
        records.write(code, {"y": np.ones(1)})
    np.testing.assert_array_equal(records.tables["y"], [[1], [0]])
    np.testing.assert_array_equal(records.tables["value"], [[0], [0]])


def test_large_finite_protected_input_does_not_overflow_rank_threshold():
    net = cd.TemporalPatchNet(1, 1, 1)
    net.set_parameters({"A": np.zeros((1, 1)), "B": np.ones((1, 1)), "C": np.ones((1, 1))})
    inputs = np.array([[[1e200]]])
    original = net.imagine(inputs).output
    memory = cd.TemporalMemory()
    report = memory.protect(net, inputs)
    assert report.ranks["B"] == 1 and report.maximum_residual == 0
    before = net.parameters()
    proposed = {key: value.copy() for key, value in before.items()}
    proposed["B"] *= -1
    net.set_parameters(memory.project(before, proposed))
    np.testing.assert_array_equal(net.imagine(inputs).output, original)


def test_coincident_detuned_paths_cannot_hide_a_displaced_midpoint():
    net = cd.TemporalPatchNet(1, 1, 1)
    free = net.imagine(np.zeros((1, 1, 1)))
    displaced = replace(free, hidden=free.hidden + 1)
    assert contrast_asymmetry(free, free, free) == 0
    assert np.isinf(contrast_asymmetry(free, displaced, displaced))
