"""Recursive populations must repair together, including the answer neurons."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def make(**options):
    return cd.PatchNet.recursive(2, [3, 2, 2], 1, seed=5, tolerance=1e-10, chunk=8, **options)


def defect(patch, phase, drive, weights, bias, *, strength=0.0, target=None):
    """Independent edge sum over every equation, without NeuralGraph's residual code."""
    graph = patch.brain.connectome
    v = phase.state.v
    activity = np.tanh(v / 2)
    residual = drive + bias - v
    for pre, post, weight in zip(graph.pre, graph.post, weights, strict=True):
        residual[:, post] += weight * activity[:, pre]
    if target is not None:
        residual[:, patch.output_index] += strength * (target - activity[:, patch.output_index])
    return residual


def qualify(patch, phase, drive, weights=None, bias=None, **options):
    weights = patch.brain.weights if weights is None else weights
    bias = patch.brain.bias if bias is None else bias
    assert np.all(phase.converged)
    np.testing.assert_allclose(phase.state.activation, np.tanh(phase.state.v / 2), atol=1e-14)
    assert np.max(np.abs(defect(patch, phase, drive, weights, bias, **options))) <= patch.tolerance
    np.testing.assert_array_equal(patch.read(phase), phase.state.activation[:, patch.output_index])
    return phase.state.activation


def fresh_solve(patch, drive):
    patch.reset()
    return qualify(patch, patch.settle(drive), drive)


def assert_snapshots_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def learned_state(patch):
    return {key: value for key, value in patch.snapshot().items()
            if key != "patch" and not key.startswith("patch/")}


@pytest.mark.parametrize("layers,contacts", [([3], 24), ([3, 2], 52), ([3, 2, 4], 124)])
def test_topology_has_reciprocal_observation_of_every_earlier_neuron(layers, contacts):
    patch = cd.PatchNet.recursive(2, layers, 2, seed=17, coupling=1.4)
    graph = patch.brain.connectome
    assert type(patch) is cd.PatchNet
    assert graph.n == 4 + sum(layers)
    assert graph.synapses == contacts
    assert cd.ep_structure(patch.brain).compatible
    edges = set(zip(graph.pre.tolist(), graph.post.tolist(), strict=True))
    assert len(edges) == contacts
    assert all((b, a) in edges and a != b for a, b in edges)
    np.testing.assert_array_equal(patch.brain.weights, patch.brain.weights[patch.learner.reverse])
    row_mass = np.bincount(graph.post, weights=np.abs(patch.brain.weights), minlength=graph.n)
    assert np.max(row_mass) == pytest.approx(1.4)
    assert patch.brain.neuron_model.leak == 1.0
    assert patch.learner.config.nudge == "quadratic"
    previous = graph.members("input", "layer_0", "output")
    for level in range(1, len(layers)):
        current = graph.populations[f"layer_{level}"]
        assert len(current) == layers[level]
        assert all((i, j) in edges for i in previous for j in current)
        assert not any((i, j) in edges for i in current for j in current)
        previous += current
    assert graph.members("hidden") == graph.members(*(f"layer_{i}" for i in range(len(layers))))
    assert graph.members("observer") == graph.members(*(f"layer_{i}" for i in range(1, len(layers))))
    assert not any((i, j) in edges for i in graph.populations["input"] for j in graph.populations["output"])


def test_seed_and_global_coupling_scale_are_reproducible():
    one = make()
    two = make()
    assert_snapshots_equal(one.snapshot(), two.snapshot())
    scaled = make(coupling=0.25)
    np.testing.assert_array_equal(one.brain.connectome.pre, scaled.brain.connectome.pre)
    np.testing.assert_allclose(scaled.brain.weights, one.brain.weights * 0.25, rtol=0, atol=0)
    different = cd.PatchNet.recursive(2, [3, 2, 2], 1, seed=6)
    assert not np.array_equal(one.brain.weights, different.brain.weights)
    numpy_widths = cd.PatchNet.recursive(np.int64(2), np.array([3, 2]), np.int64(1))
    assert numpy_widths.brain.connectome.n == 8


def test_deepest_observer_and_base_change_each_other_during_one_solve_and_cut_stops_it():
    patch = make()
    graph = patch.brain.connectome
    base = graph.members("input", "layer_0", "output")
    deepest = graph.populations["layer_2"]
    drive = patch.stimulus([[0.4, -0.2], [-0.3, 0.6]])
    free = fresh_solve(patch, drive)
    observer_drive = drive.copy()
    observer_drive[:, deepest[0]] += 0.6
    feedback = fresh_solve(patch, observer_drive)
    assert np.max(np.abs(feedback[:, base] - free[:, base])) > 1e-3
    assert np.max(np.abs(feedback[:, patch.output_index] - free[:, patch.output_index])) > 1e-5
    base_drive = drive.copy()
    base_drive[:, patch.input_index[0]] += 0.6
    readback = fresh_solve(patch, base_drive)
    assert np.max(np.abs(readback[:, deepest] - free[:, deepest])) > 1e-3

    # Remove both directions while retaining every neuron and all other contacts.
    cut = make()
    weights = cut.brain.efficacy.copy()
    boundary = np.isin(graph.pre, deepest) ^ np.isin(graph.post, deepest)
    weights[boundary] = 0
    cut.brain.efficacy = weights
    cut_free = fresh_solve(cut, drive)
    cut_feedback = fresh_solve(cut, observer_drive)
    cut_readback = fresh_solve(cut, base_drive)
    np.testing.assert_allclose(cut_feedback[:, base], cut_free[:, base], atol=1e-10, rtol=0)
    np.testing.assert_allclose(cut_readback[:, deepest], cut_free[:, deepest], atol=1e-10, rtol=0)


def test_free_and_teaching_phases_qualify_the_same_graph_including_output_and_observers():
    config = cd.LearnerConfig(nudge="quadratic", beta=1e-4, eta=0.01, eta_bias=0.01)
    patch = make(config=config)
    graph = patch.brain.connectome
    weights, bias = patch.brain.weights.copy(), patch.brain.bias.copy()
    drive = patch.stimulus([[0.4, -0.2]])
    target = np.array([[0.35]])
    report = patch.observe(drive, target, source_id="recursive:1")
    assert report.updated
    assert patch.brain.connectome is graph
    for phase, strength in ((report.free, 0.0), (report.plus, config.beta), (report.minus, -config.beta)):
        assert phase.state.v.shape == (1, graph.n)
        qualify(patch, phase, drive, weights, bias, strength=strength, target=target)
    np.testing.assert_array_equal(patch.state.v, report.free.state.v)
    assert np.max(np.abs(report.plus.state.v[:, graph.populations["observer"]] - report.free.state.v[:, graph.populations["observer"]])) > 1e-7
    # A frozen/corrupted deepest observer cannot hide behind correct answer ports.
    report.free.state.v[:, graph.populations["layer_2"]] += 0.1
    assert np.max(np.abs(defect(patch, report.free, drive, weights, bias))) > 0.05


def test_checkpoint_preserves_recursive_topology_phases_and_learning(tmp_path):
    patch = make(config=cd.LearnerConfig(nudge="quadratic", momentum=0.6, normalize=0.8))
    drive = patch.stimulus([[0.4, -0.2]])
    assert patch.observe(drive, [[0.35]], source_id="before").updated
    restored = cd.PatchNet.load(patch.save(tmp_path / "recursive"))
    assert_snapshots_equal(patch.snapshot(), restored.snapshot())
    assert restored.brain.connectome.populations == patch.brain.connectome.populations
    for net in (patch, restored):
        assert net.observe(drive, [[0.35]], source_id="before").reason == "duplicate"
        assert net.observe(drive * 0.8, [[-0.2]], source_id="next").updated
    assert_snapshots_equal(patch.snapshot(), restored.snapshot())


def test_failed_joint_phase_cannot_update_or_consume_evidence():
    patch = make(steps=0)
    before = learned_state(patch)
    drive = patch.stimulus([[0.4, -0.2]])
    report = patch.observe(drive, [[0.35]], source_id="retry")
    assert not report.updated and report.reason == "free_unconverged"
    assert_snapshots_equal(before, learned_state(patch))
    patch.reset()
    report = patch.observe(np.zeros_like(drive), [[0.35]], source_id="retry")
    assert not report.updated and report.reason == "nudge_unconverged"
    assert_snapshots_equal(before, learned_state(patch))
    assert json.loads(str(patch.snapshot()["patch"]))["sources"] == []


@pytest.mark.parametrize("bad", [0, -1, 1.0, True, np.bool_(False), "2", None])
@pytest.mark.parametrize("argument", ["inputs", "outputs", "layer"])
def test_invalid_widths(argument, bad):
    kwargs = {"inputs": 2, "layers": [3, 2], "outputs": 1}
    if argument == "layer":
        kwargs["layers"] = [3, bad]
    else:
        kwargs[argument] = bad
    with pytest.raises(ValueError, match="positive integer"):
        cd.PatchNet.recursive(**kwargs)


@pytest.mark.parametrize("bad", [[], (), None, 3, True, "32", {3: 2}, np.zeros((2, 2), dtype=int)])
def test_invalid_layer_sequences(bad):
    with pytest.raises(ValueError, match="nonempty sequence"):
        cd.PatchNet.recursive(2, bad, 1)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(True), "1", None, [1]])
def test_invalid_coupling(bad):
    with pytest.raises(ValueError, match="coupling"):
        make(coupling=bad)


@pytest.mark.parametrize("bad", [-1, 1.0, True, np.bool_(True), None])
def test_invalid_seed(bad):
    with pytest.raises(ValueError, match="seed"):
        cd.PatchNet.recursive(2, [3, 2], 1, seed=bad)


@pytest.mark.parametrize("bad", [False, cd.LearnerConfig(nudge="cross_entropy")])
def test_invalid_learner_config(bad):
    with pytest.raises(ValueError, match="quadratic"):
        make(config=bad)
