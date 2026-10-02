"""Hybrid admission changes numerical qualification, never the learning boundary."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def recursive(**options):
    return cd.PatchNet.recursive(
        2, [3, 2], 2, seed=9, coupling=0.8,
        config=cd.LearnerConfig(
            nudge="quadratic", beta=0.08, eta=0.02, eta_bias=0.01,
            momentum=0.5, normalize=0.7,
        ),
        tolerance=1e-10, **options,
    )


def snapshot_parameters(net):
    return {
        name: value for name, value in net.snapshot().items()
        if name != "patch" and not name.startswith("patch/")
    }


def equal_arrays(before, after):
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name], err_msg=name)


def small_net(*, weight=0.2, beta=0.1, steps=0, refinement_steps=64, **options):
    graph = cd.Connectome.from_synapses(
        3, pre=[0, 1, 1, 2], post=[1, 0, 2, 1], sign=[0.1, 0.1, weight, weight],
        populations={"input": (0,), "hidden": (1,), "output": (2,)},
    )
    learner = cd.Learner(
        cd.Brain(graph, cd.learning_neuron_model(leak=1)), [2],
        cd.LearnerConfig(nudge="quadratic", beta=beta, momentum=0.5, normalize=0.7),
    )
    return cd.PatchNet(
        learner, solver="hybrid", steps=steps, refinement_steps=refinement_steps,
        tolerance=1e-10, **options,
    )


def equation(brain, drive, state, *, beta=0., target=None, mask=None,
             weight=None, anchor=None, anchor_gain=None):
    """Independently sum directed effective contacts over every neuron."""
    g = brain.connectome
    s = np.tanh(state.v / 2)
    total = np.stack([
        np.bincount(g.post, weights=brain.weights * row[g.pre], minlength=g.n)
        for row in s
    ])
    defect = drive + brain.bias + total - state.v
    if target is not None:
        defect += beta * mask * weight[:, None] * (target - s)
    if anchor is not None:
        defect += anchor_gain * (anchor - s)
    np.testing.assert_allclose(state.activation, s, atol=2e-14, rtol=0)
    return np.max(np.abs(defect), axis=1)


def test_default_local_matches_explicit_local_and_direct_solver_exactly():
    default, explicit = recursive(), recursive(solver="local")
    drive = default.stimulus([[0.7, -0.2], [-0.3, 0.9]])
    direct = default.brain.equilibrate(
        drive, budget=default.steps, chunk=default.chunk, tolerance=default.tolerance,
    )
    for net in (default, explicit):
        actual = net.settle(drive)
        np.testing.assert_array_equal(actual.state.v, direct.state.v)
        np.testing.assert_array_equal(actual.residual, direct.residual)
        np.testing.assert_array_equal(actual.qualified, actual.converged)
        assert actual.steps == direct.steps
    equal_arrays(default.snapshot(), explicit.snapshot())


def test_zero_local_budget_is_refined_on_same_full_equations():
    local, hybrid = recursive(steps=0), recursive(steps=0, solver="hybrid")
    drive = local.stimulus([[0.7, -0.2], [-0.3, 0.9]])
    assert not local.settle(drive).converged.any()
    settled = hybrid.settle(drive)
    assert settled.steps == 0 and settled.converged.all() and settled.qualified.all()
    residual = equation(hybrid.brain, drive, settled.state)
    np.testing.assert_allclose(residual, settled.residual, atol=1e-14, rtol=1e-5)
    assert np.max(residual) <= hybrid.tolerance


def test_weighted_centered_phases_keep_same_temporal_boundary_and_learning_rule():
    hybrid = recursive(steps=0, solver="hybrid", context_strength=0.3)
    reference = recursive(steps=4096, chunk=8, solver="local", context_strength=0.3)
    cue = hybrid.stimulus([[0.8, -0.3], [0.2, 0.6]])
    for net in (hybrid, reference):
        assert net.settle(cue).qualified.all()
    prior = hybrid.state.activation.copy()
    brain = hybrid.brain
    drive = cue * 0.4
    target = np.array([[0.25, np.nan], [-0.2, np.nan]])
    gain = np.array([0.3, 1.7])
    known = np.array([True, False])
    actual = hybrid.observe(drive, target, observed=known, weight=gain, source_id="event")
    expected = reference.observe(drive, target, observed=known, weight=gain, source_id="event")
    assert actual.updated and expected.updated
    target_all = np.zeros_like(drive)
    target_all[:, hybrid.output_index[0]] = target[:, 0]
    mask = np.zeros(brain.connectome.n)
    mask[hybrid.output_index[0]] = 1
    anchor_gain = hybrid.context_strength * hybrid.context_mask
    for phase, beta in ((actual.free, 0.), (actual.plus, 0.08), (actual.minus, -0.08)):
        assert phase.qualified.all()
        error = equation(
            brain, drive, phase.state, beta=beta, target=target_all, mask=mask,
            weight=gain, anchor=prior, anchor_gain=anchor_gain,
        )
        assert error.max() <= hybrid.tolerance
    np.testing.assert_allclose(hybrid.brain.efficacy, reference.brain.efficacy, atol=2e-8)
    np.testing.assert_allclose(hybrid.brain.bias, reference.brain.bias, atol=2e-8)
    np.testing.assert_array_equal(hybrid.state.v, actual.free.state.v)


def test_residual_zero_saddle_rejected_without_optimizer_or_source_commit():
    net = small_net(weight=3)
    before = snapshot_parameters(net)
    result = net.observe(np.zeros((1, 3)), [[0.25]], source_id="retry")
    assert result.reason == "free_unqualified" and not result.updated
    assert result.free.converged.all() and not result.free.qualified.any()
    equal_arrays(before, snapshot_parameters(net))
    assert json.loads(str(net.snapshot()["patch"]))["sources"] == []
    # Repair the actual reciprocal weights; the same external ID remains usable.
    net.learner.brain = net.brain.with_parameters(efficacy=np.full(4, 0.1))
    retried = net.observe(np.zeros((1, 3)), [[0.25]], source_id="retry")
    assert retried.updated and net.learner.updates == 1


def test_negative_nudge_curvature_failure_cannot_update_or_consume_id():
    net = small_net(weight=0, beta=3)
    before = snapshot_parameters(net)
    result = net.observe(np.zeros((1, 3)), [[0.]], source_id="retry")
    assert result.reason == "nudge_unqualified" and not result.updated
    assert all(p.converged.all() for p in (result.free, result.plus, result.minus))
    assert result.free.qualified.all() and result.plus.qualified.all()
    assert not result.minus.qualified.any()
    equal_arrays(before, snapshot_parameters(net))
    assert json.loads(str(net.snapshot()["patch"]))["sources"] == []
    net.learner.config = replace(net.learner.config, beta=0.1)
    assert net.observe(np.zeros((1, 3)), [[0.]], source_id="retry").updated


def test_zero_refinement_budget_retains_existing_unconverged_reasons():
    net = small_net(refinement_steps=0)
    before = snapshot_parameters(net)
    free = net.observe(net.stimulus([[0.7]]), [[0.2]], source_id="retry")
    assert free.reason == "free_unconverged" and not free.updated
    net.reset()
    nudge = net.observe(np.zeros((1, 3)), [[0.2]], source_id="retry")
    assert nudge.reason == "nudge_unconverged" and not nudge.updated
    equal_arrays(before, snapshot_parameters(net))
    assert json.loads(str(net.snapshot()["patch"]))["sources"] == []


def test_v3_checkpoint_preserves_solver_and_exact_continuation(tmp_path):
    net = recursive(solver="hybrid", steps=0, refinement_steps=19, context_strength=0.2)
    drive = net.stimulus([[0.3, -0.4]])
    assert net.observe(drive, [[0.1, -0.2]], source_id="first").updated
    metadata = json.loads(str(net.snapshot()["patch"]))
    assert metadata["format"] == "cadence-patch/3"
    restored = cd.PatchNet.load(net.save(tmp_path / "hybrid"))
    assert restored.solver == "hybrid" and restored.refinement_steps == 19
    equal_arrays(net.snapshot(), restored.snapshot())
    for owner in (net, restored):
        assert owner.observe(drive, [[0.1, -0.2]], source_id="first").reason == "duplicate"
        assert owner.observe(drive * 0.5, [[0.3, 0.2]], source_id="next").updated
    equal_arrays(net.snapshot(), restored.snapshot())


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_checkpoint_loads_local_without_interpreting_new_options(tmp_path, version):
    net = recursive()
    data = net.snapshot()
    metadata = json.loads(str(data["patch"]))
    metadata["format"] = f"cadence-patch/{version}"
    metadata.pop("solver")
    metadata.pop("refinement_steps")
    if version == 1:
        metadata.pop("context_strength")
        metadata.pop("context_mask")
    data["patch"] = np.array(json.dumps(metadata))
    np.savez(tmp_path / "legacy.npz", **data)
    restored = cd.PatchNet.load(tmp_path / "legacy.npz")
    assert restored.solver == "local" and restored.refinement_steps == 64
    drive = net.stimulus([[0.2, 0.3]])
    np.testing.assert_array_equal(restored.settle(drive).state.v, net.settle(drive).state.v)


@pytest.mark.parametrize("field,value", [
    ("solver", None), ("refinement_steps", None), ("solver", "mystery"),
    ("refinement_steps", -1), ("refinement_steps", True),
])
def test_v3_missing_or_invalid_solver_configuration_rejected(tmp_path, field, value):
    data = recursive().snapshot()
    metadata = json.loads(str(data["patch"]))
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    data["patch"] = np.array(json.dumps(metadata))
    np.savez(tmp_path / "bad.npz", **data)
    with pytest.raises(ValueError):
        cd.PatchNet.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("options", [
    {"solver": "unknown"}, {"refinement_steps": -1}, {"refinement_steps": True},
    {"refinement_steps": 1.5},
])
def test_invalid_runtime_configuration_rejected(options):
    with pytest.raises(ValueError):
        recursive(**options)


@pytest.mark.parametrize("changes", [
    {"slope": 2.}, {"threshold": 0.2}, {"leak": 0.1},
    {"adaptation": cd.Adaptation()},
])
def test_unsupported_neuron_rules_are_not_silently_changed(changes):
    net = recursive()
    brain = net.brain
    net.learner.brain = cd.Brain(
        brain.connectome, brain.neuron_model.replace(**changes), efficacy=brain.efficacy,
    )
    with pytest.raises(ValueError):
        cd.PatchNet(net.learner, solver="hybrid")


def test_overlapping_output_or_anchored_input_and_recurrent_input_block_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        small_net(input_index=[0, 2])
    with pytest.raises(ValueError, match="input"):
        small_net(context_strength=0.1, context_mask=np.array([True, False, False]))
    with pytest.raises(ValueError):
        small_net(input_index=[0, 1])
    # No input anchor exists when context strength is zero.
    assert small_net(context_mask=np.ones(3, dtype=bool)).solver == "hybrid"


def test_effective_reciprocity_rechecked_after_parameter_replacement():
    net = recursive(solver="hybrid")
    drive = net.stimulus([[0.2, 0.3]])
    before = net.state
    values = net.brain.efficacy.copy()
    values[0] += 0.1
    net.learner.brain = net.brain.with_parameters(efficacy=values)
    with pytest.raises(ValueError, match="reciprocal"):
        net.settle(drive)
    assert net.state is before is None


def test_hybrid_load_honors_requested_backend_without_cpu_fallback(tmp_path, monkeypatch):
    net = recursive(solver="hybrid")
    path = net.save(tmp_path / "hybrid")
    called = []

    def alternate(cls, path, *, backend, device, precision):
        called.append((backend, device, precision))
        replacement = recursive().learner
        replacement.brain.backend = backend
        return replacement

    monkeypatch.setattr(cd.Learner, "load", classmethod(alternate))
    with pytest.raises(ValueError):
        cd.PatchNet.load(path, backend="torch", device="cpu", precision="float64")
    assert called == [("torch", "cpu", "float64")]


def test_empty_elimination_set_retains_every_neuron_in_energy_solve():
    net = small_net(input_index=[])
    drive = np.array([[0.3, -0.2, 0.1]])
    result = net.settle(drive)
    assert result.qualified.all()
    assert equation(net.brain, drive, result.state).max() <= net.tolerance


def test_effective_gain_enters_refinement_and_full_equation_audit():
    net = recursive(solver="hybrid", steps=0)
    brain = net.brain
    net.learner.brain = cd.Brain(
        brain.connectome, brain.neuron_model.replace(gain=0.4),
        efficacy=brain.efficacy, log_gain=np.full(brain.connectome.n, np.log(1.5)),
    )
    drive = net.stimulus([[0.3, -0.2]])
    phase = net.settle(drive)
    assert phase.qualified.all()
    assert equation(net.brain, drive, phase.state).max() <= net.tolerance
