"""The continuous runtime must learn only from observed, converged external phases."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def net(**kwargs):
    return cd.PatchNet.create(2, 5, 2, seed=3, density=1, tolerance=1e-7, **kwargs)


def equal_snapshots(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def parameters(p):
    return {
        key: value
        for key, value in p.snapshot().items()
        if key not in ("patch",) and not key.startswith("patch/")
    }


def test_factory_reciprocity_and_real_continuous_learning_survive_reset():
    p = cd.PatchNet.create(2, 5, 1, seed=3, density=1, tolerance=1e-7)
    assert cd.ep_structure(p.brain).compatible
    x = p.stimulus(np.array([[0.0, 1.0], [1.0, 0.0]]), amplitude=2)
    target = np.array([[0.2], [0.8]])
    before = np.mean((p.read(p.settle(x)) - target) ** 2)
    for event in range(60):
        report = p.observe(x, target, source_id=str(event))
        assert report.updated and report.reason == "updated"
        assert all(np.all(phase.converged) for phase in (report.free, report.plus, report.minus))
    weights = parameters(p)
    p.reset()
    assert p.state is None
    equal_snapshots(weights, parameters(p))
    after = np.mean((p.read(p.settle(x)) - target) ** 2)
    assert after < before / 5
    assert cd.ep_structure(p.brain).compatible


def test_observation_matches_existing_quadratic_local_rule_and_carries_free_only():
    p, reference = net(), net()
    x = p.stimulus([[0.1, 0.9], [0.8, 0.2]], amplitude=2)
    y = np.array([[0.3, 0.8], [0.6, 0.1]])
    report = p.observe(x, y, weight=np.array([1.0, 0.5]))
    free = reference.brain.equilibrate(
        x, budget=reference.steps, chunk=reference.chunk, tolerance=reference.tolerance
    )
    target = np.zeros_like(x)
    target[:, reference.output_index] = y
    mask = reference.learner.output_mask
    phases = [
        reference.brain.equilibrate(
            x,
            state=free.state,
            budget=reference.steps,
            chunk=reference.chunk,
            tolerance=reference.tolerance,
            nudge=cd.Nudge(
                target, mask, sign * reference.learner.config.beta, weight=np.array([1.0, 0.5])
            ),
        )
        for sign in (1, -1)
    ]
    reference.learner.update(free.state, phases[0].state, phases[1].state)
    assert report.updated
    equal_snapshots(parameters(p), parameters(reference))
    np.testing.assert_array_equal(p.state.v, report.free.state.v)
    assert not np.array_equal(p.state.v, report.plus.state.v)
    report.free.state.v[:] = 100  # readbacks cannot mutate persistent activity
    assert not (p.state.v == 100).any()


def test_unknown_outputs_cannot_teach_or_change_evidence_identity():
    a, b = net(), net()
    x = a.stimulus([[0.3, 0.7]])
    known = np.array([True, False])
    one = a.observe(x, [[0.8, np.nan]], observed=known, source_id="sensor:1")
    two = b.observe(x, [[0.8, -1000]], observed=known, source_id="sensor:1")
    assert one.updated and two.updated
    equal_snapshots(a.snapshot(), b.snapshot())
    saved = a.snapshot()
    assert a.observe(x, [[0.8, 1000]], observed=known, source_id="sensor:1").reason == "duplicate"
    equal_snapshots(saved, a.snapshot())
    with pytest.raises(ValueError, match="different evidence"):
        a.observe(x, [[0.1, 1000]], observed=known, source_id="sensor:1")
    equal_snapshots(saved, a.snapshot())


def test_source_window_is_bounded_and_reset_does_not_enable_double_learning():
    p = net(source_capacity=2)
    x, y = p.stimulus([[0.3, 0.7]]), [[0.2, 0.7]]
    assert p.observe(x, y, source_id="old").updated
    p.reset()
    assert p.observe(x, y, source_id="old").reason == "duplicate"
    assert p.state is None
    for key in ("new", "newer", "old"):
        assert p.observe(x, y, source_id=key).updated
    assert p.learner.updates == 4  # the oldest ID expired; no unbounded provenance claim
    assert len(json.loads(str(p.snapshot()["patch"]))["sources"]) == 2


def test_no_observed_or_zero_weight_targets_do_not_advance_optimizer_or_ledger():
    p = net()
    before = parameters(p)
    x = p.stimulus([[0.2, 0.7]])
    no_ports = p.observe(x, [[np.nan, np.nan]], observed=np.zeros(2, bool), source_id="unused")
    no_weight = p.observe(x, [[0.2, 0.7]], weight=np.zeros(1), source_id="unused")
    assert no_ports.reason == no_weight.reason == "no_observations"
    equal_snapshots(before, parameters(p))
    assert json.loads(str(p.snapshot()["patch"]))["sources"] == []
    assert p.state is not None


def test_capped_phases_cannot_commit_or_consume_source_ids():
    p = net(steps=0)
    before = parameters(p)
    x = p.stimulus([[0.3, 0.7]])
    report = p.observe(x, [[0.8, 0.2]], source_id="retry")
    assert report.reason == "free_unconverged" and not report.updated
    equal_snapshots(before, parameters(p))
    # At zero drive rest is an exact free fixed point; the target phase is not.
    p.reset()
    report = p.observe(np.zeros_like(x), [[0.8, 0.2]], source_id="retry")
    assert report.reason == "nudge_unconverged" and not report.updated
    equal_snapshots(before, parameters(p))
    assert json.loads(str(p.snapshot()["patch"]))["sources"] == []


def test_imagination_is_a_sequential_isolated_branch():
    p = net()
    x = p.stimulus([[0.2, 0.7]])
    p.observe(x, [[0.2, 0.8]], source_id="actual")
    before = p.snapshot()
    branch = p.imagine([x * 0.5, x * 1.5])
    reference = p.brain.equilibrate(
        x * 0.5, state=p.state, budget=p.steps, chunk=p.chunk, tolerance=p.tolerance
    )
    second = p.brain.equilibrate(
        x * 1.5, state=reference.state, budget=p.steps, chunk=p.chunk, tolerance=p.tolerance
    )
    np.testing.assert_array_equal(branch[-1].state.v, second.state.v)
    p.imagine([x], state=branch[-1].state)
    equal_snapshots(before, p.snapshot())
    branch[-1].state.v[:] = -100
    equal_snapshots(before, p.snapshot())
    with pytest.raises(ValueError):
        p.imagine([x, np.full_like(x, np.nan)])
    equal_snapshots(before, p.snapshot())


def test_checkpoint_resumes_fast_state_optimizer_and_dedup_exactly(tmp_path):
    config = cd.LearnerConfig(nudge="quadratic", momentum=0.6, normalize=0.8)
    p = net(config=config)
    x = p.stimulus([[0.3, 0.7], [0.8, 0.1]])
    y = np.array([[0.8, 0.2], [0.1, 0.6]])
    assert p.observe(x, y, source_id="before").updated
    resumed = cd.PatchNet.load(p.save(tmp_path / "living"))
    equal_snapshots(p.snapshot(), resumed.snapshot())
    for owner in (p, resumed):
        assert owner.observe(x, y, source_id="before").reason == "duplicate"
        assert owner.observe(x * 0.8, y[:, ::-1], source_id="after").updated
    equal_snapshots(p.snapshot(), resumed.snapshot())


def test_corrupt_continuation_and_directed_or_adapting_models_rejected(tmp_path):
    p = net()
    p.settle(p.stimulus([[0.3, 0.7]]))
    data = p.snapshot()
    data["patch/v"][0, 0] = np.nan
    bad = tmp_path / "bad.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="activity arrays"):
        cd.PatchNet.load(bad)
    data = p.snapshot()
    data["patch/activation"][0, 0] += 0.1
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="inconsistent"):
        cd.PatchNet.load(bad)
    graph = cd.layered(2, 3, 1, density=1)
    learner = cd.Learner(
        cd.Brain(graph, cd.learning_neuron_model()),
        graph.populations["output"],
        cd.LearnerConfig(nudge="quadratic"),
    )
    with pytest.raises(ValueError, match="reciprocal"):
        cd.PatchNet(learner)
    p.learner.brain = cd.Brain(
        p.brain.connectome, cd.learning_neuron_model().replace(adaptation=cd.Adaptation())
    )
    with pytest.raises(ValueError, match="adaptation"):
        cd.PatchNet(p.learner)


def test_structure_must_remain_reciprocal_under_the_parameter_update():
    graph = cd.Connectome.from_synapses(2, pre=[0, 1], post=[1, 0], count=[1, 2])
    brain = cd.Brain(graph, cd.learning_neuron_model(), efficacy=1 / graph.count)
    assert cd.ep_structure(brain).compatible  # current weights alone are insufficient
    learner = cd.Learner(brain, [1], cd.LearnerConfig(nudge="quadratic"))
    with pytest.raises(ValueError, match="contact/gain"):
        cd.PatchNet(learner)
    p = net()
    p.learner.plastic_synapses[0] = False
    with pytest.raises(ValueError, match="plasticity masks"):
        cd.PatchNet(p.learner)
