"""The readback types the public API hands back: each one obtained from its real producer,
checked to be the exported class, frozen where it is declared frozen, and consistent with the
inputs that produced it."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import cadence as cd
from cadence.life import PatchGovernor, ThresholdGovernor
from cadence.record_ports import build


def _frozen(result: object, name: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, name, None)


def _belief(seed: int = 0) -> cd.BeliefPatch:
    patch = cd.BeliefPatch(
        cd.StructuredPort(2, [cd.DenseBlock(0, 2, 6)]),
        actions=1,
        belief=8,
        outputs=2,
        cells=32,
        active=4,
        record_width=4,
        seed=seed,
    )
    params = patch.parameters()
    params["C"] = np.random.default_rng(seed).normal(size=params["C"].shape) * 0.3
    patch.set_parameters(params)
    return patch


# ---------------------------------------------------------------------------- belief patch
def test_belief_observation_and_belief_path() -> None:
    rng = np.random.default_rng(0)
    patch = _belief(0)
    o, a, y = rng.random((2, 4, 2)), rng.random((2, 4, 1)), rng.normal(size=(2, 4, 2))
    result = patch.observe(o, a, y, rate=0.0, write=False)
    assert isinstance(result, cd.BeliefObservation)
    assert isinstance(result.path, cd.BeliefPath)
    _frozen(result, "updated")
    assert not result.updated and result.writes == 0
    assert result.initial_loss is not None and np.isfinite(result.initial_loss)
    assert result.path.belief.shape == (2, 4, patch.belief)
    assert result.path.output.shape == y.shape


def test_belief_path_from_imagination() -> None:
    rng = np.random.default_rng(1)
    patch = _belief(1)
    path = patch.imagine(rng.random((3, 5, 1)))
    assert isinstance(path, cd.BeliefPath)
    _frozen(path, "belief")
    assert path.belief.shape == (3, 5, patch.belief) and path.output.shape == (3, 5, 2)
    assert path.loss is None and np.isfinite(path.output).all()


def test_steered_path() -> None:
    rng = np.random.default_rng(2)
    brain = cd.Steered(_belief(2))
    o, a = rng.random((2, 4, 2)), np.zeros((2, 4, 1))
    path = brain.run(o, a)
    assert isinstance(path, cd.SteeredPath)
    _frozen(path, "output")
    assert path.output.shape == (2, 4, 2) and path.residual.shape == (2, 4)
    np.testing.assert_allclose(path.gains, 1.0)  # no steering: fixed gains of one
    assert path.loss is None and not path.updated and path.reason == "no_step"


# ---------------------------------------------------------------------------- the life
def test_governors_are_governors() -> None:
    assert issubclass(ThresholdGovernor, cd.Governor)
    assert issubclass(PatchGovernor, cd.Governor)
    assert isinstance(PatchGovernor(), cd.Governor)


def _life(governor: cd.Governor) -> cd.Life:
    config = cd.LifeConfig(window=16, min_window=8, min_cooldown=4, passes=3, learn_rate=2.0, horizon=3)
    return cd.Life(
        _belief(3),
        governor,
        habit=lambda r: np.array([0.5 * (0.5 - r[1])]),
        propose=lambda r: np.array([[-1.0], [0.0], [1.0]]),
        advance=lambda r, y: r + y,
        cost=lambda R, A: (R[:, 0] - R[:, 1]) ** 2 + 0.01 * A[:, 0] ** 2,
        target=lambda r, r2: r2 - r,
        baseline=1e-4,
        residual0=0.1,
        config=config,
    )


def test_decision_from_the_decide_outcome_loop() -> None:
    life = _life(ThresholdGovernor({"k_imagine": 3.0, "imagine_budget": 2, "cooldown": 0}))
    assert isinstance(life.governor, cd.Governor)
    reading = np.array([0.5, 0.5])
    for t in range(4):
        action = life.decide(reading)
        reading = np.clip(reading + np.array([0.01, 0.1 * float(action[0])]), 0.0, 1.0)
        decision = life.outcome(reading)
        assert isinstance(decision, cd.Decision)
        assert decision.t == t and decision.mode in ("habit", "imagine", "learn")
        assert decision.action.shape == (1,) and np.isfinite(decision.residual)
    assert len(life.records) == 4 and all(isinstance(r, cd.Decision) for r in life.records)


# ---------------------------------------------------------------------------- temporal patch
def test_temporal_phase_observation_and_readback() -> None:
    rng = np.random.default_rng(4)
    net = cd.TemporalPatchNet(3, 6, 2, seed=5)
    inputs, target = rng.normal(size=(2, 4, 3)) * 0.2, rng.normal(size=(2, 4, 2)) * 0.2
    phase = net.imagine(inputs)
    assert isinstance(phase, cd.TemporalPhase)
    _frozen(phase, "hidden")
    assert phase.hidden.shape == (2, 4, 6) and phase.output.shape == (2, 4, 2)
    assert phase.converged and phase.reason

    result = net.observe(inputs, target, beta=0.01, rate=0.1)
    assert isinstance(result, cd.TemporalObservation)
    assert isinstance(result.free, cd.TemporalPhase)
    _frozen(result, "updated")
    assert result.updated == (result.reason == "updated")
    assert result.free.output.shape == target.shape

    readback = net.readback()
    assert isinstance(readback, cd.TemporalReadback)
    _frozen(readback, "state")
    assert readback.state is not None and readback.state.shape == (2, 6)
    assert readback.updates == int(result.updated)


def test_constraint_report() -> None:
    rng = np.random.default_rng(6)
    net = cd.TemporalPatchNet(4, 12, 3, seed=127)
    inputs = rng.normal(size=(1, 5, 4)) * 0.2
    report = cd.TemporalMemory().protect(net, inputs, state=np.zeros((1, 12)))
    assert isinstance(report, cd.ConstraintReport)
    _frozen(report, "bytes")
    assert set(report.ranks) == {"A", "B", "C"} and report.ranks["B"] <= 4
    assert report.bytes > 0 and report.maximum_residual < 1e-12


# ---------------------------------------------------------------------------- record patches
def test_record_path_observation_contrast_and_readback() -> None:
    rng = np.random.default_rng(7)
    net = cd.RecordPatchNet(2, 3, 2, seed=8, cells=64, active=4)
    inputs, target = rng.normal(size=(2, 4, 2)), rng.normal(size=(2, 4, 2))
    result = net.observe(inputs, target, rate=0.0)
    assert isinstance(result, cd.RecordObservation)
    assert isinstance(result.prediction, cd.RecordPath)
    _frozen(result, "updated")
    assert not result.updated and result.reason == "no_step" and result.writes > 0
    assert result.prediction.hidden.shape == (2, 4, 3)

    path = net.imagine(inputs, state=np.zeros((2, 3)))
    assert isinstance(path, cd.RecordPath)
    _frozen(path, "hidden")
    assert path.output.shape == target.shape and np.isfinite(path.output).all()

    contrast = net.detune(inputs, target, beta=1e-4, state=np.zeros((2, 3)))
    assert isinstance(contrast, cd.RecordContrast)
    _frozen(contrast, "converged")
    assert contrast.converged and contrast.plus_hidden.shape == contrast.minus_hidden.shape == (2, 4, 3)
    assert set(contrast.contrast) == set(net.parameters())

    readback = net.readback()
    assert isinstance(readback, cd.RecordReadback)
    _frozen(readback, "writes")
    assert readback.writes == result.writes and readback.updates == 0
    assert readback.state is not None and readback.state.shape == (2, 3)


def test_stack_observation() -> None:
    rng = np.random.default_rng(9)
    x = np.eye(5)[rng.integers(5, size=(2, 4))]
    target = np.eye(4)[rng.integers(4, size=(2, 4))]
    net = cd.RecordPatchStack(5, 6, 4, lower=7, seed=3, cells=128, active=6)
    result = net.observe(x, target, rate=0.0, write=False)
    assert isinstance(result, cd.StackObservation)
    assert isinstance(result.prediction, cd.RecordPath)
    _frozen(result, "updated")
    assert not result.updated and result.writes == 0
    assert result.prediction.output.shape == target.shape


def test_joint_observation() -> None:
    rng = np.random.default_rng(10)
    xs = [np.eye(5)[rng.integers(5, size=(2, 4))] for _ in range(2)]
    ts = [np.eye(4)[rng.integers(4, size=(2, 4))] for _ in range(2)]
    brain = build([6, 8], [5, 5], [4, 4], [cd.Port(0, 1, 1, 3)], seed=5, cells=128, active=6)
    result = brain.observe(xs, ts, rate=0.0, write=False)
    assert isinstance(result, cd.JointObservation)
    _frozen(result, "updated")
    assert not result.updated and result.writes == 0
    assert result.delta is not None and len(result.delta) == 2
    assert [p.output.shape for p in result.settled.paths] == [t.shape for t in ts]


# ---------------------------------------------------------------------------- continuous brains
def test_patch_observation_equilibrium_and_ep_structure() -> None:
    p = cd.PatchNet.create(2, 5, 1, seed=3, density=1, tolerance=1e-7)
    structure = cd.ep_structure(p.brain)
    assert isinstance(structure, cd.EPStructure)
    _frozen(structure, "tolerance")
    assert structure.compatible and structure.to_dict()["compatible"] is True

    x = p.stimulus(np.array([[0.0, 1.0], [1.0, 0.0]]), amplitude=2)
    phase = p.settle(x)
    assert isinstance(phase, cd.Equilibrium)
    _frozen(phase, "steps")
    assert phase.residual.shape == (2,) and np.all(phase.converged)

    report = p.observe(x, np.array([[0.2], [0.8]]), source_id="one")
    assert isinstance(report, cd.PatchObservation)
    assert isinstance(report.free, cd.Equilibrium)
    _frozen(report, "updated")
    assert report.updated and report.reason == "updated"
    assert all(np.all(q.converged) for q in (report.free, report.plus, report.minus))


def test_equilibrium_from_brain_equilibrate() -> None:
    brain = cd.NeuralGraph(cd.layered(2, 3, 2, seed=4), cd.learning_neuron_model())
    drive = np.zeros((2, brain.connectome.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    result = brain.equilibrate(drive, budget=256, tolerance=1e-6)
    assert isinstance(result, cd.Equilibrium)
    _frozen(result, "residual")
    assert result.residual.shape == (2,) and np.isfinite(result.residual).all()
    assert 0 < result.steps <= 256


def test_settlement_record() -> None:
    graph = cd.layered(2, 3, 2, seed=4)
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(), backend="cpu", precision="float64")
    drive = np.zeros((2, graph.n))
    drive[:, :2] = [[1.0, 0.5], [0.2, 0.8]]
    records: list[cd.SettlementRecord] = []
    with cd.record_settlements(records.append, label="probe"):
        state = brain.settle_batch(drive, steps=5)
    (record,) = records
    assert isinstance(record, cd.SettlementRecord)
    _frozen(record, "label")
    assert record.label == "probe" and record.steps == state.steps == 5
    assert record.potential.shape == record.activation.shape == (6, 2, graph.n)
    np.testing.assert_array_equal(record.drive, drive)
