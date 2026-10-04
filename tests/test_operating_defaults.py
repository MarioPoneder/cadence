"""Operating defaults chosen by the readout they serve: issues 124, 125, 126, 127.

The hand-set defaults remain the controls. These tests pin what each default
resolves to, that composed brains settle at those defaults, and that the
diagnostics name the cause when a budget or rate is the problem.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd


def motor_synapses(brain):
    """The motor-to-motor synapse signs of a composed brain."""
    w = brain.brain.connectome
    motor = np.zeros(w.n, dtype=bool)
    motor[brain.motor_index] = True
    edges = motor[w.pre] & motor[w.post]
    return w.sign[edges]


# -- issue 124: motor lateral inhibition follows the readout width


def test_small_action_menu_keeps_lateral_inhibition():
    brain = cd.Brain.compose(inputs=8, actions=4, modules=(8,), seed=0)
    signs = motor_synapses(brain)
    assert len(signs) == 4 * 3 and np.all(signs == -0.5)


def test_wide_readout_drops_lateral_inhibition_by_default():
    brain = cd.Brain.compose(inputs=8, actions=36, modules=(8,), seed=0)
    assert len(motor_synapses(brain)) == 0


def test_explicit_lateral_overrides_the_width_rule():
    brain = cd.Brain.compose(inputs=8, actions=36, modules=(8,), seed=0, lateral=-0.5)
    signs = motor_synapses(brain)
    assert len(signs) == 36 * 35 and np.all(signs == -0.5)
    assert len(motor_synapses(cd.Brain.compose(inputs=8, actions=4, modules=(8,), seed=0,
                                               lateral=0.0))) == 0


def test_build_and_genome_follow_the_same_lateral_rule():
    assert len(motor_synapses(cd.Brain.build(8, 36, hidden=8, seed=0))) == 0
    signs = motor_synapses(cd.Brain.build(8, 4, hidden=8, seed=0))
    assert len(signs) == 12 and np.all(signs == -0.5)


def test_wide_composed_brain_settles_undamped_at_the_default():
    # The issue-124 defect: at lateral -0.5 a 36-action composed brain does not
    # settle an undamped free solve at all; at the resolved default it settles
    # in the ordinary few dozen sweeps.
    rng = np.random.default_rng(3)
    brain = cd.Brain.compose(inputs=64, actions=36, modules=(64, 32), seed=1)
    drive = brain.stimulus(rng.uniform(0, 1, (8, 64)), memory=False)
    phase = brain.brain.equilibrate(drive, budget=256, tolerance=3e-3)
    assert bool(np.all(phase.qualified)) and phase.steps <= 64


def test_invalid_lateral_is_still_rejected():
    with pytest.raises(ValueError, match="lateral"):
        cd.Brain.compose(inputs=8, actions=4, modules=(8,), lateral=np.nan)
    with pytest.raises(ValueError, match="lateral"):
        cd.Brain.compose(inputs=8, actions=4, modules=(8,), lateral=True)


# -- issue 125: calibration targets the competitive operating point by default


def readout_learner(outputs=36, **config):
    """Independent output neurons: every gain settles, and the operating point is exact."""
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(outputs, pre=[], post=[]),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(graph, np.arange(outputs), cd.LearnerConfig(
        qualified=True, free_steps=512, nudged_steps=512, tolerance=3e-3, **config,
    ))


def driven_readout(**config):
    """36 input neurons each driving one output neuron: gain moves the readout."""
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(
            72, pre=np.arange(36), post=np.arange(36, 72), sign=np.ones(36)
        ),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(graph, np.arange(36, 72), cd.LearnerConfig(
        qualified=True, free_steps=512, nudged_steps=512, tolerance=3e-3, **config,
    ))


def test_default_calibration_targets_the_top_output_not_the_mean():
    rng = np.random.default_rng(0)
    drive = np.zeros((4, 72))
    drive[:, :36] = rng.uniform(0.1, 0.9, (4, 36))
    learner = driven_readout()
    competitive = learner.calibrate(drive)
    report = learner.last_calibration
    assert report["target"] == "top" and report["level"] == 0.5
    chosen = next(x for x in report["candidates"] if x["gain"] == competitive)
    assert chosen["gap"] == abs(chosen["top_output"] - 0.5)
    assert chosen["top_output"] == min(
        (x["top_output"] for x in report["candidates"] if x["admitted"]),
        key=lambda top: abs(top - 0.5),
    )

    saturating = driven_readout()
    mean_target = saturating.calibrate(drive, level=0.5)
    mean_report = saturating.last_calibration
    assert mean_report["target"] == "mean"
    picked = next(x for x in mean_report["candidates"] if x["gain"] == mean_target)
    assert picked["gap"] == abs(picked["mean_output"] - 0.5)
    # Driving the mean of 36 outputs to 0.5 needs a larger gain than placing
    # the winner at 0.5: the whole readout is pushed up together (issue 125).
    assert mean_target > competitive
    assert picked["mean_output"] > chosen["mean_output"]


def test_single_output_calibration_is_unchanged_by_the_competitive_default():
    graph = cd.Connectome.from_synapses(2, pre=[0], post=[1], count=[1])

    def learner():
        return cd.Learner(
            cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1)), [1],
            cd.LearnerConfig(qualified=True, free_steps=32, tolerance=1e-9),
        )

    drive = np.array([[1., 0.]])
    assert learner().calibrate(drive) == learner().calibrate(drive, level=0.5)


def test_slotted_calibration_takes_one_winner_per_slot():
    base = readout_learner(outputs=36)
    learner = cd.Learner(base.brain, np.arange(36), base.config, slots=(12, 24))
    rng = np.random.default_rng(1)
    drive = rng.uniform(0.0, 0.4, (2, 36))
    learner.calibrate(drive, grid=[1.0])
    candidate = learner.last_calibration["candidates"][0]
    cfg = learner.config
    phase = learner.brain.equilibrate(
        drive, budget=cfg.free_steps, tolerance=cfg.tolerance, damping=cfg.damping
    )
    outputs = phase.state.activation[:, learner.output_index]
    expected = np.concatenate(
        [outputs[:, :12].max(axis=1), outputs[:, 12:].max(axis=1)]
    ).mean()
    assert candidate["top_output"] == pytest.approx(expected, abs=1e-12)
    assert candidate["mean_output"] == pytest.approx(outputs.mean(), abs=1e-12)
    assert candidate["top_output"] > candidate["mean_output"]


# -- issue 126: the bias rate follows the synapse rate unless chosen


def test_eta_bias_defaults_to_a_tenth_of_eta():
    assert cd.LearnerConfig().eta_bias == pytest.approx(0.02)
    assert cd.LearnerConfig(eta=0.0015).eta_bias == pytest.approx(0.00015)
    assert cd.LearnerConfig(eta=0.5).eta_bias == pytest.approx(0.05)
    assert cd.LearnerConfig(eta=0.5, eta_bias=0.02).eta_bias == 0.02


def test_replace_keeps_the_resolved_bias_rate_unless_rederived():
    config = cd.LearnerConfig(eta=0.2)
    lowered = replace(config, eta=0.002, eta_bias=None)
    assert lowered.eta_bias == pytest.approx(0.0002)
    with pytest.warns(RuntimeWarning, match="bias step dominates"):
        kept = replace(config, eta=0.002)
    assert kept.eta_bias == pytest.approx(0.02)
    restored = cd.LearnerConfig(**lowered.to_dict())
    assert restored == lowered


def test_dominating_bias_rate_warns_and_zero_eta_stays_silent():
    with pytest.warns(RuntimeWarning, match="bias step dominates"):
        cd.LearnerConfig(eta=0.0015, eta_bias=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig(eta=0.0, eta_bias=0.02)  # bias-only learning is a control
        cd.LearnerConfig(eta=0.2, eta_bias=0.2)


def test_composed_default_bias_rate_follows_the_composed_eta():
    config = cd.Brain.compose(inputs=4, actions=2, modules=(4,), seed=0).learner.config
    assert config.eta == 0.5 and config.eta_bias == pytest.approx(0.05)


# -- issue 127: a finite free phase that uses its whole budget says so


def exhausted_learner(tolerance=3e-3):
    pre, post = np.where(~np.eye(36, dtype=bool))
    graph = cd.NeuralGraph(
        cd.Connectome.from_synapses(36, pre=pre, post=post, sign=np.full(len(pre), -0.5)),
        cd.learning_neuron_model(dt=1),
    )
    return cd.Learner(graph, np.arange(36), cd.LearnerConfig(
        free_steps=2, nudged_steps=2, tolerance=tolerance,
    ))


def test_exhausted_finite_free_phase_warns_and_is_counted():
    learner = exhausted_learner()
    with pytest.warns(RuntimeWarning, match="entire free_steps budget") as caught:
        _, report = learner.step(np.full((1, 36), 0.2), np.array([0]))
    assert report["free_budget_exhausted"] == 1.0
    assert report["accepted"] == 1.0  # finite teaching retains its law
    assert any("free_residual" in str(w.message) for w in caught)


def test_fixed_length_phases_and_settled_phases_stay_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = exhausted_learner(tolerance=None).step(
            np.full((1, 36), 0.2), np.array([0])
        )
    assert report["free_budget_exhausted"] == 0.0

    settled = readout_learner()
    settled.config = replace(settled.config, qualified=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, report = settled.step(np.full((1, 36), 0.2), np.array([0]))
    assert report["free_budget_exhausted"] == 0.0
    assert report["free_steps"] < 512


def test_brain_step_reports_the_exhausted_demonstration():
    composed = cd.Brain.compose(inputs=4, actions=2, modules=(4,), seed=0)
    brain = cd.Brain.compose(
        inputs=4, actions=2, modules=(4,), seed=0,
        learning=replace(composed.learner.config, free_steps=1),
    )
    # The lesson is taught and counted before the following answer, whose
    # one-sweep budget then refuses without undoing the learning.
    with pytest.warns(RuntimeWarning, match="entire free_steps budget"):
        with pytest.raises(RuntimeError, match="did not settle"):
            brain.step(np.array([[0.6, 0.1, 0.2, 0.9]]), teacher=np.array([1]))
    assert brain.last_learning["demonstration_free_budget_exhausted"] == 1.0
    assert brain.last_learning["demonstration_accepted"] == 1.0
