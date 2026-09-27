"""Independent equations and admission checks for optional warm energy refinement."""

from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence._refine import _quadratic, _Reduced, refine_equilibrium, validate_hybrid


def brain_for(w, **rule_changes):
    w = np.asarray(w, dtype=float)
    post, pre = np.nonzero(w)
    graph = cd.Connectome.from_synapses(len(w), pre=pre, post=post)
    rule = cd.NeuronModel(dt=0.5, slope=1, threshold=0, leak=1, gain=1, **rule_changes)
    return cd.Brain(graph, rule, efficacy=w[graph.post, graph.pre])


def full_residual(brain, drive, state, nudge=None):
    s = brain.neuron_model.activation(state.v)
    defect = np.atleast_2d(s) @ brain.dense() + np.atleast_2d(drive) + brain.bias - state.v
    if nudge is not None:
        defect += nudge.drive(np.atleast_2d(s))
    return np.max(np.abs(defect), axis=1)


def test_capped_local_root_refines_without_mutating_phase_or_parameters():
    brain = brain_for([[0, 1.8], [1.8, 0]])
    drive = np.array([[0.01, 0], [0.04, -0.02]])
    local = brain.equilibrate(drive, budget=2, chunk=1, tolerance=1e-11)
    assert not local.converged.any()
    before = local.state.v.copy(), local.state.activation.copy(), brain.weights.copy()
    refined = refine_equilibrium(brain, drive, local, np.array([0]))
    assert refined.qualified.all() and refined.converged.all()
    assert refined.refinement.status == ("refined", "refined")
    assert (refined.refinement.steps > 0).all()
    assert refined.steps == local.steps
    np.testing.assert_array_equal(refined.refinement.local_residual, local.residual)
    np.testing.assert_allclose(
        refined.residual, full_residual(brain, drive, refined.state), atol=1e-15
    )
    for got, wanted in zip(
        (local.state.v, local.state.activation, brain.weights), before, strict=True
    ):
        np.testing.assert_array_equal(got, wanted)
    reference = brain.equilibrate(drive, budget=4096, chunk=16, tolerance=1e-13)
    assert reference.converged.all()
    np.testing.assert_allclose(refined.state.v, reference.state.v, atol=1e-10)


def test_converged_local_state_keeps_exact_state_and_zero_newton_steps():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    drive = np.array([[0.2, -0.1]])
    local = brain.equilibrate(drive, budget=512, tolerance=1e-10)
    result = refine_equilibrium(brain, drive, local, np.array([0]))
    assert result.qualified.all() and result.refinement.status == ("local",)
    np.testing.assert_array_equal(result.state.v, local.state.v)
    np.testing.assert_array_equal(result.state.activation, local.state.activation)
    assert result.refinement.steps.tolist() == [0]


def test_zero_residual_saddle_rejected_even_without_refinement_budget():
    brain = brain_for([[0, 2.4], [2.4, 0]])
    local = brain.equilibrate(np.zeros(2), budget=0, tolerance=1e-10)
    result = refine_equilibrium(brain, np.zeros(2), local, np.array([0]), max_steps=0)
    assert result.converged.all() and not result.qualified.any()
    assert result.refinement.status == ("nonpositive_curvature",)
    assert result.refinement.min_curvature[0] < 0
    np.testing.assert_array_equal(result.state.v, local.state.v)


def test_zero_budget_audits_without_repairing_a_capped_state():
    brain = brain_for([[0, 1.8], [1.8, 0]])
    local = brain.equilibrate(np.array([0.2, 0.1]), budget=0, tolerance=1e-10)
    result = refine_equilibrium(brain, np.array([0.2, 0.1]), local, np.array([0]), max_steps=0)
    assert not result.qualified.any()
    assert result.refinement.status == ("step_cap",)
    np.testing.assert_array_equal(result.state.v, local.state.v)


def test_reduced_energy_derivatives_and_full_hessian_schur_complement():
    a = np.array([[0.4, -0.3], [0.2, 0.5]])
    b = np.array([[0.0, 0.1], [0.1, 0.0]])
    quadratic = np.array([-0.2, 0.4])
    system = _Reduced(a, b, np.array([0.1, -0.2]), np.array([0.3, 0.2]), quadratic)
    x = np.array([0.2, -0.1])
    energy, gradient, hessian = system.evaluate(x)
    h = 1e-5
    numeric_gradient = np.empty(2)
    numeric_hessian = np.empty((2, 2))
    for i in range(2):
        direction = np.eye(2)[i] * h
        plus, gp, _ = system.evaluate(x + direction)
        minus, gm, _ = system.evaluate(x - direction)
        numeric_gradient[i] = (plus - minus) / (2 * h)
        numeric_hessian[:, i] = (gp - gm) / (2 * h)
    np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-10)
    np.testing.assert_allclose(hessian, numeric_hessian, atol=2e-10)
    si = np.tanh((system.hi + a @ x) / 2)
    all_s = np.r_[si, x]
    w = np.block([[np.zeros((2, 2)), a], [a.T, b]])
    full_h = np.diag(2 / (1 - all_s**2) + np.r_[np.zeros(2), quadratic]) - w
    schur = full_h[2:, 2:] - full_h[2:, :2] @ np.linalg.solve(full_h[:2, :2], full_h[:2, 2:])
    np.testing.assert_allclose(hessian, schur, atol=1e-15)
    primitive = (1 + all_s) * np.log1p(all_s) + (1 - all_s) * np.log1p(-all_s)
    full_energy = (
        primitive.sum()
        - np.r_[system.hi, system.hc] @ all_s
        - 0.5 * all_s @ w @ all_s
        + 0.5 * quadratic @ (x * x)
    )
    assert abs(energy - full_energy) < 1e-15


@pytest.mark.parametrize("beta", [-0.3, 0.3])
def test_batch_signed_weight_and_anchors_match_original_equations(beta):
    brain = brain_for([[0, 0.7, -0.2], [0.7, 0, 0.3], [-0.2, 0.3, 0]])
    drive = np.array([[0.2, -0.1, 0], [-0.3, 0.2, 0.1]])
    nudge = cd.Nudge(
        target=np.array([[0, 0.2, -0.4], [0, -0.2, 0.1]]),
        mask=np.array([0, 1, 0.5]),
        beta=beta,
        weight=np.array([1, -2]),
        anchor=np.array([0, 0.1, -0.1]),
        anchor_gain=np.array([0, 0.3, 0.2]),
    )
    local = brain.equilibrate(drive, budget=1, tolerance=1e-11, nudge=nudge)
    refined = refine_equilibrium(brain, drive, local, np.array([0]), nudge=nudge)
    assert refined.qualified.all()
    np.testing.assert_allclose(full_residual(brain, drive, refined.state, nudge), 0, atol=1e-11)
    reference = brain.equilibrate(drive, budget=2048, tolerance=1e-13, nudge=nudge)
    assert reference.converged.all()
    np.testing.assert_allclose(refined.state.v, reference.state.v, atol=1e-10)


def test_anchor_nudge_cancellation_keeps_nonzero_linear_drive():
    nudge = cd.Nudge(
        np.array([0, 0.7]),
        np.array([0, 1]),
        -0.5,
        anchor=np.array([0, -0.3]),
        anchor_gain=np.array([0, 0.5]),
    )
    linear, coefficient = _quadratic(nudge, (1, 2), np.array([0]))
    assert coefficient[0, 1] == 0 and linear[0, 1] == -0.5
    brain = brain_for([[0, 0.4], [0.4, 0]])
    local = brain.equilibrate(np.zeros(2), budget=0, tolerance=1e-12, nudge=nudge)
    result = refine_equilibrium(brain, np.zeros(2), local, np.array([0]), nudge=nudge)
    assert result.qualified.all()
    np.testing.assert_allclose(
        full_residual(brain, np.zeros(2), result.state, nudge), 0, atol=1e-12
    )


def test_curvature_is_checked_at_returned_local_state_not_reconstructed_input():
    brain = brain_for([[0, 3], [3, 0]])
    # A deliberately loose local tolerance accepts the potential equation, but
    # the input at zero gives a negative full-Hessian Schur complement.
    state = cd.BrainState(
        np.array([0.0, 1.0]), brain.neuron_model.activation(np.array([0.0, 1.0])), np.zeros(2), 0
    )
    local = cd.Equilibrium(state, np.array([100.0]), 0, 100.0)
    result = refine_equilibrium(brain, np.zeros(2), local, np.array([0]))
    assert result.converged.all() and not result.qualified.any()
    np.testing.assert_array_equal(result.state.v, state.v)


def test_nonfinite_saturated_and_inconsistent_states_fail_closed():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    local = brain.equilibrate(np.zeros(2), budget=0, tolerance=1e-10)
    for potential, activation in (
        ([0, np.nan], [0, np.nan]),
        ([0, 1000.0], [0, 1.0]),
        ([0, 0], [0, 0.4]),
    ):
        state = cd.BrainState(
            np.array(potential, dtype=float), np.array(activation), np.zeros(2), 0
        )
        result = refine_equilibrium(brain, np.zeros(2), replace(local, state=state), np.array([0]))
        assert not result.qualified.any()
        assert result.refinement.status[0] not in ("local", "refined")


def test_unsupported_models_effective_asymmetry_and_input_coupling_rejected():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    for rule in (
        brain.neuron_model.replace(leak=0.1),
        brain.neuron_model.replace(slope=2),
        brain.neuron_model.replace(threshold=0.2),
        brain.neuron_model.replace(adaptation=cd.Adaptation()),
    ):
        with pytest.raises(ValueError, match="smooth tanh"):
            validate_hybrid(cd.Brain(brain.connectome, rule), np.array([0]))
    asymmetric = brain.with_parameters(efficacy=np.array([0.5, 0.6]))
    with pytest.raises(ValueError, match="exact reciprocal"):
        validate_hybrid(asymmetric, np.array([0]))
    with pytest.raises(ValueError, match="mutual weights"):
        validate_hybrid(brain, np.array([0, 1]))
    for ports in (np.array([0.0]), np.array([True]), np.array([0, 0]), np.array([-1])):
        with pytest.raises(ValueError, match="indices"):
            validate_hybrid(brain, ports)


def test_unsupported_nudges_and_invalid_inputs_are_rejected():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    local = brain.equilibrate(np.zeros(2), budget=0, tolerance=1e-10)
    nudges = (
        cd.Nudge(np.zeros(2), np.ones(2), 0.1),
        cd.Nudge(np.zeros(2), np.array([0, 1]), 0, anchor=np.zeros(2), anchor_gain=np.ones(2)),
        cd.Nudge(np.zeros(2), np.array([0, 1]), 0.1, softmax_temperature=1),
    )
    for nudge in nudges:
        with pytest.raises(ValueError):
            refine_equilibrium(brain, np.zeros(2), local, np.array([0]), nudge=nudge)
    for budget in (-1, True, 0.5):
        with pytest.raises(ValueError, match="max_steps"):
            refine_equilibrium(brain, np.zeros(2), local, np.array([0]), max_steps=budget)
    with pytest.raises(ValueError, match="finite"):
        refine_equilibrium(brain, np.array([np.inf, 0]), local, np.array([0]))


def test_empty_input_set_full_energy_and_all_input_independent_case():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    drive = np.array([0.3, 0.1])
    local = brain.equilibrate(drive, budget=0, tolerance=1e-11)
    result = refine_equilibrium(brain, drive, local, np.array([], dtype=int))
    assert result.qualified.all()
    isolated = brain_for(np.zeros((2, 2)))
    local = isolated.equilibrate(drive, budget=0, tolerance=1e-11)
    result = refine_equilibrium(isolated, drive, local, np.array([0, 1]))
    assert result.qualified.all()
    np.testing.assert_array_equal(result.state.v, local.state.v + drive)


def test_input_saturation_and_nonfinite_adaptation_cannot_qualify():
    brain = brain_for([[0, 0.5], [0.5, 0]])
    local = brain.equilibrate(np.zeros(2), budget=0, tolerance=1e4)
    for potential, adaptation in (([1000.0, 0.0], [0.0, 0.0]), ([0.0, 0.0], [np.nan, 0.0])):
        v = np.array(potential)
        state = cd.BrainState(v, brain.neuron_model.activation(v), np.array(adaptation), 0)
        result = refine_equilibrium(brain, np.zeros(2), replace(local, state=state), np.array([0]))
        assert not result.qualified.any()


def test_changed_state_clears_stale_local_trajectory_and_movement():
    brain = brain_for([[0, 1.8], [1.8, 0]])
    drive = np.array([[0.1, 0]])
    local = brain.equilibrate(drive, budget=1, tolerance=1e-11)
    local = replace(
        local,
        state=replace(local.state, trajectory=np.zeros((1, 1, 2)), activity_change=np.array([3.0])),
    )
    result = refine_equilibrium(brain, drive, local, np.array([0]))
    assert result.qualified.all()
    assert result.state.trajectory is None and result.state.activity_change is None
    assert local.state.trajectory is not None and local.state.activity_change is not None


def test_refinement_cap_is_failure_with_actual_returned_residual():
    brain = brain_for([[0, 1.8], [1.8, 0]])
    drive = np.array([[0.8, 0]])
    local = brain.equilibrate(drive, budget=0, tolerance=1e-14)
    result = refine_equilibrium(brain, drive, local, np.array([0]), max_steps=1)
    assert not result.qualified.any() and result.refinement.status == ("step_cap",)
    assert result.refinement.steps.tolist() == [1]
    np.testing.assert_allclose(
        result.residual, full_residual(brain, drive, result.state), atol=1e-15
    )


def test_numerical_linear_solve_failure_returns_unqualified(monkeypatch):
    brain = brain_for([[0, 0.8], [0.8, 0]])
    drive = np.array([[0.1, 0]])
    local = brain.equilibrate(drive, budget=0, tolerance=1e-12)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("fixture")

    monkeypatch.setattr(np.linalg, "solve", fail)
    result = refine_equilibrium(brain, drive, local, np.array([0]))
    assert result.refinement.status == ("linear_solve_failed",)
    assert not result.qualified.any()
    np.testing.assert_array_equal(result.state.v, local.state.v)


@pytest.mark.parametrize("saturated_port", [0, 1])
def test_public_logistic_saturation_precedes_tanh_rounding(saturated_port):
    brain = brain_for(np.zeros((2, 2)))
    drive = np.zeros((1, 2))
    drive[0, saturated_port] = 37
    assert np.tanh(37 / 2) < 1
    assert brain.neuron_model.activation(np.array([37.0]))[0] == 1
    state = cd.BrainState(
        drive.copy(), brain.neuron_model.activation(drive), np.zeros_like(drive), 0
    )
    local = cd.Equilibrium(state, np.zeros(1), 0, 1e-10)
    result = refine_equilibrium(brain, drive, local, np.array([0]))
    assert result.converged.all() and not result.qualified.any()
    assert result.refinement.status == ("nonfinite_curvature_or_saturation",)


def test_original_nudge_arithmetic_overflow_cannot_be_hidden_by_coefficient_cancellation():
    brain = brain_for(np.zeros((2, 2)))
    drive = np.array([[0.0, -1e100]])
    nudge = cd.Nudge(np.array([0.0, 1e100]), np.array([0.0, 1.0]), 1e250, weight=np.array([1e-250]))
    state = cd.BrainState(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), 0)
    local = cd.Equilibrium(state, np.array([np.inf]), 0, 1e-10)
    result = refine_equilibrium(brain, drive, local, np.array([0]), nudge=nudge)
    assert np.isinf(result.residual).all() and not result.qualified.any()


@pytest.mark.parametrize("field", ["v", "activation", "adaptation"])
def test_all_state_storage_must_be_float64_to_prevent_truncated_refinement(field):
    brain = brain_for([[0, 0.5], [0.5, 0]])
    drive = np.array([[0.2, 0.1]])
    local = brain.equilibrate(drive, budget=0, tolerance=1e-11)
    state = replace(local.state, **{field: getattr(local.state, field).astype(np.int64)})
    with pytest.raises(ValueError, match="float64 state"):
        refine_equilibrium(brain, drive, replace(local, state=state), np.array([0]))
