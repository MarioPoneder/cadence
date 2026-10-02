"""Private action repair, independent input derivatives and acceptance boundaries."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cadence import TemporalPatchNet, TemporalPlan


def same_snapshot(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        assert_array_equal(actual[key], expected[key])


def rollout(parameters, inputs, boundary):
    """Independent forward prediction: no solver or planning helper is called."""
    state = boundary.copy()
    outputs = []
    for moment in range(inputs.shape[1]):
        state = np.stack(
            [
                parameters["A"] @ np.tanh(previous) + parameters["B"] @ drive
                for previous, drive in zip(state, inputs[:, moment], strict=True)
            ]
        )
        outputs.append(np.stack([parameters["C"] @ np.tanh(row) for row in state]))
    return np.stack(outputs, axis=1)


def cost(parameters, inputs, boundary, goal, precision):
    return float(0.5 * np.mean(precision * (rollout(parameters, inputs, boundary) - goal) ** 2))


def finite_difference(function, point):
    gradient = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        gradient[index] = (function(plus) - function(minus)) / 2e-6
    return gradient


def scalar_body(*, max_iterations=80):
    net = TemporalPatchNet(2, 1, 1, tolerance=1e-12, max_iterations=max_iterations)
    net.set_parameters({"A": np.zeros((1, 1)), "B": np.array([[0.0, 1.0]]), "C": np.ones((1, 1))})
    return net


def test_weighted_batch_input_contrast_matches_independent_free_loss_gradient():
    rng = np.random.default_rng(1213)
    precision = np.array([0.3, 2.7])
    net = TemporalPatchNet(2, 3, 2, seed=1217, tolerance=1e-12, output_precision=precision)
    inputs = rng.normal(size=(2, 3, 2)) * 0.15
    boundary = rng.normal(size=(2, 3)) * 0.1
    goal = rng.normal(size=(2, 3, 2)) * 0.2
    parameters = net.parameters()
    objective = lambda path: cost(parameters, path, boundary, goal, precision)  # noqa: E731
    numeric = finite_difference(objective, inputs)
    result = net.plan(
        inputs,
        goal=goal,
        controls=np.ones(2, dtype=bool),
        state=boundary,
        beta=1e-4,
        rate=1e-3,
        max_steps=1,
        tolerance=1e-12,
        goal_tolerance=0,
    )
    assert result.iterations == 1 and result.step_sizes == (1e-3,)
    actual = (inputs - result.inputs) / result.step_sizes[0]
    assert_allclose(actual, numeric, rtol=4e-5, atol=3e-9)
    assert_allclose(result.initial_cost, objective(inputs), atol=2e-16)
    assert_allclose(result.cost, objective(result.inputs), atol=2e-16)
    assert_allclose(
        result.prediction.output, rollout(parameters, result.inputs, boundary), atol=1e-16
    )
    # A capped report must describe the last accepted point, not the old gradient.
    final_gradient = finite_difference(objective, result.inputs)
    assert_allclose(result.projected_residual, np.abs(final_gradient).max(), rtol=4e-5, atol=3e-9)
    assert not result.converged and result.reason == "step_cap"


def test_held_ports_and_bounds_allow_progress_without_faking_goal_attainment():
    net = scalar_body()
    inputs = np.array([[[5.0, 0.0], [-3.0, 0.0]]])
    goal = np.full((1, 2, 1), 0.8)
    result = net.plan(
        inputs,
        goal=goal,
        controls=np.array([False, True]),
        bounds=(np.array([-1.0, -0.2]), np.array([1.0, 0.2])),
        rate=8.0,
        beta=1e-4,
        max_steps=8,
    )
    assert_array_equal(result.inputs[:, :, 0], inputs[:, :, 0])
    assert_allclose(result.inputs[:, :, 1], 0.2, atol=0, rtol=0)
    assert_array_equal(result.prediction.output, np.tanh(result.inputs[:, :, 1:]))
    assert result.improved and result.converged
    assert result.reason == "projected_stationary" and result.projected_residual == 0
    assert not result.predicted_goal_met
    assert np.all(np.diff(result.losses) < 0)
    assert len(result.losses) == result.iterations + 1


def test_large_input_step_backtracks_until_free_prediction_cost_decreases():
    net = scalar_body()
    result = net.plan(
        np.zeros((1, 1, 2)),
        goal=np.full((1, 1, 1), 0.2),
        controls=[False, True],
        rate=100.0,
        max_steps=1,
        beta=1e-4,
    )
    assert isinstance(result, TemporalPlan)
    assert result.improved and result.iterations == 1
    assert 0 < result.step_sizes[0] < 100
    assert result.phase_calls > 6  # Initial, two pairs, accepted and rejected free paths.
    assert_allclose(result.cost, 0.5 * (np.tanh(result.inputs[0, 0, 1]) - 0.2) ** 2)


def test_failed_causal_replays_cannot_replace_valid_prediction(monkeypatch):
    net = scalar_body()
    inputs = np.zeros((1, 1, 2))
    before = net.snapshot()
    original = TemporalPatchNet._solve
    free_calls = 0

    def injected(self, path, boundary, teaching, beta):
        nonlocal free_calls
        phase = original(self, path, boundary, teaching, beta)
        if beta == 0:
            free_calls += 1
            if free_calls > 1:
                phase = replace(phase, converged=False, reason="injected_replay_failure")
        return phase

    monkeypatch.setattr(TemporalPatchNet, "_solve", injected)
    result = net.plan(
        inputs,
        goal=np.full((1, 1, 1), 0.4),
        controls=[False, True],
        max_backtracks=3,
    )
    assert result.reason == "no_decreasing_causal_step"
    assert result.prediction.converged and not result.converged
    assert not result.improved and result.iterations == 0
    assert result.projected_residual > 0
    assert result.phase_calls == 6 and free_calls == 4
    assert_array_equal(result.inputs, inputs)
    assert_array_equal(result.prediction.output, result.initial_prediction.output)
    same_snapshot(before, net.snapshot())


def test_planning_preserves_live_state_revisions_and_caller_owned_arrays():
    rng = np.random.default_rng(1223)
    net = TemporalPatchNet(2, 3, 1, seed=1229, tolerance=1e-11)
    primer = rng.normal(size=(2, 2, 2)) * 0.1
    assert net.observe(primer, np.full((2, 2, 1), 0.2)).updated
    before = net.snapshot()
    inputs = rng.normal(size=(2, 3, 2)) * 0.1
    goal = np.full((2, 3, 1), -0.2)
    controls = np.array([[[False, True], [True, False], [False, True]]])
    lower, upper = np.full((3, 2), -0.5), np.full((3, 2), 0.5)
    copies = [value.copy() for value in (inputs, goal, controls, lower, upper)]
    for value in (inputs, goal, controls, lower, upper):
        value.setflags(write=False)
    result = net.plan(inputs, goal=goal, controls=controls, bounds=(lower, upper), max_steps=3)
    same_snapshot(before, net.snapshot())
    assert_array_equal(result.boundary, before["state"])
    assert result.parameter_revision == net.readback().parameter_revision
    mask = np.broadcast_to(controls, inputs.shape)
    assert_array_equal(result.inputs[~mask], inputs[~mask])
    for expected, actual in zip(copies, (inputs, goal, controls, lower, upper), strict=True):
        assert_array_equal(actual, expected)
    # Detached results cannot be used as a back door into the continuing life.
    result.inputs[:] = 12
    result.boundary[:] = 13
    result.prediction.hidden[:] = 14
    result.prediction.output[:] = 15
    result.initial_prediction.output[:] = 16
    same_snapshot(before, net.snapshot())


def test_explicit_initial_boundary_is_fixed_private_and_changes_prediction():
    net = TemporalPatchNet(1, 1, 1, tolerance=1e-12)
    net.set_parameters({"A": np.array([[0.7]]), "B": np.zeros((1, 1)), "C": np.ones((1, 1))})
    net.advance(np.zeros((1, 1, 1)))
    before = net.snapshot()
    state = np.array([[0.4]])
    state.setflags(write=False)
    path = np.zeros((1, 3, 1))
    result = net.plan(path, goal=np.zeros((1, 3, 1)), controls=True, state=state)
    assert_array_equal(result.boundary, state)
    assert_allclose(result.prediction.output, rollout(net.parameters(), path, state), atol=1e-16)
    assert np.max(np.abs(result.prediction.output)) > 0.1
    assert result.converged and not result.predicted_goal_met
    same_snapshot(before, net.snapshot())


def test_zero_input_coupling_is_stationary_but_cannot_invent_a_reached_goal():
    net = TemporalPatchNet(1, 2, 1, tolerance=1e-12)
    parameters = net.parameters()
    parameters["B"][:] = 0
    net.set_parameters(parameters)
    path = np.zeros((1, 2, 1))
    before = net.snapshot()
    result = net.plan(path, goal=np.ones((1, 2, 1)), controls=True)
    assert result.converged and result.reason == "projected_stationary"
    assert result.projected_residual == 0 and not result.predicted_goal_met
    assert not result.improved and result.iterations == 0
    assert_array_equal(result.inputs, path)
    assert_array_equal(result.prediction.output, np.zeros((1, 2, 1)))
    assert result.cost == 0.5
    same_snapshot(before, net.snapshot())


def test_zero_step_cap_checks_stationarity_without_executing_a_step():
    net = scalar_body()
    path = np.zeros((1, 1, 2))
    capped = net.plan(path, goal=np.full((1, 1, 1), 0.4), controls=[False, True], max_steps=0)
    assert capped.reason == "step_cap" and not capped.converged
    assert capped.iterations == 0 and capped.projected_residual > 0.1
    assert capped.phase_calls == 3 and not capped.predicted_goal_met
    assert_array_equal(capped.inputs, path)
    stationary = net.plan(path, goal=np.zeros((1, 1, 1)), controls=[False, True], max_steps=0)
    assert stationary.converged and stationary.predicted_goal_met
    assert stationary.reason == "projected_stationary" and stationary.projected_residual == 0


@pytest.mark.parametrize("failed_sign", [1, -1])
def test_failed_detuned_phase_returns_last_valid_private_plan(monkeypatch, failed_sign):
    net = scalar_body()
    inputs = np.zeros((1, 2, 2))
    target = np.full((1, 2, 1), 0.6)
    before = net.snapshot()
    original = TemporalPatchNet._solve
    sign_calls = 0
    phases = []

    def injected(self, path, boundary, teaching, beta):
        nonlocal sign_calls
        phase = original(self, path, boundary, teaching, beta)
        if beta * failed_sign > 0:
            sign_calls += 1
            if sign_calls == 2:
                phase = replace(phase, converged=False, reason="injected_phase_failure")
        phases.append(phase)
        return phase

    monkeypatch.setattr(TemporalPatchNet, "_solve", injected)
    result = net.plan(inputs, goal=target, controls=[False, True], rate=0.1, max_steps=4)
    assert result.iterations == 1 and result.improved
    assert not result.converged and result.projected_residual is None
    assert result.reason == ("plus_" if failed_sign > 0 else "minus_") + "injected_phase_failure"
    assert_allclose(
        result.prediction.output, rollout(net.parameters(), result.inputs, result.boundary)
    )
    assert result.prediction.converged
    assert result.phase_calls == len(phases)
    assert result.block_chain_attempts == sum(p.block_chain_attempts for p in phases)
    assert result.energy_evaluations == sum(p.energy_evaluations for p in phases)
    assert result.peak_message_bytes == max(p.message_bytes for p in phases)
    same_snapshot(before, net.snapshot())


def test_failed_initial_prediction_is_rejected_without_mutation(monkeypatch):
    net = scalar_body()
    before = net.snapshot()
    original = TemporalPatchNet._solve

    def invalid(self, path, boundary, teaching, beta):
        phase = original(self, path, boundary, teaching, beta)
        return replace(phase, converged=False, reason="injected_invalid_free")

    monkeypatch.setattr(TemporalPatchNet, "_solve", invalid)
    with pytest.raises(ArithmeticError, match="initial"):
        net.plan(np.zeros((1, 1, 2)), goal=np.ones((1, 1, 1)), controls=True)
    same_snapshot(before, net.snapshot())


def test_failed_first_learning_phase_never_returns_its_nudged_goal_as_prediction():
    net = scalar_body(max_iterations=0)
    inputs = np.array([[[0.0, 0.1]]])
    before = net.snapshot()
    result = net.plan(inputs, goal=np.ones((1, 1, 1)), controls=[False, True])
    assert not result.converged and not result.predicted_goal_met
    assert result.projected_residual is None and result.iterations == 0
    assert result.reason.startswith("plus_")
    assert_array_equal(result.inputs, inputs)
    assert_array_equal(result.prediction.output, np.tanh(inputs[:, :, 1:]))
    same_snapshot(before, net.snapshot())


@pytest.mark.parametrize(
    "overrides",
    [
        {"controls": np.array([0, 1])},
        {"controls": np.zeros(2, dtype=bool)},
        {"controls": np.ones((4, 3), dtype=bool)},
        {"bounds": [0.0, 1.0]},
        {"bounds": (0.0,)},
        {"bounds": (np.nan, 1.0)},
        {"bounds": (1.0, -1.0)},
        {"bounds": (np.zeros(3), np.ones(3))},
        {"bounds": (0.1, 1.0)},
        {"beta": 0.0},
        {"beta": -0.1},
        {"beta": np.nan},
        {"beta": 2.0},
        {"rate": 0.0},
        {"rate": -1.0},
        {"rate": np.inf},
        {"max_steps": -1},
        {"max_steps": True},
        {"max_backtracks": 0},
        {"tolerance": 0.0},
        {"tolerance": np.nan},
        {"goal_tolerance": -1.0},
        {"goal_tolerance": np.inf},
        {"goal": np.full((1, 2, 1), np.nan)},
        {"goal": np.ones((1, 3, 1))},
        {"state": np.ones((2, 1))},
        {"state": np.array([[np.nan]])},
        {"symmetry_tolerance": 0.0},
        {"symmetry_tolerance": np.nan},
        {"max_halvings": -1},
        {"max_halvings": True},
        {"method": "newton"},
    ],
)
def test_invalid_planning_arguments_are_atomic(overrides):
    net = scalar_body()
    net.advance(np.ones((1, 1, 2)) * 0.1)
    before = net.snapshot()
    kwargs = {"goal": np.zeros((1, 2, 1)), "controls": [False, True], **overrides}
    with pytest.raises(ValueError):
        net.plan(np.zeros((1, 2, 2)), **kwargs)
    same_snapshot(before, net.snapshot())


def test_weighted_negative_phase_beta_boundary_is_enforced():
    net = TemporalPatchNet(1, 2, 2, output_precision=np.array([0.5, 8.0]))
    before = net.snapshot()
    with pytest.raises(ValueError, match="precision"):
        net.plan(np.zeros((1, 2, 1)), goal=np.ones((1, 2, 2)), controls=True, beta=0.5)
    same_snapshot(before, net.snapshot())


def off_center_case():
    """A saturating four-unit net and a 24-step goal whose default-beta phases sit off center."""
    rng = np.random.default_rng(2)
    net = TemporalPatchNet(2, 4, 1, seed=2, initial_radius=1.5)
    return net, rng.normal(size=(1, 24, 2)), np.sign(rng.normal(size=(1, 24, 1)))


def test_off_center_detuned_paths_halve_beta_until_the_contrast_is_a_derivative():
    net, inputs, goal = off_center_case()
    boundary = np.zeros((1, 4))
    parameters = net.parameters()
    expected = np.abs(
        finite_difference(lambda point: cost(parameters, point, boundary, goal, 1.0), inputs)
    ).max()
    common = {"goal": goal, "controls": np.array(True), "state": boundary, "max_steps": 0}
    before = net.snapshot()

    refused = net.plan(inputs, max_halvings=0, **common)
    assert refused.reason == "contrast_asymmetric" and refused.projected_residual is None
    assert refused.beta == 0.01 and refused.contrast_halvings == 0 and not refused.converged
    assert_array_equal(refused.inputs, inputs)
    unguarded = net.plan(inputs, symmetry_tolerance=np.inf, **common)
    assert unguarded.contrast_halvings == 0 and unguarded.beta == 0.01
    assert abs(unguarded.projected_residual - expected) > 0.05 * expected

    default = net.plan(inputs, **common)
    assert default.contrast_halvings == 1 and default.beta == 0.005
    assert abs(default.projected_residual - expected) < 0.02 * expected

    centered = net.plan(inputs, symmetry_tolerance=0.1, **common)
    assert centered.contrast_halvings == 2 and centered.beta == 0.0025
    assert abs(centered.projected_residual - expected) < 0.005 * expected
    assert centered.phase_calls == 1 + 2 * 3
    stricter = net.plan(inputs, symmetry_tolerance=0.05, **common)
    assert stricter.contrast_halvings == 3 and stricter.beta == 0.00125
    same_snapshot(before, net.snapshot())


def test_quasi_newton_direction_reaches_the_same_stationary_cost_with_fewer_phases():
    rng = np.random.default_rng(5)
    net = TemporalPatchNet(2, 3, 1, seed=5, initial_radius=0.9)
    inputs = rng.normal(size=(1, 12, 2)) * 0.3
    goal = rng.normal(size=(1, 12, 1)) * 0.5
    before = net.snapshot()
    common = {
        "goal": goal, "controls": np.array(True), "state": np.zeros((1, 3)), "beta": 1e-4,
        "rate": 10.0, "max_steps": 400, "tolerance": 1e-6,
    }
    steepest = net.plan(inputs, method="steepest", **common)
    default = net.plan(inputs, **common)
    curved = net.plan(inputs, method="bfgs", **common)
    assert steepest.method == "steepest" and curved.method == "bfgs"
    assert_array_equal(default.inputs, steepest.inputs)
    assert steepest.converged and curved.converged
    assert steepest.cost < 1e-8 and curved.cost < 1e-8
    assert curved.iterations < steepest.iterations
    assert curved.phase_calls < 0.5 * steepest.phase_calls
    assert curved.step_sizes[0] <= 10.0 and max(curved.step_sizes[1:]) <= 1.0
    assert all(later < earlier for earlier, later in zip(curved.losses, curved.losses[1:], strict=False))
    same_snapshot(before, net.snapshot())


def test_quasi_newton_step_is_clipped_to_bounds_and_still_decreases_cost():
    net = scalar_body()
    inputs = np.array([[[5.0, 0.0], [-3.0, 0.0], [1.0, 0.0]]])
    goal = np.full((1, 3, 1), 0.8)
    plan = net.plan(
        inputs, goal=goal, controls=np.array([False, True]), bounds=(-0.2, 0.2),
        rate=8.0, max_steps=8, method="bfgs",
    )
    assert plan.improved and plan.converged and not plan.predicted_goal_met
    np.testing.assert_allclose(plan.inputs[:, :, 1], 0.2)
    np.testing.assert_array_equal(plan.inputs[:, :, 0], inputs[:, :, 0])
