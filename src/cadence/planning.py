"""Private input planning with the same centered equilibrium detuning rule.

The action gradient is a finite-beta contrast, not a guarantee of a global
optimum. Every accepted action path is evaluated by target-free causal replay.
Planning does not execute actions, admit observations or change the learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temporal import TemporalPatchNet, TemporalPhase, _integer, contrast_asymmetry


@dataclass(frozen=True)
class TemporalPlan:
    """Detached action proposal and predicted cost, not a measured outcome.

    ``converged`` concerns the projected finite-beta gradient. A stationary plan
    may be unable to reach its goal. ``predicted_goal_met`` checks half the
    weighted mean-squared prediction error against the supplied threshold.
    ``beta`` is the detuning of the last contrast and ``contrast_halvings`` how
    often the supplied beta was halved to center the detuned paths. ``method``
    names the search direction: ``"steepest"`` steps along the projected
    contrast scaled by ``rate``; ``"bfgs"`` keeps a quasi-Newton estimate of
    the curvature over the controlled ports from the accepted steps, and its
    ``step_sizes`` are multiples of that direction.
    """

    inputs: np.ndarray
    prediction: TemporalPhase
    initial_prediction: TemporalPhase
    boundary: np.ndarray
    losses: tuple[float, ...]
    step_sizes: tuple[float, ...]
    projected_residual: float | None
    converged: bool
    predicted_goal_met: bool
    reason: str
    phase_calls: int
    block_chain_attempts: int
    energy_evaluations: int
    peak_message_bytes: int
    parameter_revision: int
    beta: float = 0.0
    contrast_halvings: int = 0
    method: str = "steepest"

    @property
    def cost(self) -> float:
        return self.losses[-1]

    @property
    def initial_cost(self) -> float:
        return self.losses[0]

    @property
    def improved(self) -> bool:
        return self.cost < self.initial_cost

    @property
    def iterations(self) -> int:
        return len(self.step_sizes)


def _broadcast_real(value: np.ndarray | float, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    try:
        result = np.broadcast_to(array.astype(float), shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to the input path") from error
    if np.isnan(result).any():
        raise ValueError(f"{name} must not contain NaN")
    return result


def plan_inputs(
    net: TemporalPatchNet,
    inputs: np.ndarray,
    *,
    goal: np.ndarray,
    controls: np.ndarray,
    bounds: tuple[np.ndarray | float, np.ndarray | float] | None = None,
    state: np.ndarray | None = None,
    beta: float = 0.01,
    rate: float = 1.0,
    max_steps: int = 32,
    max_backtracks: int = 16,
    tolerance: float = 1e-6,
    goal_tolerance: float = 1e-6,
    symmetry_tolerance: float = 0.15,
    max_halvings: int = 8,
    method: str = "steepest",
) -> TemporalPlan:
    """Repair controlled input ports; score only ordinary free predictions.

    Inputs and goal have shapes (batch,time,inputs/outputs). The boolean controls
    mask and optional bounds broadcast to inputs. Supplied output precision is
    the same positive metric used by ``observe``. Noncontrolled ports and the
    initial boundary stay fixed, and no actual observation is inferred from a
    desired goal. The returned proposal still needs execution and measured
    readback in the application.
    """
    max_steps = _integer("max_steps", max_steps, 0)
    max_backtracks = _integer("max_backtracks", max_backtracks, 1)
    for name, value in (("beta", beta), ("rate", rate), ("tolerance", tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(goal_tolerance) or goal_tolerance < 0:
        raise ValueError("goal_tolerance must be finite and nonnegative")
    if np.isnan(symmetry_tolerance) or symmetry_tolerance <= 0:
        raise ValueError("symmetry_tolerance must be positive; inf disables the check")
    max_halvings = _integer("max_halvings", max_halvings, 0)
    if method not in ("steepest", "bfgs"):
        raise ValueError('method must be "steepest" or "bfgs"')
    # A private checkpoint copy also freezes parameters and solver settings for
    # this entire proposal. No phase is carried into the original network.
    model = TemporalPatchNet.restore(net.snapshot())
    current = model._path(inputs, model.inputs, "inputs").copy()
    target = model._path(goal, model.outputs, "goal").copy()
    if current.shape[:2] != target.shape[:2]:
        raise ValueError("goal batch/time dimensions must match inputs")
    boundary = model._boundary(len(current), state)
    detuning = model._detuning_weights(beta, current.shape[1])
    if not np.isfinite(detuning).all() or np.any(np.asarray(detuning) >= 1.0):
        raise ValueError("beta*output_precision/(time*outputs) must be less than one")
    mask = np.asarray(controls)
    if mask.dtype.kind != "b":
        raise ValueError("controls must be a boolean mask")
    try:
        mask = np.broadcast_to(mask, current.shape).copy()
    except ValueError as error:
        raise ValueError("controls must broadcast to the input path") from error
    if not mask.any():
        raise ValueError("at least one input port must be controlled")
    if bounds is None:
        lower, upper = np.full(current.shape, -np.inf), np.full(current.shape, np.inf)
    else:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) tuple")
        lower = _broadcast_real(bounds[0], current.shape, "lower bounds")
        upper = _broadcast_real(bounds[1], current.shape, "upper bounds")
        if np.any(lower > upper):
            raise ValueError("lower bounds must not exceed upper bounds")
    if np.any(mask & ((current < lower) | (current > upper))):
        raise ValueError("initial controlled inputs must lie within bounds")
    fixed = current.copy()
    precision = model.output_precision
    projection = model.parameters()["B"]
    phase_calls = chain_attempts = energy_evaluations = peak_message_bytes = 0

    def record(phase: TemporalPhase) -> TemporalPhase:
        nonlocal phase_calls, chain_attempts, energy_evaluations, peak_message_bytes
        phase_calls += 1
        chain_attempts += phase.block_chain_attempts
        energy_evaluations += phase.energy_evaluations
        peak_message_bytes = max(peak_message_bytes, phase.message_bytes)
        return phase

    def loss(phase: TemporalPhase) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(0.5 * np.mean(precision * np.square(phase.output - target)))

    def project(value: np.ndarray) -> np.ndarray:
        return np.where(mask, np.clip(value, lower, upper), fixed)

    initial = record(model.imagine(current, state=boundary))
    if not initial.converged or not np.isfinite(loss(initial)):
        raise ArithmeticError("initial target-free prediction is invalid")
    prediction = initial
    costs, step_sizes = [loss(initial)], []
    converged, reason = False, "step_cap"
    residual: float | None = None
    contrast_beta, halvings = float(beta), 0

    def centered_contrast(free: TemporalPhase) -> tuple[np.ndarray | None, str]:
        """Contrast at the current proposal; beta halves until the paths are centered."""
        nonlocal contrast_beta, halvings
        while True:
            plus = record(
                model.settle(current, target=target, beta=contrast_beta, state=boundary)
            )
            if not plus.converged:
                return None, "plus_" + plus.reason
            minus = record(
                model.settle(current, target=target, beta=-contrast_beta, state=boundary)
            )
            if not minus.converged:
                return None, "minus_" + minus.reason
            if contrast_asymmetry(free, plus, minus) <= symmetry_tolerance:
                ep, _ = model._errors(current, plus.hidden, plus.output, boundary)
                em, _ = model._errors(current, minus.hidden, minus.output, boundary)
                # Time/output normalization is already in the detuned teaching loss.
                # This final batch factor makes the contrast match the full mean loss.
                with np.errstate(over="ignore", invalid="ignore"):
                    value = -((ep - em) @ projection) / (2.0 * contrast_beta * len(current))
                return np.where(mask, value, 0.0), ""
            if halvings == max_halvings:
                return None, "contrast_asymmetric"
            halvings += 1
            contrast_beta *= 0.5

    # A quasi-Newton estimate of the inverse curvature over the controlled
    # entries, built from accepted steps; it is dropped whenever a step is
    # clipped, fails the curvature condition or gives no decrease.
    controlled = int(mask.sum())
    curvature: np.ndarray | None = None
    last_step: tuple[np.ndarray, np.ndarray] | None = None

    # The last pass recomputes the residual at the last accepted actions; it
    # does not silently report a gradient from before their update.
    for iteration in range(max_steps + 1):
        gradient, failure = centered_contrast(prediction)
        if gradient is None:
            reason, residual = failure, None
            break
        if not np.isfinite(gradient).all():
            reason, residual = "nonfinite_input_gradient", None
            break
        residual = float(np.abs(current - project(current - gradient)).max())
        if residual <= tolerance:
            converged, reason = True, "projected_stationary"
            break
        if iteration == max_steps:
            break
        direction = -gradient
        if method == "bfgs":
            if last_step is not None:
                displacement, previous_gradient = last_step
                change = (gradient - previous_gradient)[mask]
                curve = float(displacement @ change)
                scale = float(np.linalg.norm(displacement) * np.linalg.norm(change))
                if np.isfinite(curve) and curve > 1e-12 * scale:
                    if curvature is None:
                        curvature = (curve / float(change @ change)) * np.eye(controlled)
                    left = np.eye(controlled) - np.outer(displacement, change) / curve
                    outer = np.outer(displacement, displacement) / curve
                    curvature = left @ curvature @ left.T + outer
            if curvature is not None:
                direction = np.zeros_like(gradient)
                direction[mask] = -(curvature @ gradient[mask])
                if not np.isfinite(direction).all() or float(np.sum(gradient * direction)) >= 0.0:
                    curvature, direction = None, -gradient
        accepted, clipped = False, False
        proposal = current
        for _ in range(2):
            step = 1.0 if curvature is not None else float(rate)
            for _ in range(max_backtracks):
                with np.errstate(over="ignore", invalid="ignore"):
                    proposal = project(current + step * direction)
                    slope = float(np.sum(gradient * (proposal - current)))
                if np.isfinite(proposal).all() and np.isfinite(slope) and slope < 0:
                    replay = record(model.imagine(proposal, state=boundary))
                    cost = loss(replay)
                    if (
                        replay.converged
                        and np.isfinite(cost)
                        and cost < costs[-1]
                        and cost <= costs[-1] + 1e-4 * slope
                    ):
                        clipped = bool(np.any(mask & (proposal != current + step * direction)))
                        last_step = ((proposal - current)[mask], gradient)
                        current, prediction = proposal, replay
                        costs.append(cost)
                        step_sizes.append(step)
                        accepted = True
                        break
                step *= 0.5
            if accepted or curvature is None:
                break
            # A curved direction that gives no decrease: restart from the contrast.
            curvature, direction = None, -gradient
        if not accepted:
            reason = "no_decreasing_causal_step"
            break
        if clipped:
            curvature, last_step = None, None
    return TemporalPlan(
        inputs=current.copy(),
        prediction=prediction,
        initial_prediction=initial,
        boundary=boundary.copy(),
        losses=tuple(costs),
        step_sizes=tuple(step_sizes),
        projected_residual=residual,
        converged=converged,
        predicted_goal_met=costs[-1] <= goal_tolerance,
        reason=reason,
        phase_calls=phase_calls,
        block_chain_attempts=chain_attempts,
        energy_evaluations=energy_evaluations,
        peak_message_bytes=peak_message_bytes,
        parameter_revision=model.readback().parameter_revision,
        beta=contrast_beta,
        contrast_halvings=halvings,
        method=method,
    )
