"""Temporal residual patches, bounded activity and centered equilibrium detuning.

The residual-energy construction is a predictive-coding/EP abstraction, not a
novel learning theorem. Adjacent temporal patches exchange dense hidden-width
precision messages. This is block-local computation, not scalar-synapse-only
computation. No derivative through the solver is used for learning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .planning import TemporalPlan

__all__ = [
    "TemporalPatchNet",
    "TemporalPhase",
    "TemporalObservation",
    "TemporalReadback",
    "contrast_asymmetry",
]


@dataclass(frozen=True)
class TemporalPhase:
    """A detached solved path; hidden preactivation is (batch, time, hidden)."""

    hidden: np.ndarray
    output: np.ndarray
    residual: float | None
    converged: bool
    reason: str
    iterations: int
    minimum_pivot_eigenvalue: float | None
    message_bytes: int
    energy_history: tuple[float, ...]
    accepted_step_sizes: tuple[float, ...]
    accepted_damping: tuple[float, ...]
    block_chain_attempts: int
    energy_evaluations: int

    @property
    def activity(self) -> np.ndarray:
        """Bounded tanh activity, detached from the phase and live network."""
        return np.asarray(np.tanh(self.hidden))

    @property
    def final_state(self) -> np.ndarray:
        return self.hidden[:, -1].copy()

    @property
    def energy(self) -> float | None:
        return self.energy_history[-1] if self.energy_history else None


@dataclass(frozen=True)
class TemporalObservation:
    """External teaching attempt; only its free phase may become active state."""

    updated: bool
    reason: str
    free: TemporalPhase
    plus: TemporalPhase | None = None
    minus: TemporalPhase | None = None
    delta: dict[str, np.ndarray] | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    accepted_rate: float | None = None
    replay_losses: tuple[float | None, ...] = ()
    replay_calls: int = 0
    beta: float | None = None
    contrast_halvings: int = 0


@dataclass(frozen=True)
class TemporalReadback:
    """Read-only self-observation, never automatically admitted as teaching data.

    Residual/energy describe the last carried free solve. Their parameter
    revision may precede a subsequent learning update; they do not certify
    equilibrium under changed parameters.
    """

    state: np.ndarray | None
    activity: np.ndarray | None
    residual: float | None
    energy: float | None
    updates: int
    parameter_revision: int
    state_parameter_revision: int | None


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def contrast_asymmetry(free: TemporalPhase, plus: TemporalPhase, minus: TemporalPhase) -> float:
    """How far the two detuned paths sit off center around the free path.

    The ratio divides the norm of ``plus + minus - 2 * free`` by the norm of
    ``plus - minus`` over the hidden paths. A centered contrast has a ratio
    proportional to beta on a sufficiently smooth branch when its first-order
    response is nonzero. A large ratio can signal nonlinear response or a
    branch change; a small ratio does not certify gradient accuracy or branch
    identity. This is a diagnostic, not an error bound. Coincident phases report
    zero only when they also coincide with the free path; otherwise infinity.
    """
    spread = float(np.linalg.norm(plus.hidden - minus.hidden))
    midpoint_error = float(np.linalg.norm(plus.hidden + minus.hidden - 2.0 * free.hidden))
    if spread == 0.0:
        return 0.0 if midpoint_error == 0.0 else float("inf")
    return midpoint_error / spread


class TemporalPatchNet:
    """Learn finite observed paths by repairing adjacent temporal overlaps.

    For fixed input u[t] and initial hidden boundary h[-1], each patch owns
    hidden state h[t] and output y[t]. Its residuals are::

        e[t] = h[t] - A @ tanh(h[t-1]) - B @ u[t]
        r[t] = y[t] - C @ tanh(h[t])

    Energy is half the sum of squared residuals, averaged over batch rows.
    Detuning adds beta/2 times per-output precision-weighted mean squared
    output error (batch/time/ports); supplied precisions default to one.
    Free inference is the causal zero-defect recurrence. Detuned phases repair
    the whole finite path with analytic adjacent-block Newton messages.

    Only ``observe`` changes parameters. It carries the final target-free
    hidden state, including when a detuned phase fails; ``imagine`` changes
    nothing. Past input/target paths are not retained after a call. There is no
    automatic importance, consolidation, replay, planning policy or protection
    against forgetting during continued learning.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        seed: int = 0,
        output_precision: np.ndarray | None = None,
        initial_radius: float = 0.95,
        tolerance: float = 1e-9,
        max_iterations: int = 80,
        max_backtracks: int = 24,
        max_damping_trials: int = 16,
    ) -> None:
        self.inputs = _integer("inputs", inputs, 1)
        self.hidden = _integer("hidden", hidden, 1)
        self.outputs = _integer("outputs", outputs, 1)
        self.set_output_precision(
            np.ones(self.outputs) if output_precision is None else output_precision
        )
        self.max_iterations = _integer("max_iterations", max_iterations, 0)
        self.max_backtracks = _integer("max_backtracks", max_backtracks, 1)
        self.max_damping_trials = _integer("max_damping_trials", max_damping_trials, 1)
        if not np.isfinite(initial_radius) or initial_radius < 0:
            raise ValueError("initial_radius must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        self.tolerance = float(tolerance)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(hidden, hidden)))
        self._A = initial_radius * q
        self._B = rng.normal(size=(hidden, inputs)) / np.sqrt(hidden)
        self._C = rng.normal(size=(outputs, hidden)) / np.sqrt(hidden)
        self._state: np.ndarray | None = None
        self._residual: float | None = None
        self._energy: float | None = None
        self.updates = 0
        self._revision = 0
        self._state_revision: int | None = None

    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    @property
    def output_precision(self) -> np.ndarray:
        """Detached positive per-output teaching weights, not learned parameters."""
        return self._output_precision.copy()

    def set_output_precision(self, precision: np.ndarray) -> None:
        """Atomically change supplied loss geometry, leaving free dynamics intact.

        This does not change parameter revisions or invalidate free-state
        diagnostics: free energy and recurrence do not use teaching precision.
        Protected response constraints remain bound to the unchanged A/B/C.
        """
        value = np.asarray(precision)
        if value.dtype.kind not in "fiu" or value.shape != (self.outputs,):
            raise ValueError("output_precision must be a real vector with one entry per output")
        value = value.astype(float, copy=True)
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("output_precision must be finite and strictly positive")
        self._output_precision = value

    def _detuning_weights(self, beta: float, horizon: int) -> float | np.ndarray:
        scale = beta / (horizon * self.outputs)
        # Preserve the audited default kernel's scalar arithmetic exactly.
        if np.all(self._output_precision == 1.0):
            return scale
        return np.asarray(scale * self._output_precision)

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, "_" + k).copy() for k in ("A", "B", "C")}

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        """Explicitly replace all learned arrays after validating them atomically."""
        expected = {
            "A": (self.hidden, self.hidden),
            "B": (self.hidden, self.inputs),
            "C": (self.outputs, self.hidden),
        }
        if set(parameters) != set(expected):
            raise ValueError("parameters must contain exactly A, B and C")
        values = {}
        for key, shape in expected.items():
            value = np.asarray(parameters[key])
            if value.dtype.kind not in "fiu" or value.shape != shape:
                raise ValueError(f"{key} must be a real array of shape {shape}")
            value = value.astype(float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("parameters must be finite")
            values[key] = value
        self._A, self._B, self._C = values["A"], values["B"], values["C"]
        self._revision += 1

    def _path(self, values: np.ndarray, ports: int, name: str) -> np.ndarray:
        raw = np.asarray(values)
        if (
            raw.dtype.kind not in "fiu"
            or raw.ndim != 3
            or not raw.shape[0]
            or not raw.shape[1]
            or raw.shape[2] != ports
        ):
            raise ValueError(f"{name} must be a real (batch, time, {ports}) array")
        result = raw.astype(float, copy=False)
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def _boundary(self, batch: int, state: np.ndarray | None) -> np.ndarray:
        source = self._state if state is None else state
        if source is None:
            return np.zeros((batch, self.hidden))
        value = np.asarray(source)
        if value.dtype.kind not in "fiu" or value.shape != (batch, self.hidden):
            raise ValueError("state must match (batch, hidden); reset when streams change")
        value = value.astype(float, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("state must be finite")
        return value

    def _causal(self, inputs: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        batch, horizon, _ = inputs.shape
        hidden = np.empty((batch, horizon, self.hidden))
        previous = boundary
        for t in range(horizon):
            hidden[:, t] = np.tanh(previous) @ self._A.T + inputs[:, t] @ self._B.T
            previous = hidden[:, t]
        return hidden

    def _errors(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        output: np.ndarray,
        boundary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        activity = np.tanh(hidden)
        e = hidden - inputs @ self._B.T
        e[:, 0] -= np.tanh(boundary) @ self._A.T
        e[:, 1:] -= activity[:, :-1] @ self._A.T
        return e, output - activity @ self._C.T

    def _path_energy(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        output: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray | None = None,
        beta: float = 0.0,
    ) -> float:
        e, r = self._errors(inputs, hidden, output, boundary)
        values = 0.5 * (np.square(e).sum(axis=(1, 2)) + np.square(r).sum(axis=(1, 2)))
        if beta:
            assert target is not None
            values += (
                0.5 * beta * (self._output_precision * np.square(output - target)).mean(axis=(1, 2))
            )
        return float(values.mean())

    def _state_gradient(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        output: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray | None = None,
        beta: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        first = 1.0 - np.tanh(hidden) ** 2
        e, r = self._errors(inputs, hidden, output, boundary)
        gh, gy = e - first * (r @ self._C), r.copy()
        gh[:, :-1] -= first[:, :-1] * (e[:, 1:] @ self._A)
        if beta:
            assert target is not None
            gy += self._detuning_weights(beta, hidden.shape[1]) * (output - target)
        return gh, gy

    def _reduced_energy(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        output = np.tanh(hidden) @ self._C.T
        e, _ = self._errors(inputs, hidden, output, boundary)
        b = self._detuning_weights(beta, hidden.shape[1])
        q = output - target
        lam = b / (1.0 + b)
        cost = (
            lam * np.square(q).sum(axis=(1, 2))
            if isinstance(b, float)
            else (lam * np.square(q)).sum(axis=(1, 2))
        )
        return np.asarray(0.5 * (np.square(e).sum(axis=(1, 2)) + cost))

    def _reduced_derivatives(
        self,
        inputs: np.ndarray,
        hidden: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch, horizon, width = hidden.shape
        activity = np.tanh(hidden)
        first, second = 1.0 - activity**2, -2.0 * activity * (1.0 - activity**2)
        e, _ = self._errors(inputs, hidden, activity @ self._C.T, boundary)
        b = self._detuning_weights(beta, horizon)
        lam = b / (1.0 + b)
        diagonal = np.broadcast_to(np.eye(width), (batch, horizon, width, width)).copy()
        if isinstance(b, float):
            # Exact compatibility with the all-ones frozen reference kernel.
            qt = (activity @ self._C.T - target) @ self._C
            gradient = e + lam * first * qt
            diagonal += lam * first[:, :, :, None] * (self._C.T @ self._C) * first[:, :, None, :]
            curvature = lam * second * qt
        else:
            qt = ((activity @ self._C.T - target) * lam) @ self._C
            gradient = e + first * qt
            gram = self._C.T @ (np.asarray(lam)[:, None] * self._C)
            diagonal += first[:, :, :, None] * gram * first[:, :, None, :]
            curvature = second * qt
        gradient[:, :-1] -= first[:, :-1] * (e[:, 1:] @ self._A)
        diagonal[:, :-1] += first[:, :-1, :, None] * (self._A.T @ self._A) * first[:, :-1, None, :]
        curvature[:, :-1] -= second[:, :-1] * (e[:, 1:] @ self._A)
        idx = np.arange(width)
        diagonal[:, :, idx, idx] += curvature
        lower = -self._A[None, None, :, :] * first[:, :-1, None, :]
        return gradient, diagonal, lower

    @staticmethod
    def _block_solve(
        diagonal: np.ndarray,
        lower: np.ndarray,
        rhs: np.ndarray,
        damping: float = 0.0,
    ) -> tuple[np.ndarray, float, int]:
        _, horizon, width, _ = diagonal.shape
        inverses: list[np.ndarray] = []
        information: list[np.ndarray] = []
        minimum = np.inf
        eye = np.eye(width)
        for t in range(horizon):
            pivot = diagonal[:, t].copy() + damping * eye
            local_rhs = rhs[:, t].copy()
            if t:
                edge = lower[:, t - 1]
                product = edge @ inverses[-1]
                pivot -= product @ np.swapaxes(edge, -1, -2)
                local_rhs -= np.einsum("bij,bj->bi", product, information[-1])
            pivot = 0.5 * (pivot + np.swapaxes(pivot, -1, -2))
            eigenvalue = float(np.linalg.eigvalsh(pivot).min())
            if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
                raise ArithmeticError("nonpositive local Schur pivot")
            factor = np.linalg.cholesky(pivot)
            inverse = np.linalg.solve(
                np.swapaxes(factor, -1, -2),
                np.linalg.solve(factor, np.broadcast_to(eye, pivot.shape)),
            )
            inverses.append(inverse)
            information.append(local_rhs)
            minimum = min(minimum, eigenvalue)
        solution = np.empty_like(rhs)
        for t in range(horizon - 1, -1, -1):
            local_rhs = information[t].copy()
            if t + 1 < horizon:
                local_rhs -= np.einsum("bji,bj->bi", lower[:, t], solution[:, t + 1])
            solution[:, t] = np.einsum("bij,bj->bi", inverses[t], local_rhs)
        return solution, minimum, sum(x.nbytes for x in inverses + information)

    def _solve(
        self,
        inputs: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray | None,
        beta: float,
    ) -> TemporalPhase:
        with np.errstate(over="ignore", invalid="ignore"):
            hidden = self._causal(inputs, boundary)
            free_output = np.tanh(hidden) @ self._C.T
        if not np.isfinite(hidden).all() or not np.isfinite(free_output).all():
            return TemporalPhase(
                hidden,
                free_output,
                None,
                False,
                "nonfinite_free_state",
                0,
                None,
                0,
                (),
                (),
                (),
                0,
                0,
            )
        if beta == 0.0:
            gh, gy = self._state_gradient(inputs, hidden, free_output, boundary)
            residual = max(float(np.abs(gh).max()), float(np.abs(gy).max()))
            energy = self._path_energy(inputs, hidden, free_output, boundary)
            valid = bool(np.isfinite(residual) and np.isfinite(energy))
            converged = valid and residual <= self.tolerance
            return TemporalPhase(
                hidden,
                free_output,
                residual if np.isfinite(residual) else None,
                converged,
                "exact_free" if converged else "free_residual_failed",
                0,
                None,
                0,
                (energy,) if np.isfinite(energy) else (),
                (),
                (),
                0,
                1,
            )
        assert target is not None
        b = self._detuning_weights(beta, inputs.shape[1])
        current_energy = self._reduced_energy(inputs, hidden, boundary, target, beta)
        if not np.isfinite(current_energy).all():
            return TemporalPhase(
                hidden, free_output, None, False, "nonfinite_energy", 0, None, 0, (), (), (), 0, 1
            )
        energies = [float(current_energy.mean())]
        energy_evaluations = 1
        step_sizes: list[float] = []
        dampings: list[float] = []
        message_bytes, chain_attempts = 0, 0
        minimum: float | None = None
        reason, converged = "iteration_cap", False
        for iteration in range(self.max_iterations + 1):
            gradient, diagonal, lower = self._reduced_derivatives(
                inputs,
                hidden,
                boundary,
                target,
                beta,
            )
            residual = float(np.abs(gradient).max())
            if not np.isfinite(residual):
                reason = "nonfinite_residual"
                break
            if residual <= self.tolerance:
                try:
                    chain_attempts += 1
                    _, minimum, size = self._block_solve(diagonal, lower, -gradient)
                    message_bytes = max(message_bytes, size)
                    converged, reason = True, "converged"
                except (ArithmeticError, np.linalg.LinAlgError):
                    reason = "nonminimum_stationary_point"
                break
            if iteration == self.max_iterations:
                break
            before = current_energy
            damping, accepted = 0.0, False
            for _ in range(self.max_damping_trials):
                chain_attempts += 1
                try:
                    direction, minimum, size = self._block_solve(
                        diagonal,
                        lower,
                        -gradient,
                        damping,
                    )
                    message_bytes = max(message_bytes, size)
                    slopes = np.sum(gradient * direction, axis=(1, 2))
                    if not np.isfinite(direction).all() or np.any(slopes > 0.0):
                        raise ArithmeticError("non-descent direction")
                    step = 1.0
                    for _ in range(self.max_backtracks):
                        proposed = hidden + step * direction
                        after = self._reduced_energy(inputs, proposed, boundary, target, beta)
                        energy_evaluations += 1
                        rounding = 1e-15 * np.maximum(1.0, np.abs(before))
                        if np.isfinite(after).all() and np.all(
                            after <= before + 1e-4 * step * slopes + rounding
                        ):
                            hidden, current_energy = proposed, after
                            energies.append(float(after.mean()))
                            step_sizes.append(step)
                            dampings.append(damping)
                            accepted = True
                            break
                        step *= 0.5
                except (ArithmeticError, np.linalg.LinAlgError):
                    pass
                if accepted:
                    break
                damping = 1e-8 if damping == 0.0 else damping * 10.0
            if not accepted:
                reason = "no_decreasing_spd_step"
                break
        output = (np.tanh(hidden) @ self._C.T + b * target) / (1.0 + b)
        gh, gy = self._state_gradient(inputs, hidden, output, boundary, target, beta)
        residual = max(float(np.abs(gh).max()), float(np.abs(gy).max()))
        converged = bool(converged and np.isfinite(residual) and residual <= self.tolerance)
        if reason == "converged" and not converged:
            reason = "full_residual_failed"
        return TemporalPhase(
            hidden,
            output,
            residual if np.isfinite(residual) else None,
            converged,
            reason,
            len(step_sizes),
            minimum,
            message_bytes,
            tuple(energies),
            tuple(step_sizes),
            tuple(dampings),
            chain_attempts,
            energy_evaluations,
        )

    def settle(
        self,
        inputs: np.ndarray,
        *,
        target: np.ndarray | None = None,
        beta: float = 0.0,
        state: np.ndarray | None = None,
    ) -> TemporalPhase:
        """Read-only diagnostic solve; target is wholly ignored when beta is zero.

        Nonzero detuning does not train parameters or advance live activity.
        Future goal preferences, if used this way, remain supplied constraints.
        """
        path = self._path(inputs, self.inputs, "inputs")
        boundary = self._boundary(len(path), state)
        b = self._detuning_weights(beta, path.shape[1])
        if not np.isfinite(beta) or not np.isfinite(b).all() or np.any(np.asarray(b) <= -1):
            raise ValueError("beta*output_precision/(time*outputs) must be finite and > -1")
        teaching = None
        if beta:
            if target is None:
                raise ValueError("detuning requires explicit teaching values")
            teaching = self._path(target, self.outputs, "target")
            if teaching.shape[:2] != path.shape[:2]:
                raise ValueError("target batch/time dimensions must match inputs")
        return self._solve(path, boundary, teaching, float(beta))

    def imagine(self, inputs: np.ndarray, *, state: np.ndarray | None = None) -> TemporalPhase:
        """Free private rollout; live parameters, state and readback do not change."""
        return self.settle(inputs, state=state)

    def plan(
        self,
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
        """Privately repair action ports using EP contrasts and causal replay.

        The goal is a supplied output path. Boolean ``controls`` selects input
        ports that may change; other inputs and the initial boundary stay fixed.
        Returned predictions are ordinary free rollouts, never target-nudged
        outputs. Planning neither executes the action nor trains on its goal.
        Each contrast must pass the same symmetry check as ``observe``; beta
        halves up to ``max_halvings`` times within one proposal. ``method``
        selects the search direction: ``"steepest"`` (the projected contrast
        scaled by ``rate``) or ``"bfgs"`` (a quasi-Newton direction from the
        accepted steps, with the same line search and causal replay).
        """
        from .planning import plan_inputs

        return plan_inputs(
            self,
            inputs,
            goal=goal,
            controls=controls,
            bounds=bounds,
            state=state,
            beta=beta,
            rate=rate,
            max_steps=max_steps,
            max_backtracks=max_backtracks,
            tolerance=tolerance,
            goal_tolerance=goal_tolerance,
            symmetry_tolerance=symmetry_tolerance,
            max_halvings=max_halvings,
            method=method,
        )

    def _carry(self, phase: TemporalPhase) -> None:
        if phase.converged:
            self._state = phase.final_state
            self._residual, self._energy = phase.residual, phase.energy
            self._state_revision = self._revision

    def advance(self, inputs: np.ndarray) -> TemporalPhase:
        """Advance target-free live activity without learning; reject failed state."""
        phase = self.imagine(inputs)
        self._carry(phase)
        return phase

    def _parameter_gradient(
        self,
        inputs: np.ndarray,
        boundary: np.ndarray,
        phase: TemporalPhase,
    ) -> dict[str, np.ndarray]:
        activity = np.tanh(phase.hidden)
        e, r = self._errors(inputs, phase.hidden, phase.output, boundary)
        previous = np.concatenate((np.tanh(boundary)[:, None], activity[:, :-1]), axis=1)
        batch = len(inputs)
        return {
            "A": -np.einsum("bti,btj->ij", e, previous) / batch,
            "B": -np.einsum("bti,btj->ij", e, inputs) / batch,
            "C": -np.einsum("bti,btj->ij", r, activity) / batch,
        }

    def observe(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        beta: float = 0.01,
        rate: float = 0.1,
        backtrack: bool = False,
        symmetry_tolerance: float = 0.15,
        max_halvings: int = 8,
    ) -> TemporalObservation:
        """Learn one finite external path; all phases share one fixed initial state.

        Targets never enter the free recurrence. Centered local derivatives are
        summed over shared temporal parameter occurrences, then averaged over
        batch rows. The detuning loss averages time and output ports as well.
        Invalid data cause no state change. Failed detuning retains only valid
        free activity and never changes weights or the update count.

        The two detuned paths must sit symmetrically around the free path:
        ``contrast_asymmetry`` at most ``symmetry_tolerance``. Otherwise beta is
        halved and both phases are solved again, up to ``max_halvings`` times.
        A contrast that stays off center is refused with the reason
        ``contrast_asymmetric``; the result reports the beta of its last
        contrast and the number of halvings. An infinite tolerance disables
        the check and reproduces the unguarded update.

        With ``backtrack=True``, test up to sixteen successively halved rates
        against target-free predictions from the original initial state. A
        finite decrease in the observed precision-weighted half-MSE is required
        before committing. This guards the current observation, not old skills
        or unseen data. The default preserves the fixed-rate update.
        """
        if not isinstance(backtrack, (bool, np.bool_)):
            raise ValueError("backtrack must be a boolean")
        path = self._path(inputs, self.inputs, "inputs")
        teaching = self._path(target, self.outputs, "target")
        if teaching.shape[:2] != path.shape[:2]:
            raise ValueError("target batch/time dimensions must match inputs")
        if (
            not np.isfinite(beta)
            or beta <= 0
            or beta >= path.shape[1] * self.outputs / float(self._output_precision.max())
            or not np.isfinite(rate)
            or rate < 0
        ):
            raise ValueError(
                "need 0 < beta*max(output_precision) < time*outputs and finite nonnegative rate"
            )
        if np.isnan(symmetry_tolerance) or symmetry_tolerance <= 0:
            raise ValueError("symmetry_tolerance must be positive; inf disables the check")
        max_halvings = _integer("max_halvings", max_halvings, 0)
        boundary = self._boundary(len(path), None)
        free = self._solve(path, boundary, None, 0.0)
        self._carry(free)
        if not free.converged:
            return TemporalObservation(False, "free_phase_failed", free)
        contrast_beta, halvings = float(beta), 0
        while True:
            plus = self._solve(path, boundary, teaching, contrast_beta)
            minus = self._solve(path, boundary, teaching, -contrast_beta)
            if not plus.converged or not minus.converged:
                return TemporalObservation(
                    False, "phase_failed", free, plus, minus, beta=contrast_beta,
                    contrast_halvings=halvings,
                )
            if contrast_asymmetry(free, plus, minus) <= symmetry_tolerance:
                break
            if halvings == max_halvings:
                return TemporalObservation(
                    False, "contrast_asymmetric", free, plus, minus, beta=contrast_beta,
                    contrast_halvings=halvings,
                )
            halvings += 1
            contrast_beta *= 0.5
        gp = self._parameter_gradient(path, boundary, plus)
        gm = self._parameter_gradient(path, boundary, minus)
        delta = {k: (gp[k] - gm[k]) / (2.0 * contrast_beta) for k in gp}
        if backtrack:
            return self._backtracked_observation(
                path, teaching, boundary, free, plus, minus, delta, rate, contrast_beta, halvings
            )
        proposed = {k: getattr(self, "_" + k) - rate * delta[k] for k in delta}
        if not all(np.isfinite(p).all() for p in proposed.values()):
            return TemporalObservation(
                False, "nonfinite_update", free, plus, minus, delta, beta=contrast_beta,
                contrast_halvings=halvings,
            )
        self._A, self._B, self._C = proposed["A"], proposed["B"], proposed["C"]
        self.updates += 1
        self._revision += 1
        return TemporalObservation(
            True, "updated", free, plus, minus, delta, beta=contrast_beta,
            contrast_halvings=halvings,
        )

    def _backtracked_observation(
        self,
        path: np.ndarray,
        target: np.ndarray,
        boundary: np.ndarray,
        free: TemporalPhase,
        plus: TemporalPhase,
        minus: TemporalPhase,
        delta: dict[str, np.ndarray],
        rate: float,
        beta: float,
        halvings: int,
    ) -> TemporalObservation:
        """Replay candidate parameters privately; never carry a trial's activity."""

        def loss(output: np.ndarray) -> float | None:
            with np.errstate(over="ignore", invalid="ignore"):
                value = float(0.5 * np.mean(self._output_precision * (output - target) ** 2))
            return value if np.isfinite(value) else None

        initial = loss(free.output)
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = sum(float(np.sum(value * value)) for value in delta.values())
        losses: list[float | None] = []
        replays = 0
        if initial is not None and np.isfinite(norm_squared) and norm_squared > 0 and rate > 0:
            snapshot = self.snapshot()
            for index in range(16):
                step = rate * 0.5**index
                with np.errstate(over="ignore", invalid="ignore"):
                    proposed = {k: snapshot[k] - step * delta[k] for k in delta}
                if not all(np.isfinite(value).all() for value in proposed.values()):
                    losses.append(None)
                    continue
                candidate = TemporalPatchNet.restore(snapshot)
                candidate.set_parameters(proposed)
                prediction = candidate.imagine(path, state=boundary)
                replays += 1
                current = loss(prediction.output) if prediction.converged else None
                losses.append(current)
                floor = (
                    64
                    * np.finfo(float).eps
                    * max(abs(initial), abs(current or 0.0), np.finfo(float).tiny)
                )
                if (
                    current is not None
                    and current < initial - floor
                    and current <= initial - 1e-4 * step * norm_squared
                ):
                    self._A, self._B, self._C = proposed["A"], proposed["B"], proposed["C"]
                    self.updates += 1
                    self._revision += 1
                    return TemporalObservation(
                        True,
                        "updated",
                        free,
                        plus,
                        minus,
                        delta,
                        initial,
                        current,
                        step,
                        tuple(losses),
                        replays,
                        beta,
                        halvings,
                    )
        return TemporalObservation(
            False,
            "no_decreasing_parameter_step",
            free,
            plus,
            minus,
            delta,
            initial,
            initial,
            0.0,
            tuple(losses),
            replays,
            beta,
            halvings,
        )

    def reset(self) -> None:
        """Clear active state/readback; retain learned parameters and update count."""
        self._state = None
        self._residual = self._energy = None
        self._state_revision = None

    def readback(self) -> TemporalReadback:
        """Expose detached current activity and measured free-phase diagnostics."""
        state = self.state
        return TemporalReadback(
            state,
            None if state is None else np.tanh(state),
            self._residual,
            self._energy,
            self.updates,
            self._revision,
            self._state_revision,
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        """Detached complete continuation state, without any retained training path."""
        meta = {
            "format": "cadence-temporal/1",
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "max_backtracks": self.max_backtracks,
            "max_damping_trials": self.max_damping_trials,
            "updates": self.updates,
            "parameter_revision": self._revision,
            "state_parameter_revision": self._state_revision,
            "residual": self._residual,
            "energy": self._energy,
        }
        return {
            **self.parameters(),
            "output_precision": self.output_precision,
            "state": np.empty((0, self.hidden)) if self._state is None else self._state.copy(),
            "meta": np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> TemporalPatchNet:
        """Construct a fresh net from validated complete state; never load pickle."""
        fields = {"A", "B", "C", "state", "meta"}
        if set(snapshot) not in (fields, fields | {"output_precision"}):
            raise ValueError("invalid temporal snapshot fields")
        try:
            meta = json.loads(str(snapshot["meta"]))
            if not isinstance(meta, dict) or meta.get("format") != "cadence-temporal/1":
                raise ValueError("unsupported temporal checkpoint")
            options = {
                k: meta[k]
                for k in (
                    "inputs",
                    "hidden",
                    "outputs",
                    "tolerance",
                    "max_iterations",
                    "max_backtracks",
                    "max_damping_trials",
                )
            }
            result = cls(**options, output_precision=snapshot.get("output_precision"))
            result.set_parameters({k: snapshot[k] for k in ("A", "B", "C")})
            state = np.asarray(snapshot["state"])
            if state.ndim != 2 or state.shape[1] != result.hidden:
                raise ValueError("invalid saved state shape")
            result._state = None if not len(state) else result._boundary(len(state), state)
            result.updates = _integer("updates", meta["updates"], 0)
            result._revision = _integer("parameter_revision", meta["parameter_revision"], 0)
            revision = meta["state_parameter_revision"]
            result._state_revision = (
                None
                if revision is None
                else _integer(
                    "state_parameter_revision",
                    revision,
                    0,
                )
            )
            if (result._state is None) != (result._state_revision is None) or (
                result._state_revision is not None and result._state_revision > result._revision
            ):
                raise ValueError("invalid saved state revision")
            for name in ("residual", "energy"):
                value = meta[name]
                if value is not None and (not np.isfinite(value) or value < 0):
                    raise ValueError("invalid free-state diagnostics")
                if (result._state is None) != (value is None):
                    raise ValueError("diagnostics must accompany saved state")
                setattr(result, "_" + name, value)
            return result
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid temporal checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path) -> TemporalPatchNet:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})
