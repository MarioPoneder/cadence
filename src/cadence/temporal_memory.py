"""Compressed local response constraints for explicitly protected experiences.

This is an exact-arithmetic nullspace construction, related to orthogonal weight
modification. A caller selects which learned responses to preserve. It does not
discover importance, certify truth, or supply unlimited plastic capacity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .temporal import TemporalObservation, TemporalPatchNet

__all__ = ["TemporalMemory", "ConstraintReport"]


@dataclass(frozen=True)
class ConstraintReport:
    """Numerical constraint residuals and consumed presynaptic dimensions."""

    ranks: dict[str, int]
    bytes: int
    maximum_residual: float


class TemporalMemory:
    """Keep bounded sufficient subspaces of protected local response relations.

    For every protected presynaptic vector x, require (W_new-W_old) x = 0.
    The closest admissible update to D in Frobenius norm is D(I-QQ.T), where
    Q is an orthonormal basis for those vectors. This is the normal form of a
    quadratic parameter-overlap constraint; it stores neither target phrases
    nor a runtime lookup table. A/B/C responses along protected causal paths
    are consequently unchanged in exact arithmetic by induction on time.

    Protection is explicit and applies to the supplied initial boundary and
    inputs. It is not protection for every nearby cue or arbitrary task. Once
    a basis spans its input space that connection block cannot change. There
    is no automatic relevance selection, decay, contradiction resolution or
    guarantee of retaining outputs after an unconstrained parameter change.
    """

    def __init__(self, *, relative_tolerance: float = 1e-12) -> None:
        if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
            raise ValueError("relative_tolerance must lie in (0,1)")
        self.relative_tolerance = float(relative_tolerance)
        self._basis: dict[str, np.ndarray] = {}
        self._binding: np.ndarray | None = None

    @staticmethod
    def _fingerprint(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
        digest = hashlib.sha256()
        for key in sorted(parameters):
            value = np.asarray(parameters[key], dtype="<f8")
            digest.update(key.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return np.frombuffer(digest.digest(), dtype=np.uint8).copy()

    def _check(self, parameters: Mapping[str, np.ndarray]) -> None:
        if self._binding is not None and not np.array_equal(
            self._binding, self._fingerprint(parameters)
        ):
            raise ValueError("parameters changed outside this memory's admitted update path")

    def protect(
        self,
        net: TemporalPatchNet,
        inputs: np.ndarray,
        *,
        state: np.ndarray | None = None,
    ) -> ConstraintReport:
        """Protect a read-only free path; caller supplies its importance decision.

        Default boundary is the live state's value (zero when uninitialized).
        Retain that boundary separately if the protected query must be replayed
        after clearing activity. Targets, hidden paths and observations are
        discarded after their presynaptic subspaces have been accumulated.
        """
        parameters = net.parameters()
        self._check(parameters)
        path = np.asarray(inputs, dtype=float)
        phase = net.imagine(path, state=state)
        if not phase.converged:
            raise ValueError("cannot protect an unconverged path")
        boundary = net.state if state is None else np.asarray(state, dtype=float)
        if boundary is None:
            boundary = np.zeros((len(path), parameters["A"].shape[1]))
        activity = np.tanh(phase.hidden)
        previous = np.concatenate((np.tanh(boundary)[:, None], activity[:, :-1]), axis=1)
        rows = {"A": previous, "B": path, "C": activity}
        proposed = {}
        maximum = 0.0
        for key, values in rows.items():
            vectors = values.reshape(-1, values.shape[-1]).T
            old = self._basis.get(key, np.empty((len(vectors), 0)))
            # Previously admitted directions must never be discarded when a
            # newly protected input has a much larger scale.
            remainder = vectors - old @ (old.T @ vectors)
            remainder -= old @ (old.T @ remainder)
            u, singular, _ = np.linalg.svd(remainder, full_matrices=False)
            # Squaring a finite 1e200 vector overflows and would make the
            # threshold infinite, silently discarding its protected direction.
            magnitude = float(np.max(np.abs(vectors), initial=0.0))
            threshold = (
                self.relative_tolerance * magnitude * float(np.linalg.norm(vectors / magnitude))
                if magnitude else 0.0
            )
            rank = min(len(vectors) - old.shape[1], int(np.count_nonzero(singular > threshold)))
            new = u[:, :rank]
            if rank:
                new -= old @ (old.T @ new)
                new -= old @ (old.T @ new)
                new, _ = np.linalg.qr(new, mode="reduced")
            basis = np.column_stack((old, new))
            proposed[key] = basis
            maximum = max(maximum, float(np.max(np.abs(vectors - basis @ (basis.T @ vectors)))))
        self._basis = proposed
        self._binding = self._fingerprint(parameters)
        return self.report(maximum)

    def project(
        self,
        before: Mapping[str, np.ndarray],
        proposed: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return a constrained update; no implicit learning or net mutation.

        The caller applies the result once using net.set_parameters(). This
        memory advances its expected parameter binding to that result. If
        applying it fails, restore the earlier memory snapshot before retrying.
        A rejected net.observe call should not be submitted as a new update.
        """
        self._check(before)
        if set(before) != {"A", "B", "C"} or set(proposed) != set(before):
            raise ValueError("A, B and C parameter mappings required")
        result = {}
        bases = {}
        for key, old in before.items():
            old = np.asarray(old)
            if old.ndim != 2 or old.dtype.kind not in "fiu" or not np.isfinite(old).all():
                raise ValueError("finite real parameter matrices required")
            new = np.asarray(proposed[key], dtype=float)
            if new.shape != old.shape or not np.isfinite(new).all():
                raise ValueError("finite matching proposed parameter shapes required")
            delta = new - old
            basis = self._basis.get(key, np.empty((old.shape[1], 0)))
            if basis.shape[0] != old.shape[1]:
                raise ValueError("constraint dimensions do not match parameters")
            bases[key] = basis.copy()
            if basis.shape[1]:
                delta -= (delta @ basis) @ basis.T
                # A second projection removes floating-point component left
                # by the first dense product; it changes no mathematical rule.
                delta -= (delta @ basis) @ basis.T
            result[key] = old + delta
            if not np.isfinite(result[key]).all():
                raise ValueError("nonfinite constrained update")
        self._basis = bases
        self._binding = self._fingerprint(result)
        return result

    def observe(
        self,
        net: TemporalPatchNet,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        beta: float = 0.01,
        rate: float = 0.1,
        readout_damping: float | None = None,
        symmetry_tolerance: float = 0.15,
        max_halvings: int = 8,
    ) -> TemporalObservation:
        """Stage EP and protected projection, then commit both objects together.

        Invalid data or projection errors change neither object. Failed phases
        carry only the valid free activity, as TemporalPatchNet.observe does,
        and leave the memory binding unchanged. Returned delta is the raw EP
        derivative before projection; committed weights obey the constraints.
        This single-threaded transaction does not promise concurrent mutation
        safety. Subclasses require their own complete state transaction.

        Positive readout_damping opts into a local C update metric. With free
        activities X, protected basis Q, P=I-QQ.T and S=X.T@X/(batch*time), use
        -rate*delta_C*P*(P*S*P+readout_damping*I)^-1*P; A/B stay ordinary EP.
        S omits the loss's output-count divisor and teaching precisions: it
        is an explicitly selected geometry, not the full loss Hessian. No
        optimizer state or old examples are retained. Default None preserves
        the original update path.

        The metric candidate must pass a protected-map residual check and one
        actual free replay from the ORIGINAL boundary. Weighted half-MSE must
        decrease by more than 64*eps*max(abs(old_loss),abs(new_loss),tiny).
        Ties and increases return metric_step_rejected, carry only the valid
        original free activity, and keep weights, counters and memory intact.
        This is not a phase failure or evidence of stationarity. Returned phase
        counters exclude this one additional replay and the hidden-width
        metric solve. Beta halves under the symmetry check of
        ``TemporalPatchNet.observe``; there is no parameter line search.
        """
        if type(net) is not TemporalPatchNet:
            raise TypeError("atomic protected learning requires TemporalPatchNet")
        if readout_damping is not None and (
            not np.isfinite(readout_damping) or readout_damping <= 0
        ):
            raise ValueError("readout_damping must be finite and positive")
        before = net.parameters()
        self._check(before)
        candidate = TemporalPatchNet.restore(net.snapshot())
        pending = TemporalMemory.restore(self.snapshot())
        result = candidate.observe(
            inputs,
            target,
            beta=beta,
            rate=rate,
            symmetry_tolerance=symmetry_tolerance,
            max_halvings=max_halvings,
        )
        if result.updated:
            if readout_damping is None:
                candidate.set_parameters(pending.project(before, candidate.parameters()))
            elif not self._metric_candidate(
                net, candidate, pending, before, result, inputs, target, rate, readout_damping
            ):
                # No staged weight/counter/binding mutation crosses rejection.
                candidate = TemporalPatchNet.restore(net.snapshot())
                candidate._carry(result.free)
                pending = TemporalMemory.restore(self.snapshot())
                result = TemporalObservation(
                    False,
                    "metric_step_rejected",
                    result.free,
                    result.plus,
                    result.minus,
                    result.delta,
                )
        # Every fallible calculation/validation precedes these assignments.
        net.__dict__.update(candidate.__dict__)
        self._basis, self._binding = pending._basis, pending._binding
        return result

    def _metric_candidate(
        self,
        original: TemporalPatchNet,
        candidate: TemporalPatchNet,
        pending: TemporalMemory,
        before: dict[str, np.ndarray],
        observation: TemporalObservation,
        inputs: np.ndarray,
        target: np.ndarray,
        rate: float,
        damping: float,
    ) -> bool:
        """Private staged metric and causal acceptance; never writes live objects."""
        assert observation.delta is not None
        q = self._basis.get("C", np.empty((original.hidden, 0)))
        activity = np.tanh(observation.free.hidden).reshape(-1, original.hidden)

        def project(rows: np.ndarray) -> np.ndarray:
            result = rows.copy()
            for _ in range(2):
                result -= (result @ q) @ q.T
            return result

        gradient = project(observation.delta["C"])
        innovation = project(activity)
        metric = innovation.T @ innovation / len(activity) + damping * np.eye(original.hidden)
        try:
            step = project(-rate * np.linalg.solve(metric, gradient.T).T)
        except np.linalg.LinAlgError:
            return False
        proposed = candidate.parameters()
        proposed["C"] = before["C"] + step
        if not all(np.isfinite(value).all() for value in proposed.values()):
            return False
        projected = pending.project(before, proposed)
        # Scale by the original local responses, never by a huge new step.
        for key, previous in before.items():
            basis = self._basis.get(key, np.empty((previous.shape[1], 0)))
            error = float(np.max(np.abs((projected[key] - previous) @ basis), initial=0.0))
            scale = max(1.0, float(np.max(np.abs(previous @ basis), initial=0.0)))
            if not np.isfinite(error) or not np.isfinite(scale) or error > 1e-12 * scale:
                return False
        candidate.set_parameters(projected)
        boundary = original.state
        if boundary is None:
            boundary = np.zeros((len(observation.free.hidden), original.hidden))
        replay = candidate.imagine(inputs, state=boundary)
        if not replay.converged:
            return False
        precision = original.output_precision
        teaching = np.asarray(target, dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            old_loss = float(
                0.5 * np.mean(precision * np.square(observation.free.output - teaching))
            )
            new_loss = float(0.5 * np.mean(precision * np.square(replay.output - teaching)))
        if not np.isfinite(old_loss) or not np.isfinite(new_loss):
            return False
        allowance = 64 * np.finfo(float).eps * max(
            abs(old_loss), abs(new_loss), np.finfo(float).tiny
        )
        return bool(new_loss < old_loss - allowance)

    def report(self, maximum_residual: float = 0.0) -> ConstraintReport:
        return ConstraintReport(
            {key: value.shape[1] for key, value in self._basis.items()},
            8
            + sum(a.nbytes for a in self._basis.values())
            + (32 if self._binding is not None else 0),
            maximum_residual,
        )

    def snapshot(self) -> dict[str, np.ndarray]:
        """Detached matrices and a 32-byte parameter binding (counted)."""
        return {
            "relative_tolerance": np.array([self.relative_tolerance]),
            **{"basis_" + k: v.copy() for k, v in self._basis.items()},
            "binding": np.empty(0, dtype=np.uint8)
            if self._binding is None
            else self._binding.copy(),
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> TemporalMemory:
        """Validate and restore a constraint boundary without executable payloads."""
        if not {"relative_tolerance", "binding"}.issubset(snapshot):
            raise ValueError("missing constraint checkpoint fields")
        tolerance = np.asarray(snapshot["relative_tolerance"])
        if tolerance.shape != (1,):
            raise ValueError("invalid tolerance shape")
        net = cls(relative_tolerance=float(tolerance[0]))
        keys = set(snapshot) - {"relative_tolerance", "binding"}
        binding = np.asarray(snapshot["binding"])
        if not keys:
            if binding.shape != (0,):
                raise ValueError("unexpected binding without constraints")
            return net
        if (
            keys != {"basis_" + k for k in "ABC"}
            or binding.shape != (32,)
            or binding.dtype != np.uint8
        ):
            raise ValueError("incomplete constraint checkpoint")
        for key in "ABC":
            basis = np.asarray(snapshot["basis_" + key], dtype=float)
            if (
                basis.ndim != 2
                or not np.isfinite(basis).all()
                or basis.shape[0] < 1
                or basis.shape[1] > basis.shape[0]
                or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10, rtol=0)
            ):
                raise ValueError("invalid constraint matrices")
            net._basis[key] = basis.copy()
        net._binding = binding.copy()
        return net
