"""A temporal patch with a slow linear context and a fast record inside it.

The state transition is linear in the context and gated by the input; the
nonlinearity sits at the ports. The energy is quadratic in the context path;
detuned minima are unique when their checked reduced Hessians are positive definite.
The centered detuning contrast approaches the adjoint (reverse-mode) gradient
of the same loss in the small-detuning limit. ``observe`` computes that
gradient by one backward scan; ``detune`` solves the two detuned equilibria and
returns the contrast, as the acceptance check of the identity. The slow
parameters learn the observation itself; the record, read at every moment
through an ordinary port, patches the slow readout's current error and is
written one-shot by the delta rule after the observed path. The energy
describes the slow patch; neither learning route sees the record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ports import StructuredPort
from .records import Records

__all__ = ["RecordPatchNet", "RecordPath", "RecordObservation", "RecordContrast", "RecordReadback"]

_PARAMETERS = ("G", "g", "B", "b", "C", "c")
FORMAT = "cadence-record-patch/2"
FORMAT_PORTS = "cadence-record-patch/3"  # categorical ports or batched writes


@dataclass(frozen=True)
class RecordPath:
    """A detached free path; ``hidden`` is the bounded context (batch, time, hidden).

    ``output`` is the prediction, the slow readout plus the record read;
    ``loss`` scores it. ``slow_loss`` scores the slow readout alone, which is
    what the slow parameters learn and what admission checks."""

    hidden: np.ndarray
    output: np.ndarray
    gate: np.ndarray
    read: np.ndarray
    loss: float | None
    slow_loss: float | None

    @property
    def slow_output(self) -> np.ndarray:
        return np.asarray(self.output - self.read)

    @property
    def final_state(self) -> np.ndarray:
        return self.hidden[:, -1].copy()


@dataclass(frozen=True)
class RecordObservation:
    """One teaching attempt: the prediction made before any write, and what changed.

    The losses and rates concern the slow readout, whose step was admitted;
    ``prediction.loss`` scores the prediction with the record read."""

    updated: bool
    reason: str
    prediction: RecordPath
    delta: dict[str, np.ndarray] | None = None
    initial_loss: float | None = None
    final_loss: float | None = None
    accepted_rate: float | None = None
    replay_losses: tuple[float | None, ...] = ()
    replay_calls: int = 0
    writes: int = 0


@dataclass(frozen=True)
class RecordContrast:
    """The two detuned equilibria of the quadratic energy and their centered contrast."""

    contrast: dict[str, np.ndarray]
    plus_hidden: np.ndarray
    minus_hidden: np.ndarray
    plus_energy: float
    minus_energy: float
    iterations: tuple[int, int]
    residuals: tuple[float, float]
    converged: bool


@dataclass(frozen=True)
class RecordReadback:
    """Read-only self-observation; never automatically admitted as teaching data."""

    state: np.ndarray | None
    updates: int
    writes: int
    parameter_revision: int
    state_parameter_revision: int | None
    record_entries: int


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * x)))


def _input_norms(inputs: np.ndarray) -> np.ndarray:
    """Finite row norms without squaring a large unscaled observation."""
    rows = inputs.reshape(-1, inputs.shape[-1])
    magnitude = np.max(np.abs(rows), axis=1)
    scaled = rows / np.where(magnitude > 0, magnitude, 1.0)[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        norms = magnitude * np.linalg.norm(scaled, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("input norms must be representable and finite")
    return norms


class RecordPatchNet:
    """Learn observed streams with a gated linear context and one-shot records.

    For inputs ``u[t]``, context ``h[t]`` and outputs ``y[t]``::

        s[t] = g + G u[t]                       gate preactivation
        l[t] = sigmoid(s[t])                    per-channel retention in (0, 1)
        z[t] = tanh(B u[t] + b)                 the input port
        h[t] = l[t] * h[t-1] + (1 - l[t]) * z[t]
        m[t] = read(code([u[t], r * h[t]]))     the record read, a port value
        y[t] = C h[t] + c + m[t]

    ``r`` is the fixed per-channel unit of a context deviation. The context
    is a convex combination, so ``|h| <= max(|h[-1]|, 1)``. The
    seam residual ``h[t] - l[t] h[t-1] - (1 - l[t]) z[t]`` is linear in the
    context path: the energy is quadratic and the free path is its unique
    zero-defect normal form.

    The slow parameters learn the observation itself: their gradient is
    that of the slow readout ``C h[t] + c`` against the target, and the
    record read never enters it. A record holds what the slow readout got
    wrong at its reading: the write target is ``y_obs[t] - C h[t] - c``. A
    reading the slow parameters have learned leaves a record near zero, so
    the store holds only what the slow model does not yet know, and the
    prediction ``y[t]`` is right as soon as either knows it. Records are read
    with the tables as they stood when the call began and written after the
    path, so a prediction is a function of the parameters, the records and
    the boundary alone. To let a record written at one moment inform the
    next, observe in shorter paths.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        seed: int = 0,
        output_precision: np.ndarray | None = None,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        slowest: float = 128.0,
        record_averaging: bool = False,
        record_homeostasis: float = 0.0,
        groups: Sequence[int] | None = None,
        record_writes: str = "sequential",
        record_width: int | None = None,
        port: StructuredPort | None = None,
    ) -> None:
        self.inputs = _integer("inputs", inputs, 1)
        # A structured port (tied local kernels over grids of the inputs, dense blocks
        # elsewhere) fixes the context width to its outputs; ``hidden`` must agree.
        self.port = port
        if port is not None:
            if port.inputs != self.inputs:
                raise ValueError("the port reads a different number of inputs")
            if int(hidden) != port.outputs:
                raise ValueError(f"hidden must equal the port's outputs ({port.outputs})")
        self.hidden = _integer("hidden", hidden, 1)
        self.outputs = _integer("outputs", outputs, 1)
        if groups is None:
            self.groups: tuple[int, ...] | None = None
        else:
            self.groups = tuple(_integer("groups entry", g, 2) for g in groups)
            if sum(self.groups) != self.outputs:
                raise ValueError("groups must partition the outputs")
        if record_writes not in ("sequential", "batch"):
            raise ValueError("record_writes is 'sequential' or 'batch'")
        self.record_writes = record_writes
        # A wide port need not give the store one column per output: the cells can hold a
        # fixed random code of the residual, decoded by the transpose of the same code.
        self.record_width = (
            None if record_width is None else _integer("record_width", record_width, 1)
        )
        if self.record_width is None:
            self._output_code: np.ndarray | None = None
        else:
            signs = np.random.default_rng([seed, 0x5EED]).integers(
                0, 2, (self.outputs, self.record_width)
            )
            self._output_code = (2.0 * signs - 1.0) / np.sqrt(self.record_width)
        self.set_output_precision(
            np.ones(self.outputs) if output_precision is None else output_precision
        )
        if not np.isfinite(slowest) or slowest < 2.0:
            raise ValueError("slowest must be a finite timescale >= 2")
        self.slowest = float(slowest)
        rng = np.random.default_rng(seed)
        # Retention timescales log-spaced from two moments to ``slowest`` moments.
        timescales = np.exp(np.linspace(np.log(2.0), np.log(self.slowest), self.hidden))
        retention = 1.0 - 1.0 / timescales
        self._g = np.log(retention / (1.0 - retention))
        # A channel averaging white drive of unit variance fluctuates with standard
        # deviation sqrt((1 - l) / (1 + l)); the record reads each channel in that unit,
        # so a slow channel's small deviations address records as well as a fast one's.
        self._scale = np.sqrt((1.0 + retention) / (1.0 - retention))
        if self.port is None:
            self._G = np.zeros((self.hidden, self.inputs))
            self._B = rng.normal(size=(self.hidden, self.inputs)) / np.sqrt(self.inputs)
        else:
            self._G = self._pack(self.port.initial(rng, 0.0))
            self._B = self._pack(self.port.initial(rng, 1.0))
        self._b = np.zeros(self.hidden)
        self._C = rng.normal(size=(self.outputs, self.hidden)) / np.sqrt(self.hidden)
        self._c = np.zeros(self.outputs)
        self.records = Records(
            self.inputs + self.hidden,
            {"y": self.outputs if self.record_width is None else self.record_width},
            cells=cells,
            active=active,
            rate=record_rate,
            habituation=habituation,
            bias=record_bias,
            seed=seed,
            averaging=record_averaging,
            homeostasis=record_homeostasis,
        )
        # The reading gives the input block and the context block comparable say: the
        # context is read in units of each channel's fluctuation, and the input block is
        # scaled by its witnessed rms norm to unit variance per unit, so the drive of a
        # cell has unit scale and its fixed offset stays a small preference.
        self._input_norm = 1.0
        self._state: np.ndarray | None = None
        self.updates = 0
        self._revision = 0
        self._state_revision: int | None = None

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    @property
    def output_precision(self) -> np.ndarray:
        """Detached positive per-output teaching weights, not learned parameters."""
        return self._output_precision.copy()

    def set_output_precision(self, precision: np.ndarray) -> None:
        """Atomically change the supplied loss geometry; the free path does not use it."""
        value = np.asarray(precision)
        if value.dtype.kind not in "fiu" or value.shape != (self.outputs,):
            raise ValueError("output_precision must be a real vector with one entry per output")
        value = value.astype(float, copy=True)
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("output_precision must be finite and strictly positive")
        self._output_precision = value

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, "_" + k).copy() for k in _PARAMETERS}

    # ------------------------------------------------------------- the input port
    def _pack(self, blocks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([b.ravel() for b in blocks])

    def _unpack(self, flat: np.ndarray) -> list[np.ndarray]:
        assert self.port is not None
        out, at = [], 0
        for b in self.port.blocks:
            shape = self.port.weight_shape(b)
            size = int(np.prod(shape))
            out.append(flat[at : at + size].reshape(shape))
            at += size
        return out

    def _drive(self, inputs: np.ndarray, which: str) -> np.ndarray:
        """``inputs @ B.T`` or ``inputs @ G.T``, through the port when there is one."""
        weights = self._B if which == "B" else self._G
        if self.port is None:
            return np.asarray(inputs @ weights.T)
        return self.port.apply(inputs, self._unpack(weights))

    def _drive_transpose(self, v: np.ndarray, which: str) -> np.ndarray:
        weights = self._B if which == "B" else self._G
        if self.port is None:
            return np.asarray(v @ weights)
        return self.port.transpose(v, self._unpack(weights))

    def _drive_gradient(self, v: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """The gradient of ``sum(v * (W inputs))`` with respect to ``W`` in its stored shape."""
        if self.port is None:
            return np.asarray(np.einsum("bti,btj->ij", v, inputs))
        return self._pack(self.port.gradient(v, inputs))

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        port_shape = (
            None
            if self.port is None
            else (int(sum(int(np.prod(self.port.weight_shape(b))) for b in self.port.blocks)),)
        )
        return {
            "G": (self.hidden, self.inputs) if port_shape is None else port_shape,
            "g": (self.hidden,),
            "B": (self.hidden, self.inputs) if port_shape is None else port_shape,
            "b": (self.hidden,),
            "C": (self.outputs, self.hidden),
            "c": (self.outputs,),
        }

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        """Explicitly replace all learned arrays after validating them atomically."""
        expected = self._shapes()
        if set(parameters) != set(expected):
            raise ValueError(f"parameters must contain exactly {', '.join(_PARAMETERS)}")
        values = {}
        for key, shape in expected.items():
            value = np.asarray(parameters[key])
            if value.dtype.kind not in "fiu" or value.shape != shape:
                raise ValueError(f"{key} must be a real array of shape {shape}")
            value = value.astype(float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("parameters must be finite")
            values[key] = value
        for key, value in values.items():
            setattr(self, "_" + key, value)
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

    def _teaching(self, inputs: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path = self._path(inputs, self.inputs, "inputs")
        teaching = self._path(target, self.outputs, "target")
        if teaching.shape[:2] != path.shape[:2]:
            raise ValueError("target batch/time dimensions must match inputs")
        if self.groups is not None:
            for start, end in self._bounds():
                block = teaching[..., start:end]
                if np.any(block < 0) or not np.allclose(block.sum(axis=-1), 1.0, atol=1e-9):
                    raise ValueError("a categorical target is a distribution over each group")
        return path, teaching

    # --------------------------------------------------------------- forward

    def _readings(self, inputs: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """What the record reads at each moment: the input and the scaled context, both
        with unit variance per unit."""
        return np.concatenate(
            (inputs * (np.sqrt(self.inputs) / self._input_norm), hidden * self._scale), axis=-1
        )

    def _forward(
        self, inputs: np.ndarray, boundary: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The free path: context, gate, port, code, read and output at every moment.

        The context does not depend on the reads, so the scan runs first and
        every moment's reading is coded in one batched projection."""
        batch, horizon, _ = inputs.shape
        gate = _sigmoid(self._g + self._drive(inputs, "G"))
        port = np.tanh(self._drive(inputs, "B") + self._b)
        hidden = np.empty((batch, horizon, self.hidden))
        previous = boundary
        for t in range(horizon):
            hidden[:, t] = gate[:, t] * previous + (1.0 - gate[:, t]) * port[:, t]
            previous = hidden[:, t]
        readings = self._readings(inputs, hidden).reshape(batch * horizon, -1)
        codes = self.records.code(readings, valued=False)[0].reshape(batch, horizon, -1)
        read = codes @ self.records.tables["y"]
        if self._output_code is not None:
            read = read @ self._output_code.T
        output = self._slow(hidden) + read
        return hidden, gate, port, codes, read, output

    def _bounds(self) -> list[tuple[int, int]]:
        assert self.groups is not None
        ends = np.cumsum(self.groups)
        return [(int(end - size), int(end)) for size, end in zip(self.groups, ends, strict=True)]

    def _slow(
        self, hidden: np.ndarray, C: np.ndarray | None = None, c: np.ndarray | None = None
    ) -> np.ndarray:
        """The slow readout: linear, or one softmax per group of categorical ports. ``C`` and
        ``c`` read out under proposed parameters (a replay) without touching the patch."""
        drive = hidden @ (self._C if C is None else C).T + (self._c if c is None else c)
        if self.groups is None:
            return np.asarray(drive)
        for start, end in self._bounds():
            block = drive[..., start:end]
            block = np.exp(block - block.max(axis=-1, keepdims=True))
            drive[..., start:end] = block / block.sum(axis=-1, keepdims=True)
        return np.asarray(drive)

    def _group_precision(self) -> np.ndarray:
        """One teaching weight per categorical group: the mean precision of its ports."""
        return np.array([self._output_precision[a:b].mean() for a, b in self._bounds()])

    def _loss(self, output: np.ndarray, target: np.ndarray) -> float | None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.groups is None:
                value = float(0.5 * np.mean(self._output_precision * (output - target) ** 2))
            else:
                # Cross-entropy per group; a prediction with a record read is clipped and
                # renormalised first, because a read can carry a port outside [0, 1].
                total = 0.0
                weights = self._group_precision()
                for weight, (start, end) in zip(weights, self._bounds(), strict=True):
                    p = np.clip(output[..., start:end], 1e-12, None)
                    p = p / p.sum(axis=-1, keepdims=True)
                    surprise = -np.sum(target[..., start:end] * np.log(p), axis=-1)
                    total += weight * float(np.mean(surprise))
                value = total / len(self.groups)
        return value if np.isfinite(value) else None

    def _free(
        self, inputs: np.ndarray, boundary: np.ndarray, target: np.ndarray | None
    ) -> tuple[RecordPath, np.ndarray, np.ndarray]:
        hidden, gate, port, codes, read, output = self._forward(inputs, boundary)
        loss = None if target is None else self._loss(output, target)
        slow = None if target is None else self._loss(output - read, target)
        return RecordPath(hidden, output, gate, read, loss, slow), port, codes

    def _gradient(
        self,
        inputs: np.ndarray,
        boundary: np.ndarray,
        target: np.ndarray,
        path: RecordPath,
        port: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """The adjoint scan: dL/dtheta for L = 1/2 mean(w (C h + c - target)^2), or for the
        mean cross-entropy of categorical ports, whose output error is ``softmax - target``."""
        d = self._output_error(path, target)
        return self._adjoint(inputs, boundary, path, port, d)[0]

    def _output_error(self, path: RecordPath, target: np.ndarray) -> np.ndarray:
        batch, horizon, _ = target.shape
        if self.groups is None:
            d = self._output_precision * (path.slow_output - target)
            d = d / (batch * horizon * self.outputs)
        else:
            d = path.slow_output - target
            for weight, (start, end) in zip(self._group_precision(), self._bounds(), strict=True):
                d[..., start:end] *= weight
            d = d / (batch * horizon * len(self.groups))
        return np.asarray(d)

    def _adjoint(
        self,
        inputs: np.ndarray,
        boundary: np.ndarray,
        path: RecordPath,
        port: np.ndarray,
        d: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """One backward scan from the output error ``d``: the parameter gradient, and the
        gradient with respect to the inputs (what a patch below this one would receive)."""
        batch = len(inputs)
        hidden, gate = path.hidden, path.gate
        horizon = hidden.shape[1]
        previous = np.concatenate((boundary[:, None], hidden[:, :-1]), axis=1)
        gh = d @ self._C
        carried = np.zeros((batch, self.hidden))
        for t in range(horizon - 1, -1, -1):
            gh[:, t] += carried
            carried = gate[:, t] * gh[:, t]
        gp = (1.0 - gate) * gh * (1.0 - port**2)
        gs = gh * (previous - port) * gate * (1.0 - gate)
        delta = {
            "G": self._drive_gradient(gs, inputs),
            "g": gs.sum(axis=(0, 1)),
            "B": self._drive_gradient(gp, inputs),
            "b": gp.sum(axis=(0, 1)),
            "C": np.einsum("bti,btj->ij", d, hidden),
            "c": d.sum(axis=(0, 1)),
        }
        return delta, self._drive_transpose(gp, "B") + self._drive_transpose(gs, "G")

    # ------------------------------------------------------------ interface

    def imagine(self, inputs: np.ndarray, *, state: np.ndarray | None = None) -> RecordPath:
        """Free private rollout; parameters, records, state and readback do not change."""
        path = self._path(inputs, self.inputs, "inputs")
        boundary = self._boundary(len(path), state)
        return self._free(path, boundary, None)[0]

    def _carry(self, path: RecordPath) -> None:
        self._state = path.final_state
        self._state_revision = self._revision

    def advance(self, inputs: np.ndarray) -> RecordPath:
        """Advance live activity without learning or writing."""
        path = self.imagine(inputs)
        self._carry(path)
        return path

    def observe(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        rate: float = 1.0,
        backtrack: bool = False,
        write: bool = True,
    ) -> RecordObservation:
        """Learn one finite observed path and write its observed outputs into the records.

        The prediction uses the records as they stood at the start of the
        call. The slow parameters move against the adjoint gradient of the
        precision-weighted half mean squared error; with ``backtrack=True`` a
        step is admitted only after target-free replay from the original
        boundary lowers that loss, trying up to sixteen halved rates. The
        observed outputs are then written into the records of the readings
        that produced the prediction. ``write=False`` learns without writing.
        The final free context becomes the live state. A failed record write
        restores the records and leaves the live context unchanged.
        """
        if not isinstance(backtrack, (bool, np.bool_)):
            raise ValueError("backtrack must be a boolean")
        if not isinstance(write, (bool, np.bool_)):
            raise ValueError("write must be a boolean")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        path, teaching = self._teaching(inputs, target)
        if write:
            _input_norms(path)  # validate before carrying activity or writing any record
        boundary = self._boundary(len(path), None)
        prediction, port, _ = self._free(path, boundary, teaching)
        if prediction.loss is None or prediction.slow_loss is None:
            self._carry(prediction)
            return RecordObservation(False, "nonfinite_prediction", prediction)
        delta = self._gradient(path, boundary, teaching, prediction, port)
        writes = 0
        if write:
            # Witnessing and sequential writes can mutate before a later write
            # overflows. Preserve learned state, without copying the projection.
            saved_norm, saved_records = self._input_norm, self.records.state()
            try:
                writes = self._write(path, teaching, prediction.hidden)
            except Exception:
                self._input_norm = saved_norm
                self.records.load_state(saved_records)
                raise
        self._carry(prediction)
        result = self._admit(path, teaching, boundary, prediction, delta, rate, backtrack)
        return RecordObservation(
            result.updated,
            result.reason,
            prediction,
            delta,
            result.initial_loss,
            result.final_loss,
            result.accepted_rate,
            result.replay_losses,
            result.replay_calls,
            writes,
        )

    def dream(self, inputs: np.ndarray) -> np.ndarray:
        """The store's own completion of a cue from rest: the free path with the records,
        as the patch would answer awake, made a target (for categorical ports the chosen
        category of each group). Nothing changes."""
        path = self._path(inputs, self.inputs, "inputs")
        output = self._free(path, np.zeros((len(path), self.hidden)), None)[0].output
        if self.groups is None:
            return np.asarray(output)
        target = np.zeros_like(output)
        for start, end in self._bounds():
            chosen = output[..., start:end].argmax(axis=-1)
            np.put_along_axis(target[..., start:end], chosen[..., None], 1.0, axis=-1)
        return target

    def sleep(
        self,
        cues: Sequence[np.ndarray],
        *,
        passes: int = 1,
        rate: float = 1.0,
        backtrack: bool = False,
        dawn_passes: int = 2,
    ) -> dict[str, float]:
        """A night: the slow weights take what the store holds, from dreams alone.

        Every cue is dreamed once, with the store as it stands at bedtime; the slow
        weights then learn the fixed dreams for ``passes`` passes by ``observe`` with
        ``write=False``; and at dawn each dream is written back ``dawn_passes`` times,
        so the store holds only what the slow weights did not take. Nothing outside the patch is
        consulted. A record is the residual against the slow readout at the time it
        was written: dreaming again after each update would follow the patch's own
        drift, and reading the old residuals against new slow weights would count the
        outcome twice, which is why the dreams are fixed and the store is rewritten.
        The read of a cue the store never met is what the records of its neighbours
        agree on, and it is taught with the rest. Returns admitted updates, the mean
        slow loss on the dreams before and after, and the writes of the dawn.
        """
        passes = _integer("passes", passes, 0)
        dawn_passes = _integer("dawn_passes", dawn_passes, 0)
        if not isinstance(backtrack, (bool, np.bool_)):
            raise ValueError("backtrack must be a boolean")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        dreams = [(self._path(cue, self.inputs, "inputs"), self.dream(cue)) for cue in cues]
        if not dreams:
            raise ValueError("sleep needs at least one cue")
        if dawn_passes:
            for path, _ in dreams:
                _input_norms(path)

        def slow_loss(path: np.ndarray, dream: np.ndarray) -> float:
            loss = self._free(path, np.zeros((len(path), self.hidden)), dream)[0].slow_loss
            return float("nan") if loss is None else loss

        before = float(
            np.mean(
                [slow_loss(p, d) for p, d in dreams]
            )
        )
        admitted = 0
        for _ in range(passes):
            for path, dream in dreams:
                self.reset()
                admitted += int(
                    self.observe(path, dream, rate=rate, backtrack=backtrack, write=False).updated
                )
        after = float(
            np.mean(
                [slow_loss(p, d) for p, d in dreams]
            )
        )
        writes = 0
        for _ in range(dawn_passes):
            for path, dream in dreams:
                self.reset()
                hidden = self._free(path, np.zeros((len(path), self.hidden)), None)[0].hidden
                writes += self._write(path, dream, hidden)
        self.reset()
        return {
            "cues": float(len(dreams)),
            "updates": float(admitted),
            "dream_loss_before": before,
            "dream_loss_after": after,
            "dawn_writes": float(writes),
        }

    def _admit(
        self,
        path: np.ndarray,
        target: np.ndarray,
        boundary: np.ndarray,
        prediction: RecordPath,
        delta: dict[str, np.ndarray],
        rate: float,
        backtrack: bool,
    ) -> RecordObservation:
        initial = prediction.slow_loss
        if rate == 0:
            return RecordObservation(False, "no_step", prediction, delta, initial, initial, 0.0)
        if not backtrack:
            with np.errstate(over="ignore", invalid="ignore"):
                proposed = {k: getattr(self, "_" + k) - rate * delta[k] for k in delta}
            if not all(np.isfinite(p).all() for p in proposed.values()):
                return RecordObservation(False, "nonfinite_update", prediction, delta, initial)
            self._commit(proposed)
            return RecordObservation(True, "updated", prediction, delta, initial)
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = sum(float(np.sum(value * value)) for value in delta.values())
        losses: list[float | None] = []
        replays = 0
        if initial is not None and np.isfinite(norm_squared) and norm_squared > 0 and rate > 0:
            current = self.parameters()
            trial = RecordPatchNet.restore(self.snapshot())
            for index in range(16):
                step = rate * 0.5**index
                with np.errstate(over="ignore", invalid="ignore"):
                    proposed = {k: current[k] - step * delta[k] for k in delta}
                if not all(np.isfinite(value).all() for value in proposed.values()):
                    losses.append(None)
                    continue
                trial.set_parameters(proposed)
                loss = trial._free(path, boundary, target)[0].slow_loss
                replays += 1
                losses.append(loss)
                floor = (
                    64
                    * np.finfo(float).eps
                    * max(abs(initial), abs(loss or 0.0), np.finfo(float).tiny)
                )
                if (
                    loss is not None
                    and loss < initial - floor
                    and loss <= initial - 1e-4 * step * norm_squared
                ):
                    self._commit(proposed)
                    return RecordObservation(
                        True,
                        "updated",
                        prediction,
                        delta,
                        initial,
                        loss,
                        step,
                        tuple(losses),
                        replays,
                    )
        return RecordObservation(
            False,
            "no_decreasing_parameter_step",
            prediction,
            delta,
            initial,
            initial,
            0.0,
            tuple(losses),
            replays,
        )

    def _commit(self, proposed: Mapping[str, np.ndarray]) -> None:
        for key, value in proposed.items():
            setattr(self, "_" + key, value)
        self.updates += 1
        self._revision += 1

    def _write(self, inputs: np.ndarray, target: np.ndarray, hidden: np.ndarray) -> int:
        """Write what the slow readout got wrong into the records of each reading, in order.

        Witnessed readings first move the running mean (imagined readings never
        do); the writes then land at the codes later readings will use. The
        residual is taken against the parameters that made the prediction."""
        batch, horizon, _ = inputs.shape
        norms = _input_norms(inputs)
        for value in norms:
            self._input_norm += 0.01 * (max(float(value), 1e-6) - self._input_norm)
        readings = self._readings(inputs, hidden).reshape(batch * horizon, -1)
        self.records.witness(readings)
        codes = self.records.code(readings, valued=False)[0].reshape(batch, horizon, -1)
        residual = target - self._slow(hidden)
        if self._output_code is not None:
            # The delta rule compares the coded residual with what the cells hold. Comparing
            # the decoded read instead would multiply the step by outputs / record_width.
            residual = residual @ self._output_code
        if self.record_writes == "batch":
            flat = codes.reshape(batch * horizon, -1)
            return self.records.write_batch(
                np.stack((flat, flat)), {"y": residual.reshape(batch * horizon, -1)}
            )
        written = 0
        for t in range(horizon):
            for row in range(batch):
                code = np.stack((codes[row, t], codes[row, t]))
                written += self.records.write(code, {"y": residual[row, t]})
        return written

    def detune(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        beta: float = 1e-3,
        state: np.ndarray | None = None,
        tolerance: float = 1e-14,
        max_iterations: int = 10_000,
    ) -> RecordContrast:
        """Solve both detuned equilibria of the quadratic energy and return their contrast.

        The energy describes the slow patch: its readout residual is
        ``y - C h - c`` and the record read is outside it. With ``y``
        eliminated, each detuned path is admitted only after its reduced
        quadratic is certified positive definite; conjugate gradients from the free
        path solve its normal equations to ``tolerance``. The contrast of the
        energy's parameter derivatives at the two equilibria, divided by
        ``2 beta``, is the learning signal of equilibrium detuning. On this
        regular branch its limiting value is the adjoint gradient of ``observe``;
        finite-beta error is of order ``beta^2`` before numerical solve error.
        Nothing here changes the network; this is the acceptance check of
        that identity.
        """
        if self.groups is not None:
            raise ValueError("detune needs the linear readout: categorical ports are not quadratic")
        path, teaching = self._teaching(inputs, target)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        max_iterations = _integer("max_iterations", max_iterations, 0)
        boundary = self._boundary(len(path), state)
        hidden, gate, port, _, _, _ = self._forward(path, boundary)
        batch, horizon, _ = path.shape
        b = beta * self._output_precision / (horizon * self.outputs)
        if np.any(b >= 1.0):
            raise ValueError("beta*output_precision/(time*outputs) must be below one")
        q = self._c - teaching
        converged = True

        def seam(v: np.ndarray) -> np.ndarray:
            out = v.copy()
            out[:, 1:] -= gate[:, 1:] * v[:, :-1]
            return out

        def seam_transpose(w: np.ndarray) -> np.ndarray:
            out = w.copy()
            out[:, :-1] -= gate[:, 1:] * w[:, 1:]
            return out

        constant = seam(hidden)  # the free path has zero defect: S h = a exactly

        def solve(weight: np.ndarray) -> tuple[np.ndarray, int, float]:
            nonlocal converged

            def operator(v: np.ndarray) -> np.ndarray:
                return np.asarray(seam_transpose(seam(v)) + ((v @ self._C.T) * weight) @ self._C)

            # b < 1 makes output elimination valid, but does not make the
            # negative-detuned hidden quadratic convex. Certify its block
            # Schur pivots even when the free point has zero gradient (CG
            # would otherwise accept a saddle without testing curvature).
            # Cheap sufficient bound: ||S^-1|| <= sum(max(gate)^k).
            # Positive detuning is SPD by construction. Only an unresolved
            # negative phase needs the more expensive block factorization.
            with np.errstate(over="ignore", invalid="ignore"):
                gain_bound = float(np.sum(np.max(gate) ** np.arange(horizon)))
                negative_bound = float(np.sum(self._C**2 * np.maximum(-weight, 0)[:, None]))
                certified = negative_bound * gain_bound**2 < 1 - 64 * np.finfo(float).eps
            if not certified:
                with np.errstate(over="ignore", invalid="ignore"):
                    gram = (self._C.T * weight) @ self._C
                inverse = None
                eye = np.eye(self.hidden)
                try:
                    for t in range(horizon):
                        pivot = np.broadcast_to(
                            eye + gram, (batch, self.hidden, self.hidden)
                        ).copy()
                        diagonal = np.arange(self.hidden)
                        if t + 1 < horizon:
                            pivot[:, diagonal, diagonal] += gate[:, t + 1] ** 2
                        if inverse is not None:
                            g = gate[:, t]
                            pivot -= g[:, :, None] * inverse * g[:, None, :]
                        if not np.isfinite(pivot).all():
                            raise np.linalg.LinAlgError("nonfinite quadratic")
                        factor = np.linalg.cholesky(0.5 * (pivot + pivot.swapaxes(-1, -2)))
                        inverse = np.linalg.solve(
                            factor.swapaxes(-1, -2),
                            np.linalg.solve(factor, np.broadcast_to(eye, pivot.shape)),
                        )
                except np.linalg.LinAlgError:
                    converged = False
                    return hidden.copy(), 0, float("inf")
            rhs = seam_transpose(constant) - (q * weight) @ self._C
            x = hidden.copy()
            r = rhs - operator(x)
            p = r.copy()
            rr = float(np.sum(r * r))
            scale = max(float(np.sum(rhs * rhs)), np.finfo(float).tiny)
            iterations = 0
            if not np.isfinite(rr) or not np.isfinite(scale):
                converged = False
                return x, iterations, float("inf")
            while np.sqrt(rr / scale) > tolerance and iterations < max_iterations:
                ap = operator(p)
                curvature = float(np.sum(p * ap))
                if not np.isfinite(curvature) or not curvature > 0.0:
                    converged = False
                    break
                alpha = rr / curvature
                x += alpha * p
                r -= alpha * ap
                new = float(np.sum(r * r))
                if not np.isfinite(new):
                    converged = False
                    break
                p = r + (new / rr) * p
                rr = new
                iterations += 1
            # A recursive CG residual can drift from the actual equations.
            # Admission uses a fresh equation evaluation, never NaN comparison.
            with np.errstate(over="ignore", invalid="ignore"):
                error = rhs - operator(x)
                residual = float(np.sqrt(np.sum(error * error) / scale))
            if not np.isfinite(residual) or residual > tolerance:
                converged = False
                if not np.isfinite(residual):
                    residual = float("inf")
            return x, iterations, residual

        def derivatives(h: np.ndarray, sign: float) -> tuple[dict[str, np.ndarray], float]:
            bw = sign * b
            prev = np.concatenate((boundary[:, None], h[:, :-1]), axis=1)
            e = h - gate * prev - (1.0 - gate) * port
            estimate = h @ self._C.T + self._c
            y = (estimate + bw * teaching) / (1.0 + bw)
            r = y - estimate
            energy = float(
                0.5 * (np.sum(e * e) + np.sum(r * r)) / batch
                + 0.5 * sign * beta * np.mean(self._output_precision * (y - teaching) ** 2)
            )
            gp = -e * (1.0 - gate) * (1.0 - port**2)
            gs = -e * (prev - port) * gate * (1.0 - gate)
            grads = {
                "G": self._drive_gradient(gs, path) / batch,
                "g": gs.sum(axis=(0, 1)) / batch,
                "B": self._drive_gradient(gp, path) / batch,
                "b": gp.sum(axis=(0, 1)) / batch,
                "C": -np.einsum("bti,btj->ij", r, h) / batch,
                "c": -r.sum(axis=(0, 1)) / batch,
            }
            return grads, energy

        plus, plus_iterations, plus_residual = solve(b / (1.0 + b))
        minus, minus_iterations, minus_residual = solve(-b / (1.0 - b))
        plus_grads, plus_energy = derivatives(plus, 1.0)
        minus_grads, minus_energy = derivatives(minus, -1.0)
        contrast = {k: (plus_grads[k] - minus_grads[k]) / (2.0 * beta) for k in plus_grads}
        converged = bool(
            converged
            and np.isfinite([plus_energy, minus_energy]).all()
            and all(np.isfinite(value).all() for value in contrast.values())
        )
        return RecordContrast(
            contrast,
            plus,
            minus,
            plus_energy,
            minus_energy,
            (plus_iterations, minus_iterations),
            (plus_residual, minus_residual),
            converged,
        )

    def reset(self) -> None:
        """Clear active context; retain learned parameters, records and counts."""
        self._state = None
        self._state_revision = None

    def readback(self) -> RecordReadback:
        return RecordReadback(
            self.state,
            self.updates,
            self.records.writes,
            self._revision,
            self._state_revision,
            self.records.parameters(),
        )

    # ------------------------------------------------------------ checkpoints

    def snapshot(self) -> dict[str, np.ndarray]:
        """Detached complete continuation state, without any retained training path."""
        ports = (
            self.groups is not None
            or self.record_writes != "sequential"
            or self.record_width is not None
            or self.port is not None
        )
        meta: dict[str, Any] = {
            "format": FORMAT_PORTS if ports else FORMAT,
            "inputs": self.inputs,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "slowest": self.slowest,
            "records": self.records.to_dict(),
            "updates": self.updates,
            "parameter_revision": self._revision,
            "state_parameter_revision": self._state_revision,
        }
        if ports:
            meta["groups"] = None if self.groups is None else list(self.groups)
            meta["record_writes"] = self.record_writes
            meta["record_width"] = self.record_width
            meta["port"] = None if self.port is None else self.port.to_dict()
        result = {
            **self.parameters(),
            "output_precision": self.output_precision,
            "input_norm": np.array(self._input_norm, dtype=float),
            "state": np.empty((0, self.hidden)) if self._state is None else self._state.copy(),
            "meta": np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
        }
        for key, value in self.records.state().items():
            result["records_" + key] = value
        return result

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> RecordPatchNet:
        """Construct a fresh net from validated complete state; never load pickle."""
        try:
            meta = json.loads(str(snapshot["meta"]))
            if not isinstance(meta, dict) or meta.get("format") not in (FORMAT, FORMAT_PORTS):
                raise ValueError("unsupported record-patch checkpoint")
            config: dict[str, Any] = dict(meta["records"])
            result = cls(
                meta["inputs"],
                meta["hidden"],
                meta["outputs"],
                seed=config["seed"],
                output_precision=snapshot["output_precision"],
                cells=config["cells"],
                active=config["active"],
                record_rate=config["rate"],
                habituation=config["habituation"],
                record_bias=config["bias"],
                slowest=meta["slowest"],
                record_averaging=config["averaging"],
                record_homeostasis=config["homeostasis"],
                groups=meta.get("groups"),
                record_writes=meta.get("record_writes", "sequential"),
                record_width=meta.get("record_width"),
                port=None if meta.get("port") is None else StructuredPort.from_dict(meta["port"]),
            )
            if result.records.to_dict() != config:
                raise ValueError("record configuration is not one this class constructs")
            result.set_parameters({k: snapshot[k] for k in _PARAMETERS})
            input_norm = float(np.asarray(snapshot["input_norm"]))
            if not np.isfinite(input_norm) or input_norm <= 0:
                raise ValueError("invalid saved input norm")
            result._input_norm = input_norm
            result.records.load_state(
                {
                    key[len("records_") :]: np.asarray(value)
                    for key, value in snapshot.items()
                    if key.startswith("records_")
                }
            )
            state = np.asarray(snapshot["state"])
            if state.ndim != 2 or state.shape[1] != result.hidden:
                raise ValueError("invalid saved state shape")
            result._state = None if not len(state) else result._boundary(len(state), state)
            result.updates = _integer("updates", meta["updates"], 0)
            result._revision = _integer("parameter_revision", meta["parameter_revision"], 0)
            revision = meta["state_parameter_revision"]
            result._state_revision = (
                None if revision is None else _integer("state_parameter_revision", revision, 0)
            )
            if (result._state is None) != (result._state_revision is None) or (
                result._state_revision is not None and result._state_revision > result._revision
            ):
                raise ValueError("invalid saved state revision")
            return result
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid record-patch checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path) -> RecordPatchNet:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})
