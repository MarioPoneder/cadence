"""A belief patch: a learned transition under action, evidence repair by iteration, a store read
inside the repair, and a readout the store patches.

The record patch computes its context from the input of the moment; without an input there
is no context, so it cannot imagine on its own. This patch keeps a belief ``z`` that a learned
transition carries forward under the executed action, and lets the evidence of the moment
repair it through a few iterations of one nonlinear map that reads the belief, the encoded
observation, the transition's expectation and the record store together. Scene and action
meet inside that map. An imagined moment is the transition alone under a declared action,
with the store read at the expected belief and nothing observed; imagination never writes.

    p[t]    = g * z[t-1] + (1 - g) * tanh(T [z[t-1]; a[t-1]] + t_b),  g = sigmoid(G [z; a] + g_b)
    e[t]    = gain[block] * tanh(port(o[t]) + e_b)                    the evidence, gained per block
    z(0)    = p[t]
    m(k)    = read(code([e[t], z(k)]))                                 the store, a coded residual
    z(k+1)      = (1 - alpha) z(k) + alpha tanh(F [z(k); e[t]; p[t]; m(k); 1] + f_b)
    y[t]        = C z(K) + c + decode(m(K))

Learning is one backward scan over a chunk through every iteration and the transition; the
record read is treated as given (no gradient reaches the store), as in the record patch. The
gain of each observation-port block multiplies that block's encoded evidence before the repair
map and the store read see it, so it acts in every iteration; the scan also returns the
gradient into each gain. The store holds the residual of the slow readout at the code of the
final reading, written once per observed moment. Everything private (``imagine``,
``readback``) leaves parameters, records, state and counters unchanged and consumes no
observation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ports import StructuredPort
from .records import Records

FORMAT = "cadence-belief/1"
_PARAMETERS = ("E", "e_b", "T", "t_b", "G", "g_b", "F", "f_b", "C", "c")
_HALVINGS = 16
_RECORD_KEYS = (
    "expect",
    "e",
    "e_raw",
    "gain",
    "observed",
    "repair",
    "y",
    "read",
    "code",
    "probe",
    "surprise",
)


def _integer(name: str, value: Any, low: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < low
    ):
        raise ValueError(f"{name} must be an integer >= {low}")
    return int(value)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class BeliefPath:
    """A detached path: beliefs ``(batch, time, belief)``, the transition's expectations, the
    residual of the last repair iteration at each moment, the outputs (slow readout plus
    the decoded store read) and the readback of each moment."""

    belief: np.ndarray
    expectation: np.ndarray
    residual: np.ndarray
    output: np.ndarray
    read: np.ndarray
    loss: float | None
    step: np.ndarray | None = None
    """The last repair iteration's move per belief unit, ``(batch, time, belief)``; its norm
    over the units is ``residual``. Where the belief moved under the evidence."""
    evidence: np.ndarray | None = None
    """The encoded evidence the repair and the store read, ``(batch, time, encoded)``: the
    port's output through ``tanh`` times the block's gain, zero where nothing was observed."""
    code: np.ndarray | None = None
    """The store's plain code at the final reading of each moment, ``(batch, time, cells)``:
    the cells the moment read, and wrote, with their weights."""
    gains: np.ndarray | None = None
    """The gain each block's evidence carried into the repair, ``(batch, time, blocks)``."""
    residual_alone: np.ndarray | None = None
    """The repair map's move from the expectation with only that block heard and the store
    read at zero, ``(batch, time, blocks)``; computed with ``probe=True``."""
    surprise: np.ndarray | None = None
    """Each block's reading against the reading the previous belief's slow readout implies,
    a root mean square in the block's persistence units, ``(batch, time, blocks)``; needs
    ``set_implied_reading``."""

    @property
    def slow_output(self) -> np.ndarray:
        return np.asarray(self.output - self.read)

    @property
    def final_state(self) -> np.ndarray:
        return self.belief[:, -1].copy()


@dataclass(frozen=True)
class BeliefReadback:
    """One moment's readback before its repair: the expectation ``(batch, belief)``, the
    residual-alone probe per block ``(batch, blocks)``, the surprise per block
    ``(batch, blocks)``, None without a declared implied reading, and the encoded evidence
    ``(batch, encoded)`` before any gain, what each block is reading. Nothing changes."""

    expectation: np.ndarray
    residual_alone: np.ndarray
    surprise: np.ndarray | None
    evidence: np.ndarray | None = None


@dataclass(frozen=True)
class BeliefObservation:
    """One teaching attempt: the path predicted before any write, the adjoint, what changed."""

    updated: bool
    reason: str
    path: BeliefPath
    delta: dict[str, np.ndarray]
    initial_loss: float | None
    writes: int
    final_loss: float | None = None
    """The chunk's loss replayed under the admitted parameters (``backtrack=True``)."""
    accepted_rate: float | None = None
    """The rate of the step taken; None when no step was taken."""
    replay_calls: int = 0
    """How many replays the admission ran."""
    gain_gradient: np.ndarray | None = None
    """The gradient of the loss into each block's gain per moment, ``(batch, time, blocks)``;
    a gain shared over moments or streams takes the sum over those axes."""


class BeliefPatch:
    def __init__(
        self,
        observation: StructuredPort,
        actions: int,
        belief: int,
        outputs: int,
        *,
        iterations: int = 2,
        damping: float = 0.5,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        record_width: int = 64,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        output_precision: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.port = observation
        self.inputs = observation.inputs
        self.encoded = observation.outputs
        self.block_count = len(observation.blocks)
        offsets = [int(v) for v in observation._offsets]
        self._block_slices = [slice(offsets[i], offsets[i + 1]) for i in range(self.block_count)]
        self._unit_block = np.concatenate(
            [np.full(b.outputs, i, dtype=int) for i, b in enumerate(observation.blocks)]
        )
        self.actions = _integer("actions", actions, 1)
        self.belief = _integer("belief", belief, 1)
        self.outputs = _integer("outputs", outputs, 1)
        self.iterations = _integer("iterations", iterations, 1)
        if not 0.0 < float(damping) <= 1.0:
            raise ValueError("damping lies in (0, 1]")
        self.damping = float(damping)
        self.record_width = _integer("record_width", record_width, 1)
        self.set_output_precision(
            np.ones(self.outputs) if output_precision is None else output_precision
        )
        rng = np.random.default_rng(seed)
        za = self.belief + self.actions
        self._E = np.concatenate([w.ravel() for w in observation.initial(rng, 1.0)])
        self._e_b = np.zeros(self.encoded)
        self._T = rng.normal(size=(self.belief, za)) / np.sqrt(za)
        self._t_b = np.zeros(self.belief)
        self._G = np.zeros((self.belief, za))
        self._g_b = np.full(self.belief, 1.0)  # retain most of the belief across a moment at birth
        fi = self.belief + self.encoded + self.belief + self.record_width + 1
        self._F = rng.normal(size=(self.belief, fi)) / np.sqrt(fi)
        self._f_b = np.zeros(self.belief)
        self._C = np.zeros((self.outputs, self.belief))
        self._c = np.zeros(self.outputs)
        signs = rng.integers(0, 2, (self.outputs, self.record_width))
        self._output_code = (2.0 * signs - 1.0) / np.sqrt(self.record_width)
        self.records = Records(
            self.encoded + self.belief,
            {"y": self.record_width},
            cells=cells,
            active=active,
            rate=record_rate,
            habituation=habituation,
            bias=record_bias,
            seed=seed,
        )
        self._input_norm = 1.0
        self._state: np.ndarray | None = None
        self._implied: Callable[[np.ndarray], np.ndarray] | None = None
        self._units = np.ones(self.block_count)
        self._step_size: float | None = None
        self.updates = 0
        self.cost = {"moments": 0, "macs": 0, "replays": 0}

    # ------------------------------------------------------------------ parameters
    @property
    def state(self) -> np.ndarray | None:
        return None if self._state is None else self._state.copy()

    @property
    def step_size(self) -> float | None:
        """The last admitted step; the next admission starts from twice it, at most ``rate``.
        None before any admitted step or after ``reset_step``."""
        return self._step_size

    def reset_step(self) -> None:
        self._step_size = None

    # ------------------------------------------------------------------ accounting
    def macs_per_moment(self, *, probes: bool = False) -> int:
        """Multiply-accumulates of one moment: the port, the transition and its gate, the
        store's reads and the repair map per iteration, the slow readout and the store's
        decode; with ``probes`` one evaluation of the repair map per block for the
        residual-alone probes. One accounting for every arm of a comparison."""
        port = sum(int(np.prod(self.port.weight_shape(b))) for b in self.port.blocks)
        za = self.belief + self.actions
        transition = 2 * self.belief * za
        reading = self.encoded + self.belief
        read = reading * self.records.cells + self.records.active * self.record_width
        fi = self.belief + self.encoded + self.belief + self.record_width + 1
        repair = self.iterations * (read + self.belief * fi) + read
        readout = self.outputs * self.belief + self.record_width * self.outputs
        probing = self.block_count * self.belief * fi if probes else 0
        return int(port + transition + repair + readout + probing)

    def reset_cost(self) -> None:
        """Zero the counters: ``cost["moments"]`` counts every moment the patch computed
        (assimilated, observed, imagined, and replayed in an admission), ``cost["macs"]``
        their multiply-accumulates, ``cost["replays"]`` the admission's replays."""
        self.cost = {"moments": 0, "macs": 0, "replays": 0}

    def _count(self, n: int, t: int, *, probes: bool = False) -> None:
        self.cost["moments"] += int(n * t)
        self.cost["macs"] += int(n * t) * self.macs_per_moment(probes=probes)

    def set_output_precision(self, precision: np.ndarray) -> None:
        value = np.asarray(precision, dtype=float)
        if value.shape != (self.outputs,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("output_precision must be finite and positive, one per output")
        self._output_precision = value.copy()

    def set_implied_reading(
        self,
        implied: Callable[[np.ndarray], np.ndarray] | None,
        units: np.ndarray | None = None,
    ) -> None:
        """Declare the map from the outputs ``(batch, outputs)`` to the reading each block
        should give, ``(batch, inputs)``, and the persistence error of each block's compared
        channels ``(blocks,)``: the mean squared change of those channels from one moment to
        the next on a batch of training data. Channels the map leaves ``NaN`` are not
        compared. Paths then carry ``surprise``; ``None`` withdraws the declaration. The
        declaration is not part of a snapshot."""
        if implied is not None and not callable(implied):
            raise ValueError("implied must be callable or None")
        value = np.ones(self.block_count) if units is None else np.asarray(units, dtype=float)
        if value.shape != (self.block_count,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("units must be finite and positive, one per block")
        self._implied = implied
        self._units = value.copy()

    def parameters(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, "_" + k).copy() for k in _PARAMETERS}

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        current = self.parameters()
        if set(parameters) != set(current):
            raise ValueError(f"parameters must contain exactly {', '.join(_PARAMETERS)}")
        values = {}
        for key, old in current.items():
            value = np.asarray(parameters[key], dtype=float)
            if value.shape != old.shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite with shape {old.shape}")
            values[key] = value.copy()
        for key, value in values.items():
            setattr(self, "_" + key, value)

    def _blocks(self) -> list[np.ndarray]:
        out, at = [], 0
        for b in self.port.blocks:
            shape = self.port.weight_shape(b)
            size = int(np.prod(shape))
            out.append(self._E[at : at + size].reshape(shape))
            at += size
        return out

    # ------------------------------------------------------------------ the moment
    def _encode(self, o: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pre = self.port.apply(o, self._blocks()) + self._e_b
        return np.tanh(pre), pre

    def encode(self, observations: np.ndarray) -> np.ndarray:
        """The encoded evidence of a reading before any gain, ``tanh(port(o) + e_b)``, for
        ``(..., inputs)`` observations: what each block hears, as the repair will hear it
        once the gains weigh it. A steering patch that decides a block's gain reads this
        before the gain it sets. Nothing changes."""
        o = np.asarray(observations, dtype=float)
        if o.shape[-1] != self.inputs or not np.isfinite(o).all():
            raise ValueError(f"observations must be a finite (..., {self.inputs}) array")
        return self._encode(o)[0]

    def _expect(self, z: np.ndarray, a: np.ndarray) -> dict[str, np.ndarray]:
        x = np.concatenate([z, a], axis=-1)
        g = _sigmoid(x @ self._G.T + self._g_b)
        cand = np.tanh(x @ self._T.T + self._t_b)
        return {"x": x, "g": g, "cand": cand, "p": g * z + (1.0 - g) * cand}

    def _reading(self, e: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.concatenate([e * (np.sqrt(self.encoded) / self._input_norm), z], axis=-1)

    def _read(self, e: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        code = self.records.code(self._reading(e, z), valued=False)[0]
        return code @ self.records.tables["y"], code

    def _repair(self, e: np.ndarray, p: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
        """The iterations of one moment; a row with nothing observed keeps the expectation."""
        zs, hs, us = [p], [], []
        z = p
        if observed.any():
            keep = observed[:, None]
            flag = keep.astype(float)
            for _ in range(self.iterations):
                m, _ = self._read(e, z)
                u = np.concatenate([z, e, p, m, flag], axis=-1)
                h = np.tanh(u @ self._F.T + self._f_b)
                z = np.where(keep, (1.0 - self.damping) * z + self.damping * h, z)
                zs.append(z)
                hs.append(h)
                us.append(u)
        m, code = self._read(e, z)
        step = zs[-1] - zs[-2] if len(zs) > 1 else np.zeros_like(p)
        residual = np.linalg.norm(step, axis=-1)
        return {
            "z": z,
            "zs": zs,
            "hs": hs,
            "us": us,
            "read": m,
            "code": code,
            "residual": residual,
            "step": step,
        }

    def _probe(self, e_raw: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The repair map's move from the expectation with one block's raw evidence alone and
        the store read at zero, one evaluation of the map per block: ``(batch, blocks)``."""
        n = len(p)
        zeros = np.zeros((n, self.record_width))
        flag = np.ones((n, 1))
        out = np.zeros((n, self.block_count))
        for b, sel in enumerate(self._block_slices):
            alone = np.zeros_like(e_raw)
            alone[:, sel] = e_raw[:, sel]
            h = np.tanh(np.concatenate([p, alone, p, zeros, flag], axis=-1) @ self._F.T + self._f_b)
            out[:, b] = np.linalg.norm(h - p, axis=-1)
        return out

    def _surprise(self, o: np.ndarray, z_prev: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Each block's reading against the reading the previous belief's slow readout implies,
        a root mean square over the block's compared channels in its persistence units:
        ``(batch, blocks)``, zero on a row that read nothing."""
        assert self._implied is not None
        n = len(o)
        implied = np.asarray(self._implied(z_prev @ self._C.T + self._c), dtype=float)
        if implied.shape != (n, self.inputs):
            raise ValueError(f"the implied reading must be a (batch, {self.inputs}) array")
        valid = np.isfinite(implied)
        diff = np.where(valid, implied - o, 0.0)
        out = np.zeros((n, self.block_count))
        for b, block in enumerate(self.port.blocks):
            sel = slice(block.start, block.start + block.inputs)
            count = valid[:, sel].sum(axis=-1)
            squared = (diff[:, sel] ** 2).sum(axis=-1)
            out[:, b] = np.sqrt(squared / np.maximum(count, 1) / self._units[b])
        return np.asarray(out * observed[:, None])

    def _forward(
        self,
        observations: np.ndarray | None,
        actions: np.ndarray,
        boundary: np.ndarray,
        observed: np.ndarray,
        gains: np.ndarray,
        *,
        probe: bool = False,
        surprise: bool = False,
    ) -> dict[str, Any]:
        n, t = actions.shape[:2]
        z = boundary
        record: dict[str, list[Any]] = {k: [] for k in _RECORD_KEYS}
        implied = surprise and observations is not None and self._implied is not None
        self._count(n, t, probes=bool(probe and observations is not None))
        for k in range(t):
            ex = self._expect(z, actions[:, k])
            row = observed[:, k]
            if observations is not None and row.any():
                e_raw, _ = self._encode(observations[:, k])
                e_raw = e_raw * row[:, None]
            else:
                e_raw = np.zeros((n, self.encoded))
            gain = gains[:, k]
            e = e_raw * gain[:, self._unit_block]
            rep = self._repair(e, ex["p"], row)
            probe_k = (
                self._probe(e_raw, ex["p"]) * row[:, None]
                if probe and observations is not None
                else None
            )
            surprise_k = (
                self._surprise(observations[:, k], z, row)
                if implied and observations is not None
                else None
            )
            read = rep["read"] @ self._output_code.T
            y = rep["z"] @ self._C.T + self._c + read
            for key, value in (
                ("expect", ex),
                ("e", e),
                ("e_raw", e_raw),
                ("gain", gain),
                ("observed", row),
                ("repair", rep),
                ("y", y),
                ("read", read),
                ("code", rep["code"]),
                ("probe", probe_k),
                ("surprise", surprise_k),
            ):
                record[key].append(value)
            z = rep["z"]
        return record

    def _path(
        self,
        record: dict[str, list[Any]],
        target: np.ndarray | None,
        weight: np.ndarray | None = None,
    ) -> BeliefPath:
        belief = np.stack([r["z"] for r in record["repair"]], axis=1)
        expectation = np.stack([x["p"] for x in record["expect"]], axis=1)
        residual = np.stack([r["residual"] for r in record["repair"]], axis=1)
        step = np.stack([r["step"] for r in record["repair"]], axis=1)
        output = np.stack(record["y"], axis=1)
        read = np.stack(record["read"], axis=1)
        loss = None if target is None else self._loss(output - read, target, weight)
        return BeliefPath(
            belief,
            expectation,
            residual,
            output,
            read,
            loss,
            step,
            evidence=np.stack(record["e"], axis=1),
            code=np.stack(record["code"], axis=1),
            gains=np.stack(record["gain"], axis=1),
            residual_alone=None
            if record["probe"][0] is None
            else np.stack(record["probe"], axis=1),
            surprise=None
            if record["surprise"][0] is None
            else np.stack(record["surprise"], axis=1),
        )

    def _loss(
        self, slow: np.ndarray, target: np.ndarray, weight: np.ndarray | None = None
    ) -> float | None:
        with np.errstate(over="ignore", invalid="ignore"):
            squared = self._output_precision * (slow - target) ** 2
            if weight is None:
                value = float(0.5 * np.mean(squared))
            else:
                value = float(
                    0.5 * np.sum(weight[:, :, None] * squared) / (self.outputs * weight.sum())
                )
        return value if np.isfinite(value) else None

    # ------------------------------------------------------------------ interface
    def _check(
        self, observations: Any, actions: Any, observed: Any
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        a = np.asarray(actions, dtype=float)
        if a.ndim != 3 or a.shape[2] != self.actions or not np.isfinite(a).all():
            raise ValueError(f"actions must be a finite (batch, time, {self.actions}) array")
        n, t = a.shape[:2]
        if n < 1 or t < 1:
            raise ValueError("a chunk needs at least one stream and one moment")
        if observations is None:
            o = None
            mask = np.zeros((n, t), dtype=bool)
        else:
            o = np.asarray(observations, dtype=float)
            if o.shape != (n, t, self.inputs) or not np.isfinite(o).all():
                raise ValueError(
                    f"observations must be a finite (batch, time, {self.inputs}) array"
                )
            mask = (
                np.ones((n, t), dtype=bool)
                if observed is None
                else self._rows(observed, n, t, "observed", bool)
            )
        return o, a, mask

    @staticmethod
    def _rows(value: Any, n: int, t: int, name: str, dtype: type) -> np.ndarray:
        """A per-moment array given as ``(time,)`` or ``(batch, time)``, as ``(batch, time)``."""
        v: np.ndarray = np.asarray(value, dtype=dtype)
        if v.shape == (n, t):
            return v.copy()
        if v.size == t:
            return np.ascontiguousarray(np.broadcast_to(v.reshape(t), (n, t)))
        raise ValueError(f"{name} must be a (time,) or (batch, time) array")

    def _gains(self, gains: Any, n: int, t: int) -> np.ndarray:
        """The gain of each block per moment, as ``(batch, time, blocks)``; ones when none."""
        b = self.block_count
        if gains is None:
            return np.ones((n, t, b))
        g = np.asarray(gains, dtype=float)
        if g.shape == (b,):
            g = np.broadcast_to(g, (n, t, b))
        elif g.shape == (n, b):
            g = np.broadcast_to(g[:, None, :], (n, t, b))
        elif g.shape != (n, t, b):
            raise ValueError(f"gains must have shape ({b},), (batch, {b}) or (batch, time, {b})")
        if not np.isfinite(g).all():
            raise ValueError("gains must be finite")
        return np.array(g, dtype=float)

    def _weight(self, loss_weight: Any, n: int, t: int) -> np.ndarray | None:
        if loss_weight is None:
            return None
        w = self._rows(loss_weight, n, t, "loss_weight", float)
        if not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0:
            raise ValueError("loss_weight must be finite and nonnegative with a positive sum")
        return w

    def _boundary(self, n: int, state: np.ndarray | None) -> np.ndarray:
        source = self._state if state is None else state
        if source is None:
            return np.zeros((n, self.belief))
        value = np.asarray(source, dtype=float)
        if value.shape != (n, self.belief) or not np.isfinite(value).all():
            raise ValueError("state must match (batch, belief); reset when streams change")
        return value.copy()

    def assimilate(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        observed: np.ndarray | None = None,
        *,
        state: np.ndarray | None = None,
        gains: np.ndarray | None = None,
        probe: bool = False,
        keep_live: bool = False,
    ) -> BeliefPath:
        """Advance the belief through observed moments: the executed action, then the evidence.
        Nothing is learned or written; the final belief becomes the live state. ``state`` starts
        the moments from a given boundary instead of the live belief (a window that is replayed
        from the belief that was live at its first moment), and ``keep_live=True`` leaves the
        live state as it was, so a life that has stepped through the moments can replay them
        without losing its place. ``observed`` masks moments ``(time,)`` or rows
        ``(batch, time)``; ``gains`` weighs each block's evidence inside the repair;
        ``probe=True`` computes the residual-alone probe per block."""
        o, a, mask = self._check(observations, actions, observed)
        n, t = a.shape[:2]
        record = self._forward(
            o,
            a,
            self._boundary(n, state),
            mask,
            self._gains(gains, n, t),
            probe=bool(probe),
            surprise=True,
        )
        path = self._path(record, None)
        if not keep_live:
            self._state = path.final_state
        return path

    def imagine(
        self,
        actions: np.ndarray,
        *,
        state: np.ndarray | None = None,
        gains: np.ndarray | None = None,
    ) -> BeliefPath:
        """A private continuation under declared actions from the live belief (or ``state``):
        the transition alone, the store read at the expectation, no observation, no change.
        ``gains`` are carried on the path; with nothing observed they act on nothing."""
        _, a, mask = self._check(None, actions, None)
        n, t = a.shape[:2]
        record = self._forward(None, a, self._boundary(n, state), mask, self._gains(gains, n, t))
        return self._path(record, None)

    def readback(
        self, observations: np.ndarray, actions: np.ndarray, *, state: np.ndarray | None = None
    ) -> BeliefReadback:
        """One moment's readback before its repair, from the live belief or ``state``: the
        expectation under ``actions`` ``(batch, actions)``, the residual-alone probe of each
        block of ``observations`` ``(batch, inputs)`` and, with an implied reading declared,
        each block's surprise. What a steering patch reads before it sets the moment's gains.
        Nothing changes."""
        o = np.asarray(observations, dtype=float)
        a = np.asarray(actions, dtype=float)
        if o.ndim != 2 or o.shape[1] != self.inputs or not np.isfinite(o).all():
            raise ValueError(f"observations must be a finite (batch, {self.inputs}) array")
        if a.shape != (len(o), self.actions) or not np.isfinite(a).all():
            raise ValueError(f"actions must be a finite (batch, {self.actions}) array")
        z = self._boundary(len(o), state)
        ex = self._expect(z, a)
        e_raw, _ = self._encode(o)
        rows = np.ones(len(o), dtype=bool)
        surprise = None if self._implied is None else self._surprise(o, z, rows)
        return BeliefReadback(ex["p"], self._probe(e_raw, ex["p"]), surprise, e_raw)

    def observe(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        target: np.ndarray | None = None,
        *,
        observed: np.ndarray | None = None,
        rate: float = 1.0,
        write: bool = True,
        state: np.ndarray | None = None,
        gains: np.ndarray | None = None,
        loss_weight: np.ndarray | None = None,
        output_gradient: np.ndarray | None = None,
        backtrack: bool | None = None,
        probe: bool = False,
        keep_live: bool = False,
    ) -> BeliefObservation:
        """Learn one chunk of witnessed moments and write their outcomes into the store. ``state``
        starts the chunk from a given boundary instead of the live belief; the final belief under
        the chunk becomes the live state unless ``keep_live=True``.

        ``loss_weight`` weighs each moment's error, ``(time,)`` or ``(batch, time)``, normalized
        by its sum; a moment of weight zero is neither taught nor written. ``output_gradient``
        ``(batch, time, outputs)`` replaces ``target``: the adjoint then carries that gradient of
        an external loss on the outputs into the parameters and the gains, nothing is written
        and no loss is reported. A step on a target is admitted (``backtrack`` None or True):
        it is taken only after a replay of the chunk from the same boundary, under the proposed
        parameters and with the store as it stands, lowers the chunk's loss by the Armijo
        margin, starting from twice the last admitted step (at most ``rate``) and trying up to
        sixteen halvings. ``backtrack=False`` takes the plain step at ``rate``; a step on an
        external gradient is plain, since the library can replay only the loss it can see.
        ``gains`` weighs each block's evidence inside the repair and the gradient into the
        gains comes back as ``gain_gradient``."""
        if backtrack is not None and not isinstance(backtrack, (bool, np.bool_)):
            raise ValueError("backtrack must be a boolean or None")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        if (target is None) == (output_gradient is None):
            raise ValueError("observe takes a target or an output_gradient, one of the two")
        o, a, mask = self._check(observations, actions, observed)
        if o is None:
            raise ValueError("observe needs observations; imagine is the private continuation")
        n, t = a.shape[:2]
        gain = self._gains(gains, n, t)
        weight = self._weight(loss_weight, n, t)
        y = dy = None
        if target is not None:
            y = np.asarray(target, dtype=float)
            if y.shape != (n, t, self.outputs) or not np.isfinite(y).all():
                raise ValueError("target must be a finite (batch, time, outputs) array")
        else:
            dy = np.asarray(output_gradient, dtype=float)
            if dy.shape != (n, t, self.outputs) or not np.isfinite(dy).all():
                raise ValueError("output_gradient must be a finite (batch, time, outputs) array")
            if weight is not None:
                raise ValueError(
                    "loss_weight weighs a target's loss; fold it into the output_gradient"
                )
            if backtrack:
                raise ValueError("backtrack needs a target: the admission replays the chunk's loss")
        admit = (y is not None) if backtrack is None else bool(backtrack)
        boundary = self._boundary(n, state)
        record = self._forward(o, a, boundary, mask, gain, probe=bool(probe), surprise=True)
        path = self._path(record, y, weight)
        if not keep_live:
            self._state = path.final_state
        if (y is not None and path.loss is None) or not np.isfinite(path.output).all():
            return BeliefObservation(False, "nonfinite_prediction", path, {}, None, 0)
        delta, dgains = self._adjoint(o, a, boundary, record, y, weight, dy)
        if rate > 0:
            admitted = self._admit(
                o, a, mask, gain, weight, boundary, y, delta, rate, admit, path.loss
            )
        else:
            admitted = (False, "no_step", None, None, 0)
        writes = self._write(record, y, weight) if write and y is not None else 0
        updated, reason, final, accepted, replays = admitted
        return BeliefObservation(
            updated, reason, path, delta, path.loss, writes, final, accepted, replays, dgains
        )

    def reset(self) -> None:
        """Forget the live state; the step size and the counters stay."""
        self._state = None

    # ------------------------------------------------------------------ learning
    def _adjoint(
        self,
        o: np.ndarray,
        a: np.ndarray,
        boundary: np.ndarray,
        record: dict[str, list[Any]],
        target: np.ndarray | None,
        weight: np.ndarray | None = None,
        output_gradient: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """The parameter gradient and the gradient into the gains ``(batch, time, blocks)`` of
        the weighted loss against ``target``, or of the external loss whose gradient on the
        outputs is ``output_gradient``."""
        n, t = a.shape[:2]
        alpha = self.damping
        delta = {k: np.zeros_like(getattr(self, "_" + k)) for k in _PARAMETERS}
        block_grads = [np.zeros(self.port.weight_shape(b)) for b in self.port.blocks]
        dgains = np.zeros((n, t, self.block_count))
        dz_next = np.zeros((n, self.belief))  # gradient into z[t] from moment t+1
        total = None if weight is None else self.outputs * weight.sum()
        for k in range(t - 1, -1, -1):
            rep, ex = record["repair"][k], record["expect"][k]
            z_prev = boundary if k == 0 else record["repair"][k - 1]["z"]
            if output_gradient is not None:
                dy = output_gradient[:, k]
            else:
                assert target is not None
                slow = record["y"][k] - record["read"][k]
                if weight is None:
                    dy = self._output_precision * (slow - target[:, k]) / (n * t * self.outputs)
                else:
                    dy = (
                        self._output_precision
                        * (slow - target[:, k])
                        * (weight[:, k, None] / total)
                    )
            delta["C"] += dy.T @ rep["z"]
            delta["c"] += dy.sum(axis=0)
            dz = dy @ self._C + dz_next
            keep = record["observed"][k][:, None]
            de = np.zeros((n, self.encoded))
            dp = np.zeros((n, self.belief))
            for j in range(len(rep["hs"]) - 1, -1, -1):
                h, u = rep["hs"][j], rep["us"][j]
                dh = np.where(keep, alpha * dz, 0.0)
                dz = np.where(keep, (1.0 - alpha) * dz, dz)
                dpre = dh * (1.0 - h**2)
                delta["F"] += dpre.T @ u
                delta["f_b"] += dpre.sum(axis=0)
                du = dpre @ self._F
                dz = dz + du[:, : self.belief]
                de += du[:, self.belief : self.belief + self.encoded]
                dp += du[:, self.belief + self.encoded : 2 * self.belief + self.encoded]
            dp += dz  # z(0) = p
            if np.any(de):
                e_raw, gain = record["e_raw"][k], record["gain"][k]
                product = de * e_raw
                for b, sel in enumerate(self._block_slices):
                    dgains[:, k, b] = product[:, sel].sum(axis=-1)
                dpre_e = de * gain[:, self._unit_block] * (1.0 - e_raw**2) * keep
                delta["e_b"] += dpre_e.sum(axis=0)
                for i, gb in enumerate(self.port.gradient(dpre_e, o[:, k])):
                    block_grads[i] += gb
            g, cand, x = ex["g"], ex["cand"], ex["x"]
            dz_prev = g * dp
            dg = dp * (z_prev - cand)
            dcand = dp * (1.0 - g)
            dpre_c = dcand * (1.0 - cand**2)
            delta["T"] += dpre_c.T @ x
            delta["t_b"] += dpre_c.sum(axis=0)
            dpre_g = dg * g * (1.0 - g)
            delta["G"] += dpre_g.T @ x
            delta["g_b"] += dpre_g.sum(axis=0)
            dx = dpre_c @ self._T + dpre_g @ self._G
            dz_next = dz_prev + dx[:, : self.belief]
        delta["E"] = np.concatenate([gb.ravel() for gb in block_grads])
        return delta, dgains

    def _commit(self, proposed: Mapping[str, np.ndarray]) -> None:
        for key, value in proposed.items():
            setattr(self, "_" + key, value)
        self.updates += 1

    def _admit(
        self,
        o: np.ndarray,
        a: np.ndarray,
        mask: np.ndarray,
        gains: np.ndarray,
        weight: np.ndarray | None,
        boundary: np.ndarray,
        target: np.ndarray | None,
        delta: dict[str, np.ndarray],
        rate: float,
        backtrack: bool,
        initial: float | None,
    ) -> tuple[bool, str, float | None, float | None, int]:
        """The step: plain at ``rate``, or the largest halving of the start whose replay of the
        chunk from the same boundary, with the store as it stands, lowers the loss by the
        Armijo margin; the start is twice the last admitted step, at most ``rate``. Returns
        (updated, reason, final loss, accepted rate, replays)."""
        current = self.parameters()
        if not backtrack:
            with np.errstate(over="ignore", invalid="ignore"):
                proposed = {k: current[k] - rate * delta[k] for k in _PARAMETERS}
            if not all(np.isfinite(v).all() for v in proposed.values()):
                return False, "nonfinite_update", None, None, 0
            self._commit(proposed)
            return True, "updated", None, float(rate), 0
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = sum(float(np.sum(v * v)) for v in delta.values())
        replays = 0
        start = rate if self._step_size is None else min(float(rate), 2.0 * self._step_size)
        if (
            initial is not None
            and target is not None
            and np.isfinite(norm_squared)
            and norm_squared > 0
        ):
            for index in range(_HALVINGS):
                step = start * 0.5**index
                with np.errstate(over="ignore", invalid="ignore"):
                    proposed = {k: current[k] - step * delta[k] for k in _PARAMETERS}
                if not all(np.isfinite(v).all() for v in proposed.values()):
                    continue
                for key, value in proposed.items():
                    setattr(self, "_" + key, value)
                try:
                    replay = self._forward(o, a, boundary, mask, gains)
                    loss = self._loss(
                        np.stack(replay["y"], axis=1) - np.stack(replay["read"], axis=1),
                        target,
                        weight,
                    )
                finally:
                    for key, value in current.items():
                        setattr(self, "_" + key, value)
                replays += 1
                self.cost["replays"] += 1
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
                    self._step_size = float(step)
                    return True, "updated", loss, step, replays
        return False, "no_decreasing_parameter_step", initial, None, replays

    def _write(
        self, record: dict[str, list[Any]], target: np.ndarray, weight: np.ndarray | None = None
    ) -> int:
        """Write the slow readout's residual, coded, at the final reading of each observed
        moment; a row whose loss weight is zero is not written."""
        written = 0
        t = len(record["y"])
        for k in range(t):
            rows = record["observed"][k].copy()
            if weight is not None:
                rows &= weight[:, k] > 0
            if not rows.any():
                continue  # nothing observed, or nothing to teach: nothing to write
            rep, e = record["repair"][k], record["e"][k][rows]
            readings = self._reading(e, rep["z"][rows])
            for value in np.linalg.norm(e, axis=1):
                self._input_norm += 0.01 * (max(float(value), 1e-6) - self._input_norm)
            self.records.witness(readings)
            codes = self.records.code(readings, valued=False)[0]
            slow = record["y"][k][rows] - record["read"][k][rows]
            residual = (target[rows, k] - slow) @ self._output_code
            for row in range(len(codes)):
                written += self.records.write(
                    np.stack((codes[row], codes[row])), {"y": residual[row]}
                )
        return written

    # ------------------------------------------------------------------ custody
    def snapshot(self) -> dict[str, np.ndarray]:
        meta = {
            "format": FORMAT,
            "port": self.port.to_dict(),
            "actions": self.actions,
            "belief": self.belief,
            "outputs": self.outputs,
            "iterations": self.iterations,
            "damping": self.damping,
            "record_width": self.record_width,
            "records": self.records.to_dict(),
            "updates": self.updates,
        }
        out = {
            **self.parameters(),
            "output_precision": self._output_precision.copy(),
            "output_code": self._output_code.copy(),
            "input_norm": np.array(self._input_norm),
            "step_size": np.empty(0) if self._step_size is None else np.array([self._step_size]),
            "state": np.empty((0, self.belief)) if self._state is None else self._state.copy(),
            "meta": np.array(json.dumps(meta, sort_keys=True)),
        }
        for key, value in self.records.state().items():
            out["records_" + key] = value
        return out

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> BeliefPatch:
        try:
            meta = json.loads(str(snapshot["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError("unsupported belief checkpoint")
            config = dict(meta["records"])
            result = cls(
                StructuredPort.from_dict(meta["port"]),
                meta["actions"],
                meta["belief"],
                meta["outputs"],
                iterations=meta["iterations"],
                damping=meta["damping"],
                cells=config["cells"],
                active=config["active"],
                record_rate=config["rate"],
                record_width=meta["record_width"],
                habituation=config["habituation"],
                record_bias=config["bias"],
                output_precision=snapshot["output_precision"],
                seed=config["seed"],
            )
            result.set_parameters({k: snapshot[k] for k in _PARAMETERS})
            result._output_code = np.asarray(snapshot["output_code"], dtype=float).copy()
            result._input_norm = float(np.asarray(snapshot["input_norm"]))
            if "step_size" in snapshot:
                kept = np.asarray(snapshot["step_size"], dtype=float).reshape(-1)
                result._step_size = None if kept.size == 0 else float(kept[0])
            result.records.load_state(
                {
                    k[len("records_") :]: np.asarray(v)
                    for k, v in snapshot.items()
                    if k.startswith("records_")
                }
            )
            state = np.asarray(snapshot["state"])
            result._state = None if not len(state) else state.astype(float).copy()
            result.updates = int(meta["updates"])
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid belief checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path) -> BeliefPatch:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})
