"""Carried state as a stimulus: a brain that carries its last equilibrium into the next one.

If settling has a unique attracting fixed point, fully settling the next input
erases the starting state. Multiple attractors and interrupted settling can retain
history, but are not a reliable addressed store. Here a range of *context* neurons,
one per hidden neuron, is stimulated to a leaky trace of the hidden neurons' own activation at
the previous inputs:

    c <- decay * c + (1 - decay) * h

so the state reverberates and fades over about ``1 / (1 - decay)`` inputs rather than
vanishing at once. The context neurons hear nothing (they are a source range of the block
transport, so their product is computed once per settling run) and their synapses into the
hidden neurons learn under the same free/nudged rule as every other synapse. Credit does not
flow back through time: the trace is a stimulus the rule sees, not a path it differentiates.
``stateful`` builds the connectome, ``Echo`` keeps the trace for a batch of streams.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import BrainState
from .connectome import Connectome
from .learning import embedded

__all__ = ["Trace", "Afterglow", "stateful", "Echo", "FastSynapses", "columns"]


def stateful(
    vocabulary: int,
    positions: int,
    dim: int,
    hidden: int,
    outputs: int,
    *,
    seed: int = 0,
    init: float = 1.0,
    context_init: float = 1.0,
) -> tuple[Connectome, np.ndarray]:
    """``embedded`` plus a context range of ``hidden`` neurons connected densely to the hidden ones.

    Neurons: ``positions`` blocks of ``vocabulary`` one-hot inputs, then ``hidden`` context
    neurons, then the embedding, hidden and output neurons of ``embedded``. Sets: ``input``,
    ``context``, ``embedding``, ``hidden``, ``output``. Returns the connectome and the tie groups
    of the embedding (pass to ``Learner(tie_groups=...)``).
    """
    base, base_tie = embedded(vocabulary, positions, dim, hidden, outputs, seed=seed, init=init)
    n_in = positions * vocabulary
    c0 = n_in  # the context range sits right after the inputs
    shift = hidden  # every non-input neuron of the base moves up by the context range

    def moved(index: np.ndarray) -> np.ndarray:
        return np.where(index >= n_in, index + shift, index)

    rng = np.random.default_rng(seed + 1)
    hidden_members = np.asarray(base.populations["hidden"]) + shift
    ctx, hid = np.meshgrid(np.arange(hidden), hidden_members, indexing="ij")
    magnitude = rng.uniform(0.0, 1.0, size=hidden * hidden) * np.sqrt(6.0 / (2 * hidden))
    signs = rng.choice([-1.0, 1.0], size=hidden * hidden) * magnitude * context_init
    pre = np.concatenate([moved(base.pre), c0 + ctx.ravel()])
    post = np.concatenate([moved(base.post), hid.ravel()])
    count = np.concatenate([base.count, np.ones(hidden * hidden)])
    sign = np.concatenate([base.sign, signs])
    tie = np.concatenate([base_tie, np.full(hidden * hidden, -1)])
    populations: dict[str, Iterable[int]] = {
        "input": range(0, n_in),
        "context": range(c0, c0 + hidden),
    }
    populations.update(
        {k: [int(i) + shift for i in v] for k, v in base.populations.items() if k != "input"}
    )
    connectome = Connectome.from_synapses(
        base.n + hidden,
        pre=pre,
        post=post,
        count=count,
        sign=sign,
        populations=populations,
        label=f"stateful:{positions}x{vocabulary}->{dim}->{hidden}(+{hidden} context)->{outputs}",
    )
    key = {(int(a), int(b)): int(g) for a, b, g in zip(pre, post, tie, strict=True)}
    groups = np.array(
        [key[(int(a), int(b))] for a, b in zip(connectome.pre, connectome.post, strict=True)],
        dtype=np.int64,
    )
    return connectome, groups


def columns(index: np.ndarray) -> slice | np.ndarray:
    """A slice when the neurons are one contiguous range, else the index array.

    Reading or writing the columns of a wide batch through a slice is a strided pass;
    through an index array it is a gather that comes back Fortran-ordered (and a scatter
    on write), which costs tens of times more on a batch of thousands of rows.
    """
    if len(index) and np.array_equal(index, np.arange(index[0], index[0] + len(index))):
        return slice(int(index[0]), int(index[0]) + len(index))
    return np.asarray(index, dtype=np.int64)


@dataclass
class Trace:
    """A quantity decaying across moments, kept per stream, entering the next settling run
    as a stimulus: the memory of the moment before.

    Over a range's activation it is the trace of that range (the Echo of the interpretation
    at ``focus`` 0); weighted by how much each neuron moved since the last moment (its share
    of the row's mean movement, to the power ``focus``) it is brightest where the moment
    changed, and of the input neurons it is an afterimage of the picture itself, the memory
    that reads a cue against a static background (``tests/test_child.py``). What had
    to change while settling stays lit; what stood still fades.
    """

    connectome: Connectome
    decay: float = 0.5
    amplitude: float = 1.0  # the stimulus amplitude of the neuron model
    focus: float = 0.0
    source: str = "hidden"  # the range traced
    # the range the trace enters as a stimulus, one neuron per source neuron
    target: str = "context"
    trace: np.ndarray = field(init=False)
    last: np.ndarray = field(init=False)
    cold: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("decay lies in [0, 1)")
        if not np.isfinite([self.amplitude, self.focus]).all() or self.focus < 0:
            raise ValueError("amplitude and focus must be finite; focus must be nonnegative")
        self.glow = np.asarray(self.connectome.populations[self.target], dtype=np.int64)
        self.hidden = np.asarray(self.connectome.populations[self.source], dtype=np.int64)
        if len(self.glow) != len(self.hidden):
            raise ValueError(f"one {self.target} neuron per {self.source} neuron")
        self._glow_columns = columns(self.glow)
        self._hidden_columns = columns(self.hidden)
        self.reset(0)

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Fresh for every stream, or for ``rows`` only (streams that begin again)."""
        if rows is None or len(self.trace) != batch:
            self.trace = np.zeros((batch, len(self.hidden)))
            self.last = np.zeros((batch, len(self.hidden)))
            self.cold = np.ones(batch, dtype=bool)
        else:
            self.trace[rows] = 0.0
            self.last[rows] = 0.0
            self.cold[rows] = True

    def keep(self, rows: np.ndarray) -> None:
        """Keep the traces of ``rows`` only (streams that ended are dropped)."""
        self.trace, self.last, self.cold = self.trace[rows], self.last[rows], self.cold[rows]

    def stimulate(self, drive: np.ndarray) -> np.ndarray:
        """Read the trace into a drive copy; another batch size reads zero context."""
        out = np.array(drive, dtype=float)
        out[:, self._glow_columns] = (
            self.amplitude * self.trace if len(self.trace) == len(out) else 0.0
        )
        return out

    def update(self, state: BrainState) -> None:
        """After the free phase: the trace decays toward the source neurons' activation, each
        weighted by its movement since the last moment when ``focus`` is above zero; a cold
        stream's first moment weighs one."""
        h = self._source_activation(state)
        if len(self.trace) != len(h):
            self.reset(len(h))
        if self.focus:
            moved = np.abs(h - self.last)
            weight = (moved / (moved.mean(axis=1, keepdims=True) + 1e-9)) ** self.focus
            weight[self.cold] = 1.0
            self.trace = self.decay * self.trace + (1.0 - self.decay) * weight * h
        else:
            self.trace = self.decay * self.trace + (1.0 - self.decay) * h
        self.last = h.copy()  # own the previous moment even if the caller reuses state storage
        self.cold[:] = False

    def _source_activation(self, state: BrainState) -> np.ndarray:
        """The source range's activation as a host array. A state that rests on a torch device
        is sliced there and only the slice comes to the host: for an afterimage of a retina
        of thousands of neurons that is the trace's whole cost."""
        device = getattr(state, "device", None)
        s = device.get("s") if isinstance(device, dict) else None
        if s is not None and hasattr(s, "device") and hasattr(s, "cpu"):
            cols = self._hidden_columns
            if isinstance(cols, slice):
                part = s[:, cols]
            else:
                import torch

                part = s[:, torch.as_tensor(np.asarray(cols), device=s.device)]
            return np.ascontiguousarray(part.detach().to("cpu").numpy().astype(float))
        return np.ascontiguousarray(np.atleast_2d(state.activation)[:, self._hidden_columns])

    def ringing(self, floor: float = 0.1) -> np.ndarray:
        """Each source neuron's trace magnitude over the row's mean magnitude, plus ``floor``;
        one for every other neuron: a salience for the eligibility of the synapses
        out of them, so that what is still ringing is what a signal writes through
        (``ActorCritic.salience``)."""
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        out = np.ones((len(self.trace), self.connectome.n))
        magnitude = np.abs(self.trace)
        mean = magnitude.mean(axis=1, keepdims=True) + 1e-9
        out[:, self._hidden_columns] = floor + magnitude / mean
        return out

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "focus": self.focus,
            "source": self.source,
            "target": self.target,
            "neurons": int(len(self.glow)),
        }


@dataclass
class Echo(Trace):
    """The Trace of the hidden equilibria at focus 0, into the ``context`` range."""

    def to_dict(self) -> dict[str, float | int | str]:
        return {"decay": self.decay, "amplitude": self.amplitude, "context": int(len(self.glow))}


@dataclass
class Afterglow(Trace):
    """The Trace with the afterglow's defaults: focused, into the ``afterglow`` range."""

    focus: float = 1.0
    target: str = "afterglow"

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "decay": self.decay,
            "amplitude": self.amplitude,
            "focus": self.focus,
            "source": self.source,
            "afterglow": int(len(self.glow)),
        }


@dataclass
class PatternSeparator:
    """Pattern separation: expand a key into a wider random code and keep the strongest winners.

    ``inputs`` key neurons project through a fixed random matrix onto ``expansion`` code
    neurons; the ``winners`` largest positive entries stay and the rest are zero. Expansion
    can reduce overlap between correlated keys; it does not guarantee separate codes.
    Codes with disjoint winners do not interfere: a delta write changes another key's read
    in proportion to their dot product. ``center`` > 0 keeps a running mean of the keys
    seen by ``observe`` (forgetting factor ``center``) and subtracts it first. Updating that
    mean can move the code of a previously stored key.
    """

    inputs: int
    expansion: int
    winners: int
    seed: int = 0
    center: float = 0.0
    projection: np.ndarray = field(init=False, repr=False)
    mean: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("inputs", "expansion", "winners"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.winners > self.expansion:
            raise ValueError("winners must not exceed the expansion")
        if not 0 <= self.center < 1:
            raise ValueError("center lies in [0, 1)")
        rng = np.random.default_rng(self.seed)
        self.projection = rng.standard_normal((self.inputs, self.expansion)) / np.sqrt(self.inputs)
        self.mean = np.zeros(self.inputs)

    def habituate(self, keys: np.ndarray) -> None:
        """Initialize the running mean from ``keys`` before storing records.

        With positive ``center``, later learning calls continue updating this mean;
        initialization does not freeze the codes used to address stored records.
        """
        x = np.asarray(keys, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all() or not len(x):
            raise ValueError(f"keys must be a finite nonempty (batch, {self.inputs}) array")
        self.mean = (x / len(x)).sum(axis=0)

    def code(self, key: np.ndarray, learn: bool = False) -> np.ndarray:
        """The sparse code of ``(batch, inputs)`` keys: ``(batch, expansion)``, ``winners``
        nonzero entries per row at most. ``learn`` updates the running mean (writes only)."""
        x = np.asarray(key, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"key must be a finite (batch, {self.inputs}) array")
        mean = self.mean
        if self.center > 0.0:
            if learn and len(x):
                mean = self.center * mean + (1.0 - self.center) * (x / len(x)).sum(axis=0)
            with np.errstate(over="ignore", invalid="ignore"):
                x = x - mean
        with np.errstate(over="ignore", invalid="ignore"):
            drive = np.maximum(x @ self.projection, 0.0)
        if not np.isfinite(mean).all() or not np.isfinite(drive).all():
            raise ValueError("pattern separation overflowed; scale the input keys")
        if learn:
            self.mean = mean
        if self.winners >= self.expansion:
            return np.asarray(drive)
        keep = np.argpartition(-drive, self.winners - 1, axis=1)[:, : self.winners]
        out: np.ndarray = np.zeros_like(drive)
        np.put_along_axis(out, keep, np.take_along_axis(drive, keep, axis=1), axis=1)
        return np.asarray(out)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "inputs": self.inputs,
            "expansion": self.expansion,
            "winners": self.winners,
            "seed": self.seed,
            "center": self.center,
        }


@dataclass
class FastSynapses:
    """A bounded associative memory between two ranges, one matrix per stream.

    ``rule="hebb"`` adds an outer product, preserving the original behaviour.
    ``rule="delta"`` reads before writing: for a unit key ``k``,
    ``M += rate * outer(k, value - k @ M)``. Repeated evidence then stops changing
    a correct prediction. A rate-one write corrects this key's unscaled prediction; other
    nonorthogonal keys can interfere. This is the established delta/LMS rule,
    not a guarantee of unlimited capacity or learned memory addressing.

    ``observe``/``recall`` operate directly on key/value ports. ``update``/``read``
    adapt the same operations to settled states and full drives. Key normalisation
    reads the whole key vector; a synapse uses its key coordinate and the post neuron's
    prediction error. Strengths decay once per observation/update, never on reads.
    """

    pre: np.ndarray
    post: np.ndarray
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    normalize: bool = False  # unit keys and cue, the read divided by the decayed count of writes
    replace: bool = False  # a write clears what its active pre neurons held (a slot; one-hot keys)
    rule: str = "hebb"  # "delta": unit keys, residual writes, no count-averaged read
    separator: PatternSeparator | None = None  # pattern separation of the keys before use
    strength: np.ndarray = field(init=False)
    mass: np.ndarray = field(init=False)  # (batch,) the decayed count of writes, for ``normalize``
    writes: int = 0

    def __post_init__(self) -> None:
        for name in ("pre", "post"):
            raw = np.asarray(getattr(self, name))
            if (
                raw.dtype.kind not in "iuf"
                or not np.isfinite(raw).all()
                or (raw != np.floor(raw)).any()
            ):
                raise ValueError(f"{name} must contain integer neuron indices")
        self.pre = np.asarray(self.pre, dtype=np.int64)
        self.post = np.asarray(self.post, dtype=np.int64)
        for name, neurons in (("pre", self.pre), ("post", self.post)):
            if neurons.ndim != 1 or not len(neurons) or np.any(neurons < 0):
                raise ValueError(f"{name} must be a nonempty vector of neuron indices")
            if len(np.unique(neurons)) != len(neurons):
                raise ValueError(f"{name} neuron indices must be unique")
        if not 0 <= self.decay <= 1:
            raise ValueError("decay lies in [0, 1]")
        if not np.isfinite(self.rate) or self.rate < 0 or not np.isfinite(self.amplitude):
            raise ValueError("rate must be finite and nonnegative; amplitude must be finite")
        if self.rule not in ("hebb", "delta"):
            raise ValueError("rule must be 'hebb' or 'delta'")
        if self.rule == "delta":
            if self.rate > 1:
                raise ValueError("delta rate lies in [0, 1]")
            if self.normalize or self.replace:
                raise ValueError(
                    "delta already normalises keys; normalize/replace are Hebbian modes"
                )
        if self.separator is not None and self.separator.inputs != len(self.pre):
            raise ValueError("the separator's inputs must equal the number of pre neurons")
        self._pre_columns = columns(self.pre)
        self._post_columns = columns(self.post)
        self.strength = np.zeros((0, self.key_width, len(self.post)))
        self.mass = np.zeros(0)

    @property
    def key_width(self) -> int:
        """Rows of the strength matrix: the code width with a separator, else the pre count."""
        return self.separator.expansion if self.separator is not None else int(len(self.pre))

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """Clear all streams, or only the selected streams at episode boundaries."""
        if rows is None or len(self.strength) != batch:
            self.strength = np.zeros((batch, self.key_width, len(self.post)))
            self.mass = np.zeros(batch)
        else:
            self.strength[rows] = 0.0
            self.mass[rows] = 0.0

    def keep(self, rows: np.ndarray) -> None:
        self.strength = self.strength[rows]
        self.mass = self.mass[rows]

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        # Preserve the original Hebbian normalisation, including its small-norm floor.
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12))

    @staticmethod
    def _delta_unit(x: np.ndarray) -> np.ndarray:
        # Scale first so even very small/large finite keys have a well-defined direction.
        scale = np.max(np.abs(x), axis=-1, keepdims=True)
        scaled = x / np.where(scale > 0, scale, 1.0)
        norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
        return np.asarray(scaled / np.where(norm > 0, norm, 1.0))

    @staticmethod
    def _port(values: np.ndarray, width: int, name: str) -> np.ndarray:
        out = np.asarray(values, dtype=float)
        if out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all():
            raise ValueError(f"{name} must be a finite (batch, {width}) array")
        return out

    def recall(self, key: np.ndarray) -> np.ndarray:
        """Read ``(batch, pre)`` keys as ``(batch, post)`` values without changing memory.

        A different batch size reads the baseline without erasing live streams. Delta uses
        unit keys and does not divide by the number of writes. ``amplitude``
        scales the returned drive, not the residual used when learning a value.
        """
        cue = self._port(key, len(self.pre), "key")
        same_batch = len(self.strength) == len(cue)
        strength = self.strength if same_batch else self._baseline()[None, :, :]
        if self.separator is not None:
            cue = self.separator.code(cue)
        if self.rule == "delta":
            cue = self._delta_unit(cue)
        elif self.normalize:
            cue = self._unit(cue)
        out = (cue[:, None, :] @ strength)[:, 0, :]
        if self.normalize:
            mass = self.mass if same_batch else np.zeros(len(cue))
            out = out / np.maximum(mass, 1e-12)[:, None]
        return np.asarray(self.amplitude * out)

    def _baseline(self) -> np.ndarray:
        """The unobserved stream's readback, without changing any stored stream."""
        return np.zeros((self.key_width, len(self.post)))

    def observe(self, key: np.ndarray, value: np.ndarray, write: np.ndarray | None = None) -> None:
        """Decay once, then write selected rows from the key and observed-value ports.

        Inputs are ``(batch, pre)`` and ``(batch, post)``. ``write=None`` writes
        every row; a boolean mask gates individual streams. A zero key cannot
        write a delta association. Validation happens before any state change.
        For prediction tasks call ``recall`` before revealing the new value.
        """
        key = self._port(key, len(self.pre), "key")
        value = self._port(value, len(self.post), "value")
        batch = len(key)
        if len(value) != batch:
            raise ValueError("key and value batches must match")
        gate = np.ones(batch, dtype=bool) if write is None else np.asarray(write)
        if gate.shape != (batch,) or gate.dtype != np.bool_:
            raise ValueError("write must be a boolean vector with one entry per stream")
        same_batch = len(self.strength) == batch
        strength = (
            self.strength * self.decay
            if same_batch
            else np.zeros((batch, self.key_width, len(self.post)))
        )
        mass = self.mass * self.decay if same_batch else np.zeros(batch)
        if self.rule == "delta":
            gate = gate & np.any(key != 0.0, axis=1)
        rows = np.flatnonzero(gate)
        if not len(rows):
            self.strength, self.mass = strength, mass
            return
        a, b = key[rows], value[rows]
        previous_mean = None if self.separator is None else self.separator.mean
        if self.separator is not None:
            a = self.separator.code(a, learn=True)
        if self.rule == "delta":
            a = self._delta_unit(a)
        elif self.normalize:
            a = self._unit(a)
        with np.errstate(over="ignore", invalid="ignore"):
            if self.rule == "delta":
                b = b - (a[:, None, :] @ strength[rows])[:, 0, :]
            elif self.replace:
                strength[rows] *= (a <= 0.0)[:, :, None]
            strength[rows] += self.rate * a[:, :, None] * b[:, None, :]
            mass[rows] += self.rate
        if not np.isfinite(strength).all() or not np.isfinite(mass).all():
            if self.separator is not None and previous_mean is not None:
                self.separator.mean = previous_mean
            raise ValueError("synaptic update overflowed; scale the observed keys and values")
        self.strength, self.mass = strength, mass
        self.writes += len(rows)

    def read(self, drive: np.ndarray) -> np.ndarray:
        """The post neurons' drive from the pre range's stimulus in ``drive``: ``(batch, post)``.

        Normalized Hebbian mode uses unit keys and cue, then divides the read by the
        decayed count of writes. Cosine weights can be signed, so this need not be a
        convex average. ``rate`` and ``amplitude`` also scale the returned values.
        """
        return self.recall(drive[:, self._pre_columns])

    def stimulate(self, drive: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Add the read to the post columns of ``drive`` (a copy, unless ``inplace``)."""
        out = drive if inplace else np.array(drive, dtype=float)
        out[:, self._post_columns] += self.read(out)
        return out

    def update(
        self,
        state: BrainState,
        write: np.ndarray | None = None,
        post: np.ndarray | None = None,
    ) -> None:
        """After settling: strengths fade; the rows in ``write`` add their outer product.

        ``post`` replaces the post neurons' activations for the write, ``(batch, post)``: what
        actually followed (the next symbols read) rather than what the net settled on.
        """
        s = np.atleast_2d(state.activation)
        self.observe(
            s[:, self._pre_columns],
            s[:, self._post_columns] if post is None else post,
            np.zeros(len(s), dtype=bool) if write is None else write,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre": int(len(self.pre)),
            "post": int(len(self.post)),
            "decay": self.decay,
            "rate": self.rate,
            "amplitude": self.amplitude,
            "normalize": self.normalize,
            "replace": self.replace,
            "rule": self.rule,
            "separator": None if self.separator is None else self.separator.to_dict(),
            "writes": self.writes,
        }
