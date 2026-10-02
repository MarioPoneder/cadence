"""Two record patches in depth.

A record patch's gate sees only the present input. It can hold a word, but it cannot tell
the subject of a sentence from a noun inside a later phrase, because both arrive as the same
input. A second patch that reads the first patch's context as an input has a gate that
sees what came before, and that is enough: on subject-verb agreement across two
distracting nouns one patch scored 0.03 and the stack 1.00 (the probe is described in
``docs/record-patch.md``).

The lower patch is a context alone::

    l1[t] = sigmoid(g1 + G1 u[t]),  z1[t] = tanh(B1 u[t] + b1)
    h1[t] = l1[t] * h1[t-1] + (1 - l1[t]) * z1[t]

The upper patch is a complete ``RecordPatchNet`` over the inputs ``[u[t], r1 * h1[t]]``: it
owns the readout and the records. The slow gradient is one adjoint scan per patch; the upper
scan hands its input gradient to the lower scan. The record read stays outside it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ports import StructuredPort
from .record_patch import RecordPatchNet, RecordPath

FORMAT = "cadence-record-stack/1"
FORMAT_PORTS = "cadence-record-stack/2"  # a structured port on the lower patch
_LOWER = ("G1", "g1", "B1", "b1")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class StackObservation:
    """One teaching attempt of the stack: the upper patch's prediction and what changed."""

    updated: bool
    reason: str
    prediction: RecordPath
    delta: dict[str, np.ndarray]
    initial_loss: float | None
    final_loss: float | None
    accepted_rate: float
    writes: int


class RecordPatchStack:
    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        lower: int | None = None,
        seed: int = 0,
        slowest: float = 128.0,
        groups: Sequence[int] | None = None,
        lower_port: StructuredPort | None = None,
        **upper: Any,
    ) -> None:
        self.inputs = int(inputs)
        # A structured lower port (maps over grids of the inputs) fixes the lower width.
        self.lower_port = lower_port
        if lower_port is not None:
            if lower_port.inputs != self.inputs:
                raise ValueError("the lower port reads a different number of inputs")
            if lower is not None and int(lower) != lower_port.outputs:
                raise ValueError(
                    f"lower must equal the lower port's outputs ({lower_port.outputs})"
                )
            lower = lower_port.outputs
        self.lower = int(hidden if lower is None else lower)
        if self.inputs < 1 or self.lower < 1:
            raise ValueError("inputs and lower must be positive")
        rng = np.random.default_rng(seed + 1)
        timescales = np.exp(np.linspace(np.log(2.0), np.log(float(slowest)), self.lower))
        retention = 1.0 - 1.0 / timescales
        self._g1 = np.log(retention / (1.0 - retention))
        self._scale = np.sqrt((1.0 + retention) / (1.0 - retention))
        if lower_port is None:
            self._G1 = np.zeros((self.lower, self.inputs))
            self._B1 = rng.normal(size=(self.lower, self.inputs)) / np.sqrt(self.inputs)
        else:
            self._G1 = self._pack(lower_port.initial(rng, 0.0))
            self._B1 = self._pack(lower_port.initial(rng, 1.0))
        self._b1 = np.zeros(self.lower)
        self.upper = RecordPatchNet(
            self.inputs + self.lower,
            hidden,
            outputs,
            seed=seed,
            slowest=slowest,
            groups=groups,
            **upper,
        )
        self._state: np.ndarray | None = None

    # ------------------------------------------------------------ the lower patch
    def _pack(self, blocks: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([b.ravel() for b in blocks])

    def _unpack(self, flat: np.ndarray) -> list[np.ndarray]:
        assert self.lower_port is not None
        out, at = [], 0
        for b in self.lower_port.blocks:
            shape = self.lower_port.weight_shape(b)
            size = int(np.prod(shape))
            out.append(flat[at : at + size].reshape(shape))
            at += size
        return out

    def _drive(self, inputs: np.ndarray, which: str) -> np.ndarray:
        weights = self._B1 if which == "B" else self._G1
        if self.lower_port is None:
            return np.asarray(inputs @ weights.T)
        return self.lower_port.apply(inputs, self._unpack(weights))

    def _drive_gradient(self, v: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        if self.lower_port is None:
            return np.asarray(np.einsum("bti,btj->ij", v, inputs))
        return self._pack(self.lower_port.gradient(v, inputs))

    def _context(
        self, inputs: np.ndarray, boundary: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gate = _sigmoid(self._g1 + self._drive(inputs, "G"))
        port = np.tanh(self._drive(inputs, "B") + self._b1)
        hidden = np.empty((len(inputs), inputs.shape[1], self.lower))
        previous = boundary
        for t in range(inputs.shape[1]):
            hidden[:, t] = gate[:, t] * previous + (1.0 - gate[:, t]) * port[:, t]
            previous = hidden[:, t]
        return hidden, gate, port

    def _boundary(self, batch: int, state: np.ndarray | None) -> np.ndarray:
        source = self._state if state is None else state
        if source is None:
            return np.zeros((batch, self.lower))
        value = np.asarray(source, dtype=float)
        if value.shape != (batch, self.lower) or not np.isfinite(value).all():
            raise ValueError("lower state must be a finite (batch, lower) array")
        return value.copy()

    def _above(self, inputs: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        return np.concatenate((inputs, hidden * self._scale), axis=-1)

    def parameters(self) -> dict[str, np.ndarray]:
        lower = {k: getattr(self, "_" + k).copy() for k in _LOWER}
        return {**lower, **self.upper.parameters()}

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        values = {k: np.asarray(parameters[k], dtype=float) for k in _LOWER}
        for key, value in values.items():
            if value.shape != getattr(self, "_" + key).shape or not np.isfinite(value).all():
                raise ValueError(
                    f"{key} must be finite with shape {getattr(self, '_' + key).shape}"
                )
        self.upper.set_parameters({k: v for k, v in parameters.items() if k not in _LOWER})
        for key, value in values.items():
            setattr(self, "_" + key, value.copy())

    # ------------------------------------------------------------ interface
    def reset(self) -> None:
        self._state = None
        self.upper.reset()

    def imagine(
        self,
        inputs: np.ndarray,
        *,
        state: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RecordPath:
        """Free private rollout of both patches; nothing changes."""
        path = self.upper._path(inputs, self.inputs, "inputs")
        boundary = self._boundary(len(path), None if state is None else state[0])
        hidden, _, _ = self._context(path, boundary)
        return self.upper.imagine(
            self._above(path, hidden), state=None if state is None else state[1]
        )

    def advance(self, inputs: np.ndarray) -> RecordPath:
        path = self.upper._path(inputs, self.inputs, "inputs")
        hidden, _, _ = self._context(path, self._boundary(len(path), None))
        self._state = hidden[:, -1].copy()
        return self.upper.advance(self._above(path, hidden))

    def _slow_loss(
        self, inputs: np.ndarray, target: np.ndarray, b1: np.ndarray, b2: np.ndarray
    ) -> float | None:
        hidden, _, _ = self._context(inputs, b1)
        return self.upper._free(self._above(inputs, hidden), b2, target)[0].slow_loss

    def observe(
        self,
        inputs: np.ndarray,
        target: np.ndarray,
        *,
        rate: float = 1.0,
        backtrack: bool = False,
        write: bool = True,
    ) -> StackObservation:
        """Learn one observed path through both patches and write the upper patch's records.

        With ``backtrack=True`` a joint step is admitted only after a target-free replay of
        both patches from the original boundaries lowers the slow loss, trying up to sixteen
        halved rates."""
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        upper = self.upper
        path = upper._path(inputs, self.inputs, "inputs")
        b1 = self._boundary(len(path), None)
        hidden, gate, port = self._context(path, b1)
        above, teaching = upper._teaching(self._above(path, hidden), target)
        b2 = upper._boundary(len(path), None)
        prediction, port2, _ = upper._free(above, b2, teaching)
        self._state = hidden[:, -1].copy()
        upper._carry(prediction)
        if prediction.slow_loss is None:
            return StackObservation(
                False, "nonfinite_prediction", prediction, {}, None, None, 0.0, 0
            )
        delta, gu = upper._adjoint(
            above, b2, prediction, port2, upper._output_error(prediction, teaching)
        )
        gh = gu[..., self.inputs :] * self._scale
        carried = np.zeros((len(path), self.lower))
        for t in range(path.shape[1] - 1, -1, -1):
            gh[:, t] += carried
            carried = gate[:, t] * gh[:, t]
        previous = np.concatenate((b1[:, None], hidden[:, :-1]), axis=1)
        gp = (1.0 - gate) * gh * (1.0 - port**2)
        gs = gh * (previous - port) * gate * (1.0 - gate)
        delta.update(
            G1=self._drive_gradient(gs, path),
            g1=gs.sum(axis=(0, 1)),
            B1=self._drive_gradient(gp, path),
            b1=gp.sum(axis=(0, 1)),
        )
        writes = upper._write(above, teaching, prediction.hidden) if write else 0
        current, initial = self.parameters(), prediction.slow_loss
        norm_squared = sum(float(np.sum(v * v)) for v in delta.values())
        if rate == 0 or norm_squared == 0:
            return StackObservation(
                False, "no_step", prediction, delta, initial, initial, 0.0, writes
            )
        for index in range(16 if backtrack else 1):
            step = rate * 0.5**index
            proposed = {k: current[k] - step * delta[k] for k in delta}
            if not all(np.isfinite(v).all() for v in proposed.values()):
                continue
            self.set_parameters(proposed)
            if not backtrack:
                upper.updates += 1
                return StackObservation(
                    True, "updated", prediction, delta, initial, None, step, writes
                )
            loss = self._slow_loss(path, teaching, b1, b2)
            if loss is not None and loss <= initial - 1e-4 * step * norm_squared:
                upper.updates += 1
                return StackObservation(
                    True, "updated", prediction, delta, initial, loss, step, writes
                )
        self.set_parameters(current)
        return StackObservation(
            False, "no_decreasing_parameter_step", prediction, delta, initial, initial, 0.0, writes
        )

    # ------------------------------------------------------------ checkpoints
    def snapshot(self) -> dict[str, np.ndarray]:
        meta: dict[str, Any] = {"format": FORMAT, "inputs": self.inputs, "lower": self.lower}
        if self.lower_port is not None:
            meta["format"] = FORMAT_PORTS
            meta["lower_port"] = self.lower_port.to_dict()
        result = {"upper_" + k: v for k, v in self.upper.snapshot().items()}
        result.update({k: getattr(self, "_" + k).copy() for k in _LOWER})
        result["lower_scale"] = self._scale.copy()
        result["lower_state"] = (
            np.empty((0, self.lower)) if self._state is None else self._state.copy()
        )
        result["meta"] = np.array(json.dumps(meta, sort_keys=True))
        return result

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> RecordPatchStack:
        try:
            meta = json.loads(str(snapshot["meta"]))
            if meta.get("format") not in (FORMAT, FORMAT_PORTS):
                raise ValueError("unsupported record-stack checkpoint")
            upper = RecordPatchNet.restore(
                {k[len("upper_") :]: v for k, v in snapshot.items() if k.startswith("upper_")}
            )
            result = cls.__new__(cls)
            result.inputs, result.lower, result.upper = (
                int(meta["inputs"]),
                int(meta["lower"]),
                upper,
            )
            result.lower_port = (
                None
                if meta.get("lower_port") is None
                else StructuredPort.from_dict(meta["lower_port"])
            )
            if upper.inputs != result.inputs + result.lower:
                raise ValueError("the upper patch does not read this lower patch")
            packed = (
                None
                if result.lower_port is None
                else (
                    int(
                        sum(
                            int(np.prod(result.lower_port.weight_shape(b)))
                            for b in result.lower_port.blocks
                        )
                    ),
                )
            )
            shapes = {
                "G1": (result.lower, result.inputs) if packed is None else packed,
                "g1": (result.lower,),
                "B1": (result.lower, result.inputs) if packed is None else packed,
                "b1": (result.lower,),
            }
            for key, shape in shapes.items():
                value = np.asarray(snapshot[key], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid saved {key}")
                setattr(result, "_" + key, value.copy())
            result._scale = np.asarray(snapshot["lower_scale"], dtype=float).copy()
            state = np.asarray(snapshot["lower_state"], dtype=float)
            result._state = None if not len(state) else state.copy()
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid record-stack checkpoint") from error
