"""Experimental fixed connectivity over the existing temporal energy and solver.

Components are coordinate groups of one jointly repaired recurrent state. The
mask is supplied architecture, not learned specialization. All connections are
one-time-step recurrent connections; this helper adds no same-time implicit
coupling, clock, composer policy, new energy or learning direction.

Masked gradients are computed BEFORE public parameter backtracking admission.
Snapshots retain masks in additional metadata. The core's private base-class
replay may ignore that metadata safely because it only evaluates already legal
candidate arrays. Restore through this subclass to preserve future enforcement.

TemporalMemory.observe explicitly rejects this subclass before mutation.
TemporalMemory.protect can collect response constraints, but that does not make
protected learning compatible with these masks. No combined transaction is
provided. Fixed routing is an experimental capacity control, not evidence of
learned specialization or a behavioral advantage.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..temporal import TemporalPatchNet, TemporalPhase

__all__ = ["PartitionedTemporalPatchNet", "two_group_masks"]


def _masks(
    value: Mapping[str, np.ndarray] | None, inputs: int, hidden: int, outputs: int
) -> dict[str, np.ndarray]:
    shapes = {"A": (hidden, hidden), "B": (hidden, inputs), "C": (outputs, hidden)}
    if value is None:
        value = {k: np.ones(shape, dtype=bool) for k, shape in shapes.items()}
    if not isinstance(value, Mapping) or set(value) != set(shapes):
        raise ValueError("masks must contain exactly A, B and C")
    result = {}
    for key, shape in shapes.items():
        mask = np.asarray(value[key])
        if mask.dtype.kind != "b" or mask.shape != shape:
            raise ValueError(f"{key} mask must be boolean with shape {shape}")
        result[key] = mask.copy()
        result[key].flags.writeable = False
    return result


def _legal(parameters: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray]) -> None:
    if not isinstance(parameters, Mapping) or set(parameters) != set(masks):
        raise ValueError("parameters must contain exactly A, B and C")
    for key, mask in masks.items():
        value = np.asarray(parameters[key])
        if value.dtype.kind not in "fiu" or value.shape != mask.shape:
            raise ValueError(f"{key} has invalid type or shape")
        if not np.isfinite(value).all() or np.any(value[~mask] != 0):
            raise ValueError(f"{key} must be finite and zero outside its connectivity mask")


class PartitionedTemporalPatchNet(TemporalPatchNet):
    """One temporal net with immutable, explicitly supplied A/B/C connectivity.

    Initialization draws the same dense arrays as TemporalPatchNet, then sets
    forbidden entries to zero. With all-true masks, numerical phases, parameter
    updates, carried state and revision counts equal the ordinary dense net.
    Its snapshot additionally records the masks. Arbitrary masking does not
    promise a prescribed spectral radius or a faster dense block solve.
    """

    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        masks: Mapping[str, np.ndarray] | None = None,
        **options: Any,
    ) -> None:
        super().__init__(inputs, hidden, outputs, **options)
        self._connectivity = _masks(masks, self.inputs, self.hidden, self.outputs)
        for key, mask in self._connectivity.items():
            setattr(self, "_" + key, np.where(mask, getattr(self, "_" + key), 0.0))

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Detached boolean connectivity; mutating a returned copy has no effect."""
        return {key: value.copy() for key, value in self._connectivity.items()}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(np.count_nonzero(value)) for value in self._connectivity.values())

    def set_parameters(self, parameters: Mapping[str, np.ndarray]) -> None:
        _legal(parameters, self._connectivity)
        super().set_parameters(parameters)

    def _parameter_gradient(
        self, inputs: np.ndarray, boundary: np.ndarray, phase: TemporalPhase
    ) -> dict[str, np.ndarray]:
        gradient = super()._parameter_gradient(inputs, boundary, phase)
        return {
            key: np.where(self._connectivity[key], value, 0.0) for key, value in gradient.items()
        }

    def snapshot(self) -> dict[str, np.ndarray]:
        result = super().snapshot()
        metadata = json.loads(str(result["meta"]))
        metadata["partitioned_temporal"] = {
            "format": "cadence-fixed-connectivity/1",
            "masks": {key: value.tolist() for key, value in self._connectivity.items()},
        }
        result["meta"] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
        return result

    @classmethod
    def restore(cls, snapshot: Mapping[str, np.ndarray]) -> PartitionedTemporalPatchNet:
        try:
            metadata = json.loads(str(snapshot["meta"]))
            extension = metadata["partitioned_temporal"]
            if (
                not isinstance(extension, dict)
                or set(extension) != {"format", "masks"}
                or extension["format"] != "cadence-fixed-connectivity/1"
            ):
                raise ValueError("invalid fixed-connectivity metadata")
            # Validate the complete ordinary continuation state once. Its
            # parser ignores the extra metadata; it never trains this clone.
            base = TemporalPatchNet.restore(snapshot)
            masks = _masks(extension["masks"], base.inputs, base.hidden, base.outputs)
            _legal(base.parameters(), masks)
            result = cls.__new__(cls)
            result.__dict__.update(base.__dict__)
            result._connectivity = masks
            return result
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid partitioned temporal snapshot") from error

    @classmethod
    def load(cls, path: str | Path) -> PartitionedTemporalPatchNet:
        """Restore the masks and continuation using the inherited archive reader."""
        return cast("PartitionedTemporalPatchNet", super().load(path))


def two_group_masks(
    context_hidden: int,
    motor_hidden: int,
    inputs: int,
    outputs: int,
    *,
    context_inputs: Iterable[int],
    motor_inputs: Iterable[int],
    cross_coupling: bool = True,
) -> dict[str, np.ndarray]:
    """Declared two-group wiring: separate input access, motor-only readout.

    Within-group recurrence is dense. Cross-coupling enables both delayed
    recurrent directions; it is not a hand-written semantic plan. Input sets
    can overlap for a genuinely shared observation. Empty sets are allowed.
    The returned masks are ordinary explicit arrays, not new runtime state.
    """
    for name, value in [
        ("context_hidden", context_hidden),
        ("motor_hidden", motor_hidden),
        ("inputs", inputs),
        ("outputs", outputs),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(cross_coupling, (bool, np.bool_)):
        raise ValueError("cross_coupling must be boolean")
    groups = []
    for indices in (context_inputs, motor_inputs):
        values = list(indices)
        if any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or i < 0
            or i >= inputs
            for i in values
        ):
            raise ValueError("input index outside declared ports")
        if len(set(values)) != len(values):
            raise ValueError("duplicate input port")
        groups.append(values)
    hidden = context_hidden + motor_hidden
    a = np.ones((hidden, hidden), dtype=bool)
    if not cross_coupling:
        a[:context_hidden, context_hidden:] = False
        a[context_hidden:, :context_hidden] = False
    b = np.zeros((hidden, inputs), dtype=bool)
    b[:context_hidden, groups[0]] = True
    b[context_hidden:, groups[1]] = True
    c = np.zeros((outputs, hidden), dtype=bool)
    c[:, context_hidden:] = True
    return {"A": a, "B": b, "C": c}
