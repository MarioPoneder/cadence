"""General observation encoders and optional environment calibration.

Encoders are pure context functions. They neither learn targets nor choose
outputs. Numeric grids accept declared ranges in any units; calibration can
estimate ranges and rank coordinates from any finite numeric observation vector.
No game package is imported by this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any

from ._validation import boolean, integer, number


@dataclass(frozen=True)
class IdentityFeatures:
    """Use the complete observation as context (Cortex owns a typed copy)."""

    def __call__(self, observation: Any) -> Any:
        return observation

    def specification(self) -> dict:
        """Return the portable encoder description used by checkpoints."""
        return {"type": "identity"}


@dataclass(frozen=True)
class ConstantFeatures:
    """A single shared context, useful as a coarse prior or global estimator."""

    cells = 1

    def __call__(self, observation: Any) -> tuple:
        return ()

    def specification(self) -> dict:
        """Return the portable encoder description used by checkpoints."""
        return {"type": "constant"}


class BinnedFeatures:
    """Quantize selected numeric coordinates to a finite context tuple.

    Parameters
    ----------
    bounds : sequence of (low, high)
        One finite, strictly increasing interval per selected coordinate.
    bins : int or sequence of int, default 8
        Number of bins for each coordinate; each count is at least one.
    indices : sequence of int or None
        Observation coordinates to read. None selects 0 through len(bounds)-1.
    clip : bool, default True
        Saturate outside each interval. False rejects out-of-range inputs.

    A scalar observation is accepted for the one-coordinate, index-zero case.
    ``cells`` is the product of bin counts and enables witness-mass weighting.
    Bin counts may not exceed 2**31. Configuration is immutable.
    """

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise AttributeError("Feature configuration is fixed at construction")
        super().__setattr__(name, value)

    def __init__(
        self, bounds: Sequence[Sequence[float]], bins=8, indices=None, *, clip=True
    ):
        try:
            ranges = tuple(tuple(pair) for pair in bounds)
        except TypeError as error:
            raise ValueError("bounds must contain (low, high) pairs") from error
        if not ranges or any(len(pair) != 2 for pair in ranges):
            raise ValueError("bounds must contain (low, high) pairs")
        self.bounds = tuple(
            (number(lo, "lower bound"), number(hi, "upper bound")) for lo, hi in ranges
        )
        if any(lo >= hi or not math.isfinite(hi - lo) for lo, hi in self.bounds):
            raise ValueError("Every bound must have a finite positive span")
        if isinstance(bins, int) and not isinstance(bins, bool):
            counts = (bins,) * len(ranges)
        else:
            try:
                counts = tuple(bins)
            except TypeError as error:
                raise ValueError(
                    "bins must be an integer or sequence of integers"
                ) from error
        self.bins = tuple(integer(b, "bins", 1) for b in counts)
        if any(b > 2**31 for b in self.bins):
            raise ValueError("bins must not exceed 2**31")
        selected = tuple(range(len(ranges))) if indices is None else tuple(indices)
        self.indices = tuple(integer(i, "coordinate index") for i in selected)
        if len(self.bins) != len(ranges) or len(self.indices) != len(ranges):
            raise ValueError("bounds, bins and indices must have the same length")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Coordinate indices must be distinct")
        self.clip = boolean(clip, "clip")
        self.cells = math.prod(self.bins)

    def __call__(self, observation: Any) -> tuple[int, ...]:
        try:
            values = [observation[i] for i in self.indices]
        except (TypeError, IndexError, KeyError) as error:
            if self.indices == (0,):
                values = [observation]
            else:
                raise ValueError(
                    "Observation lacks the selected coordinates"
                ) from error
        output = []
        for value, (lo, hi), bins in zip(values, self.bounds, self.bins, strict=True):
            value = number(value, "observation coordinate")
            if not self.clip and not lo <= value <= hi:
                raise ValueError("Observation coordinate outside its declared bounds")
            if value <= lo:
                output.append(0)
            elif value >= hi:
                output.append(bins - 1)
            else:
                output.append(min(bins - 1, int((value - lo) / (hi - lo) * bins)))
        return tuple(output)

    def specification(self) -> dict:
        """Return a JSON-compatible description; no executable code is stored."""
        return {
            "type": "binned",
            "bounds": [list(p) for p in self.bounds],
            "bins": list(self.bins),
            "indices": list(self.indices),
            "clip": self.clip,
        }


def features_from_specification(specification: dict):
    """Reconstruct a built-in encoder, rejecting unknown fields and types."""
    if type(specification) is not dict:
        raise ValueError("Invalid feature specification")
    if specification == {"type": "identity"}:
        return IdentityFeatures()
    if specification == {"type": "constant"}:
        return ConstantFeatures()
    if (
        set(specification) == {"type", "bounds", "bins", "indices", "clip"}
        and specification["type"] == "binned"
    ):
        return BinnedFeatures(
            specification["bounds"],
            specification["bins"],
            specification["indices"],
            clip=specification["clip"],
        )
    raise ValueError("Unknown feature specification")


def grid(bounds, *, bins=8, depth: int = 1, clip: bool = True) -> tuple:
    """Build a fine grid, ``depth-1`` coarsenings, and a global context.

    ``depth=0`` returns only the fine grid. ``depth=1`` adds the global
    context. Additional levels halve each coordinate's bins, down to one.
    All levels read the same coordinates; no sensor dimension is hand-picked.
    """
    depth = integer(depth, "depth")
    if depth > 32:
        raise ValueError("depth must not exceed 32")
    fine = BinnedFeatures(bounds, bins, clip=clip)
    levels = [fine]
    for level in range(1, depth):
        levels.append(
            BinnedFeatures(
                fine.bounds,
                tuple(max(1, b // (2**level)) for b in fine.bins),
                clip=clip,
            )
        )
    if depth:
        levels.append(ConstantFeatures())
    return tuple(levels)


def calibrate(
    env_factory: Callable,
    n_actions: int,
    *,
    steps: int = 400,
    seed: int = 0,
    observation_fn: Callable | None = None,
) -> dict:
    """Measure coordinate ranges and action-conditioned variation.

    The factory must create a reset/step environment using the five-result
    step convention. Each action receives the same reset-seed schedule; every
    environment is closed even on failure. ``observation_fn`` optionally
    projects dictionaries, images or other observations to a finite vector.
    Calibration spends ``steps * n_actions`` interactions and does not train
    a Cortex. Action separation is a diagnostic, not a causal identification.
    """
    n_actions, steps, seed = (
        integer(n_actions, "n_actions", 1),
        integer(steps, "steps", 1),
        integer(seed, "seed"),
    )
    if (
        not callable(env_factory)
        or observation_fn is not None
        and not callable(observation_fn)
    ):
        raise ValueError("env_factory and observation_fn must be callable")
    traces, length = [], None

    def vector(observation):
        nonlocal length
        projected = observation_fn(observation) if observation_fn else observation
        try:
            values = [number(v, "calibration coordinate") for v in projected]
        except TypeError as error:
            raise ValueError(
                "Calibration requires a numeric observation vector"
            ) from error
        if not values or length is not None and len(values) != length:
            raise ValueError("Calibration observation dimension changed or is empty")
        length = len(values)
        return values

    for action in range(n_actions):
        env = env_factory()
        try:
            episode = 0
            observation, _ = env.reset(seed=seed)
            rows = [vector(observation)]
            for _ in range(steps):
                observation, _, terminated, truncated, _ = env.step(action)
                rows.append(vector(observation))
                if terminated or truncated:
                    episode += 1
                    observation, _ = env.reset(seed=seed + episode)
                    rows.append(vector(observation))
            traces.append(rows)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    controllability, drive, bounds = [], [], []
    for coordinate in range(length):
        series = [[row[coordinate] for row in rows] for rows in traces]
        means = [fmean(s) for s in series]
        pooled = fmean(pstdev(s) for s in series)
        separation = (max(means) - min(means)) / (1.0 + pooled)
        if not math.isfinite(separation) or not math.isfinite(pooled):
            raise ValueError("Calibration statistics exceed the finite numeric range")
        controllability.append(separation)
        drive.append(pooled)
        lo, hi = min(min(s) for s in series), max(max(s) for s in series)
        if lo == hi:
            span = max(0.5, abs(lo) * 1e-6)
            lo, hi = lo - span, hi + span
        bounds.append((lo, hi))
    ctrl = sorted(range(length), key=lambda i: controllability[i], reverse=True)
    drv = sorted(range(length), key=lambda i: drive[i], reverse=True)
    return {
        "dimensions": length,
        "bounds": bounds,
        "probe_steps": steps,
        "probe_seed": seed,
        "controllability_rank": [(i, controllability[i]) for i in ctrl],
        "drive_rank": [(i, drive[i]) for i in drv],
    }


def wire(calibration: dict, *, depth: int = 2, width: int = 1, ladder=None) -> tuple:
    """Build portable fine-to-coarse encoders from a calibration report.

    ``width`` selects that many action-sensitive coordinates and up to that
    many additional varying coordinates. ``depth=0`` keeps only the fine
    context. Positive depth adds ``depth-1`` coarsenings of its primary
    coordinate and one global context. ``ladder`` supplies intermediate bin
    counts explicitly instead. No observation range is inferred by this step.
    """
    width = integer(width, "width", 1)
    depth = integer(depth, "depth")
    if depth > 32:
        raise ValueError("depth must not exceed 32")
    if ladder is None:
        ladder = (
            None if depth == 0 else tuple(max(1, 16 // (2**i)) for i in range(1, depth))
        )
    else:
        ladder = tuple(integer(b, "ladder bins", 1) for b in ladder)
        if len(ladder) > 31:
            raise ValueError("ladder must contain at most 31 levels")
    try:
        dimension = integer(calibration["dimensions"], "dimensions", 1)
        ctrl_rank = [
            integer(row[0], "coordinate") for row in calibration["controllability_rank"]
        ]
        drive_rank = [
            integer(row[0], "coordinate") for row in calibration["drive_rank"]
        ]
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError("Malformed calibration report") from error
    if (
        any(i >= dimension for i in ctrl_rank + drive_rank)
        or len(set(ctrl_rank)) != len(ctrl_rank)
        or len(set(drive_rank)) != len(drive_rank)
    ):
        raise ValueError("Invalid or duplicated calibration coordinates")
    if len(ctrl_rank) < width:
        raise ValueError("Calibration offers fewer coordinates than width")
    ctrl = ctrl_rank[:width]
    drv = [i for i in drive_rank if i not in ctrl][:width]
    bounds = calibration["bounds"]
    if len(bounds) != dimension:
        raise ValueError("Calibration ranges do not match its dimension")
    indices = ctrl + drv
    counts = [16] + [8] * (len(ctrl) - 1) + ([8] + [4] * (len(drv) - 1) if drv else [])
    fine = BinnedFeatures([bounds[i] for i in indices], counts, indices)
    if ladder is None:
        return (fine,)
    levels = [fine]
    levels.extend(BinnedFeatures([bounds[ctrl[0]]], bins, [ctrl[0]]) for bins in ladder)
    levels.append(ConstantFeatures())
    return tuple(levels)
