"""Records: what followed each reading, kept where the reading lands.

A reading is the drive of the neurons a patch reads: sensory fields with their missing
flags, the goal, the action, the reads of declared stores. Each unit's running mean is
subtracted, so the code follows what deviates from the usual reading and a small cue can
move it. A fixed random projection maps the reading onto many cells; the most active cells
stay and the rest are inhibited. Each predicted field keeps a table with one record per
cell. The read of a reading is the sum of its active cells' records, weighted by their
activity; the witnessed outcome is written into exactly those records by the delta rule.
A reading touches few records, so one outcome is written almost exactly and a reading
elsewhere leaves it intact: records learn from one stream without replay.

Two codes come from one reading. The plain code serves the consequence fields. The valued
code first divides every declared input pathway by its running norm, so the goal and a
remembered cue have the same say as the senses; the valued fields (reward, terminal value)
read and write through it at their own rate.

The running mean settles: it is the plain average of the readings seen until one over their
count falls to the habituation rate, and follows at that rate after. A settled mean keeps the
code of a reading where its records were written; a mean that keeps moving would carry old
readings onto cells that were never written.

Task sets keep the values of one task away from the values of another. When ``tasks`` names
the reading's task units (a goal port with one active unit), the cells are divided into one
group per task unit, and a reading's valued code draws its winners from the group of the
active task; the plain code and the consequence records stay shared by every task. A reward
earned under one goal is then written into cells no other goal reads, and a cue learned for
one task survives the rewards of the tasks that follow.

The projection is drawn from ``Mulberry32``, the generator of ``brain_scan.js``, so a page
rebuilds the same cells from the seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["Mulberry32", "Records"]

M32 = 0xFFFFFFFF


class Mulberry32:
    """The 32-bit generator of ``brain_scan.js`` (``mulberry``): uniform draws in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & M32

    def random(self) -> float:
        a = (self.state + 0x6D2B79F5) & M32
        self.state = a
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    def batch(self, n: int) -> np.ndarray:
        """``n`` draws at once, equal to ``n`` calls of ``random``."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        m32 = np.uint64(M32)
        step = np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x6D2B79F5)
        a = (np.uint64(self.state) + step) & m32
        t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & m32
        t = (t ^ ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & m32)) & m32)) & m32
        out: np.ndarray = ((t ^ (t >> np.uint64(14))) & m32).astype(np.float64) / 4294967296.0
        self.state = int((self.state + n * 0x6D2B79F5) & M32)
        return out

    def normals(self, n: int) -> np.ndarray:
        """``n`` standard normal draws by the Box-Muller transform of ``batch`` pairs."""
        pairs = (n + 1) // 2
        u = self.batch(2 * pairs)
        radius = np.sqrt(-2.0 * np.log(np.maximum(u[:pairs], 1e-12)))
        angle = 2.0 * np.pi * u[pairs:]
        out: np.ndarray = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return out[:n]


def _shuffle(n: int, generator: Mulberry32) -> np.ndarray:
    """A permutation of ``range(n)`` by Fisher-Yates from ``n - 1`` draws of the generator."""
    order = np.arange(n)
    draws = generator.batch(max(n - 1, 0))
    for i in range(n - 1, 0, -1):
        j = int(draws[n - 1 - i] * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _rate(name: str, value: float, *, upper: float) -> float:
    rate = float(value)
    if not np.isfinite(rate) or not 0.0 <= rate <= upper:
        raise ValueError(f"{name} lies in [0, {upper:g}]")
    return rate


class Records:
    """A records cortex over readings of ``inputs`` units.

    ``fields`` maps each predicted field to its width. Of ``cells`` code cells, ``active``
    stay per reading. ``rate`` is the write rate of the consequence fields; ``valued`` names
    the fields that read and write through the valued code, at ``valued_rate``.
    ``habituation`` is the slowest rate of each unit's running mean (0 subtracts nothing).
    ``pathways`` are index arrays into the reading whose running norms, at ``pathway_rate``,
    equalise their say in the valued code. ``tasks`` indexes the reading's task units; the
    cells are then divided into one group per task unit, and the valued code of a reading
    draws its winners from the group of the unit with the largest value (every cell when no
    task unit is positive). ``fan_in`` restricts each cell to that many of the pathways,
    drawn from the generator; the cell reads those pathways and every input outside the
    pathways (0 reads every input). ``bias`` scales the cells' fixed offsets. ``seed``
    starts the generator of the projection, the offsets, the division into groups and the
    pathways each cell reads. ``averaging`` makes each cell's write rate the larger of
    ``rate`` and one over the code mass written into it so far, so a fresh cell takes its
    first outcome whole and a well-used cell averages; ``homeostasis`` is the rate at which
    each cell's activation share is tracked and its offset moved toward the share
    ``active / cells``, so that no cell becomes a hub written by unrelated readings (0
    leaves the offsets fixed).
    """

    def __init__(
        self,
        inputs: int,
        fields: Mapping[str, int],
        *,
        cells: int = 8000,
        active: int = 40,
        rate: float = 0.2,
        valued: Iterable[str] = (),
        valued_rate: float = 1.0,
        habituation: float = 1e-5,
        bias: float = 0.3,
        pathways: Sequence[Sequence[int] | np.ndarray] = (),
        pathway_rate: float = 0.002,
        tasks: Sequence[int] | np.ndarray = (),
        fan_in: int = 0,
        seed: int = 0,
        averaging: bool = False,
        homeostasis: float = 0.0,
    ) -> None:
        self.inputs = _positive_int("inputs", inputs)
        self.cells = _positive_int("cells", cells)
        self.active = _positive_int("active", active)
        if self.active > self.cells:
            raise ValueError("active must not exceed cells")
        if not fields:
            raise ValueError("records need at least one field")
        self.fields = {str(k): _positive_int(f"field {k!r}", w) for k, w in fields.items()}
        self.valued = frozenset(str(name) for name in valued)
        if not self.valued <= set(self.fields):
            missing = sorted(self.valued - set(self.fields))
            raise ValueError(f"valued fields {missing} are not fields")
        self.rate = _rate("rate", rate, upper=2.0)
        self.valued_rate = _rate("valued_rate", valued_rate, upper=2.0)
        self.habituation = _rate("habituation", habituation, upper=1.0)
        self.pathway_rate = _rate("pathway_rate", pathway_rate, upper=1.0)
        self.bias = float(bias)
        if not np.isfinite(self.bias) or self.bias < 0:
            raise ValueError("bias must be finite and nonnegative")
        self.averaging = bool(averaging)
        self.homeostasis = _rate("homeostasis", homeostasis, upper=1.0)
        self.seed = int(seed) & M32
        generator = Mulberry32(self.seed)
        draws = generator.normals(self.inputs * self.cells)
        self.projection = draws.reshape(self.inputs, self.cells) / np.sqrt(self.inputs)
        self.offset = generator.normals(self.cells) * self.bias
        self.pathways = [np.asarray(p, dtype=np.int64) for p in pathways if len(p)]
        for p in self.pathways:
            if p.ndim != 1 or p.min() < 0 or p.max() >= self.inputs:
                raise ValueError("a pathway is a one-dimensional index array into the reading")
        self.tasks = np.asarray(tasks, dtype=np.int64).reshape(-1)
        if len(self.tasks) and (self.tasks.min() < 0 or self.tasks.max() >= self.inputs):
            raise ValueError("tasks is an index array into the reading")
        if len(self.tasks) and self.cells // len(self.tasks) < self.active:
            raise ValueError("every task group needs at least active cells")
        self.task_of_cell = np.zeros(self.cells, dtype=np.int64)
        if len(self.tasks):
            order = _shuffle(self.cells, generator)
            for k, group in enumerate(np.array_split(order, len(self.tasks))):
                self.task_of_cell[group] = k
        self.fan_in = int(fan_in)
        if self.fan_in < 0 or (self.fan_in and not self.pathways):
            raise ValueError("fan_in is nonnegative and needs pathways")
        if self.fan_in > len(self.pathways):
            raise ValueError("fan_in must not exceed the number of pathways")
        if self.fan_in:
            reads = np.zeros((len(self.pathways), self.cells), dtype=bool)
            for cell in range(self.cells):
                reads[_shuffle(len(self.pathways), generator)[: self.fan_in], cell] = True
            mask = np.ones((self.inputs, self.cells))
            for k, pathway in enumerate(self.pathways):
                mask[np.ix_(pathway, ~reads[k])] = 0.0
            self.projection *= mask * np.sqrt(self.inputs / mask.sum(axis=0))
        self.mean = np.zeros(self.inputs)
        self.seen = 0
        self.pathway_norm = np.ones(len(self.pathways))
        self.tables = {name: np.zeros((self.cells, width)) for name, width in self.fields.items()}
        self.writes = 0
        self.count = np.zeros(self.cells)  # code mass written into each cell
        # each cell's tracked activation share
        self.usage = np.full(self.cells, self.active / self.cells)
        self.boost = np.zeros(self.cells)  # the homeostatic offset, bounded by the drive scale
        self.drive_scale = 1.0  # running mean absolute drive of witnessed readings

    def _allowed(self, readings: np.ndarray) -> np.ndarray | None:
        """The cells each reading's active task allows for the valued code, or None."""
        if not len(self.tasks):
            return None
        units = readings[:, self.tasks]
        allowed: np.ndarray = self.task_of_cell[None, :] == units.argmax(axis=1)[:, None]
        allowed[units.max(axis=1) <= 0.0] = True
        return allowed

    def _winners(self, x: np.ndarray, allowed: np.ndarray | None, out: np.ndarray) -> None:
        """Writes the k-winner code of the drives ``x @ projection + offset`` into ``out``."""
        drive = x @ self.projection + self.offset + self.boost
        if allowed is not None:
            drive = np.where(allowed, drive, -np.inf)
        k = self.active
        index = np.argpartition(drive, self.cells - k, axis=1)[:, self.cells - k :]
        rows = np.arange(drive.shape[0])[:, None]
        values = np.maximum(drive[rows, index], 0.0)
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        out[...] = 0.0
        out[rows, index] = np.where(norm > 0, values / np.maximum(norm, 1e-12), values)

    def code(self, readings: np.ndarray, *, adapt: bool = False, valued: bool = True) -> np.ndarray:
        """The codes of ``(batch, inputs)`` readings: ``(2, batch, cells)``, plain then valued.

        ``adapt`` first moves the running mean and the pathway norms by these readings:
        witnessed readings adapt, imagined readings do not. ``valued=False`` leaves the
        valued code unset (NaN), for imagined readings whose consequence fields alone are
        read; a read of a valued field from such a code is NaN."""
        x = np.atleast_2d(np.asarray(readings, dtype=float))
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"readings must be a finite (batch, {self.inputs}) array")
        allowed = self._allowed(x)
        if self.habituation > 0:
            if adapt:
                for row in x:
                    self.seen += 1
                    self.mean += max(self.habituation, 1.0 / self.seen) * (row - self.mean)
            x = x - self.mean
        codes = np.empty((2, x.shape[0], self.cells))
        self._winners(x, None, codes[0])
        if self.pathways and adapt:
            for k, pathway in enumerate(self.pathways):
                for value in np.linalg.norm(x[:, pathway], axis=1):
                    self.pathway_norm[k] += self.pathway_rate * (value - self.pathway_norm[k])
        if not valued:
            codes[1] = np.nan
        elif self.pathways:
            v = x.copy()
            for k, pathway in enumerate(self.pathways):
                v[:, pathway] /= self.pathway_norm[k] + 1e-3
            self._winners(v, allowed, codes[1])
        elif allowed is not None:
            self._winners(x, allowed, codes[1])
        else:
            codes[1] = codes[0]
        return codes

    def _code_for(self, code: np.ndarray, name: str) -> np.ndarray:
        if code.shape[0] != 2 or code.shape[-1] != self.cells:
            raise ValueError(f"a code has shape (2, {self.cells}) or (2, batch, {self.cells})")
        selected: np.ndarray = code[1] if name in self.valued else code[0]
        return selected

    def read(self, code: np.ndarray) -> dict[str, np.ndarray]:
        """Each field's read: ``(width,)`` for one code, ``(batch, width)`` for a batch."""
        code = np.asarray(code, dtype=float)
        return {name: self._code_for(code, name) @ table for name, table in self.tables.items()}

    def _write_inputs(
        self,
        code: np.ndarray,
        targets: Mapping[str, np.ndarray],
        known: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
        """Validate every supplied field before any table or averaging count changes.

        A consequence-only code may deliberately leave its unused valued part NaN;
        only codes selected by the supplied target fields must be finite.
        """
        inputs = {}
        for name, width in self.fields.items():
            if name not in targets:
                continue
            target = np.asarray(targets[name], dtype=float)
            expected = (width,) if code.ndim == 2 else (code.shape[1], width)
            if target.shape != expected or not np.isfinite(target).all():
                raise ValueError(f"the target of {name!r} must be finite with shape {expected}")
            selected = self._code_for(code, name)
            if not np.isfinite(selected).all():
                raise ValueError(f"the code used to write {name!r} must be finite")
            if self.averaging and name not in self.valued and (selected < 0).any():
                raise ValueError("averaging writes require nonnegative code mass")
            mask = None
            if known is not None and name in known:
                mask = np.asarray(known[name])
                if mask.shape != expected or mask.dtype != np.bool_:
                    raise ValueError(
                        f"known mask for {name!r} must be boolean with shape {expected}"
                    )
            inputs[name] = (selected, target, mask)
        return inputs

    def write(
        self,
        code: np.ndarray,
        targets: Mapping[str, np.ndarray],
        known: Mapping[str, np.ndarray] | None = None,
    ) -> int:
        """Write the witnessed outcome of one reading.

        Each named field's records move toward its target by the delta rule, through the
        active cells only. ``known`` masks the entries of a target that were observed.
        Invalid inputs or nonfinite updates leave all tables and counts unchanged.
        Returns the number of fields written."""
        code = np.asarray(code, dtype=float)
        if code.shape != (2, self.cells):
            raise ValueError(f"write takes one code of shape (2, {self.cells})")
        inputs = self._write_inputs(code, targets, known)
        count = self.count.copy() if self.averaging else self.count
        updates = {}
        for name, (c, target, mask) in inputs.items():
            active = np.flatnonzero(c)  # the write touches the records of the active cells only
            error = target - c[active] @ self.tables[name][active]
            if mask is not None:
                error = np.where(mask, error, 0.0)
            rate = self.valued_rate if name in self.valued else self.rate
            if self.averaging and name not in self.valued:
                count[active] += c[active]
                steps = np.clip(1.0 / count[active], rate, 1.0)
            else:
                steps = np.full(len(active), rate)
            with np.errstate(over="ignore", invalid="ignore"):
                updated = self.tables[name][active] + steps[:, None] * np.outer(c[active], error)
            if not np.isfinite(updated).all() or not np.isfinite(count[active]).all():
                raise ValueError("record write would produce nonfinite state")
            updates[name] = (active, updated)
        for name, (active, updated) in updates.items():
            self.tables[name][active] = updated
        self.count = count
        written = len(updates)
        self.writes += written
        return written

    def write_batch(self, codes: np.ndarray, targets: Mapping[str, np.ndarray]) -> int:
        """Write the witnessed outcomes of many readings at once.

        Every reading's error is taken against the tables as they stand when the call
        begins, and each cell moves by the mean of the moves its writers would have made
        alone. One writer alone reproduces ``write``; writers that agree move a shared
        cell as far as one of them would, so a batch does not overshoot. Writers that
        disagree leave the cell at their average, which later presentations refine.
        ``codes`` has shape ``(2, batch, cells)``; each target ``(batch, width)``.
        Invalid inputs or nonfinite updates leave all tables and counts unchanged.
        Returns the number of (reading, field) writes."""
        codes = np.asarray(codes, dtype=float)
        if codes.ndim != 3 or codes.shape[0] != 2 or codes.shape[2] != self.cells:
            raise ValueError(f"write_batch takes codes of shape (2, batch, {self.cells})")
        batch = int(codes.shape[1])
        inputs = self._write_inputs(codes, targets)
        count = self.count.copy() if self.averaging else self.count
        updates = {}
        for name, (c, target, _) in inputs.items():
            width = self.fields[name]
            rows, cells = np.nonzero(c)
            values = c[rows, cells]
            table = self.tables[name]
            read = np.zeros((batch, width))
            np.add.at(read, rows, values[:, None] * table[cells])
            error = target - read
            touched, local = np.unique(cells, return_inverse=True)
            moves = np.zeros((len(touched), width))
            with np.errstate(over="ignore", invalid="ignore"):
                np.add.at(moves, local, values[:, None] * error[rows])
            writers = np.bincount(local, minlength=len(touched)).astype(float)
            rate = self.valued_rate if name in self.valued else self.rate
            if self.averaging and name not in self.valued:
                np.add.at(count, cells, values)
                steps = np.clip(1.0 / count[touched], rate, 1.0)
            else:
                steps = np.full(len(touched), rate)
            with np.errstate(over="ignore", invalid="ignore"):
                updated = table[touched] + (steps / writers)[:, None] * moves
            if not np.isfinite(updated).all() or not np.isfinite(count[touched]).all():
                raise ValueError("record write would produce nonfinite state")
            updates[name] = (touched, updated)
        for name, (touched, updated) in updates.items():
            self.tables[name][touched] = updated
        self.count = count
        written = batch * len(updates)
        self.writes += written
        return written

    def witness(self, readings: np.ndarray) -> None:
        """Move the running mean and pathway norms by witnessed readings, without coding them.

        Exactly the side effect of ``code(readings, adapt=True)``, for a caller that codes
        the readings after the outcome is known and writes into those codes."""
        x = np.atleast_2d(np.asarray(readings, dtype=float))
        if x.ndim != 2 or x.shape[1] != self.inputs or not np.isfinite(x).all():
            raise ValueError(f"readings must be a finite (batch, {self.inputs}) array")
        if self.habituation > 0:
            for row in x:
                self.seen += 1
                self.mean += max(self.habituation, 1.0 / self.seen) * (row - self.mean)
            x = x - self.mean
        if self.pathways:
            for k, pathway in enumerate(self.pathways):
                for value in np.linalg.norm(x[:, pathway], axis=1):
                    self.pathway_norm[k] += self.pathway_rate * (value - self.pathway_norm[k])
        if self.homeostasis > 0:
            # Each witnessed reading's plain code moves the cells' tracked shares; the offsets
            # follow, so cells that win too often withdraw and idle cells come forward.
            target = self.active / self.cells
            for row in x:
                drive = row @ self.projection + self.offset
                scale = float(np.mean(np.abs(drive)))
                self.drive_scale += self.homeostasis * (scale - self.drive_scale)
                cut = self.cells - self.active
                winners = np.argpartition(drive + self.boost, cut)[cut:]
                fired = np.zeros(self.cells)
                fired[winners] = 1.0
                self.usage += self.homeostasis * (fired - self.usage)
                # A bounded error: +1 for an idle cell, toward -1 for a hub; the offsets move by a
                # fraction of the drive scale measured without them, and stay within it.
                error = (target - self.usage) / (target + self.usage)
                self.boost += self.homeostasis * self.drive_scale * error
                np.clip(self.boost, -3.0 * self.drive_scale, 3.0 * self.drive_scale, out=self.boost)

    def parameters(self) -> int:
        """Record entries, the learned state; the projection and the offsets are fixed."""
        return int(sum(table.size for table in self.tables.values()))

    def state(self) -> dict[str, np.ndarray]:
        """The learned state as arrays: every table, the running mean, the readings seen,
        the pathway norms and the write count. ``to_dict`` holds the fixed configuration."""
        result = {
            "mean": self.mean.copy(),
            "seen": np.array(self.seen, dtype=np.int64),
            "pathway_norm": self.pathway_norm.copy(),
            "writes": np.array(self.writes, dtype=np.int64),
            "count": self.count.copy(),
            "usage": self.usage.copy(),
            "boost": self.boost.copy(),
            "drive_scale": np.array(self.drive_scale, dtype=float),
        }
        for name, table in self.tables.items():
            result["table_" + name] = table.copy()
        return result

    def load_state(self, state: Mapping[str, np.ndarray]) -> None:
        """Replace the learned state by arrays of the shapes ``state`` produces, atomically."""
        learned = {"mean", "seen", "pathway_norm", "writes"}
        learned |= {"count", "usage", "boost", "drive_scale"}
        expected = learned | {"table_" + name for name in self.fields}
        if set(state) != expected:
            raise ValueError("record state must contain exactly the arrays of state()")
        mean = np.asarray(state["mean"], dtype=float)
        norm = np.asarray(state["pathway_norm"], dtype=float)
        seen, writes = np.asarray(state["seen"]), np.asarray(state["writes"])
        if mean.shape != (self.inputs,) or norm.shape != (len(self.pathways),):
            raise ValueError("record state arrays have the wrong shapes")
        if not (np.isfinite(mean).all() and np.isfinite(norm).all() and (norm > 0).all()):
            raise ValueError("record state must be finite with positive pathway norms")
        if seen.shape != () or writes.shape != () or seen < 0 or writes < 0:
            raise ValueError("record counts must be nonnegative scalars")
        tables = {}
        for name, width in self.fields.items():
            table = np.asarray(state["table_" + name], dtype=float)
            if table.shape != (self.cells, width) or not np.isfinite(table).all():
                raise ValueError(f"the table of {name!r} must be a finite ({self.cells}, {width})")
            tables[name] = table.copy()
        count = np.asarray(state["count"], dtype=float)
        usage = np.asarray(state["usage"], dtype=float)
        boost = np.asarray(state["boost"], dtype=float)
        scale = float(np.asarray(state["drive_scale"]))
        for name, array in (("count", count), ("usage", usage), ("boost", boost)):
            if array.shape != (self.cells,) or not np.isfinite(array).all():
                raise ValueError(f"record {name} must be a finite ({self.cells},) array")
        if (count < 0).any() or (usage < 0).any() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("record counts and usage are nonnegative, the drive scale positive")
        self.mean, self.pathway_norm, self.tables = mean.copy(), norm.copy(), tables
        self.seen, self.writes = int(seen), int(writes)
        self.count, self.usage, self.boost = count.copy(), usage.copy(), boost.copy()
        self.drive_scale = scale

    def to_dict(self) -> dict[str, Any]:
        """The configuration that rebuilds the fixed cells (the tables are the learned state)."""
        return {
            "inputs": self.inputs,
            "fields": dict(self.fields),
            "cells": self.cells,
            "active": self.active,
            "rate": self.rate,
            "valued": sorted(self.valued),
            "valued_rate": self.valued_rate,
            "habituation": self.habituation,
            "bias": self.bias,
            "pathways": [p.tolist() for p in self.pathways],
            "pathway_rate": self.pathway_rate,
            "averaging": self.averaging,
            "homeostasis": self.homeostasis,
            "tasks": self.tasks.tolist(),
            "fan_in": self.fan_in,
            "seed": self.seed,
        }
