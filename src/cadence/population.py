"""Record patches in lockstep on a device: a population of instances, each with many
streams, every settle one batched product.

``PopulationPatch`` is the torch twin of ``RecordPatchNet`` for one-moment paths from rest
(a reading settled once, the way a decision or a school batch reads a patch). It holds
``instances`` independent parameter sets and, for each instance, ``streams`` independent
record stores, so a whole population of brains, each in many worlds, settles, reads,
writes and takes its slow step in a handful of tensor operations per moment. Nothing
crosses an instance, a stream or a moment: the slow step of each instance is the adjoint of
its own one-moment loss (autograd computes it; the NumPy patch's adjoint is the reference
and a parity test holds both to rounding; the adjoint is written out, the numbers autograd
gives at a fraction of its cost on a device), the records are the delta rule at the code,
and no gradient reaches the store or another patch. ``from_patch`` copies a NumPy patch's
parameters, projection, offsets and running statistics into every instance, which is how
the parity is checked and how a schooled patch is broadcast into a population.

What batches: instances, streams and, through several ``PopulationPatch`` objects, the
patches of a brain. What does not: the moments of one stream, which a temporal patch
settles in order. Memory: the tables are ``instances * streams * cells * outputs`` values;
a settle touches only the ``active`` rows of every stream's table (the read gathers them,
the write scatters into them), so the cost of a moment is set by the active cells and the
outputs, not by the store. ``observe`` takes a mask over the streams for a population whose
streams do not all have a moment to learn from at once (games of unequal length).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .record_patch import RecordPatchNet


def device_of(name: str | None = None) -> torch.device:
    """The named device, or the best one present: cuda, then mps, then cpu."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class _Staged:
    """A per-stream tensor's staged change: the whole new tensor when every stream is a
    moment, else the touched rows of its (instances * streams, ...) view, with each
    moment's row."""

    values: torch.Tensor
    touched: torch.Tensor | None = None
    inverse: torch.Tensor | None = None

    def per_moment(self) -> torch.Tensor:
        if self.touched is None or self.inverse is None:
            return self.values
        P, n = self.inverse.shape
        rows = self.values.index_select(0, self.inverse.reshape(-1))
        return rows.view(P, n, *self.values.shape[1:])


@dataclass
class _StagedWrite:
    """Everything one observe would write, held until the slow step is checked too, with
    the checks it must pass (0-dim booleans, read in one device sync) and whether the rows
    it lands on have room for it."""

    norm: _Staged
    seen: _Staged | None
    mean: _Staged | None
    rows: torch.Tensor
    update: torch.Tensor
    written: torch.Tensor
    code: torch.Tensor
    checks: dict[str, torch.Tensor]
    room: torch.Tensor
    exact: tuple[torch.Tensor, torch.Tensor] | None = None


class PopulationPatch:
    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        instances: int,
        streams: int,
        seed: int = 0,
        cells: int = 4096,
        active: int = 32,
        record_rate: float = 0.5,
        habituation: float = 1e-5,
        record_bias: float = 0.3,
        slowest: float = 2.0,
        groups: tuple[int, ...] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if groups is not None and (len(groups) != 1 or groups[0] != outputs):
            raise ValueError("groups: one softmax over every output (groups=(outputs,)) or None")
        self.inputs, self.hidden, self.outputs = int(inputs), int(hidden), int(outputs)
        self.P, self.B = int(instances), int(streams)
        self.cells, self.active = int(cells), int(active)
        self.record_rate = float(record_rate)
        self.habituation = float(habituation)
        self.groups: tuple[int, ...] | None = groups
        self.dev = device_of(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        # the same construction as the NumPy patch: a fresh one, copied into every instance
        seed_net = RecordPatchNet(
            inputs,
            hidden,
            outputs,
            seed=seed,
            cells=cells,
            active=active,
            record_rate=record_rate,
            habituation=habituation,
            record_bias=record_bias,
            slowest=slowest,
            groups=groups,
        )
        self._load(seed_net)
        self.settles = 0
        self.writes = 0
        # (instances, outputs): how much of each output's residual the records take
        self.record_weight: torch.Tensor | None = None
        # or "adam": the slow step scaled per parameter by the adjoint's running moments
        self.optimizer = "sgd"
        self._adam: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._adam_t = torch.zeros(self.P, dtype=self.dtype, device=self.dev)

    # --- construction ------------------------------------------------------------------------

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.dev)

    def _load(self, net: RecordPatchNet) -> None:
        P, B = self.P, self.B
        params = net.parameters()
        self.G = self._t(params["G"]).expand(P, -1, -1).clone()
        self.g = self._t(params["g"]).expand(P, -1).clone()
        self.Bm = self._t(params["B"]).expand(P, -1, -1).clone()
        self.b = self._t(params["b"]).expand(P, -1).clone()
        self.C = self._t(params["C"]).expand(P, -1, -1).clone()
        self.c = self._t(params["c"]).expand(P, -1).clone()
        self.scale = self._t(net._scale)  # the unit of a channel's fluctuation
        self.projection = self._t(net.records.projection)  # (inputs + hidden, cells)
        self.offset = self._t(net.records.offset)
        width = self.inputs + self.hidden
        self.mean = self._t(net.records.mean).expand(P, B, -1).clone()
        seen = float(net.records.seen)
        self.seen = torch.full((P, B), seen, dtype=self.dtype, device=self.dev)
        norm = float(net._input_norm)
        self.input_norm = torch.full((P, B), norm, dtype=self.dtype, device=self.dev)
        self.tables = self._t(net.records.tables["y"]).expand(P, B, -1, -1).clone()
        # writes per cell and stream
        self.written = torch.zeros(P, B, self.cells, dtype=self.dtype, device=self.dev)
        assert self.mean.shape[-1] == width

    @classmethod
    def from_patch(
        cls,
        net: RecordPatchNet,
        *,
        instances: int,
        streams: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationPatch:
        """Every instance a copy of ``net``: its parameters, its projection and offsets, its
        running statistics and its tables (one copy per stream)."""
        plain = net.record_writes == "sequential" and net._output_code is None
        if not plain or net.records.averaging:
            raise ValueError("from_patch takes a sequential patch, no output code, no averaging")
        twin = cls.__new__(cls)
        twin.inputs, twin.hidden, twin.outputs = net.inputs, net.hidden, net.outputs
        twin.P, twin.B = int(instances), int(streams)
        twin.cells, twin.active = net.records.cells, net.records.active
        twin.record_rate = float(net.records.rate)
        twin.habituation = float(net.records.habituation)
        twin.groups = tuple(net.groups) if net.groups is not None else None
        twin.dev = device_of(device) if not isinstance(device, torch.device) else device
        twin.dtype = dtype
        twin._load(net)
        twin.settles = 0
        twin.writes = 0
        twin.record_weight = None
        twin.optimizer = "sgd"
        twin._adam = {}
        twin._adam_t = torch.zeros(twin.P, dtype=twin.dtype, device=twin.dev)
        return twin

    def parameters(self) -> dict[str, torch.Tensor]:
        return {"G": self.G, "g": self.g, "B": self.Bm, "b": self.b, "C": self.C, "c": self.c}

    # --- one moment from rest -----------------------------------------------------------------

    def _forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The context (P, B, hidden) from rest with its two factors: ``h = (1 - l) z``,
        ``z = tanh(B u + b)``, ``l = sigmoid(g + G u)``."""
        z = torch.tanh(torch.matmul(u, self.Bm.transpose(1, 2)) + self.b[:, None, :])
        gate = torch.sigmoid(torch.matmul(u, self.G.transpose(1, 2)) + self.g[:, None, :])
        return (1.0 - gate) * z, z, gate

    def _context(self, u: torch.Tensor) -> torch.Tensor:
        """(P, B, hidden) from rest: ``h = (1 - l) tanh(B u + b)`` with ``l = sigmoid(g + G u)``."""
        return self._forward(u)[0]

    def _slow(self, h: torch.Tensor) -> torch.Tensor:
        drive = torch.matmul(h, self.C.transpose(1, 2)) + self.c[:, None, :]
        if self.groups is None:
            return drive
        return torch.softmax(drive, dim=-1)

    def _readings(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        stream_of: torch.Tensor | None = None,
        norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """What the records read: the input and the scaled context, unit variance per unit;
        ``norm`` (P, N) overrides the streams' input norm with a staged one."""
        if norm is None:
            norm = self._per_stream(self.input_norm, stream_of)
        scaled = u * (np.sqrt(self.inputs) / norm)[..., None]
        return torch.cat([scaled, h * self.scale], dim=-1)

    def code(self, readings: torch.Tensor) -> torch.Tensor:
        """The k-winner code (P, B, cells) of readings (P, B, inputs + hidden)."""
        indices, values = self._sparse_code(readings)
        return self._dense(indices, values)

    def _sparse_code(
        self,
        readings: torch.Tensor,
        stream_of: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The k-winner code as its active cells: indices and values, each (P, N, active),
        the values as ``code`` scatters them; ``mean`` (P, N, width) overrides the streams'
        running mean with a staged one."""
        if self.habituation > 0:
            x = readings - (self._per_stream(self.mean, stream_of) if mean is None else mean)
        else:
            x = readings
        drive = torch.matmul(x, self.projection) + self.offset
        top = torch.topk(drive, self.active, dim=-1)
        values = torch.clamp(top.values, min=0.0)
        norm = values.norm(dim=-1, keepdim=True)
        values = torch.where(norm > 0, values / torch.clamp(norm, min=1e-12), values)
        return top.indices, values

    def _dense(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            self.P, indices.shape[1], self.cells, dtype=values.dtype, device=values.device
        )
        out.scatter_(-1, indices, values)
        return out

    def read(self, code: torch.Tensor) -> torch.Tensor:
        return torch.einsum("pbc,pbco->pbo", code, self.tables)

    def _rows(self, indices: torch.Tensor, stream_of: torch.Tensor | None = None) -> torch.Tensor:
        """The active cells as rows of the table flattened to (instances * streams * cells,
        outputs). ``indices`` is (P, N, active); moment ``n`` reads stream ``n`` unless
        ``stream_of`` (N,) names its stream."""
        N = indices.shape[1]
        streams = torch.arange(N, device=self.dev)[None, :] if stream_of is None else stream_of
        if streams.dim() == 1:
            streams = streams[None, :]
        instances = torch.arange(self.P, device=self.dev)[:, None]
        base = (instances * self.B + streams)[..., None] * self.cells
        return (base + indices).reshape(-1)

    def _read_sparse(
        self, indices: torch.Tensor, values: torch.Tensor, stream_of: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The read through the active cells alone: their rows, weighted by the code."""
        return self._read_rows(indices, values, stream_of)[0]

    def _read_rows(
        self, indices: torch.Tensor, values: torch.Tensor, stream_of: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The sparse read and the largest magnitude among the rows it gathered."""
        N = indices.shape[1]
        rows = self.tables.view(-1, self.outputs).index_select(0, self._rows(indices, stream_of))
        read = (rows.view(self.P, N, self.active, self.outputs) * values[..., None]).sum(dim=2)
        return read, self._largest(rows)

    def familiarity(
        self, indices: torch.Tensor, values: torch.Tensor, stream_of: torch.Tensor | None = None
    ) -> torch.Tensor:
        """How much of a code's mass falls on cells the stream's store has written before,
        (P, N) in [0, 1]: whether this brain has been near this reading in this world."""
        N = indices.shape[1]
        rows = self._rows(indices, stream_of)
        seen = self.written.view(-1).index_select(0, rows).view(self.P, N, self.active)
        mass = ((seen > 0).to(values.dtype) * values).sum(dim=-1)
        return mass / values.sum(dim=-1).clamp(min=1e-12)

    def _per_stream(self, x: torch.Tensor, stream_of: torch.Tensor | None) -> torch.Tensor:
        """A per-stream tensor (P, B, ...) gathered to the moments (P, N, ...); ``stream_of``
        is (N,), the same streams for every instance, or (P, N)."""
        if stream_of is None:
            return x
        if stream_of.dim() == 1:
            return x[:, stream_of]
        idx = stream_of.reshape(self.P, -1, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
        return torch.gather(x, 1, idx)

    def _add_per_stream(self, x: torch.Tensor, stream_of: torch.Tensor, d: torch.Tensor) -> None:
        """``x[p, stream_of[p, n]] += d[p, n]`` in place, moments of one stream summed."""
        if stream_of.dim() == 1:
            x.index_add_(1, stream_of, d)
            return
        flat = x.view(self.P * self.B, *x.shape[2:])
        rows = (torch.arange(self.P, device=self.dev)[:, None] * self.B + stream_of).reshape(-1)
        flat.index_add_(0, rows, d.reshape(rows.shape[0], *x.shape[2:]))

    def imagine(
        self, inputs: torch.Tensor, *, stream_of: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Predictions for readings (P, N, inputs): the slow readout, the record read, their
        sum, the context and the code; private, from rest, no learning. ``N`` is the
        streams unless ``stream_of`` (N,) assigns each moment to a stream's store, which
        folds several imagined readings per stream into one settle."""
        if stream_of is not None:
            stream_of = torch.as_tensor(stream_of, device=self.dev)
        u = self._as(inputs, stream_of)
        with torch.no_grad():
            h = self._context(u)
            slow = self._slow(h)
            indices, values = self._sparse_code(self._readings(u, h, stream_of), stream_of)
            read = self._read_sparse(indices, values, stream_of)
            familiar = self.familiarity(indices, values, stream_of)
        self.settles += 1
        return {
            "output": slow + read,
            "slow": slow,
            "read": read,
            "hidden": h,
            "code": self._dense(indices, values),
            "familiarity": familiar,
        }

    def _as(
        self, x: torch.Tensor | np.ndarray, stream_of: torch.Tensor | None = None
    ) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=self.dtype, device=self.dev)
        if t.shape[-1] != self.inputs:
            raise ValueError(f"inputs must be (instances, moments, {self.inputs})")
        n = self.B if stream_of is None else int(stream_of.shape[-1])
        return t.reshape(self.P, n, -1)

    # --- learning -------------------------------------------------------------------------------

    def stream_loss(
        self, slow: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The slow loss per stream (P, B): half the mean squared error over the outputs for
        a linear readout (the NumPy patch's convention), the cross-entropy for a
        categorical port. ``weight`` (outputs,) or (P, outputs), summing to the outputs,
        reweights the squared errors (all ones is the plain mean)."""
        if self.groups is None:
            err = (slow - target) ** 2
            if weight is not None:
                err = err * (weight[:, None, :] if weight.dim() == 2 else weight)
            return 0.5 * err.mean(dim=-1)
        return -(target * torch.log(torch.clamp(slow, min=1e-12))).sum(dim=-1)

    def loss(self, slow: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The slow loss per instance, the mean over its streams."""
        return self.stream_loss(slow, target).mean(dim=-1)

    def observe(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        *,
        rate: float | torch.Tensor = 1.0,
        write: bool = True,
        mask: torch.Tensor | None = None,
        stream_of: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One moment observed in every stream: the slow parameters of every instance move
        against the adjoint of their own slow loss by ``rate`` (a float or a tensor of
        ``(instances,)``), and the records of every stream take what the readout got wrong,
        against the parameters that made the prediction. ``mask`` (instances, streams)
        names the streams that have a moment to learn from: the slow loss of an instance is
        the mean over its masked streams, and only those streams write and move their
        statistics; an instance with no masked stream does not move. ``stream_of`` (N,)
        assigns each of ``N`` moments to a stream's store, so several queued moments of one
        stream are taught in one call: the slow step averages them all and the writes of
        one stream land by the delta rule against the store as it stood before the call (a
        batched write, the NumPy patch's ``write_batch``). ``weight`` (outputs,) or
        (instances, outputs), summing to the outputs, reweights the squared errors of a
        linear readout: a patch that predicts a wide reading of which a few ports matter
        counts them by their weight rather than one in eight hundred.

        Every change is staged and checked before any is committed. The inputs must be
        finite, and the targets finite on the unmasked moments (a masked-out moment's
        target is ignored and may be NaN); a write or slow step that would leave a
        nonfinite table, statistic, parameter or moment raises ``ValueError`` and changes
        nothing, so a bad moment cannot poison a store."""
        if stream_of is not None:
            stream_of = torch.as_tensor(stream_of, device=self.dev)
        u = self._as(inputs, stream_of)
        n = u.shape[1]
        t = torch.as_tensor(target, dtype=self.dtype, device=self.dev)
        t = t.reshape(self.P, n, self.outputs)
        m = None
        if mask is not None:
            m = torch.as_tensor(mask, dtype=self.dtype, device=self.dev).reshape(self.P, n)
        w = None
        if weight is not None:
            w = torch.as_tensor(weight, dtype=self.dtype, device=self.dev)
            w = w[:, None, :] if w.dim() == 2 else w
        step = torch.as_tensor(rate, dtype=self.dtype, device=self.dev).reshape(-1)
        if step.numel() == 1:
            step = step.expand(self.P)
        with torch.no_grad():
            gate_b = torch.ones(self.P, n, dtype=self.dtype, device=self.dev) if m is None else m
            lesson = gate_b[..., None] > 0
            checks = {
                "the inputs must be finite": torch.isfinite(u).all(),
                "the targets of the unmasked moments must be finite": (
                    torch.isfinite(t) | ~lesson
                ).all(),
                "the rate must be finite": torch.isfinite(step).all(),
            }
            if w is not None:
                checks["the weight must be finite"] = torch.isfinite(w).all()
            if self.record_weight is not None:
                checks["record_weight must be finite"] = torch.isfinite(self.record_weight).all()
            h, z, gate = self._forward(u)
            slow = self._slow(h)
            t = torch.where(lesson, t, slow)  # a masked-out moment carries no lesson
            per_stream = self.stream_loss(slow, t, w)
            count = gate_b.sum(dim=-1).clamp(min=1.0)
            per_instance = (per_stream * gate_b).sum(dim=-1) / count
            # the adjoint of the one-moment loss, written out: the numbers autograd gives
            share = (gate_b / count[:, None])[..., None]  # the mask's share per moment
            if self.groups is None:
                d = (slow - t) / self.outputs * share
                if w is not None:
                    d = d * w
            else:
                d = (slow * t.sum(dim=-1, keepdim=True) - t) * share
            gC = torch.matmul(d.transpose(1, 2), h)
            gc = d.sum(dim=1)
            dh = torch.matmul(d, self.C)
            dz = dh * (1.0 - gate) * (1.0 - z * z)
            dgate = dh * (-z) * gate * (1.0 - gate)
            gB = torch.matmul(dz.transpose(1, 2), u)
            gb = dz.sum(dim=1)
            gG = torch.matmul(dgate.transpose(1, 2), u)
            gg = dgate.sum(dim=1)
            residual = t - slow
            staged_write = self._stage_write(u, h, residual, gate_b, stream_of) if write else None
            grads = {"G": gG, "g": gg, "Bm": gB, "b": gb, "C": gC, "c": gc}
            new, moments, adam_t, slow_checks = self._stage_slow_step(step, grads)
            if staged_write is not None:
                checks.update(staged_write.checks)
                checks["room"] = staged_write.room
            checks.update(slow_checks)
            failed = self._failed(checks)  # the one device sync of an observe
            if failed and failed != ["room"]:
                raise ValueError("observe: " + next(f for f in failed if f != "room"))
            if failed and staged_write is not None:
                self._sum_on_a_copy(staged_write)  # the rows could overflow: look before landing
            if staged_write is not None:
                self._commit_write(staged_write)
            self._commit_slow_step(new, moments, adam_t)
        self.settles += 1
        return {
            "loss": per_instance,
            "slow": slow,
            "hidden": h,
            "residual": residual,
            "code": staged_write.code if staged_write is not None else torch.zeros(0),
        }

    @staticmethod
    def _failed(checks: dict[str, torch.Tensor]) -> list[str]:
        """The names of the checks (0-dim booleans) that fail, read in one device sync."""
        flags = torch.stack(list(checks.values())).tolist()
        return [name for name, ok in zip(checks, flags, strict=True) if not ok]

    @staticmethod
    def _largest(x: torch.Tensor) -> torch.Tensor:
        """The largest magnitude in ``x`` as a 0-dim tensor, one reduction (``aminmax``;
        ``vector_norm`` is many times slower on MPS); NaN and infinity both make it
        nonfinite."""
        low, high = torch.aminmax(x)
        largest: torch.Tensor = torch.maximum(-low, high)
        return largest

    # --- the slow step, staged ---------------------------------------------------------------

    def _adam_step(
        self,
        name: str,
        grad: torch.Tensor,
        shape: tuple[int, ...],
        t_all: torch.Tensor,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The adjoint scaled by its running moments per instance and parameter (Adam); the
        rate is then the learning rate. Returns the scaled step with the moments it would
        leave, which the caller commits."""
        m0, v0 = self._adam.get(name, (torch.zeros_like(grad), torch.zeros_like(grad)))
        m = b1 * m0 + (1 - b1) * grad
        v = b2 * v0 + (1 - b2) * grad * grad
        t = t_all.clamp(min=1.0).reshape(shape)
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        return m_hat / (v_hat.sqrt() + eps), m, v

    def _stage_slow_step(
        self, step: torch.Tensor, grads: dict[str, torch.Tensor]
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor | None,
        dict[str, torch.Tensor],
    ]:
        """The parameters every instance would have after the step, with the Adam moments
        and count it would leave, and the check that all of them are finite."""
        adam_t = self._adam_t + (step > 0).to(self.dtype) if self.optimizer == "adam" else None
        new: dict[str, torch.Tensor] = {}
        moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        staged = []
        for name, grad in grads.items():
            leaf = getattr(self, name)
            shape = (self.P,) + (1,) * (leaf.dim() - 1)
            if adam_t is not None:
                grad, m, v = self._adam_step(name, grad, shape, adam_t)
                moments[name] = (m, v)
                staged += [m.reshape(-1), v.reshape(-1)]
            new[name] = leaf - step.reshape(shape) * grad
            staged.append(new[name].reshape(-1))
        ok = torch.isfinite(self._largest(torch.cat(staged)))  # one reduction for all of them
        check = {"the slow step would leave a nonfinite parameter or moment": ok}
        return new, moments, adam_t, check

    def _commit_slow_step(
        self,
        new: dict[str, torch.Tensor],
        moments: dict[str, tuple[torch.Tensor, torch.Tensor]],
        adam_t: torch.Tensor | None,
    ) -> None:
        for name, value in new.items():
            setattr(self, name, value)
        self._adam.update(moments)
        if adam_t is not None:
            self._adam_t = adam_t

    # --- the write, staged ------------------------------------------------------------------

    def _touched(self, stream_of: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows of a per-stream (instances * streams, ...) view the moments touch, and
        each moment's row among them."""
        streams = stream_of[None, :].expand(self.P, -1) if stream_of.dim() == 1 else stream_of
        rows = (torch.arange(self.P, device=self.dev)[:, None] * self.B + streams).reshape(-1)
        touched, inverse = torch.unique(rows, return_inverse=True)
        return touched, inverse.view(self.P, n)

    def _stage_add(
        self, x: torch.Tensor, d: torch.Tensor, touched: tuple[torch.Tensor, torch.Tensor] | None
    ) -> _Staged:
        """``x[p, stream] + d[p, n]`` summed over the moments ``n`` of each stream, staged:
        the whole new tensor when the moments are the streams, else the touched rows of
        the (instances * streams, ...) view."""
        if touched is None:
            return _Staged(x + d)
        rows, inverse = touched
        values = x.view(self.P * self.B, *x.shape[2:]).index_select(0, rows)
        values.index_add_(0, inverse.reshape(-1), d.reshape(inverse.numel(), *x.shape[2:]))
        return _Staged(values, rows, inverse)

    def _commit_stage(self, name: str, staged: _Staged) -> None:
        if staged.touched is None:
            setattr(self, name, staged.values)
            return
        x = getattr(self, name)
        flat = x.view(self.P * self.B, *staged.values.shape[1:])
        flat.index_copy_(0, staged.touched, staged.values)

    def _stage_write(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        residual: torch.Tensor,
        gate: torch.Tensor,
        stream_of: torch.Tensor | None = None,
    ) -> _StagedWrite:
        """The witnessed reading moves the running statistics; the code then takes the residual
        by the delta rule through its active cells, a scatter into their rows alone. A
        masked-out moment neither moves its stream's statistics nor writes. With
        ``stream_of`` the moments' statistics moves are summed into their streams. Nothing
        is committed here: the write travels with its checks, that the rows it read and
        the update it would add are finite, and with whether those rows have room for it."""
        n = u.shape[1]
        touched = None if stream_of is None else self._touched(stream_of, n)
        norms = u.norm(dim=-1)
        old_norm = self._per_stream(self.input_norm, stream_of)
        d_norm = 0.01 * gate * (torch.clamp(norms, min=1e-6) - old_norm)
        norm = self._stage_add(self.input_norm, d_norm, touched)
        # with the moved norm, as the NumPy patch
        readings = self._readings(u, h, stream_of, norm=norm.per_moment())
        seen = mean = None
        mean_moved = None
        if self.habituation > 0:
            seen = self._stage_add(self.seen, gate, touched)
            fresh = torch.clamp(1.0 / seen.per_moment().clamp(min=1.0), min=self.habituation)
            step = (fresh * gate)[..., None]
            d_mean = step * (readings - self._per_stream(self.mean, stream_of))
            mean = self._stage_add(self.mean, d_mean, touched)
            mean_moved = mean.per_moment()
        # with the moved mean
        indices, values = self._sparse_code(readings, stream_of, mean=mean_moved)
        held, held_max = self._read_rows(indices, values, stream_of)
        move = (residual - held) * gate[..., None]
        # an output whose target is a sample rather than a fact keeps its records damped
        if self.record_weight is not None:
            move = move * self.record_weight[:, None, :]
        update = self.record_rate * values[..., None] * move[..., None, :]
        rows = self._rows(indices, stream_of)
        # every entry of the update is at most scale * (|residual| + |row|), the code's values
        # being at most one; a row receives at most every contribution of its instance, so
        # the write has room if even that sum cannot overflow, and then it is finite too
        scale: torch.Tensor | float = self.record_rate
        if self.record_weight is not None:
            scale = scale * self._largest(self.record_weight)
        largest = self._largest(residual)
        finite = torch.isfinite(torch.stack([held_max, largest])).all()
        checks = {"the record write would leave a nonfinite table or statistic": finite}
        reach = scale * float(n * self.active) * (largest + held_max)
        room = held_max + reach < torch.finfo(self.dtype).max
        written = (values * gate[..., None] > 0).to(self.dtype).reshape(-1)
        code = self._dense(indices, values)
        return _StagedWrite(
            norm, seen, mean, rows, update.reshape(-1, self.outputs), written, code, checks, room
        )

    def _sum_on_a_copy(self, staged: _StagedWrite) -> None:
        """The rare case: the rows the write lands on could overflow, so it is summed on a
        copy of them and refused if the sum is not finite."""
        touched, inverse = torch.unique(staged.rows, return_inverse=True)
        rows = self.tables.view(-1, self.outputs).index_select(0, touched)
        rows.index_add_(0, inverse, staged.update)
        if not bool(torch.isfinite(self._largest(rows))):
            raise ValueError("observe: the record write would leave a nonfinite table or statistic")
        staged.exact = (touched, rows)

    def _commit_write(self, staged: _StagedWrite) -> None:
        self._commit_stage("input_norm", staged.norm)
        if staged.seen is not None and staged.mean is not None:
            self._commit_stage("seen", staged.seen)
            self._commit_stage("mean", staged.mean)
        flat = self.tables.view(-1, self.outputs)
        if staged.exact is None:
            flat.index_add_(0, staged.rows, staged.update)
        else:
            flat.index_copy_(0, staged.exact[0], staged.exact[1])
        self.written.view(-1).index_add_(0, staged.rows, staged.written)
        self.writes += 1

    def clear_records(self) -> None:
        self.tables.zero_()
        self.written.zero_()

    # --- population operations -------------------------------------------------------------------

    def inherit(
        self, parents: torch.Tensor, *, sigma: float = 0.0, generator: torch.Generator | None = None
    ) -> None:
        """Every instance becomes a copy of instance ``parents[p]``, with a normal step of
        ``sigma`` times each tensor's spread on the slow parameters when ``sigma > 0``; the
        stores and statistics restart."""
        idx = torch.as_tensor(parents, device=self.dev, dtype=torch.long)
        for name in ("G", "g", "Bm", "b", "C", "c"):
            v = getattr(self, name).detach()[idx].clone()
            if sigma > 0:
                noise = torch.randn(v.shape, generator=generator, dtype=v.dtype).to(self.dev)
                v = v + noise * sigma * torch.clamp(v.std(), min=1e-3)
            setattr(self, name, v)
            if name in self._adam:
                m, vv = self._adam[name]
                self._adam[name] = (m[idx].clone(), vv[idx].clone())
        self._adam_t = self._adam_t[idx].clone()
        self.tables.zero_()
        self.written.zero_()
        self.mean.zero_()
        self.seen.zero_()
        self.input_norm.fill_(1.0)

    def state(self) -> dict[str, Any]:
        return {
            "parameters": {k: v.detach().cpu().numpy() for k, v in self.parameters().items()},
            "tables": self.tables.cpu().numpy(),
            "mean": self.mean.cpu().numpy(),
            "seen": self.seen.cpu().numpy(),
            "input_norm": self.input_norm.cpu().numpy(),
            "settles": self.settles,
            "writes": self.writes,
        }

    def memory_bytes(self) -> int:
        size = self.tables.element_size()
        counted = sum(v.numel() for v in self.parameters().values())
        return int((counted + self.tables.numel() + self.mean.numel()) * size)
