"""Record patches joined by ports, evaluated by fixed Jacobi rounds at each moment.

The building block is the record patch, unchanged. A port is a declared seam between two of
them: a band of the source cortex's scaled context ``r * h`` (the unit its records read it in,
and the unit ``RecordPatchStack`` hands upward) is an input of the target cortex in the same
moment. The joint energy is the sum of every cortex's seam energy and, per port,
``1/2 |p[t] - S r h_source[t]|^2``. A simultaneous solution would make every seam zero. At a moment the
coupled equations are iterated for ``rounds`` Jacobi rounds from the previous moment's contexts:
round one is the delayed port (each cortex hears the other's context of the moment before),
and each further round re-reads the other's context of this moment. After the last round every
temporal seam is exactly zero (``damping`` one) and the port seams hold what remains: the seam
residual, logged per moment and per round as the fourth instrument. The slow gradient is one
backward scan through the moments and the rounds; with ``cross_adjoint`` the adjoint crosses
every port (the gradient of the target's loss reaches the source's context), without it the port
is a plain input. Records are read at the last-round readings after the scan and written after the
path, outside the gradient, exactly as in the record patch. With damping one and a single
one-way port, two rounds reach the fixed point; longer chains need more rounds. In general
the returned finite-round path need not be an equilibrium. The recorded movements and port
seam residuals are diagnostics, not convergence, contraction or uniqueness certificates.
The adjoint differentiates this finite computation, not an implicit equilibrium limit.

Snapshot and branch isolation: every cortex keeps its own parameters, records, running
statistics and live context; a snapshot is the list of the cortices' snapshots plus the
topology; ``imagine`` leaves all of them unchanged. Which cortex hears which, the band and the
width are a genome for ``evolve`` with ``genes``; the channels are ordered by timescale, so the
band decides what crosses.
"""
# mypy: ignore-errors
# The module's annotations follow in a later release; the finite-difference tests bind its arithmetic.
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .record_patch import RecordPatchNet, RecordPath, _input_norms, _integer

FORMAT = "cadence.record-ports/1"
_KEYS = ("G", "g", "B", "b", "C", "c")


class _NonfiniteJointPath(ValueError):
    """A numerical proposal failure, distinct from invalid public arguments."""


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass(frozen=True)
class Port:
    """The seam ``p = (r * h_source)[start:start + width]`` read by ``target``."""

    source: int
    target: int
    start: int
    width: int

    def __post_init__(self):
        for key in ("source", "target", "start", "width"):
            object.__setattr__(self, key, _integer(key, getattr(self, key), 1 if key == "width" else 0))

    def to_dict(self):
        return dict(source=self.source, target=self.target, start=self.start, width=self.width)

    @classmethod
    def from_dict(cls, value):
        return cls(value["source"], value["target"], value["start"], value["width"])


@dataclass(frozen=True)
class Settled:
    """The finite-round joint path, without an equilibrium guarantee: per cortex the full inputs,
    context, the per-round gate, port drive, port values and contexts, and the record path;
    per port the seam residual per round; per cortex the change of the context per round."""

    inputs: list[np.ndarray]
    hidden: list[np.ndarray]
    gate: list[np.ndarray]
    z: list[np.ndarray]
    p: list[np.ndarray]
    rounds: list[np.ndarray]
    paths: list[RecordPath | None]
    seam: np.ndarray
    settle: np.ndarray

    @property
    def loss(self):
        losses = [path.loss for path in self.paths]
        value = None if any(v is None for v in losses) else float(sum(losses))
        return value if value is not None and np.isfinite(value) else None

    @property
    def slow_loss(self):
        losses = [path.slow_loss for path in self.paths]
        value = None if any(v is None for v in losses) else float(sum(losses))
        return value if value is not None and np.isfinite(value) else None


@dataclass(frozen=True)
class JointObservation:
    updated: bool
    reason: str
    settled: Settled
    delta: list[dict[str, np.ndarray]] | None
    initial_loss: float | None
    final_loss: float | None
    accepted_rate: float
    replay_calls: int
    writes: int


class JointRecordPatches:
    """Several record patches joined by ports and iterated for a fixed number of rounds."""

    def __init__(self, cortices, own_inputs, ports, *, rounds=1, damping=1.0, cross_adjoint=True):
        self.cortices = list(cortices)
        if len({id(net) for net in self.cortices}) != len(self.cortices):
            raise ValueError("cortices must be distinct mutable patches")
        self.own = [_integer("own-input count", n, 0) for n in own_inputs]
        self.ports = [p if isinstance(p, Port) else Port.from_dict(p) for p in ports]
        self.rounds = _integer("rounds", rounds, 1)
        self.damping = float(damping)
        if not isinstance(cross_adjoint, (bool, np.bool_)):
            raise ValueError("cross_adjoint must be a boolean")
        self.cross_adjoint = bool(cross_adjoint)
        self.cut = False  # the ablation: every port carries zeros
        if self.rounds < 1 or not 0.0 < self.damping <= 1.0:
            raise ValueError("rounds is a positive integer and damping lies in (0, 1]")
        if not self.cortices or len(self.own) != len(self.cortices):
            raise ValueError("one own-input count per cortex")
        n = len(self.cortices)
        self.incoming = [[] for _ in range(n)]  # per target: (port, column offset within the port block)
        widths = [0] * n
        for port in self.ports:
            if not (0 <= port.source < n and 0 <= port.target < n) or port.source == port.target:
                raise ValueError("a port joins two different cortices")
            if port.width < 1 or port.start < 0 or port.start + port.width > self.cortices[port.source].hidden:
                raise ValueError("a port band lies inside the source's context")
            self.incoming[port.target].append((port, widths[port.target]))
            widths[port.target] += port.width
        self.port_width = widths
        for i, net in enumerate(self.cortices):
            if net.port is not None:
                raise ValueError("cortices with structured ports are not joined here")
            if net.inputs != self.own[i] + widths[i]:
                raise ValueError(f"cortex {i} reads {net.inputs} inputs; own {self.own[i]} plus ports {widths[i]} expected")

    # ------------------------------------------------------------------ parameters
    def parameters(self):
        return [net.parameters() for net in self.cortices]

    def parameter_count(self):
        return int(sum(v.size for p in self.parameters() for v in p.values()))

    def record_entries(self):
        return int(sum(net.records.parameters() for net in self.cortices))

    @property
    def state(self):
        return [net.state for net in self.cortices]

    def reset(self):
        for net in self.cortices:
            net.reset()

    # ------------------------------------------------------------------ the joint scan
    def _port_values(self, i, contexts, scales):
        """What cortex ``i``'s ports carry, from the given contexts (batch, hidden) of every cortex."""
        if not self.incoming[i]:
            return np.zeros((contexts[0].shape[0], 0))
        blocks = []
        for port, _ in self.incoming[i]:
            band = (contexts[port.source] * scales[port.source])[:, port.start:port.start + port.width]
            blocks.append(np.zeros_like(band) if self.cut else band)
        return np.concatenate(blocks, axis=1)

    def _settle(self, xs, boundaries, *, params=None, targets=None, reads=True):
        nets = self.cortices
        n = len(nets)
        par = [net.parameters() for net in nets] if params is None else params
        batch, horizon = xs[0].shape[:2]
        K, a = self.rounds, self.damping
        scales = [net._scale for net in nets]
        own_G = [xs[i] @ par[i]["G"][:, :self.own[i]].T + par[i]["g"] for i in range(n)]
        own_B = [xs[i] @ par[i]["B"][:, :self.own[i]].T + par[i]["b"] for i in range(n)]
        if not all(np.isfinite(v).all() for v in own_G + own_B):
            raise _NonfiniteJointPath("joint input drives must be finite")
        Gp = [par[i]["G"][:, self.own[i]:] for i in range(n)]
        Bp = [par[i]["B"][:, self.own[i]:] for i in range(n)]
        hidden = [np.empty((batch, horizon, net.hidden)) for net in nets]
        gate = [np.empty((batch, horizon, K, net.hidden)) for net in nets]
        z = [np.empty((batch, horizon, K, net.hidden)) for net in nets]
        p = [np.empty((batch, horizon, K, self.port_width[i])) for i in range(n)]
        rounds = [np.empty((batch, horizon, K + 1, net.hidden)) for net in nets]
        seam = np.zeros((batch, horizon, K, len(self.ports)))
        settle = np.zeros((batch, horizon, K, n))
        previous = [b.copy() for b in boundaries]
        for t in range(horizon):
            current = [h for h in previous]
            for i in range(n):
                rounds[i][:, t, 0] = current[i]
            for k in range(K):
                values = [self._port_values(i, current, scales) for i in range(n)]
                new = []
                for i in range(n):
                    gl = _sigmoid(own_G[i][:, t] + values[i] @ Gp[i].T)
                    zz = np.tanh(own_B[i][:, t] + values[i] @ Bp[i].T)
                    q = gl * previous[i] + (1.0 - gl) * zz
                    h = q if a == 1.0 else (1.0 - a) * current[i] + a * q
                    gate[i][:, t, k], z[i][:, t, k], p[i][:, t, k], rounds[i][:, t, k + 1] = gl, zz, values[i], h
                    settle[:, t, k, i] = np.sqrt(np.mean((h - current[i]) ** 2, axis=1))
                    new.append(h)
                for j, port in enumerate(self.ports):
                    offset = next(o for q_, o in self.incoming[port.target] if q_ is port)
                    carried = values[port.target][:, offset:offset + port.width]
                    holds = (new[port.source] * scales[port.source])[:, port.start:port.start + port.width]
                    seam[:, t, k, j] = np.sqrt(np.mean((carried - holds) ** 2, axis=1))
                current = new
            previous = current
            for i in range(n):
                hidden[i][:, t] = current[i]
        inputs = [np.concatenate((xs[i], p[i][:, :, K - 1]), axis=-1) for i in range(n)]
        paths = []
        for i, net in enumerate(nets):
            slow = net._slow(hidden[i], par[i]["C"], par[i]["c"])
            if reads:
                readings = net._readings(inputs[i], hidden[i]).reshape(batch * horizon, -1)
                if not np.isfinite(readings).all():
                    raise _NonfiniteJointPath("joint record readings must be finite")
                codes = net.records.code(readings, valued=False)[0].reshape(batch, horizon, -1)
                read = codes @ net.records.tables["y"]
                if net._output_code is not None:
                    read = read @ net._output_code.T
            else:
                read = np.zeros_like(slow)
            output = slow + read
            target = None if targets is None else targets[i]
            loss = None if target is None else net._loss(output, target)
            slow_loss = None if target is None else net._loss(slow, target)
            paths.append(RecordPath(hidden[i], output, gate[i][:, :, K - 1], read, loss, slow_loss))
        values = inputs + hidden + gate + z + p + rounds + [seam, settle]
        values += [v for path in paths for v in (path.output, path.read)]
        if not all(np.isfinite(v).all() for v in values):
            raise _NonfiniteJointPath("joint path must be finite")
        return Settled(inputs, hidden, gate, z, p, rounds, paths, seam, settle)

    # ------------------------------------------------------------------ the joint adjoint
    def _adjoint(self, F, boundaries, targets):
        nets = self.cortices
        n = len(nets)
        K, a = self.rounds, self.damping
        batch, horizon = F.hidden[0].shape[:2]
        par = [net.parameters() for net in nets]
        Gp = [par[i]["G"][:, self.own[i]:] for i in range(n)]
        Bp = [par[i]["B"][:, self.own[i]:] for i in range(n)]
        scales = [net._scale for net in nets]
        d = [net._output_error(F.paths[i], targets[i]) for i, net in enumerate(nets)]
        gH = [d[i] @ par[i]["C"] for i in range(n)]
        gs_own = [np.zeros_like(F.hidden[i]) for i in range(n)]
        gd_own = [np.zeros_like(F.hidden[i]) for i in range(n)]
        gs_port = [np.zeros((batch, horizon, K, nets[i].hidden)) for i in range(n)]
        gd_port = [np.zeros((batch, horizon, K, nets[i].hidden)) for i in range(n)]
        carried = [np.zeros((batch, nets[i].hidden)) for i in range(n)]
        for t in range(horizon - 1, -1, -1):
            prev = [boundaries[i] if t == 0 else F.hidden[i][:, t - 1] for i in range(n)]
            gk = [[np.zeros((batch, nets[i].hidden)) for _ in range(K + 1)] for i in range(n)]
            for i in range(n):
                gk[i][K] = gH[i][:, t] + carried[i]
            gprev = [np.zeros((batch, nets[i].hidden)) for i in range(n)]
            for k in range(K, 0, -1):
                for i in range(n):
                    g = gk[i][k]
                    gq = g if a == 1.0 else a * g
                    if a != 1.0:
                        gk[i][k - 1] += (1.0 - a) * g
                    gl, zz = F.gate[i][:, t, k - 1], F.z[i][:, t, k - 1]
                    gprev[i] += gl * gq
                    gs = gq * (prev[i] - zz) * gl * (1.0 - gl)
                    gd = gq * (1.0 - gl) * (1.0 - zz ** 2)
                    gs_own[i][:, t] += gs
                    gd_own[i][:, t] += gd
                    gs_port[i][:, t, k - 1] = gs
                    gd_port[i][:, t, k - 1] = gd
                    if self.cross_adjoint and self.incoming[i] and not self.cut:
                        gp = gs @ Gp[i] + gd @ Bp[i]
                        for port, offset in self.incoming[i]:
                            gk[port.source][k - 1][:, port.start:port.start + port.width] += (
                                gp[:, offset:offset + port.width] * scales[port.source][port.start:port.start + port.width])
            for i in range(n):
                carried[i] = gprev[i] + gk[i][0]
        deltas = []
        for i in range(n):
            x = F.inputs[i][..., :self.own[i]]
            G = np.concatenate((np.einsum("bth,btx->hx", gs_own[i], x), np.einsum("btkh,btkw->hw", gs_port[i], F.p[i])), axis=1)
            B = np.concatenate((np.einsum("bth,btx->hx", gd_own[i], x), np.einsum("btkh,btkw->hw", gd_port[i], F.p[i])), axis=1)
            deltas.append(dict(G=G, g=gs_own[i].sum(axis=(0, 1)), B=B, b=gd_own[i].sum(axis=(0, 1)),
                               C=np.einsum("bti,btj->ij", d[i], F.hidden[i]), c=d[i].sum(axis=(0, 1))))
        return deltas

    # ------------------------------------------------------------------ interface
    def _check(self, xs, targets=None):
        if len(xs) != len(self.cortices):
            raise ValueError("one own-input path per cortex")
        paths = [net._path(x, self.own[i], f"inputs of cortex {i}") for i, (net, x) in enumerate(zip(self.cortices, xs, strict=True))]
        if len({p.shape[:2] for p in paths}) != 1:
            raise ValueError("every cortex sees the same batch and horizon")
        teach = None
        if targets is not None:
            if len(targets) != len(self.cortices):
                raise ValueError("one target path per cortex")
            teach = []
            for i, net in enumerate(self.cortices):
                full = np.zeros((paths[i].shape[0], paths[i].shape[1], net.inputs))
                full[..., :self.own[i]] = paths[i]
                teach.append(net._teaching(full, targets[i])[1])
        return paths, teach

    def imagine(self, xs, *, states=None):
        paths, _ = self._check(xs)
        if states is not None and len(states) != len(self.cortices):
            raise ValueError("one boundary state per cortex")
        boundaries = [net._boundary(len(paths[0]), None if states is None else states[i]) for i, net in enumerate(self.cortices)]
        return self._settle(paths, boundaries)

    def advance(self, xs):
        paths, _ = self._check(xs)
        boundaries = [net._boundary(len(paths[0]), None) for net in self.cortices]
        F = self._settle(paths, boundaries)
        for net, path in zip(self.cortices, F.paths, strict=True):
            net._carry(path)
        return F

    def observe(self, xs, targets, *, rate=1.0, backtrack=False, write=True):
        """Run the finite joint scan and its adjoint, optionally writing observed records.

        Invalid inputs or exceptions during record writes leave every cortex unchanged.
        A rejected slow-parameter proposal may still retain valid activity and record writes,
        as in ``RecordPatchNet.observe``.
        """
        if not isinstance(backtrack, (bool, np.bool_)) or not isinstance(write, (bool, np.bool_)):
            raise ValueError("backtrack and write must be booleans")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        paths, teach = self._check(xs, targets)
        boundaries = [net._boundary(len(paths[0]), None) for net in self.cortices]
        F = self._settle(paths, boundaries, targets=teach)
        if F.loss is None or F.slow_loss is None:
            return JointObservation(False, "nonfinite_prediction", F, None, None, None, 0.0, 0, 0)
        deltas = self._adjoint(F, boundaries, teach)
        writes = 0
        if write:
            for inputs in F.inputs:
                _input_norms(inputs)
            # A later cortex's write can overflow after earlier writes succeeded.
            # Keep learned record state only; fixed projections and parameters are shared.
            saved = [(net._input_norm, net.records.state()) for net in self.cortices]
            try:
                for i, net in enumerate(self.cortices):
                    writes += net._write(F.inputs[i], teach[i], F.hidden[i])
            except Exception:
                for net, (norm, records) in zip(self.cortices, saved, strict=True):
                    net._input_norm = norm
                    net.records.load_state(records)
                raise
        for net, path in zip(self.cortices, F.paths, strict=True):
            net._carry(path)
        initial = F.slow_loss
        if rate == 0:
            return JointObservation(False, "no_step", F, deltas, initial, initial, 0.0, 0, writes)
        current = self.parameters()
        if not backtrack:
            proposed = [{k: current[i][k] - rate * deltas[i][k] for k in _KEYS} for i in range(len(current))]
            if not all(np.isfinite(v).all() for p in proposed for v in p.values()):
                return JointObservation(False, "nonfinite_update", F, deltas, initial, None, 0.0, 0, writes)
            for net, p in zip(self.cortices, proposed, strict=True):
                net._commit(p)
            return JointObservation(True, "updated", F, deltas, initial, None, rate, 0, writes)
        norm_squared = float(sum(np.sum(v * v) for dl in deltas for v in dl.values()))
        replays = 0
        if np.isfinite(norm_squared) and norm_squared > 0 and rate > 0:
            for index in range(16):
                step = rate * 0.5 ** index
                proposed = [{k: current[i][k] - step * deltas[i][k] for k in _KEYS} for i in range(len(current))]
                if not all(np.isfinite(v).all() for p in proposed for v in p.values()):
                    continue
                try:
                    loss = self._settle(paths, boundaries, params=proposed, targets=teach, reads=False).slow_loss
                except _NonfiniteJointPath:
                    loss = None
                replays += 1
                if loss is None:
                    continue
                floor = 64 * np.finfo(float).eps * max(abs(initial), abs(loss), np.finfo(float).tiny)
                if loss < initial - floor and loss <= initial - 1e-4 * step * norm_squared:
                    for net, p in zip(self.cortices, proposed, strict=True):
                        net._commit(p)
                    return JointObservation(True, "updated", F, deltas, initial, loss, step, replays, writes)
        return JointObservation(False, "no_decreasing_parameter_step", F, deltas, initial, initial, 0.0, replays, writes)

    # ------------------------------------------------------------------ custody
    def snapshot(self):
        meta = dict(format=FORMAT, own=self.own, ports=[p.to_dict() for p in self.ports], rounds=self.rounds,
                    damping=self.damping, cross_adjoint=self.cross_adjoint, cut=self.cut, cortices=len(self.cortices))
        out = {"meta": np.array(json.dumps(meta, sort_keys=True, allow_nan=False))}
        for i, net in enumerate(self.cortices):
            for k, v in net.snapshot().items():
                out[f"cortex{i}_{k}"] = v
        return out

    @classmethod
    def restore(cls, snapshot):
        try:
            meta = json.loads(str(snapshot["meta"]))
            if not isinstance(meta, dict) or meta.get("format") != FORMAT:
                raise ValueError("unsupported record-ports checkpoint")
            count = _integer("cortices", meta["cortices"], 1)
            cut = meta.get("cut", False)  # Older checkpoints always restored connected ports.
            if not isinstance(cut, bool):
                raise ValueError("cut must be a boolean")
            nets = []
            for i in range(count):
                prefix = f"cortex{i}_"
                nets.append(RecordPatchNet.restore({k[len(prefix):]: v for k, v in snapshot.items() if k.startswith(prefix)}))
            out = cls(nets, meta["own"], meta["ports"], rounds=meta["rounds"], damping=meta["damping"], cross_adjoint=meta["cross_adjoint"])
            out.cut = cut
            return out
        except (KeyError, TypeError, IndexError, OverflowError) as error:
            raise ValueError("invalid record-ports checkpoint") from error

    def clone(self):
        return JointRecordPatches.restore(self.snapshot())

    def save(self, path):
        from .checkpoint import _write

        return _write(self.snapshot(), path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files})


def build(hidden, own_inputs, outputs, ports, *, seed, rounds=1, damping=1.0, cross_adjoint=True, groups=None, **records):
    """Record patches of the given widths joined by the ports; ``groups[i]`` are cortex i's categorical groups."""
    ports = [p if isinstance(p, Port) else Port.from_dict(p) for p in ports]
    widths = [sum(p.width for p in ports if p.target == i) for i in range(len(hidden))]
    nets = [RecordPatchNet(own_inputs[i] + widths[i], hidden[i], outputs[i], seed=seed + 101 * i,
                           groups=None if groups is None else groups[i], **records) for i in range(len(hidden))]
    return JointRecordPatches(nets, own_inputs, ports, rounds=rounds, damping=damping, cross_adjoint=cross_adjoint)
