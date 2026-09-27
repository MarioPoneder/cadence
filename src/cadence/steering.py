"""A brain that reads itself: a cortex whose senses are weighed by a steering patch of the same rule.

The cortex is a ``BeliefPatch`` with a gain per block of its observation port. The steering patch is
a second ``BeliefPatch`` whose observation is the cortex's readback of the moment, what each sense is
saying and how far the belief disagrees with it, and whose outputs a weighing turns into the gains
the cortex's repair runs under. Nothing here is a new rule: two patches, a seam, and one admitted step.

    r[t]      = [probe_b; surprise_b; residual[t-1]; output[t-1]; evidence_b; extra[t]]   the readback
    y_s[t]    = steering(r[t])                       the steering patch's moment
    gain[t]   = W(y_s[t])                            the weighing: a softmax over the blocks, or a window
    z[t]      = cortex(o[t], a[t]; gain[t])          the cortex's moment under the gains

Which channels of the readback the steering patch hears is a gene, the mask of its port. The
weighing is the application's declaration: ``Softmax`` moves weight between senses (gains that sum
to the number of blocks), ``Gaze`` is a window over a ring of blocks whose centre the steering output
turns, under a price on the turn, and ``Rule`` is a hand-written map from the readback to the gains,
the control. Learning is within the life: the cortex's adjoint gives its deltas and the gradient
into the gains; the weighing pulls that gradient back to the steering patch's outputs; the steering
patch's adjoint under it gives its deltas; one joint step is admitted by a replay of the chunk (the
steering patch on the recorded readbacks, taken as given, the cortex under the gains it then
returns), halved while the objective does not fall by the Armijo margin, starting from twice the last
admitted step. The cortex can sleep while the steering patch learns.

Three demos wrote this composition before it was here: the ventriloquist (rung 2), the lighthouse
keeper (rung 3) and the night nursery (rung 4).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .belief import _HALVINGS, BeliefPatch, BeliefPath

FORMAT = "cadence-steered/1"


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------------- the weighings
class Weighing:
    """The map from the steering patch's outputs to the cortex's gains, with its pull (the
    Jacobian's action on the gradient into the gains) and an optional price. A weighing may
    carry a state from moment to moment (``begin`` starts it, ``gains`` advances it)."""

    outputs: int
    blocks: int

    def begin(self, n: int) -> Any:
        return None

    def gains(self, y: np.ndarray, state: Any) -> tuple[np.ndarray, Any]:
        raise NotImplementedError

    def pull(
        self, ys: np.ndarray, gains: np.ndarray, dgains: np.ndarray, states: Sequence[Any]
    ) -> np.ndarray:
        raise NotImplementedError

    def price(self, ys: np.ndarray, states: Sequence[Any]) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


class Softmax(Weighing):
    """Gains that sum to the number of blocks, ``blocks * softmax(span * tanh(y))``: the
    weighing moves weight from one sense to another and cannot amplify all of them. The
    ventriloquist's and the night nursery's."""

    def __init__(self, blocks: int, *, span: float = 1.0) -> None:
        self.blocks = int(blocks)
        self.outputs = self.blocks
        if self.blocks < 2:
            raise ValueError("a softmax weighing needs at least two blocks")
        if not np.isfinite(span) or span <= 0:
            raise ValueError("span must be finite and positive")
        self.span = float(span)

    def gains(self, y: np.ndarray, state: Any) -> tuple[np.ndarray, Any]:
        u = self.span * np.tanh(y)
        u = u - u.max(axis=-1, keepdims=True)
        w = np.exp(u)
        return self.blocks * w / w.sum(axis=-1, keepdims=True), None

    def pull(
        self, ys: np.ndarray, gains: np.ndarray, dgains: np.ndarray, states: Sequence[Any]
    ) -> np.ndarray:
        du = gains * (dgains - np.sum(dgains * gains, axis=-1, keepdims=True) / self.blocks)
        return np.asarray(du * self.span * (1.0 - np.tanh(ys) ** 2))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "softmax", "blocks": self.blocks, "span": self.span}


class Gaze(Weighing):
    """A window over a ring of blocks (bearings, retinal columns) whose centre the steering
    patch turns: ``turn = span * tanh(y)``, the centre advances by it, and each block's gain is
    a Gaussian in its distance from the centre, cut beyond ``cut`` widths, at a brightness
    that falls as the window widens (``min(1, lamp / sigma)``). The centre is the weighing's
    state and travels in the boundary. ``price`` prices the turn, ``0.5 * price * mean(turn^2)``,
    in the objective the joint step is admitted on. The pull is local in time: the dependence
    of later centres on an earlier turn is not propagated, as in the lighthouse keeper, whose
    beam this is."""

    def __init__(
        self,
        blocks: int,
        *,
        sigma: float,
        cut: float = 2.5,
        lamp: float = 0.2,
        span: float = 0.5,
        price: float = 0.0,
        start: float = 0.0,
    ) -> None:
        self.blocks = int(blocks)
        self.outputs = 1
        if self.blocks < 2:
            raise ValueError("a gaze needs at least two blocks")
        for name, value in (("sigma", sigma), ("cut", cut), ("lamp", lamp), ("span", span)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(price) or price < 0:
            raise ValueError("price must be finite and nonnegative")
        self.sigma, self.cut, self.lamp, self.span = float(sigma), float(cut), float(lamp), float(span)
        self.price_per_turn = float(price)
        self.start = float(start)
        self.bearings = 2.0 * np.pi * np.arange(self.blocks) / self.blocks

    def begin(self, n: int) -> np.ndarray:
        return np.full(n, self.start)

    def turn(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.span * np.tanh(y[..., 0]))

    def profile(self, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The gains ``(batch, blocks)`` around ``centre`` ``(batch,)`` and their derivative in
        the centre."""
        d = _wrap(self.bearings[None, :] - centre[:, None])
        g = min(1.0, self.lamp / self.sigma) * np.exp(-0.5 * (d / self.sigma) ** 2)
        g = np.where(np.abs(d) > self.cut * self.sigma, 0.0, g)
        return g, g * d / self.sigma**2

    def gains(self, y: np.ndarray, state: Any) -> tuple[np.ndarray, Any]:
        centre = _wrap(np.asarray(state, dtype=float) + self.turn(y))
        return self.profile(centre)[0], centre

    def pull(
        self, ys: np.ndarray, gains: np.ndarray, dgains: np.ndarray, states: Sequence[Any]
    ) -> np.ndarray:
        n, t = ys.shape[:2]
        turns = self.turn(ys)
        dcentre = np.zeros((n, t))
        for k in range(t):
            centre = _wrap(np.asarray(states[k], dtype=float) + turns[:, k])  # the centre the gains were read at
            _, jac = self.profile(centre)
            dcentre[:, k] = np.sum(dgains[:, k] * jac, axis=-1)
        dturn = dcentre + self.price_per_turn * turns / (n * t)
        return np.asarray((dturn * self.span * (1.0 - np.tanh(ys[..., 0]) ** 2))[..., None])

    def price(self, ys: np.ndarray, states: Sequence[Any]) -> float:
        return float(0.5 * self.price_per_turn * np.mean(self.turn(ys) ** 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "gaze",
            "blocks": self.blocks,
            "sigma": self.sigma,
            "cut": self.cut,
            "lamp": self.lamp,
            "span": self.span,
            "price": self.price_per_turn,
            "start": self.start,
        }


class Rule:
    """A hand-written map from the readback ``(batch, channels)`` to the gains ``(batch,
    blocks)``: the control of a rung, never a result. It has no steering patch and learns
    nothing; the cortex under it learns by its own admitted step."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], *, macs: int = 0) -> None:
        if not callable(fn):
            raise ValueError("a rule is a callable from the readback to the gains")
        self.fn = fn
        self.macs = int(macs)

    def __call__(self, readback: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(readback), dtype=float)


def weighing_from_dict(d: Mapping[str, Any]) -> Weighing:
    kind = d["kind"]
    if kind == "softmax":
        return Softmax(int(d["blocks"]), span=float(d["span"]))
    if kind == "gaze":
        return Gaze(
            int(d["blocks"]),
            sigma=float(d["sigma"]),
            cut=float(d["cut"]),
            lamp=float(d["lamp"]),
            span=float(d["span"]),
            price=float(d["price"]),
            start=float(d["start"]),
        )
    raise ValueError(f"unknown weighing {kind!r}")


# ---------------------------------------------------------------------------- the composition
@dataclass(frozen=True)
class Boundary:
    """Where a steered life is between two moments: the cortex's belief ``(batch, belief)``,
    the steering patch's belief, the previous moment's output and repair residual, the
    weighing's state and the count of moments seen. ``run(state=boundary)`` starts there."""

    cortex: np.ndarray
    steering: np.ndarray | None
    output: np.ndarray | None
    residual: np.ndarray
    weighing: Any
    moments: int


@dataclass(frozen=True)
class SteeredPath:
    """A chunk lived: the outputs ``(batch, time, outputs)``, the gains ``(batch, time,
    blocks)``, the readbacks ``(batch, time, channels)``, the cortex's repair residuals
    ``(batch, time)``, the steering patch's outputs, the loss and the price when a target
    was given, and what the joint step did."""

    output: np.ndarray
    gains: np.ndarray
    readback: np.ndarray
    residual: np.ndarray
    steering_output: np.ndarray | None
    loss: float | None
    price: float
    updated: bool = False
    steering_updated: bool = False
    step: float | None = None
    halvings: int = 0
    replays: int = 0
    reason: str = "no_step"
    last: BeliefPath | None = None
    last_steering: BeliefPath | None = None

    @property
    def objective(self) -> float | None:
        return None if self.loss is None else self.loss + self.price


class Steered:
    """A cortex weighed by a steering patch, a rule, or nothing (fixed gains of one: the brain
    below rung 2). ``reads_output`` adds the previous moment's outputs to the readback,
    ``evidence`` the encoded evidence of the named blocks, ``extra`` a callable from the
    moment (readback, previous output, previous residual) to further channels ``(batch, k)``
    with ``extra_channels`` of them. The steering patch's port must read exactly the
    readback's channels; its mask says which it hears. ``rate_scale`` scales the steering
    patch's step against the cortex's."""

    def __init__(
        self,
        cortex: BeliefPatch,
        steering: BeliefPatch | None = None,
        weighing: Weighing | Rule | None = None,
        *,
        reads_output: bool = False,
        evidence: Sequence[int] = (),
        extra: Callable[..., np.ndarray] | None = None,
        extra_channels: int = 0,
        rate_scale: float = 1.0,
    ) -> None:
        self.cortex = cortex
        self.steering = steering
        self.rule: Rule | None = None
        self.weighing: Weighing | None = None
        if isinstance(weighing, Rule):
            if steering is not None:
                raise ValueError("a rule sets the gains itself; give no steering patch")
            self.rule = weighing
        elif steering is not None:
            if weighing is None:
                weighing = Softmax(cortex.block_count)
            if not isinstance(weighing, Weighing):
                raise ValueError("weighing must be a Weighing or a Rule")
            if weighing.blocks != cortex.block_count:
                raise ValueError("the weighing's blocks must match the cortex's port")
            if steering.outputs != weighing.outputs:
                raise ValueError(
                    f"the steering patch must have {weighing.outputs} outputs for this weighing"
                )
            self.weighing = weighing
        elif weighing is not None:
            raise ValueError("a weighing needs a steering patch")
        self.reads_output = bool(reads_output)
        self.evidence = tuple(int(b) for b in evidence)
        for b in self.evidence:
            if not 0 <= b < cortex.block_count:
                raise ValueError("evidence names a block the cortex does not have")
        if (extra is None) != (extra_channels == 0):
            raise ValueError("extra and extra_channels come together")
        self.extra, self.extra_channels = extra, int(extra_channels)
        if not np.isfinite(rate_scale) or rate_scale <= 0:
            raise ValueError("rate_scale must be finite and positive")
        self.rate_scale = float(rate_scale)
        self.layout: list[tuple[str, int]] = (
            [(f"probe:{b}", 1) for b in range(cortex.block_count)]
            + [(f"surprise:{b}", 1) for b in range(cortex.block_count)]
            + [("residual", 1)]
            + ([(f"output:{i}", 1) for i in range(cortex.outputs)] if self.reads_output else [])
            + [(f"evidence:{b}", cortex.port.blocks[b].outputs) for b in self.evidence]
            + ([(f"extra:{i}", 1) for i in range(self.extra_channels)] if self.extra else [])
        )
        self.channels = int(sum(size for _, size in self.layout))
        if steering is not None and steering.inputs != self.channels:
            raise ValueError(
                f"the steering patch's port must read the {self.channels} readback channels"
            )
        self.ablation: str | Callable[[np.ndarray], np.ndarray] | None = None
        self.deaf: np.ndarray | None = None
        self._live: Boundary | None = None
        self._step_size: float | None = None
        self._replays = 0

    # ------------------------------------------------------------------ what it is
    @property
    def readback_names(self) -> list[str]:
        out = []
        for name, size in self.layout:
            out += [name] if size == 1 else [f"{name}:{i}" for i in range(size)]
        return out

    @property
    def probes_on(self) -> bool:
        """Whether the probes are computed: a steering patch whose port hears none of the
        probe channels, a rule, or fixed gains do without them."""
        if self.steering is None:
            return False
        mask = self.steering.port.mask
        return True if mask is None else bool(mask[: self.cortex.block_count].any())

    @property
    def step_size(self) -> float | None:
        return self._step_size

    def reset_step(self) -> None:
        self._step_size = None

    def moments_per_decision(self) -> int:
        return 1 + (1 if self.steering is not None else 0)

    def macs_per_moment(self) -> int:
        """One accounting for every arm: the cortex's moment with the probes when read, the
        steering patch's moment, or the rule's declared operations."""
        macs = self.cortex.macs_per_moment(probes=self.probes_on)
        if self.steering is not None:
            macs += self.steering.macs_per_moment()
        if self.rule is not None:
            macs += self.rule.macs
        return int(macs)

    @property
    def cost(self) -> dict[str, int]:
        """The two patches' counters summed; ``replays`` counts the joint admission's replays
        beside the cortex's own."""
        total = dict(self.cortex.cost)
        if self.steering is not None:
            for key, value in self.steering.cost.items():
                total[key] += value
        total["replays"] += self._replays
        return total

    def reset_cost(self) -> None:
        self.cortex.reset_cost()
        if self.steering is not None:
            self.steering.reset_cost()
        self._replays = 0

    def parameter_count(self) -> int:
        count = sum(v.size for v in self.cortex.parameters().values())
        if self.steering is not None:
            count += sum(v.size for v in self.steering.parameters().values())
        return int(count)

    def parameters(self) -> dict[str, dict[str, np.ndarray]]:
        out = {"cortex": self.cortex.parameters()}
        if self.steering is not None:
            out["steering"] = self.steering.parameters()
        return out

    def set_parameters(self, parameters: Mapping[str, Mapping[str, np.ndarray]]) -> None:
        self.cortex.set_parameters(parameters["cortex"])
        if self.steering is not None:
            self.steering.set_parameters(parameters["steering"])

    # ------------------------------------------------------------------ the boundary
    def _fresh(self, n: int) -> Boundary:
        return Boundary(
            np.zeros((n, self.cortex.belief)),
            None if self.steering is None else np.zeros((n, self.steering.belief)),
            None,
            np.zeros(n),
            None if self.weighing is None else self.weighing.begin(n),
            0,
        )

    def boundary(self) -> Boundary | None:
        """Where the life is, to be kept before a window and replayed from after it."""
        return self._live

    def reset(self) -> None:
        """Forget the live boundary; the step size and the counters stay."""
        self._live = None
        self.cortex.reset()
        if self.steering is not None:
            self.steering.reset()

    def _check_boundary(self, state: Boundary, n: int) -> None:
        if state.cortex.shape != (n, self.cortex.belief):
            raise ValueError("the boundary's cortex belief must match (batch, belief)")
        if (self.steering is None) != (state.steering is None):
            raise ValueError("the boundary must carry a steering belief exactly when there is a steering patch")
        if state.steering is not None and state.steering.shape != (n, self.steering.belief):  # type: ignore[union-attr]
            raise ValueError("the boundary's steering belief must match (batch, belief)")

    # ------------------------------------------------------------------ the moments
    def _readback(
        self,
        o_k: np.ndarray,
        a_k: np.ndarray,
        z_c: np.ndarray,
        prev_output: np.ndarray | None,
        prev_residual: np.ndarray,
    ) -> np.ndarray:
        n = len(o_k)
        rb = self.cortex.readback(o_k, a_k, state=z_c, probe=self.probes_on)
        parts = [
            rb.residual_alone,
            np.zeros((n, self.cortex.block_count)) if rb.surprise is None else rb.surprise,
            prev_residual[:, None],
        ]
        if self.reads_output:
            parts.append(np.zeros((n, self.cortex.outputs)) if prev_output is None else prev_output)
        for b in self.evidence:
            parts.append(rb.evidence[:, self.cortex._block_slices[b]])
        if self.extra is not None:
            more = np.asarray(self.extra(rb, prev_output, prev_residual), dtype=float)
            if more.shape != (n, self.extra_channels):
                raise ValueError(f"extra must return (batch, {self.extra_channels})")
            parts.append(more)
        r = np.concatenate(parts, axis=-1)
        if self.deaf is not None:
            r = r * self.deaf
        return np.asarray(r)

    def _weigh_one(self, r: np.ndarray, z_s: np.ndarray | None, wstate: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Any, BeliefPath | None]:
        n = len(r)
        y_s = path_s = None
        if self.steering is not None:
            assert self.weighing is not None
            path_s = self.steering.assimilate(
                r[:, None], np.zeros((n, 1, self.steering.actions)), state=z_s, keep_live=True
            )
            z_s = path_s.final_state
            y_s = path_s.output[:, 0]
            g, wstate = self.weighing.gains(y_s, wstate)
        elif self.rule is not None:
            g = self.rule(r)
            if g.shape != (n, self.cortex.block_count):
                raise ValueError("the rule must return (batch, blocks) gains")
        else:
            g = np.ones((n, self.cortex.block_count))
        g = self._ablate(np.asarray(g, dtype=float))
        return g, z_s, y_s, wstate, path_s

    def _ablate(self, g: np.ndarray) -> np.ndarray:
        """The test-time ablation of the gains: ``"cut"`` sets them to one, a callable maps them
        (the ventriloquist's shuffle, ``g[:, ::-1]``); the steering patch still runs and counts."""
        if self.ablation is None:
            return g
        if self.ablation == "cut":
            return np.ones_like(g)
        if callable(self.ablation):
            out = np.asarray(self.ablation(g), dtype=float)
            if out.shape != g.shape:
                raise ValueError("an ablation must return gains of the same shape")
            return out
        raise ValueError("ablation is None, 'cut' or a callable over the gains")

    def _forward(
        self, o: np.ndarray, a: np.ndarray, start: Boundary
    ) -> tuple[dict[str, Any], Boundary]:
        n, t = o.shape[:2]
        z_c, z_s = start.cortex.copy(), None if start.steering is None else start.steering.copy()
        prev_output = None if start.output is None else start.output.copy()
        prev_residual = start.residual.copy()
        wstate = start.weighing
        col: dict[str, list[Any]] = {k: [] for k in ("y", "g", "r", "res", "ys", "read", "wstate")}
        path = path_s = None
        for k in range(t):
            r = self._readback(o[:, k], a[:, k], z_c, prev_output, prev_residual)
            g, z_s, y_s, wstate, path_s = self._weigh_one(r, z_s, wstate)
            path = self.cortex.assimilate(
                o[:, k][:, None], a[:, k][:, None], gains=g[:, None], state=z_c, keep_live=True
            )
            z_c = path.final_state
            prev_output, prev_residual = path.output[:, 0], path.residual[:, 0]
            for key, value in (("y", path.output[:, 0]), ("g", g), ("r", r), ("res", prev_residual), ("ys", y_s), ("read", path.read[:, 0]), ("wstate", wstate)):
                col[key].append(value)
        col["last"], col["last_s"] = path, path_s
        end = Boundary(z_c, z_s, prev_output, prev_residual, wstate, start.moments + t)
        return col, end

    def _weigh_chunk(self, ys: np.ndarray, wstate: Any) -> np.ndarray:
        assert self.weighing is not None
        gains, state = [], wstate
        for k in range(ys.shape[1]):
            g, state = self.weighing.gains(ys[:, k], state)
            gains.append(g)
        out = np.stack(gains, axis=1)
        return self._ablate(out) if self.ablation is not None else out

    def _objective(
        self,
        o: np.ndarray,
        a: np.ndarray,
        target: np.ndarray,
        r: np.ndarray,
        gains: np.ndarray,
        start: Boundary,
    ) -> float | None:
        """The chunk's objective under the present parameters: the steering patch replayed on
        the recorded readbacks, taken as given, the cortex under the gains it then returns."""
        n, t = o.shape[:2]
        price = 0.0
        if self.steering is not None and self.ablation is None:
            assert self.weighing is not None
            ys = self.steering.assimilate(
                r, np.zeros((n, t, self.steering.actions)), state=start.steering, keep_live=True
            ).output
            gains = self._weigh_chunk(ys, start.weighing)
            states = [start.weighing]
            state = start.weighing
            for k in range(t):
                _, state = self.weighing.gains(ys[:, k], state)
                states.append(state)
            price = self.weighing.price(ys, states[:-1])
        with np.errstate(over="ignore", invalid="ignore"):
            replay = self.cortex.assimilate(o, a, gains=gains, state=start.cortex, keep_live=True)
            loss = self.cortex._loss(replay.slow_output, target)
        return None if loss is None else loss + price

    def run(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        target: np.ndarray | None = None,
        *,
        rate: float = 0.0,
        state: Boundary | None = None,
        keep_live: bool = False,
        learn_cortex: bool = True,
        backtrack: bool = True,
    ) -> SteeredPath:
        """Live one chunk ``(batch, time, inputs)``, ``(batch, time, actions)`` from the live
        boundary (or ``state``): the readback, the gains, the cortex's moment, in order. With a
        ``target`` ``(batch, time, outputs)`` and a ``rate``, both patches take one joint step,
        admitted by a replay of the chunk (halved while the objective does not fall by the
        Armijo margin, from twice the last admitted step, at most ``rate``); with
        ``learn_cortex=False`` the cortex sleeps and only the steering patch steps, its gradient
        still the cortex's. ``keep_live=True`` leaves the live boundary where it was."""
        o = np.asarray(observations, dtype=float)
        a = np.asarray(actions, dtype=float)
        if o.ndim != 3 or o.shape[2] != self.cortex.inputs or not np.isfinite(o).all():
            raise ValueError(f"observations must be a finite (batch, time, {self.cortex.inputs}) array")
        n, t = o.shape[:2]
        if a.shape != (n, t, self.cortex.actions) or not np.isfinite(a).all():
            raise ValueError(f"actions must be a finite (batch, time, {self.cortex.actions}) array")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be finite and nonnegative")
        start = state if state is not None else (self._live if self._live is not None and len(self._live.cortex) == n else self._fresh(n))
        self._check_boundary(start, n)
        col, end = self._forward(o, a, start)
        if not keep_live:
            self._live = end
            self.cortex._state = end.cortex.copy()
            if self.steering is not None and end.steering is not None:
                self.steering._state = end.steering.copy()
        y = np.stack(col["y"], axis=1)
        gains = np.stack(col["g"], axis=1)
        r = np.stack(col["r"], axis=1)
        residual = np.stack(col["res"], axis=1)
        ys = None if col["ys"][0] is None else np.stack(col["ys"], axis=1)
        result = dict(output=y, gains=gains, readback=r, residual=residual, steering_output=ys, loss=None, price=0.0, last=col["last"], last_steering=col["last_s"])
        if target is None:
            return SteeredPath(**result)
        target = np.asarray(target, dtype=float)
        if target.shape != (n, t, self.cortex.outputs) or not np.isfinite(target).all():
            raise ValueError("target must be a finite (batch, time, outputs) array")
        slow = y - np.stack(col["read"], axis=1)
        loss = self.cortex._loss(slow, target)
        price = 0.0
        if self.steering is not None and self.ablation is None and ys is not None:
            assert self.weighing is not None
            price = self.weighing.price(ys, [start.weighing] + col["wstate"][:-1])
        result.update(loss=loss, price=price)
        if rate == 0 or loss is None:
            return SteeredPath(**result)
        # the cortex under a rule, fixed gains or an ablation learns by its own admitted step
        if self.steering is None or self.ablation is not None:
            if not learn_cortex:
                return SteeredPath(**result)
            taught = self.cortex.observe(
                o, a, target, gains=gains, state=start.cortex, rate=rate, write=False, backtrack=backtrack, keep_live=True
            )
            result.update(updated=taught.updated, step=taught.accepted_rate, replays=taught.replay_calls, reason=taught.reason)
            if taught.updated and taught.accepted_rate is not None:
                self._step_size = float(taught.accepted_rate)
            return SteeredPath(**result)
        # the joint step
        assert self.steering is not None and self.weighing is not None
        grad = self.cortex.observe(o, a, target, gains=gains, state=start.cortex, rate=0.0, write=False, keep_live=True)
        if grad.gain_gradient is None or grad.initial_loss is None:
            return SteeredPath(**result, reason="nonfinite_prediction")
        dy = self.weighing.pull(ys, gains, grad.gain_gradient, [start.weighing] + col["wstate"][:-1])
        seam = self.steering.observe(
            r, np.zeros((n, t, self.steering.actions)), output_gradient=dy, rate=0.0, write=False, state=start.steering, keep_live=True
        )
        delta_c, delta_s = grad.delta, seam.delta
        if not delta_s:
            return SteeredPath(**result, reason="nonfinite_prediction")
        before_c, before_s = self.cortex.parameters(), self.steering.parameters()
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = (sum(float(np.sum(v * v)) for v in delta_c.values()) if learn_cortex else 0.0) + self.rate_scale**2 * sum(float(np.sum(v * v)) for v in delta_s.values())
        objective0 = loss + price
        if not np.isfinite(norm_squared) or norm_squared <= 0:
            return SteeredPath(**result, reason="no_gradient")
        start_step = float(rate) if (self._step_size is None or not backtrack) else min(float(rate), 2.0 * self._step_size)
        replays = 0
        for attempt in range(_HALVINGS if backtrack else 1):
            step = start_step * 0.5**attempt
            with np.errstate(over="ignore", invalid="ignore"):
                proposed_c = {k: before_c[k] - (step * delta_c[k] if learn_cortex else 0.0) for k in before_c}
                proposed_s = {k: before_s[k] - step * self.rate_scale * delta_s[k] for k in before_s}
            if not (all(np.isfinite(v).all() for v in proposed_c.values()) and all(np.isfinite(v).all() for v in proposed_s.values())):
                continue
            self.cortex.set_parameters(proposed_c)
            self.steering.set_parameters(proposed_s)
            if not backtrack:
                self.cortex.updates += int(learn_cortex)
                self.steering.updates += 1
                self._step_size = step
                result.update(updated=learn_cortex, steering_updated=True, step=step, halvings=attempt, replays=0, reason="updated")
                return SteeredPath(**result)
            after = self._objective(o, a, target, r, gains, start)
            replays += 1
            self._replays += 1
            floor = 64 * np.finfo(float).eps * max(abs(objective0), abs(after or 0.0), np.finfo(float).tiny)
            if after is not None and after < objective0 - floor and after <= objective0 - 1e-4 * step * norm_squared:
                self.cortex.updates += int(learn_cortex)
                self.steering.updates += 1
                self._step_size = step
                result.update(updated=learn_cortex, steering_updated=True, step=step, halvings=attempt, replays=replays, reason="updated")
                return SteeredPath(**result)
            self.cortex.set_parameters(before_c)
            self.steering.set_parameters(before_s)
        result.update(replays=replays, reason="no_decreasing_parameter_step")
        return SteeredPath(**result)

    # ------------------------------------------------------------------ custody
    def snapshot(self) -> dict[str, np.ndarray]:
        out = {"cortex_" + k: v for k, v in self.cortex.snapshot().items()}
        if self.steering is not None:
            out.update({"steering_" + k: v for k, v in self.steering.snapshot().items()})
        meta = {
            "format": FORMAT,
            "weighing": None if self.weighing is None else self.weighing.to_dict(),
            "rule": self.rule is not None,
            "rule_macs": 0 if self.rule is None else self.rule.macs,
            "reads_output": self.reads_output,
            "evidence": list(self.evidence),
            "extra_channels": self.extra_channels,
            "rate_scale": self.rate_scale,
        }
        out["meta"] = np.array(json.dumps(meta, sort_keys=True))
        out["step_size"] = np.empty(0) if self._step_size is None else np.array([self._step_size])
        return out

    @classmethod
    def restore(
        cls,
        snapshot: Mapping[str, np.ndarray],
        *,
        rule: Callable[[np.ndarray], np.ndarray] | None = None,
        extra: Callable[..., np.ndarray] | None = None,
    ) -> Steered:
        """The composition from its snapshot. A rule and an extra-channel callable are code and
        come back as arguments; the cortex's implied reading is a declaration and is not part
        of a snapshot, so declare it again on the restored cortex."""
        try:
            meta = json.loads(str(snapshot["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError("unsupported steered checkpoint")
            cortex = BeliefPatch.restore({k[len("cortex_") :]: v for k, v in snapshot.items() if k.startswith("cortex_")})
            steering = None
            if any(k.startswith("steering_") for k in snapshot):
                steering = BeliefPatch.restore({k[len("steering_") :]: v for k, v in snapshot.items() if k.startswith("steering_")})
            weighing: Weighing | Rule | None = None
            if meta["rule"]:
                if rule is None:
                    raise ValueError("this checkpoint was taken under a rule; pass rule=")
                weighing = Rule(rule, macs=int(meta.get("rule_macs", 0)))
            elif meta["weighing"] is not None:
                weighing = weighing_from_dict(meta["weighing"])
            if meta["extra_channels"] and extra is None:
                raise ValueError("this checkpoint reads extra channels; pass extra=")
            result = cls(
                cortex,
                steering,
                weighing,
                reads_output=bool(meta["reads_output"]),
                evidence=meta["evidence"],
                extra=extra if meta["extra_channels"] else None,
                extra_channels=int(meta["extra_channels"]),
                rate_scale=float(meta["rate_scale"]),
            )
            kept = np.asarray(snapshot["step_size"], dtype=float).reshape(-1)
            result._step_size = None if kept.size == 0 else float(kept[0])
            return result
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("invalid steered checkpoint") from error

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(cls, path: str | Path, **kw: Any) -> Steered:
        with np.load(path, allow_pickle=False) as arrays:
            return cls.restore({k: arrays[k] for k in arrays.files}, **kw)
