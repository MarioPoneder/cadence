"""A life: one continuing loop in which a governor reads the brain's own signals
and returns its mode, habit, imagine or learn.

Rung 1 of the ladder. The brain is a ``BeliefPatch`` or a ``Steered`` cortex; the application
supplies its body and its cheap policy. Every decision: the life reads the brain's own signals into
a readback of seven channels (the last surprise over its baseline, log-compressed; a slow average of
the same; the repair residual over its routine median; the last mode as three flags; a constant).
The governor settles on that readback and names the mode. The action is the habit's or the best
candidate imagined over a horizon under the belief's private continuation. The moment is
assimilated, and the outcome the world returns is measured against what the brain expected: the
surprise. When the governor says learn, the executed window is replayed from its boundary and one
admitted step is taken per pass; a window whose loss did not fall is undone. The baseline of the
surprise follows the quiet moments, with a floor, and habituates when a learn call was undone.

The governor is a settling patch whose every synapse is a gene (``PatchGovernor``, the dozing cat's
fourteen neurons), with the hand-set threshold rule as the control (``ThresholdGovernor``) and the
two ends of the switch (``AlwaysAwake``, ``NeverWakes``). The cost of a decision is what the brain
computed: the moments of the patch (assimilated, imagined, observed, replayed) and the governor's
settling steps converted to moments by its synapse count.

The room with the heater and the dozing cat wrote this loop before it was here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .belief import BeliefPatch, _integer
from .brain import Brain, BrainState
from .connectome import Connectome
from .neuron import NeuronModel
from .steering import Boundary, Steered

MODES = ("habit", "imagine", "learn")
READBACK = ("fast", "slow", "residual", "mode_habit", "mode_imagine", "mode_learn", "one")
GOVERNOR_MODEL = NeuronModel(dt=0.5, slope=2.0, threshold=0.5, gain=1.0, stimulus_amplitude=1.0)


def _log_surprise(value: float, baseline: float) -> float:
    """Log-compress a nonnegative error without overflowing its ratio."""
    scale = max(baseline, np.finfo(float).tiny)
    if value <= scale:
        return float(np.log1p(value / scale))
    return float(np.logaddexp(0.0, np.log(value) - np.log(scale)))


@dataclass
class Signals:
    """What a governor may read: the seven-channel readback and the raw signals behind it."""

    readback: np.ndarray
    surprise: float
    baseline: float
    residual: float
    slow: float
    mode: str
    recent: list[float]
    can_learn: bool


class Governor:
    """``settle(signals) -> (mode, steps)``: the mode for this decision and the steps the
    settling took (its cost). ``after(signals)`` sees the moment's outcome. ``synapses`` prices
    the steps in the brain's moments."""

    synapses: int = 0

    def settle(self, signals: Signals) -> tuple[str, int]:
        raise NotImplementedError

    def after(self, signals: Signals) -> None:
        return None

    def reset(self) -> None:
        return None


class NeverWakes(Governor):
    """The habit every decision: the brain below rung 1."""

    def settle(self, signals: Signals) -> tuple[str, int]:
        return "habit", 0


class ThresholdGovernor(Governor):
    """The hand-designed control: a spike of the surprise above ``k_imagine`` baselines recruits
    imagination for ``imagine_budget`` decisions (renewed by a new spike if ``renew``); a
    learn call needs the surprise above ``k_learn`` baselines on at least ``persist_share`` of
    the last ``persist`` decisions and no cooldown. Every constant is a gene."""

    HAND_SET: dict[str, Any] = {
        "k_imagine": 8.0,
        "imagine_budget": 12.0,
        "k_learn": 8.0,
        "persist": 96.0,
        "persist_share": 0.25,
        "renew": True,
        "cooldown": 64.0,
    }
    SPACE: dict[str, tuple[Any, ...]] = {
        "k_imagine": ("log", 0.3, 1.5, 100.0),
        "imagine_budget": ("log", 0.3, 1.0, 40.0),
        "k_learn": ("log", 0.3, 1.2, 30.0),
        "persist": ("log", 0.3, 4.0, 200.0),
        "persist_share": ("linear", 0.1, 0.5, 1.0),
        "renew": ("choice", True, False),
        "cooldown": ("log", 0.3, 32.0, 512.0),
    }

    def __init__(self, genome: Mapping[str, Any] | None = None) -> None:
        self.g = {**self.HAND_SET, **(genome or {})}
        for name in ("k_imagine", "k_learn", "imagine_budget", "cooldown"):
            if not np.isfinite(self.g[name]) or self.g[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.g["persist"]) or round(self.g["persist"]) < 1:
            raise ValueError("persist must round to a positive number of decisions")
        if not np.isfinite(self.g["persist_share"]) or not 0 <= self.g["persist_share"] <= 1:
            raise ValueError("persist_share must lie in [0, 1]")
        self.imagine_left = 0
        self.cooldown_left = 0

    @property
    def persist(self) -> int:
        return int(round(float(self.g["persist"])))

    def settle(self, signals: Signals) -> tuple[str, int]:
        g = self.g
        recent = signals.recent[-self.persist :]
        share = (
            float(np.mean(np.array(recent) > g["k_learn"] * signals.baseline)) if recent else 0.0
        )
        if (
            len(recent) >= self.persist
            and share >= g["persist_share"]
            and signals.can_learn
            and self.cooldown_left == 0
        ):
            self.cooldown_left = int(round(float(g["cooldown"])))
            return "learn", 0
        if self.imagine_left > 0:
            self.imagine_left -= 1
            return "imagine", 0
        return "habit", 0

    def after(self, signals: Signals) -> None:
        g = self.g
        if self.cooldown_left:
            self.cooldown_left -= 1
        spike = signals.surprise > g["k_imagine"] * signals.baseline
        if spike and (self.imagine_left == 0 or bool(g["renew"])):
            self.imagine_left = max(self.imagine_left, int(round(float(g["imagine_budget"]))))

    def reset(self) -> None:
        self.imagine_left = 0
        self.cooldown_left = 0


class AlwaysAwake(ThresholdGovernor):
    """Imagination every decision; learning by the threshold rule."""

    def settle(self, signals: Signals) -> tuple[str, int]:
        mode, steps = super().settle(signals)
        return ("learn" if mode == "learn" else "imagine"), steps


class PatchGovernor(Governor):
    """A graph rate-neuron governor: a readback port of seven units,
    a cortex of ``cortex`` units, a motor of three. Every synapse is a gene (``rc_i_j``
    readback to cortex, ``rm_i_k`` readback to motor, ``cm_j_k`` cortex to motor, biases
    ``cb_j`` and ``mb_k``, one lateral weight within the cortex and one within the motor);
    the settled motor state is the mode; nothing in it learns within a life. ``warm`` starts
    each settle from the last (hysteresis as a gene). This graph solve is separate
    from the belief cortex's finite repair and does not form a joint equilibrium."""

    def __init__(
        self,
        genome: Mapping[str, Any] | None = None,
        *,
        cortex: int = 4,
        model: NeuronModel = GOVERNOR_MODEL,
        budget: int = 200,
        chunk: int = 10,
        tolerance: float = 1e-3,
    ) -> None:
        self.cortex_units = _integer("cortex", cortex, 1)
        self.g = {**self.hand_set(self.cortex_units), **(genome or {})}
        self.model = model
        self.budget, self.chunk = _integer("budget", budget, 0), _integer("chunk", chunk, 1)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        self.tolerance = float(tolerance)
        nr, nc, nm = len(READBACK), self.cortex_units, len(MODES)
        self.readback = np.arange(0, nr)
        self.cortex = np.arange(nr, nr + nc)
        self.motor = np.arange(nr + nc, nr + nc + nm)
        self.n = nr + nc + nm
        pre: list[int] = []
        post: list[int] = []
        w: list[float] = []
        g = self.g

        def synapse(a: int, b: int, value: float) -> None:
            if value != 0.0:
                pre.append(int(a))
                post.append(int(b))
                w.append(float(value))

        for i in range(nr):
            for j in range(nc):
                synapse(self.readback[i], self.cortex[j], g[f"rc_{i}_{j}"])
            for k in range(nm):
                synapse(self.readback[i], self.motor[k], g[f"rm_{i}_{k}"])
        for j in range(nc):
            for k in range(nm):
                synapse(self.cortex[j], self.motor[k], g[f"cm_{j}_{k}"])
            for j2 in range(nc):
                if j != j2:
                    synapse(self.cortex[j], self.cortex[j2], g["lateral_cortex"])
        for k in range(nm):
            for k2 in range(nm):
                if k != k2:
                    synapse(self.motor[k], self.motor[k2], g["lateral_motor"])
        self.connectome = Connectome.from_synapses(
            self.n,
            pre=pre,
            post=post,
            sign=w,
            populations={"readback": self.readback, "cortex": self.cortex, "motor": self.motor},
            label="governor",
        )
        bias = np.zeros(self.n)
        bias[self.cortex] = [float(g[f"cb_{j}"]) for j in range(nc)]
        bias[self.motor] = [float(g[f"mb_{k}"]) for k in range(nm)]
        self.brain = Brain(self.connectome, model, bias=bias)
        self.synapses = int(self.connectome.synapses)
        self.state: BrainState | None = None
        self.activation = np.zeros(self.n)
        self.converged = False
        self.residual: float | None = None

    @classmethod
    def hand_set(cls, cortex: int = 4) -> dict[str, Any]:
        """The threshold rule written as synapses: a spike drives the imagine unit, the mode
        feeds itself back, the slow average drives the learn unit, the habit unit rests on the
        constant, the motor units compete. The dozing cat's wiring."""
        g: dict[str, Any] = {"warm": False, "lateral_cortex": -0.5, "lateral_motor": -1.0}
        for i in range(len(READBACK)):
            for j in range(cortex):
                g[f"rc_{i}_{j}"] = 0.0
            for k in range(len(MODES)):
                g[f"rm_{i}_{k}"] = 0.0
        for j in range(cortex):
            g[f"cb_{j}"] = 0.0
            for k in range(len(MODES)):
                g[f"cm_{j}_{k}"] = 0.0
        for k in range(len(MODES)):
            g[f"mb_{k}"] = 0.0
        g["rm_0_1"] = 1.0  # fast surprise -> imagine
        g["rm_4_1"] = 0.4  # imagine -> imagine: the chase holds while the surprise stays up
        g["rm_6_0"] = 0.9  # one -> habit: the rest state
        g["rm_1_2"] = 1.5  # slow surprise -> learn
        g["rm_6_2"] = -0.9  # one -> learn: a bias against learning
        return g

    @classmethod
    def space(cls, cortex: int = 4) -> dict[str, tuple[Any, ...]]:
        """The genome's space for ``genes``: every synapse and bias linear in [-2.5, 2.5], the
        laterals, and ``warm``."""
        keys = [
            k for k in cls.hand_set(cortex) if k not in ("warm", "lateral_cortex", "lateral_motor")
        ]
        space: dict[str, tuple[Any, ...]] = {k: ("linear", 0.3, -2.5, 2.5) for k in keys}
        space["lateral_cortex"] = ("linear", 0.3, -2.5, 0.0)
        space["lateral_motor"] = ("linear", 0.3, -2.5, 0.0)
        space["warm"] = ("choice", True, False)
        return space

    def settle(self, signals: Signals) -> tuple[str, int]:
        drive = np.zeros((1, self.n))
        drive[0, self.readback] = signals.readback
        result = self.brain.equilibrate(
            drive,
            budget=self.budget,
            chunk=self.chunk,
            tolerance=self.tolerance,
            state=self.state if self.g.get("warm") else None,
        )
        self.converged = bool(result.converged.all())
        self.residual = float(np.max(result.residual))
        if not self.converged:
            # A capped numerical attempt is not a settled neural mode. The
            # declared fallback is the cheap habit, not its provisional argmax.
            return "habit", int(result.steps)
        self.state = result.state
        self.activation = np.asarray(result.state.activation[0]).copy()
        motor = self.activation[self.motor]
        mode = MODES[int(np.argmax(motor))] if motor.max() > 1e-6 else "habit"
        return mode, int(result.steps)

    def reset(self) -> None:
        self.state = None
        self.activation = np.zeros(self.n)
        self.converged = False
        self.residual = None


# ----------------------------------------------------------------------- the patch behind a life
class _Belief:
    def __init__(self, patch: BeliefPatch) -> None:
        self.patch = patch

    def boundary(self) -> Any:
        return self.patch.state

    def moment(self, r: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        path = self.patch.assimilate(r[None, None], action[None, None])
        return path.output[0, 0], float(path.residual[0, 0])

    def imagine_state(self, state: Any, n: int) -> np.ndarray:
        return (
            np.zeros((n, self.patch.belief))
            if state is None
            else np.repeat(np.asarray(state), n, axis=0)
        )

    def imagine(self, actions: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path = self.patch.imagine(actions[:, None, :], state=state)
        return path.output[:, 0], path.final_state

    def learn(
        self,
        o: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        boundary: Any,
        rate: float,
        write: bool,
        learn_cortex: bool = True,
    ) -> tuple[bool, float | None, float | None]:
        boundary = self.patch._boundary(
            len(o), np.zeros((len(o), self.patch.belief)) if boundary is None else boundary
        )
        result = self.patch.observe(
            o, a, y, rate=rate if learn_cortex else 0.0, write=write, state=boundary, keep_live=True
        )
        return result.updated, result.initial_loss, result.final_loss

    def loss(self, o: np.ndarray, a: np.ndarray, y: np.ndarray, boundary: Any) -> float | None:
        boundary = np.zeros((len(o), self.patch.belief)) if boundary is None else boundary
        return self.patch.observe(
            o, a, y, rate=0.0, write=False, state=boundary, keep_live=True
        ).initial_loss

    def checkpoint(self) -> dict[str, np.ndarray]:
        return self.patch.snapshot()

    def rollback(self, checkpoint: Mapping[str, np.ndarray]) -> None:
        # Keep the caller's patch, record store and declarations; retain compute
        # counters because rejected attempts still consumed that computation.
        restored = BeliefPatch.restore(checkpoint)
        self.patch.set_parameters(restored.parameters())
        self.patch.records.load_state(restored.records.state())
        self.patch._state = restored.state
        self.patch._input_norm = restored._input_norm
        self.patch._step_size = restored.step_size
        self.patch.updates = restored.updates

    def parameters(self) -> Any:
        return self.patch.parameters()

    def set_parameters(self, p: Any) -> None:
        self.patch.set_parameters(p)

    def set_boundary(self, boundary: Any) -> None:
        self.patch._state = None if boundary is None else np.asarray(boundary).copy()

    def macs_per_moment(self) -> int:
        return self.patch.macs_per_moment()

    @property
    def cost(self) -> dict[str, int]:
        return dict(self.patch.cost)

    def reset_cost(self) -> None:
        self.patch.reset_cost()


class _Steer:
    def __init__(self, patch: Steered) -> None:
        self.patch = patch

    def boundary(self) -> Boundary | None:
        return self.patch.boundary()

    def moment(self, r: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        path = self.patch.run(r[None, None], action[None, None])
        return path.output[0, 0], float(path.residual[0, 0])

    def imagine_state(self, state: Boundary | None, n: int) -> np.ndarray:
        belief = self.patch.cortex.belief
        return np.zeros((n, belief)) if state is None else np.repeat(state.cortex, n, axis=0)

    def imagine(self, actions: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        path = self.patch.cortex.imagine(actions[:, None, :], state=state)
        return path.output[:, 0], path.final_state

    def learn(
        self,
        o: np.ndarray,
        a: np.ndarray,
        y: np.ndarray,
        boundary: Any,
        rate: float,
        write: bool,
        learn_cortex: bool = True,
    ) -> tuple[bool, float | None, float | None]:
        boundary = self.patch._fresh(len(o)) if boundary is None else boundary
        before = self.patch.run(o, a, y, state=boundary, keep_live=True).objective
        path = self.patch.run(
            o, a, y, rate=rate, state=boundary, keep_live=True, learn_cortex=learn_cortex
        )
        after = (
            None
            if not (path.updated or path.steering_updated)
            else self.patch.run(o, a, y, state=boundary, keep_live=True).objective
        )
        return bool(path.updated or path.steering_updated), before, after

    def loss(self, o: np.ndarray, a: np.ndarray, y: np.ndarray, boundary: Any) -> float | None:
        boundary = self.patch._fresh(len(o)) if boundary is None else boundary
        return self.patch.run(o, a, y, state=boundary, keep_live=True).objective

    def checkpoint(self) -> dict[str, Any]:
        return {
            "cortex": _Belief(self.patch.cortex).checkpoint(),
            "steering": None
            if self.patch.steering is None
            else _Belief(self.patch.steering).checkpoint(),
            "boundary": deepcopy(self.patch.boundary()),
            "step": self.patch.step_size,
        }

    def rollback(self, checkpoint: Mapping[str, Any]) -> None:
        _Belief(self.patch.cortex).rollback(checkpoint["cortex"])
        if self.patch.steering is not None:
            _Belief(self.patch.steering).rollback(checkpoint["steering"])
        self.patch._live = deepcopy(checkpoint["boundary"])
        self.patch._step_size = checkpoint["step"]

    def parameters(self) -> Any:
        return self.patch.parameters()

    def set_parameters(self, p: Any) -> None:
        self.patch.set_parameters(p)

    def set_boundary(self, boundary: Any) -> None:
        self.patch._live = deepcopy(boundary)
        self.patch.cortex._state = None if boundary is None else boundary.cortex.copy()
        if self.patch.steering is not None:
            self.patch.steering._state = None if boundary is None else boundary.steering.copy()

    def macs_per_moment(self) -> int:
        return self.patch.macs_per_moment()

    @property
    def cost(self) -> dict[str, int]:
        return dict(self.patch.cost)

    def reset_cost(self) -> None:
        self.patch.reset_cost()


# ---------------------------------------------------------------------------- the life
@dataclass
class Decision:
    """One decision's record."""

    t: int
    mode: str
    action: np.ndarray
    expected: np.ndarray
    residual: float
    readback: np.ndarray
    governor_steps: int
    imagined: int
    learned: dict[str, Any] | None = None
    surprise: float | None = None
    baseline: float | None = None
    target: np.ndarray | None = None


@dataclass
class LifeConfig:
    """The life's constants; every one that looks designed is a candidate gene."""

    window: int = 96
    passes: int = 6
    learn_rate: float = 0.1
    rollback: bool = True
    validity: float = 0.9
    min_window: int = 32
    min_cooldown: int = 32
    cooldown: int = 0
    keep: int = 512
    horizon: int = 6
    hold: bool = True
    baseline_rate: float = 0.02
    floor: float = 1.0
    habituate: float = 2.0
    slow_rate: float = 0.02
    recent: int = 96
    write: bool = False
    learn_cortex: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("window", "passes", "min_window", "keep", "horizon", "recent"):
            _integer(name, getattr(self, name), 1)
        for name in ("min_cooldown", "cooldown"):
            _integer(name, getattr(self, name), 0)
        for name in ("learn_rate", "validity", "floor", "habituate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("baseline_rate", "slow_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


class Life:
    """The loop. The application gives the body's side: ``habit(reading) -> action``, the cheap
    policy; ``propose(reading) -> (candidates, actions)``, what imagination weighs;
    ``advance(reading, output) -> reading``, how a predicted change moves the reading;
    ``cost(reading, action) -> (candidates,)``, the task's price; ``target(reading, next) ->
    (outputs,)``, what the brain should have predicted. ``baseline`` and ``residual0`` are the
    routine surprise and residual (medians on a quiet stretch) the readback is scaled by.

    ``decide(reading)`` returns the action to execute; ``outcome(next_reading)`` closes the
    moment with what the world returned. ``records`` holds every decision, ``learns`` every
    learn call, ``compute()`` the cost."""

    def __init__(
        self,
        patch: BeliefPatch | Steered,
        governor: Governor,
        *,
        habit: Callable[[np.ndarray], np.ndarray],
        propose: Callable[[np.ndarray], np.ndarray],
        advance: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cost: Callable[[np.ndarray, np.ndarray], np.ndarray],
        target: Callable[[np.ndarray, np.ndarray], np.ndarray],
        baseline: float = 1.0,
        residual0: float = 1.0,
        config: LifeConfig | None = None,
        refit: Callable[[Life], dict[str, Any] | None] | None = None,
        keep_records: bool = True,
    ) -> None:
        if isinstance(patch, Steered):
            self.brain: _Belief | _Steer = _Steer(patch)
        elif isinstance(patch, BeliefPatch):
            self.brain = _Belief(patch)
        else:
            raise ValueError("a life runs on a BeliefPatch or a Steered cortex")
        self.patch = patch
        self.governor = governor
        self.habit, self.propose, self.advance, self.task_cost, self.target = (
            habit,
            propose,
            advance,
            cost,
            target,
        )
        self.c = config or LifeConfig()
        self.c.__post_init__()
        if isinstance(patch, Steered) and self.c.write:
            raise ValueError("a Steered life does not support record writes")
        if (
            not np.isfinite(baseline)
            or baseline <= 0
            or not np.isfinite(residual0)
            or residual0 <= 0
        ):
            raise ValueError("baseline and residual0 must be finite and positive")
        self.baseline0 = float(baseline)
        self.residual0 = float(residual0)
        self.floor = float(self.c.floor) * self.baseline0
        if not np.isfinite(self.floor):
            raise ValueError("the baseline floor must remain finite")
        self.baseline = max(self.baseline0, self.floor, np.finfo(float).tiny)
        self.refit = refit
        self.keep_records = keep_records
        self.t = 0
        self.mode = "habit"
        self.slow = 0.0
        self.surprise_last = 0.0
        self.residual_last = 0.0
        self.recent: list[float] = []
        self.cooldown_left = 0
        self.since_learn = 10**9
        self.o: list[np.ndarray] = []
        self.a: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.boundaries: list[Any] = []
        self.records: list[Decision] = []
        self.learns: list[dict[str, Any]] = []
        self.pending: Decision | None = None
        self.totals: dict[str, Any] = {
            "decisions": {m: 0 for m in MODES},
            "governor_steps": 0,
            "governor_moments": 0.0,
            "imagined": 0,
            "learn_calls": 0,
            "kept": 0,
            "undone": 0,
        }
        self.brain.reset_cost()

    # ------------------------------------------------------------------ the readback
    def readback(self) -> np.ndarray:
        one_hot = np.eye(len(MODES))[MODES.index(self.mode)]
        return np.array(
            [
                _log_surprise(self.surprise_last, self.baseline),
                self.slow,
                self.residual_last / self.residual0,
                *one_hot,
                1.0,
            ]
        )

    def signals(self) -> Signals:
        can_learn = (
            self.since_learn >= self.c.min_cooldown
            and len(self.a) - self._continuation_start() >= self.c.min_window
            and self.cooldown_left == 0
        )
        return Signals(
            self.readback(),
            self.surprise_last,
            self.baseline,
            self.residual_last,
            self.slow,
            self.mode,
            self.recent.copy(),
            can_learn,
        )

    # ------------------------------------------------------------------ the modes
    def _continuation_start(self) -> int:
        """History before a reset is retained, but is not the current causal path."""
        if self.brain.boundary() is None:
            return len(self.a)
        for index in range(len(self.boundaries) - 1, -1, -1):
            if self.boundaries[index] is None:
                return index
        # The first boundary of this continuation may have left the bounded history.
        return 0

    def _imagine(self, r: np.ndarray, boundary: Any) -> tuple[np.ndarray, int]:
        candidates = np.asarray(self.propose(r), dtype=float)
        patch = self.patch.cortex if isinstance(self.patch, Steered) else self.patch
        if (
            candidates.ndim != 2 or candidates.shape[1] != patch.actions
            or not len(candidates) or not np.isfinite(candidates).all()
        ):
            raise ValueError("propose must return finite (candidates, actions) with candidates > 0")
        n = len(candidates)
        state = self.brain.imagine_state(boundary, n)
        readings = np.repeat(r[None], n, axis=0)
        actions = candidates.copy()
        total = np.zeros(n)
        for _ in range(int(self.c.horizon)):
            outputs, state = self.brain.imagine(actions, state)
            readings = np.stack([self.advance(readings[i], outputs[i]) for i in range(n)])
            if readings.shape != (n, patch.inputs) or not np.isfinite(readings).all():
                raise ValueError("advance must return a finite (inputs,) predicted reading")
            cost = np.asarray(self.task_cost(readings, actions), dtype=float)
            if cost.shape != (n,) or not np.isfinite(cost).all():
                raise ValueError("cost must return a finite (candidates,) vector")
            with np.errstate(over="ignore", invalid="ignore"):
                total += cost
            if not np.isfinite(total).all():
                raise ValueError("the accumulated imagined cost must remain finite")
            if not self.c.hold:
                actions = np.stack([self.habit(readings[i]) for i in range(n)])
                if actions.shape != candidates.shape or not np.isfinite(actions).all():
                    raise ValueError("habit must return a finite (actions,) predicted action")
        return candidates[int(np.argmin(total))], n * int(self.c.horizon)

    def _learn(self) -> dict[str, Any]:
        c = self.c
        i0 = max(len(self.a) - int(c.window), self._continuation_start())
        w = len(self.a) - i0
        if w == 0:
            raise RuntimeError("learning needs completed moments in the current continuation")
        o = np.array(self.o[i0:])[None]
        a = np.array(self.a[i0:])[None]
        y = np.array(self.y[i0:])[None]
        boundary = self.boundaries[i0]
        checkpoint = self.brain.checkpoint()
        loss0, passes_kept = None, 0
        try:
            for _ in range(int(c.passes)):
                updated, initial, final = self.brain.learn(
                    o, a, y, boundary, float(c.learn_rate), bool(c.write), bool(c.learn_cortex)
                )
                if loss0 is None:
                    loss0 = initial
                if not updated:
                    break
                passes_kept += 1
            loss1 = self.brain.loss(o, a, y, boundary)
        except Exception:
            self.brain.rollback(checkpoint)
            raise
        valid = loss0 is not None and loss1 is not None and loss1 < c.validity * loss0
        entry: dict[str, Any] = {
            "t": self.t,
            "window": w,
            "loss_before": loss0,
            "loss_after": loss1,
            "valid": bool(valid),
            "kept": bool(valid or not c.rollback),
            "passes_kept": passes_kept,
            "refit": None,
        }
        if not valid and c.rollback:
            self.brain.rollback(checkpoint)
            self.totals["undone"] += 1
            habituated = min(
                self.baseline * float(c.habituate),
                float(np.median(self.recent)) if self.recent else self.baseline,
            )
            self.baseline = max(self.floor, habituated, np.finfo(float).tiny)
        else:
            self.totals["kept"] += 1
            if loss1 is not None:
                self.baseline = max(
                    self.floor, float(loss1) * 2.0, np.finfo(float).tiny
                )  # the window's mean squared error per moment
            if self.refit is not None:
                entry["refit"] = self.refit(self)
        self.totals["learn_calls"] += 1
        self.recent = []
        self.slow = 0.0
        self.cooldown_left = int(c.cooldown)
        self.since_learn = 0
        self.learns.append(entry)
        return entry

    # ------------------------------------------------------------------ one decision
    def decide(self, reading: np.ndarray) -> np.ndarray:
        """The action for this reading: the readback, the governor's mode, the action, the
        moment assimilated. ``outcome`` must follow with what the world returned."""
        if self.pending is not None:
            raise RuntimeError("outcome() must close the previous decision before the next")
        r = np.array(reading, dtype=float, copy=True)
        patch = self.patch.cortex if isinstance(self.patch, Steered) else self.patch
        if r.shape != (patch.inputs,) or not np.isfinite(r).all():
            raise ValueError("reading must be a finite (inputs,) array")
        want, steps = self.governor.settle(self.signals())
        if want not in MODES:
            raise ValueError("the governor must return habit, imagine or learn")
        steps = _integer("governor steps", steps, 0)
        can_learn = self.signals().can_learn
        learned = self._learn() if want == "learn" and can_learn else None
        imagined = 0
        if want in ("imagine", "learn"):
            action, imagined = self._imagine(r, self.brain.boundary())
            mode = "imagine"
        else:
            action, mode = np.asarray(self.habit(r), dtype=float), "habit"
        if learned is not None:
            mode = "learn"
        action = np.array(action, dtype=float, copy=True)
        if action.shape != (patch.actions,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite (actions,) array")
        boundary = deepcopy(self.brain.boundary())
        expected, residual = self.brain.moment(r, action)
        macs = max(1, self.brain.macs_per_moment())
        governor_moments = steps * self.governor.synapses / macs
        self.totals["governor_steps"] += steps
        self.totals["governor_moments"] += governor_moments
        self.totals["imagined"] += imagined
        self.totals["decisions"][mode] += 1
        self.pending = Decision(
            self.t,
            mode,
            action.copy(),
            expected.copy(),
            residual,
            self.readback(),
            steps,
            imagined,
            learned,
        )
        self.o.append(r.copy())
        self.a.append(action.copy())
        self.boundaries.append(boundary)
        return action.copy()

    def outcome(self, next_reading: np.ndarray) -> Decision:
        """Close the moment: the target from the world's next reading, the surprise against the
        expectation, the baseline and the slow average, the window, the governor's ``after``."""
        d = self.pending
        if d is None:
            raise RuntimeError("decide() comes before outcome()")
        c = self.c
        patch = self.patch.cortex if isinstance(self.patch, Steered) else self.patch
        next_value = np.array(next_reading, dtype=float, copy=True)
        if next_value.shape != (patch.inputs,) or not np.isfinite(next_value).all():
            raise ValueError("next_reading must be a finite (inputs,) array")
        y = np.array(self.target(self.o[-1].copy(), next_value), dtype=float, copy=True)
        if y.shape != (patch.outputs,) or not np.isfinite(y).all():
            raise ValueError("target must be a finite (outputs,) array")
        with np.errstate(over="ignore", invalid="ignore"):
            surprise = float(np.mean((d.expected - y) ** 2))
        if not np.isfinite(surprise):
            raise ValueError("outcome produced a nonfinite surprise")
        self.y.append(y.copy())
        if len(self.a) > int(c.keep):
            del self.o[0], self.a[0], self.y[0], self.boundaries[0]
        self.recent.append(surprise)
        del self.recent[: -int(c.recent)]
        if surprise <= 3.0 * self.baseline:
            self.baseline = max(
                self.floor, self.baseline + float(c.baseline_rate) * (surprise - self.baseline),
                np.finfo(float).tiny,
            )
        self.slow = self.slow + float(c.slow_rate) * (
            _log_surprise(surprise, self.baseline) - self.slow
        )
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        self.since_learn += 1
        self.surprise_last, self.residual_last, self.mode = surprise, d.residual, d.mode
        d.surprise, d.baseline, d.target = surprise, self.baseline, y.copy()
        self.governor.after(self.signals())
        if self.keep_records:
            self.records.append(deepcopy(d))
        self.pending = None
        self.t += 1
        return d

    def step(
        self, reading: np.ndarray, world: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[np.ndarray, Decision]:
        """One decision closed by the world: ``world(action)`` returns the next reading."""
        action = self.decide(reading)
        next_reading = np.asarray(world(action), dtype=float)
        return next_reading, self.outcome(next_reading)

    # ------------------------------------------------------------------ the cost
    def compute(self) -> dict[str, Any]:
        """Moments and multiply-accumulates the brain computed, the governor's steps as
        moments, per decision and by mode."""
        cost = self.brain.cost
        decisions = max(1, sum(self.totals["decisions"].values()))
        moments = cost["moments"] + self.totals["governor_moments"]
        return {
            **deepcopy(self.totals),
            "moments": cost["moments"],
            "macs": cost["macs"],
            "replays": cost["replays"],
            "moment_macs": self.brain.macs_per_moment(),
            "total_moments": float(moments),
            "moments_per_decision": float(moments / decisions),
        }

    def reset(self) -> None:
        """Forget live state and the governor's; parameters, history and counters stay.

        A later learning window uses only the new continuation, never a fabricated
        transition from the pre-reset history into the reset state.
        """
        self.patch.reset()
        self.governor.reset()
        if self.pending is not None:
            # The unexecuted/open moment has no target and cannot remain in a
            # completed learning window after cancellation.
            self.o.pop()
            self.a.pop()
            self.boundaries.pop()
        self.pending = None
