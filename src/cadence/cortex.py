"""General contextual prediction through jointly settled column banks.

``observe`` admits supplied targets immediately and ``predict`` reads qualified
means. The optional ``act`` / ``learn`` / ``flush`` interface supplies a discrete
reinforcement-learning workflow. Its targets, reward horizon and exploration
rule belong to that workflow, not to the cortical-column repair equation.

Each level has a reciprocal belief/precision observer loop. Coarse-to-fine prior
ports couple levels in one settlement; these directed ports do not themselves
constitute reciprocal observation between levels. Arbitrary custom feature maps
are application code and must be pure and deterministic.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import element as el
from ._validation import (
    boolean,
    canonical,
    context_key,
    integer,
    number,
    restore_key,
    strict_json,
)
from .column import _initial_rate, stage_ports
from .wiring import IdentityFeatures, features_from_specification, grid


class SettlementError(RuntimeError):
    """An operation requiring a qualified belief exhausted its repair budget."""


class LevelBelief:
    """Fuse one level's evidence with the adjacent coarse belief by precision."""

    def __init__(self, weight, mean, level, coupling, prior_dispersion=0.0):
        self.weight, self.mean, self.level = weight, mean, level
        self.coupling, self.prior_dispersion = coupling, prior_dispersion

    def emit(self, port, inbox):
        """Propose mean/variance from current precision and the optional prior."""
        own_precision = self.weight * inbox[f"feedback{self.level}"]
        prior = inbox.get(f"prior{self.level}")
        if prior is None or self.coupling == 0.0:
            return self.mean, 1.0 / own_precision
        prior_mean, prior_variance = prior
        prior_precision = self.coupling / (prior_variance + self.prior_dispersion)
        total = own_precision + prior_precision
        mean = (own_precision * self.mean + prior_precision * prior_mean) / total
        return mean, 1.0 / total


class LevelObserver(el.PrecisionObserver):
    """Read the level's live mean and uncertainty, including prior displacement."""

    def __init__(self, weight, scatter, level, mean, *, prior_shape, prior_rate):
        super().__init__(
            weight, scatter, prior_shape=prior_shape, prior_rate=prior_rate
        )
        self.level, self.mean = level, mean
        self.shape, self.rate = prior_shape, prior_rate

    def emit(self, port, inbox):
        """Propose precision from the expected squared evidence residual."""
        mean, variance = inbox[f"readback{self.level}"]
        rate = inbox.get(f"{self.level}:meta_feedback", self.rate)
        displacement = (mean - self.mean) ** 2
        return (self.shape + self.weight / 2.0) / (
            rate + (self.scatter + self.weight * (variance + displacement)) / 2.0
        )


@dataclass
class ValueColumn:
    """Private floating-point evidence for one level/context/output."""

    weight: float
    linear: float = 0.0
    square: float = 0.0
    version: int = 0

    def stats(self) -> tuple[float, float, float]:
        """Return the natural evidence parameters, without a mutable alias."""
        return self.weight, self.linear, self.square

    def prospective(self, target, decay, mass):
        """Prepare tempered evidence; the caller qualifies it before commit."""
        result = (
            el.EvidenceFactor(*self.stats())
            .temper(decay**mass)
            .combine(el.EvidenceFactor(mass, mass * target, mass * target * target))
        )
        return ValueColumn(
            result.weight, result.linear, result.square, self.version + 1
        )


@dataclass(frozen=True)
class _Transition:
    contexts: tuple
    action: int
    reward: float
    next_contexts: tuple | None
    terminal: bool
    truncated: bool
    event: int


class Cortex:
    """A bounded hierarchy of contextual scalar-output column banks.

    Parameters
    ----------
    feature_maps : sequence of callable or None
        Pure context functions, finest first. None uses the complete observation
        as one context. Use :func:`cadence.grid` for numeric generalization.
    n_outputs : int, default 1
        Output channels, default one. Channels can represent measured quantities,
        supplied targets, or discrete action values in the optional RL workflow.
    decay : float, default .99
        Evidence retention per unit witness mass, in (0,1].
    discount : float, default .97
        RL bootstrap discount in [0,1); unused by direct target observation.
    optimism : float, default .5
        Nonnegative uncertainty bonus for RL action selection and RL targets.
    epsilon : float, default .02
        Random-action probability in [0,1]; unused by prediction.
    coupling : float, default 1
        Nonnegative coarse-prior precision multiplier; zero disconnects priors.
    target_bound : float or None, default 8
        Symmetric RL-target clipping bound. Direct supplied targets are never
        clipped. None disables RL clipping; all statistics must remain finite.
    settle_budget : int, default 256
        Maximum sweeps, including zero for diagnostic rejection tests.
    seed : int, default 0
        Nonnegative seed for action exploration and tie breaking.
    learning_enabled : bool, default True
        False prevents new admissions and preserves queued experience.
    height : int, default 1
        Observer stages inside each column, not hierarchy depth.
    update_mode : {'episode', 'step'}, default 'episode'
        RL admission timing. Direct ``observe`` always admits immediately.
    tolerance : float, default 1e-11
        Positive residual threshold for settlement qualification.
    damping : float, default 1
        Scalar message update fraction in (0,1]. Qualification uses full proposals.
    prior_weight, prior_shape, prior_rate : float, default 1
        Positive zero-mean evidence weight and precision-observer prior parameters.
    meta_shape, meta_rate : float, default 4
        Positive hyper-observer parameters used above height one.
    level_weights : sequence of float or None
        Witness mass multipliers, one positive value per level. None uses declared
        context counts relative to the finest level, or ones for custom maps.
        Witness mass is a task-scale choice, not a free improvement: masses far
        below one let long episodes admit without one flush washing a coarse
        level's whole decay window, but on short episodes they slow coarse
        welding until learning looks dead (flat means, exhausted novelty).
        Compare episode length times mass against the 1/(1-decay) window, and
        pass explicit ones to disable mass weighting for short-episode tasks.
    max_columns : int, default 100000
        Maximum stored (level, context, output) entries. Reads do not allocate them.
    max_cache : int, default 4096
        Maximum cached readouts; zero disables caching.
    max_pending : int, default 10000
        Maximum buffered RL transitions before explicit admission is required.
    max_checkpoint_bytes : int, default 8388608
        UTF-8 size limit for checkpoint loading and production.
    wiring_id : str or None
        Required to checkpoint custom callables. The application owns this semantic
        version identifier and must change it when their behavior changes.
    self_observation : bool, default True
        With ``height > 1``, watch the cortex's own prediction error and gate
        every context by one of ``height`` self-selected modes: on a sustained
        error spike, re-score the modes against the latest evidence and switch
        to the best or recruit a fresh one. Height one is unaffected. False
        disables the mode machinery entirely. Mode selection runs in ``observe``;
        RL transitions keep the current mode.
    reflect_rate : float, default .2
        Fast surprise-trace rate in (0, 1]; the slow baseline uses a tenth of it.
    reflect_threshold : float, default 2
        Fast-to-slow surprise ratio above one that triggers mode re-evaluation.

    Numerical configuration is fixed at construction; ``learning_enabled`` and
    the diagnostic ``prior_ports_cut`` switch remain mutable bool properties.
    Checkpoints preserve evidence, pending transitions, event ownership and RNG.
    Derived caches are rebuilt, so subsequent cache-cost counters may differ.
    """

    _CONFIG_NAMES = frozenset(
        {
            "n_outputs",
            "feature_maps",
            "levels",
            "decay",
            "discount",
            "optimism",
            "epsilon",
            "coupling",
            "target_bound",
            "settle_budget",
            "seed",
            "height",
            "update_mode",
            "tolerance",
            "damping",
            "prior_weight",
            "prior_shape",
            "prior_rate",
            "meta_shape",
            "meta_rate",
            "level_weights",
            "max_columns",
            "max_cache",
            "max_pending",
            "max_checkpoint_bytes",
            "wiring_id",
            "self_observation",
            "reflect_rate",
            "reflect_threshold",
        }
    )

    def __setattr__(self, name, value):
        if name in self._CONFIG_NAMES and name in self.__dict__:
            raise AttributeError(
                f"{name} is fixed at construction; create a new Cortex"
            )
        super().__setattr__(name, value)

    def __init__(
        self,
        n_outputs: int = 1,
        feature_maps: Sequence[Callable[[Any], Any]] | None = None,
        *,
        decay: float = 0.99,
        discount: float = 0.97,
        optimism: float = 0.5,
        epsilon: float = 0.02,
        coupling: float = 1.0,
        target_bound: float | None = 8.0,
        settle_budget: int = 256,
        seed: int = 0,
        learning_enabled: bool = True,
        height: int = 1,
        update_mode: str = "episode",
        tolerance: float = 1e-11,
        damping: float = 1.0,
        prior_weight: float = 1.0,
        prior_shape: float = 1.0,
        prior_rate: float = 1.0,
        meta_shape: float = 4.0,
        meta_rate: float = 4.0,
        level_weights=None,
        max_columns: int = 100000,
        max_cache: int = 4096,
        max_pending: int = 10000,
        max_checkpoint_bytes: int = 8388608,
        wiring_id: str | None = None,
        self_observation: bool = True,
        reflect_rate: float = 0.2,
        reflect_threshold: float = 2.0,
    ):
        self.n_outputs = integer(n_outputs, "n_outputs", 1)
        maps = (IdentityFeatures(),) if feature_maps is None else tuple(feature_maps)
        if not maps or not all(callable(f) for f in maps):
            raise ValueError("feature_maps must be a nonempty sequence of callables")
        # Own built-in descriptors rather than borrow mutable encoder objects.
        self.feature_maps = tuple(
            features_from_specification(f.specification())
            if type(f).__module__ == "cadence.wiring" and hasattr(f, "specification")
            else f
            for f in maps
        )
        self.levels = len(maps)
        for name, value in (
            ("decay", decay),
            ("discount", discount),
            ("optimism", optimism),
            ("epsilon", epsilon),
            ("coupling", coupling),
            ("tolerance", tolerance),
            ("damping", damping),
            ("prior_weight", prior_weight),
            ("prior_shape", prior_shape),
            ("prior_rate", prior_rate),
            ("meta_shape", meta_shape),
            ("meta_rate", meta_rate),
        ):
            setattr(self, name, number(value, name))
        if not 0 < self.decay <= 1 or not 0 <= self.discount < 1:
            raise ValueError("decay must be in (0,1] and discount in [0,1)")
        if not 0 <= self.epsilon <= 1 or self.optimism < 0 or self.coupling < 0:
            raise ValueError(
                "epsilon must be in [0,1]; optimism and coupling nonnegative"
            )
        if self.tolerance <= 0 or not 0 < self.damping <= 1:
            raise ValueError("tolerance must be positive and damping in (0,1]")
        if (
            min(
                self.prior_weight,
                self.prior_shape,
                self.prior_rate,
                self.meta_shape,
                self.meta_rate,
            )
            <= 0
        ):
            raise ValueError("Prior and meta parameters must be positive")
        if self.prior_shape + (self.prior_weight - 1.0) / 2.0 <= 0:
            raise ValueError(
                "Priors must permit a positive finite precision equilibrium"
            )
        precision = self.prior_shape / self.prior_rate
        total_precision = self.prior_weight * precision
        if not all(math.isfinite(v) and v > 0 for v in (precision, total_precision)):
            raise ValueError("Prior precision exceeds the finite numeric range")
        if not math.isfinite(1.0 / total_precision):
            raise ValueError("Prior variance exceeds the finite numeric range")
        meta_precision = self.meta_shape / self.meta_rate
        if not math.isfinite(meta_precision) or meta_precision <= 0:
            raise ValueError("Meta precision exceeds the finite numeric range")
        self.target_bound = (
            None if target_bound is None else number(target_bound, "target_bound")
        )
        if self.target_bound is not None and self.target_bound <= 0:
            raise ValueError("target_bound must be positive or None")
        self.height = integer(height, "height", 1)
        if self.height > 1:
            _initial_rate(self.prior_rate, self.meta_shape, self.meta_rate)
        if self.height > 2:
            _initial_rate(self.meta_rate, self.meta_shape, self.meta_rate)
        self.settle_budget, self.seed = (
            integer(settle_budget, "settle_budget"),
            integer(seed, "seed"),
        )
        self.max_columns, self.max_cache = (
            integer(max_columns, "max_columns", 1),
            integer(max_cache, "max_cache"),
        )
        self.max_pending = integer(max_pending, "max_pending", 1)
        self.max_checkpoint_bytes = integer(
            max_checkpoint_bytes, "max_checkpoint_bytes", 1
        )
        if update_mode not in ("episode", "step"):
            raise ValueError("update_mode must be 'episode' or 'step'")
        self.update_mode = update_mode
        if wiring_id is not None and (
            not isinstance(wiring_id, str) or not wiring_id or len(wiring_id) > 1024
        ):
            raise ValueError(
                "wiring_id must be a nonempty string of at most 1024 characters"
            )
        self.wiring_id = wiring_id
        if level_weights is None:
            cells = [getattr(f, "cells", None) for f in maps]
            try:
                level_weights = (
                    [min(1.0, c / cells[0]) for c in cells]
                    if all(type(c) is int and c > 0 for c in cells)
                    else [1.0] * self.levels
                )
            except OverflowError as error:
                raise ValueError(
                    "Context counts overflow; supply explicit level_weights"
                ) from error
        self.level_weights = tuple(number(w, "level weight") for w in level_weights)
        if len(self.level_weights) != self.levels or any(
            w <= 0 for w in self.level_weights
        ):
            raise ValueError(
                "level_weights must contain one positive finite mass per level"
            )
        self.self_observation = boolean(self_observation, "self_observation")
        self.reflect_rate = number(reflect_rate, "reflect_rate")
        self.reflect_threshold = number(reflect_threshold, "reflect_threshold")
        if not 0 < self.reflect_rate <= 1 or self.reflect_threshold <= 1:
            raise ValueError("reflect_rate must be in (0,1] and reflect_threshold > 1")
        # Recursive self-observation: with height > 1 the column watches its own
        # prediction error and, on sustained surprise, re-evaluates which of
        # ``height`` internal modes explains the latest evidence best. The mode
        # is part of every context, so height changes answers, not only variance.
        self._reflect = {"mode": 0, "fast": 0.0, "slow": 0.0, "seen": 0, "switches": 0}
        self.learning_enabled = learning_enabled
        self.prior_ports_cut = False
        self._rng = random.Random(self.seed)
        self._columns = [{} for _ in maps]
        self._cache = {}
        self._episode = []
        self._cursor, self._last_record = 0, None
        self.calibration = None
        self.counters = {
            "updates": 0,
            "rejected_updates": 0,
            "settles": 0,
            "cache_hits": 0,
            "greedy_actions": 0,
            "random_actions": 0,
        }

    @property
    def learning_enabled(self) -> bool:
        """Whether new evidence may be admitted; disabling preserves pending data."""
        return self._learning_enabled

    @learning_enabled.setter
    def learning_enabled(self, value):
        self._learning_enabled = boolean(value, "learning_enabled")

    @property
    def prior_ports_cut(self) -> bool:
        """Diagnostic lesion: read only the finest level; never deletes memory."""
        return self._prior_ports_cut

    @prior_ports_cut.setter
    def prior_ports_cut(self, value):
        self._prior_ports_cut = boolean(value, "prior_ports_cut")

    @classmethod
    def from_dimensions(
        cls,
        n_inputs: int,
        n_outputs: int = 1,
        *,
        bounds=(-1.0, 1.0),
        bins=8,
        depth: int = 1,
        **options,
    ) -> Cortex:
        """Create numeric grid contexts from input/output sizes and unit ranges.

        A single ``(low,high)`` pair applies to every input; a sequence of pairs
        declares each input's units. Depth counts levels above the fine grid.
        Remaining options are passed to the Cortex constructor.
        """
        n_inputs = integer(n_inputs, "n_inputs", 1)
        bounds = tuple(bounds)
        if len(bounds) == 2 and all(isinstance(v, (int, float)) for v in bounds):
            bounds = (bounds,) * n_inputs
        if len(bounds) != n_inputs:
            raise ValueError("Provide one bounds pair per input")
        return cls(
            n_outputs=n_outputs,
            feature_maps=grid(bounds, bins=bins, depth=depth),
            **options,
        )

    @classmethod
    def for_environment(
        cls,
        env_factory,
        n_actions=None,
        *,
        depth=2,
        width=1,
        calibration_steps=400,
        seed=0,
        **options,
    ) -> Cortex:
        """Calibrate a finite numeric-vector environment; no Gym import required.

        Creates and closes temporary environments. Supply custom feature maps
        directly for images or structured observations; automatic calibration
        requires a numeric vector and a finite discrete action count.
        """
        from .wiring import calibrate, wire

        if not callable(env_factory):
            raise ValueError("env_factory must be callable")
        if n_actions is None:
            env = env_factory()
            try:
                n_actions = getattr(getattr(env, "action_space", None), "n", None)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            if n_actions is None:
                raise ValueError("Pass n_actions when action_space.n is unavailable")
        calibration = calibrate(
            env_factory, n_actions, steps=calibration_steps, seed=seed
        )
        cortex = cls(
            n_actions, wire(calibration, depth=depth, width=width), seed=seed, **options
        )
        cortex.calibration = calibration
        return cortex

    @property
    def reflective(self) -> bool:
        """Whether recursive self-observation drives contexts (needs height > 1)."""
        return self.self_observation and self.height > 1

    @property
    def mode(self) -> int:
        """The self-selected hypothesis mode currently gating contexts."""
        return self._reflect["mode"]

    def _contexts(self, observation, mode=None):
        if not self.reflective:
            return tuple(context_key(f(observation)) for f in self.feature_maps)
        mode = self._reflect["mode"] if mode is None else mode
        return tuple(
            context_key(["mode", mode, f(observation)]) for f in self.feature_maps
        )

    def _mode_error(self, observation, values, mode):
        contexts = self._contexts(observation, mode)
        return sum(
            abs(self._value(contexts, k)["mean"] - v) for k, v in values.items()
        ) / len(values)

    def _reflect_step(self, observation, values):
        """Observe own surprise; on a sustained spike, retrospectively pick a mode."""
        r = self._reflect
        error = self._mode_error(observation, values, r["mode"])
        a = self.reflect_rate
        r["fast"] = (1 - a) * r["fast"] + a * error
        r["slow"] = (1 - a / 10) * r["slow"] + (a / 10) * error
        r["seen"] += 1
        if r["seen"] > 10 and r["fast"] > self.reflect_threshold * r["slow"] + 1e-12:
            errors = [
                (self._mode_error(observation, values, m), m)
                for m in range(self.height)
            ]
            best = min(errors)[1]
            if best == r["mode"]:
                # Nobody explains it yet: recruit the least-used fresh mode.
                best = (r["mode"] + 1) % self.height
            if best != r["mode"]:
                r["mode"], r["switches"] = best, r["switches"] + 1
            r["fast"] = r["slow"]
        return self._contexts(observation)

    def _output(self, output):
        output = integer(output, "output")
        if output >= self.n_outputs:
            raise ValueError("Output outside the declared channel count")
        return output

    def _chain(self, contexts, output):
        return [
            table.get((key, output), ValueColumn(self.prior_weight))
            for table, key in zip(self._columns, contexts, strict=True)
        ]

    def _chain_ports(self, stats_list):
        beliefs, ports = [], []
        precision = self.prior_shape / self.prior_rate
        for level, statistics in enumerate(stats_list):
            weight, mean, scatter = el._stats(*statistics)
            dispersion = 0.0
            if level + 1 < len(stats_list):
                upper_weight, _, upper_scatter = el._stats(*stats_list[level + 1])
                dispersion = upper_scatter / upper_weight
            belief = LevelBelief(weight, mean, level, self.coupling, dispersion)
            observer = LevelObserver(
                weight,
                scatter,
                level,
                mean,
                prior_shape=self.prior_shape,
                prior_rate=self.prior_rate,
            )
            beliefs.append((belief, observer, weight, mean))
        for level in reversed(range(len(beliefs))):
            belief, observer, weight, mean = beliefs[level]
            if level + 1 < len(beliefs):
                upper, _, up_weight, up_mean = beliefs[level + 1]
                ports.append(
                    el.Port(
                        f"prior{level}",
                        upper,
                        belief,
                        "moments",
                        (up_mean, 1.0 / (up_weight * precision)),
                    )
                )
            ports.extend(
                (
                    el.Port(
                        f"readback{level}",
                        belief,
                        observer,
                        "moments",
                        (mean, 1.0 / (weight * precision)),
                    ),
                    el.Port(f"feedback{level}", observer, belief, "scalar", precision),
                )
            )
            if self.height > 1:
                ports.extend(
                    stage_ports(
                        observer,
                        self.height,
                        precision,
                        prefix=f"{level}:",
                        prior_rate=self.prior_rate,
                        meta_shape=self.meta_shape,
                        meta_rate=self.meta_rate,
                    )
                )
        return ports

    def _settle(self, columns):
        result = el.settle(
            self._chain_ports([c.stats() for c in columns]),
            budget=self.settle_budget,
            tolerance=self.tolerance,
            damping=self.damping,
        )
        self.counters["settles"] += 1
        return result

    def _value(self, contexts, output):
        chain = self._chain(contexts, output)
        if self.prior_ports_cut:
            chain = chain[:1]
        key = (
            output,
            self.prior_ports_cut,
            contexts[: len(chain)],
            tuple(c.version for c in chain),
        )
        if key in self._cache:
            self.counters["cache_hits"] += 1
            return dict(self._cache[key])
        result = self._settle(chain)
        mean, variance = result["messages"]["readback0"]
        novelty = sum(
            1.0 / (c.weight * result["messages"][f"feedback{i}"])
            for i, c in enumerate(chain)
        )
        outcome = {
            "mean": mean,
            "variance": variance,
            "novelty": novelty,
            "qualified": result["converged"],
            "sweeps": result["sweeps"],
            "residual": result["full_residual"],
        }
        if outcome["qualified"] and self.max_cache:
            if len(self._cache) >= self.max_cache:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = dict(outcome)
        return outcome

    def value(self, observation: Any, output: int = 0) -> dict:
        """Return one output's mean, variance and qualification diagnostics.

        This diagnostic method can return ``qualified=False``. Use ``predict``
        when consuming a mean: it raises instead of using an unsettled result.
        """
        return self._value(self._contexts(observation), self._output(output))

    def query(self, observation: Any) -> tuple[dict, ...]:
        """Return diagnostics for every channel without changing retained evidence."""
        contexts = self._contexts(observation)
        return tuple(self._value(contexts, output) for output in range(self.n_outputs))

    @staticmethod
    def _qualified(belief):
        if belief["qualified"] is not True or not all(
            math.isfinite(belief[k])
            for k in ("mean", "variance", "novelty", "residual")
        ):
            raise SettlementError(
                "Belief did not qualify; increase the budget or revise the model"
            )
        return belief

    def predict(self, observation: Any) -> tuple[float, ...]:
        """Return all qualified means; raise SettlementError if any is unsettled."""
        return tuple(
            self._qualified(belief)["mean"] for belief in self.query(observation)
        )

    def act(self, observation: Any) -> int:
        """Choose a discrete output using qualified value/novelty or epsilon exploration."""
        contexts = self._contexts(observation)
        beliefs = [
            self._qualified(self._value(contexts, output))
            for output in range(self.n_outputs)
        ]
        if self._rng.random() < self.epsilon:
            self.counters["random_actions"] += 1
            return self._rng.randrange(self.n_outputs)
        order = list(range(self.n_outputs))
        self._rng.shuffle(order)
        action = max(
            order,
            key=lambda i: (
                beliefs[i]["mean"] + self.optimism * math.sqrt(beliefs[i]["novelty"])
            ),
        )
        self.counters["greedy_actions"] += 1
        return action

    def _event(self, event, record):
        event = self._cursor + 1 if event is None else integer(event, "event", 1)
        encoded = canonical(record)
        if event == self._cursor:
            if encoded == self._last_record:
                return event, encoded, True
            raise ValueError("Conflicting retry of the latest event")
        if event != self._cursor + 1:
            raise ValueError(f"Expected event {self._cursor + 1}")
        return event, encoded, False

    def _admit(self, contexts, targets, mass=1.0):
        planned = []
        missing = sum(
            (context, output) not in self._columns[level]
            for output in targets
            for level, context in enumerate(contexts)
        )
        if sum(map(len, self._columns)) + missing > self.max_columns:
            raise ValueError("max_columns exhausted; existing evidence is unchanged")
        for output, target in targets.items():
            chain = self._chain(contexts, output)
            prospective = [
                c.prospective(target, self.decay, mass * weight)
                for c, weight in zip(chain, self.level_weights, strict=True)
            ]
            # _stats rejects non-finite or impossible moments before settlement.
            result = self._settle(prospective)
            if result["converged"] is not True:
                self.counters["rejected_updates"] += 1
                return 0
            planned.append((output, prospective))
        for output, prospective in planned:
            for level, (context, column) in enumerate(
                zip(contexts, prospective, strict=True)
            ):
                self._columns[level][(context, output)] = column
        admitted = len(planned) * self.levels
        self.counters["updates"] += admitted
        self._cache.clear()
        return admitted

    def observe(
        self,
        observation: Any,
        targets,
        *,
        event: int | None = None,
        weight: float = 1.0,
    ) -> dict:
        """Atomically admit supplied targets across all affected outputs and levels.

        Targets may be one number for a one-output cortex, a full sequence, or
        a nonempty mapping of output indices to numbers. No RL bonus, clipping
        or bootstrap is applied. ``weight`` is positive evidence mass. Explicit
        consecutive event IDs support latest-event idempotency; omitted IDs
        allocate the next ID and cannot identify application-level retries.
        Flush pending RL experience before mixing in direct observations.
        """
        weight = number(weight, "weight")
        if weight <= 0:
            raise ValueError("weight must be positive")
        if isinstance(targets, Mapping):
            values = {self._output(k): number(v, "target") for k, v in targets.items()}
        elif self.n_outputs == 1 and not isinstance(targets, (Sequence, Mapping)):
            values = {0: number(targets, "target")}
        else:
            try:
                sequence = tuple(targets)
            except TypeError as error:
                raise ValueError(
                    "Provide a target sequence or output mapping"
                ) from error
            if len(sequence) != self.n_outputs:
                raise ValueError("Target count must match n_outputs")
            values = {i: number(v, "target") for i, v in enumerate(sequence)}
        if not values:
            raise ValueError("At least one target is required")
        contexts = self._contexts(observation)
        if self.reflective and self.learning_enabled and not self._episode:
            saved = dict(self._reflect)
            contexts = self._reflect_step(observation, values)
        record = {
            "kind": "targets",
            "contexts": contexts,
            "targets": sorted(values.items()),
            "weight": weight,
        }
        try:
            event, encoded, duplicate = self._event(event, record)
        except ValueError:
            if self.reflective and self.learning_enabled and not self._episode:
                self._reflect = saved
            raise
        if duplicate:
            if self.reflective and self.learning_enabled and not self._episode:
                self._reflect = saved
            return {
                "accepted": False,
                "qualified": True,
                "duplicate": True,
                "admitted": 0,
            }
        if self._episode:
            raise ValueError("Flush pending transitions before direct observation")
        if not self.learning_enabled:
            return {
                "accepted": False,
                "qualified": False,
                "duplicate": False,
                "admitted": 0,
            }
        admitted = self._admit(contexts, values, weight)
        if admitted:
            self._cursor, self._last_record = event, encoded
        return {
            "accepted": bool(admitted),
            "qualified": bool(admitted),
            "duplicate": False,
            "admitted": admitted,
        }

    def learn(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        terminal: bool = False,
        *,
        truncated: bool = False,
        event: int | None = None,
    ) -> dict:
        """Own a transition and optionally flush it using a supplied TD rule.

        True terminals have no bootstrap and may pass None as next_observation.
        Truncation flushes while preserving the next-state bootstrap. Contexts
        are copied at intake, so reusing an environment observation buffer is safe.
        A failed flush retains the unadmitted transitions for an explicit retry.
        """
        action, reward = self._output(action), number(reward, "reward")
        terminal, truncated = (
            boolean(terminal, "terminal"),
            boolean(truncated, "truncated"),
        )
        contexts = self._contexts(observation)
        next_contexts = None if terminal else self._contexts(next_observation)
        record = {
            "kind": "transition",
            "contexts": contexts,
            "action": action,
            "reward": reward,
            "next_contexts": next_contexts,
            "terminal": terminal,
            "truncated": truncated,
        }
        event, encoded, duplicate = self._event(event, record)
        if duplicate or not self.learning_enabled:
            return {
                "buffered": len(self._episode),
                "admitted": 0,
                "duplicate": duplicate,
            }
        if len(self._episode) >= self.max_pending:
            raise ValueError(
                "max_pending exhausted; call flush before adding experience"
            )
        if self._episode and (
            self._episode[-1].terminal
            or self._episode[-1].truncated
            or self.update_mode == "step"
        ):
            raise ValueError(
                "Retry flush or clear_pending before beginning another transition"
            )
        self._episode.append(
            _Transition(
                contexts, action, reward, next_contexts, terminal, truncated, event
            )
        )
        self._cursor, self._last_record = event, encoded
        admitted = (
            self.flush() if terminal or truncated or self.update_mode == "step" else 0
        )
        return {
            "buffered": len(self._episode),
            "admitted": admitted,
            "duplicate": False,
        }

    def flush(self) -> int:
        """Admit queued transitions backward, atomically per transition.

        On failure, earlier completed transitions remain committed, and the
        failing transition plus all remaining ones stay queued. Frozen learning
        leaves the queue untouched. The returned count is column admissions.
        """
        if not self.learning_enabled:
            return 0
        admitted = 0
        while self._episode:
            transition = self._episode[-1]
            bootstrap = 0.0
            if not transition.terminal:
                bootstrap = max(
                    self._qualified(self._value(transition.next_contexts, output))[
                        "mean"
                    ]
                    for output in range(self.n_outputs)
                )
            novelty = self._qualified(
                self._value(transition.contexts, transition.action)
            )["novelty"]
            bonus = self.optimism * (1.0 - self.discount) * math.sqrt(novelty)
            target = number(
                transition.reward + bonus + self.discount * bootstrap, "TD target"
            )
            if self.target_bound is not None:
                target = max(-self.target_bound, min(self.target_bound, target))
            count = self._admit(transition.contexts, {transition.action: target})
            if not count:
                raise SettlementError(
                    "Transition did not qualify; pending experience is retained"
                )
            admitted += count
            self._episode.pop()
        return admitted

    def clear_pending(self) -> int:
        """Explicitly discard queued transitions, retaining their consumed event IDs."""
        count = len(self._episode)
        self._episode.clear()
        return count

    def _configuration(self):
        names = self._CONFIG_NAMES - {"feature_maps", "levels", "wiring_id"}
        return {name: getattr(self, name) for name in sorted(names)}

    def _wiring(self):
        specs = []
        for encoder in self.feature_maps:
            if type(encoder).__module__ == "cadence.wiring" and hasattr(
                encoder, "specification"
            ):
                specs.append(encoder.specification())
            else:
                if self.wiring_id is None:
                    raise ValueError(
                        "Checkpointing custom feature maps requires an explicit wiring_id"
                    )
                specs.append({"type": "custom"})
        return {"id": self.wiring_id, "features": specs}

    def _restored_contexts(self, values):
        if not isinstance(values, list) or len(values) != self.levels:
            raise ValueError("Checkpoint contexts do not match the hierarchy")
        return tuple(restore_key(value) for value in values)

    def _restored_transition(self, record, event):
        fields = {
            "kind",
            "contexts",
            "action",
            "reward",
            "next_contexts",
            "terminal",
            "truncated",
        }
        if (
            type(record) is not dict
            or set(record) != fields
            or record["kind"] != "transition"
        ):
            raise ValueError("Malformed transition record")
        terminal = boolean(record["terminal"], "terminal")
        truncated = boolean(record["truncated"], "truncated")
        contexts = self._restored_contexts(record["contexts"])
        next_contexts = record["next_contexts"]
        if terminal:
            if next_contexts is not None:
                raise ValueError("Terminal transition has next contexts")
        else:
            next_contexts = self._restored_contexts(next_contexts)
        return _Transition(
            contexts,
            self._output(record["action"]),
            number(record["reward"], "reward"),
            next_contexts,
            terminal,
            truncated,
            event,
        )

    def _validate_record(self, record, event):
        if type(record) is not dict:
            raise ValueError("Malformed latest event")
        if record.get("kind") == "transition":
            self._restored_transition(record, event)
            return
        if (
            set(record) != {"kind", "contexts", "targets", "weight"}
            or record["kind"] != "targets"
        ):
            raise ValueError("Malformed target record")
        self._restored_contexts(record["contexts"])
        if number(record["weight"], "weight") <= 0:
            raise ValueError("Recorded mass must be positive")
        rows = record["targets"]
        if not isinstance(rows, list) or not rows or len(rows) > self.n_outputs:
            raise ValueError("Malformed recorded targets")
        previous = -1
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("Malformed recorded target")
            output = self._output(row[0])
            if output <= previous:
                raise ValueError("Recorded target indices are not ordered and distinct")
            number(row[1], "target")
            previous = output

    def snapshot(self) -> str:
        """Serialize continuation without executable functions or derived caches."""
        rows = [
            [level, context, output, *column.stats(), column.version]
            for level, table in enumerate(self._columns)
            for (context, output), column in table.items()
        ]
        rows.sort(key=lambda row: canonical(row[:3]))
        pending = [
            [
                t.contexts,
                t.action,
                t.reward,
                t.next_contexts,
                t.terminal,
                t.truncated,
                t.event,
            ]
            for t in self._episode
        ]
        text = canonical(
            {
                "schema": "cortex-state/1",
                "config": self._configuration(),
                "wiring": self._wiring(),
                "rows": rows,
                "rng": self._rng.getstate(),
                "pending": pending,
                "cursor": self._cursor,
                "last_record": self._last_record,
                "learning_enabled": self.learning_enabled,
                "prior_ports_cut": self.prior_ports_cut,
                "counters": self.counters,
                **({"reflection": dict(self._reflect)} if self.reflective else {}),
            }
        )
        if len(text.encode("utf-8")) > self.max_checkpoint_bytes:
            raise ValueError("Checkpoint exceeds max_checkpoint_bytes")
        return text

    def restore(self, text: str) -> None:
        """Atomically restore a compatible full continuation checkpoint.

        Arbitrary functions are never loaded. Configuration and encoder identity
        must match. Impossible/non-finite evidence, duplicate rows and malformed
        pending experience are rejected before any state is replaced.
        """
        state = strict_json(text, self.max_checkpoint_bytes)
        fields = {
            "schema",
            "config",
            "wiring",
            "rows",
            "rng",
            "pending",
            "cursor",
            "last_record",
            "learning_enabled",
            "prior_ports_cut",
            "counters",
        }
        if self.reflective:
            fields = fields | {"reflection"}
        if (
            type(state) is not dict
            or set(state) != fields
            or state["schema"] != "cortex-state/1"
        ):
            raise ValueError("Unknown or malformed Cortex checkpoint")
        if canonical(state["config"]) != canonical(self._configuration()) or canonical(
            state["wiring"]
        ) != canonical(self._wiring()):
            raise ValueError(
                "Checkpoint configuration or wiring does not match this Cortex"
            )
        columns = [{} for _ in range(self.levels)]
        if not isinstance(state["rows"], list) or len(state["rows"]) > self.max_columns:
            raise ValueError("Invalid checkpoint column count")
        for row in state["rows"]:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError("Invalid checkpoint row")
            level, key, output, weight, linear, square, version = row
            level, output = integer(level, "level"), self._output(output)
            if level >= self.levels:
                raise ValueError("Checkpoint level outside hierarchy")
            context = restore_key(key)
            if (context, output) in columns[level]:
                raise ValueError("Duplicate checkpoint column")
            statistics = tuple(number(v, "statistic") for v in (weight, linear, square))
            el._stats(*statistics)
            columns[level][(context, output)] = ValueColumn(
                *statistics, integer(version, "version")
            )
        cursor = integer(state["cursor"], "cursor")
        last_record = state["last_record"]
        if (
            (cursor == 0) != (last_record is None)
            or last_record is not None
            and not isinstance(last_record, str)
        ):
            raise ValueError("Checkpoint cursor and latest record disagree")
        if last_record is not None:
            last = strict_json(last_record, self.max_checkpoint_bytes)
            if type(last) is not dict or canonical(last) != last_record:
                raise ValueError("Malformed latest-event record")
            self._validate_record(last, cursor)
        pending = []
        if (
            not isinstance(state["pending"], list)
            or len(state["pending"]) > self.max_pending
        ):
            raise ValueError("Invalid pending transition count")
        last_event = 0
        for row in state["pending"]:
            if not isinstance(row, list) or len(row) != 7:
                raise ValueError("Malformed pending transition")
            contexts, action, reward, next_contexts, terminal, truncated, event = row
            event = integer(event, "event", 1)
            if (
                not last_event < event <= cursor
                or last_event
                and event != last_event + 1
            ):
                raise ValueError("Pending events are not ordered within cursor")
            if pending and (pending[-1].terminal or pending[-1].truncated):
                raise ValueError("Pending transitions cross an episode boundary")
            last_event = event
            record = {
                "kind": "transition",
                "contexts": contexts,
                "action": action,
                "reward": reward,
                "next_contexts": next_contexts,
                "terminal": terminal,
                "truncated": truncated,
            }
            pending.append(self._restored_transition(record, event))
            if event == cursor and canonical(record) != last_record:
                raise ValueError(
                    "Pending transition conflicts with the latest event record"
                )
        if self.update_mode == "step" and len(pending) > 1:
            raise ValueError("Step mode cannot have multiple pending transitions")
        rng_state = state["rng"]
        if (
            not isinstance(rng_state, list)
            or len(rng_state) != 3
            or type(rng_state[0]) is not int
            or not isinstance(rng_state[1], list)
            or any(type(v) is not int for v in rng_state[1])
        ):
            raise ValueError("Malformed RNG state")
        if rng_state[2] is not None:
            number(rng_state[2], "RNG Gaussian cache")
        rng = random.Random()
        try:
            rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid RNG state") from error
        counters = state["counters"]
        if type(counters) is not dict or set(counters) != set(self.counters):
            raise ValueError("Malformed counters")
        counters = {key: integer(value, key) for key, value in counters.items()}
        if (
            sum(column.version for table in columns for column in table.values())
            != counters["updates"]
        ):
            raise ValueError("Column versions and update count disagree")
        if cursor == 0 and any(columns):
            raise ValueError("A cortex without events cannot contain retained evidence")
        enabled = boolean(state["learning_enabled"], "learning_enabled")
        cut = boolean(state["prior_ports_cut"], "prior_ports_cut")
        reflect = dict(self._reflect)
        if self.reflective:
            data = state["reflection"]
            if type(data) is not dict or set(data) != set(reflect):
                raise ValueError("Malformed reflection state")
            reflect = {
                "mode": integer(data["mode"], "mode"),
                "fast": number(data["fast"], "fast"),
                "slow": number(data["slow"], "slow"),
                "seen": integer(data["seen"], "seen"),
                "switches": integer(data["switches"], "switches"),
            }
            if reflect["mode"] >= self.height:
                raise ValueError("Reflection mode outside height")
        self._columns, self._episode, self._rng = columns, pending, rng
        self._cursor, self._last_record = cursor, last_record
        self.counters, self.learning_enabled, self.prior_ports_cut = (
            counters,
            enabled,
            cut,
        )
        self._reflect = reflect
        self._cache.clear()

    @classmethod
    def from_snapshot(
        cls,
        text: str,
        *,
        feature_maps=None,
        wiring_id=None,
        max_checkpoint_bytes=8388608,
    ) -> Cortex:
        """Reconstruct built-in wiring and configuration from a checkpoint.

        Custom callables require feature_maps and their original wiring_id.
        max_checkpoint_bytes bounds parsing before the stored config is trusted.
        """
        limit = integer(max_checkpoint_bytes, "max_checkpoint_bytes", 1)
        state = strict_json(text, limit)
        if (
            type(state) is not dict
            or state.get("schema") != "cortex-state/1"
            or type(state.get("config")) is not dict
            or type(state.get("wiring")) is not dict
        ):
            raise ValueError("Malformed Cortex checkpoint")
        wiring = state["wiring"]
        if set(wiring) != {"id", "features"} or not isinstance(
            wiring["features"], list
        ):
            raise ValueError("Malformed checkpoint wiring")
        if feature_maps is None:
            feature_maps = tuple(
                features_from_specification(spec) for spec in wiring["features"]
            )
            wiring_id = wiring["id"] if wiring_id is None else wiring_id
        if (
            integer(
                state["config"].get("max_checkpoint_bytes", limit),
                "max_checkpoint_bytes",
                1,
            )
            > limit
        ):
            raise ValueError("Stored checkpoint limit exceeds caller's parsing bound")
        try:
            cortex = cls(
                feature_maps=feature_maps, wiring_id=wiring_id, **state["config"]
            )
        except TypeError as error:
            raise ValueError("Malformed constructor configuration") from error
        cortex.restore(text)
        return cortex

    def stats(self) -> dict:
        """Return counters and resource sizes without exposing mutable internals."""
        return {
            **self.counters,
            "columns_per_level": [len(table) for table in self._columns],
            "level_weights": list(self.level_weights),
            "levels": self.levels,
            "height": self.height,
            "n_outputs": self.n_outputs,
            "pending": len(self._episode),
            "cursor": self._cursor,
            "cache_entries": len(self._cache),
            **(
                {
                    "mode": self._reflect["mode"],
                    "mode_switches": self._reflect["switches"],
                }
                if self.reflective
                else {}
            ),
        }
