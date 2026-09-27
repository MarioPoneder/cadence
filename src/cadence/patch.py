"""Continuous observations on the existing nonlinear, reciprocal neural core.

Fast activity is carried between real observations; efficacy and bias are learned
by the existing local free/nudged contrast. There is no separate fact store,
importance detector, replay, or weight-consolidation potential here. A small
equation residual certifies neither a minimum nor a unique normal form. The
gradient interpretation additionally needs a smooth stable equilibrium branch
and the usual small-nudge and parameter-scaling assumptions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .brain import Backend, Brain, BrainState, Equilibrium, Nudge
from .certificate import ep_structure
from .connectome import Connectome
from .learning import Learner, LearnerConfig, layered, learning_neuron_model

__all__ = ["PatchNet", "PatchObservation"]


def _copy_state(state: BrainState | None) -> BrainState | None:
    if state is None:
        return None
    return BrainState(
        state.v.copy(),
        state.activation.copy(),
        state.adaptation.copy(),
        state.steps,
        activity_change=None if state.activity_change is None else state.activity_change.copy(),
    )


@dataclass(frozen=True)
class PatchObservation:
    """An external observation's measured phases, before the resulting weight update.

    ``reason`` distinguishes a committed update, a duplicate, absent evidence,
    or an unconverged phase. Duplicate commits do not even advance fast activity.
    The free state is always target-free and is the only phase carried forward.
    """

    free: Equilibrium | None
    plus: Equilibrium | None = None
    minus: Equilibrium | None = None
    updated: bool = False
    reason: str = "no_observations"
    metrics: dict[str, float] = field(default_factory=dict)


class PatchNet:
    """Stateful continuous learning over a reciprocal ``Brain`` and ``Learner``.

    All three phases use the same explicit step budget and full equation
    residual. A weight update is committed only if every required phase
    converges. A rejected attempt retains finite target-free activity but does not
    change weights, optimizer history, or consumed source IDs. Batch rows are
    persistent streams: call ``reset`` when their identities change.

    ``observe`` is the external-evidence boundary. Targets, observation masks,
    gains and source IDs are supplied by the caller, not discovered internally.
    IDs suppress repeats only inside a bounded FIFO window; distinct IDs do not
    establish independent evidence. Internal predictions never call ``observe``.

    Opt-in ``context_strength`` adds a quadratic overlap with the preceding
    free activity on ``context_mask`` (default: the supplied hidden ports).
    That preceding value is fixed across all phases of one observation, not
    updated at each numerical step. It is a temporal boundary, not a witnessed
    target. The update does not differentiate through prior observations.
    Fresh or reset streams use an explicit zero-activity boundary.
    Strength zero preserves the original warm-start-only dynamics. Positive
    strength can slow forgetting but does not guarantee useful memory; large
    values may need a smaller integration step to settle.
    """

    def __init__(
        self,
        learner: Learner,
        *,
        input_index: Sequence[int] | None = None,
        steps: int = 512,
        chunk: int = 32,
        tolerance: float = 1e-5,
        source_capacity: int = 256,
        context_strength: float = 0.0,
        context_mask: np.ndarray | None = None,
    ) -> None:
        for name, value, minimum in (
            ("steps", steps, 0),
            ("chunk", chunk, 1),
            ("source_capacity", source_capacity, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not np.isfinite(context_strength) or context_strength < 0:
            raise ValueError("context_strength must be finite and nonnegative")
        self.learner = learner
        self.steps, self.chunk, self.tolerance = int(steps), int(chunk), float(tolerance)
        self.source_capacity = int(source_capacity)
        indices = (
            learner.brain.connectome.populations.get("input", ())
            if input_index is None
            else input_index
        )
        raw = np.asarray(list(indices))
        if raw.size and (
            raw.ndim != 1
            or raw.dtype.kind not in "iu"
            or (raw < 0).any()
            or (raw >= self.brain.connectome.n).any()
            or len(np.unique(raw)) != len(raw)
        ):
            raise ValueError("input_index must contain distinct valid neuron indices")
        self.input_index = raw.astype(np.int64)
        self.context_strength = float(context_strength)
        if context_mask is None:
            context_mask = np.zeros(self.brain.connectome.n, dtype=bool)
            context_mask[list(self.brain.connectome.populations.get("hidden", ()))] = True
        mask = np.asarray(context_mask)
        if mask.shape != (self.brain.connectome.n,) or mask.dtype != np.bool_:
            raise ValueError("context_mask must be a boolean vector over neurons")
        if self.context_strength and not mask.any():
            raise ValueError("positive context_strength needs at least one context port")
        self.context_mask = mask.copy()
        self._state: BrainState | None = None
        self._sources: dict[str, str] = {}
        self._validate_structure()

    @property
    def brain(self) -> Brain:
        return self.learner.brain

    @property
    def output_index(self) -> np.ndarray:
        return self.learner.output_index.copy()

    @property
    def state(self) -> BrainState | None:
        """A detached copy; editing a readback cannot alter the live state."""
        return _copy_state(self._state)

    def _validate_structure(self) -> None:
        learner, brain = self.learner, self.brain
        if learner.config.nudge != "quadratic" or learner.slot_count != 1:
            raise ValueError("PatchNet requires a quadratic, single-slot Learner")
        if not learner.reciprocal or not ep_structure(brain).compatible:
            raise ValueError("PatchNet requires reciprocal effective weights and no adaptation")
        reverse = learner.reverse
        if (reverse < 0).any():
            raise ValueError("every PatchNet contact needs a reciprocal candidate contact")
        factors = (
            brain.neuron_model.gain
            * brain.connectome.count
            * np.exp(brain.log_gain[brain.connectome.pre])
        )
        if not np.allclose(factors, factors[reverse], rtol=0, atol=1e-12):
            raise ValueError("reciprocal contacts must have equal contact/gain factors")
        assert learner.plastic_synapses is not None
        if not np.array_equal(learner.plastic_synapses, learner.plastic_synapses[reverse]):
            raise ValueError("reciprocal contacts must have equal plasticity masks")

    @classmethod
    def create(
        cls,
        inputs: int,
        hidden: int,
        outputs: int,
        *,
        seed: int = 0,
        density: float = 0.3,
        init: float = 1.0,
        config: LearnerConfig | None = None,
        backend: Backend = "cpu",
        device: str | None = None,
        **runtime_options: Any,
    ) -> PatchNet:
        """Build a fully reciprocal layered graph with unit contact/gain factors.

        The inherited topology and named ports are supplied, not learned roles.
        Inputs are soft external drives and also receive recurrent feedback.
        This factory does not guarantee a memory basin or successful learning.
        """
        graph = layered(inputs, hidden, outputs, seed=seed, density=density, init=init)
        keys = set(zip(graph.pre.tolist(), graph.post.tolist(), strict=True))
        missing = np.array(
            [(int(b), int(a)) not in keys for a, b in zip(graph.pre, graph.post, strict=True)]
        )
        graph = Connectome.from_synapses(
            graph.n,
            pre=np.concatenate([graph.pre, graph.post[missing]]),
            post=np.concatenate([graph.post, graph.pre[missing]]),
            count=np.concatenate([graph.count, graph.count[missing]]),
            sign=np.concatenate([graph.sign, graph.sign[missing]]),
            populations=graph.populations,
            label="patch:" + graph.label,
        )
        learner = Learner(
            Brain(graph, learning_neuron_model(), backend=backend, device=device),
            graph.populations["output"],
            config or LearnerConfig(nudge="quadratic"),
        )
        return cls(learner, **runtime_options)

    @classmethod
    def recursive(
        cls,
        inputs: int,
        layers: Sequence[int],
        outputs: int,
        *,
        seed: int = 0,
        coupling: float = 1.0,
        config: LearnerConfig | None = None,
        backend: Backend = "cpu",
        device: str | None = None,
        **runtime_options: Any,
    ) -> PatchNet:
        """Build observing populations inside one ordinary reciprocal ``PatchNet``.

        ``layers[0]`` is the base hidden population, fully joined to input and
        output neurons. Each later population is joined in both directions to
        every earlier neuron, including inputs, outputs and older observers.
        There are no within-population contacts or privileged overriding layer.
        Outputs use the same neuron rule and joint solve as the observers.

        A single global scale preserves symmetric random weights while setting
        their largest absolute incoming row sum to ``coupling``. With the smooth
        ``tanh(v / 2)`` rule, the initial free map has Lipschitz bound
        ``coupling / 2`` (default 0.5). This initialization bound need not survive
        learning and does not qualify arbitrary nudges or temporal anchors.
        The usual full equation checks still decide whether a phase converged.
        Widths and couplings are supplied candidate genes, not learned roles.
        Construction uses synapse lists, with storage proportional to contacts.
        """
        def width(name: str, value: object) -> int:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
            return int(value)

        inputs, outputs = width("inputs", inputs), width("outputs", outputs)
        if not isinstance(layers, (Sequence, np.ndarray)) or isinstance(layers, (str, bytes)):
            raise ValueError("layers must be a nonempty sequence of positive integers")
        if isinstance(layers, np.ndarray) and layers.ndim != 1:
            raise ValueError("layers must be a nonempty sequence of positive integers")
        sizes = [width(f"layers[{i}]", value) for i, value in enumerate(layers)]
        if not sizes:
            raise ValueError("layers must be a nonempty sequence of positive integers")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if (
            isinstance(coupling, (bool, np.bool_))
            or not isinstance(coupling, (int, float, np.integer, np.floating))
            or not np.isfinite(coupling)
            or coupling <= 0
        ):
            raise ValueError("coupling must be finite and positive")
        selected = LearnerConfig(nudge="quadratic") if config is None else config
        if not isinstance(selected, LearnerConfig) or selected.nudge != "quadratic":
            raise ValueError("PatchNet requires a quadratic LearnerConfig")

        rng = np.random.default_rng(seed)
        populations = {
            "input": tuple(range(inputs)),
            "layer_0": tuple(range(inputs, inputs + sizes[0])),
            "output": tuple(range(inputs + sizes[0], inputs + sizes[0] + outputs)),
        }
        pres: list[np.ndarray] = []
        posts: list[np.ndarray] = []
        signs: list[np.ndarray] = []

        def join(left: Sequence[int], right: Sequence[int]) -> None:
            pre = np.repeat(left, len(right))
            post = np.tile(right, len(left))
            weight = rng.normal(size=len(pre))
            pres.extend((pre, post))
            posts.extend((post, pre))
            signs.extend((weight, weight))

        join(populations["input"], populations["layer_0"])
        join(populations["layer_0"], populations["output"])
        base_size = inputs + sizes[0] + outputs
        count = base_size
        for level, size in enumerate(sizes[1:], start=1):
            members = tuple(range(count, count + size))
            populations[f"layer_{level}"] = members
            join(range(count), members)
            count += size
        populations["observer"] = tuple(range(base_size, count))
        populations["hidden"] = populations["layer_0"] + populations["observer"]
        pre, post, weights = np.concatenate(pres), np.concatenate(posts), np.concatenate(signs)
        row_mass = np.bincount(post, weights=np.abs(weights), minlength=count)
        weights = (weights / row_mass.max()) * float(coupling)
        graph = Connectome.from_synapses(
            count,
            pre=pre,
            post=post,
            sign=weights,
            populations=populations,
            label="patch:recursive",
        )
        learner = Learner(
            Brain(graph, learning_neuron_model(leak=1.0), backend=backend, device=device),
            graph.populations["output"],
            selected,
        )
        return cls(learner, **runtime_options)

    def stimulus(self, inputs: np.ndarray, *, amplitude: float = 1.0) -> np.ndarray:
        """Place continuous inputs on the declared input ports as a soft drive."""
        values = np.asarray(inputs, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or not len(values) or values.shape[1] != len(self.input_index):
            raise ValueError("inputs must have shape (batch, input ports)")
        if not np.isfinite(values).all() or not np.isfinite(amplitude):
            raise ValueError("inputs and amplitude must be finite")
        drive = np.zeros((len(values), self.brain.connectome.n))
        drive[:, self.input_index] = amplitude * values
        return drive

    def read(self, phase_or_state: Equilibrium | BrainState | None = None) -> np.ndarray:
        state = self._state if phase_or_state is None else phase_or_state
        if isinstance(state, Equilibrium):
            state = state.state
        if state is None:
            raise ValueError("there is no activity to read")
        return np.atleast_2d(state.activation)[:, self.learner.output_index].copy()

    def _solve(
        self, drive: np.ndarray, state: BrainState | None, nudge: Nudge | None = None
    ) -> Equilibrium:
        return self.brain.equilibrate(
            drive,
            budget=self.steps,
            chunk=self.chunk,
            tolerance=self.tolerance,
            state=state,
            nudge=nudge,
        )

    def _anchor_nudge(self, prior: BrainState | None, nudge: Nudge | None = None) -> Nudge | None:
        """Fix the same preceding activity as a boundary, independent of target detuning."""
        if self.context_strength == 0:
            return nudge
        if nudge is None:
            nudge = Nudge(np.zeros(self.brain.connectome.n), np.zeros(self.brain.connectome.n), 0.0)
        return replace(
            nudge,
            anchor=np.zeros(self.brain.connectome.n) if prior is None else prior.activation,
            anchor_gain=self.context_strength * self.context_mask,
        )

    def settle(self, drive: np.ndarray) -> Equilibrium:
        """Advance live, target-free activity without any learned-parameter update."""
        phase = self._solve(drive, self._state, self._anchor_nudge(self._state))
        candidate = _copy_state(phase.state)
        assert candidate is not None
        if all(np.isfinite(v).all() for v in
               (candidate.v, candidate.activation, candidate.adaptation)):
            self._state = candidate
        return phase

    def observe(
        self,
        drive: np.ndarray,
        target: np.ndarray,
        *,
        observed: np.ndarray | None = None,
        weight: np.ndarray | None = None,
        source_id: str | None = None,
    ) -> PatchObservation:
        """Learn once from continuous external targets on explicitly observed ports.

        ``target`` is (batch, outputs), or (outputs,) for one stream. ``observed``
        is a shared boolean vector over output ports; unknown target entries may
        be NaN and are ignored. ``weight`` is a nonnegative external gain per
        batch row. The loss is the weighted sum of squared observed-port errors,
        averaged over batch rows. It is not normalized by the observed-port count.
        """
        x = np.asarray(drive, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if (
            x.ndim != 2
            or not len(x)
            or x.shape[1] != self.brain.connectome.n
            or not np.isfinite(x).all()
        ):
            raise ValueError("drive must be a finite (batch, neurons) array")
        count = len(self.learner.output_index)
        known = np.ones(count, dtype=bool) if observed is None else np.asarray(observed)
        if known.shape != (count,) or known.dtype != np.bool_:
            raise ValueError("observed must be a boolean vector over output ports")
        values = np.asarray(target, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape != (len(x), count) or not np.isfinite(values[:, known]).all():
            raise ValueError(
                "target must match batch and output ports, with finite observed values"
            )
        gain = np.ones(len(x)) if weight is None else np.asarray(weight, dtype=float)
        if gain.shape != (len(x),) or not np.isfinite(gain).all() or (gain < 0).any():
            raise ValueError("weight must be a finite nonnegative vector over batch rows")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id or len(source_id) > 1024
        ):
            raise ValueError("source_id must be a nonempty string of at most 1024 characters")
        values = np.where(known, values, 0.0)
        digest = sha256()
        for item in (x, values, known, gain):
            array = np.ascontiguousarray(item)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        fingerprint = digest.hexdigest()
        if source_id in self._sources:
            if self._sources[source_id] != fingerprint:
                raise ValueError("source_id was already committed with different evidence")
            return PatchObservation(None, reason="duplicate")
        self._validate_structure()
        prior = self._state
        free = self.settle(x)
        if not known.any() or not gain.any():
            return PatchObservation(free, reason="no_observations")
        if not np.all(free.converged):
            return PatchObservation(free, reason="free_unconverged")
        target_all = np.zeros_like(x)
        target_all[:, self.learner.output_index] = values
        mask = np.zeros(x.shape[1])
        mask[self.learner.output_index] = known
        beta = self.learner.config.beta
        plus = self._solve(
            x, free.state, self._anchor_nudge(prior, Nudge(target_all, mask, beta, weight=gain))
        )
        minus = (
            self._solve(
                x,
                free.state,
                self._anchor_nudge(prior, Nudge(target_all, mask, -beta, weight=gain)),
            )
            if self.learner.config.centered
            else None
        )
        if not np.all(plus.converged) or (minus is not None and not np.all(minus.converged)):
            return PatchObservation(free, plus, minus, reason="nudge_unconverged")
        metrics = self.learner.update(
            free.state, plus.state, None if minus is None else minus.state
        )
        if source_id is not None and self.source_capacity:
            self._sources[source_id] = fingerprint
            if len(self._sources) > self.source_capacity:
                del self._sources[next(iter(self._sources))]
        return PatchObservation(free, plus, minus, True, "updated", metrics)

    def imagine(
        self, drives: Iterable[np.ndarray], *, state: BrainState | None = None
    ) -> tuple[Equilibrium, ...]:
        """Settle a sequential private branch without learning or changing live state.

        Pass the last returned state to a subsequent call to build a feedback
        rollout from its predictions. No branch is promoted to external evidence.
        The caller still controls any later, explicit call to ``observe``.
        """
        current = _copy_state(self._state if state is None else state)
        result = []
        for drive in drives:
            phase = self._solve(drive, current, self._anchor_nudge(current))
            current = _copy_state(phase.state)
            result.append(phase)
        return tuple(result)

    def reset(self) -> None:
        """Clear fast activity; retain learned parameters, optimizer and source IDs."""
        self._state = None

    def snapshot(self) -> dict[str, np.ndarray]:
        """Detached arrays containing all continuation state (including the ID window)."""
        from .checkpoint import _learner_data

        data = _learner_data(self.learner)
        meta = {
            "format": "cadence-patch/2",
            "steps": self.steps,
            "chunk": self.chunk,
            "tolerance": self.tolerance,
            "source_capacity": self.source_capacity,
            "input_index": self.input_index.tolist(),
            "sources": list(self._sources.items()),
            "state_steps": None if self._state is None else self._state.steps,
            "context_strength": self.context_strength,
            "context_mask": self.context_mask.tolist(),
        }
        if self._state is not None:
            for name in ("v", "activation", "adaptation", "activity_change"):
                value = getattr(self._state, name)
                if value is not None:
                    data["patch/" + name] = value
        data["patch"] = np.array(json.dumps(meta, sort_keys=True, allow_nan=False))
        return {name: np.asarray(value).copy() for name, value in data.items()}

    def save(self, path: str | Path, *, compressed: bool = True) -> Path:
        """Atomically checkpoint parameters, optimizer, activity and ID window."""
        from .checkpoint import _write

        return _write(self.snapshot(), path, compressed=compressed)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend = "cpu",
        device: str | None = None,
        precision: str | None = None,
    ) -> PatchNet:
        """Resume; exact continuation requires the same numerical backend/precision."""
        with np.load(path, allow_pickle=False) as data:
            try:
                meta = json.loads(str(data["patch"]))
                if not isinstance(meta, dict) or meta.get("format") not in (
                    "cadence-patch/1",
                    "cadence-patch/2",
                ):
                    raise ValueError("unsupported PatchNet checkpoint")
                learner = Learner.load(path, backend=backend, device=device, precision=precision)
                if meta["format"] == "cadence-patch/2" and not all(
                    key in meta for key in ("context_strength", "context_mask")
                ):
                    raise ValueError("missing saved temporal boundary configuration")
                result = cls(
                    learner,
                    context_strength=meta.get("context_strength", 0.0),
                    context_mask=np.asarray(meta["context_mask"])
                    if "context_mask" in meta
                    else None,
                    **{
                        name: meta[name]
                        for name in (
                            "input_index",
                            "steps",
                            "chunk",
                            "tolerance",
                            "source_capacity",
                        )
                    },
                )
                sources = meta["sources"]
                if not isinstance(sources, list) or len(sources) > result.source_capacity:
                    raise ValueError("invalid saved source window")
                for entry in sources:
                    if not isinstance(entry, list) or len(entry) != 2:
                        raise ValueError("invalid saved source entry")
                    key, value = entry
                    if (
                        not isinstance(key, str)
                        or not 0 < len(key) <= 1024
                        or key in result._sources
                        or not isinstance(value, str)
                        or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)
                    ):
                        raise ValueError("invalid saved source entry")
                    result._sources[key] = value
                steps = meta["state_steps"]
                if steps is not None:
                    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                        raise ValueError("invalid saved activity steps")
                    arrays = [
                        np.asarray(data["patch/" + name])
                        for name in ("v", "activation", "adaptation")
                    ]
                    shape = arrays[0].shape
                    if len(shape) != 2 or shape[0] == 0 or shape[1] != result.brain.connectome.n:
                        raise ValueError("invalid saved activity dimensions")
                    if any(
                        a.shape != shape or a.dtype.kind != "f" or not np.isfinite(a).all()
                        for a in arrays
                    ):
                        raise ValueError("invalid saved activity arrays")
                    # A portable checkpoint may have been produced in float32.
                    # Its published activity must still match its potentials;
                    # arbitrary teacher activations are not valid continuation state.
                    expected = result.brain.neuron_model.activation(arrays[0])
                    if not np.allclose(arrays[1], expected, rtol=1e-6, atol=1e-6):
                        raise ValueError("saved activation is inconsistent with potentials")
                    if np.any(arrays[2] != 0):
                        raise ValueError("adaptation-free PatchNet has nonzero saved adaptation")
                    change = (
                        data["patch/activity_change"].copy()
                        if "patch/activity_change" in data
                        else None
                    )
                    if change is not None and (
                        change.shape != (shape[0],)
                        or not np.isfinite(change).all()
                        or (change < 0).any()
                    ):
                        raise ValueError("invalid saved activity change")
                    result._state = BrainState(
                        arrays[0].copy(),
                        arrays[1].copy(),
                        arrays[2].copy(),
                        steps,
                        activity_change=change,
                    )
                elif any(name.startswith("patch/") for name in data.files):
                    raise ValueError("activity arrays without saved activity")
                return result
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid or incomplete PatchNet checkpoint") from exc
