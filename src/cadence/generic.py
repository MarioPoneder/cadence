"""A generic brain for simple tasks: senses, cortex, a motor choice, dopamine and memory.

``GenericBrain`` composes the standard regions into one connectome that settles as a whole:

* a sensory region: a blank population for a vector observation, or a ``visual_cortex`` for
  an image;
* an association ``cortex`` that every sensory region projects to;
* a ``motor_cortex`` with one neuron per action, connected to the association cortex by
  reciprocal synapse pairs so that a nudge on the choice reaches the cortex;
* a basal-ganglia analogue, an external ``ActorCritic`` whose linear critic reads the
  association cortex and whose
  dopamine prediction error moves every plastic synapse through its eligibility trace;
* optionally a ``prefrontal_cortex`` driven by a ``Trace`` of the association cortex, a
  working memory of the moments before;
* a default hippocampal-memory analogue, ``SynapticMemory`` from sensory to motor
  neurons, recording outcomes quickly and consolidating them through repetition/salience.

``step`` runs one ongoing perceive/feedback/act loop; demonstrations and rewards are
signals in that loop, without a train/eval mode. The lower-level ``fit``, ``act``
and ``learn`` operations expose individual mechanisms for controlled experiments.
This composition does not include a learned world model, hierarchical goals or language.
Its connectome comes from a ``Genome``, so ``evolve`` can select its region sizes/densities, and a
designed region can replace any of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .brain import Backend, Brain, BrainState, Equilibrium
from .connectome import Connectome
from .genome import Genome, Projection, develop
from .learning import Learner, LearnerConfig, learning_neuron_model
from .memory import SynapticMemory
from .plasticity import ActorCritic, ActorCriticConfig
from .regions import Region, motor_cortex, prefrontal_cortex, visual_cortex
from .stream import FastSynapses, PatternSeparator, Trace

__all__ = ["GenericBrain"]

_REWARD_ARRAYS = (
    "w_critic",
    "trace",
    "trace_bias",
    "trace_critic",
    "second_moment",
    "second_moment_bias",
    "velocity",
    "velocity_bias",
    "salience",
    "_drive",
)


def _learning() -> LearnerConfig:
    # Retain the recovered finite teaching law; numerical damping belongs to
    # qualified free settlement, where it does not change these learning phases.
    return LearnerConfig(
        beta=0.1,
        eta=0.5,
        temperature=0.2,
        tolerance=3e-3,
        free_steps=1024,
        nudged_steps=12,
        momentum=0.9,
    )


def _reward() -> ActorCriticConfig:
    return ActorCriticConfig(gamma=0.9, lam=0.8, eta=1.0, eta_critic=0.3)


def _load_memory(metadata: Any, data: Mapping[str, Any], neurons: int) -> FastSynapses | None:
    """Validate and restore a detached memory before constructing the resumed composition."""
    prefix = "episodic/"

    def array(name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        key = prefix + name
        if key not in data:
            raise ValueError(f"missing saved memory state: {name}")
        value: np.ndarray = data[key]
        if (
            value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
            or (shape is not None and value.shape != shape)
        ):
            raise ValueError(f"invalid saved memory state: {name}")
        return value.copy()

    if metadata is None:
        if any(key.startswith(prefix) for key in data):
            raise ValueError("saved memory arrays require hippocampus metadata")
        return None
    if not isinstance(metadata, dict):
        raise ValueError("invalid hippocampus metadata")
    try:
        kind = metadata.get("kind")
        if kind not in (None, "consolidating"):
            raise ValueError("unsupported saved memory kind")
        for name in ("normalize", "replace"):
            if not isinstance(metadata[name], bool):
                raise ValueError(f"saved memory {name} must be boolean")
        for name in ("pre", "post", "writes"):
            value = metadata[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"saved memory {name} must be a nonnegative integer")
        pre, post = array("pre", (metadata["pre"],)), array("post", (metadata["post"],))
        if np.any(pre >= neurons) or np.any(post >= neurons):
            raise ValueError("saved memory indices exceed the connectome")
        separator = None
        spec = metadata.get("separator")  # Older unseparated checkpoints omitted this field.
        if spec is None:
            if any(key.startswith(prefix + "separator/") for key in data):
                raise ValueError("saved separator arrays require separator metadata")
        else:
            if not isinstance(spec, dict):
                raise ValueError("invalid separator metadata")
            if set(spec) != {"inputs", "expansion", "winners", "seed", "center"}:
                raise ValueError("incomplete or unsupported separator metadata")
            for name in ("inputs", "expansion", "winners", "seed"):
                value = spec[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"saved separator {name} must be a nonnegative integer")
            if spec["inputs"] != len(pre):
                raise ValueError("saved separator inputs must match memory pre neurons")
            # Do not regenerate these arrays from the seed: projections can be customized,
            # and the running center of FastSynapses can have changed after observations.
            projection = array("separator/projection", (spec["inputs"], spec["expansion"]))
            mean = array("separator/mean", (spec["inputs"],))
            separator = PatternSeparator(**spec)
            separator.projection, separator.mean = projection, mean
        options = {
            name: metadata[name]
            for name in ("decay", "rate", "amplitude", "normalize", "replace", "rule", "writes")
        }
        memory: FastSynapses
        if kind == "consolidating":
            memory = SynapticMemory(
                pre, post, separator=separator, consolidation=metadata["consolidation"], **options
            )
        else:
            if prefix + "consolidated" in data:
                raise ValueError("consolidated weights require consolidating memory metadata")
            memory = FastSynapses(pre, post, separator=separator, **options)
        strength = array("strength")
        if strength.ndim != 3 or strength.shape[1:] != (memory.key_width, len(post)):
            raise ValueError("invalid saved memory strength dimensions")
        mass = array("mass", (len(strength),))
        if np.any(mass < 0):
            raise ValueError("saved memory mass must be nonnegative")
        if isinstance(memory, SynapticMemory):
            memory.consolidated = array("consolidated", (memory.key_width, len(post)))
            if (
                metadata.get("persistent_parameters", memory.consolidated.size)
                != memory.consolidated.size
            ):
                raise ValueError("inconsistent saved persistent parameter count")
        memory.strength, memory.mass = strength, mass
        return memory
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid or incomplete saved memory metadata") from exc


def _validate_life_state(meta: dict[str, Any], data: Mapping[str, Any], learner: Learner) -> None:
    """Reject corrupt continuation state before exposing a resumed agent."""
    n = learner.brain.connectome.n
    populations = learner.brain.connectome.populations
    association = len(populations["association"])
    sensory = len(populations.get("sensory", populations.get("visual/input", ())))

    def array(name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        if name not in data:
            raise ValueError(f"missing saved state: {name}")
        value: np.ndarray = data[name]
        if (
            value.dtype.kind not in "biuf"
            or not np.isfinite(value).all()
            or (shape is not None and value.shape != shape)
        ):
            raise ValueError(f"invalid saved state: {name}")
        return value

    def count(name: str) -> int:
        value = meta.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid saved count: {name}")
        return value

    count("updates")
    bias = meta.get("b_critic")
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not np.isfinite(bias):
        raise ValueError("invalid saved critic bias")
    for name in ("pending", "free_current"):
        if name in meta and not isinstance(meta[name], bool):
            raise ValueError(f"invalid saved flag: {name}")
    batch = 0
    if meta["free_steps"] is not None:
        count("free_steps")
        v = array("free/v")
        if v.ndim != 2 or v.shape[1] != n or not len(v):
            raise ValueError("invalid saved free-state dimensions")
        batch = len(v)
        for name in ("activation", "adaptation"):
            array("free/" + name, (batch, n))
    elif meta.get("pending", False) or meta.get("free_current", False):
        raise ValueError("saved pending action or cache needs a free state")
    if batch:
        array("actor/_drive", (batch, n))
    if ("actor/trace" in data) != ("actor/trace_bias" in data):
        raise ValueError("incomplete saved eligibility traces")
    array("actor/w_critic", (association,))
    for name in ("second_moment", "second_moment_bias", "velocity", "velocity_bias"):
        size = n if name.endswith("bias") else learner.brain.connectome.synapses
        value = array("actor/" + name, (size,))
        if name.startswith("second_moment") and (value < 0).any():
            raise ValueError(f"negative saved second moment: {name}")
    for name, width in (
        ("trace", learner.brain.connectome.synapses),
        ("trace_bias", n),
        ("trace_critic", association + 1),
        ("salience", n),
        ("_drive", n),
    ):
        if "actor/" + name in data:
            array("actor/" + name, (batch, width))
    for name in ("mean", "var"):
        value = array("valence/" + name)
        if value.shape not in ((), (batch,)) or (name == "var" and (value < 0).any()):
            raise ValueError(f"invalid saved valence: {name}")
    if "prepared" in data:
        array("prepared", (batch, sensory))
    if meta.get("pending", False):
        array("pending/value", (batch,))
        for phase in ("plus", "minus"):
            count("pending_" + phase + "_steps")
            for name in ("v", "activation", "adaptation"):
                array(f"pending/{phase}/{name}", (batch, n))
    if "moment/observations" in data or "moment/action" in data:
        if not meta.get("pending", False):
            raise ValueError("saved action observations need pending credit")
        array("moment/observations", (batch, sensory))
        action = array("moment/action", (batch,))
        if (
            action.dtype.kind not in "iu"
            or (action < 0).any()
            or (action >= len(learner.output_index)).any()
        ):
            raise ValueError("invalid saved action indices")
    working = meta["working_memory"]
    if working is not None:
        source, target = working["source"], working["target"]
        if source not in populations or target not in populations:
            raise ValueError("invalid saved working-memory ports")
        width = len(populations[source])
        array("working/trace", (batch, width))
        array("working/last", (batch, width))
        if array("working/cold", (batch,)).dtype != np.bool_:
            raise ValueError("invalid saved working-memory cold flags")


class GenericBrain:
    """Senses, an association cortex, a motor cortex, basal ganglia and consolidating memory.

    ``connectome`` needs populations ``sensory`` (or ``visual/input`` for an image),
    ``association`` and ``motor``, and ``prefrontal`` for a working memory; ``genome`` builds
    one. Use ``GenericBrain.build`` for the default layout.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        episodic: bool = True,
        consolidation: float = 0.05,
        working_memory_decay: float = 0.2,
        working_memory_amplitude: float = 3.0,
        learning: LearnerConfig | None = None,
        reward: ActorCriticConfig | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        device: str | None = None,
    ) -> None:
        populations = connectome.populations
        for name in ("association", "motor"):
            if name not in populations:
                raise ValueError(f"a generic brain needs a population named {name!r}")
        sensory = "sensory" if "sensory" in populations else "visual/input"
        if sensory not in populations:
            raise ValueError("a generic brain needs a 'sensory' or 'visual/input' population")
        self.connectome = connectome
        self.sensory_index = np.asarray(populations[sensory], dtype=np.int64)
        self.motor_index = np.asarray(populations["motor"], dtype=np.int64)
        self.association_index = np.asarray(populations["association"], dtype=np.int64)
        brain = Brain(connectome, learning_neuron_model(dt=1.0), backend=backend, device=device)
        self.learner = Learner(brain, list(self.motor_index), learning or _learning())
        self.basal_ganglia = ActorCritic(
            self.learner, list(self.association_index), reward or _reward(), seed=seed
        )
        self.working_memory: Trace | None = None
        if "prefrontal" in populations:
            self.working_memory = Trace(
                connectome,
                decay=working_memory_decay,
                amplitude=working_memory_amplitude,
                source="association",
                target="prefrontal",
            )
        self.hippocampus: FastSynapses | None = None
        if episodic:
            self.hippocampus = SynapticMemory(
                self.sensory_index, self.motor_index, consolidation=consolidation
            )
        self.rng = np.random.default_rng(seed)
        self._moment: tuple[np.ndarray, np.ndarray] | None = None
        self._prepared: np.ndarray | None = None  # the observations ``learn`` settled
        self.last_learning: dict[str, float] = {}

    @property
    def brain(self) -> Brain:
        """The current learned dynamics; learning can replace this Brain instance."""
        return self.learner.brain

    # -- construction

    @staticmethod
    def genome(
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float = -0.5,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
    ) -> Genome:
        """The default layout: ``inputs`` is a vector length or an image shape
        ``(height, width)`` or ``(height, width, channels)``."""
        if isinstance(inputs, (int, np.integer)):
            sensory = Region("sensory", inputs)
            forward = Projection("sensory", "association", density=density, reciprocal=False)
        else:
            shape = tuple(inputs)
            if len(shape) not in (2, 3):
                raise ValueError(
                    "image inputs must be (height, width) or (height, width, channels)"
                )
            height, width, *rest = shape
            sensory = visual_cortex(
                height,
                width,
                channels=rest[0] if rest else 1,
                features=features,
                field=field,
                seed=seed,
            )
            forward = Projection("visual", "association", density=density)
        regions = [sensory, Region("association", hidden), motor_cortex(actions, lateral=lateral)]
        projections = [forward, Projection("association", "motor")]
        if working_memory:
            regions.insert(1, prefrontal_cortex(hidden))
            # the trace of a settled cortex is small; a strong projection lets it steer the next
            # moment
            projections.append(
                Projection("prefrontal", "association", scale=memory_scale, reciprocal=False)
            )
        return Genome(tuple(regions), tuple(projections), label="generic-brain")

    @classmethod
    def build(
        cls,
        inputs: int | Sequence[int],
        actions: int,
        *,
        hidden: int = 64,
        density: float = 1.0,
        lateral: float = -0.5,
        working_memory: bool = False,
        memory_scale: float = 12.0,
        episodic: bool = True,
        features: int = 8,
        field: int = 3,
        seed: int = 0,
        **options: Any,
    ) -> GenericBrain:
        """Develop the default genome and wrap it; ``options`` go to the constructor."""
        genome = cls.genome(
            inputs,
            actions,
            hidden=hidden,
            density=density,
            lateral=lateral,
            working_memory=working_memory,
            memory_scale=memory_scale,
            features=features,
            field=field,
            seed=seed,
        )
        return cls(develop(genome, seed=seed), episodic=episodic, seed=seed, **options)

    @classmethod
    def compose(
        cls,
        inputs: int,
        actions: int,
        *,
        modules: Sequence[int] = (64,),
        observers: Sequence[int] = (),
        seed: int = 0,
        **options: Any,
    ) -> GenericBrain:
        """Compose a modular brain with working memory and consolidation.

        ``modules`` specifies a chain of reciprocally connected processing
        regions. The final region is the association cortex. ``observers``
        optionally adds regions with reciprocal access to all processing and
        motor regions, including earlier observers. Every region participates
        in the same neural settlement; observer regions never run as a separate
        critic or override an already completed answer.

        This reuses the existing trace, synaptic memory and learning mechanisms.
        Observer wiring is an experiment, not evidence of learned self-reflection.
        Inputs are fixed external drives; their neural representations can vary.
        ``options`` configures the existing constructor's learning and memory.
        Use ``genome``/``Genome`` for custom ports, sparsity and named wiring.
        """

        def size(value: Any) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("population sizes must be positive integers")
            if value < 1:
                raise ValueError("population sizes must be positive integers")
            return int(value)

        inputs, actions = size(inputs), size(actions)
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if "episodic" in options and not isinstance(options["episodic"], bool):
            raise ValueError("episodic must be boolean")
        if isinstance(modules, (str, bytes)) or isinstance(observers, (str, bytes)):
            raise ValueError("modules and observers must be sequences of population sizes")
        try:
            widths = tuple(size(value) for value in modules)
            observer_widths = tuple(size(value) for value in observers)
        except TypeError as exc:
            raise ValueError("modules and observers must be sequences of population sizes") from exc
        if not widths:
            raise ValueError("at least one processing module is required")
        names = [f"module_{index}" for index in range(len(widths) - 1)] + ["association"]
        regions = [Region("sensory", inputs)]
        regions.extend(Region(name, width) for name, width in zip(names, widths, strict=True))
        regions.extend((prefrontal_cortex(widths[-1]), motor_cortex(actions, lateral=-0.5)))
        projections = [Projection("sensory", names[0], reciprocal=False)]
        projections.extend(
            Projection(left, right) for left, right in zip(names, names[1:], strict=False)
        )
        projections.extend(
            (
                Projection("association", "motor"),
                Projection("prefrontal", "association", scale=12.0, reciprocal=False),
            )
        )
        observed = [*names, "motor"]
        for index, width in enumerate(observer_widths):
            name = f"observer_{index}"
            regions.append(Region(name, width))
            projections.extend(Projection(source, name) for source in observed)
            observed.append(name)
        genome = Genome(tuple(regions), tuple(projections), label="composed-brain")
        return cls(develop(genome, seed=seed), seed=seed, **options)

    # -- stimulus

    def _observations(self, observations: Any) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim < 2 or not x.shape[0] or not np.isfinite(x).all():
            raise ValueError(
                "observations must be a nonempty finite batch; use [observation] for one"
            )
        x = x.reshape(len(x), -1)
        if x.shape[1] != len(self.sensory_index):
            raise ValueError(
                f"observations need {len(self.sensory_index)} values per row, got {x.shape[1]}"
            )
        return x

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        """The drive of a batch of observations: the sensory neurons stimulated, plus the
        working memory and the hippocampal recall when ``memory`` is set."""
        x = self._observations(observations)
        drive = np.zeros((len(x), self.connectome.n))
        drive[:, self.sensory_index] = x * self.brain.neuron_model.stimulus_amplitude
        if memory and self.working_memory is not None:
            drive = self.working_memory.stimulate(drive)
        if memory and self.hippocampus is not None:
            drive = self.hippocampus.stimulate(drive)
        return drive

    # -- independent-sample convenience operations (no mode switch)

    def fit(
        self,
        observations: Any,
        labels: Any,
        *,
        epochs: int = 30,
        batch: int = 32,
    ) -> list[float]:
        """Supervised learning on independent samples; the training accuracy after each epoch."""
        drive = self.stimulus(observations, memory=False)
        labels = self.learner._labels(np.asarray(labels))
        if len(labels) != len(drive):
            raise ValueError("observations and labels must have the same batch size")
        for name, value, minimum in (("epochs", epochs, 0), ("batch", batch, 1)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        # Supervision changes the parameters under any previously cached decision.
        self.reset()
        history = []
        for _ in range(epochs):
            order = self.rng.permutation(len(labels))
            for start in range(0, len(labels), batch):
                rows = order[start : start + batch]
                self.learner.step(drive[rows], labels[rows])
            history.append(self.learner.accuracy(drive, labels))
        return history

    def _equilibrate(
        self, drive: np.ndarray, state: BrainState | None, *, budget: int, tolerance: float
    ) -> Equilibrium:
        """One bounded free solve, with underrelaxation if fast repair does not qualify.

        Reserve half the sweeps for numerical damping. The second attempt uses
        identical parameters and fixed-point equations, with half the integration
        step. It never changes the live model or its recovered finite teaching
        law. Both attempts together respect the caller's exact sweep budget.
        This numerical fallback is independent of optional observer wiring.
        """
        from .patch import _copy_state

        brain = self.brain
        first_budget = budget if budget < 2 else (budget + 1) // 2
        phase = brain.equilibrate(
            drive, state=_copy_state(state), budget=first_budget, tolerance=tolerance
        )
        if np.all(phase.qualified) or phase.steps >= budget:
            return phase
        if not all(
            np.isfinite(value).all()
            for value in (phase.state.v, phase.state.activation, phase.state.adaptation)
        ):
            return phase
        damped = Brain(
            brain.connectome,
            brain.neuron_model.replace(dt=brain.neuron_model.dt / 2),
            backend=brain.backend,
            efficacy=brain.efficacy,
            log_gain=brain.log_gain,
            bias=brain.bias,
            device=str(brain._torch.device) if brain._torch is not None else None,
            dense_limit=brain.dense_limit,
            layout=brain.layout,
            precision=brain.precision,
        )
        following = damped.equilibrate(
            drive, state=_copy_state(phase.state), budget=budget - phase.steps, tolerance=tolerance
        )
        # Certify against the original model, not a changed interpretation.
        residual = brain.residual(drive, following.state)
        return Equilibrium(following.state, residual, phase.steps + following.steps, tolerance)

    def _qualified(self, drive: np.ndarray, state: BrainState | None = None) -> BrainState:
        """Check the complete current graph before publishing an answer."""
        cfg = self.learner.config
        if cfg.tolerance is None:
            raise ValueError("GenericBrain answers require a finite residual tolerance")
        phase = self._equilibrate(drive, state, budget=cfg.free_steps, tolerance=cfg.tolerance)
        if not np.all(phase.qualified):
            raise RuntimeError(
                f"brain did not settle within {cfg.free_steps} steps: "
                f"residual={float(np.max(phase.residual)):.6g}, tolerance={cfg.tolerance:g}; "
                "no action issued"
            )
        return phase.state

    def predict(self, observations: Any) -> np.ndarray:
        """Qualified independent observations, without reading or changing live memory."""
        state = self._qualified(self.stimulus(observations, memory=False))
        return np.asarray(np.argmax(state.activation[:, self.motor_index], axis=1))

    def accuracy(self, observations: Any, labels: Any) -> float:
        x = self._observations(observations)
        expected = self.learner._labels(np.asarray(labels))
        if len(expected) != len(x):
            raise ValueError("observations and labels must have the same batch size")
        hits = sum(
            int(np.sum(self.predict(x[start : start + 256]) == expected[start : start + 256]))
            for start in range(0, len(x), 256)
        )
        return hits / expected.size

    # -- ongoing interaction

    def step(
        self,
        observations: Any,
        *,
        reward: Any = None,
        done: Any = None,
        teacher: Any = None,
        salience: Any = None,
        bootstrap: Any = None,
    ) -> np.ndarray:
        """One operating mode: observe, learn from the last outcome, and act again.

        Reward/done describe the preceding action; an omitted reward means no
        reward event on a real transition (zero), not an unknown outcome. Wait for
        the issued action to finish before the next call. ``teacher`` labels the
        current observation. The first
        call has no previous action to reward. Later calls retain the batch's row
        identities; call ``reset`` before starting different streams. Salience is
        nonnegative, per row, and defaults to absolute reward for consolidation.
        There is no training/inference switch. For a frozen measurement use a
        separate instance's ``predict``/greedy ``act``. If the next action fails
        to settle, already accepted feedback remains learned. Retry with ``act``
        after adjusting the solver; do not submit that outcome a second time.
        """
        x = self._observations(observations)
        batch = len(x)
        current = self.basal_ganglia.state
        if current is not None and len(np.atleast_2d(current.v)) != batch:
            raise ValueError("action batch must match the live streams; reset for new streams")
        r = np.zeros(batch) if reward is None else np.asarray(reward, dtype=float)
        ended = np.zeros(batch, bool) if done is None else np.asarray(done)
        if r.shape != (batch,) or not np.isfinite(r).all():
            raise ValueError("reward must be a finite vector matching the observation batch")
        if ended.shape != (batch,) or ended.dtype != np.bool_:
            raise ValueError("done must be a boolean vector matching the observation batch")
        importance = SynapticMemory.salience_vector(
            np.abs(r) if salience is None else salience, batch
        )
        labels = None if teacher is None else self.learner._labels(np.asarray(teacher))
        if labels is not None and len(labels) != batch:
            raise ValueError("teacher must match the observation batch")
        if bootstrap is not None:
            bootstrap = np.asarray(bootstrap, dtype=float)
            if bootstrap.shape != (batch,) or not np.isfinite(bootstrap).all():
                raise ValueError("bootstrap must be a finite vector matching the batch")
        pending = self.basal_ganglia._pending is not None
        if not pending and (reward is not None or done is not None or bootstrap is not None):
            raise RuntimeError("feedback needs a preceding action; start with step(observations)")
        # All feedback and demonstrations are checked before changing any state.
        if pending:
            self.last_learning = self.learn(r, ended, x, bootstrap=bootstrap, salience=importance)
        else:
            self.last_learning = {}
        if labels is not None:
            drive = self.stimulus(x)
            self.learner.step(drive, labels)
            # A demonstration teaches the slow policy, not an invented reward value.
            self._prepared = None
            self.basal_ganglia._drive = None
            self.last_learning["demonstrations"] = float(batch)
        return self.act(x)

    # -- lower-level interaction operations

    def imagine(
        self,
        observations: Sequence[Any],
        *,
        budget: int = 1024,
        tolerance: float = 1e-6,
    ) -> tuple[Equilibrium, ...]:
        """Privately settle responses to a supplied sequence of possible observations.

        Each item is a batch, with the same stream identities throughout. A
        private trace advances along the imagined branch; durable memory is read
        without writes. Parameters, live activity, random generators and pending
        real outcomes are unchanged. Motor responses are at ``motor_index`` in
        each returned state's activation. A refused phase ends the branch and
        remains in the result; inspect ``qualified`` before using a response.

        This imagines the brain's responses, not an unobserved environment's
        dynamics. ``TemporalPatchNet.plan`` supplies the separate learned
        consequence-model interface for continuous action planning.
        """
        from copy import copy

        from .patch import _copy_state

        if (
            isinstance(budget, (bool, np.bool_))
            or not isinstance(budget, (int, np.integer))
            or budget < 0
        ):
            raise ValueError("budget must be a nonnegative integer")
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        sequence = tuple(self._observations(value) for value in observations)
        if not sequence:
            return ()
        batch = len(sequence[0])
        if any(len(value) != batch for value in sequence):
            raise ValueError("imagined observations must preserve the batch's stream identities")
        current = _copy_state(self.basal_ganglia.state)
        if current is not None and len(np.atleast_2d(current.v)) != batch:
            raise ValueError("imagined batch must match the live streams; reset for new streams")
        trace = copy(self.working_memory)
        if trace is not None:
            for name in ("trace", "last", "cold"):
                setattr(trace, name, getattr(trace, name).copy())
        phases = []
        for value in sequence:
            drive = self.stimulus(value, memory=False)
            if trace is not None:
                drive = trace.stimulate(drive)
            if self.hippocampus is not None:
                drive = self.hippocampus.stimulate(drive)
            phase = self._equilibrate(drive, current, budget=budget, tolerance=tolerance)
            phases.append(phase)
            if not np.all(phase.qualified):
                break
            current = _copy_state(phase.state)
            if trace is not None:
                trace.update(phase.state)
        return tuple(phases)

    def act(self, observations: Any, *, greedy: bool = False) -> np.ndarray:
        """Qualify the whole graph and choose an action for each continuing stream.

        ``learning.free_steps`` bounds repair; ``learning.tolerance`` checks the
        full potential/adaptation equations, including optional observers.
        Exhaustion raises ``RuntimeError`` before changing activity, memory,
        random state or pending feedback. A cached state is always checked anew.
        Reward eligibility retains its separate finite nudged-phase contract.
        """
        x = self._observations(observations)
        current = self.basal_ganglia.state
        if current is not None and len(np.atleast_2d(current.v)) != len(x):
            raise ValueError("action batch must match the live streams; reset for new streams")
        drive = self.stimulus(x)
        free = self._qualified(drive, current)
        self.basal_ganglia._free = free
        self.basal_ganglia._free_brain = self.brain
        self.basal_ganglia._drive = drive.copy()
        self._prepared = None
        action = self.basal_ganglia.act(drive, greedy=greedy)
        state = self.basal_ganglia.state
        if self.working_memory is not None and state is not None:
            self.working_memory.update(state)
        self._moment = None if greedy else (x.copy(), action.copy())
        return action

    def learn(
        self,
        reward: Any,
        done: Any,
        next_observations: Any,
        *,
        bootstrap: Any = None,
        salience: Any = None,
    ) -> dict[str, float]:
        """Dopamine from the reward of the last action; ``done`` rows start a new episode.

        The hippocampus records the reward of the chosen action for the situation it was
        chosen in, and the next choice in that situation reads the record.
        """
        following = self._observations(next_observations)
        # Preflight without reading or resetting memories: invalid input must not write an episode.
        reward, done, _, bootstrap = self.basal_ganglia._validated_transition(
            reward, done, self.stimulus(following, memory=False), bootstrap
        )
        importance = SynapticMemory.salience_vector(
            np.abs(reward) if salience is None else salience, len(following)
        )
        if self.hippocampus is not None and self._moment is not None:
            keys, action = self._moment
            if isinstance(self.hippocampus, SynapticMemory):
                target = np.zeros((len(action), len(self.motor_index)))
                target[np.arange(len(action)), action] = reward
                observed = np.zeros(target.shape, bool)
                observed[np.arange(len(action)), action] = True
                self.hippocampus.observe(keys, target, salience=importance, value_mask=observed)
            else:
                target = self.hippocampus.recall(keys)
                target[np.arange(len(action)), action] = reward
                self.hippocampus.observe(keys, target)
        if self.working_memory is not None and done.any():
            self.working_memory.reset(len(done), rows=np.flatnonzero(done))
        report = self.basal_ganglia.learn(reward, done, self.stimulus(following), bootstrap)
        self._moment = None
        self._prepared = following.copy()
        return report

    def reset(self) -> None:
        """Start every stream afresh; hippocampal records are kept."""
        self.basal_ganglia.reset()
        if self.working_memory is not None:
            self.working_memory.reset(0)
        self._moment = None
        self._prepared = None
        self.last_learning = {}

    def parameters(self) -> int:
        """Actor/critic and consolidated weights; per-stream transient storage is additional."""
        return self.basal_ganglia.parameters() + (
            self.hippocampus.consolidated.size
            if isinstance(self.hippocampus, SynapticMemory)
            else 0
        )

    def save(self, path: str | Path) -> Path:
        """Save the complete composition, including an action awaiting its real outcome.

        Includes critic, optimizer, random generators, working and episodic memories,
        eligibility, the current free phase and any pending action's nudged states.
        ``Learner.save`` remains available for the slow learned response alone.
        """
        from .checkpoint import _learner_data, _write
        from .receipts import canonical_json

        agent = self.basal_ganglia
        data = _learner_data(self.learner)
        metadata: dict[str, Any] = {
            "format": "cadence-generic/2",
            "reward": agent.config.to_dict(),
            "rng": self.rng.bit_generator.state,
            "actor_rng": agent.rng.bit_generator.state,
            "updates": agent.updates,
            "b_critic": agent.b_critic,
            "working_memory": None
            if self.working_memory is None
            else self.working_memory.to_dict(),
            "hippocampus": None if self.hippocampus is None else self.hippocampus.to_dict(),
            "free_steps": None if agent.state is None else agent.state.steps,
            "free_current": agent._free_brain is self.brain,
            "pending": agent._pending is not None,
            "last_learning": self.last_learning,
        }
        if agent._pending is not None:
            kind, plus, minus, value = agent._pending
            if kind != "states":
                raise ValueError("GenericBrain checkpoints require its standard action states")
            data["pending/value"] = value
            for name, phase in (("plus", plus), ("minus", minus)):
                metadata["pending_" + name + "_steps"] = phase.steps
                for field in ("v", "activation", "adaptation"):
                    data[f"pending/{name}/{field}"] = np.asarray(getattr(phase, field))
            if self._moment is not None:
                data["moment/observations"], data["moment/action"] = self._moment
        for name in _REWARD_ARRAYS:
            value = getattr(agent, name)
            if value is not None:
                data["actor/" + name] = np.asarray(value)
        if agent._trace_device is not None:
            for name, tensor in zip(("trace", "trace_bias"), agent._trace_device, strict=True):
                data["actor/" + name] = tensor.detach().cpu().double().numpy()
        for name in ("mean", "var"):
            data["valence/" + name] = np.asarray(getattr(agent.valence, name))
        if agent.state is not None:
            for name in ("v", "activation", "adaptation"):
                data["free/" + name] = np.asarray(getattr(agent.state, name))
        if self._prepared is not None:
            data["prepared"] = self._prepared
        if self.working_memory is not None:
            for name in ("trace", "last", "cold"):
                data["working/" + name] = getattr(self.working_memory, name)
        if self.hippocampus is not None:
            for name in ("pre", "post", "strength", "mass"):
                data["episodic/" + name] = getattr(self.hippocampus, name)
            if self.hippocampus.separator is not None:
                for name in ("projection", "mean"):
                    data["episodic/separator/" + name] = getattr(self.hippocampus.separator, name)
            if isinstance(self.hippocampus, SynapticMemory):
                data["episodic/consolidated"] = self.hippocampus.consolidated
        data["generic"] = np.array(canonical_json(metadata))
        return _write(data, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        backend: Backend = "cpu",
        device: str | None = None,
        precision: str | None = None,
    ) -> GenericBrain:
        """Resume a complete saved brain, defaulting to portable CPU inference and learning.

        Keep the saved batch row identities, or call ``reset`` for new episodes.
        Cross-backend results are subject to floating-point differences.
        """
        import json

        with np.load(path, allow_pickle=False) as data:
            if "generic" not in data:
                raise ValueError("not a GenericBrain checkpoint; use Learner.load for a learner")
            meta = json.loads(str(data["generic"]))
            if not isinstance(meta, dict) or meta.get("format") not in (
                "cadence-generic/1",
                "cadence-generic/2",
            ):
                raise ValueError("unsupported GenericBrain checkpoint format")
            if "hippocampus" not in meta:
                raise ValueError("missing hippocampus metadata")
            learner = Learner.load(path, backend=backend, device=device, precision=precision)
            try:
                _validate_life_state(meta, data, learner)
            except (KeyError, TypeError, OverflowError) as exc:
                raise ValueError("invalid or incomplete saved continuation state") from exc
            memory = _load_memory(meta["hippocampus"], data, learner.brain.connectome.n)
            result = cls(
                learner.brain.connectome, episodic=False, reward=ActorCriticConfig(**meta["reward"])
            )
            result.learner = learner
            agent = result.basal_ganglia
            agent.learner = learner
            for name in _REWARD_ARRAYS:
                if "actor/" + name in data:
                    value = data["actor/" + name].copy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite saved actor state: {name}")
                    setattr(agent, name, value)
            agent.updates, agent.b_critic = int(meta["updates"]), float(meta["b_critic"])
            result.rng.bit_generator.state = meta["rng"]
            agent.rng.bit_generator.state = meta["actor_rng"]
            agent.valence.mean = data["valence/mean"].copy()
            agent.valence.var = data["valence/var"].copy()
            if meta["free_steps"] is not None:
                agent._free = BrainState(
                    data["free/v"].copy(),
                    data["free/activation"].copy(),
                    data["free/adaptation"].copy(),
                    int(meta["free_steps"]),
                )
                agent._free_brain = learner.brain if meta.get("free_current", False) else None
            result._prepared = data["prepared"].copy() if "prepared" in data else None
            result.last_learning = meta.get("last_learning", {})
            if meta.get("pending", False):
                phases = []
                for name in ("plus", "minus"):
                    phases.append(
                        BrainState(
                            data[f"pending/{name}/v"].copy(),
                            data[f"pending/{name}/activation"].copy(),
                            data[f"pending/{name}/adaptation"].copy(),
                            int(meta["pending_" + name + "_steps"]),
                        )
                    )
                agent._pending = ("states", phases[0], phases[1], data["pending/value"].copy())
                if "moment/observations" in data:
                    result._moment = (
                        data["moment/observations"].copy(),
                        data["moment/action"].copy(),
                    )
            working = meta["working_memory"]
            result.working_memory = None
            if working is not None:
                result.working_memory = Trace(
                    result.connectome,
                    **{
                        name: working[name]
                        for name in ("decay", "amplitude", "focus", "source", "target")
                    },
                )
                for name in ("trace", "last", "cold"):
                    setattr(result.working_memory, name, data["working/" + name].copy())
            result.hippocampus = memory
        return result
