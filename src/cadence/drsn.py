"""Population layouts with live recursive observation and joint local repair.

``inputs`` carry sensor or represented values. ``observes`` additionally reads
the exact current prediction errors of other populations. Both are constraints
in one energy: their feedback participates in the same settlement.

The nonlinear residual-energy model is a candidate learning architecture.
Qualification means projected stationarity, not unique equilibrium, zero
prediction error, biological fidelity or a demonstrated depth advantage.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from itertools import islice
from types import MappingProxyType

from . import _repair
from ._validation import canonical, integer, number, strict_json

SCHEMA = "population-brain/1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
IMPLEMENTATION = MappingProxyType(
    {
        name: hashlib.sha256(files(__package__).joinpath(name).read_bytes()).hexdigest()
        for name in ("_repair.py", "drsn.py", "_validation.py")
    }
)


class SettlementError(RuntimeError):
    """A population solve did not meet its full stationarity threshold."""


def _shape(shape):
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, (tuple, list)) or len(shape) > 8:
        raise ValueError("shape must have at most eight positive dimensions")
    return tuple(integer(n, "shape dimension", 1) for n in shape)


def _values(value, shape, name):
    """Accept a shaped nested array or an explicitly flattened numeric vector."""
    count = math.prod(shape)
    if callable(getattr(value, "tolist", None)):
        array_shape = getattr(value, "shape", None)
        if array_shape is not None:
            if not isinstance(array_shape, (tuple, list)) or tuple(array_shape) not in (
                shape,
                (count,),
            ):
                raise ValueError(f"{name} does not match shape {shape}")
        value = value.tolist()
    if not shape:
        return (number(value, name),)
    if isinstance(value, (list, tuple)) and len(value) == count:
        if all(not isinstance(v, (list, tuple)) for v in value):
            return tuple(number(v, name) for v in value)

    def flatten(item, dimensions):
        if not dimensions:
            return [number(item, name)]
        if not isinstance(item, (list, tuple)) or len(item) != dimensions[0]:
            raise ValueError(f"{name} does not match shape {shape}")
        return [v for child in item for v in flatten(child, dimensions[1:])]

    return tuple(flatten(value, shape))


@dataclass(frozen=True, slots=True, eq=False)
class Input:
    """A named sensor boundary; ``shape`` describes externally clamped data."""

    name: str
    shape: tuple[int, ...]
    _owner: object = field(repr=False)

    @property
    def size(self):
        """Number of scalar sensor samples, independent of processing capacity."""
        return math.prod(self.shape)


@dataclass(frozen=True, slots=True, eq=False)
class Population:
    """An exact patch count and its data and internal-observation connections."""

    name: str
    patches: int
    inputs: tuple
    observes: tuple
    _owner: object = field(repr=False)

    def __repr__(self):
        inputs = tuple(source.name for source in self.inputs)
        observes = tuple(source.name for source in self.observes)
        return (
            f"Population(name={self.name!r}, patches={self.patches!r}, "
            f"inputs={inputs!r}, observes={observes!r})"
        )

    @property
    def role(self):
        """Layout role; observation function still requires causal testing."""
        return "observer" if self.observes else "processing"


@dataclass(frozen=True, slots=True, eq=False)
class Output:
    """A shaped selection of settled patch values, without a separate policy head."""

    name: str
    shape: tuple[int, ...]
    reads: Population
    indices: tuple[int, ...]
    _owner: object = field(repr=False)


class Cortex:
    """Declare a population graph and compile it with :meth:`build`.

    ``seed`` fixes sparse wiring and initial relations. ``fan_in`` is the
    minimum sample count per source per target patch, raised when necessary
    to cover every source coordinate across the destination population.
    ``initial_scale`` bounds random weights before fan-in normalization.

    ``settle_budget``, ``tolerance``, ``step`` and ``backtracks`` configure the
    common projected-gradient repair. ``state_prior`` penalizes activity;
    ``parameter_prior`` anchors relation changes to pre-experience parameters.
    Both priors are strictly positive. ``state_bound`` and ``parameter_bound``
    bound the state and relation boxes. Values outside them are rejected.

    ``max_inputs``, ``max_patches`` and ``max_connections`` bound construction.
    They count scalar sensor samples, processing patches and directed signal
    connections respectively, not physical process memory or latency.
    """

    def __init__(
        self,
        *,
        seed=0,
        fan_in=8,
        initial_scale=0.3,
        settle_budget=512,
        tolerance=1e-6,
        state_prior=0.01,
        parameter_prior=0.1,
        state_bound=1.0,
        parameter_bound=4.0,
        step=1.0,
        backtracks=32,
        max_patches=10000,
        max_connections=1000000,
        max_inputs=1000000,
    ):
        config = {
            "seed": integer(seed, "seed"),
            "settle_budget": integer(settle_budget, "settle_budget"),
        }
        for name, value in (
            ("fan_in", fan_in),
            ("backtracks", backtracks),
            ("max_patches", max_patches),
            ("max_connections", max_connections),
            ("max_inputs", max_inputs),
        ):
            config[name] = integer(value, name, 1)
        for name, value in (
            ("initial_scale", initial_scale),
            ("tolerance", tolerance),
            ("state_prior", state_prior),
            ("parameter_prior", parameter_prior),
            ("state_bound", state_bound),
            ("parameter_bound", parameter_bound),
            ("step", step),
        ):
            config[name] = number(value, name)
            if config[name] <= 0:
                raise ValueError(f"{name} must be positive")
        if config["initial_scale"] > config["parameter_bound"]:
            raise ValueError("initial_scale must not exceed parameter_bound")
        self._config = MappingProxyType(config)
        self._owner = object()
        self._nodes = {}
        self._inputs = []
        self._populations = []
        self._outputs = []
        self._built = False

    @property
    def config(self):
        """Read-only resolved construction and numerical configuration."""
        return self._config

    def _name(self, name, prefix):
        if self._built:
            raise ValueError("A built layout is frozen; create a new Cortex")
        if name is None:
            index = 1
            while f"{prefix}{index}" in self._nodes:
                index += 1
            name = f"{prefix}{index}"
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("Names must be nonempty strings of at most 256 characters")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Names must be valid UTF-8 text") from error
        if name in self._nodes:
            raise ValueError(f"Duplicate layout name: {name}")
        return name

    def _sources(self, values, types):
        if isinstance(values, types):
            values = (values,)
        try:
            values = tuple(islice(values, len(self._nodes) + 1))
        except TypeError as error:
            raise ValueError("Connections require layout references") from error
        if len(values) > len(self._nodes):
            raise ValueError("Too many source references for this Cortex")
        for value in values:
            if (
                not isinstance(value, types)
                or not isinstance(value.name, str)
                or self._nodes.get(value.name) is not value
            ):
                raise ValueError(
                    "Connections must reference existing nodes in this Cortex"
                )
        if len(set(values)) != len(values):
            raise ValueError("Duplicate source reference")
        return values

    def input(self, name, *, shape):
        """Add a sensor with declared shape; samples stay fixed during a solve."""
        name, shape = self._name(name, "input"), _shape(shape)
        if (
            math.prod(shape) + sum(i.size for i in self._inputs)
            > self.config["max_inputs"]
        ):
            raise ValueError("Input sample budget exceeded")
        node = Input(name, shape, self._owner)
        self._inputs.append(node)
        self._nodes[name] = node
        return node

    def _population(self, name, patches, inputs, observes):
        patches = integer(patches, "patches", 1)
        if (
            patches + sum(p.patches for p in self._populations)
            > self.config["max_patches"]
        ):
            raise ValueError("Processing patch budget exceeded")
        node = Population(name, patches, inputs, observes, self._owner)
        self._populations.append(node)
        self._nodes[name] = node
        return node

    def column(self, name=None, *, patches, inputs=()):
        """Add processing patches reading sensor or represented data ports."""
        name = self._name(name, "column")
        inputs = self._sources(inputs, (Input, Population))
        return self._population(name, patches, inputs, ())

    def observer(self, name=None, *, patches, inputs=(), observes):
        """Add the same patches reading live states and exact prediction errors.

        Observed populations must already exist, so residual readback has an
        acyclic definition. Its energy feedback acts on lower populations in
        the same joint solve; it is not a post-processing mode switch.
        """
        name = self._name(name, "observer")
        inputs = self._sources(inputs, (Input, Population))
        observes = self._sources(observes, (Population,))
        if not observes:
            raise ValueError("An observer must observe at least one population")
        return self._population(name, patches, inputs, observes)

    def output(self, name, *, shape, reads, indices=None):
        """Expose selected patch coordinates; default indices start at zero."""
        name, shape = self._name(name, "output"), _shape(shape)
        (reads,) = self._sources((reads,), (Population,))
        count = math.prod(shape)
        if count > reads.patches:
            raise ValueError("Output size exceeds its source population")
        if indices is None:
            indices = tuple(range(count))
        else:
            try:
                indices = tuple(
                    integer(i, "output index") for i in islice(indices, count + 1)
                )
            except TypeError as error:
                raise ValueError(
                    "Output indices must be an integer sequence"
                ) from error
        if len(indices) != count or any(i >= reads.patches for i in indices):
            raise ValueError("Output indices do not match shape/source")
        if len(set(indices)) != len(indices):
            raise ValueError("Output indices must be distinct")
        node = Output(name, shape, reads, indices, self._owner)
        self._outputs.append(node)
        self._nodes[name] = node
        return node

    def build(self):
        """Resolve sparse wiring once and construct one jointly settling brain."""
        return self._compile(self.config["max_connections"])

    def _compile(self, edge_limit):
        if self._built or not self._populations or not self._outputs:
            raise ValueError(
                "Build requires an unbuilt layout with patches and outputs"
            )
        rng = random.Random(self.config["seed"])
        input_ranges, population_ranges = {}, {}
        n_inputs = n_patches = 0
        for source in self._inputs:
            input_ranges[source.name] = range(n_inputs, n_inputs + source.size)
            n_inputs += source.size
        for population in self._populations:
            population_ranges[population.name] = range(
                n_patches, n_patches + population.patches
            )
            n_patches += population.patches
        edges, seen = [], set()
        for population in self._populations:
            sources = [
                ("input" if isinstance(s, Input) else "state", s)
                for s in population.inputs
            ]
            sources += [
                (kind, s) for s in population.observes for kind in ("state", "residual")
            ]
            targets = population_ranges[population.name]
            source_ports = set()
            for kind, source in sources:
                if (kind, source.name) in source_ports:
                    continue
                source_ports.add((kind, source.name))
                ranges = input_ranges if kind == "input" else population_ranges
                indices = ranges[source.name]
                fan_in = min(
                    len(indices),
                    max(self.config["fan_in"], math.ceil(len(indices) / len(targets))),
                )
                if len(edges) + len(targets) * fan_in > edge_limit:
                    raise ValueError("Connection budget exceeded")
                indices = list(indices)
                rng.shuffle(indices)
                for local_target, target in enumerate(targets):
                    for slot in range(fan_in):
                        source_index = indices[
                            (local_target * fan_in + slot) % len(indices)
                        ]
                        edge = (kind, source_index, target)
                        if edge not in seen:
                            seen.add(edge)
                            edges.append(edge)
        graph = _repair.Graph(n_inputs, n_patches, tuple(edges))
        fan_counts = [0] * n_patches
        for _, _, target in edges:
            fan_counts[target] += 1
        scale = self.config["initial_scale"]
        weights = tuple(
            rng.uniform(-1.0, 1.0) * scale / math.sqrt(fan_counts[t])
            for _, _, t in edges
        )
        layout = {
            "inputs": [{"name": i.name, "shape": i.shape} for i in self._inputs],
            "populations": [
                {
                    "name": p.name,
                    "patches": p.patches,
                    "inputs": [s.name for s in p.inputs],
                    "observes": [s.name for s in p.observes],
                }
                for p in self._populations
            ],
            "outputs": [
                {
                    "name": o.name,
                    "shape": o.shape,
                    "reads": o.reads.name,
                    "indices": o.indices,
                }
                for o in self._outputs
            ],
        }
        brain = Brain(self, graph, weights, input_ranges, population_ranges, layout)
        self._built = True
        return brain


class Brain:
    """A compiled layout with private parameters, live state and atomic admission.

    Construct with ``Cortex.build`` or ``Brain.from_snapshot``. Query methods
    freeze parameters; ``step`` can retain qualified live state. ``observe``
    jointly repairs live state and local relations under actual output witnesses,
    then commits only a qualified complete proposal. This is supervised witness
    admission, not an implemented reward/temporal-credit algorithm.
    """

    def __init__(
        self, builder, graph, weights, input_ranges, population_ranges, layout
    ):
        self._config = MappingProxyType(dict(builder.config))
        self._graph = graph
        self._inputs = tuple(builder._inputs)
        self._populations = tuple(builder._populations)
        self._outputs = tuple(builder._outputs)
        self._input_ranges = input_ranges
        self._population_ranges = population_ranges
        self._layout = canonical(layout)
        self._weights = tuple(weights)
        self._biases = (0.0,) * graph.n_patches
        self._state = (0.0,) * graph.n_patches
        self._event_id = -1
        self._event_digest = None
        self._admissions = 0
        self._fingerprint = hashlib.sha256(
            canonical(
                {"layout": layout, "config": dict(self.config), "edges": graph.edges}
            ).encode()
        ).hexdigest()

    @property
    def config(self):
        """Read-only resolved configuration, bound into checkpoints."""
        return self._config

    @property
    def graph(self):
        """Immutable topology with input, state and residual connection kinds."""
        return self._graph

    @property
    def state(self):
        """Immutable current live patch values, in layout declaration order."""
        return self._state

    @property
    def weights(self):
        """Immutable local relation coefficients, aligned with ``graph.edges``."""
        return self._weights

    @property
    def biases(self):
        """Immutable local prediction offsets, one per patch."""
        return self._biases

    def _mapping(self, supplied, expected, *, partial=False):
        if not isinstance(supplied, Mapping):
            raise ValueError("Values must map layout names or references to samples")
        expected = {node.name: node for node in expected}
        result = {}
        for key, value in supplied.items():
            name = key if isinstance(key, str) else getattr(key, "name", None)
            if (
                not isinstance(name, str)
                or name not in expected
                or (not isinstance(key, str) and key is not expected[name])
            ):
                raise ValueError("Unknown or foreign layout reference")
            if name in result:
                raise ValueError("Duplicate named values")
            result[name] = value
        if not partial and result.keys() != expected.keys():
            raise ValueError("Supply every declared sensor exactly once")
        return result

    def _arguments(self, inputs, targets=None, interventions=None):
        supplied = self._mapping(inputs, self._inputs)
        flat = tuple(
            v
            for source in self._inputs
            for v in _values(supplied[source.name], source.shape, source.name)
        )
        clamps = {}

        def add(indices, values):
            for index, value in zip(indices, values, strict=True):
                if abs(value) > self.config["state_bound"]:
                    raise ValueError("Clamp exceeds state_bound")
                if index in clamps and clamps[index] != value:
                    raise ValueError(
                        "Conflicting clamps alias the same processing patch"
                    )
                clamps[index] = value

        if targets is not None:
            supplied = self._mapping(targets, self._outputs, partial=True)
            for output in self._outputs:
                if output.name in supplied:
                    indices = tuple(
                        self._population_ranges[output.reads.name][i]
                        for i in output.indices
                    )
                    add(
                        indices,
                        _values(supplied[output.name], output.shape, output.name),
                    )
        if interventions is not None:
            supplied = self._mapping(interventions, self._populations, partial=True)
            for population in self._populations:
                if population.name in supplied:
                    add(
                        self._population_ranges[population.name],
                        _values(
                            supplied[population.name],
                            (population.patches,),
                            population.name,
                        ),
                    )
        return flat, clamps

    def _solve(self, inputs, clamps, *, learn, budget):
        config = self.config
        result = _repair.settle(
            self.graph,
            inputs,
            self._state,
            self._weights,
            self._biases,
            clamps=clamps,
            learn=learn,
            budget=config["settle_budget"]
            if budget is None
            else integer(budget, "budget"),
            tolerance=config["tolerance"],
            state_prior=config["state_prior"],
            parameter_prior=config["parameter_prior"],
            state_bound=config["state_bound"],
            parameter_bound=config["parameter_bound"],
            step=config["step"],
            backtracks=config["backtracks"],
        )
        result["outputs"] = {
            output.name: tuple(
                result["state"][self._population_ranges[output.reads.name][i]]
                for i in output.indices
            )
            for output in self._outputs
        }
        return result

    def settle(self, inputs, *, targets=None, interventions=None, budget=None):
        """Query a full coupled solve without changing live or durable state.

        Optional target/intervention clamps are hypothetical diagnostics here;
        they never become learning evidence. Outputs are flat tuples in declared
        shape order. Inspect ``qualified`` before using them.
        """
        flat, clamps = self._arguments(inputs, targets, interventions)
        return self._solve(flat, clamps, learn=False, budget=budget)

    def predict(self, inputs, *, budget=None):
        """Return qualified output values without admitting experience."""
        result = self.settle(inputs, budget=budget)
        if not result["qualified"]:
            raise SettlementError(f"Population solve refused: {result['reason']}")
        return result["outputs"]

    def step(self, inputs, *, budget=None):
        """Continue live activity with frozen relations; refuse partial state."""
        result = self.settle(inputs, budget=budget)
        if result["qualified"]:
            self._state = tuple(result["state"])
        return {**result, "accepted": result["qualified"]}

    def observe(self, inputs, targets, *, event_id=None, budget=None):
        """Jointly repair and atomically retain an actual input/target experience.

        Targets clamp at least one declared output. Ordered nonnegative event
        IDs recognize an identical latest retry; changed or older IDs are
        rejected. Omitting the ID allocates the next one only on acceptance.
        A refusal leaves live state, parameters and event ownership unchanged.
        """
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("Observation requires at least one actual output target")
        flat, clamps = self._arguments(inputs, targets)
        event_id = (
            self._event_id + 1 if event_id is None else integer(event_id, "event_id")
        )
        digest = hashlib.sha256(
            canonical([flat, sorted(clamps.items())]).encode()
        ).hexdigest()
        if event_id <= self._event_id:
            if event_id == self._event_id and digest == self._event_digest:
                return {
                    "accepted": False,
                    "qualified": True,
                    "duplicate": True,
                    "event_id": event_id,
                }
            raise ValueError(
                "Event is older than, or conflicts with, the latest admitted event"
            )
        result = self._solve(flat, clamps, learn=True, budget=budget)
        if result["qualified"]:
            self._state = tuple(result["state"])
            self._weights = tuple(result["weights"])
            self._biases = tuple(result["biases"])
            self._event_id, self._event_digest = event_id, digest
            self._admissions += 1
        return {
            **result,
            "accepted": result["qualified"],
            "duplicate": False,
            "event_id": event_id,
        }

    def inspect(self):
        """Return owned layout data, actual graph counts and readback roles."""
        import json

        layout = json.loads(self._layout)
        for population in layout["populations"]:
            population["role"] = "observer" if population["observes"] else "processing"
            population["indices"] = tuple(self._population_ranges[population["name"]])
        used = {source for kind, source, _ in self.graph.edges if kind == "input"}
        return {
            **layout,
            "config": dict(self.config),
            "patches": self.graph.n_patches,
            "input_samples": self.graph.n_inputs,
            "connections": len(self.graph.edges),
            "edges": self.graph.edges,
            "observed_fields": ("state", "prediction_error"),
            "sensor_coverage": len(used),
            "fingerprint": self._fingerprint,
            "implementation": dict(IMPLEMENTATION),
            "admissions": self._admissions,
            "last_event_id": self._event_id,
        }

    def snapshot(self):
        """Serialize the entire layout and continuation state as validated JSON."""
        import json

        result = canonical(
            {
                "schema": SCHEMA,
                "implementation": dict(IMPLEMENTATION),
                "config": dict(self.config),
                "layout": json.loads(self._layout),
                "fingerprint": self._fingerprint,
                "state": self._state,
                "weights": self._weights,
                "biases": self._biases,
                "event_id": self._event_id,
                "event_digest": self._event_digest,
                "admissions": self._admissions,
            }
        )
        if len(result.encode()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint exceeds text-size budget")
        return result

    @classmethod
    def from_snapshot(cls, text):
        """Reconstruct the declared graph and validate all state before use."""
        data = strict_json(text, MAX_CHECKPOINT_BYTES)
        fields = {
            "schema",
            "implementation",
            "config",
            "layout",
            "fingerprint",
            "state",
            "weights",
            "biases",
            "event_id",
            "event_digest",
            "admissions",
        }
        if (
            not isinstance(data, dict)
            or set(data) != fields
            or data["schema"] != SCHEMA
        ):
            raise ValueError("Unsupported population checkpoint")
        if data["implementation"] != IMPLEMENTATION:
            raise ValueError("Checkpoint repair/layout implementation mismatch")
        try:
            builder = Cortex(**data["config"])
            if set(data["config"]) != set(builder.config):
                raise ValueError("Checkpoint must include the complete configuration")
            nodes = {}
            layout = data["layout"]
            if set(layout) != {"inputs", "populations", "outputs"}:
                raise ValueError("Invalid layout fields")
            if any(
                not isinstance(data[key], list)
                for key in ("state", "biases", "weights")
            ):
                raise ValueError("Checkpoint parameters must be arrays")
            patch_count = sum(
                integer(record["patches"], "patches", 1)
                for record in layout["populations"]
            )
            if patch_count != len(data["state"]) or patch_count != len(data["biases"]):
                raise ValueError("State length does not match declared patch count")
            for record in layout["inputs"]:
                node = builder.input(**record)
                nodes[node.name] = node
            for record in layout["populations"]:
                if set(record) != {"name", "patches", "inputs", "observes"}:
                    raise ValueError("Invalid population fields")
                inputs = tuple(nodes[n] for n in record["inputs"])
                observes = tuple(nodes[n] for n in record["observes"])
                method = builder.observer if observes else builder.column
                extra = {"observes": observes} if observes else {}
                node = method(
                    record["name"], patches=record["patches"], inputs=inputs, **extra
                )
                nodes[node.name] = node
            for record in layout["outputs"]:
                if set(record) != {"name", "shape", "reads", "indices"}:
                    raise ValueError("Invalid output fields")
                builder.output(
                    record["name"],
                    shape=record["shape"],
                    reads=nodes[record["reads"]],
                    indices=record["indices"],
                )
            brain = builder._compile(
                min(builder.config["max_connections"], len(data["weights"]))
            )
            if brain._fingerprint != data["fingerprint"]:
                raise ValueError("Checkpoint layout/configuration fingerprint mismatch")
            initial_weights = brain.weights
            for key, length, bound in (
                ("state", brain.graph.n_patches, brain.config["state_bound"]),
                ("weights", len(brain.graph.edges), brain.config["parameter_bound"]),
                ("biases", brain.graph.n_patches, brain.config["parameter_bound"]),
            ):
                raw = data[key]
                if not isinstance(raw, list) or len(raw) != length:
                    raise ValueError(f"Invalid {key} length")
                values = tuple(number(v, key) for v in raw)
                if any(abs(v) > bound for v in values):
                    raise ValueError(f"Invalid {key} bound")
                setattr(brain, "_" + key, values)
            event_id = integer(data["event_id"], "event_id", -1)
            admissions = integer(data["admissions"], "admissions")
            digest = data["event_digest"]
            if (admissions == 0) != (event_id == -1) or admissions > event_id + 1:
                raise ValueError("Invalid event ownership")
            if event_id == -1:
                if digest is not None:
                    raise ValueError("Unexpected event digest")
                if brain.weights != initial_weights or any(brain.biases):
                    raise ValueError(
                        "Retained parameters changed without an admitted event"
                    )
            elif (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid event digest")
            brain._event_id, brain._event_digest, brain._admissions = (
                event_id,
                digest,
                admissions,
            )
            return brain
        except (KeyError, TypeError, OverflowError, AttributeError) as error:
            raise ValueError("Malformed population checkpoint") from error

    def restore(self, text):
        """Atomically replace continuation state for this exact layout/config."""
        proposal = type(self).from_snapshot(text)
        if proposal._fingerprint != self._fingerprint:
            raise ValueError("Cannot restore a different population layout")
        self._state, self._weights, self._biases = (
            proposal.state,
            proposal.weights,
            proposal.biases,
        )
        self._event_id = proposal._event_id
        self._event_digest = proposal._event_digest
        self._admissions = proposal._admissions
