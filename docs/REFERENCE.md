# API reference

Cadence constructs Deep Recursive Settlement Networks from populations of
processing patches. Ordinary data connections and recursive observation share
one jointly repaired state. Public exports are `Cortex`, `Brain`, `Input`,
`Population`, `Output`, `SettlementError` and `__version__`.

```python
from cadence import Cortex, Brain, Input, Population, Output, SettlementError, __version__
```

The same classes are available from `cadence.drsn`. See the
[quickstart](QUICKSTART.md) for a first example and the
[architecture guide](DRSN.md) for equations and layout patterns.

## Cortex: declare a layout

```text
Cortex(
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
)
```

All arguments are keyword-only. Integer parameters reject booleans; real
parameters must be finite and strictly positive. Defaults are starting points
for small numerical and learning examples, not universal task settings.

| Parameter | Contract |
| --- | --- |
| `seed` | Nonnegative integer controlling sparse wiring and initial relation coefficients. Reproduction also requires the same layout, configuration and implementation. |
| `fan_in` | Positive minimum source-coordinate sample count per destination patch, capped at source size and raised when necessary to cover the complete source across that destination population. Applies separately to each source and signal kind. |
| `initial_scale` | Positive initial weight scale, no greater than `parameter_bound`. Each weight is sampled uniformly in `[-initial_scale, initial_scale]` and divided by the square root of its target's total incoming connection count. Biases and live state start at zero. |
| `settle_budget` | Nonnegative maximum accepted repair sweeps per solve. Zero can qualify an already stationary state. A sweep may evaluate several rejected proposals. |
| `tolerance` | Positive maximum complete projected-gradient residual for qualification. An absolute numerical threshold, not prediction accuracy. |
| `state_prior` | Positive coefficient of the quadratic activity penalty. Changes the model's preferred states, not merely solver speed. |
| `parameter_prior` | Positive coefficient anchoring weights and biases to their pre-experience values during `observe`. This anchor stays fixed throughout the experience solve. |
| `state_bound` | Positive absolute bound on processing-patch states and output/intervention clamps. Sensor values are not clipped to it. |
| `parameter_bound` | Positive absolute bound on weights and biases; repair projects eligible parameter coordinates into this box. |
| `step` | Positive initial projected-gradient step tried anew each sweep. |
| `backtracks` | Positive maximum line-search proposals per sweep. Rejection halves the proposed step. |
| `max_patches` | Positive integer cap on processing patches across all populations. Sensor samples and output aliases do not consume this count. |
| `max_connections` | Positive integer cap on compiled directed input, state and error-readback connections. |
| `max_inputs` | Positive integer cap on scalar samples across all sensors. |

`cortex.config` is a read-only mapping of resolved settings. Construction caps
count graph elements; they do not bound actual RAM or runtime. Larger inputs
increase coverage cost even when `fan_in` is small.

### Layout methods

| Method | Return and behavior |
| --- | --- |
| `input(name, *, shape)` | An `Input` boundary whose supplied values remain fixed for the complete solve. |
| `column(name=None, *, patches, inputs=())` | A `Population` containing exactly `patches` processing patches, a positive integer. `inputs` accepts existing sensor or population references. |
| `observer(name=None, *, patches, inputs=(), observes=...)` | The same patch primitive with at least one observed population. Reads its live states and exactly recomputed prediction errors; may also receive ordinary `inputs`. |
| `output(name, *, shape, reads, indices=None)` | An `Output` exposing selected coordinates of one population, with no separate output network. |
| `build()` | A `Brain` with resolved sparse wiring. Requires at least one population and one output. A successful build freezes the layout; further construction or another build raises `ValueError`. |

Names are unique across node types, valid UTF-8 strings of 1–256 characters.
`None` generates a name using the node type and an unused positive integer.
References must be the actual handles returned by this `Cortex`; a matching
name or a handle from another layout does not establish ownership.

`inputs` and `observes` accept one handle or an iterable of distinct handles.
Sources must already exist. `inputs` reads sensor values or population states;
`observes` accepts populations only and additionally reads their errors.
Declaring a population in both fields does not duplicate its state connection.
Empty-input columns are permitted for internal-state controls.

`shape` accepts a positive integer or a tuple/list of at most eight positive
integer dimensions. `shape=()` denotes one scalar. Shape describes data layout,
not learned interpretation. Output size cannot exceed `reads.patches`.
`indices` selects distinct zero-based local coordinates from `reads`, in output
order, with length equal to output size. The default is `range(size)`.
Different outputs may alias the same patch.

Every coordinate of a connected source is covered across its destination
population; each destination patch need not read every coordinate. A sensor
with no connections remains unobserved. `inspect()` reports actual coverage.
Normalization, features and motor interpretation are supplied by the application.

### Immutable layout handles

Obtain these objects from the builder instead of constructing them directly.

| Class | Public fields and properties |
| --- | --- |
| `Input` | `name`, `shape`; `size` counts scalar samples. |
| `Population` | `name`, `patches`, `inputs`, `observes`; `role` is `"observer"` when `observes` is nonempty, otherwise `"processing"`. |
| `Output` | `name`, `shape`, `reads`, `indices`. |

`role` describes wiring. Useful self-observation still needs causal tests;
assigning a name does not demonstrate metacognitive ability. Population width
is `patches`; observation nesting is defined by `observes`.

## Brain: query, continue and learn

Construct with `Cortex.build()` or `Brain.from_snapshot(text)`; the `Brain`
constructor itself is an internal compilation interface. Use one serial owner
per brain. Methods do not provide thread synchronization. Admission and restore
are all-or-nothing under serial access, not concurrent database transactions.

| Method | Contract |
| --- | --- |
| `settle(inputs, *, targets=None, interventions=None, budget=None)` | Pure query returning a complete solve result. Optional output targets and population interventions are hypothetical clamps. Changes neither retained live state nor parameters. |
| `predict(inputs, *, budget=None)` | Pure query returning an output-name-to-flat-tuple mapping. Raises `SettlementError` if the solve does not qualify. |
| `step(inputs, *, budget=None)` | Repair activity with parameters frozen. Retain proposed state only if qualified. Returns the full result plus `accepted`. |
| `observe(inputs, targets, *, event_id=None, budget=None)` | Jointly repair state, weights and biases under at least one actual output witness. A qualified solve atomically retains state, parameters and event ownership; a refusal retains none of the proposal. |
| `inspect()` | Owned layout description, resolved graph counts, topology, observation roles and continuation metadata. |
| `snapshot()` | Complete continuation as a JSON string, limited to 32 MiB of UTF-8 text. |
| `Brain.from_snapshot(text)` | Class method validating and reconstructing a new brain from compatible JSON. |
| `restore(text)` | Validate before replacing this brain's continuation. Layout and configuration fingerprint must match. Returns `None`. |

`budget` overrides `settle_budget` for an attempted solve and must be a
nonnegative integer; `None` uses the configured budget. An identical latest-event
retry performs no solve and ignores `budget`. Queries begin at currently retained
live state. They are read-only, not automatic resets.

### Values, clamps and events

`inputs` maps every declared sensor exactly once, using names or that brain's
original `Input` handles. Values may be correctly nested lists/tuples or flat
lists/tuples with the declared element count. Array objects with `tolist()` are
accepted without a NumPy dependency; any exposed shape must match the declared
shape or its flat vector shape. Scalar boundaries receive scalar values.
All samples must be finite real numbers; booleans are rejected. No automatic
clipping or image/audio preprocessing is applied.

`targets` maps selected output names or owned `Output` handles to values of the
output shape. `interventions` maps selected population names or owned
`Population` handles to vectors of `population.patches` values. Clamps must lie
inside `state_bound`. Aliased output/intervention clamps must agree exactly on
shared patches or the call raises `ValueError`. Returned output values are
always flat tuples, including multidimensional outputs.

In `settle`, clamps express hypothetical queries and never become learning
witnesses. `observe` requires a nonempty target mapping and treats it as an
actual witness; it has no intervention argument. The application supplies
witness provenance. Training outputs equal their clamps, so measure learning
using subsequent predictions without those clamps.

`event_id` is an ordered nonnegative integer. Omission selects the latest
admitted identifier plus one; initially this is `0`. Gaps are allowed. The
latest accepted identifier may be retried with identical inputs and clamps,
performing no solve or second admission. Reusing it with different content, or
supplying an older identifier, raises `ValueError`. Refused proposals consume
no inferred identifier and retain no event content.

### Read-only properties

| Property | Value |
| --- | --- |
| `config` | Read-only resolved configuration mapping. |
| `state` | Tuple of current patch values, concatenated in population declaration order. |
| `weights` | Tuple of retained coefficients in `graph.edges` order. |
| `biases` | Tuple of retained prediction offsets in patch order. |
| `graph` | Immutable topology with `n_inputs`, `n_patches`, `edges`, `incoming` and `residual_order`. |

An edge is `(kind, source, target)`, where `kind` is `"input"`, `"state"` or
`"residual"`. Input sources index the flattened sensor vector; other sources
and all targets index the combined patch vector. `incoming` contains edge-index
tuples per target. `residual_order` orders recomputation of derived errors.
Edges expose read dependencies. Feedback uses derivatives of the same energy,
not a separately stored reverse edge.

### Solve results

`settle`, `step` and newly attempted `observe` calls return a dictionary:

| Key | Meaning |
| --- | --- |
| `outputs` | Names mapped to selected state tuples. Refused outputs are diagnostic proposals and must not drive actions. |
| `state`, `weights`, `biases` | Proposed final coordinate tuples. A pure query leaves the brain's properties unchanged. |
| `predictions` | Exactly recomputed `tanh` predictions, one per patch. |
| `errors` | Exact `state - prediction` values, one per patch. |
| `energy` | Final objective, including the fixed parameter anchor penalty during learning. |
| `stationarity` | Maximum absolute projected-gradient component over eligible coordinates. |
| `prediction_residual` | Largest absolute prediction error; may remain nonzero at qualified stationarity. |
| `qualified` | Whether a fresh final evaluation satisfies the stationarity tolerance. |
| `reason` | `"qualified"`, `"budget"` or `"line_search"`. |
| `sweeps` | Accepted repair sweeps. |
| `energy_history` | Initial energy followed by each accepted proposal's energy. |
| `work` | Algorithmic work counts, including attempted work on refused solves. |

`step` adds `accepted`, equal to `qualified`. A newly attempted `observe` adds
`accepted`, `duplicate=False` and `event_id`. An identical latest-event retry
returns only `accepted=False`, `qualified=True`, `duplicate=True` and `event_id`.
Its qualification refers to the prior admission; it is not a fresh prediction
or qualification of live state after intervening calls.

`work` contains `evaluations`, `patch_visits`, `edge_visits`, `proposals` and
`backtracks`. Evaluations count attempted energy/gradient computations,
including the final check. Visits count prediction/error and derivative
traversals, including partial work before numeric failure. Proposals count
line-search attempts; backtracks count rejected proposals, including the last
rejection when line search fails. These are not CPU instruction counts or
complete memory/latency measurements.

### Inspector result

`inspect()` returns layout `inputs`, `populations` and `outputs`, plus `config`,
`patches`, `input_samples`, `connections`, `edges`, `observed_fields`,
`sensor_coverage`, `fingerprint`, `implementation`, `admissions` and
`last_event_id`. Population records add `role` and global patch `indices`.
`observed_fields` is `("state", "prediction_error")`. `sensor_coverage` counts
unique input coordinates with a compiled connection. `last_event_id` is `-1`
before any admission. Editing inspector copies does not modify the brain.

## Numerical and learning contract

Each patch computes `p = tanh(b + sum(weight * signal))` and error `e = x - p`.
Query energy is `sum(e²)/2 + state_prior * sum(x²)/2`. Learning adds
`parameter_prior * sum((parameter - pre_experience_parameter)²)/2`, making
weights and biases eligible alongside unclamped live states.

Repair is synchronized projected-gradient descent with Armijo backtracking.
The residual is `z - clip(z - gradient, -bound, bound)` for each eligible
coordinate. Clamped states and frozen query parameters are excluded.
Derivatives include transitive error-readback influence. Observers and observed
populations participate in the same objective and final qualification, not a
sequence of independently settled answers.

Qualification establishes bounded projected stationarity, not zero disagreement,
a unique normal form, a global optimum or asynchronous confluence. Nonconvex
energy can retain different stationary points. Budget or line-search exhaustion
is a numerical refusal. Invalid arguments and nonrepresentable initial numeric
quantities raise `ValueError`. `SettlementError` is a `RuntimeError` used by
`predict` for a valid but unqualified solve.

The learning operation is supervised witness admission. Reward credit assignment,
automatic episodic retrieval, planning policies and learned structural growth
are not provided by `observe`. Useful perception, behavior and benefits from
observation depth require separate application tests.

## Checkpoints

Checkpoints contain schema, full configuration and layout, graph fingerprint,
live state, weights, biases, admission count and latest event identity. They
bind the exact source hashes of `drsn.py`, `_repair.py` and `_validation.py`.
Loading refuses different source implementations even if package version labels
match; compatibility is therefore stricter than version compatibility.

Loading checks structure, configuration, deterministic topology, array lengths,
finite values, bounds and event-ownership consistency. Restoration builds and
validates a complete proposal before replacing continuation. Newly loaded brains
own different handles; address them by names when using reconstructed layouts.
`restore` retains the receiving brain's existing handles.

JSON is data, not executable deserialization. A structurally valid checkpoint
does not authenticate its maker, prove its witness history or certify current
stationarity. Check external provenance separately and solve before acting.
The 32 MiB text limit does not bound peak decoding or construction memory.
