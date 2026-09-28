# API reference

Cadence constructs Deep Recursive Settlement Networks from populations of
processing patches. Ordinary data connections and recursive observation share
one jointly repaired state. Public exports are `Cortex`, `Brain`, `Input`,
`Population`, `Output`, `SettlementError`, `bootstrap` and `__version__`.

```python
from cadence import Cortex, Brain, Input, Population, Output, SettlementError, bootstrap, __version__
```

Use the package-level imports above. `Brain` and `SettlementError` live in
`brain.py`, `Cortex` in `cortex.py`, `Population` in `column.py`, and `Input` and
`Output` with boundary shape/value validation in `ports.py`. Numerical repair
and numeric/JSON validation remain private in `_repair.py` and `_validation.py`.
Private `_tensor.py` supplies optional device execution of the same analytic
repair law, with final qualification by the float64 reference engine.
`bootstrap.py` orchestrates example replay and unclamped checks through the
existing brain methods; it adds no solver or phase state.
See the [quickstart](QUICKSTART.md) for a first example and the
[architecture guide](DRSN.md) for equations and layout patterns.

## Cortex: declare a layout

```text
Cortex(
    *,
    seed=0,
    fan_in=None,
    initial_scale=0.3,
    settle_budget=2048,
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
    device="python",
    dtype=None,
)
```

All arguments are keyword-only. Integer parameters reject booleans; `fan_in`
also accepts `None` for full connectivity. Real parameters must be finite and
strictly positive. Defaults are starting points
for small numerical and learning examples, not universal task settings.

| Parameter | Contract |
| --- | --- |
| `seed` | Nonnegative integer controlling sampled wiring and initial relation coefficients. Reproduction also requires the same layout, configuration and implementation. |
| `fan_in` | `None` (default) connects every coordinate of each declared source to every destination patch. A positive integer requests sparse sampling: a minimum count per destination, capped at source size and raised to cover the complete source across the population. Applies separately to each source and signal kind. |
| `initial_scale` | Positive initial weight scale, no greater than `parameter_bound`. Each weight is sampled uniformly in `[-initial_scale, initial_scale]` and divided by the square root of its target's total incoming connection count. Biases and live state start at zero. |
| `settle_budget` | Nonnegative maximum accepted repair sweeps per solve. Zero can qualify an already stationary state. A sweep may evaluate several rejected proposals. |
| `tolerance` | Positive maximum complete projected-gradient residual for qualification. An absolute numerical threshold, not prediction accuracy. |
| `state_prior` | Positive coefficient of the quadratic activity penalty. Changes the model's preferred states, not merely solver speed. |
| `parameter_prior` | Positive coefficient anchoring weights and biases to their pre-experience values during `observe`. This anchor stays fixed throughout the experience solve. |
| `state_bound` | Positive absolute bound on processing-patch states and output/intervention clamps. Sensor values are not clipped to it. |
| `parameter_bound` | Positive absolute bound on weights and biases; repair projects eligible parameter coordinates into this box. |
| `step` | Positive initial and fallback projected-gradient step. Subsequent trials estimate a scalar step from the previous accepted displacement and gradient change. Unsafe estimates fall back to this value; growth is bounded by the available backtracking budget. |
| `backtracks` | Positive maximum line-search proposals per sweep. Rejection halves the proposed step. |
| `max_patches` | Positive integer cap on processing patches across all populations. Sensor samples and output aliases do not consume this count. |
| `max_connections` | Positive integer cap on compiled directed input, state and error-readback connections. |
| `max_inputs` | Positive integer cap on scalar samples across all sensors. |
| `device` | `"python"` (default) uses the standard-library reference engine. `"cpu"`, `"mps"`, `"cuda"` and `"cuda:N"` select optional PyTorch tensor execution. `"cuda"` resolves to `"cuda:0"`; `N` is a nonnegative device index. Availability is checked on the first solve. |
| `dtype` | `None` resolves to `"float32"` for `"mps"`, otherwise `"float64"`. Explicit `"float32"` or `"float64"` selects tensor proposal precision. `"python"` requires `"float64"`; `"mps"` requires `"float32"`. Final qualification always uses Python float64 with the original data. |

`cortex.config` is a read-only mapping of resolved settings. Construction caps
count graph elements; they do not bound actual RAM or runtime. Larger inputs
increase coverage cost even when `fan_in` is small. Tensor execution requires
`pip install "cadence-net[gpu]"`. Declaring, building or loading a brain does
not import PyTorch or allocate GPU resources. The first solve raises
`ImportError` if PyTorch is missing or `ValueError` for an unavailable device;
it never silently selects a different device. Device/precision affect numerical
trajectories and cost, not the declared patch law. See [acceleration](ACCELERATION.md).

### Layout methods

| Method | Return and behavior |
| --- | --- |
| `input(name, *, shape)` | An `Input` boundary whose supplied values remain fixed for the complete solve. |
| `column(name=None, *, patches, inputs=())` | A `Population` containing exactly `patches` processing patches, a positive integer. `inputs` accepts existing sensor or population references. |
| `observer(name=None, *, patches, inputs=(), observes)` | The same patch primitive with at least one observed population. Reads its live states and exactly recomputed prediction errors; may also receive ordinary `inputs`. |
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
Two scalar outputs reading the same population both default to index zero.
Use explicit `indices=(0,)` and `indices=(1,)` for independent named controls,
or one vector output with `shape=2`.

With default full connectivity, each destination patch reads every coordinate
of every source it declares. Explicit sparse wiring guarantees only aggregate
coverage across the population. A selected output can miss information read by
other, unconnected patches. A sensor with no connections remains unobserved.
`inspect()` reports actual edges, aggregate coverage and structural sensor
coverage for each output coordinate.
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

Construct with `Cortex.build()` or `Brain.from_snapshot`; the `Brain`
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
| `Brain.from_snapshot(text, *, device=None, dtype=None)` | Class method validating the complete original snapshot before constructing a new brain. Optional execution overrides retain arrays and witness identity while changing configuration/fingerprint. See [checkpoints](#checkpoints). |
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
witness provenance. Outputs returned by `observe` equal their supplied target
clamps, so measure learning
using subsequent predictions without those clamps.

`event_id` is an ordered nonnegative integer that must be serializable by the
runtime's JSON integer encoder. Excessive digit counts raise `ValueError`
before solving or admitting experience. Omission selects the latest
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
| `energy_history` | With the Python engine, initial energy followed by each accepted proposal's energy. Tensor execution records its approximate device trajectory and any reference refinement; do not treat that combined trace as a float64 monotonicity certificate. |
| `work` | Algorithmic work counts, including attempted work on refused solves. Tensor and reference qualification/refinement work are both counted. |
| `execution` | Present for tensor solves: selected device/precision, tensor library version and work split; see below. |

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

Tensor `execution` contains `device`, `dtype`, `torch` (library version),
`tensor_sweeps`, `reference_sweeps`, `reference_evaluations` and
`reference_restart`. The last flag reports whether an unacceptable float64
energy increase discarded the device candidate and restarted from the original
coordinates. Reference checking may refine a device candidate, using only the
remaining total sweep allowance. For a positive budget, float32 device repair
uses at most `max(1, budget // 2)` accepted sweeps, reserving at least half of
budgets of two or more for reference refinement. Float64 may use the full
allowance. The device stopping hint `max(tolerance, 64 * dtype_epsilon)` never
relaxes final admission tolerance. `sweeps` includes both stages. Returned
`energy`, `predictions`, `errors`, `stationarity` and `qualified` come from the
reference check using original inputs, exact clamps, original learning anchors
and original frozen query parameters. Device proposals may follow a different
trajectory; matching device results bit for bit is not promised.

### Inspector result

`inspect()` returns layout `inputs`, `populations` and `outputs`, plus `config`,
`patches`, `input_samples`, `connections`, `edges`, `observed_fields`,
`sensor_coverage`, `output_connected_patches`, `fingerprint`, `implementation`, `admissions` and
`last_event_id`. Population records add `role` and global patch `indices`.
`observed_fields` is `("state", "prediction_error")`. `sensor_coverage` counts
unique input coordinates attached anywhere; it does not certify their influence
on a selected output. Each output record adds `sensor_coverage_by_coordinate`,
a flat tuple of distinct reachable sensor-coordinate counts in output order.
`output_connected_patches` counts processing patches in any output's coupled
component. State and residual connections join components in both directions;
shared fixed inputs do not join otherwise independent patches. These are
structural possibilities, not guarantees of causal influence: zero weights,
saturation or learned cancellation can suppress a path. `last_event_id` is `-1`
before any admission. Editing inspector copies does not modify the brain.

## Numerical and learning contract

Each patch computes `p = tanh(b + sum(weight * signal))` and error `e = x - p`.
Query energy is `sum(e²)/2 + state_prior * sum(x²)/2`. Learning adds
`parameter_prior * sum((parameter - pre_experience_parameter)²)/2`, making
weights and biases eligible alongside unclamped live states.

Repair is synchronized projected-gradient descent with a scalar secant step
and Armijo backtracking. A fully stationary final proposal may finish within
eight energy ulps when rounding prevents sufficient decrease; the requested
stationarity tolerance is unchanged. The energy and analytic gradient do
not change when the step adapts. See the specification for the update formula.
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
`predict` for a valid but unqualified solve. Its message includes the refusal
reason, stationarity, tolerance and sweep count; use `settle` for full diagnostics.

The learning operation is supervised witness admission. Reward credit assignment,
automatic episodic retrieval, planning policies and learned structural growth
are not provided by `observe`. Useful perception, behavior and benefits from
observation depth require separate application tests.

## bootstrap

```text
bootstrap(
    brain,
    examples,
    *,
    checks,
    max_error,
    epochs=20,
    seed=0,
    budget=None,
)
```

Convenience orchestration for the **bootstrapping phase**, returning a plain
report and admitting examples to the supplied `Brain`. The **live phase** uses
that same brain, with continued `observe` calls when actual witnesses arrive.
These are application phases; no solver mode changes at the boundary.

| Argument | Contract |
| --- | --- |
| `brain` | Existing `Brain` to bootstrap. Earlier admitted experience is preserved. |
| `examples` | Nonempty finite sequence of `(inputs, targets)` pairs, with the same named/owned-handle mappings and shapes as `observe`. Each pair needs at least one target. Rows may repeat; generators are not accepted. |
| `checks` | Nonempty sequence in the same format, used only for unclamped evaluation. These cases influence stopping and are development data; reserve a separate final test. |
| `max_error` | Required finite nonnegative maximum absolute error in encoded output units. This is an application error limit, separate from solver `tolerance`. |
| `epochs` | Nonnegative maximum complete replay passes, default 20. Zero evaluates the current brain without learning. |
| `seed` | Nonnegative integer for a private RNG that shuffles example indices each epoch. Does not alter Python's global random state. |
| `budget` | Nonnegative solve-budget override applied to every admission and check; `None` uses the brain's configured ceiling. |

All options, examples and checks are validated and samples copied before any
solve or admission. Inputs and targets are not normalized automatically.
Malformed examples fail with their collection/index and leave the brain
unchanged. Numerical exceptions during subsequent solves propagate as in the
underlying brain methods; earlier accepted experiences remain committed.

Before the first epoch and after each complete epoch, pure `settle` calls score
the examples (recall), then the checks, with **no targets supplied to the solve**.
Only actual targeted state coordinates are compared; agreeing output aliases
count once within a pair. Every query must qualify and both maximum errors must
meet `max_error` to pass. An already satisfactory brain returns without replay.
A refusal stops immediately; unqualified outputs never contribute a score.

Each admitted presentation uses the ordinary `observe` rule and a fresh automatic
event ID. Replay is repeated supervised experience, not new environmental data.
The helper is not a batch transaction: a later refusal keeps earlier admitted
examples. Starting a new helper call starts a new shuffle sequence and report;
it does not resume an interrupted helper cursor. For external retry identities,
streaming data or custom environment metrics, use `observe` and `settle` directly.

| Report field | Meaning |
| --- | --- |
| `options` | Requested `max_error`, epoch allowance and shuffle `seed`, plus the resolved integer solve `budget`. Reuse these with the same starting checkpoint and data to repeat the call. |
| `passed` | All current recall/check queries qualified and both errors met the declared limit. Applies only to the supplied cases. |
| `reason` | `"passed"`, `"epochs"` (allowance exhausted), or `"refused"`. |
| `epochs` | Number of fully admitted replay passes; excludes a partly completed pass. |
| `presentations`, `accepted` | Attempted and admitted example presentations in this call, including replay. |
| `examples`, `checks` | Counts of supplied rows, not deduplicated experiences. |
| `history` | Epoch-zero assessment and an assessment after each complete epoch. Each entry has `epoch`, `recall` and `checks`. |
| `work` | Summed solver work counters over admissions and all checks, including refused solves. Does not count Python preprocessing/bookkeeping. |
| `failure` | `None`, or `stage` (`"observe"`, `"recall"`, `"checks"`), original row `index`, solver `reason`, and `stationarity`. |

Each assessment metric contains `evaluated`, `qualified` and `max_error`.
A refused query sets that collection's `max_error` to `None`, never a score
over only the successful subset. A collection not reached after refusal has
metric `None`. Checkpoint the brain and retain the report plus preprocessing
alongside it; the report is not stored in the brain's snapshot. Exact replay
also needs the **starting** checkpoint and the original examples/checks. The
final checkpoint is the continuation used in the live phase.

## Checkpoints

Checkpoints contain schema, full configuration and layout, graph fingerprint,
live state, weights, biases, admission count and latest event identity. They
bind the exact source hashes of `brain.py`, `cortex.py`, `column.py`, `ports.py`,
`_repair.py`, `_tensor.py` and `_validation.py`. Loading rejects previous source sets or
different hashes even if package version labels match; compatibility is
therefore stricter than version compatibility.

Loading checks structure, configuration, deterministic topology, array lengths,
finite values, bounds and event-ownership consistency. Restoration builds and
validates a complete proposal before replacing continuation. Newly loaded brains
own different handles; address them by names when using reconstructed layouts.
`restore` retains the receiving brain's existing handles and requires the
complete configuration, including device and dtype, to match.

`Brain.from_snapshot(text, device=..., dtype=...)` first validates the complete
original snapshot against its saved configuration and current implementation.
An override cannot bypass an invalid fingerprint, source mismatch or malformed
array. Only then are explicit execution settings applied. With both overrides
omitted, saved settings are retained. Supplying `device` without `dtype` uses
the new device's default precision; supplying only `dtype` preserves the saved
device. Arrays, bounds, topology and event ownership are preserved exactly;
execution changes alter the configuration fingerprint. Choosing float32 does
not round stored checkpoint arrays: conversion occurs for device proposals
on the next solve. Hardware availability remains a first-solve check.

JSON is data, not executable deserialization. A structurally valid checkpoint
does not authenticate its maker, prove its witness history or certify current
stationarity. Check external provenance separately and solve before acting.
The 32 MiB text limit does not bound peak decoding or construction memory.
