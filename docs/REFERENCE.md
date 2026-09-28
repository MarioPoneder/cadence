# API reference

The main entry points are `CorticalColumn` for a scalar stream and `Cortex`
for context-dependent outputs. Built-in feature maps make construction and
checkpoint reconstruction straightforward. The low-level port and finite-factor
APIs expose the repair machinery directly. See [Quickstart](QUICKSTART.md) for
complete dependency-free examples and [Element](ELEMENT.md) for equations.
`cadence.__version__` reports the installed package version.

## CorticalColumn

```text
CorticalColumn(height=1, *, decay=0.5, capacity=None, value_bound=8,
               settle_budget=256, tolerance=1e-11, damping=1.0,
               prior_weight=1, prior_shape=1.0, prior_rate=1.0,
               meta_shape=4.0, meta_rate=4.0, max_statistic_bits=16384)
```

| Parameter | Meaning and accepted range |
| --- | --- |
| `height` | Positive integer. One means belief plus precision observer; larger values add rate-observer stages. |
| `decay` | Evidence retention per accepted witness, in `(0, 1]`. One pools without decay; it still learns. |
| `capacity` | Positive integer event limit, or `None` for no event-count cap. |
| `value_bound` | Positive maximum absolute witnessed value, in the application's units. Out-of-range values are rejected, not silently clipped. |
| `settle_budget` | Positive integer maximum message sweeps per query/admission. |
| `tolerance` | Positive finite equation-defect threshold. Smaller is stricter and may need more sweeps. |
| `damping` | Finite proposal mixing in `(0, 1]`. Changes the iteration, not its fixed-point equations. |
| `prior_weight` | Positive initial zero-valued pseudo-evidence weight. |
| `prior_shape`, `prior_rate` | Positive finite precision-observer parameters; initial precision is their ratio. |
| `meta_shape`, `meta_rate` | Positive finite parameters of additional rate-observer stages; relevant above height 1. |
| `max_statistic_bits` | Integer from 64 through 65,536. Maximum numerator/denominator bit length of each exact retained statistic. |

Boolean values do not stand in for numeric configuration. The column's
`config` is read-only; `count` reports committed distinct witnesses. Exact
statistics and configured limits can reject a long stream even when
`capacity=None`. The priors must satisfy
`prior_shape + (prior_weight - 1)/2 > 0` for a positive finite precision
equilibrium at the initial evidence state; derived initial precision and
variance must also be finite.

### Evidence and readback

| Method | Contract |
| --- | --- |
| `add(value, *, budget=None)` | Uses the next local event ID. Two calls are two observations even if the values match. |
| `observe(event, value, *, kind="witness", budget=None)` | Admits the next ordered event, starting at one. An identical latest retry is a duplicate. |
| `query()` | Returns `answer`, `variance`, `precision`, `stages`, `qualified`, `residual` without teaching. `stages` maps extra stage numbers to their rate messages. |
| `observer_intervention()` | Read-only perturbation/recovery and two lesion diagnostics for the live belief–precision loop. Requires a qualified starting state. |
| `solve_factors(factors, edges)` | Delegates the separate exact finite-factor operation below; does not modify scalar evidence. |

Witnesses accept finite real scalars, `Fraction` and `Decimal`. Floating-point
witnesses are retained using their shortest decimal representation as an exact
rational value. Only `kind="witness"` is accepted. No API can verify that a
caller-labeled witness was actually measured.

Admission returns `accepted`, `qualified`, `duplicate` and `sweeps`. A
nonconverged attempt returns false admission/qualification and leaves the
retained state unchanged. A duplicate returns `duplicate=True` without a new
solve; its `qualified=False` describes the absence of a fresh admission, not a
claim that the existing model became invalid. Invalid inputs, conflicting or
out-of-order events, capacity exhaustion and statistic overflow raise
`ValueError` before commit. `budget=0` is useful for an intentional refusal;
`None` selects the configured budget.

### Column checkpoints

`snapshot() -> str` returns JSON containing exact statistics, configuration,
event cursor and latest witness. `restore(text)` validates a complete candidate
before installing it. `CorticalColumn.from_snapshot(text)` constructs a model
from a current configuration-bound checkpoint.

Restore checks configuration identity, moment admissibility, event count and
latest-witness consistency. These are consistency checks, not authentication
of the original sensor history. Only the current configuration-bound schema is accepted.

## Cortex

```text
Cortex(n_outputs=1, feature_maps=None, *,
       decay=0.99, discount=0.97, optimism=0.5, epsilon=0.02,
       coupling=1.0, target_bound=8.0, settle_budget=256, seed=0,
       learning_enabled=True, height=1, update_mode="episode",
       tolerance=1e-11, damping=1.0,
       prior_weight=1.0, prior_shape=1.0, prior_rate=1.0,
       meta_shape=4.0, meta_rate=4.0, level_weights=None,
       max_columns=100000, max_cache=4096, max_pending=10000,
       max_checkpoint_bytes=8388608, wiring_id=None)
```

A default Cortex has one output and uses the complete observation as its
context. For continuously varying vectors, prefer `from_dimensions` or
`BinnedFeatures` so nearby readings share a declared context. Retained columns
are indexed by level, context and output; arbitrary new contexts consume new
columns when taught. Contexts may contain finite numbers, strings, booleans,
`None`, lists, tuples and mappings; arrays exposing `tolist()` are copied through
that interface. Context keys preserve these types. Each encoded context is
limited to 4,096 items, 32 nesting levels and 16,384 characters per string.

### Model and resource parameters

| Parameter | Meaning |
| --- | --- |
| `n_outputs` | Positive number of output channels; default one. |
| `feature_maps` | Nonempty sequence of deterministic context functions, finest first. Default identity map. Built-in maps carry checkpoint descriptors; custom functions must keep their behavior fixed. |
| `decay` | Evidence retention in `(0, 1]`; default 0.99. Applies per admitted unit of witness mass. |
| `coupling` | Nonnegative coarse-prior strength; zero removes its influence. |
| `target_bound` | Positive clipping bound for constructed RL targets, or `None` to disable clipping. Native supplied targets are never clipped. |
| `height` | Positive observer-stage count per column, independent of the number of context levels. |
| `prior_weight`, `prior_shape`, `prior_rate` | Positive evidence/precision prior parameters. |
| `meta_shape`, `meta_rate` | Positive parameters for additional within-column observer stages. |
| `settle_budget` | Nonnegative integer maximum message sweeps for each required solve; zero allows diagnostic refusal. |
| `tolerance`, `damping` | Residual threshold and proposal mixing, as above. |
| `learning_enabled` | Boolean control. False prevents admissions; querying and action selection still operate. |
| `level_weights` | One positive finite witness-mass multiplier per level, or `None` for the context-count rule below. |
| `max_columns` | Positive integer limit on retained context/output columns across levels. Not a byte or RSS limit. |
| `max_cache` | Nonnegative integer bound on cached settled readouts; zero disables caching. |
| `max_pending` | Positive integer bound on buffered RL transitions. |
| `max_checkpoint_bytes` | Positive integer UTF-8 byte limit for checkpoint serialization/parsing; default 8 MiB. |
| `wiring_id` | Nonempty string up to 1,024 characters, or `None`. Custom feature functions require this identity for checkpointing. |

With `level_weights=None`, when every feature map declares a positive integer `cells`, level `l` receives
mass `min(1, cells_l / cells_finest)` per unit witness. Otherwise every level
uses unit mass. This is a declared scaling rule, not an estimate of actual
context occupancy. The constructor rejects an unrepresentable mass rather than
silently zeroing it; supply explicit weights for such wiring. Prior parameters
obey the same initial precision condition as the scalar column.

Retained Cortex statistics use floating-point arithmetic. Large offsets and
small dispersion can cause cancellation in moment subtraction; arithmetic
checks reject invalid statistics, but do not make those moments exact.
Choose useful sensor/target units and monitor qualification.

### Immediate prediction and observation

```text
value(observation, output=0) -> dict
query(observation) -> tuple[dict, ...]
predict(observation) -> tuple[float, ...]
observe(observation, targets, *, event=None, weight=1) -> dict
```

`value` returns `mean`, `variance`, `novelty`, `qualified`, `sweeps` and
`residual` for one output. `query` returns one such dictionary per output.
`predict` returns their means and raises `SettlementError` if a required answer
is unqualified. `SettlementError` is a `RuntimeError` raised when a required
belief cannot qualify; malformed inputs and resource refusals use `ValueError`.
Novelty is `sum(1/(w_l*tau_l))` across active levels, using their jointly settled
precisions. It omits direct coarse-prior precision from each denominator, but
can still depend on priors through `tau_l`; it differs from the final fused
variance.

Targets may be a scalar for one output, a full output sequence, or a sparse
mapping `{output_index: observed_value}`. Omitted outputs are not taught zero.
`weight` is positive finite observation mass. Native observation is immediate regardless
of `update_mode`, and all affected outputs/levels are admitted atomically.
The result identifies `accepted`, `qualified`, `duplicate` and `admitted`
column updates. An explicit event supplies retry ordering; omitting it allocates
the next event locally. Distinct IDs do not establish independent evidence.

Native observations and RL transitions share one ordered event cursor. Flush or
explicitly discard pending transitions before switching to native observation.
A latest identical native retry returns `duplicate=True`, `accepted=False`,
`qualified=True`, `admitted=0`; no fresh evidence or solve is added. Unqualified
admission or disabled learning returns false acceptance. Invalid input and
resource overflow raise `ValueError`.

Queries do not teach or retain new context columns; they may update caches/work counters.
Caller-owned mutable observations and returned results must not alias retained
continuation state.

### Optional reinforcement learning

| Parameter | Default | Meaning |
| --- | --- | --- |
| `discount` | 0.97 | Bootstrap discount in `[0, 1)`. Relevant to RL targets, not native observed targets. |
| `optimism` | 0.5 | Nonnegative novelty bonus weight for RL action scores and targets. |
| `epsilon` | 0.02 | Probability in `[0, 1]` of a uniform-random action. |
| `seed` | 0 | Nonnegative integer for exploration and tie-breaking. No training data are generated by it. |
| `update_mode` | `"episode"` | `"episode"` buffers transitions for backward admission; `"step"` processes them immediately. |

```text
act(observation) -> int
learn(observation, action, reward, next_observation, terminal=False,
      *, truncated=False, event=None) -> dict
flush() -> int
clear_pending() -> int
```

`act` qualifies every output belief before selecting either a random action or
the highest `mean + optimism*sqrt(novelty)` score. An unqualified readout raises
`SettlementError` even during random exploration. It does not execute an action
in the world. `learn` receives the actual executed action and measured transition.
In episode mode, a terminal/truncated boundary or explicit flush processes the
pending episode. `learn` returns `buffered`, `admitted` and `duplicate`.
Intake consumes the event ID when a transition is buffered. An identical retry
does not append it or retry a failed flush; call `flush()` explicitly.

Each transition's affected columns are committed atomically. A failed flush
keeps its failing and remaining transitions queued; transitions already
processed by that flush remain committed. Its return value counts column
admissions, not transitions. `clear_pending` discards the queue and returns its
length, without reusing its consumed event IDs. Disabling learning preserves
the pending queue.

The supplied target construction is

```text
target = reward
       + optimism*(1-discount)*sqrt(novelty)
       + discount*max(next_output_means)       # absent for terminal transitions
```

When `target_bound` is not `None`, the target is clipped to
`[-target_bound, target_bound]`. Bootstrapping uses means,
not optimistic action scores. Truncation ends a buffered episode without
silently declaring an absorbing terminal. Processing each buffered entry once
is not proof that its bootstrap target is an independently observed fact.
The wrapper is a TD policy layered around the column mechanism.

### Cortex construction and continuation

```text
Cortex.from_dimensions(n_inputs, n_outputs=1, *, bounds=(-1, 1),
                       bins=8, depth=1, **options)
Cortex.for_environment(env_factory, n_actions=None, *, depth=2, width=1,
                       calibration_steps=400, seed=0, **options)
snapshot() -> str
restore(text)
Cortex.from_snapshot(text, *, feature_maps=None, wiring_id=None,
                     max_checkpoint_bytes=8388608)
stats() -> dict
```

`from_dimensions` creates built-in numeric grid maps; a shared `(low, high)`
pair applies to every input, or supply one pair per input. `bins` selects
resolution and `depth` the grid hierarchy. Additional model keywords pass to
the constructor.

`for_environment` is optional. It uses the environment's discrete
`action_space.n` when no action count is supplied, calibrates observations and
constructs the maps. It spends real environment interactions and must not
calibrate on a held-out evaluation stream. Dictionary/image observations need
an explicit projection to a finite numeric vector; use `calibrate` with
`observation_fn`, then `wire`, and apply the same projection at runtime.

Cortex checkpoints use `cortex-state/1`. They bind configuration, feature
specifications or a custom wiring identity, retained statistics, random state,
event cursor, pending transitions and counters. `from_snapshot` reconstructs
built-in maps. Custom functions must be supplied again with the declared
compatible identity. A checkpoint cannot serialize arbitrary Python behavior
or the external environment.

`restore` requires matching configuration, including the original seed;
`from_snapshot` reconstructs it using the stored seed. Derived caches are not
stored. `stats` reports `updates` (column admissions), `rejected_updates`,
`settles` (solve calls), `cache_hits`, `greedy_actions`, `random_actions`,
`columns_per_level`, `level_weights`, `levels`, `height`, `n_outputs`, `pending`,
`cursor` and `cache_entries`. These are work/resource diagnostics, not total
operation counts or process memory measurements.

Configuration is immutable after construction except the documented Boolean
controls `learning_enabled` and `prior_ports_cut`. The latter cuts the
coarse-prior readout path for an intervention; it does not erase stored columns.

## Feature maps

```text
IdentityFeatures()
ConstantFeatures()
BinnedFeatures(bounds, bins=8, indices=None, *, clip=True)
grid(bounds, *, bins=8, depth=1, clip=True) -> tuple[feature_map, ...]
```

`IdentityFeatures` uses the complete observation as context.
`ConstantFeatures` returns `()` and declares one context cell.
Both expose `specification()` for checkpoint reconstruction and take no
configuration arguments.

`BinnedFeatures` takes one finite increasing `(low, high)` pair per selected
coordinate. `bins` is an integer from 1 through `2**31`, or one count per coordinate;
`indices=None` reads coordinates from zero in order, otherwise supply distinct
nonnegative indices of the same length. `clip=True` saturates out-of-range
coordinates to the first/last bin; false rejects them. Selected coordinates
must be finite; unselected coordinates are ignored. A scalar is accepted for
the one-coordinate, index-zero case.
`cells` is the product of bin counts; `specification()` is its serializable
description.

`grid` reads the same coordinates at every non-global level. Depth zero is the
fine map only; depth one adds a global context. Larger depths add `depth-1`
coarsenings, halving each bin count down to one, then the global context.
Depth is an integer from zero through 32.

## Calibration and ranked wiring

```text
calibrate(env_factory, n_actions, *, steps=400, seed=0,
          observation_fn=None) -> dict
wire(calibration, *, depth=2, width=1, ladder=None) -> tuple[feature_map, ...]
```

The factory must create an environment whose `reset(seed=...)` returns
`observation, info` and whose `step(action)` returns
`observation, reward, terminated, truncated, info`. Calibration holds each
action for `steps` interactions, using the same reset-seed schedule per action,
and closes every environment. `observation_fn` can project observations to
finite numeric vectors of a fixed size. It must also be applied consistently
when using the resulting maps afterward.

The report contains coordinate ranges and rankings by action-conditioned mean
separation and within-run variation. These are diagnostic associations, not
proofs of causal controllability. Exactly `steps*n_actions` step calls are
spent; resets and observation processing are additional work.

`wire` selects `width` action-sensitive coordinates plus up to `width`
additional varying coordinates. Width is a positive integer, bounded by the
number of ranked coordinates. The fine map uses 16 bins for the first
action-sensitive coordinate and 8 for each additional one; it uses 8 bins for
the first additional varying coordinate and 4 for each remaining one.

Depth is an integer from 0 through 32. Zero returns the fine map alone. Positive
depth adds `depth-1` maps of the primary action-sensitive coordinate at
`max(1, 16//2**i)` bins for `i=1..depth-1`, then a global map. `ladder` overrides
the intermediate counts with up to 31 positive bin counts and always includes
the fine and global maps; an empty ladder means just those two. This differs
from `grid`, which keeps every selected coordinate at every non-global level.
No coordinate is assumed to occupy a fixed range.

## Direct repair

```text
Port(name, source, target, family, message, meta=None)
settle(ports, *, budget=256, lesion=None, tolerance=1e-11,
       damping=1.0, lesion_precision=1.0) -> dict
```

`Port` is exported from `cadence`. Names must be nonempty and unique
within a solve. A source supplies `emit(port, inbox)`; the target owns the
corresponding incoming message. `family` is `"scalar"`, `"moments"` or
`"table"`; `message` is its initial payload; `meta` is optional metadata.
Emitters must be deterministic and must not mutate hidden state while queried.
Each call receives a fresh inbox dictionary. Initial messages and emitted
proposals are validated and normalized; canonical floats and moment tuples can
be shared safely, while table dictionaries are copied for each inbox. This
avoids repeated validation of unchanged messages inside the repair loop.

`budget` is a nonnegative integer. Scalar/moment damping lies in `(0, 1]`;
table messages require one. `lesion=None` is normal operation; the built-in
`"variance_blind"` and `"feedback_cut"` interventions refer to the scalar
column's named ports. `lesion_precision` supplies the cut feedback value.

Results contain `messages`, `sweeps`, `converged`, `executed_residual`,
`full_residual`, `stationarity` and `executed_stationarity`. Qualification
checks the undamped executed-equation residual and a fresh executed sweep.
Intact diagnostics are separate when a lesion is active. A zero budget does
not qualify a nonempty port graph. Invalid/nonfinite payloads raise
`ValueError`; finite nonconvergence returns `converged=False`.

## Exact factor federation

```text
solve_cluster_forest(factors, edges, budget=64) -> list[belief]
```

Each factor is `{"scope": (...) , "values": {binary_tuple: potential, ...}}`.
Scopes contain distinct variable names, and the table covers every binary
assignment exactly once. Potentials are nonnegative integers or `Fraction`
values; floats are rejected. Edges join distinct factor indices without
cycles or duplicates. Shared-variable clusters must satisfy running
intersection. Returned beliefs contain the original scope and an exact
normalized `Fraction` table.

```python
from fractions import Fraction
from cadence import solve_cluster_forest

factors = [{"scope": ("signal",), "values": {(0,): 1, (1,): 3}}]
beliefs = solve_cluster_forest(factors, [])
assert beliefs[0]["values"][(1,)] == Fraction(3, 4)
```

Uncertifiable structure, contradictory zero support, invalid input or a cap
that prevents qualification raises `ValueError`. This exact finite operation
has no automatic capacity advantage: tables grow with binary scope size.
