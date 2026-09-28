# Runtime and mathematical specification

This document specifies the population DRSN engine. The equations are in
[the processing-patch description](ELEMENT.md); all arguments and result fields
are in [the API reference](REFERENCE.md).

## Layout and state

- `Cortex` is a declaration builder. `build()` requires at least one population
  and output, resolves deterministic wiring, and freezes declarations. Every
  declared source coordinate is connected by default. Positive `fan_in` requests
  sparse wiring; connection-budget overflow raises instead of dropping inputs.
- `Input`, `Population` and `Output` are immutable identity handles owned by a
  layout. Their counts and shapes are validated before compilation.
- `inputs` connects samples or live population states. `observes` connects live
  states and derived prediction errors. Those constraints affect the same joint
  repair; observers do not query completed lower-layer answers.
- `Brain` owns state, relation weights and biases. Public state and parameter
  views are tuples; configuration is read-only; returned diagnostics are owned
  copies. Callers serialize access to each brain.
- Bounds apply to live state, weights, biases and clamps. Sensory samples need
  only be finite; applications choose normalization appropriate to their task.

## Repair and qualification

The solver minimizes the stated nonlinear energy by projected analytic-gradient
steps with sufficient-decrease backtracking. Observed error derivatives include
all transitive dependencies. A solve qualifies only when the full projected
stationarity residual is at most `tolerance` over every eligible coordinate.
Output clamps are excluded; learned parameters are included during admission.

The first trial uses `step`. After acceptance, let `s` be the change in eligible
coordinates and `y` the change in their exact gradient under the same objective.
The next trial uses the scalar secant estimate `dot(s,s) / dot(s,y)` when its
curvature and result are positive and finite. Unchanged coordinates contribute
nothing. Unsafe arithmetic falls back to `step`; growth is capped at
`ldexp(step, min(backtracks - 1, 1023))`, with an overflowing cap also causing
fallback. Every trial projects onto the same boxes and requires a finite,
negative slope. Ordinary steps satisfy
`E_new <= E_old + 1e-4 * dot(gradient_old, displacement)`.
Near floating-point precision, a final proposal may instead qualify when its
full projected residual meets the requested `tolerance` and
`abs(E_new - E_old) <= 8 * ulp(E_old)`. This narrow finishing allowance avoids
rejecting a stationary proposal because rounded energy appears a few ulps
higher. It never admits an unqualified proposal or a larger energy increase.
The final stationarity check is freshly recomputed in either case.
Step adaptation resets on each solve; it is not additional learned memory.
The estimate is the first Barzilai–Borwein step from
[Two-Point Step Size Gradient Methods (1988)](https://doi.org/10.1093/imanum/8.1.141),
used here inside bounded projected repair with the finishing allowance above.

`prediction_residual` is a different quantity: the largest absolute local
prediction error. Priors, bounds and competing constraints can leave this
nonzero at a qualified point. Qualification establishes constrained numerical
stationarity to the declared tolerance. It establishes neither a unique normal
form nor a global minimum, and says nothing by itself about task accuracy.

`budget` limits accepted sweeps, with at most `backtracks` attempted steps per
sweep. Already stationary proposals may qualify with budget zero. Exhaustion
or inability to find a descent step refuses the proposal. Invalid arguments
raise `ValueError`. Numerical overflow may refuse a proposal or raise
`ValueError`; neither outcome commits continuation. Applications must not act
on diagnostic outputs from a refused solve.

## Continuation and admission

| Operation | Live state | Retained parameters | External event record |
| --- | --- | --- | --- |
| `settle` / `predict` | Unchanged | Unchanged | Unchanged |
| `settle` with hypothetical clamps | Unchanged | Unchanged | Unchanged |
| Qualified `step` | Commit solved state | Unchanged | Unchanged |
| Qualified `observe` | Commit solved state | Commit solved relations | Advance once |
| Refused operation | Unchanged | Unchanged | Unchanged |

`observe` requires at least one output witness. It fixes these values throughout
joint state/parameter repair and anchors parameters to their pre-experience
values. The entire proposal qualifies before admission. There are no partial
parameter commits and no event ID consumption on refusal.

Explicit nonnegative event IDs are monotonically increasing. Retrying the latest
admitted ID with identical sensory samples and physical clamps is idempotent;
it returns `duplicate=True` without solving or committing. Changed or older IDs
are rejected. This is latest-event deduplication, not a history of all events.
Omitting an ID allocates the next identity on acceptance, so applications that
need retry safety must retain their external IDs.

## Checkpoint contract

Snapshots contain the complete layout, configuration, arrays, admission cursor
and digest. JSON must have exactly the expected fields, finite numeric values,
consistent bounds, counts and identity. Duplicate JSON fields are invalid.
Size is limited to 32 MiB. Reconstructed connections and dimensions are checked
before installing a proposed continuation.

A layout/configuration fingerprint and exact hashes of `brain.py`, `cortex.py`,
`column.py`, `ports.py`, `_repair.py` and `_validation.py` bind compatibility.
Previous source sets and different hashes are rejected, including across
releases. `restore` only installs a validated continuation for the same
graph/configuration; `from_snapshot` constructs one. Source identity and digest
checks are integrity checks, not cryptographic authentication of a witness or
proof of its truth. Treat caller-provided files as bounded data, never executable
code.

## Evidence boundary

Tests check numerical derivatives independently against finite differences,
analytic optima on small cases, causal feedback into observed populations,
energy descent, constrained boundaries, source coverage, witnessed acquisition,
unclamped recall, refusal rollback, event custody and continuation. Executable
documentation uses the same released API.

The engine supplies supervised witness learning and persistent joint activity.
It does not yet supply reward-driven temporal credit, autonomous task discovery,
learned structural growth or a general biological physiology model. Performance
and advantages from recursive depth remain empirical questions. Biological
inspiration is not evidence that a numerical qualification reproduces a brain.
