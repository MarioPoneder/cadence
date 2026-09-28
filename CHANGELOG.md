# Changelog

## 0.47.0 — 2026-09-28

- Query repair computes state derivatives without allocating or validating
  unused parameter derivatives. This fixes a finite-input query that failed
  solely because a derivative of a frozen weight overflowed. Learning and the
  full derivative evaluator still validate parameter gradients.
- Remove frozen parameter coordinates from query line-search and secant work.
  Joint state/error feedback, energy, qualification, refusal and learning rules
  are unchanged. The public API gains no new flag or primitive.
- Avoid abstract-number type checks for native Python floats and integers;
  retain finite-value, positive-value, boolean, extended-real and conversion
  validation. This reduces repeated validation overhead in larger layouts.
- Add regressions for the public query failure, exact full-result parity through
  recurrent state/error graphs, learning and numeric validation. Document the
  distinction between recursive observation depth and planning horizon.
- Checkpoints remain bound to exact engine sources; these engine changes require
  checkpoints produced by the new implementation.

## 0.46.0 — 2026-09-28

- Add `bootstrap(brain, examples, checks=..., max_error=...)`: one standard-library
  helper for validated example replay and unclamped recall/readiness checks.
  It shuffles reproducibly, reports work and error history, and stops on the
  declared limit, exhausted epochs or numerical refusal. Checks never teach
  the brain; earlier admitted examples persist if a later solve refuses.
- Use **bootstrapping phase** and **live phase** throughout current guides and
  quickstarts. Both use the same brain and repair rule; actual new witnesses
  can continue learning during the live phase.
- Rename the preparation guide to `docs/BOOTSTRAP.md`, with runnable bootstrap,
  calibration and live-handoff examples plus staged perception/body-control
  curricula. These are generic procedures, not supplied learned abilities.
- Add tests for acquisition, check custody, deterministic replay, refusal,
  prevalidation, sample ownership, shapes/aliases and complete solve accounting.
  Minimal-install CI bootstraps and continues learning without optional packages.
- The six checkpoint-bound engine modules and the repair equations are unchanged
  from 0.45.0; orchestration reports and application preprocessing remain separate
  from brain snapshots.

## 0.45.0 — 2026-09-28

- Raise the default repair ceiling from 512 to 2,048 sweeps. Small recursive
  teaching examples exhausted the old ceiling; six tested initializations
  acquire independent relations with the larger allowance. Already qualified
  solves still stop early. The equations and qualification threshold are unchanged.
- Report structural sensor coverage per output coordinate and the number of
  patches connected to any output. Shared fixed inputs do not falsely join
  otherwise independent populations. These diagnostics expose disconnected
  decisions and unused width without rejecting deliberate constant branches.
- Reject event IDs that cannot be JSON-encoded before solving or admitting
  them, preserving checkpointability and atomic retries.
- Make missing inputs, invalid targets and numerical refusals explain the
  affected names, bounds or qualification measurements. Accept scalar integer
  shapes consistently with integer dimensions from array libraries.
- Add fast acquisition gates across flat, composed and recursive layouts:
  multiple seeds, independent inputs/outputs, unclamped answers, replay,
  input ablation and checkpoint continuation. Clarify target scaling,
  classification scores and output aliases in the training guide.
- Exact source-bound checkpoints from earlier implementations remain incompatible.

## 0.44.0 — 2026-09-28

- Adapt the projected repair step to observed curvature using a guarded scalar
  secant estimate. Keep the same energy, analytic gradient, Armijo acceptance,
  finite budget and final stationarity check. This reduces measured work on
  small learning tasks; it does not guarantee qualification or task accuracy.
- Handle rounding at the end of a solve: a fully stationary proposal may finish
  within eight energy ulps when strict decrease is numerically indistinguishable.
  Ordinary steps retain Armijo decrease; the requested qualification tolerance
  and fresh final check are unchanged. This fixes a Linux tight-tolerance stall.
- **Default wiring change:** `fan_in=None` connects every declared source
  coordinate to each destination patch. Positive integers explicitly request
  sparse sampling. Aggregate coverage alone does not ensure a selected output
  receives every input. Connection limits still refuse oversized layouts before
  materialization; there is no silent sparse fallback.
- Add training and size guidance covering connected capacity, input conditioning,
  supervised witnesses, replay, closed-loop evaluation and matched controls.
  Use smaller runnable layouts and numerical defaults in the quickstarts.

- Share numeric validation across construction and repair; remove unused
  ownership tokens, runtime input ranges and redundant compiler bookkeeping.
  Ownership still requires the exact handle registered with its cortex.
- Separate runtime (`brain.py`), construction (`cortex.py`), population definitions
  (`column.py`) and sensory/output boundaries (`ports.py`). Top-level imports
  remain `from cadence import Cortex, Brain`; module imports use those focused
  files. Remove the combined `cadence.drsn` module.
- The module split preserves behavior, with exact before/after comparisons;
  adaptive repair changes numerical trajectories. Bind checkpoints to all six
  semantic modules; source-bound snapshots from a
  different implementation remain incompatible.
- Clarify that minimalism concerns concepts, dependencies and indirection;
  focused source files are encouraged.

## 0.43.0 — 2026-09-28

- Make population DRSNs the public API: `Cortex` declares sensors, processing
  columns, recursive observers and outputs; `build()` produces a `Brain`.
- Settle observed and observing populations jointly using exact live state/error
  readback, analytic energy derivatives and bounded projected repair.
- Support supervised witness admission, pure queries, live-state continuation,
  explicit refusal diagnostics and complete validated JSON checkpoints.
- Harden layout identity, dense/deep observer construction, bounded iterator
  reads, shape validation before array copying, and checkpoint source identity.
- Replace all guides and README examples with the population architecture and
  test every Python documentation example. Check derivatives and learning
  against independent numerical and analytic calculations.
- **Breaking API change:** replace scalar/context-bank `Cortex` and
  `CorticalColumn`, feature-map helpers and their modules with the population
  interface. The package has no compatibility layer. Installations relying on
  those interfaces must stay on their existing release until their applications
  adopt the population API. The game-specific optional dependency group is
  removed; integrations own their environment dependencies.

Qualification means constrained stationarity, not a unique global optimum or a
proven benefit from recursive depth. Performance evaluations remain ongoing.
