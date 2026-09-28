# Changelog

## Unreleased

- Separate runtime (`brain.py`), construction (`cortex.py`), population definitions
  (`column.py`) and sensory/output boundaries (`ports.py`). Top-level imports
  remain `from cadence import Cortex, Brain`; module imports use those focused
  files. Remove the combined `cadence.drsn` module.
- Preserve settlement and learning behavior, with exact before/after comparisons.
  Bind checkpoints to all six semantic modules; source-bound snapshots from a
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
