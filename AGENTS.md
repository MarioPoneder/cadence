# Building on Cadence

Read [the guide](docs/DRSN.md), [API reference](docs/REFERENCE.md) and
[specification](docs/SPECIFICATION.md) before changing semantics.

## Architecture

Import `Cortex` and `Brain` from `cadence`. `Cortex` declares populations;
`build()` returns a persistent `Brain`. `column(patches=..., inputs=...)` and
`observer(patches=..., observes=...)` use the same processing-patch rule.
Width counts processing states. Recursive depth comes from observation wiring.
`observes` reads current state and exact prediction error and contributes
feedback to the same coupled solve. Do not implement recursion by chaining
completed population predictions.

The only runtime modules are the population builder/brain (`drsn.py`), numerical
repair (`_repair.py`) and validation (`_validation.py`), plus package exports.
Keep the public API small and application-independent. The package is standard
library only, Python 3.11+.

## Changes and verification

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests execute Python examples in the README and every documentation page.
Constructor signatures and package exports must match the reference. Add
independent mathematical or adversarial tests for changes to derivatives,
qualification, witness custody or serialization. Preserve bounded construction
and refuse malformed shapes before expensive materialization.

## Application rules

- `settle` and `predict` are pure queries. `step` retains qualified live state;
  `observe` also repairs parameters from actual output witnesses.
- Check `qualified`/`accepted`. Refused or capped outputs are diagnostics,
  not admitted actions. `predict` raises `SettlementError` on refusal.
- Evaluate learning on subsequent **unclamped** predictions. A training output
  equal to its clamp is not a learning result.
- `observe` is supervised learning. Rewards require a separately justified
  temporal-credit design; do not label reward-as-target as reinforcement learning.
- Check input sufficiency before interpreting failure. Sparse coverage does
  not provide learned visual/audio features or recover omitted information.
- Serialize calls to a brain. Checkpoints bind exact implementation sources;
  source hashes are compatibility checks, not authenticated witness evidence.

## Claims

Report stationarity and prediction error separately. The global energy check
and analytic derivatives are part of this reference implementation; it is not
a proof of asynchronous distributed confluence. Extra observers do not
establish a capability advantage without matched-information, matched-capacity
and declared-work controls. Preserve unsuccessful outcomes and receipts.
