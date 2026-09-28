# Building on Cadence

Read [the guide](docs/DRSN.md), [API reference](docs/REFERENCE.md) and
[specification](docs/SPECIFICATION.md) before changing semantics.

## Three mandatory review gates

**Minimalism, user-friendliness and agent-friendliness govern every change.**

| Pillar | A change is ready only when |
| --- | --- |
| Minimalism | It uses the common patch/repair substrate, keeps the public API small, and adds no compatibility machinery, application-specific core rule or mandatory runtime dependency. Optional acceleration must preserve the same mathematical rule and admission contract. A new abstraction must remove real duplication or enable a demonstrated general need. |
| User-friendliness | A first-time user can construct, query, teach and save a brain from the quickstart. Names separate width, sensor shape and recursive observation. Configuration has validated defaults, errors explain the violated contract, and failure leaves continuation intact. |
| Agent-friendliness | Public signatures, defaults, return fields, mutation rules and failure behavior match the reference. State and topology are inspectable; examples execute; checkpoint identity and retry semantics are explicit. No undocumented preprocessing or hidden fallback changes the task. |

Minimalism measures concepts and dependencies, not file count. Separate
responsibilities into focused modules when that makes ownership clearer.

Before merging, explain how the change meets each applicable gate and run the
checks below. Prefer improving an existing primitive to adding another. Do not
add aliases or wrappers solely to make a second way to express the same thing.
Performance or convenience must preserve qualification and witness custody.
Tests enforce executable examples and signature/export parity; architectural
simplicity and clarity still require review rather than a test-count claim.

## Architecture

Import `Cortex` and `Brain` from `cadence`. `Cortex` declares populations;
`build()` returns a persistent `Brain`. `column(patches=..., inputs=...)` and
`observer(patches=..., observes=...)` use the same processing-patch rule.
Width counts processing states. Recursive depth comes from observation wiring.
`observes` reads current state and exact prediction error and contributes
feedback to the same coupled solve. Do not implement recursion by chaining
completed population predictions.

`brain.py` owns `Brain`, `SettlementError`, runtime state and checkpoints;
`cortex.py` owns the `Cortex` builder and compiler; `column.py` owns immutable
`Population` handles; `ports.py` owns `Input`, `Output` and boundary shape/value
validation. Private `_repair.py` supplies numerical repair and `_validation.py`
supplies numeric/JSON validation. Package exports keep the public API small and
application-independent. The default engine is standard library only, Python 3.11+. Private `_tensor.py`
provides optional PyTorch tensor execution with analytic derivatives; it supplies
no autograd optimizer or separate learning rule. Final tensor proposals are
qualified against the original float64 objective by the reference engine.
`bootstrap.py` supplies bounded replay and unclamped readiness checks through
those existing brain methods. It owns orchestration, not a second learning rule.
`observe_batch` uses private per-example activity with shared parameters under
mean example energy plus one pre-batch parameter anchor. It commits parameters
and one event identity, preserving live state. Tensor execution vectorizes rows;
it does not average separately learned checkpoints or introduce another solver.

## Changes and verification

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests execute Python examples in the README and every documentation page.
Basic learning tests must also pass: multiple seeds, independently varied inputs,
unclamped recall after replay, and saved continuation across flat, composed and
observing populations. Keep these gates small enough to run in ordinary CI.
Numerical qualification alone cannot pass an acquisition test.
Constructor signatures and package exports must match the reference. Add
independent mathematical or adversarial tests for changes to derivatives,
qualification, witness custody or serialization. Preserve bounded construction
and refuse malformed shapes before expensive materialization.
Batch tests must check shared-parameter derivatives, unaveraged per-row
qualification, all-or-nothing admission, order-sensitive retry identity,
preserved live state and separate example/update counters. Check both one-row
equivalence of parameter solves and the distinct live-state commitment rules.

## Application rules

- Use **bootstrapping phase** for initial preparation and **live phase** for
  ongoing operation. These application lifecycle terms use the same repair law;
  do not introduce an automatic phase toggle or phase-wide parameter freeze.
  The live phase can continue learning through actual witnesses supplied to
  `observe`; this remains supervised witness admission, not automatic reward
  credit or a complete biological learning mechanism.
- `settle` and `predict` are pure queries. `step` retains qualified live state;
  `observe` also repairs parameters from actual output witnesses.
- `observe_batch` repairs private row states and shared parameters, retaining
  parameters only. One batch owns one admission/event. Rows share no implicit
  temporal state; batching does not provide delayed credit or sequence memory.
  `bootstrap(batch_size=1)` keeps ordered `observe`; larger sizes change the
  learning trajectory. Count `presentations`/`accepted` as examples and
  `updates` as atomic admissions, and measure accuracy as well as throughput.
- Check `qualified`/`accepted`. Refused or capped outputs are diagnostics,
  not admitted actions. `predict` raises `SettlementError` on refusal.
- Evaluate learning on subsequent **unclamped** predictions. A witnessed output
  equal to its clamp is not a learning result.
- `observe` is supervised learning. Rewards require a separately justified
  temporal-credit design; do not label reward-as-target as reinforcement learning.
- Check input sufficiency before interpreting failure. Sparse coverage does
  not provide learned visual/audio features or recover omitted information.
- Start with [the bootstrapping guide](docs/BOOTSTRAP.md). Default wiring includes
  every declared source coordinate; sparse `fan_in` is an explicit choice.
  An unused flat patch does not provide hidden capacity to a separate output.
- Parallelize independent brains or environment collection; keep each brain's
  experience admissions ordered. Device availability is checked on first solve;
  do not silently substitute a different device.
- Serialize calls to a brain. Checkpoints bind exact implementation sources;
  source hashes are compatibility checks, not authenticated witness evidence.

## Claims

Report stationarity and prediction error separately. The global energy check
and analytic derivatives are part of this reference implementation; it is not
a proof of asynchronous distributed confluence. Extra observers do not
establish a capability advantage without matched-information, matched-capacity
and declared-work controls. Preserve unsuccessful outcomes and receipts.
