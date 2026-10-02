# Building on Cadence

Read [the guide](docs/DRSN.md), [API reference](docs/REFERENCE.md) and
[specification](docs/SPECIFICATION.md) before changing semantics.
Use [the agent recipe](docs/AGENTS.md) to construct and assess an application;
it also defines the documentation rules for System 1 and System 2.

## Three mandatory review gates

**Minimalism, user-friendliness and agent-friendliness govern every change.**

| Pillar | A change is ready only when |
| --- | --- |
| Minimalism | It builds on bounded patches, ports, readback and explicit learning/repair, keeps the public API small, and adds no compatibility machinery, application-specific core rule or mandatory runtime dependency. Optional acceleration must preserve the same mathematical rule and admission contract. A new abstraction must remove real duplication or enable a demonstrated general need. |
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
`memory.py` supplies explicit bounded sensory history and an error-progress
heuristic. `reinforcement.py` supplies discrete Q-learning target orchestration
and transition replay through that same repair law; derived teaching targets
must use `source="estimate"`. `runtime.py` supplies a serial callback owner and
actuator rate limits, without hard deadlines or solver cancellation. These
helpers are not new patch primitives or biological chemistry.

## System 1 and System 2

**Recommend System 1 for applications.** Start with the smallest adequate flat
`column`, then add ordinary `column(..., inputs=previous)` layers when the task
needs intermediate representations. Most examples and application recipes must
use this ordinary path. Ordinary depth is not System 2. Check held-out quality
and end-to-end latency; depth alone does not guarantee fast settlement.

**Recursive self-observation is experimental.** `observer(..., observes=...)`
is explicit opt-in and participates in every whole-brain solve. It can slow
routine responses; the public runtime does not automatically put it to sleep
or recruit it only on surprise. Its advantage over capable ordinary layers
is not established. Keep recursive examples clearly labeled experiments and
link the [experimental boundary](docs/EXPERIMENTAL.md). Never add observers to
a production-oriented recipe merely because a task is complex or long-term.

**System 1** means acquired routine competence with inexpensive repair.
**System 2** means additional recursive observation and correction when routine
behavior cannot maintain equilibrium, including longer-term outcomes. These are
roles within one brain, not public classes or synonyms for flat and deep.
Ordinary deep populations can learn specialized routines. Recursive observers
add exact current error readback; their presence alone proves no useful correction.
Bootstrapping and live operation are lifecycle phases, distinct from these roles.

The current API builds flat, ordinary composed and observing layouts, including
mixed layouts, under one whole-brain solve. It does **not** yet implement
automatic recruitment of reflection, independently progressing fast and slow
populations, or integrated shared outcome responsibility. Describe these as
requirements until the runtime and behavioral evidence support them.

Keep the intended application contract small: one brain/body interface for
observations, qualified actions and actual outcomes. Specialized learned roles
must not require per-population evaluators or a user-managed attention scheduler.
Do not add a System 1/System 2 mode flag, extra brain wrapper or migration layer
to express this recommendation. Ordinary construction is already the default.
Currently all populations use the same patch rule. Future specialized mechanisms
need explicit bounded state, readback, learning and repair semantics, a declared
qualification contract and measured general benefit. Minimalism does not establish
that the current primitive can replace every useful memory mechanism.
Any future internal scheduling must preserve the declared whole-brain
qualification, execution custody and saved continuation. Do not certify a
sleeping or delayed component by omitting it from the check.

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
  `observe`. The `Reinforcement` helper assigns explicit estimated action-value
  targets from observed transitions; neither primitive admission alone nor that
  helper is a complete biological learning mechanism.
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
- `observe` is supervised learning. Use `source="estimate"` for derived targets;
  keep source labels bound to retry identity. `Reinforcement` implements declared
  normalized, clipped Q targets and replay; do not label a raw reward-as-action
  target as reinforcement learning. Check behavior on later free decisions.
- A `History` buffer is explicit external memory; do not claim learned recurrent
  memory from it. Retention after replay requires old-skill checks.
- Never discard unprocessed reward/action evidence in `LiveController`'s latest
  sensory slot. Count candidate queries, learning work and complete command age;
  a solver sweep budget is not a real-time guarantee.
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
