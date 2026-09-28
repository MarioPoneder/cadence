# Building on Cadence

Guidance for coding agents and their humans. The API reference is
[docs/REFERENCE.md](docs/REFERENCE.md); runnable examples are in
[docs/QUICKSTART.md](docs/QUICKSTART.md); qualification and claim
boundaries are in [docs/SPECIFICATION.md](docs/SPECIFICATION.md). This
file holds what those pages cannot: the mistakes already made once,
with their symptoms.

## Commands

```sh
python -m pip install -e ".[dev]"   # library is stdlib-only; dev adds pytest/ruff
pytest -q                            # full suite, a few seconds
ruff check src tests && ruff format --check src tests   # CI gates
```

Python 3.11+. The `games` extra adds Gymnasium/ALE for Atari work.

## The API in five lines

- `CorticalColumn(...)`: one scalar estimator with uncertainty; `add` /
  `observe` admit evidence, `query` reads, `snapshot`/`restore` persist.
- `Cortex(n_outputs, feature_maps, ...)`: banks of columns over
  contexts; `observe`/`predict` for supervised targets, `act`/`learn`/
  `flush` for reinforcement, `value` for diagnostics.
- Feature maps are supplied, stateless callables; `wire`, `grid`,
  `BinnedFeatures` build common ones. `Cortex.from_dimensions` and
  `Cortex.for_environment` remove the wiring boilerplate.
- Checkpoints are validated JSON; custom map callables need a
  `wiring_id` you version yourself.
- `settle` and `solve_cluster_forest` are the underlying repair law and
  the exact finite solver; most applications never call them directly.

## Pitfalls, from experience

Each row was hit in practice by an experienced agent. Read the row
before "fixing" the symptom somewhere else.

| Symptom you will see | Cause | Fix |
| --- | --- | --- |
| RL learns on long episodes but is near-dead on short ones (values flat at a small constant, novelty exhausted, policy wanders) | Feature maps declare `cells`, so coarse levels admit at witness mass cells_level/cells_finest and move too slowly to weld short episodes | Witness mass is a task-scale choice: keep `cells` when episode length x mass is large next to the decay window 1/(1-decay); drop `cells` or pass explicit `level_weights` of ones when episodes are short |
| RL oscillates between near-perfect and near-zero episodes on long tasks | The opposite error: no witness mass, so one long episode's flush washes a coarse column's whole decay window and erases the previous episode's evidence | Declare `cells` on the maps (builders in `wiring` already do) so every level shares one evidence timescale |
| `rejected_updates` climbing in `counters` | Settlement did not converge inside `settle_budget`; tall columns (`height > 1`) with heavy evidence contract at ~0.73/sweep and need the full default budget | Keep `settle_budget=256`; do not lower it to "speed things up" - an unsettled state is rejected, never used |
| `ValueError: Retry flush or clear_pending before beginning another transition` | The previous episode ended (terminal/truncated) but its queue was left unflushed, or `update_mode` was mixed up | Call `flush()` at every episode boundary you own; `learn(..., terminal=True)` and `truncated=True` flush themselves |
| `SettlementError` from `predict`/`act` | A belief did not qualify; `predict` refuses unsettled numbers by design (`value` reports `qualified=False` instead) | Treat it as a real refusal: raise the budget, simplify the model, or read `value` diagnostics - do not catch-and-ignore |
| Learner stuck at random-level scores on a game whose value depends on object relations (the recorded Pong case) | Calibration ranked clock/counter bytes above game state, and absolute positions cannot express "controllable thing relative to moving thing" | Filter counter bytes (constant nonzero delta most steps) out of calibration ranks; build relational features (differences between ranked bytes, motion signs over a two-frame stack). Feature maps are stateless - stack frames in the environment wrapper |
| Exploration noise drowns a real value gradient | `optimism` too high; the novelty term outweighs learned means | Keep `optimism` at its default; explore through novelty, keep `epsilon` small (it only breaks ties) |
| "Random baseline" that is not random | `learning_enabled=False` alone is the frozen-memory control, which still acts greedily on its priors | Uniform-random control is `learning_enabled=False, epsilon=1.0`; report both controls |
| Old tall-column checkpoint refuses to load under 0.42+ | Self-observation modes are on by default; pre-0.42 `height > 1` state predates them | Load with `self_observation=False` |
| Custom feature maps refuse to checkpoint | Callables cannot be serialized safely without an identity | Pass `wiring_id` and bump it whenever the maps' behavior changes |

## Claims discipline

Results built on Cadence follow the same rules the library was built
under: run the uniform-random and frozen-memory controls at identical
interfaces, declare thresholds before confirmation runs, use fresh
seeds for confirmation, and keep every run's returns, counters,
versions and source hashes in a receipt. A tuned single seed is a
search result, not a finding. Negative results stay in the receipt.

## What the library will not do for you

Feature choice is model choice: Cadence learns evidence within the
contexts you supply and does not discover representations. Settlement
qualification is not task success. Passing library tests does not
qualify an application; see docs/SPECIFICATION.md for what the API
guarantees and what remains yours.
