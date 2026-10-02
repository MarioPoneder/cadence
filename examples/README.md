# Cadence examples

These examples run on the `0.62.0` checkout with Python 3.11 or later. First
acquire a small relation with two coupled populations, then use a learned body
model to choose actions. Neither needs an optional dependency. The
[quickstart](../docs/QUICKSTART.md) covers installation;
[brain design](../docs/BRAIN_DESIGN.md) explains how to choose a larger layout.

## One graph and one learning interface

Population sizes and wiring determine which states and errors each patch reads.
All populations form one connected graph and repair together. Adding an
intermediate population changes the paths in that graph; it does not select a
new kind of brain. Optional error readback uses the same patch law and solve.
See the [experimental boundary](../docs/EXPERIMENTAL.md) for behavioral claims.

`settle` and `predict` are pure queries, `step` retains qualified activity, and
`observe` learns from supplied output targets. Check `qualified` or `accepted`;
`predict` raises on refusal. Learned relations persist and remain plastic; a
warm activity state is not a guarantee of temporal memory.

## Change wiring through the same learning loop

[layout_learning.py](layout_learning.py) is the smallest complete example:
teach a scalar relation with the default `small` layout (four `features`
patches read the sensor, one `response` patch reads them), check new unclamped
inputs and verify saved continuation. Select `deep` to add an intermediate population.
It uses the default Python engine with no optional dependency:

```sh
PYTHONPATH=src python examples/layout_learning.py
PYTHONPATH=src python examples/layout_learning.py --layout deep
```

Explicit `--layout recursive` runs the observer experiment; `--layout all`
includes it in a comparison. The small case has five connected patches, the
deep and recursive cases seven each. Observation adds error contacts. They are
API examples with different capacities and costs. The CLI labels identify
these particular examples; all their populations settle together. For a larger task, follow
[brain design](../docs/BRAIN_DESIGN.md) before scaling this small relation.

## Use an acquired model to control a body

[live_control.py](live_control.py) teaches the consequence of position and
commanded velocity in a small simulated body. It checks new inputs without
target clamps, compares candidate actions using the learned next-position
model, executes a command, and learns from the measured transition:

```sh
PYTHONPATH=src python examples/live_control.py --decisions 20 --seed 0
```

Read `bootstrap.passed`, `final_position`, `initial_need` and `final_need` in
the JSON report. Here `need` is squared distance to zero; the overall `passed`
field requires every controller response to qualify and the final distance
to improve. The report also retains actual transitions, their learning
admissions and observed command latency. A nonzero exit means the example's
check failed.

The application supplies the action candidates, distance/effort score and
actuator limits. This is a small learned-model control example, not automatic
recursive planning or an observer advantage. `LiveController` serializes one
brain's work while the caller polls; this simulation waits for each command.
Use `layout_learning.py` for the separate save/resume example.

## Choose an example

| Example | Actual layout | What it demonstrates |
| --- | --- | --- |
| [layout_learning.py](layout_learning.py) | Five-patch two-population, seven-patch deeper and seven-patch nested observer layouts. | The same public build/teach/query/save contract, fresh free predictions, exact resumption and separate work counts. |
| [layout_cost.py](layout_cost.py) | Six patches: state-only composition, composition with a sensory skip, and two observer levels. | Pure query cost at fixed parameters. No training or capability score. |
| [batch_bootstrap.py](batch_bootstrap.py) | Two populations of 12 and four patches, with direct sensory inputs to both. | Batched supervised preparation on two simple continuous relations, followed by fresh unclamped checks; separates startup, learning and test time across requested devices. |
| [cuda_qualification.py](cuda_qualification.py) | Existing single-example and batch tensor test fixtures. | Records CUDA correctness checks from a clean committed checkout, including exact source identity, hardware and per-case outcomes; `--full` includes the complete test suite. |
| [parallel_bootstrap.py](parallel_bootstrap.py) | Independent six-patch brains: populations of four and two patches, both reading supplied body inputs. | Learning one-step outcomes of a tiny moving body, with process parallelism across independent brains and ordered admissions within each brain. |
| [live_control.py](live_control.py) | Populations of four and two patches, both reading position and candidate action. | Learning a one-dimensional body's next position and querying candidate actions through `LiveController`, with actuator limits and actual outcome observations. |
| [live_learning.py](live_learning.py) | Six-patch coupled layouts for cue/retention gates; a nine-patch coupled layout for reward learning. | Explicit-history cue recall, retention with old-example replay, reward-based acquisition/reversal and saved continuation in separate small fixtures. |
| [temporal_credit.py](temporal_credit.py) | Memory fixture: four patches in each state-coupled and observer layout, with sensory skips. Reward fixture: two coupled populations. | Layout comparisons with explicit sensory history, erased-history and state-reset controls; separate delayed-reward/replay controls; saved continuation. |

The research scripts [credit_horizon.py](credit_horizon.py),
[credit_diagnostic.py](credit_diagnostic.py) and
[credit_exposure.py](credit_exposure.py) use the two-population reward fixture to
compare credit horizons, target construction and replay order. They preserve
explicit collection and assessment records; they are bounded research screens,
not claims of strategic learning or extra public API. `credit_exposure.py`
requires a diagnostic collection. Read each script's `--help` and declared
budgets before running a screen.

`layout_cost.py` and `temporal_credit.py` deliberately retain observer arms as
experimental comparisons. Their stored receipts remain historical; the layouts
in other examples do not turn those records into new-version results.

`History` is supplied external memory. `Reinforcement` supplies explicit
action-value targets and transition replay through the same learning API.
These fixtures do not establish learned recurrent memory or an integrated
autonomous agent. The live-control example supplies its action search and
waits for each simulated command; its callback interface is not a hard
real-time guarantee. See [live operation](../docs/LIVE.md).

Batch size changes the learning objective per update. Treat a batch comparison
as a learning-and-throughput comparison, and check both accuracy and work.
Parallel bootstrapping runs separate lives; it does not merge brains or make
one brain accept concurrent experiences. See
[bootstrapping](../docs/BOOTSTRAP.md) and
[acceleration](../docs/ACCELERATION.md).

## Run from the repository root

For the remaining examples, these commands select the checkout's implementation:

```sh
PYTHONPATH=src python examples/layout_cost.py --out /tmp/cadence-layout-cost.json
PYTHONPATH=src python examples/batch_bootstrap.py --devices python --batch-size 8 --repeats 1
PYTHONPATH=src python examples/parallel_bootstrap.py --lives 4 --workers 2
PYTHONPATH=src python examples/live_learning.py --seeds 0 2 7
PYTHONPATH=src python examples/temporal_credit.py --out /tmp/cadence-temporal-credit.json
```

Choose a fresh output path for `layout_cost.py`; it refuses to overwrite a
receipt and records its protocol before measurement. Learning examples can
take substantially longer than the query-cost probe. Optional tensor devices
in `batch_bootstrap.py` require the `gpu` extra and the corresponding runtime;
an unavailable requested device fails explicitly.

## Interpret saved receipts and public demos

Saved receipts describe their recorded source version; rerunning a command
on this checkout produces a new result. Historical Amen, Atari, Patchworld
and other application demonstrations used their own versioned engines,
models and body adapters. Their published scores or musical quality are not
current `0.62` reproduction results. See the
[demo evidence boundary](../docs/EXPERIMENTAL.md)
before comparing or replacing one.

- [Cadence 0.50.0 query-cost receipt](receipts/layout_cost.json): protocol, per-query
  outcomes, work counts, timings and implementation hashes. It includes an
  input-only arm that `0.61.0` no longer builds; the record stays as recorded.
  Equal patch counts do not make graph geometry or output-connected capacity
  equal; the composition arm also has fewer edges than the other arms.
- [Temporal-credit receipt](receipts/temporal_credit.json): the recorded
  qualification run with its original source hashes, explicit memory and
  credit controls. It is a historical run, not a fresh execution on every
  checkout; its equal-width layouts have different connection counts, and its
  input-only arms no longer build.
- [Historical comparison excerpts](receipts/layout_cost_history.json):
  source-hashed summaries of older CartPole and changing-body runs, with their
  versions and conditions. These are excerpts, not full reproduction bundles.
- [CUDA qualification receipt](receipts/cuda_audit_verified.json): the audited
  full-suite run on an NVIDIA RTX 4000 Ada laptop GPU at commit `8892927`,
  with source hashes before and after the run, hardware and library versions
  and every CUDA case outcome. Two earlier runs,
  [cuda_qualification.json](receipts/cuda_qualification.json) and
  [cuda_audit_encoding_failure.json](receipts/cuda_audit_encoding_failure.json),
  are retained for their failures: a platform-dependent sweep cap and a Windows
  subprocess encoding fault. They are history, not current results.

The [performance guide](../docs/PERFORMANCE.md) explains what these comparisons
support. Prefer a trained task comparison with declared information, capacity,
learning work and query cost when deciding whether recursive observation helps.
