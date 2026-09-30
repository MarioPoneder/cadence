# Cadence examples

These examples use the current `Cortex`/`Brain` API. Start with the
[design patterns](../docs/VARIANTS.md) for runnable constructions of the three
layouts, and the [performance guide](../docs/PERFORMANCE.md) for their measured
costs and evidence limits.

## Three patterns, one settlement rule

| Pattern | Connections | Useful starting point |
| --- | --- | --- |
| Flat / input-only | Patches read external inputs without reading other patches. | Small direct input-to-output relations and an inexpensive baseline. |
| State-coupled / ordinary composition | Populations read other populations' live states, without error ports; they may also read external inputs. | Learned intermediate representations and a control for the value of error readback. |
| Recursive observer | Observers read other populations' live states and exact prediction errors; an observer can itself be observed. | Testing whether internal state-and-error readback improves the task enough to justify its cost. |

**All three patterns settle.** They use the same bounded patches, energy,
repair rule and numerical qualification. State-coupled and recursive layouts
solve their interacting states together; they do not chain completed layer
predictions. “States-only” describes their internal contacts, not the absence
of external sensory inputs. An unused flat patch supplies no hidden capacity
to another output patch.

For every layout, `settle` and `predict` are pure queries, `step` retains
qualified activity, and `observe` learns from supplied output witnesses.
Check `qualified` or `accepted`; `predict` raises on refusal. Numerical
qualification and useful learned behavior are separate checks.

## Choose an example

| Example | Actual layout | What it demonstrates |
| --- | --- | --- |
| [layout_cost.py](layout_cost.py) | Six patches: input-only flat, state-only composition, composition with a sensory skip, and two observer levels. | Pure query cost at fixed parameters, including an independent exact-optimum check for flat queries. No training or capability score. |
| [batch_bootstrap.py](batch_bootstrap.py) | 12 processing patches and four observers, with direct sensory inputs to both populations. | Batched supervised preparation on two simple continuous relations, followed by fresh unclamped checks; separates startup, learning and test time across requested devices. |
| [parallel_bootstrap.py](parallel_bootstrap.py) | Independent six-patch brains: four processing patches and two observers, both reading supplied body inputs. | Learning one-step outcomes of a tiny moving body, with process parallelism across independent brains and ordered admissions within each brain. |
| [live_control.py](live_control.py) | Four processing patches and two observers, both reading position and candidate action. | Learning a one-dimensional body's next position and querying candidate actions through `LiveController`, with actuator limits and actual outcome observations. |
| [live_learning.py](live_learning.py) | Six-patch observer layouts for cue/retention gates; a nine-patch observer layout for reward learning. | Explicit-history cue recall, retention with old-example replay, reward-based acquisition/reversal and saved continuation in separate small fixtures. |
| [temporal_credit.py](temporal_credit.py) | Memory fixture: four patches in each flat, ordinary-composed and observer layout, with sensory skips in the latter two. Reward fixture: four input-only patches. | Layout comparisons with explicit sensory history, erased-history and state-reset controls; separate delayed-reward/replay controls; saved continuation. |

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

Use Python 3.11 or later. These commands select the checkout's implementation:

```sh
PYTHONPATH=src python examples/layout_cost.py --out /tmp/cadence-layout-cost.json
PYTHONPATH=src python examples/batch_bootstrap.py --devices python --batch-size 8 --repeats 1
PYTHONPATH=src python examples/parallel_bootstrap.py --lives 4 --workers 2
PYTHONPATH=src python examples/live_control.py --decisions 20 --seed 0
PYTHONPATH=src python examples/live_learning.py --seeds 0 2 7
PYTHONPATH=src python examples/temporal_credit.py --out /tmp/cadence-temporal-credit.json
```

Choose a fresh output path for `layout_cost.py`; it refuses to overwrite a
receipt and records its protocol before measurement. Learning examples can
take substantially longer than the query-cost probe. Optional tensor devices
in `batch_bootstrap.py` require the `gpu` extra and the corresponding runtime;
an unavailable requested device fails explicitly.

## Inspect the receipts

- [Current query-cost receipt](receipts/layout_cost.json): protocol, per-query
  outcomes, work counts, timings and implementation hashes. Equal patch counts
  do not make graph geometry or output-connected capacity equal; the ordinary
  composition arm also has fewer edges than the other three arms.
- [Temporal-credit receipt](receipts/temporal_credit.json): the recorded
  qualification run with its original source hashes, explicit memory and
  credit controls. It is a historical run, not a fresh execution on every
  checkout; its equal-width layouts have different connection counts.
- [Historical comparison excerpts](receipts/layout_cost_history.json):
  source-hashed summaries of older CartPole and changing-body runs, with their
  versions and conditions. These are excerpts, not full reproduction bundles.

The [performance guide](../docs/PERFORMANCE.md) explains what these comparisons
support. Prefer a trained task comparison with declared information, capacity,
learning work and query cost when deciding whether recursive observation helps.
