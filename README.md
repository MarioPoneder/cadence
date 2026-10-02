<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: brains that settle into one equilibrium" width="100%">
</p>

# Cadence

[Documentation](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/index.md) · [Quickstart](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/QUICKSTART.md) · [Examples](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/README.md) · [Interactive overview](https://floatingpragma.io/cadence/) · [Preprint](https://philpapers.org/rec/MUECAP-2) · [Pragma Research](https://floatingpragma.io/)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/muellerberndt/cadence/blob/v0.62.0/LICENSE)

**Patches repair local disagreement to reach a coherent brain state. Further
repair is driven by that state's mismatch with reality.**

Cadence is an alpha learning library built from bounded patches with local
state, ports, prediction-error readback and retained relations. Input samples
are held fixed while connected patches repair a shared state. The whole
settled state is the brain's current interpretation; selected patch states
provide its outputs. Actual outcomes can also be clamped, allowing the same
repair to change learned relations.

**This README describes `0.62.0`.** Every compiled patch must belong to one
connected network of state or error contacts. Shared sensor inputs alone do
not connect patches. This is checked after sparse wiring is resolved, as well
as at the population level.

Connected does not mean fully connected. Sparse chains, branches and modules
joined by a few contacts can all repair one shared state. Independent groups
can be useful separate computations; this builder requires a repair path
between the patches declared as one brain.

## One network, different connections

There are no flat/deep operating modes. Size tells you how many patches there
are; wiring tells you which disagreements can influence one another. The two
matter independently. Populations are convenient groups of the same patch
primitive. Start with a sensing population and a response population that reads
it. Add connected capacity or branches when a measured task needs them.

`column(..., inputs=population)` reads live states. Its constraints influence
the source population through the joint repair. The optional
`observer(..., observes=population)` also reads exact current errors. Both
participate in the same equilibrium; an observer is never a separately settled
critic. See [wiring examples](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/VARIANTS.md) and the
[recursive observation experiment](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/EXPERIMENTAL.md).

## Repair, reality and memory

Use `step` for live operation: it starts from retained activity and commits a
qualified new state. An unchanged equilibrium needs no repair sweeps. Changed
inputs or factual outcome clamps can disturb it; the solver performs the work
needed to meet the same tolerance, within its budget. The numerical disturbance
is distinct from surprise measured against an earlier forecast. Neither one
alone tells the brain whether a long-term goal was achieved.

`observe` retains changes to both activity and learned relations after a
qualified experience. Those relations carry learned information across calls
and saves. Retained activity is a warm start, not a guarantee of sequence
memory, and later learning can overwrite earlier skills. For tasks requiring
explicit recent context, `History` provides bounded storage. Automatic memory
allocation, protected consolidation and surprise-gated recursive attention
remain research work; [the experimental guide](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/EXPERIMENTAL.md) describes
the integration contract.

Every answer is checked against the whole brain. Equilibrium here means that
no eligible projected repair direction exceeds the tolerance. Competing
constraints may leave prediction errors, and a settled answer may still be
wrong about reality. Evaluate free predictions against actual observations.

## How the computation works

Each patch predicts its state with a weighted `tanh` relation. Repair minimizes
the sum of local squared disagreements plus a state prior. Queries adjust
activity; learning also adjusts relations under a prior anchored to the
preceding experience. A refusal preserves the previous continuation.

The implementation uses analytic gradients, including a reverse traversal to
carry returning state and error influence. Its distinction from a feed-forward
predictor is the jointly adjustable activity and equilibrium answer, not the
absence of derivative computation. The synchronized reference solver does not
establish asynchronous distributed convergence. See the
[patch equations](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/DRSN.md#what-a-processing-patch-computes) and
[qualification contract](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/SPECIFICATION.md#repair-and-qualification).

## Install

Python 3.11 or later. The default engine needs only the standard library:

```sh
python -m pip install "cadence-net==0.62.0"
python -c "import cadence; print(cadence.__version__)"
```

For reproducible work, retain the installed package and its source with saved
brains; see [checkpoint requirements](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/REFERENCE.md#checkpoints).

Optional PyTorch execution uses the same learning rule and final reference
check. Install the `gpu` extra, then choose
`Cortex(device="cpu")`, `Cortex(device="mps")` or `Cortex(device="cuda")`:

```sh
python -m pip install "cadence-net[gpu]==0.62.0"
```

Small brains can be faster on the default engine. Measure the complete workload;
see [devices, precision and batching](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/ACCELERATION.md).

## Teach a small body model

This brain learns how a supplied one-dimensional simulator moves. It sees
position and commanded velocity, then predicts the next position. Four
`features` patches read the body and one `response` patch reads them; the two
populations settle against each other, so the forecast is one equilibrium of
five patches. Teaching, readiness checks and final probes use different
inputs. Predictions come from the settled brain; the simulator supplies only
measured teaching and test values.

```python
from cadence import Brain, Cortex, bootstrap

# Supplied simulator: one quarter-second of movement.
def advance(position, velocity):
    return position + 0.25 * velocity


def measurements(positions, velocities):
    return [
        ({"body": [x, u]}, {"next_position": [advance(x, u)]})
        for x in positions for u in velocities
    ]


layout = Cortex(seed=2)
body = layout.input("body", shape=2)
features = layout.column("features", patches=4, inputs=body)
response = layout.column("response", patches=1, inputs=features)
layout.output("next_position", shape=1, reads=response)
brain = layout.build()

report = bootstrap(
    brain,
    measurements((-0.5, 0.5), (-0.8, 0.8)),
    checks=measurements((-0.25, 0.25), (-0.4, 0.4)),
    max_error=0.06, epochs=30,
)
assert report["passed"], report

# Final free predictions: no answer is clamped or supplied as an input.
for position, velocity in ((0.35, -0.4), (-0.35, 0.4)):
    forecast = brain.predict({"body": [position, velocity]})["next_position"][0]
    assert abs(forecast - advance(position, velocity)) < 0.06

# Live operation: retain activity, execute, then learn the actual consequence.
inputs = {"body": [0.35, -0.4]}
activity = brain.step(inputs)
assert activity["accepted"]
measured = advance(0.35, -0.4)
assert brain.observe(inputs, {"next_position": [measured]})["accepted"]

saved = brain.snapshot()  # JSON text: state, learned relations and source identity
restored = Brain.from_snapshot(saved)
assert restored.predict(inputs) == brain.predict(inputs)
```

This learns a small forward model, not a navigation policy. The
[live-control example](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/live_control.py) uses a learned model to compare
candidate actions and move an actual simulated body toward a goal. Its action
search is supplied application code. It does not demonstrate automatic attention
or a benefit from recursion.

`settle` and `predict` query without changing the brain. `step` retains qualified
activity. `observe` also learns from supplied output witnesses; `observe_batch`
learns from several independent examples while preserving live activity.
Check `qualified` or `accepted`; `predict` raises `SettlementError` on refusal.
A teaching clamp matching its target is not evidence of learning—check later
predictions without targets.

## Run the current examples

From a checkout of this version:

```sh
python -m pip install -e .
python examples/layout_learning.py
python examples/layout_learning.py --layout deep
python examples/live_control.py --decisions 20 --seed 0
python examples/live_learning.py --seeds 0 2 7
```

| Example | What you can verify |
| --- | --- |
| [Layout learning](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/layout_learning.py) | Start with the two-population brain, then try additional connected populations; check fresh predictions, work and exact saved continuation. The optional `--layout recursive` experiment has different capacity and does not establish an advantage. |
| [Learned body control](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/live_control.py) | Bootstrap a body model, select actions through explicit candidate search, execute them and admit actual outcomes. |
| [History, retention and rewards](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/live_learning.py) | Separate small tests of explicit sensory history, old-skill replay, reward learning/reversal and saved continuation. |

[All examples](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/README.md) include batch learning, independent parallel
brains, delayed reward and layout costs. `History` supplies explicit external
memory; `Reinforcement` supplies discrete action-value learning and replay.
Neither is an automatic planner or a guarantee of long-term success.

The [public demos](https://github.com/muellerberndt/cadence-demos) also include
Amen, Atari, Patch World and Doom experiments. Their original engines and
results have different versions and representations. They are **historical
application evidence**, not completed reproductions on `0.62.0`. In particular,
the original Amen record-cell brain is not equivalent to one current
population. Consult the [versioned performance evidence](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/PERFORMANCE.md) and
[current capability boundary](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/EXPERIMENTAL.md) before comparing them.

## Learn more

| Guide | What it helps you do |
| --- | --- |
| [Quickstart](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/QUICKSTART.md) | Build, teach, query and save your first brain |
| [Layout quickstarts](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/VARIANTS.md) | Choose connections and size, then optional error readback |
| [Experimental capabilities](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/EXPERIMENTAL.md) | Understand observer costs, unfinished System 2 behavior and current evidence limits |
| [Brain design](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/BRAIN_DESIGN.md) | Choose sufficient observations, connected capacity and useful evaluation checks |
| [Bootstrapping](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/BOOTSTRAP.md) | Prepare a skill and measure acquisition, retention and learning cost |
| [Live operation](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/LIVE.md) | Connect observations, actual outcomes, history, reward and control callbacks |
| [Agent recipe](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/AGENTS.md) | Build integrations with the right contracts and capability claims |
| [Architecture](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/DRSN.md) / [API reference](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/REFERENCE.md) / [Specification](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/SPECIFICATION.md) | Understand the equations, exact calls and numerical guarantees |

Qualification means constrained numerical stationarity, not a unique global
minimum or task success. Saved brains bind exact implementation sources; retain
those sources and application preprocessing with checkpoints. Broader capability
and comparative efficiency require measured task evidence.

## Development

Changes follow **minimalism**, **user-friendliness** and **agent-friendliness**,
and the principle in the [contributor instructions](https://github.com/muellerberndt/cadence/blob/v0.62.0/AGENTS.md): every brain is
one equilibrium of patches settling against each other.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests execute the README and documentation examples and check learning,
mathematical derivatives, refusal and checkpoint continuation.
Licensed under [GPL-3.0-or-later](https://github.com/muellerberndt/cadence/blob/v0.62.0/LICENSE).
