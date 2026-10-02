<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: brains that settle into one equilibrium" width="100%">
</p>

# Cadence

[Documentation](docs/index.md) · [Quickstart](docs/QUICKSTART.md) · [Examples](examples/README.md) · [Interactive overview](https://floatingpragma.io/cadence/) · [Preprint](https://philpapers.org/rec/MUECAP-2) · [Pragma Research](https://floatingpragma.io/)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**Patches repair local disagreement to reach a coherent brain state. Further
repair is driven by that state's mismatch with reality.**

That is the whole idea. The coherent state is one equilibrium of the entire
brain, and it is the brain's world model.

Cadence is an experimental learning library built from bounded patches. Every
patch is an observer: it reads its ports, predicts its own state from what it
reads, and carries the disagreement between that prediction and its state as a
live error. Connected patches settle those disagreements together until the
whole brain is stationary. That settled state is the brain's answer and its
model of the world. A new observation, or an outcome that contradicts a
prediction, disturbs the equilibrium, and the brain pays exactly the repair the
disturbance requires. Learning is the same repair with the relations made
eligible. No loss is propagated backwards through layers.

**This README describes `0.61.0`.** Since this release the builder refuses a
layout in which a population settles with no other population, and a layout in
which a group of populations settles apart from the rest: a patch that reads
sensors alone and is read by nobody is an isolated regression, not part of a
brain, and two unconnected groups are two brains. Neither builds. The smallest
brain is two populations, one reading the sensors and one reading the first.

**Error readback is experimental.** Every patch repairs its own disagreement.
`observer(...)` additionally lets a population read other patches' current
errors as signals. Those contacts join every synchronous solve and may slow
routine work, and their benefit over state-coupled populations is unproven.
Automatic on-demand attention and independent population clocks are not public
features. See the [experimental capability boundary](docs/EXPERIMENTAL.md).

## Choose how the brain is wired

| Wiring | What patches read | When to use it |
| --- | --- | --- |
| **Two coupled populations** (the smallest brain) | One population reads the sensors, the next reads its live states. All states settle together; the later population's constraint moves the earlier states through the shared energy. | Every application starts here. |
| **Deeper composition** | Several populations reading earlier populations, parallel sensory branches, fusion. | Routines that need learned intermediate features or combined sensory information. |
| **Experimental error readback** | Live states and exact current prediction errors of other populations; an observer can itself be observed. | A declared experiment against a state-coupled control, with all extra work measured. |

All three use the **same patch rule**, learning methods and whole-brain
qualification. They are choices of wiring, not neuron classes or speed
settings. Latency depends on size, coupling, learning and the task.

**System 1** means learned routine competence; it can need several coupled
populations. **System 2** names the intended useful recursive correction when
routine competence fails. The complete cycle—cheap routine, useful correction,
then retained cheap routine—has not been demonstrated. A coherent changing
beat, a familiar game situation or walking a known path can all be routine.
Equilibrium in this behavioral sense means sustained competence, not an
unchanging output. Numerical settlement alone can still give a wrong answer
about the world: compare forecasts with later observations and measure actual
task outcomes.

Build the layout inside one `Cortex`. The application supplies observations,
executes actions and reports outcomes through one brain/body interface; it does
not attach an evaluator to every population. Start with the
[layout quickstarts](docs/VARIANTS.md). Explicit observer wiring
remains available in the [experimental recipes](docs/VARIANTS.md#experimental-recursive-observer-settlement).

## How this differs from a feed-forward network trained by backpropagation

A conventional network computes its answer in one pass, layer after layer, and
learns by propagating the derivative of a global loss backwards through those
layers. Cadence does neither.

| | Feed-forward network with backpropagation | Cadence brain |
| --- | --- | --- |
| What an answer is | The output of one pass through fixed layers | A settled equilibrium: every patch's live state has stopped disagreeing with its own prediction of that state, within tolerance, across the whole connected brain |
| Influence while answering | Forward only | Each patch settles against the ports it reads; a later population's constraint moves earlier states through the shared energy, so influence returns upstream during the same answer |
| Where errors live | Only during training, at the output | Every patch carries its own prediction error at all times; the errors are part of the equilibrium, and an observer can read another patch's current error as a signal |
| What learning changes | All weights, through a backward pass of the global loss | The relations of patches whose disagreements remain once an actual outcome is clamped: `observe` repairs live states and relation coefficients together against the same energy, anchored to the relations held before that experience |
| Admission | Every gradient step is applied | A repair is admitted only if the whole brain reaches a qualified equilibrium; a refusal keeps the previous states and relations |
| Training and running | Separate phases | One persistent brain; the same repair serves queries, retained activity (`step`) and learning (`observe`), including during live operation |
| Cost of an answer | Fixed per input | The repair sweeps the disturbance requires; an unchanged familiar input from retained activity may need no repair sweep, a surprising one many, and every answer is still evaluated and qualified |
| What it reports | A loss value | Whether the brain qualified, the work it did and its remaining prediction residual, separately from task accuracy |

What stays familiar: a patch's relation is a weighted sum with a bias through
`tanh`, and repair follows the analytic derivatives of the energy with a line
search. The energy, however, is the sum of local disagreements plus a state
prior, not a global output loss, and the coordinates being repaired include
the states themselves. The claim is the equilibrium: a world model is the
joint state the patches settle on, and a disturbance, a new observation or an
outcome that contradicts a prediction, is what calls for repair. The
[architecture guide](docs/DRSN.md#what-a-processing-patch-computes) gives the
exact energy and the [specification](docs/SPECIFICATION.md#repair-and-qualification)
the admission rule.

## Install

Python 3.11 or later. The default engine needs only the standard library:

```sh
python -m pip install "cadence-net==0.61.0"
python -c "import cadence; print(cadence.__version__)"
```

For reproducible work, retain the installed package and its source with saved
brains; see [checkpoint requirements](docs/REFERENCE.md#checkpoints).

Optional PyTorch execution uses the same learning rule and final reference
check. Install the `gpu` extra, then choose
`Cortex(device="cpu")`, `Cortex(device="mps")` or `Cortex(device="cuda")`:

```sh
python -m pip install "cadence-net[gpu]==0.61.0"
```

Small brains can be faster on the default engine. Measure the complete workload;
see [devices, precision and batching](docs/ACCELERATION.md).

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
[live-control example](examples/live_control.py) uses a learned model to compare
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
| [Layout learning](examples/layout_learning.py) | Start with the two-population brain, then try deeper composition; check fresh predictions, work and exact saved continuation. The optional `--layout recursive` experiment has different capacity and does not establish an advantage. |
| [Learned body control](examples/live_control.py) | Bootstrap a body model, select actions through explicit candidate search, execute them and admit actual outcomes. |
| [History, retention and rewards](examples/live_learning.py) | Separate small tests of explicit sensory history, old-skill replay, reward learning/reversal and saved continuation. |

[All examples](examples/README.md) include batch learning, independent parallel
brains, delayed reward and layout costs. `History` supplies explicit external
memory; `Reinforcement` supplies discrete action-value learning and replay.
Neither is an automatic planner or a guarantee of long-term success.

The [public demos](https://github.com/muellerberndt/cadence-demos) also include
Amen, Atari, Patch World and Doom experiments. Their original engines and
results have different versions and representations. They are **historical
application evidence**, not completed reproductions on `0.61.0`. In particular,
the original Amen record-cell brain is not equivalent to one current
population. Consult the [versioned performance evidence](docs/PERFORMANCE.md) and
[current capability boundary](docs/EXPERIMENTAL.md) before comparing them.

## Learn more

| Guide | What it helps you do |
| --- | --- |
| [Quickstart](docs/QUICKSTART.md) | Build, teach, query and save your first brain |
| [Layout quickstarts](docs/VARIANTS.md) | Construct coupled and deeper populations, then optional observer experiments |
| [Experimental capabilities](docs/EXPERIMENTAL.md) | Understand observer costs, unfinished System 2 behavior and current evidence limits |
| [Brain design](docs/BRAIN_DESIGN.md) | Choose sufficient observations, connected capacity and useful evaluation checks |
| [Bootstrapping](docs/BOOTSTRAP.md) | Prepare a skill and measure acquisition, retention and learning cost |
| [Live operation](docs/LIVE.md) | Connect observations, actual outcomes, history, reward and control callbacks |
| [Agent recipe](docs/AGENTS.md) | Build integrations with the right contracts and capability claims |
| [Architecture](docs/DRSN.md) / [API reference](docs/REFERENCE.md) / [Specification](docs/SPECIFICATION.md) | Understand the equations, exact calls and numerical guarantees |

Qualification means constrained numerical stationarity, not a unique global
minimum or task success. Saved brains bind exact implementation sources; retain
those sources and application preprocessing with checkpoints. Broader capability
and comparative efficiency require measured task evidence.

## Development

Changes follow **minimalism**, **user-friendliness** and **agent-friendliness**,
and the principle in the [contributor instructions](AGENTS.md): every brain is
one equilibrium of patches settling against each other.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests execute the README and documentation examples and check learning,
mathematical derivatives, refusal and checkpoint continuation.
Licensed under [GPL-3.0-or-later](LICENSE).
