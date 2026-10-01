<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: learning through flat, state-coupled and recursive settlement" width="100%">
</p>

# Cadence

[Interactive overview](https://floatingpragma.io/cadence/) · [Public demos](https://github.com/muellerberndt/cadence-demos) · [Preprint](https://philpapers.org/rec/MUECAP-2) · [Documentation](docs/index.md) · [Pragma Research / investors](https://floatingpragma.io/investors/)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**Brains that learn through joint settlement.**

Cadence is an experimental learning architecture from
[Pragma Research](https://floatingpragma.io/), with embodied AI as a target
application. Its building blocks are bounded patches with local state, sensory
and output ports, prediction errors and a shared repair rule. Connected patches
settle together to produce an answer; experience can change their retained
parameters. Recursive observers read other patches' states and exact errors,
feeding back into the same solve. This makes the brain an observer-like,
self-reading system.

The research goal is an alternative to transformers that learns throughout a
continuing life. The library supplies supervised learning, reward-learning
helpers and saved continuation. The evidence below covers small simulated tasks;
broad capability, physical-robot deployment and comparative energy efficiency
require separate validation.

## Demonstrated behavior

| Result | Evidence and interpretation |
| --- | --- |
| **Doom Basic: 63/64 and 64/64 wins** | Two selected descendants improved native return over their respective founder comparisons in separate reserved confirmations. This is single-room combat after finite teaching and simulator practice, without a demonstrated benefit from recursive depth. [Experiment record and receipts](https://github.com/muellerberndt/cadence-demos/blob/main/doom-lab/docs/EXPERIMENTS.md). |
| **CartPole: 180/180 episodes at the 500-step ceiling** | One-patch flat, six-patch flat and four-plus-two observer layouts, with three seeds each. Each model received 128 numerical-state teaching examples on Cadence 0.43.0. Even one flat patch reached the ceiling. [Source-hashed receipt excerpts](https://floatingpragma.io/evidence/cadence/cartpole-confirmations.json). |
| **Learning and continuation in the core library** | Small tests check acquired relations on unclamped inputs across flat, composed and recursive layouts, including restored checkpoints. The [runnable examples](examples/README.md) cover live control, explicit history, replay and layout costs. |

These results use their declared versions and task interfaces. The
[performance guide](docs/PERFORMANCE.md) separates query cost, learning work
and complete control decisions. It does not establish a general speed or
capability advantage over other architectures.

For an application to explore, start with
[Doom Lab](https://github.com/muellerberndt/cadence-demos/tree/main/doom-lab)
or compare all three layouts on a changed simulated body in
[Rover Lab](https://github.com/muellerberndt/cadence-demos/tree/main/rover-lab).
Both have local setup instructions. The
[browser explorer](https://floatingpragma.io/cadence/#inside-the-brain) illustrates the
settlement mechanism with a small deterministic model.

## Three settlement design patterns

| Pattern | What patches read | Useful starting point |
| --- | --- | --- |
| **Flat settlement** | Fixed sensory inputs | Small direct sensor-to-answer relations and an inexpensive baseline |
| **State-coupled settlement** | Other populations' live states, optionally with sensory inputs | Learned intermediate representations and sensory fusion |
| **Recursive observer settlement** | Live states and exact prediction errors, including other observers' | Testing whether internal state-and-error feedback improves decisions |

**All three settle.** They share the patch rule, learning API and numerical
qualification, and can be combined in one brain. Coupled populations participate
in one solve; they do not chain completed predictions. Start with the smallest
layout that learns the behavior and measure the benefit of added coupling.
Ordinary deep layouts already return influence through the common energy.
Observers add an explicit residual channel; they are not separate evaluators.
A capable routine response can need several ordinary layers. The
[design-pattern quickstarts](docs/VARIANTS.md) teach and resume all three through
one external interface; [brain design](docs/BRAIN_DESIGN.md) explains when to
try each and how to measure its cost.

## Install

Python 3.11 or later. The default engine uses only the standard library.
The published baseline is Cadence 0.50.0:

```sh
python -m pip install "cadence-net==0.50.0"
```

Packages are also available from
[GitHub Releases](https://github.com/muellerberndt/cadence/releases).
For optional PyTorch execution on CPU, Apple Silicon GPU or NVIDIA GPU:

```sh
python -m pip install "cadence-net[gpu]==0.50.0"
```

This checkout also contains an unreleased 0.60 candidate. Its changed outcome
API requires installing the reviewed checkout with `python -m pip install -e .`;
see [migration](docs/MIGRATION_060.md). Automatic internal attention and the
complete routine–disturbance–correction cycle are not yet supplied.

Choose `Cortex(device="mps")` or `Cortex(device="cuda")` for the corresponding
GPU. Execution uses the same learning rule and final float64 admission check.
Small brains can be faster with the default Python engine; see
[execution and precision](docs/ACCELERATION.md).

## Learn a relation and keep learning

A `Cortex` declares a layout; `build()` returns its persistent `Brain`.
This one-patch flat layout learns a signed input/output relation. Deep ordinary
and recursive layouts use these same calls.

```python
from cadence import Brain, Cortex, bootstrap

layout = Cortex(seed=2)
signal = layout.input("signal", shape=(1,))
response = layout.column("response", patches=1, inputs=signal)
layout.output("answer", shape=(1,), reads=response)
brain = layout.build()

# Bootstrapping phase: learn from supplied outcomes, then check without clamps.
examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
checks = [({"signal": [x]}, {"answer": [x]}) for x in (-0.4, 0.4)]
report = bootstrap(brain, examples, checks=checks, max_error=0.2)
assert report["passed"], report

# Fresh inputs, absent from teaching and readiness checks.
assert brain.predict({"signal": [-0.6]})["answer"][0] < -0.3
assert brain.predict({"signal": [0.6]})["answer"][0] > 0.3

# Live phase: retain qualified activity and learn when an actual outcome arrives.
assert brain.step({"signal": [0.3]})["accepted"]
assert brain.observe({"signal": [0.3]}, {"answer": [0.3]})["accepted"]

# A JSON snapshot preserves state and learned parameters for continuation.
saved = brain.snapshot()
restored = Brain.from_snapshot(saved)
assert restored.predict({"signal": [0.6]}) == brain.predict({"signal": [0.6]})
```

`settle` and `predict` are pure queries. `step` retains qualified live state;
`observe` also learns from supplied output witnesses. Check `qualified` or
`accepted`; `predict` raises `SettlementError` on refusal. A clamped teaching
output is not evidence of learning: evaluate later predictions without targets.

The **bootstrapping phase** and **live phase** use the same patch rule and
retained parameters. Applications supply sensors, teaching evidence and actuator
interpretation. Batch learning, explicit sensory `History`, discrete
`Reinforcement` and `LiveController` are documented below.

## Scope and reproducibility

The reference engine uses analytic derivatives and bounded descent to repair
a nonlinear residual energy. Qualification checks constrained stationarity;
it does not guarantee zero prediction error or a unique global minimum.
Budget exhaustion can cause refusal. Numerical qualification, useful behavior
and control-loop deadlines are separate measurements.

Checkpoints bind implementation sources. Use the runtime and reproduction
instructions associated with each receipt; a package version alone does not
identify every experimental artifact. Explicit history is external memory,
and retention tests with replay do not establish indefinite learned memory.
The [specification](docs/SPECIFICATION.md) states the exact contract.

| Documentation | Start here for |
| --- | --- |
| [Quickstart](docs/QUICKSTART.md) | Queries, teaching, diagnostics and saved continuation |
| [Brain design](docs/BRAIN_DESIGN.md) | Choose sensors, context, reachable capacity and learning budgets; measure useful behavior and cost |
| [0.60 candidate migration](docs/MIGRATION_060.md) | Unreleased changes, executed-outcome ownership and versioned demo reproduction |
| [Bootstrapping](docs/BOOTSTRAP.md) | Calibration, individual/batch learning and readiness checks |
| [Live operation](docs/LIVE.md) | History, reward credit, replay and control callbacks |
| [Architecture guide](docs/DRSN.md) | Population layouts, recursive observation and equations |
| [Performance](docs/PERFORMANCE.md) | Layout costs, versioned evidence and capability comparisons |
| [API reference](docs/REFERENCE.md) | Public methods, configuration and refusal behavior |

## Development

Core changes follow three principles: **minimalism**, **user-friendliness** and
**agent-friendliness**. A small public API, runnable examples and inspectable
state keep the mechanism usable. See the [contributor instructions](AGENTS.md).

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

The test suite executes the README and documentation examples and checks
learning, mathematical derivatives, refusal and checkpoint continuation.
Licensed under [GPL-3.0-or-later](LICENSE).
