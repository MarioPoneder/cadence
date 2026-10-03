<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: a continuing brain with local state, memory and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/README.md) · [Application demos](https://github.com/muellerberndt/cadence-demos) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/)

[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**One continuing equilibrium brain: learn a world, act in it, repair what fails.**

Cadence is not a feed-forward deep neural network. Its primary application is
one acquired brain continuing through experience.

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. Its default **System 1** is a continuing animal-like brain with
perception, plastic connections, memory and action. Optional **System 2** adds
recursive feedback through observing cortical regions in the same neural graph.
The base can already be deep and modular. Biological names describe functional
software roles.

Bounded, observer-like regions carry local neural state. The composition declares
sensory and motor indices, exposes activity for readback and retains working
traces and cue/outcome associations. Reciprocal regions constrain one another
as the neural graph settles; actions require its full equations to qualify.
Actual observations and consequences guide local plasticity and memory writes.
The aim is to bootstrap a useful interpretation, use it in an ongoing life, and
repair witnessed failures while retaining the same acquired brain.

Learned parameters and memories support a **family of equilibria** under changing
evidence and context. They do not hold the brain at one frozen state. An internally
consistent answer can still be wrong about the world: numerical settlement and
useful understanding need separate evidence.

Start with [one continuing equilibrium brain](https://github.com/muellerberndt/cadence/blob/main/docs/world-model.md), the canonical
guide to this lifecycle and its current implementation boundaries. `Brain.compose`
already supplies continuing action, local learning and memory. Integrated learned
world prediction and automatic failure-triggered repair with cheap stable
operation remain development goals. Cadence is alpha research software.

The examples and guides use Cadence 0.72.0, including action diagnostics through
`Brain.last_settlement`.

## How a Cadence brain differs from a feed-forward network

`Brain.compose(inputs=4, actions=2, modules=(64, 32, 16))` builds reciprocal
processing regions whose activity settles together. Later regions can shape
earlier ones while an answer forms. Working and associative memory contribute to
the current drive; the selected motor action comes from the qualified neural
state. An external trained readout would be a separate answer-producing model.

The graph learner compares local activities in free and nudged phases, without
a backward differentiation pass through those phases. Default teaching and
reward eligibility are finite; `LearnerConfig(qualified=True, ...)` explicitly
requires supervised phases to qualify before updating. Other Cadence model
families have [their own learning contracts](https://github.com/muellerberndt/cadence/blob/main/docs/contracts.md), including
explicit adjoints in record and belief models.

Independent classification and calibration remain useful mechanism tests.
They do not exercise the complete continuing brain. Recurrent networks trained
by backpropagation can also learn online and use memory: the relevant comparison
is acquired behavior and total work on the same task and information, not an
architectural label. See [the update mechanisms](https://github.com/muellerberndt/cadence/blob/main/docs/concepts.md#compared-with-backprop-networks).

## Why Cadence

The library makes continuing state, reciprocal interpretation, local plasticity
and memory available in one small interface. A saved brain can resume an action
awaiting its actual outcome. Private imagination can inspect responses without
rewriting live experience. These are mechanisms for testing acquisition,
retention and recovery through a life.

The target includes full-sentence language and reusable world understanding.
A small fixed-label task is a control, not that destination. Greater capability,
scalability and efficiency than transformers must be established with matched
comparisons. A small numerical residual does not measure low physical energy
or guarantee a correct answer.

<a id="get-started"></a>

## Start with System 1

Python 3.11+ and NumPy are required.
Install the published release for this basic example:

```sh
python -m pip install cadence-net==0.72.0
```

Its [released documentation](https://github.com/muellerberndt/cadence/blob/v0.72.0/docs/README.md)
describes the APIs included in that package.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, modules=(16, 8), seed=7)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)

# A tiny environment rewards action 0 and supplies the next observation.
reward = (action == 0).astype(float)
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.step(next_observation, reward=reward, done=np.array([False]))
assert action.shape == (1,)
```

`step` learns from the **preceding action's** measured reward, then chooses the
next action. `teacher=` can label the **current observation**. Keep each batch
row attached to the same life. There is no training/inference mode switch.
[Continuous interaction](https://github.com/muellerberndt/cadence/blob/main/docs/continuous.md)
covers teaching, resets and saved continuation.
The [continuing brain example](https://github.com/muellerberndt/cadence/blob/main/examples/continuing_brain.py) keeps the same brain
through bootstrap, unchanged conditions, disruption and correction, then checks
a checkpoint awaiting feedback. It records task outcomes and free-answer work
separately, and teaches a repeated cue only after a witnessed mistake. Real
reward learning and associative writes still process every observed outcome.

The constructor includes a working trace and fast/persistent associative memory.
The trace carries recent activity; learned graph parameters and consolidated
associations retain changes across resets. A current teacher changes graph
parameters. Actual chosen-action outcomes write associative memory. `act` reads
both memory pathways; independent `predict` and `accuracy` read neither, so they
measure the graph's learned response. Capacity is finite; overlapping associations
and further plasticity can interfere with recall.

```python
phases = brain.imagine([observation, next_observation])
assert phases  # Inspect phase.converged before using an imagined response.
```

Imagination carries a private trace without changing live memory, random state
or pending feedback. It evaluates responses to the observations you supply.
For learned environmental consequences and action planning, use the separate
[temporal model](https://github.com/muellerberndt/cadence/blob/main/docs/interaction.md).

<a id="development-checkout"></a>

To work from source, install from the library repository root:

```sh
python -m pip install -e .
```

This installs the local source as an editable package.

The API keeps each completed action solve's residual, qualification,
sweeps and check counts in read-only `brain.last_settlement`, including refused
attempts. These diagnostics exclude learning, reward eligibility and memory work.
The [quickstart](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md#inspect-the-work-of-answering) shows how to
inspect them; the continuing example also records other settling phases.
Cheap routine operation and a useful response to surprise must be measured,
not inferred from a small residual.

## Add optional System 2

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
)
```

Observer regions read and return influence to the base, motor regions and earlier
observers. They join the same settlement and use the same interaction interface.
This makes recursive feedback available; learning when it helps remains a task
for experience and evaluation.

Actions and independent predictions require the full neural equation residual
to meet the configured tolerance. Exhausting the budget refuses an action without
changing its live state, memory or pending feedback. If `step` has learned a real
outcome before the next action refuses, retry `act` without submitting that reward
again. Numerical damping stays within the total budget and checks the original
equations. It does not change the teaching rule. See [contracts](https://github.com/muellerberndt/cadence/blob/main/docs/contracts.md).

## Go further

[Build a brain](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md)
for custom wiring, [memory](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md)
for traces and associations, and [the memory/planning example](https://github.com/muellerberndt/cadence/blob/main/examples/memory_imagination.py)
for a bounded demonstration with actual toy-body outcomes.
[Record patches](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md)
provide event records and consolidation. The advanced
[population solver](https://github.com/muellerberndt/cadence/blob/main/docs/equilibrium/index.md)
provides exact state-and-error readback under its own numerical contract.

[cadence-demos](https://github.com/muellerberndt/cadence-demos) contains the active
application demos. [cadence-examples](https://github.com/muellerberndt/cadence-examples)
preserves research examples and viewer tools with their own declared library pins.

[Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/index.md) ·
[API](https://github.com/muellerberndt/cadence/blob/main/docs/api.md) ·
[Contributing](https://github.com/muellerberndt/cadence/blob/main/CONTRIBUTING.md) ·
[Changelog](https://github.com/muellerberndt/cadence/blob/main/CHANGELOG.md) ·
[Research tasks](https://github.com/muellerberndt/cadence/issues)

Licensed under [GPL-3.0](https://github.com/muellerberndt/cadence/blob/main/LICENSE).
