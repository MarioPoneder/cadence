<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: a continuing brain with local state, memory and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Demos](https://floatingpragma.io/demos/) · [Documentation](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/index.md) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/)

[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/muellerberndt/cadence/blob/v0.70.0/LICENSE)

**An experimental brain that learns, remembers, imagines and acts.**

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. Its default **System 1** is a continuing animal-like brain with
perception, plastic connections, memory and action. Optional **System 2** adds
recursive feedback through observing cortical regions in the same neural graph.
The base can already be deep and modular.

Bounded, observer-like regions carry local state, communicate through ports,
read back activity and retain records. They repair disagreement to find a coherent
state; actual observations and consequences guide learning. A settled answer can
still be wrong about the world, so capability is measured through free behavior.
Cadence is alpha research software, not a claim of human-level intelligence.

## Why Cadence

Animal and human brains learn from experience and not by backpropagation with
gradient descent. Cadence is designed the same way. Each synapse changes from the
activity of the two neurons it connects, compared between a free settled state
and one nudged by the outcome. Reward scales that change, and memory writes are
local too. The brain keeps no backward computation graph and has no separate
training mode.

For embodied AI this design gives:

- **Learning on the job.** The same brain acts and learns from every measured
  outcome. There is no difference between training and inference.
- **Adaptation to changed conditions.** A changed body or world shows up in what
  the brain measures, and the live brain adjusts without being told what changed.
- **Local learning.** No backward pass through the network is needed, so learning
  can run where the brain runs.
- **Memory.** A working trace carries the recent past, and fast and persistent
  associative memory keep what mattered.
- **Settled answers.** Every action is a qualified settled state of the whole
  brain. A brain that does not settle refuses to act.
- **Inspection.** Region activity can be read while the brain runs, and private
  imagination tests a response before the body commits to it.

Under idealized conditions the local contrast follows the gradient that
backpropagation would compute. [The learning rule](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/learning.md)
states those conditions, and [the comparison](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/concepts.md#compared-with-backprop-networks)
is of update mechanisms. These properties are shown in simulation in the
[demos](https://floatingpragma.io/demos/). An advantage on a physical robot is a
separate test.

<a id="get-started"></a>

## Start with System 1

Python 3.11+ and NumPy are required.

```sh
python -m pip install cadence-net==0.70.0
```

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
[Continuous interaction](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/continuous.md)
covers teaching, resets and saved continuation.

The constructor includes a working trace and fast/persistent associative memory.
Earlier activity can affect later answers, and actual outcomes change associations.
Capacity is finite; correlated memories can interfere.

```python
phases = brain.imagine([observation, next_observation])
assert phases  # Inspect phase.converged before using an imagined response.
```

Imagination carries a private trace without changing live memory, random state
or pending feedback. It evaluates responses to the observations you supply.
For learned environmental consequences and action planning, use the separate
[temporal model](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/interaction.md).

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
again. Numerical damping stays within the total budget and does not change the
teaching rule. See [contracts](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/contracts.md).

## Go further

[Build a brain](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/brain.md)
for custom wiring, [memory](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/memory.md)
for traces and associations, and [the memory/planning example](https://github.com/muellerberndt/cadence/blob/v0.70.0/examples/memory_imagination.py)
for a bounded demonstration with actual toy-body outcomes.
[Record patches](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/record-patch.md)
provide event records and consolidation. The advanced
[population solver](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/equilibrium/index.md)
provides exact state-and-error readback under its own numerical contract.

[Documentation](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/index.md) ·
[API](https://github.com/muellerberndt/cadence/blob/v0.70.0/docs/api.md) ·
[Contributing](https://github.com/muellerberndt/cadence/blob/v0.70.0/CONTRIBUTING.md) ·
[Changelog](https://github.com/muellerberndt/cadence/blob/v0.70.0/CHANGELOG.md) ·
[Research tasks](https://github.com/muellerberndt/cadence/issues)

Licensed under [GPL-3.0](https://github.com/muellerberndt/cadence/blob/v0.70.0/LICENSE).
