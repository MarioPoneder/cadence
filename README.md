<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: a continuing brain with local state, memory and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Demos](https://floatingpragma.io/demos/) · [Documentation](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/index.md) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/) · [Changelog](https://github.com/muellerberndt/cadence/blob/v0.62.0/CHANGELOG.md)

[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

**A simulated human-like brain, built from simplified biological mechanisms.**

Cadence aims to grow a continuing brain that learns through experience, remembers,
imagines alternatives and acts in the world. Its starting point is an animal-like
sensorimotor system with plastic connections and memory. Cortical columns and
recursive observation extend that foundation. Human-level intelligence is the
goal; the software models selected mechanisms rather than detailed biology.

The central idea is local disagreement repair. Bounded, observer-like patches
carry state, communicate through ports, read back activity and retain records.
Observations disturb their relationships; repair seeks a coherent state, and
actual consequences guide learning. A numerically settled answer can still be
wrong about the world. Learning has to improve subsequent free behavior.

<a id="get-started"></a>

## A continuing brain

The base can contain specialized regions and deep, modular connections.
`GenericBrain.compose` brings together sensory and association regions,
reciprocal motor connections, reward learning, a working trace and consolidating
associative memory. `modules` chooses the base regions; `observers` optionally
adds regions that read and return influence to the same live graph. Its `step` method receives the next observation and the actual outcome
of the preceding action. There is no application-wide training/inference switch.

**Cadence 0.62.0 is alpha software.** Its memory, learning and imagination
mechanisms are implemented and tested under their stated contracts; human-like
general intelligence remains the research goal. Python 3.11+ and NumPy are
required. PyTorch, MLX, Numba and SciPy support optional execution paths.

```sh
python -m pip install cadence-net==0.62.0
```

```python
import numpy as np
from cadence import GenericBrain

brain = GenericBrain.compose(
    inputs=4, actions=2, modules=(16, 8), seed=7,
)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)
assert action.shape == (1,)
```

The body executes that action. On the next call, pass its measured reward and
termination flag with the next observation; a teacher can instead label the
current observation. Keep each batch row attached to the same continuing life.
[Continuous interaction](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/continuous.md) explains the timing and complete
save/restore. [Build a brain](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/brain.md) shows custom regions and wiring.

## Memory and imagination are working mechanisms

Short-term traces preserve earlier activity across events. `Trace` and
`Afterglow` feed that retained information into later processing; they are more
than a warm numerical starting point. `GenericBrain` can use a working trace and
`SynapticMemory`, whose fast associations and persistent matrix support recall,
consolidation and revision. Correlated memories can interfere, and finite capacity
limits what can be retained. See [memory](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/memory.md) and
[continued learning](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/continuous.md).

`RecordPatchNet` combines persistent event context, slow learned parameters and
a writable record store. It can retain individual outcomes, privately imagine
continuations and transfer record completions into slow weights with `sleep`.
This mechanism powers the original Amen composer and Connect Four evaluator.
The [record-patch guide](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/record-patch.md) gives executable examples and
explains the different jobs of records and learned relations.

`TemporalPatchNet` learns a model of observed paths. Its `imagine` operation
queries a private continuation; `plan` repairs proposed continuous actions under
that model while holding actual observations and learned parameters fixed.
It executes no action and creates no new evidence. `TemporalMemory` can protect
declared responses under its finite-capacity contract. Use
[the learned-model interaction loop](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/interaction.md),
[planning](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/planning.md) and [response protection](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/temporal-memory.md).
`GenericBrain.imagine` can also examine a sequence of hypothetical observations
on a private copy of its working trace, without changing learned memory, pending
outcomes or random state. That predicts the brain's responses to supplied
observations; predicting the environment's response to an action requires the
learned world model above.

Run [the memory and imagination example](https://github.com/muellerberndt/cadence/blob/v0.62.0/examples/memory_imagination.py) for a
bounded demonstration of cue retention, response protection, private planning
and actual toy-body outcomes. Goals and protected responses are explicit. These
APIs provide memory and imagination; their usefulness and limits still need to
be measured on the task they serve.

## Optional recursive cortical observation

A column can observe another region's current state and send feedback while
both participate in the same recurrent repair process. Nesting this readback adds recursive
observation to a brain that can already be deep and modular. It gives no column
an unconditional final answer.

```python
from cadence import GenericBrain

recursive = GenericBrain.compose(
    inputs=8, actions=2, modules=(32, 16), observers=(8,), seed=7,
)
```

This constructs one recurrent graph with two base modules and an observing
region. The same `step`, memory and continuation interface applies. Before
returning an action, the full state equations must meet the declared tolerance;
exhausting the repair budget raises an error without issuing an action.
The lower-level `PatchNet.recursive` constructor remains available. [Recursive settlement](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/recursive-settlement.md) explains the
connections and causal checks; [recursive training](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/recursive-training.md)
covers experience and saved continuation. Useful recursive correction must earn
its cost while preserving the base brain's skills and memory.

The newer state-and-error population solver remains available separately as
`cadence.experimental.equilibrium`; its [guide](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/equilibrium/index.md)
documents its own API and numerical guarantees. It does not replace the restored
memory, record and temporal mechanisms. The implementations have different
learning contracts: graph/temporal models use equilibrium contrasts, while
record and belief models also use explicit adjoints and record writes. The
[contracts guide](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/contracts.md) states those differences.

## Recover and preserve actual behavior

The original applications provide concrete baselines:

- [Amen](https://github.com/muellerberndt/cadence-examples/tree/main/amen): a
  0.11-trained record patch generates music from silence while hearing what it
  plays. Ordinary playback keeps record writes off; its optional self-primer
  writes played events into records. It is not an Afterglow demo.
- [Connect Four](https://github.com/muellerberndt/cadence-examples/tree/main/connect4):
  a 0.12-trained record evaluator supplies values to a declared game-tree search.
  Browser play uses fixed learned parameters and an empty record store.
- [Atari Arcade](https://github.com/muellerberndt/cadence-demos/tree/main/atari-arcade):
  a separate 0.50 population-engine browser port learns from demonstrations and
  outcomes. Its runtime and parity checks have their own source identity.

Each application keeps its original checkpoints, receipts and supplied body
logic. Restored library operations must pass their relevant reproduction and
continuation checks before an application is described as ported. A newer
package version does not establish musical quality or playing strength.

Start with [the documentation](https://github.com/muellerberndt/cadence/blob/v0.62.0/docs/index.md). Development follows three active
research owners: [short-term memory](https://github.com/muellerberndt/cadence/issues/84),
[long-term memory and plasticity](https://github.com/muellerberndt/cadence/issues/85)
and [optional recursive integration](https://github.com/muellerberndt/cadence/issues/86).
These track stronger guarantees and integrated behavior while existing
capabilities remain available. See [contributing](https://github.com/muellerberndt/cadence/blob/v0.62.0/CONTRIBUTING.md) for checks.

Licensed under [GPL-3.0](https://github.com/muellerberndt/cadence/blob/v0.62.0/LICENSE). Historical attribution and license notices
remain with the preserved source and evidence.
