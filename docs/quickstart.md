# Start a continuing brain

Use `Brain.compose` to build **System 1**: connected processing regions,
motor choices, a working trace and fast/persistent associative memory. Regions
carry local state, exchange signals and repair disagreement in one neural
settlement. Optional **System 2** adds observing regions with returning feedback
in that same graph.

Python 3.11+ and NumPy are required:

```bash
python -m pip install cadence-net==0.71.0
```

## Observe, act and learn

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, modules=(16, 8), seed=7)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)

# Execute the choice in a tiny environment: action 0 earns one unit.
reward = (action == 0).astype(float)
next_observation = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.step(next_observation, reward=reward, done=np.array([False]))
assert action.shape == (1,)
assert brain.learner.updates > 0
```

Each input row is one continuing stream; its output is an action index.
`modules=(16, 8)` gives two reciprocally connected processing regions. A deeper
base is still System 1. `step` learns from the **previous action's actual outcome**
before choosing the next action. `teacher=` instead labels the current
observation. Keep row identities fixed until `reset()`.

This short example exercises a feedback update, not a learned policy benchmark.
Memory and learned associations can affect later choices; capacity is finite
and memories can interfere. [Continuous interaction](continuous.md) covers
teaching, episodes and memory timing.

## Imagine privately and resume

```python
phases = brain.imagine([observation, next_observation])
assert len(phases) == 2 and all(np.all(phase.qualified) for phase in phases)

brain.save("brain.npz")
resumed = Brain.load("brain.npz")

# Resume the same pending action with the same measured outcome.
reward = (action == 0).astype(float)
continued = brain.step(observation, reward=reward, done=np.array([False]))
replayed = resumed.step(observation, reward=reward, done=np.array([False]))
assert np.array_equal(continued, replayed)
```

`imagine` carries a private trace through supplied observations. It leaves live
memory, parameters, random state and pending feedback unchanged. Inspect every
phase's `qualified` flags: the result stops at the first refused phase. This
evaluates the brain's responses to supplied observations; a learned model of
environmental consequences belongs to [temporal planning](planning.md).

Save/load preserves the current brain's full continuation, including pending
feedback. Save the environment separately and resume its stream identities too.

## Add optional observers

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8,), seed=7,
)
assert recursive.step(observation).shape == (1,)
```

Observers read and return influence to processing regions, motor regions and
earlier observers. They participate in the same settlement and interaction API.
This provides recursive feedback; a useful task advantage must be learned and
measured.

Actions and independent predictions require the full neural equation residual
to meet the configured tolerance. A refused `act` leaves live state, memory,
randomness and pending feedback unchanged. If `step` already learned an outcome
before the next action refused, retry `act` without submitting that outcome
again. See [numerical contracts](contracts.md).

## Specialist guides

- [Compose a brain](brain.md): custom regions, wiring and checkpoint semantics.
- [Memory](memory.md): working traces, fast associations and consolidation.
- [Temporal models](temporal.md) and [response protection](temporal-memory.md):
  learned consequences, private planning and selected durable responses.
- [Record patches](record-patch.md): one-write event records, categorical ports
  and learning from stored completions.
- [Reciprocal patches](patchnet.md): explicit patch ports and local contrasts.
- [The population solver](equilibrium/index.md): advanced exact state-and-error
  readback under its own numerical contract.
