# Compose a brain

Start with `Brain.compose`. It builds **System 1**: one continuing neural
graph with sensory input, reciprocal processing regions, motor choices, a working
trace and fast/persistent associative memory. Optional observer regions add
**System 2** feedback within that graph.

## Brain

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, modules=(16, 8), seed=7)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)
assert action.shape == (1,)
```

Inputs have shape `(streams, inputs)`; each output is an action index.
`modules=(16, 8)` creates two reciprocally connected processing regions. The
last is the association region. Sensory input reaches the first, the association
region exchanges signals with motor neurons, and its working trace enters through
context neurons. `SynapticMemory` records the actually chosen action's reward;
its fast and persistent associations influence later choices.

The base can already be deep. Add recursive readback separately:

```python
recursive = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), observers=(8, 4), seed=7,
)
assert recursive.step(observation).shape == (1,)
```

Each observer exchanges activity with the base, motor regions and earlier
observers. All regions participate in the same settlement. This provides the
connections for recursive correction; it does not automatically learn useful
reflection. These observers read neural state. The advanced
[population solver](equilibrium/index.md) separately implements exact state-and-error
readback under its own equations.

## One experience step by hand

`step` combines learning from the previous outcome and choosing the next action.
Use `act` and `learn` separately when the body needs to manage that timing:

```python
body_brain = Brain.compose(4, 2, modules=(16, 8), seed=7)
chosen = body_brain.act(observation)

# Execute the choice in a tiny environment: action 0 earns one unit.
reward = (chosen == 0).astype(float)
following = np.array([[0.0, 1.0, 0.0, 0.0]])
body_brain.learn(reward, np.array([False]), following)
next_action = body_brain.act(following)
```

Reward and termination describe the preceding executed action. A demonstration
labels the current observation instead. Keep batch-row identities fixed until
`reset()`. [Continuous interaction](continuous.md) covers episodes, teaching,
private imagination and retries. Independent [Records](memory.md#records) can
store declared observation/action/outcome fields; their reads and writes have
an explicit record rule rather than a neural-settlement certificate.

## Settle and check

`Brain.act`, `predict` and `accuracy` check the full potential/adaptation
equations before returning answers, including observer state. The defaults allow
1024 free steps at residual tolerance `3e-3`. A difficult free solve may use
half-step numerical damping within that same total budget; its final residual
is checked against the original model. Finite teaching uses its own nudged-phase
contract.

An exhausted `act` raises `RuntimeError` without changing live activity, memory,
randomness or pending feedback. If `step` learned an outcome before the next action
refused, keep the learning and retry `act`; do not send the same reward again.
A qualified state satisfies the equations to tolerance. Accuracy, uniqueness
and stability require their own evidence.

The lower-level `NeuralGraph.equilibrate` returns an `Equilibrium` with per-row
`residual` and `converged`, the state and total steps. `NeuralGraph.residual` checks an
existing state without settling. See [contracts](contracts.md) and
[certificates](certificate.md).

## Imagination and checkpoints

<a id="checkpoints"></a>

```python
phases = brain.imagine([observation, following], budget=1024, tolerance=1e-6)
assert phases
brain.save("brain.npz")
resumed = Brain.load("brain.npz")
```

Inspect each phase's `converged` flags before using it. Imagination stops at the
first refused phase and includes that phase in its result. It privately advances
a copied trace; it does not change live memory, parameters, randomness or pending
outcomes. It evaluates observations you supply. [Temporal planning](planning.md)
uses a learned environmental model to consider action consequences.

`Brain.save/load` includes neural parameters, critic, optimizers, random
state, traces, both associative-memory timescales and a pending action's feedback
state. Save the body separately and resume the same stream identities. Memory
shapes and numerical values are validated before use.

For advanced compositions, `Learner.save/load` saves the learner rather than an
entire body loop. Independently owned `Records` need their configuration, `tables`,
`mean`, `pathway_norm`, `seen` and `writes`; independently owned traces and critics
also belong to the caller's saved state. [API details](api.md) define each contract.

## Genome, development, brain

Use `Genome` for named regions, custom projections or evolved wiring:

```python
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(16), motor_cortex(3)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
)
connectome = cd.develop(genome, seed=0)
network = cd.NeuralGraph(connectome, cd.learning_neuron_model())
assert network.connectome.n == connectome.n
```

[Write a cortex](cortex.md) describes ports and projections;
[evolution](evolution.md) describes genomes and selection.
`Brain.build` offers the image/vector builder, and
`Brain(connectome, ...)` accepts the named populations required by its
interaction loop. These are advanced construction options for specific wiring
needs. A custom `NeuralGraph` alone does not install the complete Brain loop.

## The brain in a browser page

The [viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
visualizes a connectome and recorded settlement. `record_settlements` supplies
actual iterations; the library does not provide a browser environment or body.
