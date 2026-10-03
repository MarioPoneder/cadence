# Compose a brain

Start with `Brain.compose`. It builds **System 1**: one continuing neural
graph with sensory input, reciprocal processing regions, motor choices, a working
trace and fast/persistent associative memory. Optional observer regions add
**System 2** feedback within that graph.

Keep this brain through acquisition, ordinary use and correction. The
[world-model guide](world-model.md) explains why its learned relations and memory
support changing equilibria, and distinguishes that design from the mechanisms
implemented today. The calibration and isolated response examples below are
controls for those mechanisms.

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

## Operating point and motor competition

`Brain.compose(..., lateral=-0.5)` configures motor competition. This is
the signed weight between each distinct pair of motor neurons; zero removes
those lateral connections while keeping reciprocal processing/motor feedback.
The default stays at -0.5. A larger action vocabulary adds more inhibitory
inputs per motor neuron, so inspect free activity and its full residual on the
actual task before selecting a different value.

Global gain changes synaptic drive throughout the graph; population bias
changes selected neurons' operating points. Integration steps and numerical
damping affect how the equations are solved. A faster qualified solve does not
establish that its motor state responds usefully to teaching. The
[calibration guide](learning.md#calibrating-the-operating-point) shows how to
check candidate biases before installing them. Choose gain, bias targets and
lateral wiring on development inputs, retain their hand-set controls, and
freeze them before confirmation.

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
private imagination and retries. The
[continuing brain example](../examples/continuing_brain.py) shows the complete
reward, teacher and checkpoint loop using the published `0.71.0` API.
Independent [Records](memory.md#records) can
store declared observation/action/outcome fields; their reads and writes have
an explicit record rule rather than a neural-settlement certificate.

## Settle and check

`Brain.act`, `predict` and `accuracy` check the full potential/adaptation
equations before returning answers, including observer state. The defaults allow
1024 free steps at residual tolerance `3e-3`. A difficult free solve may use
half-step numerical damping within that same total budget; its final residual
is checked against the original model. Default teaching uses finite nudged phases.
Select qualified teaching explicitly:

```python
from cadence import LearnerConfig

qualified = Brain.compose(
    inputs=4, actions=2, modules=(16, 8), seed=7,
    learning=LearnerConfig(
        qualified=True, damping=3, free_steps=1024, nudged_steps=1024,
        tolerance=3e-3,
    ),
)
qualified.step(observation, teacher=np.array([0]))
assert qualified.last_learning["demonstration_qualified"] == 1.0
```

This requires every free and teaching phase to satisfy the original full
equations before a supervised update. `LearningPhaseError` retains the attempted
phases and their cost report; a refusal preserves parameters and optimizer
history. `last_learning` records accepted and refused teaching diagnostics under
`demonstration_` keys, including attempted presentations and row work.
These qualified solves use configurable damping: a stalled attempt
can move to a smaller numerical step sooner, within the same total budget and
without relaxing the original equation check.
Qualified teaching does not change the reward eligibility contract: the
Brain default allows up to 12 finite nudged steps, configured separately through
`ActorCriticConfig.eligibility_steps`.

An exhausted `act` raises `RuntimeError` without changing live activity, memory,
randomness or pending feedback. If `step` learned an outcome before the next action
refused, keep the learning and retry `act`; do not send the same reward again.
A qualified state satisfies the equations to tolerance. Accuracy, uniqueness
and stability require their own evidence.

`predict` and `accuracy` start independent cold graph solves without reading the
working trace or associative memory. Use greedy `act` on a separately loaded
brain to measure the complete memory-aware response. Reset live state and the
trace for each independent query; clear fast associative residuals too when the
question is recall from consolidated associations alone. Keep those resets
separate from erasing durable parameters or records. [Memory](memory.md) describes
the three stores and their limits.

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

The archived [viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
visualizes a connectome and recorded settlement in its declared environment.
Current `record_settlements` captures actual graph iterations for an
application-owned display. [cadence-demos](https://github.com/muellerberndt/cadence-demos)
contains current application examples; the library does not provide a browser
environment or body.
