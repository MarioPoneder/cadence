# Cadence documentation

**Cadence is not a feed-forward deep neural network.** A Cadence brain is one continuing
equilibrium brain: it bootstraps an interpretation of its world into its settled state,
carries that state through experience, acts from it, and repairs it locally when experience
disagrees. The equilibrium is the world model. While the world model holds, the brain runs
cheaply; a witnessed failure is what calls for repair, and the repair moves the brain to a
neighbouring equilibrium. Its intended structure is that of a biological brain, regions that
constrain one another while settling, not a stack of layers evaluated once.

Read this page first. It states the principle, says what the library implements today and
what is still design direction, gives the test that tells a Cadence brain from a layered
network, and lists the guides in reading order. [index.md](index.md) is the catalogue.

## The principle

- **The answer is a settled state.** Regions settle together under the present drive, the
  retained trace and memory; the motor choice reads that state. Later regions shape earlier
  ones while the answer forms. A brain that does not settle refuses to act.
- **The equilibrium is the world model.** Learned parameters and memories encode a family of
  equilibria under different evidence and context. Learning changes the family; ordinary use
  moves within it. A new observation changes the boundary of the solve, not the model.
- **Low energy while the model holds.** A world the brain has learned is settled in few
  sweeps with small updates; the work rises when something is new or wrong.
- **Repair on failure.** A lesson is the local repair of a witnessed mismatch, the brain's
  prediction or action against the observed outcome. The repair uses local contrast, no
  backward pass, and leaves the rest of the model in place.
- **One life.** The same brain bootstraps, acts, fails, repairs and continues. Saving and
  loading preserve pending feedback and memory.

## The test that tells a Cadence brain from a layered network

A brain is set up like a layered network when its answer is a function of the present
observation alone: a whole window of input pressed into one observation, one processing
region, a readout, trained by teacher labels on every presentation. Such a brain learns
exactly what a one-hidden-layer network learns, and the measurements say so. The test:

1. **Stream, not window.** The brain receives one frame or one event per step, and a copy
   given only the present frame cannot answer. The answer lives in the state the past left.
2. **State carried through `step`.** The brain runs through `step` with its activity, trace
   and memory retained; `predict` and `fit` are controls that discard both.
3. **Regions that return.** Observers or recurrent wiring return influence to earlier regions;
   memory enters the settle through the trace and associative recall.
4. **Lessons on failure.** A correction teaches; a confirmation does not. Count the lessons.
5. **Energy watched.** The sweeps a settle needs and the residual it reaches are measured at
   rest and at a surprise; a rise is the signal.
6. **Judged on a life.** Free behaviour over time, retention after interference and recovery
   after a disturbance, with a backpropagation network as the baseline, never as the goal.

## What the library implements today

| Part | Implemented | Design direction, not yet a default |
| --- | --- | --- |
| Composition | `Brain.compose(inputs, actions, modules, observers, lateral)`: a chain of processing regions, reciprocal between neighbours and with the motor cortex, the sensory projection one way, a working trace into the association region, optional observer regions returning influence to every processing and motor region | Regions with internal connections, readback ports and records inside the composed brain |
| Answers | `act`, `predict`, `accuracy` qualify the full equations to tolerance or refuse; `imagine` settles supplied observations privately | A residual or sweep count returned with every answer (today `imagine`, `equilibrate` and `last_learning` carry them) |
| Memory | Working trace (one to two steps), fast and persistent associative memory, separate record and temporal patches | Context across a sentence or an episode inside the composed brain ([issue 121](https://github.com/muellerberndt/cadence/issues/121)) |
| Learning | Local free and nudged contrast; finite by default, qualified with damping on opt-in (`LearnerConfig(qualified=True)`); reward plasticity with eligibility; memory writes on outcomes | Updates gated on witnessed failure and cheap stable operation ([issue 122](https://github.com/muellerberndt/cadence/issues/122)); the step size already shrinks with the error, the work does not |
| Consequences | `TemporalPatchNet` learns transitions and plans privately | Integrated into the composed brain ([issue 93](https://github.com/muellerberndt/cadence/issues/93)) |
| Language | Record patches over symbol streams | Hearing a sentence and writing it through one continuing brain ([issue 103](https://github.com/muellerberndt/cadence/issues/103)) |

A numerical fixed point checks the declared equations. It does not establish that the
interpretation is right, that the memory is useful or that the computation was cheap; each of
those needs its own measurement.

## Read in this order

**The principle**

1. [One continuing equilibrium brain](world-model.md): the lifecycle, what is implemented and the boundaries.
2. [Equilibrium world models](equilibrium-world-models.md): three clocks, state estimation against model learning, cheap habitual use.
3. [Architecture](architecture.md): the map of System 1, optional System 2 and the separate engines.

**Build and run one brain**

4. [Quickstart](quickstart.md): one brain through observations, outcomes, imagination and a saved continuation.
5. [The continuing-brain example](../examples/continuing_brain.py): bootstrap, unchanged and changed conditions, a pending checkpoint.
6. [Continuous interaction](continuous.md): event order, refusal and retry, memory writes, reset, defaults.
7. [Compose a brain](brain.md): modules, observers, `lateral`, qualified teaching, genome wiring.
8. [Experience design](experience.md): causal order, ports, curriculum.
9. [Memory](memory.md): the trace, synaptic memory, records and their limits.
10. [Reward](reward.md): eligibility, the TD error, centring and floor.

**The mechanisms, as controls**

11. [Concepts](concepts.md): neuron equations, residual against movement, state lifetimes.
12. [The learning rule](learning.md): free and nudged phases, the gradient conditions, calibration, every knob.
13. [Task recipes](tasks.md): interfaces, with prediction heads as controls.

**Evaluate**

14. [Task design](task-design.md): from a task to ports, closing the action-consequence loop.
15. [Missteps](missteps.md): how a good-looking number is wrong.
16. [Scaling](scaling.md): experience, exposure and capacity apart; counting work.
17. [Troubleshooting](troubleshooting.md): symptom to page.

**Contracts and reference**

18. [Contracts](contracts.md): equations, qualification and refusal per interface.
19. [Certificates](certificate.md): the contraction bound and its scope.
20. [Protocols](protocols.md) and [receipts](receipts.md): held-out tests and numbers bound to sources.
21. [API](api.md): signatures and defaults.
22. [Backends](backends.md): devices and precision.

**Other engines**

23. [Interaction](interaction.md), [temporal model](temporal.md), [planning](planning.md), [temporal memory](temporal-memory.md): learned consequences and private planning.
24. [Record patch](record-patch.md): event records and sleep.
25. [Belief](belief.md), [steering](steering.md), [the rung guide](howto-rung.md): belief patches and the `Life` governor.
26. [PatchNet](patchnet.md), [recursive settlement](recursive-settlement.md), [recursive training](recursive-training.md): observers in one equilibrium.
27. [Cortices](cortex.md), [evolution](evolution.md), [connectomes](connectomes.md): custom regions, genomes, measured wiring.
28. [Build](build.md): data-to-model recipes; its classifier is a control, not a brain.
29. [Orientation](orientation.md): the glossary for readers who come from machine learning.
30. [The population solver](equilibrium/index.md): exact state-and-error readback under its own equations.

## Evaluate and contribute

[Contributing](../CONTRIBUTING.md) lists the checks; [the changelog](../CHANGELOG.md) the
releases; [the roadmap issue](https://github.com/muellerberndt/cadence/issues/109) the
numbered build sequence. Report a capability that is missing or untested as its own issue.
