# Cadence documentation

Cadence researches general intelligence through overlap consensus, equilibrium
detuning and functional self-reflection. One continuing system should acquire
skills, retain useful experience, imagine alternatives and repair its behavior
across applications. The guides distinguish implemented operations from that
broader research goal. This page is the map, in the order a builder needs it.

## Start here

1. [Cadence for machine-learning people](orientation.md): the four brains against
   the models you know, where each learning signal comes from, a comparison with
   backprop, the shapes of every array, and a glossary.
2. [Quickstarts: three kinds of brains](quickstart.md): a record patch that learns a
   stream and sleeps, a settling brain that decides, a temporal patch that plans; each
   also runs [in your browser](demos.md) with the whole brain animated,
   `cadence-demo stream|decide|body`.
3. [Build your own brain](build.md): your own data to a trained, evaluated and saved
   brain of each kind, with the sizes to start from and the checks to run.
4. [Troubleshooting](troubleshooting.md): the first questions, each answered in a
   paragraph with a pointer.

## Two primitives, and what composes them

- The **settling patch**: bounded local state, ports, an equilibrium under
  constraints, learned by the contrast of a free and a nudged settle. Built from a
  genome of regions (`Genome`, `develop`, `Brain`, `Learner`) or as one temporal
  patch (`TemporalPatchNet`).
- The **record patch** (`RecordPatchNet`): a gated linear context with a record
  store inside the patch. By day an observation is written once; by night the
  slow weights learn from the store's own dreams (`sleep`).

Everything else composes these two through ports. The [belief patch](belief.md)
is the composition toward a learned world model: a transition under action, a
repair of the belief by iteration with the store read inside it, and private
imagination; its input port reads grids through [maps](record-patch.md#maps-a-structured-input-port).

### The record patch

[The record patch](record-patch.md): one observed path, the reading and its scale,
what records hold, diagnosing a store, categorical ports, batched writes, a store
narrower than its port, acquisition in two phases (records by day, weights by night),
maps at the port, [two patches in depth](record-patch.md#two-patches-in-depth) and
[several patches joined by ports](record-patch.md#several-patches-joined-by-ports).

### The belief patch

[The belief patch](belief.md): a transition under action, evidence repair by
iteration, the store inside the repair, imagination that consumes no observation,
and the imagination loss that trains the transition.

### A brain that reads itself

[The steered cortex, the governor and the life](steering.md): a steering patch of the same
rule sets the cortex's gains from the cortex's readback, a governor settles on the brain's own
signals and names its mode, and the life runs both, with the three-way switch and one
accounting; [how to build a rung demo](howto-rung.md) is the order of work from the world to
the receipt.

### The settling brain, from regions

[Compose a brain](brain.md) (genome, development, settling checks, one experience
step with records beside a policy, `GenericBrain`, checkpoints, the brain in a page),
[write a cortex](cortex.md) (regions, the catalogue, projections, ports, which
synapses a head owns), [local learning](learning.md) (the free/nudged rule, why the
contrast is a gradient, every knob), [records and memory](memory.md) (the records
cortex, fast synapses, pattern separation), [reward](reward.md) (eligibility traces
and dopamine), [evolve a brain](evolution.md) (mutation, selection, any genome as
genes, detuning inside a life) and [concepts](concepts.md) (the neuron model,
settling and equilibrium, state lifetimes, evidence and controls).

### The temporal patch

[Temporal learning](temporal.md) (observed paths, context, the energy and the
update, the centered-contrast check, backtracking), [private planning](planning.md)
(repair of continuous controls under the learned model), [learn, act and observe](interaction.md)
(the loop end to end on a small body), [response protection](temporal-memory.md)
(conditional retention of chosen responses) and [experimental fixed connectivity](partitioned.md)
(routing constraints for comparisons, not learned specialization).

- [Learning in lockstep: populations on a device](population.md): the torch twin of the record patch for a population of brains in many worlds at once; what batches (instances, streams, patches) and what does not (moments); the parity tests; the measured throughput; the kernel at the width of a game (moments folded by stream, masks, output weights, familiarity, Adam on the adjoint) and the recommended structure for training across several devices.

## Measure

[Task design](task-design.md), [common missteps](missteps.md) and
[scaling](scaling.md) say how to validate observations, actions, learning and
retained behavior before scaling. [Convergence certificates](certificate.md),
[protocols](protocols.md), [receipts](receipts.md), [the brain viewer](pages.md) and
[backends, devices, precision](backends.md) are the instruments.
[Brains from a connectome](connectomes.md) is the recipe for a measured wiring as one brain:
custody, the gain by protocol, the sub-net a page settles, what a rate model cannot carry,
and the traps of learning on it.

## Reference

The [API reference](api.md) lists every public name by module. The
[changelog](../CHANGELOG.md) lists every release; [contributing](../CONTRIBUTING.md)
says how to run the checks and what a change needs; the
[Lean library](../lean/README.md) holds the checked conditional theorems.

## The ideas

[Architecture and integration](architecture.md): the contract of the temporal patch and
the fixed-model actor. [Equilibrium and learned world models](equilibrium-world-models.md):
three clocks, changing expected trajectories, evidence and branch isolation, and the
distinction between current APIs and a proposed shared architecture.
[Creativity and self-reflection](creativity.md): novel proposal evaluation, recursive
readback and transfer as research requirements. The
[paper](https://philpapers.org/rec/MUECAP-2) states the
hypotheses, the theorems and the evidence.

## Kept for existing experiments

These compositions have their own state and learning contracts and are kept for the
experiments that used them: [PatchNet](patchnet.md), the `GenericBrain` loop in
[continuous interaction](continuous.md) and [experience](experience.md), and
[task recipes](tasks.md). `GenericBrain` is kept for its one remaining application;
new lives are written on [the life](steering.md#a-life-with-a-governor).

Worked applications with their receipts and checks live in the
[examples repository](https://github.com/muellerberndt/cadence-examples): the worm,
Connect Four, the Amen composer and Patch World. Examples illustrate a general
mechanism. Their scores are evidence for the named task, not proof that the
architecture solves arbitrary problems.
