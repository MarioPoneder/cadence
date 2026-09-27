<p align="center">
  <img src="https://raw.githubusercontent.com/muellerberndt/cadence/main/docs/assets/cadence-logo.png" alt="Cadence: connected patches with local state, readback and repair" width="100%">
</p>

# Cadence

[Website](https://floatingpragma.io/cadence/) · [Examples](https://github.com/muellerberndt/cadence-examples) · [Paper](https://philpapers.org/rec/MUECAP-2) · [PyPI](https://pypi.org/project/cadence-net/) · [Documentation](https://github.com/muellerberndt/cadence/blob/main/docs/index.md) · [Changelog](https://github.com/muellerberndt/cadence/blob/main/CHANGELOG.md)

[![PyPI](https://img.shields.io/pypi/v/cadence-net)](https://pypi.org/project/cadence-net/)
[![CI](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml/badge.svg)](https://github.com/muellerberndt/cadence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/cadence-net)](https://pypi.org/project/cadence-net/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/muellerberndt/cadence/blob/main/LICENSE)

**Research toward general intelligence through overlap consensus, equilibrium detuning and self-reflection.**

Cadence's goal is a continuing learning system with the flexibility of animal
and human problem solving: acquiring skills from experience, retaining useful
knowledge, imagining alternatives and creating solutions across domains.
The mission is to find the smallest persistent state and local update rule
that can support these abilities. The building block stays as simple as possible,
like in nature; every part of a brain answers with a settled state of that one
rule; and where a choice appears, evolution across lives is preferred to design. General intelligence is the research goal;
the current library establishes bounded learning, memory and control results.

The building block is a bounded, observer-like patch with local state,
ports, records, readback and repair. A disturbance exposes disagreement. The
network can explore a possible response, test it against actual consequences
and settle into a revised organization. We seek fewer mechanisms that solve
more problems.

## The roadmap

We want to reach a point where we can effortlessly evolve a human-like brain,
teach it first by imitation and then through its own life, and give it an
experience identical to that of a human living in our world. Brains with
capabilities far beyond ours are thinkable on the same path; human-level
competence comes first, as a sensible milestone. The steps from the
brains in this library to that milestone are tracked as issues in this repository,
rung by rung, each with its task, its control and its falsifier.

## In familiar terms

Cadence is a library for computation through local state, ports and repair,
with NumPy implementations and optional accelerated backends. Its graph and
temporal learners use free and nudged equilibrium contrasts. Record and belief
models use causal or finite-iteration computations and train slow weights by
an explicit adjoint backward scan. Record stores additionally learn by local
writes. These are distinct implementations with different guarantees; the
[numerical and learning contracts](docs/contracts.md) explain how to choose and
check them. A night of `sleep` can transfer a record store's completions into
slow weights under its declared training procedure.
[Cadence for machine-learning people](https://github.com/muellerberndt/cadence/blob/main/docs/orientation.md)
maps each brain to the model you know, says where each learning signal comes from,
and lists the shapes and the words.

## Get started

```bash
python -m pip install cadence-net     # Python 3.11+ and NumPy; torch, MLX and Numba are optional
```

Three brains, each trained in front of you in seconds from fixed seeds, with every
neuron and synapse animated, the distance from equilibrium as a heat on the neurons,
the last change on the synapses, the learning plotted as it is measured, and a line
of text for each phase:

```bash
git clone https://github.com/muellerberndt/cadence-examples.git
cd cadence-examples
python -m quickstart.demo stream     # or: decide, body
```

Nothing is hosted and there is no checkpoint; [the quickstarts in your browser](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart)
says what equilibrium means in each brain and what detuning buys. The same three
brains in Python are the [quickstarts](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md).
This is the record patch: it hears a stream, recalls every outcome after one pass by
day, and after a night with the stream closed says every outcome from its weights alone:

```python
import numpy as np
from cadence import RecordPatchNet

rng = np.random.default_rng(21)
net = RecordPatchNet(
    12, 12, 5, seed=4, cells=2048, active=16, record_rate=1.0, groups=(5,), slowest=8.0
)
symbols = rng.integers(12, size=(3, 8))
heard = np.eye(12)[symbols]
outcome = np.eye(5)[(symbols + np.roll(symbols, 1, axis=1)) % 5]  # the last two symbols decide

for _ in range(8):  # the day: one write per moment, slow weights at rate zero
    net.reset()
    net.observe(heard, outcome, rate=0.0)
net.reset()
awake = net.imagine(heard, state=np.zeros((3, 12)))
assert np.array_equal(awake.output.argmax(-1), outcome.argmax(-1))  # the store recalls

night = net.sleep([heard], passes=240, rate=8.0, backtrack=True)  # dreams, then dawn
alone = RecordPatchNet.restore(net.snapshot())
alone.records.tables["y"][:] = 0.0  # the same weights with an empty store
assert np.array_equal(alone.imagine(heard, state=np.zeros((3, 12))).output.argmax(-1), outcome.argmax(-1))
print(night)
```

[Build your own brain](https://github.com/muellerberndt/cadence/blob/main/docs/build.md)
takes your own data to a trained, evaluated and saved brain of each kind, and
[troubleshooting](https://github.com/muellerberndt/cadence/blob/main/docs/troubleshooting.md)
answers the first questions.

## Deep Recursive Settlement Networks

**Deep Recursive Settlement Networks (DRSNs)** place observing patches and the
patches they observe in one equilibrium. Each observer reads current activity
and sends feedback while all populations, including the answer neurons, keep
settling together. Adding a level adds participants to the same computation;
it gives no level an unconditional final word.

```python
from cadence import PatchNet

brain = PatchNet.recursive(inputs=8, layers=[32, 16, 8], outputs=2, seed=7)
# One base population, two observer populations, one joint solver and learning rule.
```

The builder lets you choose the number and size of these populations in one
graph. The
[recursive settlement guide](docs/recursive-settlement.md) explains the wiring,
whole-state residual checks, local learning and causal interventions.
The [training guide](docs/recursive-training.md) walks through
`recursive(1371, [24, 8], 6)`, observed action returns and saved continuation.
Version **0.19.0** adds optional CPU energy refinement when a compatible brain
exhausts its local settling budget, with full-equation and local-curvature
checks before learning. Local settlement remains the default.
**Performance evaluations are ongoing:** numerical and causal tests establish
the joint-settlement mechanism; a capability advantage from recursive depth
and competitive training efficiency against transformers remain hypotheses.

## Which brain

The public families share state and port concepts, with distinct solver and learning contracts:

| You want | Start with | What it supplies |
| --- | --- | --- |
| recursive observers and their observed populations to settle together | [one joint PatchNet](docs/recursive-settlement.md) | Reciprocal current-state readback and feedback, with answer neurons in the same checked equilibrium. `PatchNet.recursive` selects the base and observer widths without introducing another solver. |
| to learn from a stream of events, keep single facts after one exposure, and generalise overnight | the record patch, [RecordPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md) | A gated linear context with a record store inside the patch: an observation is written once by day, and by night the slow weights learn from the store's own dreams (`sleep`), with nothing outside the patch consulted. Categorical ports, batched writes, a store narrower than its port and a two-patch stack. One pass of writes, with no gradient, gives a small grammar for 0.8 of its never-taught combinations; one night lifts the slow weights alone to 1.0. |
| a decision or evaluation over a fixed set of inputs, an explicit wiring, a policy that learns from reward | the settling brain: a [brain of regions](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md) (`Genome`, `develop`, `Brain`, `Learner`) | Local repair of a settled state under any wiring, including a measured connectome; learning by the contrast of a free and a nudged settle; [cortices](https://github.com/muellerberndt/cadence/blob/main/docs/cortex.md), a records cortex, [reward](https://github.com/muellerberndt/cadence/blob/main/docs/reward.md) through eligibility traces, [evolution](https://github.com/muellerberndt/cadence/blob/main/docs/evolution.md) of the genome, and a [certificate](https://github.com/muellerberndt/cadence/blob/main/docs/certificate.md) on the settling. |
| continuous observations and actions, a learned dynamics model, private planning | the temporal patch, [TemporalPatchNet](https://github.com/muellerberndt/cadence/blob/main/docs/temporal.md) | Local repair of observed paths, persistent context, private imagination, [continuous planning](https://github.com/muellerberndt/cadence/blob/main/docs/planning.md) and [protected responses](https://github.com/muellerberndt/cadence/blob/main/docs/temporal-memory.md); learning by the contrast of a free and a nudged settle of the whole path. |
| a brain that reads itself: senses weighed by a steering patch, a governor that decides when to think | the composition, [Steered and Life](https://github.com/muellerberndt/cadence/blob/main/docs/steering.md) | Two belief patches and a seam with one admitted step; a settling brain of a dozen neurons as the governor; the three-way switch and the accounting every rung demo shares |
| a belief carried under action, repaired by evidence, that imagines with no observation | the belief patch, [BeliefPatch](https://github.com/muellerberndt/cadence/blob/main/docs/belief.md) | A belief carried by a learned transition under the executed action and repaired by a few iterations of one nonlinear map with the record store read inside it. The composition toward a learned world model, trained with the [imagination loss](https://github.com/muellerberndt/cadence/blob/main/docs/belief.md#training-the-transition-the-imagination-loss) so the transition carries the belief. |

Everything composes through ports: two record patches in depth (`RecordPatchStack`),
several coupled through finite port-repair rounds (`JointRecordPatches`), a grid read through a tied
kernel at the port (`StructuredPort`), and a genome that `evolve` mutates and selects
across lives. The [architecture guide](https://github.com/muellerberndt/cadence/blob/main/docs/architecture.md)
is the contract of the temporal patch.

## Examples

The worked applications live in the [examples repository](https://github.com/muellerberndt/cadence-examples),
each a static page that runs its brain in the browser with the library's arithmetic, with the
receipts behind every number it states and a check that recomputes them; the
[examples page](https://github.com/muellerberndt/cadence-examples#readme) lists them all. Two
of them:

<table>
<tr>
<td width="50%"><a href="https://floatingpragma.io/cadence-examples/fly-matrix/"><img src="https://raw.githubusercontent.com/muellerberndt/cadence-examples/main/fly-matrix/screenshot.png" alt="A fly in the Matrix: a connectome model with sensory input and a supplied body controller"></a><br><b>A fly in the Matrix</b> · <a href="https://floatingpragma.io/cadence-examples/fly-matrix/">live</a> · <a href="https://github.com/muellerberndt/cadence-examples/tree/main/fly-matrix">code</a><br>Import 150,802 retained neurons, supply sensory inputs and read settled neural outputs. Local mushroom-body learning changes responses; a supplied body controller executes goals. The example documents its model, assistance and source-bound checks.</td>
<td width="50%"><a href="https://floatingpragma.io/cadence-examples/amen-beats/"><img src="https://raw.githubusercontent.com/muellerberndt/cadence-examples/main/amen/screenshot.png" alt="Amen: one record patch computes a jungle track from silence"></a><br><b>Amen, the jungle composer</b> · <a href="https://floatingpragma.io/cadence-examples/amen-beats/">live</a> · <a href="https://github.com/muellerberndt/cadence-examples/tree/main/amen">code</a><br>One record patch starts from silence, hears each half-beat it plays and computes sixteen bars of drums, bass and texture.</td>
</tr>
</table>

They are application tests, not definitions of the architecture. A result in one
does not establish transfer to the others. Every example states what is supplied,
what is learned, what was measured and what it does not show, and pins the release
its checks were run against.

Build on an example with explicit sensory and action interfaces, reproducible
checks and measured resource costs. Report what is supplied, what is learned
and which behaviors the evidence supports. The examples repository's
[contributing section](https://github.com/muellerberndt/cadence-examples#contributing)
says what an example needs to live there, including the card every example README carries.

## Documentation

- **Start:** [for machine-learning people](https://github.com/muellerberndt/cadence/blob/main/docs/orientation.md) · [quickstarts](https://github.com/muellerberndt/cadence/blob/main/docs/quickstart.md) · [in your browser](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart) · [build your own brain](https://github.com/muellerberndt/cadence/blob/main/docs/build.md) · [troubleshooting](https://github.com/muellerberndt/cadence/blob/main/docs/troubleshooting.md)
- **Build:** [record patch](https://github.com/muellerberndt/cadence/blob/main/docs/record-patch.md) · [belief patch](https://github.com/muellerberndt/cadence/blob/main/docs/belief.md) · [a brain that reads itself](https://github.com/muellerberndt/cadence/blob/main/docs/steering.md) · [how to build a rung demo](https://github.com/muellerberndt/cadence/blob/main/docs/howto-rung.md) · [temporal learning](https://github.com/muellerberndt/cadence/blob/main/docs/temporal.md), [planning](https://github.com/muellerberndt/cadence/blob/main/docs/planning.md), [learn, act and observe](https://github.com/muellerberndt/cadence/blob/main/docs/interaction.md), [response protection](https://github.com/muellerberndt/cadence/blob/main/docs/temporal-memory.md) · [compose a brain](https://github.com/muellerberndt/cadence/blob/main/docs/brain.md), [write a cortex](https://github.com/muellerberndt/cadence/blob/main/docs/cortex.md), [local learning](https://github.com/muellerberndt/cadence/blob/main/docs/learning.md), [records and memory](https://github.com/muellerberndt/cadence/blob/main/docs/memory.md), [reward](https://github.com/muellerberndt/cadence/blob/main/docs/reward.md), [evolve a brain](https://github.com/muellerberndt/cadence/blob/main/docs/evolution.md)
- **Measure:** [task design](https://github.com/muellerberndt/cadence/blob/main/docs/task-design.md) · [common missteps](https://github.com/muellerberndt/cadence/blob/main/docs/missteps.md) · [scaling](https://github.com/muellerberndt/cadence/blob/main/docs/scaling.md) · [certificate](https://github.com/muellerberndt/cadence/blob/main/docs/certificate.md) · [protocols](https://github.com/muellerberndt/cadence/blob/main/docs/protocols.md) · [receipts](https://github.com/muellerberndt/cadence/blob/main/docs/receipts.md) · [the brain viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer) · [backends](https://github.com/muellerberndt/cadence/blob/main/docs/backends.md)
- **Reference:** [API](https://github.com/muellerberndt/cadence/blob/main/docs/api.md) · [changelog](https://github.com/muellerberndt/cadence/blob/main/CHANGELOG.md) · [Lean proofs](https://github.com/muellerberndt/cadence/blob/main/lean/README.md) · [contributing](https://github.com/muellerberndt/cadence/blob/main/CONTRIBUTING.md)
- **The ideas:** [architecture](https://github.com/muellerberndt/cadence/blob/main/docs/architecture.md) · [equilibrium world models](https://github.com/muellerberndt/cadence/blob/main/docs/equilibrium-world-models.md) · [creativity and self-reflection](https://github.com/muellerberndt/cadence/blob/main/docs/creativity.md) · [the paper](https://philpapers.org/rec/MUECAP-2)

Many brains at once on a graphics processor: `cadence.population.PopulationPatch` runs a population of record patches, each in many worlds, in lockstep; the [guide](https://github.com/muellerberndt/cadence/blob/main/docs/population.md) has the two-line use, what batches, the kernel at the width of a game, and how a population trains across several devices.

The [index](https://github.com/muellerberndt/cadence/blob/main/docs/index.md) is the full map.

## Four shared principles

- **Overlap consensus:** patches repair disagreement across their shared
  boundaries. The resulting equilibrium is an internally consistent model;
  its predictions have to agree with experience.
- **Equilibrium detuning:** observed outcomes perturb that equilibrium.
  Local positive/negative contrasts change learned relationships; the same
  operation can adjust proposed actions while holding the model fixed.
- **Self-readback:** a patch's proposed actions and predicted consequences are
  available to the same brain and tested by its next observation. Physical
  readback is evidence; an imagined outcome never is.
- **Metacognition as recursive self-observation:** a patch of the same kind reads
  the beliefs, residuals and surprises of the rest of the brain through ordinary
  ports, and its settled state steers them: which evidence counts, where a sense
  samples, what a habit holds, how long a repair runs. Because it is bounded it
  cannot steer everything at once, so it must select what matters for what it is
  computing; that selection is attention. Because it is a patch it can be read in
  turn, and the levels form a ladder from reflex to a robot that acts and speaks
  among people. The genome decides which readbacks exist, and a rung is earned
  only by a task the brain below it fails at matched information and compute.
  The rungs, their experiments and their issues are tracked in this repository's
  issues. The ladder orders what a brain earns; its nursery orders what the world
  supplies in one life, from the self by contingency to words, and its breeder what
  pays for a brain across lives.

The graph implementation supports joint recursive settlement; other families
provide detached self-readback, sequential steering or private proposal revision
under their own contracts. A learned advantage from recursive depth, curiosity and reliable
creativity are hypotheses with their tests on the ladder. [Creativity and self-reflection](https://github.com/muellerberndt/cadence/blob/main/docs/creativity.md)
defines these goals and their behavioral tests.

## Equilibrium world models

Cadence is being developed toward **evolving equilibrium world models**: brains
whose internal representation of an actor in its world grows more accurate and
expressive through experience. The representation should carry what is happening,
what persists out of sight, what the actor controls and what its actions could
cause. It need not describe those relationships in words to use them.

An equilibrium here need not mean motionless activity. A skilled actor can
follow a coherent, changing trajectory of perceptions, expectations and actions.
When events unfold as expected, the carried state should be close to a
useful interpretation of the next moment. Familiar danger can prompt a learned
response immediately. Extra inference is needed when competing interpretations
or consequential choices warrant it.

The proposed mechanism is recursive composition through bounded, observer-like,
self-reading patches. Scene, body, candidate action and retrieved experience
meet through learned nonlinear ports. One interpretation can become input to
another, allowing the system to revise its understanding before acting. Actual
evidence anchors that revision; changing the interpretation and learning new
relationships are distinct operations. A random outcome can require a new
response and be consistent with a correctly learned probability
distribution. Teaching detuning uses a specified target to compute a learning
signal; it is not synonymous with surprise.

Skilled demonstrations and the actor's own actions supply complementary
experience. Watching reveals useful behavior and situations; acting tests what
controls actually cause. The same learned relationships should support private
branches that explore possible futures without altering factual memory. Plans
must be judged by subsequent real outcomes. Grounded replay and sleep should
then consolidate reliable relationships and successful decisions into cheaper
habits, while unfamiliar situations can reopen deliberation.

This is the architectural vision. Current APIs provide components and bounded
demonstrations, not the complete evolving world model. Each successive release
is intended to take a concrete step toward this goal. Progress must be shown in
prediction, retention, useful internal planning and behavior at declared
resource budgets; a new version alone does not establish it. The
[equilibrium world-model guide](docs/equilibrium-world-models.md) explains the
mechanism, the distinction between evidence repair and learning, and the current
implementation boundaries.

## What the library establishes

**Imagined continuations are isolated.** Branches use the learned network
without changing live activity, parameters or factual bookkeeping. Controlled
experiments demonstrate useful planning. Creativity requires additional
evidence that novel proposals satisfy meaningful constraints and survive
actual evaluation; musical improvisation is one possible example.

**Retained experience and new learning are tested together.** Protected-path
memory is conditional and finite. Importance is supplied; automatic
relevance, selective forgetting, specialization and broad skill transfer are
research requirements. The [task-design guide](https://github.com/muellerberndt/cadence/blob/main/docs/task-design.md)
and [common missteps](https://github.com/muellerberndt/cadence/blob/main/docs/missteps.md) explain how to measure them.

**General mechanisms, different applications.** Games, language, multimodal
perception, embodied control and creative work should use the same learning and
memory mechanisms with declared observation and action ports. The scaling goal is
better learned behavior from more experience and training, with as little manual
design as possible. Measure unique experience, repeated training and model
capacity separately while keeping port meanings and task evaluation fixed. The
[scaling guide](https://github.com/muellerberndt/cadence/blob/main/docs/scaling.md)
defines these comparisons and the current computational limits.

Application demonstrations are published only when they establish their
claimed behavior. Recall and interpolation are useful development tests;
original creation requires stronger evidence. Every example states what is
supplied, what is learned, what was measured and what it does not show, and
carries a check that recomputes its numbers. Research receipts are
available with the paper without presenting those tests as finished products.

## Proofs and compatibility

The [bundled Lean library](https://github.com/muellerberndt/cadence/blob/main/lean/README.md)
contains 169 checked conditional theorems about the mathematical components
and their limits. It does not certify the complete Python implementation or
prove intelligence. The paper identifies assumptions and reproducible evidence.

The [PatchNet graph interface](https://github.com/muellerberndt/cadence/blob/main/docs/patchnet.md)
and `GenericBrain` are kept for the experiments that used them; they are distinct
compositions, not parts of the two primitives. [API reference](https://github.com/muellerberndt/cadence/blob/main/docs/api.md).

Development installs use `python -m pip install -e ".[dev]"`; [contributing](https://github.com/muellerberndt/cadence/blob/main/CONTRIBUTING.md)
lists the checks. [Optional backends](https://github.com/muellerberndt/cadence/blob/main/docs/backends.md)
apply to their documented graph APIs; the temporal implementation is NumPy.
Pin a release or exact commit for reproducible work. MIT licensed.

## Origins

Cadence started as an offshoot of a physics theory,
[Observer Patch Holography](https://github.com/FloatingPragma/observer-patch-holography),
which models reality itself as a metaphorical brain: a distributed network of
observers that finds global equilibria by repairing local conflict. The two
projects complement each other and share many of their theorems.
