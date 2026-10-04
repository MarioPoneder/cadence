# One continuing equilibrium brain

**Cadence is not a feed-forward deep neural network.** Its organizing idea is a
continuing equilibrium brain that acquires a reusable interpretation of
its world, carries it through experience, and repairs it when experience disagrees.
Perception, memory and action should constrain one another through reciprocal
connections. The answer belongs to the brain's settled state. Reading its motor
choice exposes that state; a separately trained answer-producing decoder would
be another model.

Start with `Brain.compose`: the default **System 1** has reciprocal processing
regions, local plasticity, working trace and fast/persistent associative memory.
It can already be deep. Optional **System 2** adds observing regions with returning
feedback inside the same neural settlement. Biological names describe software
roles, not a literal biological implementation.

| Wrong as the flagship application | Right organizing lifecycle |
| --- | --- |
| Rebuild a classifier for each observation, bypass memory, and count label accuracy as a world model. | Bootstrap one brain, retain its acquired relations and memory, use it, witness failures, repair and continue. |
| Have an external trained readout or language model supply the answer. | Let reciprocal regions constrain the answer read from the brain's declared motor population. |
| Call any small residual a correct, cheap interpretation of the world. | Check the numerical equations, actual task outcomes, retention and measured work separately. |

Isolated classifiers and calibration remain useful unit controls. The distinction
is what capability the application demonstrates, not whether those tests exist.

![A feed-forward network passes activity one way and takes gradients from an
outside controller; a patch net's neurons hold local state and settle together
over reciprocal synapses, with no layers, no order and no controller](assets/patchnet.svg)

## Bootstrap, use, repair, continue

1. **Bootstrap a useful interpretation.** Present real observations, consequences
   and demonstrations to one brain. Let connections and memory acquire relations
   that can support later answers with the teaching signal absent. Judge usefulness
   on held-out experience and retention, separately from numerical settlement.
2. **Operate with what was learned.** New evidence and remembered context change
   the boundary of the solve. Reciprocal regions seek a compatible present state;
   the brain acts and the environment supplies the actual consequence.
3. **Repair a witnessed failure.** The application compares an issued action or,
   when its chosen model supplies one, a saved prediction with the observed
   outcome before correction. A false expectation or missed goal supplies task
   evidence; a large equation residual instead means the numerical solve has
   not qualified. `Brain.step` receives that measured reward; it does not make
   an environmental transition prediction or compare a corrected utterance.
4. **Resume the same life.** Retain acquired parameters, relevant memories and
   pending feedback through correction. Measure recovery and old capabilities
   after the disturbance, without rebuilding the brain for each observation.

The stored relationships and memories encode a **family of possible equilibria**
under different evidence and context. Learning changes that family. Ordinary use
does not mean holding one activation vector forever: even a well-learned world
requires different states as observations, goals and context change.

These are bounded observer-like software patches. In `Brain.compose`, the
boundaries are sensory and motor neuron indices; readback exposes neural state
and numerical residuals; the retained records are working traces and learned
cue/outcome associations. Reciprocal signals and local plasticity provide
feedback and repair. Explicit record-field ports and the population solver's
exact error-readback mechanism are separate interfaces. Public evidence must
support behavioral claims. A numerical fixed point only checks the declared internal
equations. It does not establish external truth, useful meaning, retention or a
unique stable interpretation.

## What the current interface provides

The current `Brain` supports a continuing equilibrium interpretation, action and
memory. This realizes part of the world-model hypothesis; it does not yet
integrate a learned model of environmental transitions. Its
`step` consumes the preceding action's measured outcome, optionally teaches the
current observation, then settles and selects another action. This small loop
keeps one stream and both memory pathways active:

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(2, 2, modules=(8,), seed=7)
observations = np.eye(2)
cue = 0
action = brain.step(observations[[cue]], teacher=np.array([1 - cue]))

for transition in range(8):
    # The toy body's rule changes halfway through this continuing life.
    target = 1 - cue if transition < 4 else cue
    reward = (action == target).astype(float)  # actual executed action
    cue = 1 - cue
    action = brain.step(
        observations[[cue]], reward=reward, done=np.array([False]),
    )

assert brain.learner.updates > 0
assert brain.hippocampus is not None and brain.hippocampus.writes > 0
```

This exercises continuing feedback through a changed environment; eight outcomes
do not establish acquisition or recovery. The [runnable lifecycle example](../examples/continuing_brain.py)
separates bootstrap, unchanged conditions and changed conditions, records actual
outcomes, and verifies identical continuation from a saved pending action.
[Continuous interaction](continuous.md) specifies event order, refusal/retry,
memory writes and reset semantics. The environment must be saved separately.

| Mechanism | Implemented contract and boundary |
| --- | --- |
| Present interpretation and action | `Brain.act` checks the whole neural-graph residual with held trace and memory input. It does not jointly equilibrate the auxiliary memory stores. |
| Inspectable settling work | `Brain.last_settlement` retains free-answer residuals, sweeps, checks and qualification, including refused attempts. It excludes eligibility, teaching, feedback and memory work. |
| Short and long memory | Working traces and fast/persistent associations influence actions; graph parameters also retain learning. Their capacities, update clocks and interference differ. |
| Local correction | Current teacher labels change graph parameters; actual chosen-action outcomes drive reward plasticity and associative writes. A teacher label does not automatically become a stored event or credit an earlier sequence. |
| Private imagination | `Brain.imagine` evaluates **supplied** observation sequences with private trace and read-only durable memory. It does not learn or generate environmental transitions. |
| Dreaming and sleep | `RecordPatchNet.dream` completes cues using the model and records; `sleep` teaches fixed completions to slow weights and rewrites residual records. This supported [sleep cycle](record-patch.md#acquisition-in-two-phases-records-by-day-weights-by-night) is distinct from `Brain.imagine` and online associative consolidation, and is not integrated into `Brain.compose`. |
| Learned consequences | `TemporalPatchNet` provides a separate learned temporal model and planning interface. Records provide other explicit prediction mechanisms. These are not automatically integrated into `Brain.compose`. |
| State-and-error population experiments | `cadence.experimental.equilibrium` qualifies a joint stationary state under its own energy. It uses explicit `History`, not the default Brain's trace and associative memory; its law and guarantees stay distinct. |

The full world-model lifecycle is the design direction, not a completed default
capability. In particular, current `step` processes real feedback even when the
task succeeded; it does not gate every update on witnessed failure. Warm state
can help settlement, but a general contract for cheap stable operation,
automatic mismatch-triggered local repair and integrated learned consequences
still needs implementation and behavioral evidence. This guide adds no new
threshold or success policy. Use each model's [actual contract](contracts.md).
The implementation work is owned by [learned-consequence integration](https://github.com/muellerberndt/cadence/issues/93)
and [failure-driven repair and qualified reuse](https://github.com/muellerberndt/cadence/issues/122).

Two narrower mechanisms already exist. [Centered dopamine](reward.md#centered-dopamine-and-selective-activity)
can suppress familiar actor modulation when `dopamine_center` and
`dopamine_floor` are configured; critic updates, momentum, eligibility and
memory writes can still do work. [Life](api.md#life-cadencelife) selects habit,
imagination and learning for a separate belief/steering composition. Neither
silently changes `Brain.step` into a universal failure gate.

The default sensory projection feeds a chain of processing regions; neighboring
regions and the association/motor pair exchange reciprocal signals. Processing
regions have no internal synapses by default. Even `modules=(64,)` therefore
uses recurrent settlement with memory, rather than a single feed-forward pass.
More modules or optional observers change connectivity; their behavioral benefit
must be measured. See [equilibrium world models](equilibrium-world-models.md)
for the distinction between stochastic innovation and a learning error.

## Build and measure the whole life

Keep the same brain across acquisition, use, interference and repair. Preserve
stream identities, and reset transient state only at declared boundaries.
`predict`, `accuracy` and `fit` are useful independent-sample controls: they omit
working and associative memory, and `fit` clears pending stream state. Label them
as controls when investigating graph learning or calibration. They are not the
default demonstration of a continuing memory-using brain. A frozen evaluation
can load a checkpoint; it must retain the state relevant to the tested capability.

For language, the intended result is understanding and producing full sentences
with compositional vocabulary through that continuing system. A fixed word
classifier tests a much narrower mechanism. An external language model or
trained readout that supplies the answer must be counted as a separate baseline,
not presented as the brain's acquired language. Output coding alone does not
establish sequence learning, meaning or language competence.

Measure numerical residuals, task quality, memory retention, recovery and work
separately. Count bootstrap, stable use, learning, replay, imagination and refused
attempts. A small residual is neither low measured energy nor evidence of
transformer-level performance. Greater capability, scalability and efficiency
are targets for matched comparisons, not consequences of the word equilibrium.

Continue with the [quickstart](quickstart.md), [composition API](brain.md),
[experience design](experience.md) and [temporal planning](planning.md).
