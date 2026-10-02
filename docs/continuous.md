# Continuous interaction with GenericBrain

This page documents the existing `GenericBrain` composition and its feedback
timing. For the current temporal learner and private continuous-input planner,
see [learn, act and observe](interaction.md).

`GenericBrain.step` connects this composition to an environment. Each moment brings an observation,
feedback from the preceding action, and optionally a demonstration. The same recurrent
brain responds and changes its synapses throughout its life; there is no `train()` or
`eval()` switch. No replay buffer or separate training network is required.
Neurons hold bounded local state, exchange signals through declared synapses and read
back their current activity; records and feedback make the system self-reading.

Start with [a settling brain that decides](quickstart.md#a-settling-brain-that-decides);
[compose a brain](brain.md#genericbrain) builds this composition.

Reward and `done` concern the **previous action**; a teacher labels the **current
observation**. Each has one entry per batch row. Omitted reward means no reward event,
numerically zero; this is appropriate for an environment that supplies reward only
when events occur. The critic and eligibility still advance. It is not a way to skip
unknown transitions in offline data. For an ended row, supply its next episode's reset
observation; truncation can supply the previous episode's final value with `bootstrap`.
The first call has no previous action to reward. Keep stream identities until `reset()`.

## Fast activity, slow plasticity

In a parallel environment pool, some slots may be empty after their last episode.
With the lower-level `ActorCritic`, pass `learn(..., observed=active_rows)` so those
padding rows cannot teach from invented transitions. The boolean mask refers to
the action just taken, including a real terminal action. Inactive eligibility
resets, reward statistics ignore padding, and the update is averaged over real
transitions. When no slots remain active, stop the loop instead of calling `learn`.


One ongoing system still has distinct physical quantities and numerical timescales.
Neuron potentials change during settling. Synaptic weights change as observations,
demonstrations or reward supply a learning signal. Cadence alternates these updates:
weights are held fixed during each free/nudged phase, then local contrasts update them.
Records are outside this alternation: a write is one delta-rule step at the moment the
outcome is witnessed.
That separation preserves the numerical rule and its conditional gradient interpretation.
It does not claim that a biological brain runs these exact phases or that perception
and plasticity are identical processes.

An input is not automatically a correct action label. Repeating the brain's own guesses
as targets would strengthen mistakes. `step` uses actual rewards for the actor/critic
and actual teacher labels for imitation. The associative memory records only the chosen
action's observed reward; it never treats unobserved actions' predictions as evidence.
Lower-level `act`/`learn` and `Learner` remain available for experiments and custom wiring.
Use a separate instance for frozen `predict` or greedy `act` measurements.

## Repetition and salience become lasting synaptic changes

`GenericBrain.build(...)` includes `SynapticMemory` by default (`episodic=True`).
Use `episodic=False` to omit this associative pathway. A [records cortex](memory.md#records)
writes reward by the same delta rule through a sparse code, at `valued_rate` (1.0 by
default). `SynapticMemory` has a persistent
matrix `C` shared across streams and a transient residual `F` per stream. Each entry is
a synapse from a declared key neuron to a declared value neuron. The effective weight
is `C + F`; there is no list of remembered examples. For one unit key `k` and an actually
observed value `v`, its update is:

```text
F *= decay
alpha = min(1, consolidation * (1 + salience))
C += alpha * outer(k, v - k @ C)
F += rate * outer(k, v - k @ (C + F))
```

Defaults are `decay=0.9`, `consolidation=0.05`, `rate=1`. Repetition updates the slow
matrix even when the immediate fast response is already correct. Salience accelerates
that slow change. In the generic reward loop it defaults to `abs(reward)`; explicit
`salience=` is a nonnegative vector. This is a supplied importance signal. Positive and
aversive outcomes can both be salient, while the signed observed value determines what
is remembered.

For a batch, slow updates average over writing rows using the same pre-update matrix.
Fast corrections remain per stream. `value_mask` limits learning to observed output
components. Keys are normalized; orthogonal keys preserve each other, correlated keys
can interfere. Storage capacity stays fixed, and another observed value can revise a
consolidated association. The mechanism does not guarantee recall of every experience.

```python
import numpy as np
import cadence as cd

memory = cd.SynapticMemory(np.arange(3), np.arange(3, 5))
cue, outcome = np.array([[1., 0., 0.]]), np.array([[1., 0.]])
for _ in range(40):
    memory.observe(cue, outcome)
memory.reset(1)  # remove the transient residual; retain persistent synapses
assert memory.recall(cue)[0, 0] > 0.85
memory.observe(cue, np.array([[0., 1.]]), salience=np.array([19.]))
memory.reset(1)
assert np.allclose(memory.recall(cue), [[0., 1.]])
```

The controlled retention test leaves **0.05** of a unit target after one ordinary
exposure, **0.8715** after 40 repetitions, and **1.0** after one exposure with salience
19, after clearing all transient residuals. These are responses of the model above.
[Executable tests](../tests/test_continuous.py) also cover distraction,
correction, unseen value components, ongoing reward learning and checkpoint recovery.

## What changes in the wiring

Plasticity changes connection strengths, including previously zero weights. The declared
key/value contacts and neuron counts stay fixed. Adding/pruning anatomical connections
is a separate structural mechanism, and is not necessary to make a lasting change in
these synapses. This implementation does not model dendritic growth, protein synthesis,
biological synaptic tags or automatic transfer into the association cortex's weights.
The policy's own plastic weights continue learning from contrasts and eligibility.

Biological experiments motivate separating transient changes from their persistence:
repeated stimulation can establish lasting synaptic potentiation, and dopamine can
modulate spine plasticity in a restricted time window. The equations above are an
engineering model.
[Frey and Morris (1997)](https://pubmed.ncbi.nlm.nih.gov/9020359/),
[Yagishita et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/25258080/).

## Reset, save and cost

`brain.reset()` clears the current neural/eligibility state while retaining its memories.
`brain.hippocampus.reset(batch)` clears transient residuals and keeps consolidated
synapses. `brain.hippocampus.clear()` explicitly erases both. Observing a different
memory batch size clears transient residuals while retaining the consolidated matrix.
Reading a different batch size uses that consolidated baseline and preserves live records.
Unlike independent `FastSynapses` streams, these streams share long-term knowledge.

`GenericBrain.save/load` saves both memory timescales, policy, critic, optimizers,
working activity, random generators, and pending-action states. Resume the same row
identities and supply the pending action's actual outcome once. The environment/body
must be saved separately. Old generic checkpoints retain their original fast-memory rule.
When memory uses a `PatternSeparator`, the checkpoint includes its actual projection,
running mean and expanded memory matrices. Restoring does not regenerate the projection
from its seed. Ordinary older checkpoints without a separator remain supported; older
separated checkpoints that omitted the projection or mean are rejected because their
original coordinate system cannot be recovered reliably. Memory and continuation state
are validated for dimensions and finite values before exposing the resumed brain.
Invalid counters, action indices, variances or incomplete eligibility are rejected.
`parameters()` includes the consolidated matrix; per-stream residuals and eligibility
are additional storage. Persistent memory costs `key_width × value_width` numbers,
plus the same amount per stream for effective fast weights. Reads do not consolidate
or decay memory; only a new observation advances its update clock.

## Defaults and the thinking clock

| Mechanism | Default | When it advances |
| --- | --- | --- |
| Persistent neuronal state | On in `GenericBrain.step` | Each actual interaction continues the previous state |
| Reward plasticity and current demonstrations | On in `step` | Real transitions and supplied labels; no train/eval switch |
| Lasting associative synapses | On: `episodic=True`, `consolidation=0.05` | Observed outcomes; repetition and salience change persistent weights |
| Extra working-memory trace population | Off: `working_memory=False` | Opt in when the task needs a separate fading trace |
| Deliberation between actions | Available through `Deliberator`; application calls `tick` | Internal hypotheses, using supplied actions, transition and evaluator |
| Hidden background thread | None | The application owns scheduling, pause and shutdown |

These are defaults for the composed `GenericBrain`.
Existing checkpoints preserve their saved memory configuration. Enabling the default
associative pathway adds `sensory_width × action_count` persistent parameters and the
same number of fast weights per stream; disable it explicitly when reproducing an old
memory-free control. Supervised `fit`/`predict` remain independent-sample operations.

A task can stay active while the world waits. Use a separate **thinking clock** for
hypotheses and retained neural activity. Do not call `step` merely because another UI
frame passed: that would consume a real-action transition and replace its eligibility.
Continue raw neuronal dynamics with `Brain.settle(..., state=state)` when needed;
repeated settling under an unchanged drive may reach the same fixed point.
Deliberation changes hypothetical input so there is something new to evaluate.

`cadence.circuits.Deliberator` retains unfinished search across bounded ticks. It shares
its search rule with synchronous `imagine`, and publishes only fully completed depths.
Here a tempting immediate choice hides a bad later outcome:

```python
from cadence.circuits import Deliberator

# Supplied toy rules and evaluator, always scored for the same decision maker.
def value(path):
    if len(path) == 1:
        return 1.0 if path[0] == "tempting" else 0.0
    return -1.0 if path[0] == "tempting" else 1.0

thought = Deliberator(
    actions=lambda path: ("tempting", "safe") if not path else ("continue",),
    transition=lambda path, move: (*path, move),
    evaluate=value,
    terminal=lambda path: len(path) == 2,
    depth=2,
)
thought.start(())
while thought.pending:  # In a UI, call tick once per scheduled slice instead.
    completed = thought.tick(nodes=1)
assert completed.futures[0].action == "safe"
assert completed.depth == 2
```

In a running application, process incoming events first, then give thought a bounded
slice. `pause()` retains unfinished work, `resume()` permits more, and `cancel()` discards
obsolete work and candidate actions. `start(new_state)` snapshots the new observation
and replaces old work. Once the depth or total node budget is reached, ticks stop
spending compute; a persistent system need not busy-loop. If the budget cannot complete
even depth one, the result is `None`: the application must wait or use a labeled fallback.
`nodes` counts all visited positions, including unfinished and earlier search depths;
`result.nodes` describes the work at its last completed depth.

Use a fixed, read-only model/evaluator during each search. For a learned value readout,
`brain.basal_ganglia.value_of(brain.stimulus(hypothetical_observation))` settles an
isolated evaluation without replacing pending action credit. Imagined outcomes do not
write the live hippocampus or train the actor. After actual feedback updates the model,
restart the search so candidates do not mix old and new weights. The application must
explicitly connect completed candidate scores to its action-selection circuit; adding
a `Deliberator` does not automatically override `GenericBrain.step`'s sampled action.

The node budget bounds transitions and evaluations; a callback's own wall time lies
outside it.
Expensive world models need their own bounded evaluation or a worker.
This core planner restarts on new observations and
retains work between ticks of the same search; it does not cache across observations.

[Deliberation tests](../tests/test_deliberator.py) compare against independent minimax,
exercise pause/cancel/budget behavior and check real-action eligibility and memory
remain unchanged during hypothetical evaluation. [Memory tests](../tests/test_continuous.py)
measure lasting retention, correction and checkpoint recovery. Continuous operation
and prospective search are useful architectural functions; neither establishes
subjective experience or guarantees good plans with an inaccurate world model.

For a replay synchronized with a task, wrap each decision's feedback and action
in `record_settlements` and attach the records to the exact observation that
produced them. Save the actual motor command, signed `ActorCritic.learn` dopamine,
and the task's own frame or timestamp alongside the neural records. A normal-speed
screen cannot display every iteration separately: keep all steps in the recording
and provide pause and single-step inspection. Do not synthesize extra oscillations
or infer a signed reward signal from the absolute `delta` statistic. See the
[recording API](api.md#record-every-settling-step) for its cost and precision limits.
