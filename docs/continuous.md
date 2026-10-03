# Continuous interaction with Brain

`Brain.compose` creates a continuing **System 1** brain with working
trace, plastic connections and fast/persistent associative memory. Optional
observer regions add **System 2** feedback in the same neural graph. `step`
connects either layout to its body: observe, learn from the preceding outcome,
then act again. There is no training/inference mode switch.

This is the runtime loop for [one continuing equilibrium brain](world-model.md).
Bootstrap, unchanged conditions and witnessed disruption belong to the same
life. Current reward learning processes each actual outcome; it does not yet
implement a universal policy of learning only after failure. Keep a task error
separate from a numerical failure to settle.

## Observations, actions and reward

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(4, 2, modules=(16, 8), seed=7)
observation = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(observation)

# A tiny body rewards action 0, then reports its next observation.
reward = (action == 0).astype(float)
following = np.array([[0.0, 1.0, 0.0, 0.0]])
action = brain.step(following, reward=reward, done=np.array([False]))
assert action.shape == (1,)
```

Reward and `done` describe the **previous action**. `teacher=` labels the
**current observation**. Every vector has one entry per stream; observations
have shape `(streams, inputs)`. Keep the same stream in each batch row until
`reset()`. The first call has no preceding action to reward.

For an ended row, pass its next episode's reset observation with `done=True`.
A truncation can supply a final value through `bootstrap`. Omitted reward means
zero reward on an actual transition, not an unknown outcome or permission to
advance before the body acts. The critic and eligibility still advance.

```python
# A separate life receives a demonstration for its first observation.
student = Brain.compose(4, 2, modules=(16, 8), seed=7)
student.step(observation, teacher=np.array([0]))
```

Use actual labels and consequences. Teaching from the brain's own guesses can
reinforce mistakes. `teacher` changes graph parameters; it does not write its
label into associative memory. The reward loop records the chosen action's
observed reward, with unobserved action entries masked. For frozen measurements,
use a separate instance. Its `predict` and `accuracy` ignore both the working
trace and associative memory; greedy `act` reads both and advances the trace.
Lower-level `act`/`learn` separates action and feedback timing.
The [continuing brain example](../examples/continuing_brain.py) runs this loop
through bootstrap, unchanged conditions and changed conditions, then checks a
saved pending action's continuation. It reports actual task outcomes without
assuming that settlement guarantees recovery.

The example supplies a corrective label only after an executed mistake and
repeats that cue for the next action. Thus its teacher labels the **current**
observation while reward still describes the previous action. Successes still
reach reward learning and associative memory. This is an application teaching
policy, not automatic suppression of all successful-outcome updates. See
[centered dopamine and Life](reward.md#centered-dopamine-and-selective-activity)
for the narrower selective mechanisms already available.

## Qualification and refusal

Every `Brain` action and independent prediction must satisfy the full
neural equation residual, including optional observers. Default free answers
have a 1024-step budget and tolerance `3e-3`. The solver may use half-step
numerical damping within that total budget, then checks the original model's
residual. This numerical fallback does not alter the finite teaching rule or add
a second controller.

A refused `act` raises `RuntimeError` before changing activity, memory, random
state or pending feedback. If `step` has learned a real outcome and its following
action refuses, that learning remains. Retry `act` after adjusting the solve;
do not submit that outcome twice. `tolerance=None` cannot disable action
qualification. See [contracts](contracts.md) for the separate finite
free/nudged learning and eligibility rules.

Inspect the latest free-answer attempt without changing the action return type:

```python
answer_work = brain.last_settlement
assert answer_work is not None
assert answer_work["qualified"]
assert answer_work["steps"] <= answer_work["budget"]
assert answer_work["max_residual"] <= answer_work["tolerance"]
```

`last_settlement` is a read-only snapshot with per-row residuals and admission,
sweeps, residual checks, damping halvings and stagnation comparisons. A refused
answer leaves its report available after the exception. It describes the last
completed free-answer solve: validation errors and failures before that solve
do not replace it. `operation` distinguishes `act` from independent `predict`;
`step` delegates to `act`. `reset` and loading a checkpoint clear this diagnostic.

This is not a total cost report. It excludes feedback, eligibility, teaching and
memory work; `last_learning` reports learning separately. Optional
[settlement recording](api.md#record-every-settling-step) captures the other
solver calls at additional cost. The continuing example records those sweeps
too. No numerical counter substitutes for measured task success or hardware
energy, and a surprise need not increase settling work.

Teaching diagnostics, eligibility duration and feedback rollback have
separate contracts:
`LearnerConfig(qualified=True, ...)` requires full-equation qualification
of every supervised free and teaching phase before changing parameters.
`LearningPhaseError` reports an attempted lesson that could not qualify.
Qualified solves use configurable damping; stagnation detection can
admit a smaller numerical step sooner within the same budget and original
equation check. The default finite configuration keeps its half-step fallback.
Accepted and refused teaching diagnostics appear in `brain.last_learning` with
`demonstration_` prefixes, alongside any preceding accepted reward report. They
count attempted presentations and solver work. See [building a brain](brain.md)
for a runnable configuration. Reward eligibility is configured independently:
`ActorCriticConfig.eligibility_steps=12` is the Brain default. For a standalone
`ActorCritic`, `None` uses its learner's `nudged_steps`.

If learning cannot qualify the next state needed for a reward bootstrap, it
preserves the preceding action, memories, optimizer state and random state.
Adjust the solve and retry that same outcome. Once the outcome has been accepted,
a later action refusal has the different retry rule above: call `act` alone.

## Private imagination

```python
phases = brain.imagine([observation, following], budget=1024, tolerance=1e-6)
assert phases
```

Each phase reports its state and `converged` flags. A refused phase is returned
and ends the branch. The private trace can carry earlier hypothetical observations
forward, while live activity, durable memory, random state and pending outcomes
stay unchanged. These are responses to supplied observations, not predictions of
what the environment will do. [Temporal learning and planning](interaction.md)
provide the separate learned action-consequence interface.

## Repetition and salience become lasting synaptic changes

`compose` includes a working `Trace` and `SynapticMemory`; `episodic=False` omits
the associative pathway. The trace retains earlier activity as input to later
settlement. Durable graph plasticity retains learned weights and biases, while
the associative pathway retains cue-to-outcome records. These are separate
mechanisms with separate acquisition and retention tests. Losing an old response
after new teaching can be parameter interference even when no working trace is
involved. A warm numerical starting state alone does not guarantee recall.

`SynapticMemory` has persistent matrix `C`, shared across streams, and fast
residual `F` for each stream. With a normalized key `k` and an observed value `v`:

```text
F *= decay
alpha = min(1, consolidation * (1 + salience))
C += alpha * outer(k, v - k @ C)
F += rate * outer(k, v - k @ (C + F))
```

Defaults are `decay=0.9`, `consolidation=0.05` and `rate=1`. In the reward loop,
salience defaults to `abs(reward)`; explicit `salience=` supplies a nonnegative
vector. This is a supplied importance signal. The signed value determines what
is remembered. Batch writes average persistent updates over observed rows,
and a value mask excludes unobserved actions.

```python
from cadence import SynapticMemory

memory = SynapticMemory(np.arange(3), np.arange(3, 5))
cue = np.array([[1.0, 0.0, 0.0]])
for _ in range(40):
    memory.observe(cue, np.array([[1.0, 0.0]]))
memory.reset(1)  # Clear fast residuals, retaining persistent associations.
assert memory.recall(cue)[0, 0] > 0.85
```

Orthogonal keys preserve one another under the stated rule; correlated keys can
interfere. Storage is fixed and new observations can revise associations.
Rehearsing actual earlier observations can protect recall, but consumes extra
writes and does not guarantee retention. Test old and new responses in the same
brain after intervening experience, with answers free and the relevant transient
state cleared. A successful record-store test does not establish acquisition by
the graph's contrast rule.
Persistent memory costs `key_width × value_width` numbers, plus that amount per
stream for fast weights. Reads neither consolidate nor decay memory. Learning
weights does not require growing new anatomical connections.

## Reset and save

`brain.reset()` clears live neural/eligibility state and the working trace,
retaining associative memories. `brain.hippocampus.reset(batch)` clears fast
residuals but retains persistent synapses; `clear()` erases both. Changing memory
batch size resets fast residuals on a write; reads use the persistent baseline
without changing the live memory.

```python
brain.save("continuing-brain.npz")
resumed = Brain.load("continuing-brain.npz")
```

Save/load includes parameters, critic, optimizers, traces, fast and persistent
memory, random state and an action awaiting feedback. Resume the same rows and
supply that action's actual outcome once. Save the environment separately.
If a pattern separator is used, its actual projection and running mean are
saved too. Shapes, finite values and continuation state are validated on load.

## Defaults and the thinking clock

| Mechanism | Default in `Brain.compose` | Advances on |
| --- | --- | --- |
| Neural activity | Retained | Actual interaction |
| Working trace | Included | Each admitted action's free state |
| Reward plasticity and demonstrations | Available through `step` | Actual outcomes and supplied current labels |
| Fast/persistent associations | Included | Observed chosen-action outcomes |
| Recursive observers | Empty unless requested | The same neural solve when included |
| Private imagination | Explicit call | Supplied hypothetical observations |

`Brain.build` is a separate configurable builder; its working trace is
opt-in. No hidden thread drives either interface. Do not call `step` for every UI
frame: it consumes a real transition. The application owns scheduling.

Advanced `Deliberator` search retains unfinished work across bounded `tick`
calls using supplied actions, transition and evaluator. It does not automatically
override the brain's action. Keep the model fixed during a search and restart
after learning; imagined outcomes must not train the live brain. Its node budget
does not bound a callback's wall time. See [the API](api.md) and
[deliberation tests](../tests/test_deliberator.py).

For partially filled environment pools, lower-level `ActorCritic.learn(...,
observed=active_rows)` masks padding transitions. Stop when no real rows remain.
For visualizations, record actual iterations and their action/outcome identity
with [record_settlements](api.md#record-every-settling-step); do not invent
oscillations or a reward signal from an absolute update statistic.
