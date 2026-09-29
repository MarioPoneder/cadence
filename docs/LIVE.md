# Learning through a continuing life

Cadence separates three kinds of memory. **Live activity** is the last admitted
settled state. **Long-term memory** is the retained weights and biases repaired
by experience; their ability to change is plasticity. **Temporal context** is
an explicit record of recent observations. These are different mechanisms.
Neither saved activity nor a larger observer population automatically provides
working memory, episodic retrieval, or protection against forgetting.

The helpers on this page are available in Cadence 0.50.0. They reuse
the same patch equation and qualified admission. They add no mandatory dependency
and do not simulate neurotransmitter chemistry. See the [reference](REFERENCE.md)
for every parameter and failure contract.

## Remember a recent observation

```python
from cadence import Cortex, History

history = History(2, steps=3)  # two sensory values per moment
cortex = Cortex(seed=2)
context = cortex.input("history", shape=history.shape)
base = cortex.column(patches=4, inputs=context)
observer = cortex.observer(patches=2, observes=base)
cortex.output("answer", shape=1, reads=observer)
brain = cortex.build()

history.push([0.8, 1.0])
history.push([0.0, 0.0])
inputs = {"history": history.push([0.0, 0.0])}
result = brain.step(inputs)
assert result["accepted"]
```

Each history block contains the supplied values and a presence mask; blocks run
oldest to newest. Padding has mask zero, so a real zero-valued sample remains
distinguishable. An occluded object needs an application visibility indicator,
as in the second coordinate above. The presence mask indicates an actual frame,
not whether every object was visible in it. `preview` computes the next encoding
without consuming a frame; `reset` clears the window at an episode boundary.

This is bounded **external history**, presented to the jointly settling brain.
The brain must learn how to use it. After a cue leaves the window, this mechanism
cannot recover it. Long-lived learned recurrent memory remains a distinct task.
Do not call adding history alone a demonstrated recursive-memory advantage.

## Learn choices from consequences

`Reinforcement` evaluates discrete actions through a scalar action-value output
of the same brain. The action enters as a one-hot sensor. Every processing and
observer population remains in each coupled solve. Epsilon exploration and the
comparison between completed action queries are explicit orchestration outside
that equilibrium; they are not a new neural readout or an emergent planner.

For a small fixed action set, a more efficient option exposes **one scalar output
per action from a single jointly settling brain**. Set `action_input=None` and
pass those output names as `value_output`. A learning update clamps only the
chosen action's output; the other patches remain free in that same solve. This
avoids a separate query for every action and lets action ranks vary directly
with the current sensory context. Cadence Pet uses this form.

```python
from cadence import Reinforcement

choices = Cortex(seed=2)
odor = choices.input("odor", shape=2)
values = choices.column(patches=3, inputs=odor)
choices.observer(patches=2, observes=values)
for i, name in enumerate(("rest", "left", "right")):
    choices.output(name, shape=(), reads=values, indices=(i,))
policy = Reinforcement(choices.build(), actions=3, action_input=None,
                       value_output=("rest", "left", "right"))
```

```python
from cadence import Reinforcement

layout = Cortex(seed=2, initial_scale=1.5)
senses = layout.input("senses", shape=3)
action = layout.input("action", shape=2)
perception = layout.column(patches=6, inputs=(senses, action))
reflection = layout.observer(patches=3, observes=perception)
layout.output("value", shape=(), reads=reflection)
learner = Reinforcement(layout.build(), actions=2, seed=2)

decision = learner.act({"senses": [0.3, 0.1, 0.0]})
assert decision["accepted"]
action_index = decision["action"]
# The environment executes action_index and returns its actual consequence.
admission = learner.feedback(0.0, {"senses": [0.2, 0.1, 0.0]})
assert admission["stored"]
```

Call `act` only when there is no pending action; call `feedback` once for that
action's actual consequence. `feedback(..., terminal=True)` has no next inputs
and no future-value term. An arbitrary collection timeout is not necessarily
terminal: use the next observation when future rewards continue. If reward arrives
later, intervening transitions can have zero reward; subsequent replay propagates
the later reward backward through learned value predictions. This is one-step
Q-learning with replay, not an eligibility trace or unlimited-delay guarantee.

For reward `r`, discount `g`, reward scale `R` and value scale `S`, the estimated
target is:

```text
y = (1-g) * S * r/R + g * clip(max_a Q(next_context, a), -S, S)
```

The second term is zero at a terminal state. This normalization keeps targets
inside the declared output scale for bounded rewards, rather than silently
truncating accumulated returns. It represents `(1-g)*S/R` times discounted
return. Higher discount reduces immediate target magnitude: it is not a free
increase in horizon. The finite output range and function approximation still
limit accuracy. No convergence or task-independent default guarantee is made.

Replay samples stored transitions, computes every target using the pre-update
brain, then calls `observe_batch(..., source="estimate")`. The numerical patch
rule is unchanged. The source label distinguishes a derived teaching target
from an actual observation, including in retry identity. It does not authenticate
the caller's evidence. Real reward and next sensing are observed; the fitted
action value is an estimate. Ordinary witnessed demonstrations still use
`observe` or `observe_batch` with their default `source="witness"`.

Invalid feedback does not consume the pending action. Valid feedback **does**
store the transition and consume the action even if its subsequent learning
attempt refuses. Retry with `replay`, not by pretending the outcome happened
twice. Query/fit refusals do not change learned parameters. Check `accepted`
before applying actions and after learning; `stored` alone does not mean learning
succeeded. `feedback(..., learn=False)` records a frozen-learning control.
The stored record also survives an exception during the subsequent fit; it
remains an actual experience even though no fitted update was admitted.

## Retain skills and predict the body

`parameter_prior` limits parameter movement within one admission. It is not
protected consolidation, a biological decay constant, or a guarantee that old
skills survive new experience. Batch replay mixes old and new experiences under
the same repair law; always measure old-skill recall after learning the new one.
The reinforcement store is bounded FIFO, so experiences eventually leave it.
Its latest transition is always included; the remaining batch is sampled without
replacement. `replay()` can run between decisions, with a declared work budget.

Action-conditioned consequence learning already uses witnessed targets:

```text
inputs  = previous sensory history + actual executed action
targets = subsequently measured position, contact or other body consequence
brain.observe(inputs, targets)  # actual measurements, not imagined outcomes
```

Design those input/output ports explicitly. Teaching a body predictor does not
by itself teach which action to prefer. Conversely, estimating reward does not
learn an accurate visual model of the body. The `live_control.py` example learns
a small body relation, queries actions, and chooses by predicted need reduction;
that comparison is an application controller, explicitly separate from neural
settlement. `live_learning.py` exercises reward-based choice and reversal.

## Drives and curiosity

Keep physical state in the body adapter: energy, fatigue, contact and actuator
strain are measured quantities. Supply relevant needs as sensors, and define
their consequences as bounded rewards. For example, food increases an energy
reserve, motion consumes it, and a reward can measure reduced energy deficit.
That supplies a preference; it does not script a route or select the food.

```python
from cadence import LearningProgress

curiosity = LearningProgress(rate=0.1)
assert curiosity.update("body_prediction", 0.5) == 0.0
bonus = curiosity.update("body_prediction", 0.2)
assert 0.0 < bonus <= 1.0
```

Supply prediction error against a **subsequently observed outcome**, measured
before teaching that outcome. Do not use numerical stationarity as curiosity.
The helper tracks an exponentially smoothed error by a bounded context key and
returns positive relative error reduction. This is a progress heuristic; noisy
decreases can also earn a bonus. It is neither information gain nor a reliable
noise detector. Check it against constant-error, random-noise and bonus-disabled
controls before relying on it for exploration. Context categories and mixing
weights are declared adapter choices or candidate genes, not learned facts.

A possible bounded composition is
`reward = (1-curiosity_weight)*external_reward + curiosity_weight*bonus`
with external reward in [-1,1] and weight in [0,1]. The application owns this
choice. Built-in homeostatic chemistry, learned neuromodulation, dynamic
plasticity gating and consolidation are not supplied. `exploration`, replay
frequency and `parameter_prior` are explicit functional controls; do not relabel
them as measured dopamine or claim they reproduce biological physiology.

## Keep the body responsive

```python
from cadence import LiveController, slew

def decide(observation):
    result = brain.settle(observation)
    return {"qualified": result["qualified"],
            "command": result["outputs"]["answer"]}

controller = LiveController(decide, fallback=(0.0,), max_age=0.25)
try:
    controller.submit(inputs)
    # A rendering/physics tick never waits for a completed settlement.
    command = controller.read()
    actuator = slew((0.0,), command, rate=2.0, dt=1/60)
finally:
    exited = controller.close(timeout=1.0)
```

One worker owns the callback and all brain calls it performs, including learning.
Do not access that brain from another thread. One pending sensory sample may be
replaced by a newer sample; never put the only copy of an unprocessed reward or
executed action into that lossy slot. Keep ordered experience in the serial owner
or a separate lossless application queue. A control decision that was never
executed must not become an action/reward transition.
When a callback uses `Reinforcement.act`, the body adapter must acknowledge
execution before supplying `feedback`. If the command expired or was discarded,
call `reset` from that same serial owner to abandon the pending action. The
generic worker does not infer whether a physical command was executed.

Command age starts at sensory submission, including queue time. Refused,
erroneous, expired or closed results use the supplied fallback. `slew` limits
actuator change, leaving target selection to the controller. Physics, animation
and low-level support remain identifiable. This thread wrapper is best effort:
Python's GIL and OS scheduling affect latency, GPU startup takes time, and an
in-flight solve is not cancelled. A false result from `close` means the worker
still owns its brain. Sweep budgets are not wall-clock deadlines. Measure both
decision latency tails and fresh-command rate, separately from rendering FPS.

## Save a whole life and run the small gates

`Reinforcement.snapshot()` includes the brain, replay records, pending action and
random generator. Restore it with `Reinforcement.from_snapshot`. History and
curiosity have their own validated snapshots; save them at the same paused
boundary as the learner and the environment. A brain-only snapshot has no
external sensory history, replay dataset or body state.

Run the bounded examples from the repository:

```sh
python examples/live_learning.py
python examples/live_control.py
```

These are small capability checks, not a complete pet or proof that recursion
beats a flat model. The tests cover hidden-cue recall through explicit history,
delayed reward and reversal, replay retention, saved continuation, consequence
prediction, numerical refusal and stale-command handling. Longer memory,
continuous-action reinforcement, learned planning and a browser creature remain
separate demonstrations.


## Qualify temporal context and delayed credit

`examples/temporal_credit.py` separates two bounded capabilities. In the cue
fixture, opposite initial cues have identical distractor suffixes and final
observations. Delays are 2, 4 and 8 observation ticks; the supplied `History`
window is `delay + 1`. Training uses cue amplitudes ±0.8 and two distractor
sequences, stopping checks use ±0.4 and a third sequence, and reserved tests
use ±0.3/±0.6 and two new sequences. Flat, ordinary-connected and observing
layouts each contain four patches. Their edges and parameters differ; this
is a capability comparison at matched patch count, not a depth advantage claim.
Each layout is queried with retained or reset activity, and again with history
removed. A saved brain and history resume partway through every test episode.
Retaining activity alone is not assumed to preserve an occluded cue.

The credit fixture has two actions and a supplied one-hot observation of the
current stage and the first executed action. Subsequent actions leave that
choice unchanged. Only the final transition supplies reward, +1 or −1 according
to the first choice. Delays 0/2/4/8 therefore mean 1/3/5/9 executed transitions.
The two possible rewarded choices are tested separately for every seed. Neither
the preferred choice nor a desired value enters the sensors or a witness target.
This fixture provides sufficient observed state to isolate reward credit from
the separate history experiment; it does not establish learned recurrent memory.

Each life collects 60 episodes, then executes 40 evaluation episodes with
learning disabled and exploration still 0.4. Success measures those actual
choices, not just greedy value rankings. The declared behavioral gate is at
least 0.65 success per TD case; an optimal policy with this exploration has
expected success 0.8. Frozen parameters and discount-zero learning are controls,
not additional acquisition claims. Disabling bootstrapping can still change
unvisited-state values through shared parameters, so its measured behavior is
reported rather than assumed to be chance. Discount is 0.8 for TD, replay stores
256 transitions and samples up to eight per update. The ideal first-choice
value magnitude is `0.18 * 0.8**delay`; attenuation and finite approximation
error limit the useful horizon. This tests one-step TD with replay, not eligibility
traces, unrestricted delays or a comparison of every return estimator.

One serial owner pairs each `act` with the transition it executes. Event time
is an integer simulation tick with one discount factor per transition; there is
no variable wall-time discount. At most one action awaits feedback. Terminal
means the reward horizon has actually ended. A collection pause instead saves
and resumes the learner, including a pending executed action, replay and RNG;
it does not create a terminal transition. A refused action is never executed.
A refused learning attempt preserves its actual transition but no parameter
update. Abandoned commands use `reset`; hypothetical `settle` queries own no
pending action and cannot receive feedback.

Run the bounded confirmation on seeds 2 and 7 (seed 0 is the development and
CI fixture):

```sh
python examples/temporal_credit.py --out /tmp/temporal-credit.json
```

The JSON preserves every scheduled case, trial outcome, resumed comparison,
source hash, learning configuration and helper-level solver-work sum, including
candidate queries, replay, evaluation, refused calls and continuation twins.
The report also records complete wall and CPU time. The source hashes must
remain unchanged during the run. These are two controlled demonstrations, not
one integrated autonomous life, learned episodic retrieval or evidence that
observers outperform conventional recurrent models.

The [source-bound confirmation receipt](../examples/receipts/temporal_credit.json)
contains all 66 cases for seeds 2 and 7. All 18 memory cases pass, with maximum
reserved full-history error 0.107. All 16 TD cases exceed the 0.65 executed-choice
gate; their individual success rates range from 0.725 to 0.900. The following
means pool both rewarded choices and both seeds (160 evaluation episodes per
cell):

| Reward delay | TD with replay | Discount-zero learning | Frozen parameters |
| --- | ---: | ---: | ---: |
| 0 | 0.750 | 0.750 | 0.500 |
| 2 | 0.806 | 0.625 | 0.500 |
| 4 | 0.794 | 0.363 | 0.500 |
| 8 | 0.781 | 0.631 | 0.500 |

Every saved/resumed comparison matched and no solve refused. The complete run
recorded 291 seconds wall time and 275 seconds CPU time while other local checks
were running; these are execution receipts, not production latency claims.
Controls sometimes succeed individually, and two seeds do not establish broad
statistical superiority. The seed-0 development and CI cases can be rerun with `--seeds 0`.
