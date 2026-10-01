# Designing an efficient, capable brain

Start with a task, sufficient observations and a measurable acquisition check.
Choose the smallest connected layout that passes that check at an acceptable
cost. Add representation, temporal context or recursive observation to solve a
specific remaining failure, then measure again. There is no universally optimal
width or depth.

Cadence uses one processing-patch rule throughout a brain. Each patch has local
state, incoming ports, retained relations and exact prediction-error readback;
connected constraints return influence through the same joint repair. A qualified
answer covers the whole declared graph. It does not certify task success, a
unique answer or the global energy minimum.

This guide uses the public API. The [quickstart](QUICKSTART.md) introduces its
operations; [migration notes](MIGRATION_060.md) describe the 0.60 candidate and
checkpoint compatibility. The [reference](REFERENCE.md) gives exact signatures.

## Define the body's information and outcomes

Write down what the brain observes, what its outputs mean, what the body actually
executes, and which later observation establishes success. For navigation, a
current image may omit velocity, the last turn or a recently hidden landmark.
For music, a few recent notes may suffice for local rhythm while omitting the
phrase whose return matters much later. More patches cannot reconstruct
information that never reaches them.

One body adapter can supply all named sensors and receive all named outputs.
Internal populations do not need separate external evaluators. Supervised
witnesses enter through selected output names in `observe`; reward-driven
choices use `Reinforcement`. The application still defines the measurements,
reward units and actuator behavior. The library cannot infer the objective of
an unfamiliar environment from a sensor shape.

Normalize inputs consistently and encode targets comfortably within both the
`tanh` prediction range and the state bound. With defaults, values such as
`-0.6` and `0.6` are useful starting units. Increasing `state_bound` does not
expand `tanh`. Fit preprocessing using training data only and preserve it with
the application checkpoint. For classification, expose distinct score patches
and decode the largest score; scores are not normalized probabilities.

Split recordings by independent episodes, performances or environments before
extracting overlapping windows. Reserve development data for choosing settings
and fresh assessment data for the final measurement. Future observations may
supply teaching targets; they must not enter the inputs used to forecast them.

## Choose a layout for the missing capability

| Pattern | A sensible first use | What to check |
| --- | --- | --- |
| Input-only flat population | Independent scalar relations or a small immediate control mapping | One connected patch per required output may suffice. Unused neighboring patches supply no hidden representation. |
| Ordinary composition | A learned combination of intermediate features | Later populations read earlier live states; returning energy derivatives already provide feedback. |
| Recursive observation | Testing whether current representation errors help a downstream relation | An observer reads both states and exact errors. Compare with ordinary composition at declared information, parameter count and work. |
| Parallel branches with fusion | Sensors with different local structure or update meaning | Every output must have a useful path to the observations it needs. |

Fast routine behavior does not imply a single input-only layer. A familiar
skill may require learned intermediate features and temporal memory even when
it needs no recursive error observation. The flat pattern above has no hidden
representation: each output predicts from its own weighted sensor inputs.
Additional unconnected output patches do not recover the missing computation.
Compare a capable ordinary layout with its observer extension before attributing
a failure or improvement to self-observation.

Width counts processing states. Observation depth means an observer reads
another observer. Neither a population called `reflection` nor extra settling
sweeps demonstrates reasoning. The public builder accepts previously declared
sources only: it supports jointly coupled state/error constraints, but does not
expose explicit recurrent state cycles. Retaining activity with `step` alone is
not evidence of useful temporal memory.

The [three layout quickstarts](VARIANTS.md) and
[shared executable example](../examples/layout_learning.py) teach, query and
save each pattern with the same `signal`/`answer` body interface. They are
construction and acquisition examples, not matched-capacity comparisons.

Inspect `connections`, `output_connected_patches` and each output's
`sensor_coverage_by_coordinate`, not just total patch count. These report
structural reachability, not measured causal influence: saturation or learned
cancellation can suppress a connection. Perturb relevant inputs and measure
changes in free outputs. For several scalar outputs from one population, use
explicit different `indices`; otherwise each defaults to patch zero.

Default wiring connects all coordinates of each declared source. A positive
`fan_in` requests sparse wiring and changes what each patch can read. Dense
input-to-population connections cost roughly input width times population
width; dense population-to-population connections cost the product of both
widths. Observation adds error contacts as well as state contacts. Check the
compiled graph against `max_connections` before increasing image resolution,
history length or population size. See [layout variants](VARIANTS.md).

## Make temporal information explicit

When the latest sample is ambiguous, first establish a sufficient-information
control using a bounded `History` window. Its presence masks distinguish initial
padding from observed zeros. This is explicit application memory, and its size
must count in a comparison.

```python
from cadence import Brain, Cortex, History, bootstrap

history = History(1, steps=3)
history.push([0.2])
encoded = history.push([0.4])
assert len(encoded) == history.size == 6

history_layout = Cortex(seed=2)
recent = history_layout.input("recent", shape=history.shape)
response = history_layout.column("response", patches=1, inputs=recent)
history_layout.output("answer", shape=1, reads=response)
history_brain = history_layout.build()
assert history_brain.settle({"recent": encoded})["qualified"]
```

Choose the window from the task's causal horizon and sampling rate. Test longer
occlusions or delays separately. A model given an explicit history window has
more temporal information than one given only the latest observation. To claim
learned memory, remove that extra history in a declared control, disable learning
during recall, and test writes, retention, replacement and reset. Ordinary
activity retention and recursive error readback do not guarantee those abilities.

`observe_batch` does not make a sequence: its rows have private activities,
each initialized from the same retained live state. An accepted batch retains
shared parameters and preserves live state. Ordered environment interaction
requires ordered application records and calls.

## Acquire a small ability before scaling

Begin with defaults and one directly sensing patch per simple output. For a
small nonlinear relation, a connected representation of 4–16 patches feeding
1–4 output patches is a starting search range, not a capability guarantee.
Use [bootstrapping and size](BOOTSTRAP.md) for more detailed recipes.

`state_prior` changes the activity penalty and therefore the preferred answers;
it is not simply a speed setting. `parameter_prior` anchors relations to their
values immediately before each admission, with that anchor fixed throughout
the solve. A larger value penalizes each parameter change more strongly and can
require more exposure; it is not permanent protection for old skills. Start
with defaults and compare a small declared set of settings on development data.
Freeze the choice before final assessment, and count that calibration work.

Here the supplied body relation is `answer = 0.6 * signal`. Development checks
control stopping; the final four probes are separate from both teaching and
those checks.

```python
layout = Cortex(seed=2)
signal = layout.input("signal", shape=1)
response = layout.column("response", patches=1, inputs=signal)
layout.output("answer", shape=1, reads=response)
brain = layout.build()
examples = [
    ({"signal": [x]}, {"answer": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
checks = [
    ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)
]
report = bootstrap(
    brain, examples, checks=checks, max_error=0.1,
    epochs=20, batch_size=4, seed=2,
)
assert report["passed"], report
assert report["accepted"] == 4 * report["updates"]

for x in (-0.7, -0.2, 0.2, 0.7):
    query = brain.settle({"signal": [x]})  # no answer clamp
    assert query["qualified"]
    assert abs(query["outputs"]["answer"][0] - 0.6 * x) < 0.1
```

The example shows acquisition on a tiny numerical relation. It is not a recipe
for raw vision, autonomous navigation or long-term planning. Set a task-specific
error or behavior threshold before comparing architectures. For control, run
complete episodes with the teacher disconnected; agreement on demonstration
states alone can hide failure on states reached by the learned policy.

Give each candidate enough admissions to measure a learning curve. Count
distinct experiences, replayed presentations and accepted updates separately.
A 16-row batch is one parameter admission involving 16 examples, not 16 serial
updates. Check earlier skills after new learning: neither parameter anchoring
nor replay guarantees protected retention.

Compare fresh initializations and preserve refusals, timeouts and unsuccessful
candidates. If ordinary composition and observation see different inputs or
parameter counts, state the difference. Report both equal-exposure and declared
work-budget comparisons when sample efficiency and compute efficiency differ.
Increasing depth after failed acquisition is a hypothesis to test, not a repair
for missing inputs, poorly scaled targets or insufficient training.

## Distinguish settlement, prediction error and surprise

| Measurement | What it tells you | What it cannot establish |
| --- | --- | --- |
| `stationarity` and `qualified` | The complete projected repair residual meets the requested numerical tolerance | Correct predictions, good actions or a globally minimal state |
| `prediction_residual` and patch `errors` | Current disagreement between internal state and the current local prediction | Whether an earlier forecast was contradicted by a later real observation |
| Actual forecast error or task outcome | How a committed prediction or executed behavior compared with later evidence | Which earlier internal relation deserves credit without a declared learning procedure |

Keep the original forecast, observation context, action identity and model
identity before its outcome arrives. Compare that fixed forecast with the later
measurement. Recomputing a prediction after learning can erase the very error
you wanted to measure. `LearningProgress` summarizes changes in supplied
predictor errors; it is not an automatic surprise detector or information-gain
measure. Even with unchanged parameters, an internal prediction and the final
settled output can differ because of priors and coupling. Use the original
issued output when measuring forecast error.

Targets also change the settling problem. An observer's error inputs during
joint teaching can differ from the signals available when the future answer is
free. Evaluate with that answer unclamped. When diagnosing an apparent readback
benefit, compare those two signal conditions with identical parameters and
causally available observations. A low teaching loss is not sufficient evidence
of a useful free prediction. Observers also read states, so zero error does not
make their input or influence vanish; it is not a safe automatic sleep signal.

## Keep actions and outcomes attached to the right life

Use `settle` for a pure query, `step` to retain qualified activity, and `observe`
for actual labeled targets. Check `qualified` before using query outputs and
`accepted` before treating an update as committed. `predict` raises
`SettlementError` on numerical refusal. Diagnostic outputs from a refused call
are not actions.

For discrete reward-driven choices, `Reinforcement.act` proposes an action and
issues a `decision_id` when accepted. Execute through the body, then call
`feedback` with the same identifier and the actual `executed_action`; actuator
overrides must be recorded. Do not reward a hypothetical or abandoned command.
`reset()` discards pending ownership: record an executed action's outcome before
resetting for a new episode. An identical latest feedback retry is
idempotent; stale or conflicting outcomes are rejected.

The helper stores actual rewards and transitions, then fits derived Q targets
through `observe_batch(source="estimate")`. Those targets are estimates, not
measured future rewards. A stored outcome survives a refused fit; retry fitting
with `replay()`. Direct supervised `observe` calls instead use `source="witness"`
for measured targets and explicit event IDs when external retries need identity.

A reward can influence connected relations through joint repair, but there is
no automatic brain-wide emotion, responsibility assignment or task evaluator.
The current temporal-credit interface and its bounded scope are documented in
[the live guide](LIVE.md) and [reference](REFERENCE.md#reinforcement-discrete-reward-driven-choices).
A larger credit horizon is not a learned world model or a planning algorithm.
Keep the single body-level outcome stream lossless; `LiveController`'s replaceable
sensory slot must not hold reward or transition records.

## Spend compute according to measured need

“System 1” and “System 2” can describe cheap familiar responses and more costly
context-dependent correction. They are not Cadence modes or constructor flags.
They also do not specify layer count or distinct patch types. The current
library uses the same patch rule in ordinary and observing populations; this
design choice does not establish that it can replace every specialized memory
or processing mechanism at an acceptable cost.
Flat and observing populations can coexist in one brain, with the same input
and output boundary. Every participating population remains in the whole-brain
energy, and qualification checks every eligible free coordinate. Adding a slow
observer does not automatically let a fast
branch issue actions while that observer sleeps.

> **Current capability boundary.** The intended cycle is learned routine →
> actual disturbance → useful corrective processing → restored, inexpensive
> routine, while retaining the skill. Automatic internal allocation of attention
> and shared long-term outcome responsibility are not yet implemented and
> validated as that integrated cycle. No extra application attention flag or
> per-population evaluator should be needed for the intended design. The body
> still has to supply observations and actual outcomes.

Test the whole cycle before claiming that a correction mechanism works. First
establish autonomous routine competence with teaching disconnected. Apply a
specified disturbance to the body; feed back what it actually does, including
actuator overrides and exhausted resources. Then measure recovery, retained
skill and the return to inexpensive decisions. Do not substitute low internal
residuals, training agreement or a nominal output flag for observed behavior.
A music command to hold a note, for example, does not prove a finite sample
continues sounding. Legitimate variation also needs to remain possible; a goal
of minimizing every deviation would suppress a useful musical fill.

Current tools do not establish that a deep observer hierarchy becomes useful
merely through long training. Measure any scheduling, curriculum and body-level
task measure as part of the application. Numerical reuse of invariant arithmetic
is a supported optimization; learned selective attention is a different claim.

Measure the complete workload: queries, selected actions, teaching, replay,
readiness checks and refused attempts. Sum `work` counters from results or use
the helper's aggregate counters without counting the same work twice. Record
wall time, latency percentiles, qualification rate and task quality alongside
`evaluations`, `edge_visits`, `patch_visits`, `proposals`, `backtracks` and sweeps.
The sweep `budget` is not an elapsed-time deadline. A stationary answer can be
cheap while still being wrong about the body.

Pure input-only queries can settle quickly; coupled state and error contacts
add dependencies and returning derivatives. The current single-row reference query cache
reuses mathematically invariant input-only and bias-only predictions, with fresh
final qualification. It does not freeze a slow population, skip error feedback or
change the learning objective. See [performance](PERFORMANCE.md).

Start with `device="python"` for small graphs. Optional `device="cpu"` uses
PyTorch on the CPU; `"cuda"` or `"mps"` select supported GPU execution. Device
proposals still require float64 reference qualification and may need reference
refinement. Include that cost, transfer/setup overhead and changed acquisition
behavior when comparing backends. Small brains may lose time to device overhead.
The [acceleration guide](ACCELERATION.md) explains precision and batch choices.

Parallelize independent lives or environment collection. Serialize calls that
read or mutate one brain's continuation; independently learned checkpoints
cannot be averaged as though they were one sequence of experiences.

## Preserve and diagnose continuation

```python
saved = brain.snapshot()
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"signal": [0.5]}) == brain.predict({"signal": [0.5]})
```

Brain snapshots bind exact implementation sources, layout and configuration.
An older snapshot is not automatically compatible with a newer release; retain
the originating installation and follow the [migration notes](MIGRATION_060.md).
Allowed device/dtype overrides validate the original snapshot before changing
execution settings. They do not bypass source compatibility.

Save preprocessing, body state, external `History`, environment RNG and data
position alongside the brain. A reward learner needs `Reinforcement.snapshot()`
so pending action ownership, replay and RNG survive together. Do not execute a
restored pending action twice. A checkpoint preserves continuation; its source
hashes do not authenticate an external observation.

| Observed failure | First investigation |
| --- | --- |
| Solves refuse | Inspect `reason`, stationarity and attempted work; test budget headroom from the unchanged continuation. Report any tolerance change. |
| Solves qualify but training recall fails | Check input sufficiency, output connectivity, target units, useful capacity, anchoring and exposure. |
| Recall passes but new cases fail | Inspect split independence, coverage, overfitting and the states reached by the learned controller. |
| A temporal task fails | Establish whether the supplied observation/history distinguishes the required answers before attributing failure to depth. |
| New learning destroys an old skill | Measure retention and compare explicit replay curricula; do not assume protected memory. |
| More depth raises cost without improving behavior | Retain the simpler qualified model and the unsuccessful comparison; give deeper observation a different, explicitly motivated task. |

Choose the next increase in task complexity from those measurements. Good local
control, temporal information, outcome credit and planning are separate abilities
that need their own acquisition and behavioral checks.
