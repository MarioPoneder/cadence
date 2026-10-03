# Choose, train and run a brain

Start from [one continuing equilibrium brain](../world-model.md). Design for
bootstrap, ordinary use, witnessed disruption and local repair in the same
acquired model; retain its learned relations and declared memory. An independent
classifier or external answer network is a control, not this application.
This guide owns the experimental population model's joint stationary energy
solve. Its explicit history and parameter admission must not be confused with
`Brain.compose`'s memory pathways and neural contrast rule.

A brain is one connected graph whose patches repair disagreement together.
Start with a sensing population read by an output population, then choose sizes
and connections from the information the task needs. More patches add capacity;
an intermediate population changes the paths and constraints in the graph.
Measure free behavior and complete learning/query cost before scaling.

Every population uses the same patch rule. `inputs=` supplies sensors or live
states, and optional `observes=` adds exact current error readback in the same
solve. The [experimental boundary](EXPERIMENTAL.md) explains the intended
routine/correction roles and which capabilities remain unproved.

For a first program, use [the quickstart](QUICKSTART.md). For exact signatures,
use [the reference](REFERENCE.md); for saved continuation, see
[checkpoints](REFERENCE.md#checkpoints). The [agent recipe](AGENTS.md) gives a short
application workflow.

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

<a id="coupled-deeper-and-recursive-layouts"></a>

## Choose sizes and connections

Every patch predicts `p = tanh(bias + weighted incoming signals)` and has current
error `state - p`. `column(..., inputs=earlier_population)` reads live states;
`observer(..., observes=earlier_population)` reads both states and exact errors.
An output exposes selected patch states. There is no separate output network.

A two-population graph is a useful starting point. Increase a population's width
when it needs more represented coordinates; add an intermediate population when
you want a learned composition; use branches and a shared reader when sensors
need separate representations. These choices change actual connectivity, not
just a category called "flat" or "deep". Every population must belong to the
same connected graph.

State contacts return influence during repair: a reader's error contributes to
repair of the states it reads. Error readback adds derivative paths through the
current errors. All states remain eligible together, with one final qualification.
Choose connections by measured task benefit and cost.

### Experimental observer wiring

Choose an observer's inputs deliberately and compare it with a capable
state-coupled control. During teaching, a clamped motor
state records the action that actually happened. A later relation reading only
that fixed state cannot send its error into the motor's own parameters through
that connection. Reading the motor's error adds a parameter-learning path,
because that error depends on the motor's prediction. This can change what is
learned; it does not guarantee a better action or assign reward credit. Ordinary
coupling already affects free states, so an output change alone does not
demonstrate a benefit from error readback. For an input-only observed patch at
fixed parameters, its error is just its state minus a fixed sensory prediction.
An equally informed ordinary control may represent that same feature. A useful
recursive comparison must establish a behavioral contribution, not merely the
presence of an error edge. The [experimental recipes](VARIANTS.md#experimental-recursive-observer-settlement)
show how to wire this explicitly.

### Capacity and connectivity

Width counts processing states. Observation depth means an observer reads
another observer. Neither a population called `reflection` nor extra settling
sweeps demonstrates reasoning. The public builder accepts previously declared
sources only: it supports jointly coupled state/error constraints, but does not
expose explicit recurrent state cycles. Retaining activity with `step` alone is
not evidence of useful temporal memory.

The [ordinary and experimental layout quickstarts](VARIANTS.md) and
[shared executable example](../../examples/equilibrium/layout_learning.py) teach, query and
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
history length or population size. All compiled patches must remain connected;
sparse wiring that produces disconnected components is rejected. See [layout variants](VARIANTS.md).

## Make temporal information explicit

When the latest sample is ambiguous, first establish a sufficient-information
control using a bounded `History` window. Its presence masks distinguish initial
padding from observed zeros. This is explicit application memory, and its size
must count in a comparison.

```python
from cadence.experimental.equilibrium import Brain, Cortex, History, bootstrap

history = History(1, steps=3)
history.push([0.2])
encoded = history.push([0.4])
assert len(encoded) == history.size == 6

history_layout = Cortex(seed=2)
recent = history_layout.input("recent", shape=history.shape)
features = history_layout.column("features", patches=4, inputs=recent)
response = history_layout.column("response", patches=1, inputs=features)
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

Begin with defaults and the smallest brain: a few sensing patches read by one
output patch per simple output. For a small nonlinear relation, a connected
representation of 4–16 patches feeding 1–4 output patches is a starting search
range, not a capability guarantee.
Use [bootstrapping and size](BOOTSTRAP.md) for more detailed recipes.
Keep the simpler ordinary network when added depth does not improve measured
quality enough to justify its complete query and learning costs.

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
features = layout.column("features", patches=4, inputs=signal)
response = layout.column("response", patches=1, inputs=features)
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

## Keep numerical error, forecast surprise and task value separate

| Measurement | What it tells you | What it cannot establish |
| --- | --- | --- |
| `stationarity` and `qualified` | Every eligible coordinate in the whole graph meets the projected repair tolerance under this call's clamps | Correct predictions, good actions, a unique answer or a global energy minimum |
| `prediction_residual` and patch `errors` | Current disagreement between internal state and the current local prediction | Whether an earlier forecast was contradicted by a later real observation |
| Historical forecast surprise | Difference between an immutable issued forecast and its later actual observation | Whether that outcome is good or bad for the task |
| Reward or goal deficit | Observed value or predicted shortfall in declared task units | Automatic credit assignment to every population |

Keep the original forecast, observation context, action identity and model
identity before its outcome arrives. Compare that fixed forecast with the later
measurement. Recomputing a prediction after learning can erase the very error
you wanted to measure. `LearningProgress` summarizes changes in supplied
predictor errors; it is not an automatic surprise detector or information-gain
measure. Even with unchanged parameters, a patch's local prediction and its
settled output can differ because of priors and coupling. Declare which value
your application issues as its forecast and preserve that exact value.

A predictable failure can have zero surprise and still require correction.
An unexpected good result can have large surprise without negative value.
Neither signal should silently stand in for the other.

Targets also change the settling problem. An observer's error inputs during
joint teaching can differ from the signals available when the future answer is
free. Evaluate with that answer unclamped. When diagnosing an apparent readback
benefit, compare those two signal conditions with identical parameters and
causally available observations. A low teaching loss is not sufficient evidence
of a useful free prediction. Observers also read states, so zero error does not
make their input or influence vanish; it is not a safe automatic sleep signal.

## Keep actions and outcomes attached to the right life

Choose operations by what should persist:

| Operation | Retained activity | Retained parameters and event |
| --- | --- | --- |
| `settle` / `predict` | Unchanged | Unchanged |
| Qualified `step` | Complete solved activity | Unchanged |
| New accepted `observe` | Complete solved activity | Parameters and one admission |
| New accepted `observe_batch` | Unchanged | Shared parameters and one admission |

These Brain operations retain none of a refused proposal. Check `qualified`
before using query outputs and
`accepted` before treating an update as committed. `predict` raises
`SettlementError` on numerical refusal. Diagnostic outputs from a refused call
are not actions.

`step` and `settle` accept the same optional `targets` and `interventions`.
These clamps condition a frozen-parameter solve; they are not teaching
admissions. `step` retains the resulting activity, but the constraints expire
at the end of that call. Supply any continuing constraint again on the next
call. A measured past outcome can condition a current decision. A desired
future output is an intention; leave it free in a separate query to obtain a
forecast. Even an all-clamped, immediately qualified call proves no prediction
or acquired skill.

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

Recording an executed action establishes what happened, not whether the action
was useful. Repeatedly teaching a motor output to copy its own choices can erase
an acquired routine. Use measured consequences to teach a body predictor and a
declared outcome-credit procedure to teach preferences; test routine retention
throughout live learning.

<a id="spend-compute-according-to-measured-need"></a>

## Measure speed and retained correction

Current calls solve synchronously. Every participating population remains in
the energy and every eligible coordinate remains in final qualification.
Naming a branch `fast` or `reflection` does not schedule it independently.
An explicit observer can therefore add latency to every solve, even when the
routine is familiar and its predictions are good.
`LiveController` keeps a caller responsive while a serial callback owns the
brain; it cannot certify a fresh action while an unresolved part of that
same state is silently omitted. Parallelize independent lives or collection,
and serialize access to one brain's continuation.

The intended application boundary is one brain receiving observations,
issuing qualified actions and acknowledging actual outcomes. Users should not
need a second brain, a wake flag or an evaluator per population. A future
internal scheduler must define dependency invalidation and qualification
before claiming independent population speeds. Concurrent learning proposals,
coordinate scheduling and arithmetic reuse are distinct mechanisms.

The current reference query cache reuses predictions depending only on fixed
sensory inputs, including bias-only predictions, within one solve. Errors and
returning derivatives remain current, and final qualification is fresh. The
cache does not sleep an observer or carry learned attention between calls.
See [performance](PERFORMANCE.md).

Measure queries, candidate actions, teaching, replay, checks and refusals.
Sum primitive `work` counters or use a helper's aggregate, without counting
both. Report task quality, qualification rate, latency and wall time alongside
`evaluations`, `edge_visits`, `patch_visits`, `proposals`, `backtracks` and
sweeps. A sweep budget is not an elapsed-time deadline; zero sweeps still
require evaluation.

For a correction test, first establish competent autonomous routine behavior.
Disturb the actual body, measure recovery, then freeze learning and test
retained quality and subsequent work. Include a routine-plus-factual-fit
control: improvement after fitting does not by itself show that planning or
observation was necessary. Count successful correction fits as well as
failed attempts; a controller that keeps requesting correction but never
consolidates it has not demonstrated a return to inexpensive routine.

Start with `device="python"` for small graphs. Optional `device="cpu"` uses
PyTorch on CPU; `"cuda"` and `"mps"` select supported GPU execution. Device
proposals still need float64 reference qualification and may need refinement.
Include transfer, setup and reference work in comparisons. Small brains can
lose time to device overhead; see [acceleration](ACCELERATION.md).

## Preserve and diagnose continuation

```python
saved = brain.snapshot()
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"signal": [0.5]}) == brain.predict({"signal": [0.5]})
```

Brain snapshots bind exact implementation sources, layout and configuration.
An older snapshot is not automatically compatible with a newer release; retain
the originating installation and follow the [checkpoint contract](REFERENCE.md#checkpoints).
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
