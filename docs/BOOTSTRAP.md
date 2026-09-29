# Bootstrapping phase and live phase

The **bootstrapping phase** prepares a brain's basic abilities through guided
experience: interpreting sensors, responding to relevant objects, controlling
a body, and retaining those abilities together. The **live phase** uses that
same brain in a continuing environment. Both phases use the same patch rule;
actual new witnesses can keep changing learned relations during the live phase.

Cadence supplies supervised witness learning and an explicit `Reinforcement`
helper for discrete choices, one-step Q credit and replay. The application
provides sensors, measured outcomes or rewards, and the environment used to
check behavior. A declared eye port does not by itself teach vision; a motor
port does not by itself teach walking. These are abilities to bootstrap and
measure, using the same generic interfaces. See [live learning](LIVE.md) for
reward-driven interaction, explicit history and runtime control.

## A complete small bootstrap

`bootstrap` validates examples, replays them in a reproducible shuffled order,
and measures answers without target clamps. It checks both recall and separate
check cases before starting and after each completed epoch. It stops once both
meet your error limit, the epoch allowance runs out, or a solve refuses.

```python
from cadence import Brain, Cortex, bootstrap

cortex = Cortex(seed=2)
signal = cortex.input("signal", shape=1)
base = cortex.column("perception", patches=4, inputs=signal)
observer = cortex.observer("reflection", patches=2, inputs=signal, observes=base)
cortex.output("answer", shape=1, reads=observer)
learner = cortex.build()

examples = [({"signal": [x]}, {"answer": [x]}) for x in (-0.8, 0.8)]
checks = [({"signal": [x]}, {"answer": [x]}) for x in (-0.4, 0.4)]
report = bootstrap(learner, examples, checks=checks, max_error=0.2)
assert report["passed"], report

saved = learner.snapshot()
live = Brain.from_snapshot(saved)
activity = live.step({"signal": [0.3]})
assert activity["accepted"]
print(activity["outputs"]["answer"])

# An actual new witness can still teach the same brain in the live phase.
assert live.observe({"signal": [0.3]}, {"answer": [0.3]})["accepted"]
```

`max_error` is an absolute output error in your encoded units. It is separate
from numerical settlement `tolerance`. A passing report covers the supplied
examples and checks, not arbitrary environments. Checks influence when the
helper stops, so they are development data. Reserve fresh cases for final
evaluation. The report includes per-epoch errors, admitted presentations and
all counted solve work; see the [exact helper contract](REFERENCE.md#bootstrap).

The helper operates on the supplied brain. Each accepted admission persists;
a later refusal does not undo earlier admissions. Invalid examples are caught
before the first admission. It does not normalize data, choose an architecture,
change the repair rule or reinterpret rewards as desired outputs.

## Choose individual or batch admissions

`bootstrap(..., batch_size=1)` uses the usual ordered `observe` calls. To jointly
learn a group of witnessed examples, choose a larger positive `batch_size`.
Each epoch still shuffles the examples with the supplied seed, then groups
that order into batches; the last may be shorter. All recall and readiness
checks remain separate unclamped queries.

A batch minimizes the **mean** example energy plus one shared parameter-anchor
penalty. Each row has its own clamped outputs and private processing/observer
states, initialized from the same pre-call live state. The shared parameters
repair against all those experiences together. On success they are retained
as one atomic admission; the live state stays unchanged. A refused batch
admits none of its examples and stops the helper, leaving earlier successful
admissions intact.

The [quickstart](QUICKSTART.md#learn-several-experiences-together) gives a runnable
batch example. For custom streams or retry identities, call
`brain.observe_batch(examples, event_id=...)` directly. Keep the same ordered
rows and method when retrying an event. Even a one-row `observe_batch` preserves
live state; the helper's default `batch_size=1` deliberately continues to use
`observe`, which retains its solved state.

Count `presentations` and `accepted` as **examples**, and `updates` as accepted
atomic calls. Four accepted rows in one batch mean four accepted examples and
one update. `inspect()["admissions"]` also advances once per batch. A refused
group contributes its full size to attempted presentations and zero accepted
examples; the report identifies all original row indices in that group.

Changing group size changes the number of anchors and updates per epoch;
increasing epochs changes exposure too. A short final batch has its own mean
and anchor, not a share of the preceding batch's objective. Calibrate batch
size, `parameter_prior` and epoch allowance together on fixed development
data, then check unclamped accuracy and older skills. Equal presentation counts
do not imply equal repair work or an identical learning trajectory. Repeated
rows change their frequency in the batch mean; replay remains supplied evidence,
not new environment experience.

Tensor devices can vectorize rows, with temporary memory growing with batch
size. The Python engine supports the same objective. Use bounded batches and
measure full acquisition time, refusals, validation and memory rather than
assuming speedup. Rows do not acquire a temporal relationship by being placed
in one batch; delayed credit and missing history still need their own design.

## Calibrate before increasing the task

Calibrate **units** first: choose one input transform and one reversible output
encoding, fit any statistics on bootstrap examples only, and reuse them in
the live phase. For a measured position within known limits, an encoding such
as `0.6 * (2 * (position - low) / (high - low) - 1)` maps that interval to
`[-0.6, 0.6]`. Check out-of-range measurements explicitly. Preserve these units
and transforms with the brain's checkpoint; the library does not infer them.

Then calibrate **configuration** against a small fixed development set. Start
with defaults, change one setting at a time, and build a fresh brain with the
same seed and examples for each comparison. Record `report["passed"]`, errors
and `report["work"]`; count failed configurations and search work too.
`report["options"]` retains the requested error limit, epoch allowance, seed,
batch size and resolved solve budget. Save the starting checkpoint and original data to
reproduce calibration; save the final checkpoint to continue into the live phase. An
epoch allowance is a work limit, not evidence that an ability was acquired.
`epochs=0` runs the same unclamped checks without admitting any examples.

For example, compare default anchoring with one more conservative setting on
the same small relation. This is a two-candidate calibration, not an automatic
search or a universal prescription:

```python
for prior in (0.1, 1.0):
    candidate = Cortex(seed=2, parameter_prior=prior)
    sensor = candidate.input("signal", shape=1)
    response = candidate.column("response", patches=1, inputs=sensor)
    candidate.output("answer", shape=1, reads=response)
    trial = bootstrap(candidate.build(), examples, checks=checks, max_error=0.2)
    print(prior, trial["passed"], trial["history"][-1], trial["work"])
```

| Ability to bootstrap | Guided experience to supply | Check before using it live |
| --- | --- | --- |
| Sensor interpretation | Actual sensor arrays paired with measured properties or desired responses | New positions, lighting or noise; perturb the relevant sensor and measure the output change |
| Selection of relevant objects | Scenes with distractors and witnessed selection/response targets | Move the relevant object, change distractors and test which information changes the decision |
| Body control | Proprioception, recent motion and demonstrated actuator commands or measured next states | Closed-loop starts, disturbances, speed, contacts and actuator limits in the simulator |
| Several retained abilities | Interleave earlier examples with the new ability's examples | Recheck every earlier ability after each stage |

These are curriculum templates, not supplied pretrained vision, attention or
walking systems. The application supplies the body, observations and witnesses.
Begin with tiny tasks and unlock larger ones only when these checks hold. For
classification or sequential control, also check the actual decoded decision
or complete behavior; a numerical paired-example error is only one check.

## Choose a size

Start with the smallest connected model that could express your target. Add
width or observation depth when a controlled comparison improves **unclamped
task performance** enough to justify its work. Settlement qualification and
task accuracy measure different things.

## Practical starting sizes

| Use case | First layout to try | Interpretation |
| --- | --- | --- |
| Scalar regression or simple control from a few measured values | One output patch reading all inputs | A nonlinear scalar relation; often enough for a simple control decision. |
| Several independent output relations | One patch per output; start with 2–8 | Extra unused flat patches do not create hidden capacity. |
| A small relation requiring learned nonlinear combinations | 4–16 representation patches feeding a separate 1–4-patch output population | Start at the lower end. `column(inputs=base)` gives ordinary composition. |
| Test whether observation of that representation helps | Replace the output population with `observer(observes=base)`; start with 1–4 output patches | A matched ordinary-composition control is essential. All states still settle together. |
| Small parallel sensor branches | 4–16 patches per branch, then 4–8 fusion/observer patches | An initial search range, not a demonstrated multimodal solution. |

These are search starting points. Small experiments support one-patch control,
five-patch nonlinear acquisition and eight-patch conditioned classification.
They do not establish a recommended size for raw-image learning, general
planning or full board-game mastery. Two or three observation levels are
candidate layouts to test, not prerequisites for intelligence.

For a connected nonlinear candidate:

```python
from cadence import Cortex

layout = Cortex(seed=7, initial_scale=1.5, parameter_prior=10.0)
cue = layout.input("cue", shape=2)
representation = layout.column("representation", patches=4, inputs=cue)
decision = layout.observer("decision", patches=1, observes=representation)
layout.output("choice", shape=1, reads=decision)
brain = layout.build()
assert brain.inspect()["patches"] == 5
# Every decision coordinate must have a path to both cue coordinates here.
assert brain.inspect()["outputs"][0]["sensor_coverage_by_coordinate"] == (2,)
assert brain.inspect()["output_connected_patches"] == 5
```

The two nondefault values above are a tested candidate for small nonlinear
relations. A larger `initial_scale` can give hidden `tanh` relations more
curvature; it also raises numerical work on some tasks. A larger
`parameter_prior` makes each experience change retained relations more
conservatively. It can improve retention across conflicting examples and
require more exposures. Neither value is a universal improvement. Begin with
`Cortex(seed=...)` for simple relations; compare changes one at a time.
The five-patch confirmation used four distinct examples replayed for 4,096
updates per model, then tested nearby unseen inputs across three initializations.
Both ordinary and recursive layouts succeeded; it did not show a depth advantage.

## The parameter prior governs supervised extraction

On supervised mappings with many binary outputs and rare positive targets,
`parameter_prior` decides how much of the linearly available structure the
settled readout recovers. In the
[supervised extraction study](https://github.com/muellerberndt/cadence/issues/70),
a one-column network trained on eight linear rules at a 5% positive rate
reached mean test AUC 0.733 at prior 0.005, 0.922 at the default 0.1, and
0.977 at 0.4, against 0.951 for closed-form ridge regression on identical
rows. Single-witness admission at the default prior reached 0.796, so
batched admission helps extraction on such tasks. Widening the column from
8 to 24 patches left every per-epoch AUC unchanged to three decimals and
eight epochs oscillated without a trend: the settled readout is a fixed
point that capacity and exposure do not move, and the prior places it.
Before judging a layout on a supervised task, sweep `parameter_prior`
upward and compare the settled readout's ranking against a ridge baseline
on the same rows; the linear baseline is the floor an extraction run
should clear.

## Read outputs from enough of the network

Where an output reads from bounds what the mapping can use. An output read
only from one small late observer can extract less task structure than the
declared inputs support, while a direct read from the sensing population, or
a read from a wide observer over every processing stage, recovers it. When a
supervised readout underperforms its linear baseline, widen what the output
reads before adding capacity anywhere else; capacity behind a narrow read
does not reach the output.

## A qualified starting recipe for small coordinate relations

For immediate scalar regression, two-output linear mixing and a single `tanh`
relation, start with one directly sensing processing patch per output:

```python
cortex = Cortex(seed=0)
signal = cortex.input("signal", shape=1)
response = cortex.column("response", patches=1, inputs=signal)
cortex.output("answer", shape=1, reads=response)
brain = cortex.build()
teaching = [({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.8, 0.8, -0.4, 0.4)]
checks = [({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6, -0.2, 0.2)]
report = bootstrap(brain, teaching, checks=checks, max_error=0.1, epochs=20, seed=0)
assert report["passed"], report
```

Keep inputs and outputs in these normalized units. The selected settings use
the unchanged defaults: `parameter_prior=0.1`, `state_prior=0.01`,
`settle_budget=2048`, `tolerance=1e-6`, and bounds `1` for state and `4` for
parameters. Width and data change with the declared relation; no source edits
are needed. Save the units and data split alongside `brain.snapshot()`.

The [bounded defaults study](https://github.com/muellerberndt/cadence/issues/66)
screened nine recipes over two development seeds per family, then froze settings
before 30 fresh seed blocks. All 90 family/seed cases passed acquisition,
stable replay, witnessed sign reversal, explicitly taught restoration and
checkpoint continuation. Independent assessment errors stayed below `0.12`
in original units; all cases met the declared two-second limit.

This measures initialization/replay sensitivity on three fixed tasks. The 30
shared seed blocks give a one-sided 95% reference lower reliability bound of
90.5%, not 99% or a guarantee for new tasks. Restoration is supervised relearning,
not protected long-term memory. Recursive layouts had development checks only.
Aliased inputs and oversized targets failed readiness; multiplying input scale
by 100 also caused failures. Inspect units, coverage and unclamped errors before
increasing depth or replay. Delayed reward credit and raw perception are outside
this recipe's support envelope.

## Supply an informative, well-scaled teaching stream

`observe(inputs, targets)` learns an actual desired output or measured outcome.
It does not turn an environment reward into action credit. A controller can
learn from demonstrated actions; a dynamics model can learn measured next
observations. State who supplies those witnesses. Calling repeated examples
through `observe` is replayed supervised learning, not fresh environment
experience or reward-only discovery. For discrete reward-driven choices, use
`Reinforcement`: it records actual transitions and separately derives Q targets,
fitted with `source="estimate"`. Do not label a predicted return as an observed
fact. Valid feedback consumes its pending action even when numerical fitting
refuses; use `replay()` to retry fitting that retained experience.

1. Include the information needed to distinguish different answers. A position
   alone cannot identify velocity, and a demonstrator's action that depends on
   hidden state, such as an internal counter, cannot be agreed with from the
   supplied observations at all; measure a linear baseline on the same rows to
   know what the stream supports before judging the learner. A numeric-array interface does not provide
   learned vision or recover discarded observations. `History` can preserve a
   bounded explicit sample window with validity masks; it is an input encoding,
   not learned recurrent memory or automatic object tracking.
2. Scale inputs consistently and use targets comfortably inside both the
   `tanh` prediction range and `state_bound`, such as `-0.6` and `0.6` with
   the default bounds. Fit preprocessing on bootstrap examples only. Centering
   or standardization can help; strongly correlated inputs may require an
   explicitly fitted whitening transform. Whitening is external preprocessing,
   not a representation learned by Cadence.
3. Shuffle representative examples and check retention of earlier ones.
   Record distinct examples separately from replayed updates. Start with
   hundreds of examples/updates for a small relation, then inspect learning
   curves instead of assuming more depth will replace sufficient exposure.
4. Evaluate with targets absent, on data and environment starts reserved before
   tuning. Check `qualified`/`accepted`; refusing an action is different from
   making a wrong one. A refused admission changes no state or event identity.
   Track the qualification rate at the deployment budget across training:
   settling cost can grow as content accumulates, and a fixed budget that was
   generous early can start refusing held-out queries later.

The default `fan_in=None` gives each patch every coordinate of each declared
source. Opting into sparse `fan_in` changes the information available to each
patch. `sensor_coverage` means connected **somewhere**.
Each inspected output's `sensor_coverage_by_coordinate` counts the sensor
coordinates connected to each selected state through the jointly coupled graph.
`output_connected_patches` counts patches connected to at least one output;
extra disconnected width cannot supply a hidden representation to those outputs.
These are structural
checks, not measured causal effects or accuracy. Zero weights and saturation
can still suppress influence. Parallel branches may deliberately use different
sensors. See [layout variants](VARIANTS.md).

Check the expressive capacity of supplied encodings as well as their size.
For example, multiplying a centered category vector by context basis functions
creates interaction features whose category mean is zero in every context.
A flat patch with one bias then cannot express independently varying mean
**drives** for those contexts. Include the public context basis itself when
that variation is needed, or test a connected nonlinear representation. The
cases can have distinct input vectors and still exceed a flat patch's capacity.
For a small finite task, inspect rank or fit the required pre-activation drives
before attributing underfitting to settlement or recursive depth.

Increasing `state_bound` permits larger clamps; it does **not** rescale `tanh`.
For an output patch that no other patch reads or observes, unconstrained free
settlement gives `x = tanh(drive) / (1 + state_prior)`, so its magnitude stays
below `1 / (1 + state_prior)` even when the bound is larger. A target of `2`
with `state_bound=4` can qualify during teaching and still be impossible to
recall. Encode regression targets into a suitable range and decode predictions
back into application units. Bounds are not a substitute for that transform.
For classification, expose one score per class and decode the largest score;
these values are not normalized probabilities. Integer class IDs are labels,
not suitable scalar regression targets by default. For multiple named controls
from one population, select different `indices` explicitly: scalar outputs both
default to patch zero when no indices are supplied.

Before a long run, teach a small representative set, then query **every example
without targets** after replay. Compare against the untrained model, check that
changing relevant inputs changes answers, and save/restore the result. Track
the worst seed as well as averages. If bootstrap recall fails, stop and diagnose
that failure before spending work on a larger task. The core's small learning
tests exercise this sequence for flat, composed and observing populations;
they do not establish that every task is learnable with the defaults.

## Test control in the environment

Even 99% action agreement on teaching states can produce a poor controller:
one wrong action changes the states encountered next. Measure complete episodes
with the teacher disconnected. If the model visits unfamiliar states, collect
teaching witnesses there, combine them with earlier examples, and compare
against an equal-update replay-only control. This is an explicit supervised
curriculum; disclose it instead of claiming the network discovered a policy
from reward alone. A weak teacher also limits what imitation can achieve.
This follows the dataset-aggregation approach studied by
[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html);
it is a bootstrapping procedure, not an additional Cadence primitive.

Report rare but important decisions separately: if almost every demonstration
says to keep moving straight, high aggregate agreement can hide failure to
acquire any evasive maneuver. Balanced replay is a supplied curriculum choice;
count repeated presentations and recheck ordinary behavior after reweighting.

## Diagnose before scaling

The default ceiling is 2,048 accepted repair sweeps per solve; qualification
stops work early. Small nine-patch recursive acquisition tests needed up to
777 sweeps, so a 512-sweep ceiling refused some otherwise successful examples.
This is numerical headroom, not extra bootstrap examples or a guarantee that
every layout settles within the budget.

If solves refuse, inspect `stationarity`, `reason` and `work`, then test a larger
budget from the unchanged continuation. The solver adapts a scalar step while
retaining the same energy and qualification check. Changing `tolerance` changes
the admission criterion and must be reported.

If qualified bootstrap recall is poor, examine connectivity, input conditioning,
target scale, exposure count, initialization and parameter anchoring. If recall
is good but new cases fail, examine data coverage and generalization. Compare
ordinary composition and recursive observation at declared capacity and work.
Report evaluations/backtracks as well as wall time; more observers can add cost
without improving a task already solved by a small model.

## Accelerate simulated experience

Use `Cortex(device="mps")` or `Cortex(device="cuda")` with the optional
`cadence-net[gpu]` installation to run repair proposals on a GPU. Use
`device="cpu"` for tensor execution on CPU. `bootstrap`, `observe_batch`, `observe` and live
queries keep their existing interfaces and witness rules. Admission always
uses the original float64 objective and requested tolerance, including when
tensor proposals use float32. See [the acceleration guide](ACCELERATION.md).

Environment collection and independent brains can run in parallel. The runnable
[parallel bootstrap example](../examples/parallel_bootstrap.py) owns one small
simulated body and one brain per process. Each brain predicts fresh outcomes
before admitting the measured outcomes during its live phase:

```sh
python examples/parallel_bootstrap.py --lives 4 --workers 2
```

A single brain's admission calls remain ordered: each individual witness or
joint batch is anchored to the parameters retained by the preceding call.
Within a batch, examples settle in parallel with shared parameters. Collecting
data concurrently does not justify concurrent mutation or averaging independently
learned brains. For one brain, queue witnesses or batches with explicit event
identities and admit those calls serially. Count simulator steps, distinct witnesses, replayed
presentations, refused solves and total wall time separately. GPU execution
and faster simulation do not provide missing information or guarantee that
added observers improve acquisition.

`bootstrap` validates and copies the entire supplied dataset before starting;
`batch_size` bounds the joint solve, not that dataset allocation. For recordings
too large to hold in memory, load bounded lists of examples and pass each list
to `observe_batch`. Keep a separate held-out evaluation stream and checkpoint
your data position alongside the brain snapshot.
