# Training and choosing a size

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

## Supply an informative, well-scaled teaching stream

`observe(inputs, targets)` learns an actual desired output or measured outcome.
It does not turn an environment reward into action credit. A controller can
learn from demonstrated actions; a dynamics model can learn measured next
observations. State who supplies those witnesses. Calling repeated examples
through `observe` is replayed supervised learning, not fresh environment
experience or reward-only discovery.

1. Include the information needed to distinguish different answers. A position
   alone cannot identify velocity. A numeric-array interface does not provide
   learned vision or recover discarded observations.
2. Scale inputs consistently and use targets comfortably inside both the
   `tanh` prediction range and `state_bound`, such as `-0.6` and `0.6` with
   the default bounds. Fit preprocessing on training data only. Centering
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
the worst seed as well as averages. If training recall fails, stop and diagnose
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
it is a training procedure, not an additional Cadence primitive.

## Diagnose before scaling

The default ceiling is 2,048 accepted repair sweeps per solve; qualification
stops work early. Small nine-patch recursive acquisition tests needed up to
777 sweeps, so a 512-sweep ceiling refused some otherwise successful examples.
This is numerical headroom, not extra training examples or a guarantee that
every layout settles within the budget.

If solves refuse, inspect `stationarity`, `reason` and `work`, then test a larger
budget from the unchanged continuation. The solver adapts a scalar step while
retaining the same energy and qualification check. Changing `tolerance` changes
the admission criterion and must be reported.

If qualified training recall is poor, examine connectivity, input conditioning,
target scale, exposure count, initialization and parameter anchoring. If recall
is good but new cases fail, examine data coverage and generalization. Compare
ordinary composition and recursive observation at declared capacity and work.
Report evaluations/backtracks as well as wall time; more observers can add cost
without improving a task already solved by a small model.
