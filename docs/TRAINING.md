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
2. Scale inputs consistently and use targets comfortably inside `state_bound`,
   such as `-0.6` and `0.6`. Fit preprocessing on training data only. Centering
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
patch. `sensor_coverage` means connected **somewhere**, not necessarily usable
by your chosen output. See [layout variants](VARIANTS.md).

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
