# Task recipes

Start with the operation your experience stream requires. These recipes describe
interfaces; observation encoding, feedback timing and held-out lifetimes belong
to your application.

| Task | Input | Operation and output | Guide |
|---|---|---|---|
| Known interacting constraints | A drive and a declared connectome | Settle; read output activations and residual | [Neural dynamics](concepts.md) |
| Learn consequences and reward | Reading and observed outcome | Code the reading, read the records, write the outcome into the active cells | [Records](memory.md#records) |
| Complete a partial reading | Drive with missing flags | Settle the regions fed by the record reads; repair from the observed outcome | [Prediction repair](learning.md#4-one-observation-repairs-a-settled-prediction) |
| Revise an addressed association per stream | Key and observed value | `FastSynapses.observe`, then `recall` | [Memory](memory.md#reading-and-writing-directly) |
| Imitation | An observation and a teacher's action | Nudge toward the current demonstrated action; retain legal-action constraints | [Learning recipes](learning.md), [learning life](experience.md) |
| Regression or reconstruction | Features and an output pattern | Quadratic nudge; read continuous output activations | [Pattern targets below](#pattern-targets) |
| A continuing stream | Each observation before its label arrives | Predict, score, then update; retain history explicitly when needed | [Memory](memory.md), [ongoing loop](continuous.md) |
| Reward-driven action | Observation, chosen action, reward | Weight action-target nudges by advantage, or use eligibility traces | [Feedback](continuous.md), [reward](reward.md) |
| A measured synapse list | Supplied topology and declared stimuli | Settle and score held-out predicates with controls | [Protocols](protocols.md) |

## Put features on input neurons

A batched drive has one column per neuron, including hidden and output neurons.
Here is a small prediction head with four input values:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(4, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()), connectome.populations["output"]
)
x = np.array([[1.0, 0.0, 0.0, 1.0]])  # current room and proposed action
levels = np.zeros((len(x), learner.brain.connectome.n))
levels[:, list(learner.brain.connectome.populations["input"])] = x
drive = learner.brain.stimulus_levels(levels)
```

`stimulus_levels` multiplies by the neuron model's `stimulus_amplitude`; it does not pad
or clip the input. Fit feature scaling, category dictionaries, and missing-value
handling on training data. For a scalar with a wide range, a few localized bumps
over training-fitted bin centers are one possible representation. Compare that
encoding against simpler scaling using validation data.

## Pattern targets

`step` accepts integer class indices. For continuous or multi-label patterns,
construct targets with one column per neuron and use the phase operations directly:

```python
import numpy as np
import cadence as cd

connectome = cd.layered(2, 8, 2, density=1.0, seed=0)
learner = cd.Learner(
    cd.Brain(connectome, cd.learning_neuron_model()),
    connectome.populations["output"],
    cd.LearnerConfig(nudge="quadratic"),
)
drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["input"])] = np.eye(2)
target = np.zeros_like(drive)
target[:, learner.output_index] = np.array([[0.2, 0.8], [0.8, 0.2]])

free = learner.free(drive)
plus = learner.nudged(drive, free, target)
minus = learner.nudged(drive, free, target, sign=-1.0)
learner.update(free, plus, minus)
prediction = learner.free(drive).activation[:, learner.output_index]
print(prediction.shape)  # (2, 2)
```

This performs one update; it does not establish fit quality. Choose target scaling
within the activation's useful range using training data. Select thresholds for
multi-label outputs or calibrate continuous readouts on validation data, then
hold those choices fixed on test data. `predict` returns the most active outcome index; read
activations directly for these pattern tasks.

## Streams and independent episodes

Score a prediction before revealing its label. Keep the order of observations,
actions, and rewards explicit. `Learner.free(..., warm=state)` carries transient
state; `Trace` carries fading activity; `FastSynapses` carries observations.
These have different memory contracts. Settling to a fixed point alone does not
guarantee recall of earlier inputs.

Rows of a batch represent independent streams. If a stream ends, clear its
episodic state. For `FastSynapses` and `Trace`, `reset(batch, rows=...)` resets
selected rows and `keep(rows)` retains a subset without changing identities.
Updating a different batch size initializes fresh fast state. Reads of a different
batch size use the empty-stream baseline without changing live records. Use the
explicit stream operations when changing which stream identities are present.

## Several learners in one net

Use separate `Learner` objects on one brain with `plastic_synapses` and
`plastic_neurons` masks to restrict updates and decay. Each learner owns its
optimizer history; manually swapping just its momentum arrays is insufficient
when normalization or step counters also matter.

An update replaces `learner.brain` with a `Brain` carrying the updated parameters.
If two learners should operate on one evolving parameter set, assign the updated
brain to the other learner's `brain` before its next phase, while preserving that
learner's masks and history. Check checkpoint and reset behaviour for the complete
composition. [Write a cortex](cortex.md#which-synapses-a-head-owns) builds the masks from
port pairs.
