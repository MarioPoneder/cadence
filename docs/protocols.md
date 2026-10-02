# Protocols: held-out tests with preconditions

A protocol declares the questions a settled brain is asked, and what it may be
shown before it is asked. It can score a supplied graph model. The examples follow the
same discipline in their stage verifiers: held-out lifetimes apart from the learning lives,
every scheduled seed, listed controls, and metrics recomputed from the event logs.

## Stimuli, rows, training facts

This example builds a brain whose connectome has `sensors` and `motors` populations inside a
connectome of 100 neurons:

```python
import cadence as cd
from cadence.protocol import Levels

connectome = cd.Connectome.from_synapses(
    100, pre=[0, 1], post=[2, 3], count=[200, 200],
    populations={"sensors": [0, 1], "motors": [2, 3]},     # the other 96 neurons stay at rest
)
neuron_model = cd.NeuronModel(gain=0.02)
brain = cd.Brain(connectome, neuron_model)

protocol = cd.Protocol(
    stimuli={"rest": (), "touch": ("sensors",)},              # name -> populations stimulated at full amplitude
    training=[("touch", "motors", "active")],                  # what the model may be shown
    rows=[
        cd.Row("R1", "rest", "motors", "inactive", "no input, no output"),
        cd.Row("R2", "touch", "motors", "reduced", "cutting the sensors", ablate=("sensors",)),
    ],
    levels=Levels(active=0.5, inactive=0.2, margin=0.15),
    steps=60,
)
report = protocol.score(brain)        # {"rows": [...], "passed": k, "total": m, ...}
```

A `Row` names a stimulus, a readout population, a predicate, a citation-style reference,
and optionally populations to ablate (neurons zeroed for that row) and a second readout
to compare against (`relative_to`). For `exceeds` and `lateralized`, a matching readout
name takes priority over a stimulus name; other predicates prefer a matching
stimulus. An unknown nonempty name raises `KeyError`. `tier` groups results.

## Predicates and their preconditions

| predicate | passes when | precondition |
|---|---|---|
| `active` | mean(readout) ≥ `active` | |
| `inactive` | mean(readout) ≤ `inactive` | |
| `reduced` | mean(readout) ≤ mean(reference) − `margin` | the reference (intact) brain was active |
| `retained` | mean(readout) ≥ `active` | the reference was active |
| `released` | mean(readout) ≥ mean(reference) + `margin` | the reference was inactive |
| `exceeds` | mean(readout) ≥ `active` and ≥ mean(other) + `margin` | |
| `lateralized` | absolute mean difference ≥ `margin` | at least one mean ≥ `inactive` |
| `sparse` | `sparse_min` ≤ fraction active ≤ `sparse_max` | |
| `densified` | fraction(readout) ≥ fraction(reference) + `densify_margin` | the reference was sparse |
| `specific` | the readout's codes under the row's stimulus and under `versus` share at most `specific_max` of their union (a code: the members at or above `code_level`) | both codes have a member |

A `specific` row names its second stimulus: `Row("fruit and yeast are told apart", "fruit", "kc",
"specific", versus="yeast")`, and a training fact does the same with a fourth element,
`("fruit", "kc", "specific", "yeast")`. It is the fact to select a population's gain on before
learning at that population (the fly's local neurons; [brains from a connectome](connectomes.md)):
a code that is the same for every stimulus carries nothing a lesson can attach to.

A precondition that fails makes the row fail. That is what stops a silent brain from
passing `reduced` or a saturated one from passing `released`. `evaluate_predicate(predicate,
value, reference, levels)` is the pure function, and a receipt's verifier reruns it on the
stored readings.

## The shuffled control

`cd.shuffled(connectome, seed)` permutes the postsynaptic endpoints of every synapse and keeps
everything else: every count, every sign, every neuron's out-degree, every named population,
and no autapses. Score the protocol on it with the same neuron model and gain. A difference
is evidence that the particular connections matter under the declared model and
protocol. Use several shuffle seeds; one shuffled failure does not isolate a
biological explanation or every possible graph confound.

## Gain selection under a sparsity cap

```python
gain, table = cd.select_gain(lambda g: cd.Brain(connectome, neuron_model.replace(gain=g)), protocol,
                             grid=(0.01, 0.02, 0.03, 0.05), sparsity_cap=0.05)
```

Every gain on the grid is tried. A gain is admissible when no more than the
cap's fraction of neurons is active under any training stimulus. Selection
maximizes training facts passed among admissible gains, breaking ties by smallest
gain. It can return a gain that fails some or all training facts, so inspect the
table's `facts_passed` as well as `admissible`.

An empty grid or no admissible candidate raises `ValueError`. Declare a wider
grid or a different cap before evaluation instead of silently using a rejected
candidate. Pass `sparsity_cap=None` for toy brains deliberately allowed to become
fully active. Include the complete selection table in the receipt.

## For learned brains

The examples do not use `Protocol`. Their run receipts (`cadence-experience-run/v1`, see
[receipts](receipts.md)) carry every acceptance predicate with its value, threshold,
aggregation and pass flag, and each stage's `verify.py` recomputes the metrics from the
event logs and fails closed on an incomplete run. The same habits apply to any learned
brain: selection on a validation split of the training data only, the test set read once
after selection, and a control that should fail (a brain trained on shuffled labels
scoring at chance).
