# Write a cortex

A cortex is a named group of neurons with declared projections to other cortices. In a
`Genome` it is a `Region`; `develop` lays every region into one connectome, where each
becomes a named population, a port. A learning head owns the synapses between ports and
changes them from settled phases. Three parts of a brain keep their state outside the
connectome: the records cortex (`Records`), the basal ganglia (`ActorCritic`) and the
hippocampus (`FastSynapses`, `SynapticMemory`). The records cortex has cells of its own, a
fixed expansion of a reading, and learns with one delta-rule write per witnessed outcome.

## Blank and designed regions

`Region(name, size)` is a blank region: a group of neurons with no synapses of its own. The
genome's projections connect it, and `mutate` may change its size.
`Region(name, circuit=connectome, inputs=None, outputs=None)` is a designed region: it
carries its own synapses and named populations, takes its size from the circuit and keeps
that size under mutation. `inputs` names the circuit population that receives projections
and `outputs` the population that sends them; each defaults to the whole region. A region
name is nonempty and holds no `/`.

```python
import numpy as np
import cadence as cd

ring = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3, 4, 5], post=[1, 2, 3, 4, 5, 0], sign=[0.5] * 6,
    populations={"cue": [0], "readout": [3]}, label="ring",
)
loop = cd.Region("loop", circuit=ring, inputs="cue", outputs="readout")
print(loop.designed, loop.size, loop.neurons("readout"))   # True 6 (3,)
```

`neurons(population=None)` returns region-local indices, of one circuit population or of
every neuron.

## The catalogue

`cadence.regions` builds the standard cortices, engineering analogues named for their
intended functions.

| Builder | Region | Populations and synapses |
|---|---|---|
| `visual_cortex(height, width, *, channels=1, features=8, field=3, stride=1, init=1.0, seed=0, name="visual")` | designed | `input`: one neuron per pixel and channel, in the order of a `(height, width, channels)` image flattened row by row. `output`: `features` maps; each neuron reads one `field` by `field` window of the input, the windows stepping by `stride`. Efficacies start with random signs and fan-scaled magnitudes times `init`. |
| `cortex(size, *, lateral=0.0, name="association")` | blank; designed for a nonzero `lateral` | With a nonzero `lateral`, every ordered pair of its neurons carries a synapse of that weight; a negative weight makes the neurons compete. |
| `motor_cortex(actions, *, lateral=0.0, name="motor")` | designed | `actions`: one neuron per action, with a synapse of weight `lateral` between every ordered pair when `lateral` is nonzero. |
| `prefrontal_cortex(holds, *, name="prefrontal")` | blank | One neuron per neuron of `holds` (a `Region` or a size), for a `Trace` of the held region that drives it as a stimulus. |

```python
from cadence.regions import cortex, motor_cortex, prefrontal_cortex, visual_cortex

retina = visual_cortex(8, 8, features=4)          # 64 pixels, then 4 maps of 6 by 6
association = cortex(24)
motor = motor_cortex(3, lateral=-0.5)
held = prefrontal_cortex(association)
print(retina.size, retina.inputs, retina.outputs)                  # 208 input output
print(association.designed, motor.designed, held.name, held.size)  # False True prefrontal 24
```

## Projections

`Projection(pre, post, density=1.0, sign=0.0, scale=1.0, count=1.0, reciprocal=True)`
declares synapses from the neurons of `pre` to the neurons of `post`. An end names a
region, meaning its `outputs` population when it sends and its `inputs` population when it
receives (every neuron of a blank region), or one population as `region/population`.

- `density`: the probability that a pair of neurons, one from each end, carries a synapse.
- `sign`: the mean sign, from -1 (every synapse inhibitory) through 0 (mixed) to +1 (every
  synapse excitatory). Each synapse is excitatory with probability `(1 + sign) / 2`.
- `scale`: multiplies the magnitudes, which are drawn uniformly from
  `[0, sqrt(6 / (pre neurons + post neurons))]`.
- `count`: the synaptic contacts per synapse.
- `reciprocal`: `True` adds the reverse of every synapse with the same signed weight, so a
  learner can tie the two into one reciprocal synapse and a nudge travels back; `False`
  declares a one-way projection.

Development draws every magnitude and sign from its seed. Synapses that two projections
place on the same ordered pair of neurons merge into one, whose count is the sum and whose
sign is the count-weighted mean; a synapse from a neuron onto itself is dropped. `Genome`
checks every projection end when it is built.

## Ports after development

`develop(genome, seed=0)` lays the regions out in order as contiguous populations named
after them, adds each designed region's synapses and its populations as
`region/population`, and draws every projection. The result is one `Connectome` for one
`Brain`, the same for the same seed. A port is a named population of that connectome.

```python
genome = cd.Genome(
    regions=(cd.Region("senses", 6), association, motor, cd.Region("value", 1), loop),
    projections=(
        cd.Projection("senses", "association", density=0.5, reciprocal=False),
        cd.Projection("association", "motor"),
        cd.Projection("association", "value", scale=0.5),
        cd.Projection("loop", "association", sign=1.0, reciprocal=False),
        cd.Projection("association", "loop/cue", sign=-1.0, density=0.25, reciprocal=False),
    ),
    label="two-heads",
)
connectome = cd.develop(genome, seed=0)
for name, members in connectome.populations.items():
    print(f"{name:13s} {members[0]:2d} to {members[-1]:2d}")


def between(connectome, pre, post):
    """The synapses from the neurons of port ``pre`` to the neurons of port ``post``."""
    return np.isin(connectome.pre, connectome.populations[pre]) & np.isin(
        connectome.post, connectome.populations[post]
    )


print(between(connectome, "senses", "association").sum())   # 64 of 144 pairs at density 0.5
print(between(connectome, "association", "senses").sum())   # 0: a one-way projection
print(between(connectome, "motor", "association").sum())    # 72: the reverse synapses
print((connectome.sign[between(connectome, "loop", "association")] > 0).all())  # True
```

The ports are `senses`, `association`, `motor`, `motor/actions`, `value`, `loop`,
`loop/cue` and `loop/readout`.

## Which synapses a head owns

A learning head is a `Learner` with two boolean masks: `plastic_synapses`, one entry per
synapse, and `plastic_neurons`, one entry per neuron whose bias moves. Synapses and biases
outside the masks keep their values through the head's updates, decay and clipping. Build
the masks from port pairs.

A reciprocal pair shares one efficacy. The learner zeroes the steps of frozen synapses,
averages the two directions of every pair and zeroes the frozen steps again, so a synapse
whose reverse is frozen moves by half its own step. A head owns both directions of every
reciprocal projection it learns. `Learner(synapse_rate=...)` multiplies each synapse's step
before that averaging, so one head can learn its projections at different rates.

Heads with disjoint masks share one brain. An update gives the updating head a new `Brain`
with the changed parameters (`learner.brain`); hand that brain to the other head before
its next phase.

```python
def head(connectome, *pairs):
    """Both directions of the synapses between each pair of ports."""
    owned = np.zeros(connectome.synapses, bool)
    for pre, post in pairs:
        owned |= between(connectome, pre, post) | between(connectome, post, pre)
    return owned


def ports(connectome, *names):
    """A neuron mask of the named ports."""
    mask = np.zeros(connectome.n, bool)
    for name in names:
        mask[list(connectome.populations[name])] = True
    return mask


brain = cd.Brain(connectome, cd.learning_neuron_model())
policy = cd.Learner(
    brain,
    connectome.populations["motor/actions"],
    plastic_synapses=head(connectome, ("association", "motor")),
    plastic_neurons=ports(connectome, "motor"),
)
value = cd.Learner(
    brain,
    connectome.populations["value"],
    cd.LearnerConfig(nudge="quadratic"),
    plastic_synapses=head(connectome, ("association", "value")),
    plastic_neurons=ports(connectome, "value"),
)
assert not (policy.plastic_synapses & value.plastic_synapses).any()
```

## A learner over one head

`Learner.step(drive, labels)` settles a free phase, the nudged phases toward the labelled
outputs and one update. The phase operations `free`, `nudged` and `update` take an
explicit target pattern, here a value for the quadratic head.

```python
drive = np.zeros((2, connectome.n))
drive[:, list(connectome.populations["senses"])] = np.eye(6)[:2]
before = brain.efficacy.copy()

policy.step(drive, np.array([0, 2]))          # one action index per row
value.brain = policy.brain                    # continue on the updated parameters
target = np.zeros((2, connectome.n))
target[:, list(connectome.populations["value"])] = [[1.0], [0.0]]
free = value.free(drive)
plus = value.nudged(drive, free, target)
minus = value.nudged(drive, free, target, sign=-1.0)
value.update(free, plus, minus)
policy.brain = value.brain

moved = value.brain.efficacy != before
owned = policy.plastic_synapses | value.plastic_synapses
assert moved[owned].any() and not moved[~owned].any()
```

## The records cortex

`cd.Records` is a cortex whose cells lie outside the connectome. Its reading is a vector
the application assembles: the activation of a port, the action, the reads of declared
stores, missing flags. A fixed random expansion codes the reading, and one delta-rule write
per witnessed outcome changes its records. No settling phase, residual check or
certificate involves it, and a write leaves every `Brain` parameter as it was.
[Records](memory.md#records) describes the mechanism.

```python
records = cd.Records(24 + 3, {"value": 1}, valued=["value"], cells=2000, active=20, seed=0)
association_port = list(connectome.populations["association"])
settled = policy.free(drive)
reading = np.concatenate([settled.activation[0, association_port], np.eye(3)[0]])
code = records.code(reading, adapt=True)[:, 0]
efficacy = policy.brain.efficacy.copy()
records.write(code, {"value": np.array([1.0])})
print(records.read(code)["value"])                       # [1.]
assert np.array_equal(efficacy, policy.brain.efficacy)
```

## Settle or record

A cortex needs settling when its answer depends on a joint state of many neurons:

- **Completion.** Part of the reading is missing, and the missing part must agree with the
  rest: the settled regions reach a fixed point that satisfies every synapse at once.
- **Context across a delay.** The outcome depends on something the present reading lacks.
  A recurrent state, or a `Trace` into a `prefrontal_cortex`, carries it into the next
  settling.
- **A policy under late credit.** Reward arrives after other decisions. The eligibility
  trace at every synapse holds a decaying sum of the contrasts of the actions taken, and
  the prediction error of the reward writes through it (`ActorCritic`, [reward](reward.md)).

Records suffice when the outcome follows the reading it belongs to (the consequence of an
action, the reward of a choice, a fact seen at a place), when the reading holds what
decides the outcome, and when readings that should generalise to each other share active
cells. A record read is one product of the code with each table and a write is one
delta-rule step; neither runs a settling step. [Compose a brain](brain.md) wires records
and a settled policy into one experience step.
