# Quickstarts: three kinds of brains

Cadence has two primitives. A **settling patch** finds the state that agrees with what its
ports hold and with its weights, and learns by the contrast of a free and a nudged settle.
A **record patch** adds a store that takes an observation in one write and a night in
which the slow weights learn from the store's own dreams. Every brain below is one of
these, or a few of them joined by ports. Each snippet runs on NumPy alone, and each runs
behind a local page as well, with the whole brain animated and the learning plotted as it
[the quickstarts in your browser](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart).

```bash
python -m pip install cadence-net
```

## A patch that learns a stream, remembers in one shot, and sleeps

The record patch hears a stream of symbols and predicts what follows. By day every
outcome is written once into its records and the slow weights do not move; at night the
slow weights take what the store holds, from the store's own completions, with the
stream closed. Afterwards the weights alone know the rule.

```python
import numpy as np
from cadence import RecordPatchNet

rng = np.random.default_rng(21)
net = RecordPatchNet(
    12, 12, 5, seed=4, cells=2048, active=16, record_rate=1.0, groups=(5,), slowest=8.0
)
symbols = rng.integers(12, size=(3, 8))
heard = np.eye(12)[symbols]
outcome = np.eye(5)[(symbols + np.roll(symbols, 1, axis=1)) % 5]  # the last two symbols decide

for _ in range(8):  # the day: one write per moment, slow weights at rate zero
    net.reset()
    net.observe(heard, outcome, rate=0.0)
net.reset()
awake = net.imagine(heard, state=np.zeros((3, 12)))
assert np.array_equal(awake.output.argmax(-1), outcome.argmax(-1))  # the store recalls

night = net.sleep([heard], passes=240, rate=8.0, backtrack=True)  # dreams, then dawn
alone = RecordPatchNet.restore(net.snapshot())
alone.records.tables["y"][:] = 0.0  # the same weights with an empty store
assert np.array_equal(alone.imagine(heard, state=np.zeros((3, 12))).output.argmax(-1), outcome.argmax(-1))
print(night)
```

the records of each reading; `sleep` dreams every cue once, teaches the fixed dreams by
`observe(write=False)`, and rewrites the store at dawn. Categorical ports (`groups`), batched writes, a store narrower
than its port and a two-patch stack are in [the record patch guide](record-patch.md).

## A settling brain that decides

A genome names regions and projections; `develop` lays them out as one connectome; a
`Brain` settles it; a `Learner` over the motor neurons nudges the chosen action and moves
the synapses on the contrast. Here five senses map to three actions.

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(24), motor_cortex(3, lateral=-0.5)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
)
connectome = cd.develop(genome, seed=0)
brain = cd.Brain(connectome, cd.learning_neuron_model())
senses = list(connectome.populations["senses"])
learner = cd.Learner(brain, connectome.populations["motor/actions"], cd.LearnerConfig(eta=1.0))

drive = np.zeros((5, connectome.n))
drive[np.arange(5), senses] = 1.0
labels = np.arange(5) % 3  # sense k asks for action k mod 3
learner.calibrate(drive)  # the gain that puts the free motor activity in its responsive range
for _ in range(80):
    learner.step(drive, labels)
assert learner.accuracy(drive, labels) == 1.0
```

reads the most active motor neuron.
Regions, ports, records beside a policy head and evolution of the genome are in
[compose a brain](brain.md), [write a cortex](cortex.md) and [evolve a brain](evolution.md).

## A temporal patch that learns a consequence and plans

The temporal patch learns how its inputs move its outputs by centered detuning, keeps its
context between calls, imagines privately, and repairs a continuous action toward a goal
under its own learned model.

```python
import numpy as np
from cadence import TemporalPatchNet

net = TemporalPatchNet(2, 8, 1, seed=151)
heard = np.array([[[1.0, 0.0]]])
net.reset()
learned = net.observe(heard, np.array([[[0.2]]]))
assert learned.updated
private = net.imagine(np.zeros((1, 4, 2)))
assert private.converged
```

is replayed without its goal before acceptance, and actual readback repairing the next
proposal; the same loop as text is [learn, act and observe](interaction.md), and protection
of chosen responses is [response protection](temporal-memory.md).

## Which one

| You want | Start with |
| --- | --- |
| to learn from a stream of events, keep single facts, and generalise overnight | the record patch |
| a decision or evaluation over a fixed set of inputs, an explicit graph, a policy that learns from reward | the settling brain |
| continuous observations and actions with a learned dynamics model and private planning | the temporal patch |

The reciprocal `PatchNet` and the older `GenericBrain` composition have their
own learning and state contracts; see the
[index](index.md#kept-for-existing-experiments).
