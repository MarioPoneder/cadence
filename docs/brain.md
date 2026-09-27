# Compose a brain

A brain is a `Genome` of cortices and projections, developed into one connectome and run
as one `Brain`. This page composes a small brain, checks its settling, writes one
experience step by hand with a records cortex beside a policy head, describes
`GenericBrain`, saves every part and draws the brain in a browser page.

## Genome, development, brain

```python
import numpy as np
import cadence as cd
from cadence.regions import cortex, motor_cortex

genome = cd.Genome(
    regions=(cd.Region("senses", 5), cortex(16), motor_cortex(3, lateral=-0.5)),
    projections=(
        cd.Projection("senses", "association", reciprocal=False),
        cd.Projection("association", "motor"),
    ),
    label="corridor-brain",
)
connectome = cd.develop(genome, seed=0)
brain = cd.Brain(connectome, cd.learning_neuron_model())
senses = list(connectome.populations["senses"])
actions = list(connectome.populations["motor/actions"])
print(connectome.summary()["populations"])
```

The regions are populations of one connectome, and every settling step updates all of
their neurons together, so the regions reach one joint state. [Write a cortex](cortex.md)
describes regions, projections and ports.

## Settle and check

`equilibrate(drive, *, budget=512, chunk=32, tolerance=1e-5, state=None, mask=None,
nudge=None)` continues a state in chunks of settling steps until every row's fixed-point
equation residual is at most `tolerance` or the step budget is spent. It returns an
`Equilibrium` with the `state`, the per-row `residual`, the total `steps` and the per-row
`converged` flags. `residual(drive, state)` measures the same equations for any state
without settling.

```python
drive = np.zeros((1, connectome.n))
drive[0, senses] = np.eye(5)[2]
result = brain.equilibrate(drive, budget=256, chunk=16, tolerance=1e-6)
assert result.converged.all()
print(result.steps, brain.residual(drive, result.state))

cert = cd.certificate(brain)
print(cert.certified, round(cert.row_mass, 2), cert.mass_limit)   # False 5.71 2.0
```

The largest incoming weight sum of this brain exceeds the certificate's limit of 2, so the
certificate gives no bound and the residual is the check. A converged row satisfies the
equations at its state; uniqueness and stability need the [certificate](certificate.md) or
tests of their own.

## One experience step by hand

A body lives in a corridor of five cells and moves left, stays or moves right. The reward
is the reduction in distance to the goal cell, and reaching the goal places the body on a
random cell other than the goal. The records cortex reads the body's cell and a candidate
action and keeps two fields: `next`, the cell that followed, at the consequence rate, and
`reward`, a valued field at the fast rate. The policy head is a `Learner` over the motor
actions.

Each moment runs seven operations in causal order:

1. Observe: the cell drives the senses.
2. Code the reading of every candidate action. Imagined readings leave the running mean
   as it is.
3. Read the records: the expected reward of each candidate.
4. Choose: on one moment in five a random action, otherwise the largest expected reward
   plus a tenth of the policy's preference.
5. Act: the environment moves the body.
6. Write the witnessed outcome into the records of the executed reading, whose code moves
   the running mean.
7. Update the policy head from reward: a nudge toward the executed action, weighted by its
   advantage, the reward less the mean expected reward of the candidates.

```python
records = cd.Records(
    5 + 3, {"next": 5, "reward": 1}, valued=["reward"], cells=2000, active=20, seed=1
)
policy = cd.Learner(brain, actions, cd.LearnerConfig(eta=0.5))
rng = np.random.default_rng(0)
goal, cell = 4, 0


def reading(cell, action):
    return np.concatenate([np.eye(5)[cell], np.eye(3)[action]])


for moment in range(200):
    drive = np.zeros((1, connectome.n))
    drive[0, senses] = np.eye(5)[cell]                                     # 1. observe
    codes = records.code(np.stack([reading(cell, a) for a in range(3)]))   # 2. code
    expected = records.read(codes)["reward"][:, 0]                         # 3. read
    free = policy.free(drive)
    preference = free.activation[0, actions]
    if rng.random() < 0.2:                                                 # 4. choose
        action = int(rng.integers(3))
    else:
        action = int(np.argmax(expected + 0.1 * preference))
    following = int(np.clip(cell + action - 1, 0, 4))                      # 5. act
    reward = float(abs(goal - cell) - abs(goal - following))
    witnessed = records.code(reading(cell, action), adapt=True)[:, 0]      # 6. write
    records.write(witnessed, {"next": np.eye(5)[following], "reward": np.array([reward])})
    advantage = reward - expected.mean()                                   # 7. learn
    policy.step(drive, np.array([action]), warm=free, weight=np.array([advantage]))
    cell = int(rng.integers(4)) if following == goal else following
```

The records answer questions about imagined moves without settling. The goal cell has no
records: reaching it moves the body elsewhere before a reading there is witnessed.

```python
questions = np.stack([reading(c, a) for c in range(4) for a in range(3)])
imagined = records.read(records.code(questions))
print(imagined["next"].argmax(axis=1).reshape(4, 3))    # the cell each move leads to
print(imagined["reward"][:, 0].reshape(4, 3).round(2))  # the expected reward of each move
assert (imagined["reward"][:, 0].reshape(4, 3).argmax(axis=1) == 2).all()

drives = np.zeros((4, connectome.n))
drives[:, senses] = np.eye(5)[:4]
print(policy.predict(drives))                            # the policy's greedy action per cell
```

The loop keeps no replay store: each outcome is written once, when it is witnessed. The
policy settles one free phase for its preference and three phases for its update; the
records' reads and writes settle nothing. Where reward arrives several decisions after the
action that earned it, step 7 belongs to an `ActorCritic`, whose eligibility traces carry
the credit ([reward](reward.md)). Imagined readings follow the rules in
[records](memory.md#records), including the missing flags.

## GenericBrain

`GenericBrain.build(inputs, actions, *, hidden=64, density=1.0, lateral=-0.5,
working_memory=False, memory_scale=12.0, episodic=True, features=8, field=3, seed=0,
**options)` develops `GenericBrain.genome(...)` and wraps it in the ready composition:

- `sensory`, a blank region of `inputs` neurons, or `visual`, a `visual_cortex` when
  `inputs` is an image shape `(height, width)` or `(height, width, channels)`;
- `association`, a blank cortex of `hidden` neurons that the senses reach one-way (a
  visual cortex reaches it reciprocally);
- `motor`, a `motor_cortex` of `actions` neurons with lateral weight `lateral`, joined to
  the association cortex by reciprocal synapses;
- `basal_ganglia`, an `ActorCritic` whose critic reads the association cortex and whose
  dopamine moves every synapse through its eligibility trace;
- with `working_memory=True`, `working_memory`: a `Trace` of the association cortex that
  drives a `prefrontal_cortex`, which projects one-way to the association cortex at
  `memory_scale`;
- with `episodic=True`, `hippocampus`: a `SynapticMemory` from sensory to motor neurons
  that records the reward of each chosen action.

`GenericBrain.step` runs the ongoing loop described in [continuous interaction](continuous.md).
`GenericBrain(connectome, ...)` wraps any connectome with `sensory` (or `visual/input`),
`association` and `motor` populations, and a `prefrontal` population for working memory, so
a genome from `GenericBrain.genome` can be edited or [evolved](evolution.md) first. The
composition holds no records cortex; a `Records` beside it reads the observation and the
action, as in the step above.

```python
generic = cd.GenericBrain.build(5, 3, hidden=16, working_memory=True, seed=0)
print(sorted(generic.connectome.populations))
print(type(generic.basal_ganglia).__name__, type(generic.working_memory).__name__,
      type(generic.hippocampus).__name__)                 # ActorCritic Trace SynapticMemory
action = generic.step([np.eye(5)[0]])
```

## Checkpoints

| Part | Save | Restore | Contents |
|---|---|---|---|
| `GenericBrain` | `save(path)` | `GenericBrain.load(path, *, backend="cpu", device=None, precision=None)` | The complete composition: parameters, critic, both optimizers, random state, working memory, both hippocampal timescales, eligibility and an action awaiting its outcome. |
| A head (`Learner`) | `save(path, *, compressed=True)`, also `cd.save(learner, path)` | `Learner.load(path, *, backend=None, device=None, config=None, precision=None)`, also `cd.load` | The connectome, efficacies, gains and biases, neuron model, configuration, outputs and slots, plasticity masks, tie groups, synapse rates, optimizer history and update counts. |
| `Records` | the configuration from `to_dict()`, the arrays `tables`, `mean` and `pathway_norm`, and the counters `seen` and `writes` | `Records(**configuration)`, then the arrays and counters | The configuration rebuilds the fixed cells from the seed; the arrays and counters are the learned state. |

An `ActorCritic`, `FastSynapses` or `Trace` built outside `GenericBrain` is saved by the
code that owns it.

```python
import json

policy.save("policy.npz")
policy = cd.Learner.load("policy.npz")

np.savez(
    "records.npz",
    configuration=json.dumps(records.to_dict()),
    mean=records.mean,
    seen=records.seen,
    pathway_norm=records.pathway_norm,
    writes=records.writes,
    **{f"table/{name}": table for name, table in records.tables.items()},
)
with np.load("records.npz") as saved:
    restored = cd.Records(**json.loads(str(saved["configuration"])))
    restored.tables = {name: saved[f"table/{name}"].copy() for name in restored.fields}
    restored.mean = saved["mean"].copy()
    restored.seen = int(saved["seen"])
    restored.pathway_norm = saved["pathway_norm"].copy()
    restored.writes = int(saved["writes"])
assert np.array_equal(restored.projection, records.projection)
assert np.allclose(restored.read(restored.code(questions))["next"], imagined["next"])

generic.save("generic.npz")
generic = cd.GenericBrain.load("generic.npz")
```

## The brain in a browser page

The whole-brain viewer, the atlas that lays a connectome out and the self-contained replay page
live in the examples repository, [cadence-examples/viewer](https://github.com/muellerberndt/cadence-examples/tree/main/viewer);
the library ships the brains alone, and `record_settlements` supplies the settlings a page replays.

