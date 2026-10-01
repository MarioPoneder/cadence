# Quickstart

Use Python 3.11 or later. Cadence has no mandatory runtime dependencies.
For the released baseline, install `cadence-net==0.50.0`. To run **all examples
in this development checkout**, including the candidate reward acknowledgment
API, install the checkout itself from its root:

```sh
python -m pip install -e .
```

Without a local checkout, install the `0.60.0.dev0` development line from `main`:

```sh
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@main"
```

Pin a reviewed commit instead of `main` when preserving an experiment.
The [migration guide](MIGRATION_060.md) separates this development rollout from
the published package. Nothing here enables automatic System 1/System 2 attention.

## Build a flat brain

A `Cortex` declares a layout; `build()` returns its persistent `Brain`. Start
with one patch that learns a small sensor-to-answer relation:

```python
from cadence import Brain, Cortex, bootstrap

layout = Cortex(seed=2)
signal = layout.input("signal", shape=1)
response = layout.column("response", patches=1, inputs=signal)
layout.output("answer", shape=1, reads=response)
brain = layout.build()
assert brain.inspect()["patches"] == 1
assert brain.inspect()["outputs"][0]["sensor_coverage_by_coordinate"] == (1,)
```

`patches` counts processing states; `shape` describes sensor or output data.
The output exposes a patch state, with no separate readout network. This flat
layout is enough for the relation below. For learned intermediate features,
use [deep ordinary or recursive layouts](VARIANTS.md); their operations and
external interface are the same. Adding unused flat patches supplies no hidden
representation to this output.

## Inspect an unbootstrapped brain

```python
before = brain.snapshot()
result = brain.settle({"signal": [0.4]})
assert result["qualified"]
assert brain.snapshot() == before  # queries never change continuation
print(result["outputs"], result["stationarity"])

activity = brain.step({"signal": [0.4]})
assert activity["accepted"]  # retain this live state; weights stay fixed
```

`settle` returns diagnostics even on refusal. Use outputs only when `qualified`
is true. `predict` returns qualified outputs directly and raises
`SettlementError` otherwise. A numerical budget limits work; it does not promise
convergence. A qualified state may still have prediction error.

## Bootstrapping phase

```python
examples = [
    ({"signal": [x]}, {"answer": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
checks = [
    ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)
]
report = bootstrap(
    brain, examples, checks=checks, max_error=0.1, epochs=20, batch_size=4,
)
assert report["passed"], report

# Target-free recall at amplitudes absent from teaching and readiness checks.
assert abs(brain.predict({"signal": [-0.5]})["answer"][0] + 0.3) < 0.1
assert abs(brain.predict({"signal": [0.5]})["answer"][0] - 0.3) < 0.1
```

The helper validates every example before making changes, then shuffles and
replays examples through `observe_batch` here (`observe` when `batch_size=1`).
It scores recall and the separate checks without target clamps, stopping when both meet the declared error limit or work stops.
`observe` fixes the witnessed outputs and jointly repairs state and retained
relation parameters. It commits only a fully qualified proposal. This is
supervised learning; provide the actual outcome you intend to teach, not a
reward disguised as a desired motor value. `observe` and `bootstrap` do not
compute reward-to-action credit; use the `Reinforcement` helper below for that.
Accuracy must be evaluated later without target clamps, since a clamped
bootstrap output equals its target by construction.

Use consistently scaled inputs and targets comfortably inside both the `tanh`
prediction range and the state bounds (for example, ±0.6 with default bounds).
Increasing `state_bound` does not expand the `tanh` range.
Fit any centering, scaling or whitening on bootstrap examples only. Shuffle or replay
representative experiences when learning a reusable relation. In control tasks,
check performance in the actual environment: high agreement on demonstration
states can still fail on states the learned policy visits.

`report["passed"]` covers the supplied recall and development checks. The checks
affect stopping, so reserve fresh cases for final evaluation. A refused solve
stops the helper; earlier admitted examples remain learned. For event-by-event
control or external retry identities, use `observe` directly.

For an interrupted admission, retry with the same explicit `event_id` and the
same inputs, physical output clamps and `source` label. An identical latest retry
reports `duplicate=True`, `accepted=False` and performs no learning. Older or changed
identities are rejected. If IDs are omitted, the next ID is allocated on
acceptance; automatic IDs do not identify a retried external event.

## Save and resume the learned brain

```python
saved = brain.snapshot()
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"signal": [0.5]}) == brain.predict({"signal": [0.5]})
```

Save this JSON text with your application. Loading binds the exact implementation
sources, layout and configuration. Save preprocessing and body/history state
alongside it. See [continuation](#save-a-whole-reward-learner) for reward learners.

This completes the first learning loop. Run the same loop with all three
layouts using `python examples/layout_learning.py --layout all`, or continue
with the [layout quickstarts](VARIANTS.md) and [brain-design guide](BRAIN_DESIGN.md).

## Learn several experiences together

Use `batch_size` when bootstrapping independent witnessed examples. This small
scalar relation uses one directly sensing output patch:

```python
batch_layout = Cortex(seed=7)
batch_signal = batch_layout.input("signal", shape=1)
batch_response = batch_layout.column("response", patches=1, inputs=batch_signal)
batch_layout.output("answer", shape=1, reads=batch_response)
batch_brain = batch_layout.build()

batch_examples = [
    ({"signal": [x]}, {"answer": [0.6 * x]})
    for x in (-0.8, -0.4, 0.4, 0.8)
]
batch_checks = [
    ({"signal": [x]}, {"answer": [0.6 * x]}) for x in (-0.6, 0.6)
]
live_state = batch_brain.state
batch_report = bootstrap(
    batch_brain, batch_examples, checks=batch_checks,
    max_error=0.1, batch_size=4,
)
assert batch_report["passed"], batch_report
assert batch_report["accepted"] == 4 * batch_report["updates"]
assert batch_brain.state == live_state
assert abs(batch_brain.predict({"signal": [0.5]})["answer"][0] - 0.3) < 0.1

# Direct batch admission uses the same pairs and one event identity.
batch_update = batch_brain.observe_batch(batch_examples)
assert batch_update["accepted"]
assert len(batch_update["states"]) == 4
assert batch_brain.state == live_state
```

Each row gets private activity initialized from the same live state. The rows
jointly repair one set of shared parameters under a mean-example objective
with one pre-batch parameter anchor. An accepted batch is one update/event;
`presentations` and `accepted` count examples. Its private solved states are
returned for inspection, while live state is preserved. Query or `step` the
current input when you want current activity under the updated relations.

The default `batch_size=1` retains ordinary ordered `observe` calls. Larger
batches change the learning trajectory, so recheck acquisition and retention;
they are not equivalent to parallel serial updates. Batching supplies neither
temporal credit nor implicit sequence memory. Tensor execution can parallelize
the batch; [measure the complete workload](ACCELERATION.md) before claiming a
speedup.

## Live phase

```python
activity = brain.step({"signal": [0.3]})
assert activity["accepted"]
print(activity["outputs"]["answer"])
assert brain.observe({"signal": [0.3]}, {"answer": [0.18]})["accepted"]
```

Use the same brain, sensory encoding and output decoding in the live phase.
`step` retains settled activity; actual teaching signals can continue to arrive
through `observe`. Learning remains available. The application decides when
the demonstrated ability is sufficient for its environment; there is no hidden
phase switch in the solver.

## Learn from a reward

When the environment supplies a reward instead of a desired output, use
`Reinforcement`. Here a supplied toy body moves left or right; moving closer to
zero earns positive reward. Cadence selects the action, the body executes it,
and only then does the helper record its consequence:

```python
from cadence import Reinforcement

reward_layout = Cortex(seed=2)
position_sensor = reward_layout.input("position", shape=1)
values = reward_layout.column(patches=2, inputs=position_sensor)
for i, name in enumerate(("left", "right")):
    reward_layout.output(name, shape=(), reads=values, indices=(i,))
reward_learner = Reinforcement(
    reward_layout.build(), actions=2, action_input=None,
    value_output=("left", "right"), seed=2,
)

position = 0.6
choice = reward_learner.act({"position": [position]})
assert choice["accepted"]
next_position = position + (-0.1, 0.1)[choice["action"]]  # actual body step
reward = abs(position) - abs(next_position)
feedback = reward_learner.feedback(
    reward, {"position": [next_position]},
    decision_id=choice["decision_id"], executed_action=choice["action"],
)
assert feedback["stored"]
assert feedback["accepted"]
```

This one transition illustrates ownership, not an acquired navigation skill.
Continue collecting actual transitions and evaluate later choices to test
learning. A final episode outcome uses `terminal=True` and no next inputs.
For delayed rewards, record intervening transitions with their actual rewards
(often zero); replay can propagate the later value backward. `History` supplies
an explicit recent-observation window when current sensing is insufficient.
See [the live guide](LIVE.md) for bounded demonstrations and exact failure rules.

## Save a whole reward learner

A reward learner needs its own complete snapshot, not just its brain:

```python
saved_reward_life = reward_learner.snapshot()
resumed_reward_life = Reinforcement.from_snapshot(saved_reward_life)
assert resumed_reward_life.snapshot() == saved_reward_life
```

This also saves replay, pending action and exploration/replay RNG. Save the body,
external history and any environment RNG at the same boundary; they are not
contained in the learner snapshot. Do not execute a pending action twice after
restoring. See [saving a whole life](LIVE.md#save-a-whole-life-and-run-the-small-gates).

See [the DRSN guide](DRSN.md) for multimodal and nested layouts, and the
[reference](REFERENCE.md) for all options and result fields.
