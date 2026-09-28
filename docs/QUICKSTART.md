# Quickstart

Install Python 3.11+ and `python -m pip install --upgrade "cadence-net>=0.45.0"`.
The [release wheel](https://github.com/muellerberndt/cadence/releases/latest)
is also available when a package index has not listed the new version yet. Cadence has no
runtime dependencies. A `Cortex` declares a layout; `build()` returns its
persistent `Brain`.

## Connect populations

```python
from cadence import Brain, Cortex

layout = Cortex(seed=2)
signal = layout.input("signal", shape=(1,))
base = layout.column("perception", patches=4, inputs=signal)
observer = layout.observer(
    "reflection", patches=2, inputs=signal, observes=base,
)
layout.output("answer", shape=(1,), reads=observer)
brain = layout.build()
assert brain.inspect()["patches"] == 6
assert brain.inspect()["outputs"][0]["sensor_coverage_by_coordinate"] == (1,)
```

`patches` counts live processing coordinates. `inputs` reads fixed sensor data
or other populations' states. `observes` additionally reads live prediction
errors. Observer feedback participates in the same solve as the populations it
observes. Increase width with `patches`; add recursive depth by observing an
observer. These are separate choices.

Default wiring reads every coordinate of each declared source. For a scalar
control or regression problem, start with one output patch reading the sensors;
the six-patch layout above illustrates observation. A nonlinear hidden
representation needs a connected second population, not unused patches beside
an output. See [training and size](TRAINING.md) for starting ranges and setup.

## Query and continue activity

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

## Admit experience and test recall

```python
for event in range(40):
    value = (-0.8, 0.8)[event % 2]
    result = brain.observe(
        {"signal": [value]}, {"answer": [value]}, event_id=event,
    )
    assert result["accepted"]

# Target-free recall at two amplitudes absent from teaching.
assert brain.predict({"signal": [-0.4]})["answer"][0] < -0.2
assert brain.predict({"signal": [0.4]})["answer"][0] > 0.2
```

`observe` fixes the witnessed outputs and jointly repairs state and retained
relation parameters. It commits only a fully qualified proposal. This is
supervised learning; provide the actual outcome you intend to teach, not a
reward disguised as a desired motor value. Reward-to-action temporal credit is
not supplied by this API. Accuracy must be evaluated later without target
clamps, since a clamped training output equals its target by construction.

Use consistently scaled inputs and targets comfortably inside both the `tanh`
prediction range and the state bounds (for example, ±0.6 with default bounds).
Increasing `state_bound` does not expand the `tanh` range.
Fit any centering, scaling or whitening on training data only. Shuffle or replay
representative experiences when learning a reusable relation. In control tasks,
check performance in the actual environment: high agreement on demonstration
states can still fail on states the learned policy visits.

For an interrupted admission, retry with the same explicit `event_id` and the
same inputs and physical output clamps. An identical latest retry reports
`duplicate=True`, `accepted=False` and performs no learning. Older or changed
identities are rejected. If IDs are omitted, the next ID is allocated on
acceptance; automatic IDs do not identify a retried external event.

## Save and resume

```python
saved = brain.snapshot()
resumed = Brain.from_snapshot(saved)
assert resumed.snapshot() == saved
assert resumed.predict({"signal": [0.4]}) == brain.predict({"signal": [0.4]})
```

Save the returned JSON text with your application. It contains layout,
configuration, state, parameters and admission identity. Loading requires the
same layout/repair/validation source hashes. Restored brains accept sensor and
output names; handles from another brain are foreign. Checkpoints are bounded
continuation records, not authenticated proof of an experience.

See [the DRSN guide](DRSN.md) for multimodal and nested layouts, and the
[reference](REFERENCE.md) for all options and result fields.
