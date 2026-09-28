# Quickstart

These examples use Python's standard library only. They demonstrate the API,
not a benchmark or a claim that a small model solves a particular application.
Use the [reference](REFERENCE.md) for every option and the
[specification](SPECIFICATION.md) for qualification and evidence boundaries.

## A scalar sensor

A column estimates a changing scalar center and its uncertainty. Normalize or
choose the value bound to match the sensor's units.

```python
from cadence import CorticalColumn

sensor = CorticalColumn(decay=0.9, value_bound=8.0)
for value in (0.2, 0.4, 0.3):
    reading_before = sensor.query()
    assert reading_before["qualified"]
    result = sensor.add(value)
    assert result["accepted"]

reading_after = sensor.query()
assert reading_after["qualified"]
print(reading_after["answer"], reading_after["variance"])
```

`add` allocates the next event number. `decay` controls how much earlier evidence
survives each accepted observation. It is a model choice: a smaller value can
track change faster while discarding older information sooner. A value of one
keeps pooling evidence; it does not disable learning.

## Delivery with retries

If the application owns event ordering, use `observe`. IDs start at one and
advance only when an observation is admitted.

```python
from cadence import CorticalColumn

sensor = CorticalColumn()
first = sensor.observe(1, 0.25)
assert first["accepted"]
saved = sensor.snapshot()

retry = sensor.observe(1, 0.25)
assert retry["duplicate"] and not retry["accepted"]
assert sensor.snapshot() == saved
```

Only an identical retry of the latest committed event is a duplicate. A
changed value for that ID, an older ID or a skipped ID is an error. Queries
are not observations. Do not feed a prediction back as if a sensor measured it.

A capped admission returns an unqualified status; a resource refusal raises
`ValueError`. Retain the real event and handle the declared failure before
advancing its ID. Capacity and exact-statistic limits belong in the application
budget; `capacity=None` is not a promise of unlimited storage.

## Predict from several sensors

A Cortex associates observed outputs with contexts. This example uses a
supplied two-dimensional grid and learns two outputs from the same reading.
The coordinates and outputs use units in the chosen bounds.

```python
from cadence import Cortex

model = Cortex.from_dimensions(
    n_inputs=2, n_outputs=2, bounds=(-1.0, 1.0), bins=8, depth=1,
)
context = (-0.5, 0.25)
before = model.predict(context)
assert len(before) == 2

result = model.observe(context, (-0.3, 0.6))
assert result["accepted"]
after = model.predict(context)
assert len(after) == 2
print(before, after)
```

Binning determines which inputs share evidence. It is an explicit representation,
not learned feature extraction. Coarser levels supply priors to finer contexts;
they can share evidence but can also blur important distinctions. Use a flat
model as a control when evaluating whether the hierarchy helps.

For a partially observed target, supply only the known output indices:

```python
from cadence import Cortex

model = Cortex.from_dimensions(2, n_outputs=2)
result = model.observe((0.1, -0.2), {1: 0.4})
assert result["accepted"]
beliefs = model.query((0.1, -0.2))
assert len(beliefs) == 2
assert all(belief["qualified"] for belief in beliefs)
```

The absent output is not taught zero. This does not make the Cortex a general
missing-input inference engine: its context maps still specify which inputs
a query requires. A vector of predictions is a tuple; richer uncertainty and
qualification diagnostics come from `query` or `value`.

## Resume a built-in model

Built-in feature maps have serializable descriptors, so a Cortex can reconstruct
them from its checkpoint.

```python
from cadence import Cortex

model = Cortex.from_dimensions(2, n_outputs=1)
context = (0.1, -0.2)
assert model.observe(context, 0.4)["accepted"]
saved = model.snapshot()

twin = Cortex.from_snapshot(saved)
assert twin.predict(context) == model.predict(context)
```

Custom feature-map functions are not serialized as executable code. Give them
a stable `wiring_id` and supply compatible functions when restoring. Changing
the meaning of a context map while keeping its ID is an application error that
a string identifier cannot detect. Checkpoints bind the declared model
configuration and retained continuation state; they do not include a robot or
environment's state.

## Add observer stages

```python
from cadence import CorticalColumn

column = CorticalColumn(height=2)
for value in (-0.2, 0.2, -0.2):
    assert column.add(value)["accepted"]
answer = column.query()
assert answer["qualified"]
print(answer["precision"], answer["stages"])
```

Height 1 already has reciprocal belief–precision feedback. Height 2 adds an
observer of the live precision proposal; further stages repeat that structure.
This changes the model's uncertainty, not merely its solve budget. Test its
usefulness separately from adding context levels or more bins.

## Optional action values

`Cortex` also offers an explicit temporal-difference wrapper for discrete
actions. The application supplies executed transitions and rewards. Native
`observe` learns supplied output targets immediately; `learn` constructs its
own value targets using reward, bootstrapping and the configured novelty bonus.

```python
from cadence import Cortex

agent = Cortex(n_outputs=2, update_mode="step", optimism=0.0, seed=3)
observation = (0,)
action = agent.act(observation)

# A synthetic transition illustrates the interface. In an application,
# these three values must come from the action's actual measured outcome.
next_observation, reward, terminal = (1,), 0.25, True
agent.learn(observation, action, reward, next_observation, terminal)
agent.flush()
```

With `update_mode="episode"`, learning buffers transitions until a terminal or
truncated transition, or an explicit `flush`; the wrapper processes that
buffer backward. With `update_mode="step"`, each transition is processed
immediately. An environment truncation is not automatically an absorbing
terminal: use the distinct `truncated` flag and your task's termination rule.
Episode storage, calibration and all settling work count toward the budget.

`predict` and `act`, including random exploration, refuse unqualified numerical
answers. Catch `SettlementError` at the application boundary and choose an
explicit fallback rather than treating an unfinished solve as a valid
prediction. Numerical qualification does not guarantee safe control or correct
predictions.
