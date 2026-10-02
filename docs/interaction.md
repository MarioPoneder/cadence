# Learn consequences, then act

This example learns how a small nonlinear body moves, privately repairs an
action proposal, executes its first action, and observes the result before
planning again. The same vector-port interface can carry measurements from
another application. The application defines what those vectors mean and
which input ports are controllable.

The body has one measured position and one bounded action. Its equation lives
in the simulator below; only executed actions and measured positions reach the
learner. A supplied goal is a preference, never a replacement for a measurement.

| Input port | Meaning | May planning change it? |
| --- | --- | --- |
| 0 | Measured position at the start of this path | No |
| 1 | Initial measurement is present | No |
| 2 | Executed action during learning; proposed action during planning | Yes, within `[-1, 1]` |

Only the first temporal patch receives the position and its presence flag.
Later patches receive actions and predict later positions. The zero presence
flag is our declared encoding; Cadence does not automatically interpret a zero
input as missing data. This fully observed body permits a fresh hidden boundary
at each replan. A partially observed environment would need a tested state
estimate or remembered context instead.

The Python blocks below run together, in order. They require NumPy and Cadence
with `TemporalPatchNet.plan`. They perform a small CPU experiment and create no
files.

```python
import numpy as np
from cadence import TemporalPatchNet


def actual_step(position, action):
    # The environment's transition, used to execute actions and read sensors.
    return 0.85 * position + 0.2 * np.tanh(2 * action) - 0.05 * position**3


def collect_actual_paths(rng, batch, horizon=6):
    position = rng.uniform(-0.6, 0.6, batch)
    actions = rng.uniform(-1.0, 1.0, (batch, horizon))
    inputs = np.zeros((batch, horizon, 3))
    inputs[:, 0, 0] = position
    inputs[:, 0, 1] = 1.0
    inputs[:, :, 2] = actions
    measured = np.empty((batch, horizon, 1))
    for t in range(horizon):
        position = actual_step(position, actions[:, t])
        measured[:, t, 0] = position
    return inputs, measured


net = TemporalPatchNet(inputs=3, hidden=16, outputs=1,
                       seed=709, initial_radius=0.8)
training_rng = np.random.default_rng(10709)
for _ in range(256):
    executed, measured = collect_actual_paths(training_rng, batch=8)
    net.reset()  # These are independent body episodes, not one continuous path.
    learned = net.observe(executed, measured, beta=0.01, rate=1.0)
    assert learned.updated

# Different states and action paths: no updates use these measured outcomes.
test_inputs, test_measured = collect_actual_paths(
    np.random.default_rng(20709), batch=128
)
test_prediction = net.imagine(test_inputs, state=np.zeros((128, 16)))
assert test_prediction.converged
forecast_mse = float(np.mean((test_prediction.output - test_measured) ** 2))
assert forecast_mse < 0.004
```

The model now predicts consequences. During the following control trial its
parameters remain fixed. Each call to `plan` copies the model and uses centered
equilibrium detuning to change only action ports. It accepts a change only
after target-free causal replay improves the predicted goal cost. Execution
and measurement remain outside the private branch.

```python
frozen = net.snapshot()
position = -0.45
zero_action_position = position  # Separate comparison body; not a planner input.
goal_position = 0.35
measured_positions = []
zero_action_positions = []
predicted_positions = []
plan_reasons = []

for t in range(12):
    if t == 6:
        # An unannounced physical disturbance. The controller sees its effect
        # only through the next measured position, not through a disturbance flag.
        position -= 0.25
        zero_action_position -= 0.25

    inputs = np.zeros((1, 6, 3))
    inputs[0, 0, :2] = [position, 1.0]
    proposal = net.plan(
        inputs,
        goal=np.full((1, 6, 1), goal_position),
        controls=np.array([False, False, True]),
        bounds=(-1.0, 1.0),
        state=np.zeros((1, 16)),
        beta=0.01,
        rate=8.0,
        max_steps=32,
    )
    assert proposal.prediction.converged
    assert proposal.reason in {
        "projected_stationary", "step_cap", "no_decreasing_causal_step"
    }
    np.testing.assert_array_equal(proposal.inputs[:, :, :2], inputs[:, :, :2])
    assert np.all(np.abs(proposal.inputs[:, :, 2]) <= 1.0)

    # Execute only the first action; later actions will be replanned after readback.
    action = float(proposal.inputs[0, 0, 2])
    predicted_positions.append(float(proposal.prediction.output[0, 0, 0]))
    position = float(actual_step(position, action))
    measured_positions.append(position)
    zero_action_position = float(actual_step(zero_action_position, 0.0))
    zero_action_positions.append(zero_action_position)
    plan_reasons.append(proposal.reason)

    # Planning has changed no learned arrays, active state, diagnostics or revisions.
    current = net.snapshot()
    for name, value in frozen.items():
        np.testing.assert_array_equal(value, current[name])

late_error = float(np.mean((np.array(measured_positions[-3:]) - goal_position) ** 2))
zero_action_error = float(
    np.mean((np.array(zero_action_positions[-3:]) - goal_position) ** 2)
)
assert late_error < 0.01
assert late_error < 0.25 * zero_action_error
print(f"Held-out forecast MSE: {forecast_mse:.6f}; late actual goal MSE: {late_error:.6f}")
```

The prediction and the measured consequence are separate records. The actual
measurement supplies the next planning boundary; the desired or imagined
output is never admitted as an observation. For later adaptation, train on
actual executed-input/measured-output records through `observe`, then plan
again under the updated model. That additional learning can change previous
responses and requires its own retention checks.

This example uses one validated initialization. Its assertions check this
bounded demonstration, not arbitrary bodies or seeds. The six-step proposals
may reach their iteration cap while still improving control: inspect
`plan_reasons`, `proposal.converged` and actual outcomes separately. The model
can be wrong even when its private prediction is internally consistent.

A one-step controller also works well on this monotone constant-goal task.
The example demonstrates learned action consequences and repeated repair
after measured disturbance; it does not establish an advantage from a longer
planning horizon, autonomous curiosity, a recursive hierarchy or hidden-state
memory. See [planning](planning.md) for result and failure semantics, and
[temporal learning](temporal.md) for how real observations update the network.
