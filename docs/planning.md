# Private input planning

`TemporalPatchNet.plan` proposes changes to declared input ports using the same
temporal energy and centered detuning used for learning. It holds the model and
initial boundary fixed. Each accepted proposal must improve an ordinary free
prediction of the supplied goal. The method executes no action and admits no
new observation.

This API is a local optimizer for
continuous controls under an existing model. A useful application still needs a
model that predicts actual consequences and an external action/readback loop.
The [interaction guide](interaction.md) provides an executable learned-model loop.

## A bounded proposal

The following supplied toy model responds to its second input port. Its first
port represents a fixed observation; both ports remain explicitly declared.
The example illustrates action repair, not body-model acquisition.

```python
import numpy as np
from cadence import TemporalPatchNet

net = TemporalPatchNet(inputs=2, hidden=1, outputs=1, tolerance=1e-12)
net.set_parameters({
    "A": np.zeros((1, 1)),
    "B": np.array([[0.0, 1.0]]),
    "C": np.ones((1, 1)),
})
inputs = np.array([[[5.0, 0.0], [-3.0, 0.0]]])
before = net.snapshot()
proposal = net.plan(
    inputs,
    goal=np.full((1, 2, 1), 0.8),
    controls=np.array([False, True]),
    bounds=(-0.2, 0.2),
    rate=8.0,
    max_steps=8,
)

assert proposal.improved
np.testing.assert_array_equal(proposal.inputs[:, :, 0], inputs[:, :, 0])
np.testing.assert_allclose(proposal.inputs[:, :, 1], 0.2)
assert proposal.converged            # No feasible local descent remains.
assert not proposal.predicted_goal_met  # The bound prevents reaching 0.8.
for name, values in before.items():
    np.testing.assert_array_equal(values, net.snapshot()[name])
```

Inputs have shape `(batch, time, inputs)` and the goal has shape
`(batch, time, outputs)`. The boolean `controls` mask broadcasts to the input
shape and must select at least one port. Only selected values may change.
Optional bounds are a `(lower, upper)` tuple whose entries broadcast to that
shape; selected initial values must already satisfy them. Fixed ports may lie
outside the bounds. An explicit `state` supplies a private initial hidden
boundary; otherwise the current live boundary is copied, or zero is used for a
network with no active state.

## What is optimized

The reported cost is half the weighted mean squared error of a free prediction:

```text
cost = 0.5 * mean_batch,time,outputs(output_precision * (free_output - goal)²)
```

At each proposal, the solver repairs positive and negative goal detunings from
the same fixed boundary. If `e_plus` and `e_minus` are their transition defects
and `B` maps input ports into hidden activity, the input contrast is

```text
gradient = -((e_plus - e_minus) @ B) / (2 * beta * batch)
```

Time and output normalization already enter through the detuned loss. The
gradient is set to zero on fixed ports. A projected step clips controlled
values to their bounds; backtracking accepts it only after a converged,
target-free replay gives both a strict cost decrease and the Armijo decrease.
Goal-detuned outputs never stand in for the returned prediction.

Under smooth stable-branch assumptions, this contrast approaches the free-loss
gradient as beta tends to zero. Finite beta and solver tolerances introduce
error. A stationary result is not a global-optimality certificate. Use
`0 < beta * max(output_precision) < time * outputs` so both detunings are valid.
Teaching precision is the caller's supplied metric, not learned confidence.

## A centered contrast

The contrast is a derivative only while the two detuned paths sit
symmetrically around the free path. On a long path, or a path far from its
goal, one detuned solve can converge on another branch of the energy; both
phases then report convergence and the contrast points somewhere else. On a
learned cart-pole model over sixteen steps, the default beta gave a contrast
25 times too large and 58 degrees off the true derivative, and the line search
then found no decreasing step.

`cadence.temporal.contrast_asymmetry(free, plus, minus)` measures this: the
norm of `plus + minus - 2 * free` over the norm of `plus - minus`, both over
the hidden paths. A centered contrast has a ratio proportional to beta, and
its error against the derivative grows with the square of the ratio. Every
contrast in `plan` and `observe` must have a ratio at most `symmetry_tolerance`
(default 0.15, a contrast error of a few percent). Otherwise beta is halved and
both phases are solved again, up to `max_halvings` times (default 8). A
contrast that stays off center ends the proposal with the reason
`contrast_asymmetric`. The result reports the `beta` of its last contrast and
the number of `contrast_halvings`. An infinite tolerance disables the check.

## A quasi-Newton direction

The cost over an action path is badly conditioned: an early action moves
every later output, a late one almost nothing, so the steepest direction
zigzags. `method="bfgs"` keeps an estimate of the inverse curvature over the
controlled entries, built from the accepted steps by the BFGS update, and
steps along `-curvature @ contrast` with the same projected line search and
causal replay. The first step, and every step after the estimate is dropped,
is the steepest step scaled by `rate`. The estimate is dropped when a step is
clipped by the bounds, when the curvature condition fails, when the direction
does not descend, or when no step along it decreases the cost; the proposal
then restarts from the contrast. `step_sizes` are multiples of the direction,
with 1 the full quasi-Newton step. On a twelve-step toy the same stationary
cost takes 15 iterations and 52 phase calls instead of 25 and 121. The
default `method="steepest"` is unchanged.

## Reading the result

`TemporalPlan` contains detached input and prediction arrays, the fixed
boundary, the initial prediction and the frozen model's `parameter_revision`.

| Field | Meaning |
| --- | --- |
| `cost`, `initial_cost`, `losses` | Costs of the initial and accepted free predictions. Every accepted cost strictly decreases. |
| `improved`, `iterations`, `step_sizes` | Whether cost decreased, accepted step count and actual step scales including `rate`. |
| `projected_residual` | Maximum absolute change under a unit projected input-gradient step at the last accepted proposal; `None` when its contrast failed. |
| `converged` | The projected finite-beta residual met `tolerance`. It does not mean the goal is attainable. |
| `predicted_goal_met` | The returned free prediction's cost is at most `goal_tolerance`. It does not certify a real outcome. |
| `reason` | Projected stationarity, step cap, failed phase, off-center contrast, nonfinite gradient or absence of a decreasing replay step. |
| `beta`, `contrast_halvings` | The detuning of the last contrast and how often the supplied beta was halved to center the detuned paths. |
| `method` | `steepest` or `bfgs`, the search direction that produced the proposal. |
| Work counters | Phase calls, attempted block-chain solves, energy evaluations and peak message storage across all attempted phases, including rejected work. |

`max_steps=0` still evaluates a free prediction and both detunings to report
stationarity. Reaching a cap does not count as convergence. If a detuned phase
fails, or the contrast stays off center after every halving, the last valid
free proposal is returned without claiming stationarity.
If every attempted replay fails or cannot decrease cost, that same valid
proposal remains. An invalid initial free prediction raises `ArithmeticError`.
Invalid arguments raise `ValueError` before any live mutation.

Planning runs on a private checkpoint copy. It changes no learned arrays,
active state, readback or revision of the original network, and it does not
modify caller arrays. Its work counters exclude checkpoint copies, Python
overhead and numerical-library workspace; they are not a full runtime or
memory measurement.

## Closing the loop

Execute a selected control through the real application, obtain its measured
consequence, and then supply that actual record to the normal learning path.
Do not teach the model that its desired goal or predicted future was observed.
If the live state or parameters change before execution, recompute the proposal
from the current boundary.

This interface supplies action search under a declared goal. It does not
generate goals, choose informative experiments, infer hidden body parameters,
perform categorical sample selection, or add an automatic higher-level
self-model. Such behaviors need matched causal tests with actual consequences;
achieving a small private planning cost does not establish them.
