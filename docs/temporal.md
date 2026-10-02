# TemporalPatchNet

`TemporalPatchNet` is the temporal patch: a settling patch over a path. Each
moment owns hidden activity and an output, with an overlap to the preceding
moment. Observations enter declared input ports. Whole-path repair lets later
observed discrepancies affect earlier temporal patches during learning.

It uses residual predictive coding and centered equilibrium propagation on
time-varying input and output vectors with a persistent initial boundary. It is
a NumPy CPU implementation; the optional backends of the settling brain do not
accelerate it.

## A finite observed path

All paths have shape `(batch, time, ports)`. Batch rows are independent streams
sharing parameters. Targets contain externally supplied observations and are
read only in the detuned learning phases.

```python
import numpy as np
from cadence import TemporalPatchNet

net = TemporalPatchNet(inputs=2, hidden=4, outputs=1, seed=7)
observations = np.zeros((2, 6, 2))
observations[:, 0, 0] = [-1.0, 1.0]  # one initial cue per stream
observations[:, :2, 1] = 1.0        # a supplied observation-presence port
heard = np.array([0.0, 0.0, 0.2, -0.2, 0.2, -0.2])
target = np.stack([heard, -heard])[:, :, None]

result = net.observe(observations, target, beta=0.01, rate=0.1)
assert result.updated
assert result.free.converged and result.plus.converged and result.minus.converged
np.testing.assert_array_equal(net.state, result.free.final_state)
```

One update does not establish accurate prediction. Measure accuracy on withheld
paths, delayed-cue tests and continued learning. `observe` returns the free,
positive and negative phases, an update decision and the centered parameter
derivatives. Failed detuning changes no parameters or update count. A valid
free final state still becomes the current state. Invalid inputs change nothing.
No target-detuned hidden state becomes the live continuation boundary.

## Context, private continuation and readback

The learned arrays `A`, `B`, `C` persist separately from the active hidden state.
`advance` carries target-free activity without learning. `imagine` starts at the
current boundary and returns a detached path without changing the network.
To start a private branch cold, supply an explicit zero state. `reset()` clears
active context and diagnostics while retaining all learned arrays and updates.

```python
net.reset()
primer = net.advance(observations[:, :2])
assert primer.converged
before = net.snapshot()
future_inputs = np.zeros((2, 8, 2))
branch = net.imagine(future_inputs)
assert branch.converged
for name, values in before.items():
    np.testing.assert_array_equal(values, net.snapshot()[name])

readback = net.readback()
assert readback.state.shape == (2, 4)
assert np.all(np.abs(readback.activity) <= 1.0)
assert readback.state_parameter_revision == readback.parameter_revision
```

This is functional self-readback: detached activity, equation residual and energy
are available as observations to an application. It is not an automatic
higher-level self-model or evidence of consciousness. After `observe`, readback
labels the parameter revision used for its free phase; that phase precedes the
learning update. Its old residual does not certify equilibrium under new weights.

The caller decides which inputs represent observations, intention or external
controls. Zeros mean zero drive, not a built-in missing-data flag. Add a declared
presence port when an observed zero must differ from an absent observation.
Neither the solver nor checkpoints insert a clock, future cue or desired output
into free inference. A supplied intention is not learned goal formation.

## Energy and update

For inputs `u[t]` and a fixed initial hidden boundary `h[-1]`, the defects are

```text
e[t] = h[t] - A tanh(h[t-1]) - B u[t]
r[t] = y[t] - C tanh(h[t])
E = mean_batch(0.5 sum_time(||e[t]||² + ||r[t]||²))
L = 0.5 mean_batch,time,outputs(output_precision * (y - target)²)
```

Free inference is the causal recurrence and has zero defects up to roundoff.
That says the path agrees with the model; it says nothing about its accuracy,
usefulness or creative quality. Tanh bounds activity, not the stored hidden
preactivation or output. Nonfinite paths are rejected.

The two teaching phases repair `E + beta L` and `E - beta L`, starting at the
same free path and holding the same initial boundary. Outputs are eliminated
analytically per output with `b[j] = beta * output_precision[j]/(time * outputs)`.
Every negative-phase coefficient must satisfy `b[j] > -1`. Adjacent hidden-width blocks pass
precision matrices and information vectors through an exact block Hessian solve.
The Hessian includes residual times activation-curvature terms. This is a
matrix computation local to temporal blocks, not a scalar-synapse-only solver.

Damped Newton proposals need positive Schur pivots and an Armijo energy decrease;
the rounding allowance is `1e-15 * max(1, abs(previous_energy))` per stream.
Convergence requires the full state gradient to meet `tolerance` and the final
**unshifted** Hessian to have positive Schur pivots. Iteration caps, failed
curvature checks and nonfinite updates are reported as failures. A positive
Hessian certifies only the returned local minimum, not a unique equilibrium or
that both detuned phases belong to the same smooth branch.

For each phase the parameter derivatives are local products:

```text
dE/dA = -mean_batch(sum_time(e[t] tanh(h[t-1])ᵀ))
dE/dB = -mean_batch(sum_time(e[t] u[t]ᵀ))
dE/dC = -mean_batch(sum_time(r[t] tanh(h[t])ᵀ))
delta = (dE_plus - dE_minus)/(2 beta)
parameter <- parameter - rate * delta
```

No derivative through the solver is used. Credit crosses the supplied finite
path through state repair. Its initial boundary is held fixed, so learning does
not differentiate through earlier calls. Under the usual smooth stable-branch
assumptions the centered contrast approaches the gradient of free prediction
loss as beta tends to zero. Finite beta and numerical tolerance introduce error.
The tests include independently evaluated energy/loss derivatives, a negative
curvature rejection and parity with the frozen scalar-cue engine.

## Explicit output teaching precision

Teaching precision defaults to one for every output. A caller may supply finite,
strictly positive per-output weights to express a declared task metric or noise
assumption. This changes the quadratic loss; it is not learned importance,
precision inference, a new optimizer or a temporal-variation guarantee.

```python
precision = np.array([2.0])
net.set_output_precision(precision)
np.testing.assert_array_equal(net.output_precision, precision)
```

The constructor also accepts `output_precision=...`. Updating precision is atomic
and leaves A/B/C, live activity, parameter revisions and free-energy diagnostics
unchanged: none of them depends on the teaching loss. Explicit response-memory
bindings remain valid. Checkpoints store the supplied vector; older checkpoints
without it load with all ones. The implementation preserves the default kernel
arithmetic for all-one weights.

For centered teaching, require `0 < beta * max(output_precision) < time * outputs`.
Large weights can therefore require smaller beta. If weights are estimated from
training statistics, freeze the formula, floors and normalization before
evaluation and retain the original unweighted task metrics. Weighting a loss
cannot alone establish useful autonomous continuation.

## API and checkpoints

| Operation | Effect |
| --- | --- |
| `observe(inputs, target, beta=0.01, rate=0.1, backtrack=False)` | Repair free/positive/negative paths; carry only free state. Optional parameter backtracking requires improvement in causal replay before committing. |
| `advance(inputs)` | Free inference, carrying a converged final state; no learning. |
| `imagine(inputs, state=None)` | Read-only free path from live state or the explicit supplied boundary. |
| `settle(inputs, target=None, beta=0, state=None)` | Read-only diagnostic detuning; target is wholly ignored at beta zero. |
| `reset()` | Clear active state/readback; keep parameters and update count. |
| `state`, `parameters()`, `readback()` | Detached copies, safe for private inspection. |
| `output_precision`, `set_output_precision(vector)` | Detached supplied positive teaching weights and atomic loss-configuration replacement; free dynamics unchanged. |
| `set_parameters(mapping)` | Explicit validated atomic replacement of A/B/C; preserves active state and marks the new parameter revision. |
| `snapshot()`, `TemporalPatchNet.restore(snapshot)` | Detached complete state and construction of a fresh continuation. |
| `save(path)`, `TemporalPatchNet.load(path)` | Atomic NumPy archive and validated restoration without pickle. |

```python
path = net.save("temporal_brain.npz")
restored = TemporalPatchNet.load(path)
np.testing.assert_array_equal(
    restored.imagine(future_inputs).output,
    net.imagine(future_inputs).output,
)
```

Every phase exposes `hidden`, `output`, `converged`, `reason`, `residual`, energy
history, accepted step sizes/damping, minimum Schur pivot eigenvalue and work
counts. `block_chain_attempts` counts attempted whole-chain solves, including
failed attempts and final curvature checks. Each successful chain factors one
hidden-width block per time step and batch row; it is not one scalar operation.
`message_bytes` is the largest successful chain's explicit precision/vector
storage, excluding arrays, Hessian blocks and numerical-library workspace.
`energy_evaluations` counts path-energy evaluations, including line searches.

## Checking that the contrast is centered

The centered contrast is a derivative of the free loss only while the two
detuned paths sit symmetrically around the free path. A detuned solve can
converge on another branch of the energy while still reporting convergence,
and the contrast then points somewhere else. `observe` and `plan` therefore
check `contrast_asymmetry(free, plus, minus)`, the norm of
`plus + minus - 2 * free` over the norm of `plus - minus` on the hidden paths,
against `symmetry_tolerance` (default 0.15). The ratio grows in proportion to
beta and the contrast error with its square, so an off-center pair is solved
again at half the beta, up to `max_halvings` times (default 8). A pair that
stays off center is refused with the reason `contrast_asymmetric`; parameters,
revisions and the update count do not change, and the valid free activity is
carried as for any failed phase. `TemporalObservation` reports the `beta` of
the last pair and the number of `contrast_halvings`. An infinite tolerance
disables the check. The measure is public as
`cadence.temporal.contrast_asymmetry`.

## Checking a learning step

`observe(..., backtrack=True)` checks
the proposed parameter change against the actual teaching path. Converged
positive and negative phases alone do not ensure that a finite parameter step
improves the model. A good direction can overshoot.

The optional check tries the supplied rate and up to fifteen halvings. Each
candidate makes a target-free prediction from the **original hidden boundary**.
Its precision-weighted half-MSE must decrease beyond a floating-point floor
and satisfy `new_loss <= old_loss - 1e-4 * rate * ||delta||²`. This is a
measured acceptance condition; a finite-beta EP estimate need not be an exact
gradient. There is no extra persistent optimizer state or retained episode.

`TemporalObservation` reports `initial_loss`, `final_loss`, `accepted_rate`,
`replay_losses` and `replay_calls`. A `None` trial loss denotes a nonfinite or
unsolved candidate. Nonfinite parameter proposals need no replay, so the
number of trials can exceed the replay count. Rejection returns
`no_decreasing_parameter_step`, preserves parameters, revisions and update
count, and carries only the original valid free activity. Failed phases keep
their distinct failure reason. With backtracking, rate zero cannot commit.
The fixed rate is the default.

This check improves the current observed objective only. It does not establish
generalization, protect earlier skills, choose important experiences or supply
new observations. Use the separate `TemporalMemory.observe` transaction when
response constraints are bound to the model; bypassing it with a direct
parameter update can invalidate those constraints. The new option is not
automatically applied by the memory transaction.

Free-phase residual and energy measure agreement within the model. The causal
recurrence has zero residual by construction even when its predictions are
wrong. Use actual observation-minus-prediction error to assess surprise or
model accuracy; do not read zero free energy as confidence about the world.

Persistent numeric parameters occupy `8*(hidden² + hidden*inputs + outputs*hidden)`
bytes; supplied teaching precision adds `8*outputs` bytes, and active state
occupies `8*batch*hidden` bytes when present. Metadata,
Python objects and transient solved paths cost additional space. Training paths
are not retained by the net. There is no replay buffer, protection against
interference, learned importance or guarantee of lifelong retention. Gaussian
past-message compression belongs to the separate fixed-model actor. A free
rollout is a prediction; the [planning interface](planning.md)
adds declared continuous action ports and preferences, using the same temporal
model for private input repair. [Actual interaction](interaction.md) checks
learned consequences against an executed body. Long-range context, retention,
transfer and creativity remain behavioral tests, not consequences of attaining
a small equation residual.
