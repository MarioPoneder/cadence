# Deep recursive settlement: one shared equilibrium

An observer can read another population and feed back into it **inside one
`PatchNet` graph**. Both populations use the same rate-neuron rule; every
settling round updates the combined state. No external steering callback or
second completed solve supplies the observer's answer.

**The observer becomes part of what settles.** Write the complete state as
`x = (base, observer_1, ..., observer_D, outputs)` and solve
`x = F_theta(x; input)` for all of it. Each observer reads the evolving current
state, feeds back into it, and is itself changed by that feedback. A previously
settled base can initialize the larger system; it must remain free to change.
An immutable copy of that base passed into a second network tests a different
architecture.

## Choose the depth and widths

On library `main` after 0.17.0, the `PatchNet.recursive` factory constructs that
wiring directly. The explicit construction below also runs on released 0.17.0.

```python
import numpy as np
import cadence as cd

net = cd.PatchNet.recursive(
    inputs=3, layers=[12, 8, 4], outputs=2, seed=7,
    coupling=1.0, steps=1024, tolerance=1e-9,
    config=cd.LearnerConfig(nudge="quadratic", beta=1e-3),
)
drive = net.stimulus([[0.2, 0.4, 0.1]])
phase = net.settle(drive)
assert np.all(phase.converged)
answer = net.read(phase)  # activity of output neurons inside this same graph

lesson = net.observe(drive, [[0.3, 0.1]], source_id="example:recursive-1")
assert lesson.updated, lesson.reason
assert all(np.all(p.converged) for p in (lesson.free, lesson.plus, lesson.minus))
```

`layers[0]` is the base hidden population. Every later entry adds an observer
population with reciprocal connections to **all earlier populations**, including
the input and output neurons. `[12, 8, 4]` therefore gives one base and two
observer levels; `[12]` supplies the shallow control. Use `[32] * 9` for one
base and eight observer levels, or provide any positive widths explicitly.
The named populations `layer_0`, `layer_1`, and so on make interventions and
per-level measurements possible. `observer` selects the observing populations;
`hidden` contains the base and all observers.

This factory returns an ordinary `PatchNet`. It shares the existing solver,
learning, checkpoint and rejection rules. All three teaching phases solve the
complete graph. It adds no trainable feed-forward answer head and no second
controller. Inputs are soft external drives, so their neural representations
also receive feedback. External input values themselves remain fixed during a
solve.

The supplied wiring, widths and initialization are architectural choices to
test or evolve, not discovered cognitive roles. The factory uses the smooth
`tanh(v/2)` rule and scales all initial reciprocal weights together so their
largest absolute incoming row sum equals `coupling`. At the default of one,
the free fixed-point map has Lipschitz bound one-half. Learning can change that
bound, and nudges change the equations; qualification is still required for
every phase. This is a convenient starting topology, not a claim that dense
reciprocal wiring is the most scalable architecture. Its edge count grows with
the sizes of the connected populations.

## What the deepest observer can do

The deepest observer can read every earlier population through its ports and
influence their settled state. It does not receive unconditional authority:
the lower populations also change it. Which influence dominates depends on
the learned couplings, evidence and dynamics. Choosing the final level as a
decision port would select where an answer is measured; it would not exempt
that level from the shared equations. A limited-width observer can also fail
to preserve information available elsewhere in the graph.

All-to-all reciprocal access does not by itself establish a meaningful
cognitive hierarchy. Layer labels describe construction, not intelligence.
Test whether another observing level adds useful capability against widening,
cut-feedback and frozen-readback controls at matched resources.

## Inspect the mechanism explicitly

This small example uses two base neurons and two observer neurons. Their names,
couplings and external ports are supplied candidate genes, not learned roles.
The reciprocal interface carries current activity in both directions. Inputs
are soft drives: input neurons also receive feedback during settlement.

```python
import numpy as np
import cadence as cd

# W[post, pre]: the complete observer–observed system.
W = np.array([
    [0.00, 0.12, 0.22, 0.00],
    [0.12, 0.00, -0.04, 0.18],
    [0.22, -0.04, 0.00, 0.12],
    [0.00, 0.18, 0.12, 0.00],
])
tolerance, beta = 1e-10, 1e-4


def make_patch(weights):
    post, pre = np.nonzero(weights)
    graph = cd.Connectome.from_synapses(
        4, pre=pre, post=post, sign=weights[post, pre],
        populations={"input": [0], "output": [1],
                     "base": [0, 1], "observer": [2, 3]},
    )
    model = cd.learning_neuron_model(leak=1.0)
    brain = cd.Brain(graph, model, bias=np.full(4, 0.15), backend="cpu")
    learner = cd.Learner(
        brain, [1], cd.LearnerConfig(
            nudge="quadratic", beta=beta, eta=0.01, eta_bias=0.01,
        ),
    )
    return cd.PatchNet(learner, steps=1024, chunk=8, tolerance=tolerance)


def qualify(phase, weights, drive, strength=0.0):
    # With this model, activation is tanh(v/2). Independently evaluate
    # every neuron's potential equation, including the observer neurons.
    v = phase.state.v
    activity = np.tanh(v / 2)
    defect = activity @ weights.T + drive + 0.15 - v
    defect[:, 1] += strength * (0.35 - activity[:, 1])
    assert np.all(phase.converged)
    assert np.max(np.abs(defect)) <= tolerance
    return activity


def solve(patch, weights, drive):
    patch.reset()
    return qualify(patch.settle(drive), weights, drive)


patch = make_patch(W)
drive = np.array([[0.4, 0.0, 0.0, 0.0]])
free = solve(patch, W, drive)
observer_input = drive + [[0.0, 0.0, 0.0, 0.4]]
feedback = solve(patch, W, observer_input)
assert np.max(np.abs(feedback[:, :2] - free[:, :2])) > 1e-3
base_input = drive + [[0.4, 0.0, 0.0, 0.0]]
readback = solve(patch, W, base_input)
assert np.max(np.abs(readback[:, 2:] - free[:, 2:])) > 1e-3

# Cut both directions of the interface, retaining reciprocity and all neurons.
# Changing observer input can now no longer change the base equilibrium.
cut = W.copy()
cut[:2, 2:] = cut[2:, :2] = 0
control = make_patch(cut)
cut_free = solve(control, cut, drive)
cut_feedback = solve(control, cut, observer_input)
np.testing.assert_allclose(cut_feedback[:, :2], cut_free[:, :2], atol=1e-10, rtol=0)

# Explicit teaching uses the SAME combined graph in all three phases.
patch.reset()
lesson = patch.observe(drive, [[0.35]], source_id="example:teaching-1")
assert lesson.updated
for phase, strength in [(lesson.free, 0), (lesson.plus, beta), (lesson.minus, -beta)]:
    qualify(phase, W, drive, strength)  # W and bias are the pre-update parameters.
np.testing.assert_array_equal(patch.state.v, lesson.free.state.v)
```

Here the fixed-point map has Lipschitz bound
`0.5 * max(sum(abs(W), axis=1)) = 0.19`, below one. The small nudges also
stay within a contraction bound. Arbitrary graphs need their own stability
analysis; a small measured residual alone does not prove uniqueness or a
minimum. The contrast's gradient interpretation additionally requires the
appropriate parameter scaling and a smooth stable branch in the small-nudge
limit. See the [numerical contracts](contracts.md) and [PatchNet guide](patchnet.md).

To add another observing level, add another population with reciprocal ports
to the same graph and check the equations over **all** populations. This
demonstrates a mechanism for recursive coupling, not that labels create
metacognition or that depth improves learned capability. Tests of depth need
matched parameters, information, experience and total computational work,
including failed solves, against shallow, sequential and feed-forward controls.

`BeliefPatch` instead performs a fixed number of repair iterations and learns
through an adjoint scan. `Steered` runs readback, steering and cortex
sequentially; replay can admit their parameter step without producing a shared
observer–observed equilibrium. `JointRecordPatches` exchanges contexts across
finite Jacobi rounds, with a finite-round adjoint and diagnostic seams. These
are useful comparison implementations with different contracts. Raising their
iteration budgets does not alone establish the checked shared-equilibrium
learning contract demonstrated here.

## Before comparing with a transformer

1. **Declare the complete state and equations.** Include every claimed observing,
   observed and decision population. A residual-reading or gain-changing
   observer must participate in those equations; an external diagnostic
   callback is not such an observer. Distinguish previous-event boundaries
   from current-event populations: the former can stay fixed, the latter
   must settle together.
2. **Qualify the whole solution.** Independently evaluate the maximum equation
   defect over every state component at the returned state. Report per-level
   residuals, iteration budget, caps and nonfinite outcomes. Small update
   motion, a port seam RMS, or `rounds >= depth + 1` is not this check.
   Preserve failed cases in the denominator and charge their work.
3. **Name the training rule.** `PatchNet.observe` uses qualified free and
   positive/negative phases with local contrasts. A finite unroll with its
   backward scan is a separate training arm. An implicit equilibrium gradient
   requires its own converged linear solve and gradient check. Convergence of
   the forward state alone does not make these gradients interchangeable.
4. **Test causal participation.** Perturb base and observer populations in both
   directions. Cut their feedback, freeze the observer's readback, and compare
   with a sequential observer. Verify that every required phase still includes
   the same complete state and that only target-free activity is carried.
5. **Match the actual experiment.** Use the same available information, labels,
   masks, splits and history. Separate architecture selection from held-out
   evaluation. Report parameter-matched and compute-matched comparisons, with
   comparable tuning budgets and controlled random seeds for every framework.
   Count all settling phases, backwards/adjoint work, retries, memory and
   selection work; distinguish inference from total training cost. Hardware
   timings must identify device, precision and implementation.

A successful mechanism test establishes shared settlement and feedback. A
learned task advantage, a benefit from depth and training efficiency versus
transformers require their own results; they do not follow from the builder.
