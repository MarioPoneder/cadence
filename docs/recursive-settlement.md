# Recursive observation in one equilibrium

An observer can read another population and feed back into it **inside one
`PatchNet` graph**. Both populations use the same rate-neuron rule; every
settling round updates the combined state. No external steering callback or
second completed solve supplies the observer's answer.

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
