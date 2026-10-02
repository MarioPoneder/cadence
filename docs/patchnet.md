# PatchNet: continuous observations and isolated rehearsal

`PatchNet` composes the settling brain for continuous observations and
[recursive observers that settle together](recursive-settlement.md).
The graph interface composes the nonlinear `Brain` and local contrastive `Learner` without
requiring an external associative store. Its default factory creates a fully
reciprocal graph; the older `layered` factory alone does not add every reverse
input contact.

```python
import numpy as np
import cadence as cd

net = cd.PatchNet.create(
    inputs=2, hidden=8, outputs=1, seed=7, density=1.0,
    config=cd.LearnerConfig(nudge="quadratic", eta=0.2, eta_bias=0.02),
    steps=512, tolerance=1e-5,
    context_strength=1.0,  # experimental temporal overlap; default is zero
)

observation = np.array([[0.2, 0.8]])
drive = net.stimulus(observation)
result = net.observe(drive, np.array([[0.6]]), source_id="sensor:1")
print(result.reason, net.read(result.free))

# Hypothetical continuations use a private activity branch.
branch = net.imagine([
    net.stimulus(np.array([[0.3, 0.7]])),
    net.stimulus(np.array([[0.4, 0.6]])),
])
print(net.read(branch[-1]))

path = net.save("patch-life.npz")
resumed = cd.PatchNet.load(path)
resumed.reset()  # clears activity, preserving learned parameters and evidence IDs
```

The numbers in this example are supplied observations and teaching targets,
not a demonstrated memory or prediction benchmark. Check `result.updated`
and `result.reason`. The possible outcomes include `updated`, `duplicate`,
`no_observations`, `free_unconverged` and `nudge_unconverged`.

## What persists

The network has current activity and learned relationships. Activity is
carried between calls; synaptic efficacy and biases change only through
learning or an explicit parameter edit. `reset()` clears the former and
retains the latter, along with optimizer history and the evidence-ID window.
An activity attractor can retain a cue, but the supplied graph and its learned
parameters must actually admit that behavior. Passing a state between calls
does not by itself establish useful short-term memory.

With `context_strength=kappa > 0`, a selected set of neurons also overlaps
with the previous free activity. If that boundary is `c`, each solve adds
the drive `kappa * context_mask * (c - activity)`, corresponding to the
quadratic energy `kappa/2 * sum(context_mask * (activity - c)**2)`.
The default boolean mask selects the factory's hidden neurons. A custom
graph needs declared hidden neurons or an explicit nonempty `context_mask`.
Both configuration fields are checkpointed. Fresh and reset streams use a
zero boundary; an explicit zero `BrainState` gives the same equation.

The boundary is held fixed across the free, positive and negative phases
of one observation. It advances to the free activity only between events;
imagined events advance their own private boundary. There is no gradient
through previous events and no additional persistent context array.
The target-nudged state never becomes the next boundary. A unique equilibrium
for each fixed boundary can therefore coexist with history-dependent behavior.

This introduces a measurable tradeoff. The first small library comparison at
`kappa=1` preserved cue differences for one blank event, but those differences
almost vanished after 16 blanks. Repeated warm-context associations were
learned; querying them immediately after clearing context failed the declared
accuracy threshold. Holding a zero boundary adds a restoring force that is
absent once consecutive activities agree. Temporal overlap is therefore an
experimental option, not a completed short-term or long-term memory mechanism.
Increasing it can also require a smaller integration step for convergence.

There is no autonomous importance detector, replay archive, permanent weight
lock or synaptic consolidation potential in this interface. Ordinary idle
settling does not decay learned weights. If the learner's optional uniform
weight decay is enabled, it applies indiscriminately to trainable parameters.
Protection of rare useful memories and selective forgetting during continued
learning are open capability tests, not consequences of serialization.

## Observed ports and local detuning

`stimulus(inputs)` places a continuous external drive on named input neurons.
Inputs are soft fields, not hard clamps: the reciprocal network can feed back
onto them. `observe(drive, target, observed=mask, weight=gain)` accepts a target
array of shape `(batch, outputs)`, a shared boolean output mask, and optional
nonnegative gain per batch row. Unobserved target entries may be NaN; they
are excluded from the nudge. Zero gain or an empty observation mask advances
only the free activity and does not commit learning.

The teaching target is absent from the free phase. A quadratic nudge pulls observed
outputs toward it; centered learning compares positive and negative nudges
started from that same free state. Existing synaptic updates compare the
activities of each contact's endpoints. The loss sums squared errors over
observed output ports and averages over batch rows. Gain is supplied teaching
strength; it is not inferred importance or truth reliability.

Each phase has an explicit step budget and checks the potential/adaptation
equations through `Brain.equilibrate`. If a required phase does not converge,
the free activity is retained but weights, optimizer history and source IDs
are not updated. The carried state is never replaced by the target-nudged
state. After a successful parameter update it is an initial state for the
next solve, not a certificate of equilibrium under the changed parameters.

`PatchNet` requires reciprocal effective weights, matching reciprocal
contact/gain factors and plasticity masks, and no adaptation. The usual
equilibrium-propagation gradient interpretation additionally assumes a smooth
stable equilibrium branch and sufficiently small nudge. A finite residual
check does not prove those assumptions. The factory's unit gain/contact
factors avoid a parameter-unit conversion; arbitrary compatible learners may
require accounting for those factors in interpreting the raw contrast.

Batch rows identify persistent streams. Call `reset()` when their identities
change. Reordering examples while carrying the same row states changes the
experiment and must not masquerade as a chronological lifetime.

## Evidence and imagined continuations

Only an explicit `observe` call accepts teaching evidence. With `source_id`,
an already committed identical event is skipped without advancing live state.
Reusing that ID with a different payload raises an error. The oldest ID
expires when `source_capacity` is exceeded; a replay after expiry can be
accepted again. This bounded deduplication window is not a full causal poset,
authentication system or test of independent real-world evidence.

`imagine(drives, state=...)` returns a sequence of `Equilibrium` objects on a
private branch. Each starts from the previous branch state. A caller can feed
predicted outputs into a subsequent branch call by passing its last state;
that feedback mapping and any action selection are explicit application
choices. Live state, parameters, optimizer and evidence IDs remain unchanged,
including when a later branch input raises an error. No branch automatically
becomes a witnessed fact or a training target. The application remains
responsible for the truth of a later explicit `observe` call.

This implements a substrate for internal rehearsal. Creative problem solving
requires a learned consequence model, useful candidate generation, measured
selection and successful autonomous output. An isolated branch alone proves
none of those capabilities.

## Checkpoint and readback

`save` atomically writes a versioned NumPy archive containing the complete
learner and continuation state. `load` validates it and supports an explicit
backend choice. Exact continuation requires the same backend and numerical
precision. `state` and `snapshot()` return detached copies, so editing them
does not mutate the live network. Snapshot array sizes count payload storage;
they do not include Python objects, numerical scratch or external run logs.

## Research reproduction

The retention comparisons behind this page were run against the library with
frozen sources, checkpoints and an independent endpoint verifier, and both the
activity-only and the temporal-overlap outcomes were kept. They are development
probes, not evidence of a competent controller or animal-like lifelong memory.

The existing `GenericBrain` and `Records` APIs remain supported for prior
applications. They contain additional mechanisms and should not be treated
as evidence that the smaller `PatchNet` composition has already reproduced
their capabilities.
