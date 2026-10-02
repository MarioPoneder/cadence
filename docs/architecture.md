# Architecture and integration

Cadence aims at generalized intelligence through overlap consensus, equilibrium
detuning and evolving functional self-reflection. This page is the integration
contract of the temporal patch (`TemporalPatchNet`, its planner and
`TemporalMemory`): local repair, persistent
activity, explicit response protection, private imagination and continuous action
planning, what each operation changes, and what its bounded tests establish. The
[record patch](record-patch.md) and the [belief patch](belief.md) state their
contracts in their own guides, and [Cadence for machine-learning people](orientation.md)
compares the four brains. Bounded tests do not establish the full flexibility of
an animal or human brain. The APIs below need no experiment-repository imports.

```bash
python -m pip install cadence-net
```

Development installs use `pip install -e .`; pin a release or a commit for reproducible
work. The graph interfaces of earlier applications stay available.

## What carries the individual forward

| Requirement | Base API | Exact current meaning |
| --- | --- | --- |
| Short-term context | `TemporalPatchNet.advance`, `state`, `reset` | Hidden activity persists between calls. Reset clears that activity, not weights. Whether a cue survives a particular delay is tested. |
| Acquired relationships | `TemporalPatchNet.observe` | Centered equilibrium detuning repairs a finite observed path and changes A/B/C. Supplied per-output teaching precision defines the task metric; its default is one. No gradient propagates through calls before the supplied initial boundary. |
| Protected long-term responses | `TemporalMemory.protect`, `memory.observe` | Caller-selected local response subspaces constrain later updates. An optional local readout metric improves conditioning, with checked causal replay. Exact-path retention is conditional and available capacity is finite. |
| Recursive temporal computation | `TemporalPatchNet` recurrence | Each moment depends on the previous hidden activity. This is recurrence, not an already learned hierarchy that observes itself. |
| Functional self-readback | `TemporalPatchNet.readback`, plan diagnostics | Detached state, residual/energy, revision, uncertainty and proposal information can be inspected or explicitly fed back by an application. |
| Private imagination | `TemporalPatchNet.imagine`, the planner | Private predicted paths leave live state and learned parameters unchanged. Their usefulness depends on model quality. |
| Goal-directed action | `TemporalPatchNet.plan` | The temporal learner privately repairs bounded continuous input ports and accepts only decreasing target-free prediction cost. |
| Continued life | Model and protection snapshots/checkpoints | Parameters, activity and supplied task settings can resume; save explicit protection together with its net. |

The nonlinear temporal learner can acquire an action/consequence relation and
use that same model to plan continuous controls. Its input gradient is another
centered equilibrium contrast, with parameters held fixed. Every candidate is
replayed without a goal nudge before acceptance. Its measured prediction accuracy
and the subsequent executed outcome remain separate checks.

Application adapters define sensory meanings, executable actions and teaching
access. The base library contains no musical structure, game policy, language
task or physical objective that silently supplies the missing skill.

The application must preserve distinctions needed for its task. A sensory
summary need not specify an action: an average event count, for example, loses
the timing of individual events. Validate known actions and their observed
consequences before treating forecast accuracy as performed competence. This
applies equally to physical control, games, language and creative work.

[Creativity and evolving self-reflection](creativity.md) describes the intended
progression from private proposals to useful novel solutions, learned internal
readback and transfer. Current readback is implemented; automatically growing
recursive coordination remains a behavioral research requirement.

## One temporal learning life

```python
import numpy as np
from cadence import TemporalPatchNet, TemporalMemory

net = TemporalPatchNet(2, 8, 1, seed=151)
memory = TemporalMemory()
heard = np.array([[[1.0, 0.0]]])
net.reset()
learned = net.observe(heard, np.array([[[0.2]]]))
assert learned.updated

# Supplied importance choice: preserve this current response from cold context.
cold = np.zeros((1, 8))
memory.protect(net, heard, state=cold)
net.reset()
changed = memory.observe(net, np.array([[[0.0, 1.0]]]), np.array([[[0.3]]]))
assert changed.updated
net.reset()
assert net.advance(heard).converged
private = net.imagine(np.zeros((1, 4, 2)))
assert private.converged
```

Only free activity becomes live after learning; target-detuned states stay
private. Protection is explicit and protects the current response, not an
unobserved truth. Long delay, competing learning, cue changes and novelty need
separate behavioral checks. There is no inferred importance, automatic fading,
learned specialization or unlimited long-term memory guarantee.

## Choosing evidence over architectural labels

A causal record order helps distinguish actual observations from imagined
branches and binds compressed state to its model. It does not itself decide
which experiences matter. A normal form means the declared equations or
constraints agree; it can still describe a poor predictor. Equilibrium and
memory are therefore tested by what the continuing system can recall, learn,
predict and do after disturbances and competing experience.

Detailed guides: [temporal learning](temporal.md),
[private continuous-control planning](planning.md), [general creativity and self-reflection](creativity.md),
[explicit response protection](temporal-memory.md), [the graph PatchNet](patchnet.md) and
[conditional Lean proofs](https://github.com/FloatingPragma/oph-meta/blob/main/cadence-flagship/lean/README.md).
