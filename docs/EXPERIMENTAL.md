# Experimental capabilities and demo evidence

**`0.62.0` implements one connected graph with a joint repair and qualification
contract.** Population sizes and connections determine the graph. `inputs=`
reads sensors or live states; optional `observes=` also reads exact current
prediction errors. All populations use the same patch law and participate in
the same solve. The [quickstart](QUICKSTART.md) starts with two populations.

<a id="system-2-warning"></a>

## Recursive observation and the intended correction cycle

Error readback is implemented; its task advantage remains experimental. An
observer participates in every synchronous solve. The public runtime does not
automatically put it to sleep during familiar behavior, recruit it on surprise
or give it an independent clock. Compare its learned behavior with a capable
state-coupled control using the same available information, disclosed capacity
and complete learning/query work. Preserve failures and recheck old skills.

**System 1** describes acquired routine competence with inexpensive repair.
**System 2** describes useful recursive correction when a routine misses a
forecast or cannot meet a goal. These are intended roles within one brain, not
architecture classes, depth settings or bootstrapping/live switches. The complete
routine → disturbance → useful correction → retained inexpensive routine cycle
has not been demonstrated by this public runtime. Private experiments linked
from [historical issue 72](https://github.com/muellerberndt/cadence/issues/72) do not become
released capabilities merely because the package version changes.

The current design/proof goals have three owners:
[short-term memory](https://github.com/muellerberndt/cadence/issues/84),
[long-term memory and plasticity](https://github.com/muellerberndt/cadence/issues/85),
and [recursive cortical-column integration](https://github.com/muellerberndt/cadence/issues/86).
They develop simplified biological abstractions within the common equilibrium.
Historical `0.11.0` Trace/Afterglow supplied decaying context to later events,
and SynapticMemory combined a transient component with persistent consolidation.
Those modules were removed in the `0.20` rewrite; current activity retention,
plastic parameters and external `History` are not evidence of their recovery.
The formal library is developed beside the paper in `cadence-flagship/lean`;
its model-specific theorems need an explicit bridge to any claimed runtime guarantee.

Current patch disagreement, a miss of a previously issued forecast, and task
value are distinct quantities. A stationary state can retain prediction error
and still make a poor decision. Retaining qualified activity with `step` can
reduce later repair work, but does not guarantee useful temporal memory.
Relations learned through `observe` persist and remain plastic; later learning
can overwrite them. Whole-brain qualification must include every eligible
coordinate, including declared observers.

## Longer credit and saved state

`Reinforcement` defaults to `credit_horizon=1`. Larger horizons are experimental
return-construction controls, not a recommendation for every long-term task
or evidence of learned planning. They can increase replay work. Use the
[reward-learning guide](LIVE.md) and [API reference](REFERENCE.md) for actual
execution acknowledgments and outcome ownership.

Save the complete application continuation, including body state and pending
outcomes. Brain checkpoints bind their implementation sources; the
[checkpoint contract](REFERENCE.md#checkpoints) specifies what can be restored.
There is no automatic checkpoint conversion or additional compatibility API.

## What the website demos establish

The existing demos remain versioned application evidence, not completed
reproductions on this release:

| Demo | Existing implementation | Evidence required for a new-version claim |
| --- | --- | --- |
| Atari learner | Python server and separately implemented browser engine | Source-pinned parity for the separate browser port, plus native gameplay and learning under the same information and work accounting. |
| Connect Four | Historical record evaluator with supplied game-tree search | Matched teaching data and search budget, free play and native query latency on the new implementation. |
| Amen | Archived 0.11 record-patch brain and its browser engine | Learned free continuations through the same sequencing rules and instruments, with listening assessment. |
| Patch World | Separate JavaScript patch-law implementation | Declared numerical/refusal semantics and native behavior under matched sensing, seeds and conserved mass. |

The library does not include these browser implementations. Updating its Python
package does not port their engines or convert their saved models.

The old Amen record patch contains 128 gated context channels and 8,192 record
cells. One input-only output population does not reproduce that mechanism or
its learned capacity. Preserve the original tracks and checkpoints as the
baseline. A page loading, a solver qualifying or a baseline verifier passing
does not establish equivalent musical quality or a recursive advantage.
