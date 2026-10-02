# Experimental features and the System 2 boundary

**`0.60.0.dev1` recommends ordinary flat or deep settling networks for
applications.** Build them with `Cortex.column`, teach through `bootstrap` or
`observe`, and use `step` to retain qualified live activity. Add ordinary depth
when it improves the task. No extra mode, critic or attention scheduler is
required. See the [quickstart](QUICKSTART.md) and [brain design](BRAIN_DESIGN.md).

## System 2 warning

**Recursive self-observation is experimental and can slow routine responses.**
Calling `Cortex.observer` explicitly adds state-and-error contacts. These
participate in every synchronous whole-brain solve. The public runtime does
not put an observer to sleep during familiar behavior, wake it only on
surprise, or let it run at an independent speed. An observer's presence does
not establish useful correction or a benefit over capable ordinary layers.

Keep ordinary learned routine (System 1) dominant. For most tasks, use flat
or ordinary deep layers without self-observation. Ordinary depth remains a
coupled settling network and can require additional work; measure held-out
quality and complete decision latency before adding capacity. A small
settlement residual establishes numerical qualification, not success at the
task. Repeated qualified `step` calls may need zero repair sweeps while still
performing full evaluation and qualification.

The [experimental layout recipes](VARIANTS.md) keep observers available for
deliberate comparisons. Compare against a capable ordinary model with the same
available information, disclosed capacity and complete learning/inference cost.
Preserve failures and check old skills after any new learning. Do not silently
skip an explicitly declared observer or weaken whole-brain qualification to
make a latency result look better.

The intended System 2 cycle remains: routine → surprise or unmet longer-term
goal → useful correction → retained, inexpensive routine. Predictable failure
must still matter. The private timing and learning experiments linked from
[issue 72](https://github.com/muellerberndt/cadence/issues/72) do not yet
demonstrate that complete cycle or a measured recursive advantage. They are
not part of this release's public runtime. Stable 0.60.0 remains subject to
those capability gates; this is a development prerelease.

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
reproductions on this development release:

| Demo | Existing implementation | Evidence required for a new-version claim |
| --- | --- | --- |
| Atari learner | Python server and separately implemented browser engine | Source-pinned numerical parity plus native gameplay and learning under the same information and work accounting. |
| Amen | Archived 0.11 record-patch brain and its browser engine | Learned free continuations through the same sequencing rules and instruments, with listening assessment. |
| Patch World | Separate JavaScript patch-law implementation | Declared numerical/refusal semantics and native behavior under matched sensing, seeds and conserved mass. |

The old Amen record patch contains 128 gated context channels and 8,192 record
cells. One input-only output population does not reproduce that mechanism or
its learned capacity. Preserve the original tracks and checkpoints as the
baseline. A page loading, a solver qualifying or a baseline verifier passing
does not establish equivalent musical quality or a recursive advantage.
