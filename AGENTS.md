# Building on Cadence

Read [the documentation index](docs/index.md), [numerical contracts](docs/contracts.md)
and [API reference](docs/api.md) before changing semantics.

## Goal and governing mechanism

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. A continuing animal-like foundation learns, remembers, imagines and
acts; optional cortical columns add recursive observation and feedback. The
foundation can already be deep and modular. Do not equate ordinary depth with
recursive observation or require a separate application controller for each region.

The central mechanism is local disagreement repair through bounded state,
ports, readback, records and plastic relationships. Observers and observed
regions must participate in the equilibrium claimed for their connected graph.
A historical trace may be a fixed boundary for a present solve; distinguish it
from a live coordinate that must still qualify. Never freeze unresolved live
state or weaken a certificate to claim a whole-brain answer.

Different restored numerical implementations have different contracts.
Graph and temporal learners use free/nudged equilibrium contrasts; record and
belief models also use explicit adjoints and record writes. Preserve those
mathematics honestly. A successful record scan is not a certificate for a
joint graph equilibrium, and an imagined outcome is not an actual witness.

## Preserve working capabilities

The default package restores the pre-reset foundation from 930ee807, including
memory and imagination mechanisms present in 0.11 and subsequent hardening.
The narrower population solver is preserved under
`cadence.experimental.equilibrium`, with its own docs, examples and tests.
Do not let experimental construction restrictions silently remove a supported
foundation mechanism or reinterpret an old checkpoint.

Before removing or replacing a mechanism:

1. Identify its actual consumers, source-bound checkpoints and behavioral tests.
2. Preserve the original source/evidence and every relevant failed comparison.
3. Make the replacement pass acquisition, free behavior, memory, continuation
   and complete-cost checks that the capability requires.
4. State any remaining differences. A renamed API, green solver tests or fewer
   public classes cannot substitute for recovered behavior.

Memory is implemented. `Trace`/`Afterglow` retain temporal input;
`SynapticMemory` has fast and persistent associations; record patches keep
context, learned relations and a record store; temporal models retain a boundary
and support private imagination, action planning and response protection.
These mechanisms have bounded capacities and specific learning rules. Neither
their existence nor their removal establishes general lifelong retention.

`GenericBrain.compose` is the simple entry: base `modules` form reciprocal
paths and optional `observers` read and return to that same graph. Working trace
and consolidating memory are included by default. `GenericBrain.build` retains
its original construction contract. `GenericBrain.imagine` evaluates supplied
hypothetical observations with a private trace and read-only durable memory;
it does not predict environmental transitions. Use `TemporalPatchNet.plan`
with a learned world model for action-consequence planning.

`GenericBrain` owns a continuing observation/action/outcome loop. Reward concerns
the preceding executed action; demonstration labels concern the current input.
Preserve event order, stream identity, actual-outcome custody and complete saved
continuation. Hypothetical queries must not teach from their own predictions.
`GenericBrain.act`, `predict` and `accuracy` qualify the full state equations
through `Brain.equilibrate`; cached activity is checked afresh. A refused `act`
preserves live state, memory, randomness and pending feedback. If `step` has
learned an actual outcome before the next action refuses, retain that learning
and retry `act` rather than resubmitting the reward. Free-answer qualification
does not certify finite nudged eligibility or every training phase. Check each
API's contract rather than transferring experimental result fields or source
hash rules to another model.

## Three review gates

- **Minimalism:** reuse a working primitive and remove demonstrated duplication.
  Added complexity needs a general measured purpose. Simplicity is not permission
  to discard memory, imagination or plasticity because a smaller model lacks them.
- **User-friendliness:** one clear body interface, explicit units and timing,
  useful defaults and complete learning/save/restore examples. Columns extend
  the base rather than forcing users to assemble disconnected brains.
- **Agent-friendliness:** documented signatures and mutation rules match code;
  state and topology are inspectable; numerical assumptions, refusal, retries,
  source identities and checkpoint compatibility are explicit and tested.

The default runtime requires NumPy. Optional PyTorch, MLX, Numba and SciPy paths
must retain the appropriate numerical/admission contract. Do not advertise the
restored foundation as dependency-free. Keep current GPL-3.0 licensing and
historical source attribution; do not restore an obsolete top-level license.

## Verification and evidence

Run the checks in [CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need
independent derivative/reference or adversarial checks as appropriate. Use
bounded local tests first; charge all learning, query, replay, planning and
refused work. Preserve held-out cases, failed runs and actual behavioral limits.
Learning accuracy is assessed after teaching with outputs free.

Changes to the default runtime need its tests. Changes to the preserved
population solver need `tests/equilibrium` as well. Executable documentation and
local links must pass. Test installed wheel/sdist behavior rather than relying
only on a source checkout. Do not repin immutable receipts to make them pass.

Amen, Connect Four and Atari are separate versioned application baselines.
Their original engines, supplied search/body logic and checkpoints must stay
identified. Static browser parity, a library checkpoint load and new native
behavior are different checks; report which actually ran.

Short-term memory is tracked in #84, long-term memory/plasticity including the
historical reversal regression in #85, and optional recursive integration in #86.
Earlier closed issue metadata is not proof of their old capability contracts.
Release only after capability, continuation and installed-artifact checks pass
on the final source; publication status belongs in the release record.
