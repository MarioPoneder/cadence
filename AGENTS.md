# Building on Cadence

Read [the documentation index](docs/index.md), [numerical contracts](docs/contracts.md)
and [API reference](docs/api.md) before changing semantics.

## Goal and governing mechanism

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. **System 1 is the default:** the recovered continuing animal-like foundation
learns, remembers, imagines and acts. **System 2 is optional:** cortical observers
add recursive feedback and self-correction within that same neural graph. The
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

The 0.70 package restores the pre-reset foundation from 930ee807, including
memory and imagination mechanisms present in 0.11 and subsequent hardening.
The narrower population solver is preserved under
`cadence.experimental.equilibrium`, with its own docs, examples and tests. Its
frozen implementation keeps its own 0.62 identity inside package 0.70; preserve
its source bytes and historical receipts.
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
does not certify finite nudged eligibility or every training phase. Qualified
free solves may use numerical damping within their one declared budget, with
final residuals checked against the original model. Preserve the recovered
finite teaching law; numerical fallback is not System 2. Check each
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

Use live GitHub issues for individual missing, unproved or untested capabilities
and optimization. Preserve historical issue criteria and GPU work when splitting
broad owners; a closed issue label is not proof of its scientific claim. System 2
may ship as an optional implemented mechanism without a demonstrated advantage.
Do not require a completed theory of cognition or automatic reflection before
releasing System 1 and the optional feedback interface. Release gates remain
correctness, capability preservation, continuation and installed-artifact checks
on the final source; publication status belongs in the release record.
