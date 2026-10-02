# Building on Cadence

Read [the documentation index](docs/index.md), [numerical contracts](docs/contracts.md)
and [API reference](docs/api.md) before changing semantics.

## Goal and mechanism

Cadence aims to build a simulated human-like brain from simplified biological
mechanisms. **System 1 is the default:** a continuing animal-like brain with
memory, plasticity, private imagination and action. **System 2 is optional:**
observing cortical regions add recursive feedback in that same neural graph.
Base modules can already be deep and specialized; ordinary depth is not recursive
observation.

Bounded observer-like regions have local state, ports, readback, records and
plastic relationships. Local disagreement repair seeks a coherent shared state.
A retained trace may be a held boundary during a present solve; a live coordinate
must still qualify. Do not freeze unresolved live state or weaken qualification
to manufacture a whole-brain answer.

## Keep the main interface simple

Use `Brain.compose` for a continuing brain. `modules` defines reciprocal
base regions; optional `observers` read and return to the same graph. Working
trace and consolidating associative memory are included. Add a new abstraction
only for a demonstrated general need; changing an interface must preserve the
behavior it serves.

`step` receives a current observation and the preceding executed action's actual
outcome. `teacher` labels the current observation. Preserve stream identity,
event order and pending feedback. `imagine` evaluates supplied observations
using a private trace and read-only durable memory; it does not predict the
world's transitions or turn predictions into witnessed experience.

Advanced APIs have explicit contracts: `TemporalPatchNet.plan` uses a learned
world model; `TemporalMemory` protects selected responses at finite capacity;
record patches combine context, learned relations and writable records. The
population solver under `cadence.experimental.equilibrium` provides exact
state-and-error readback. Do not transfer its mathematical guarantees to a
neural-graph or record operation without an actual correspondence.

Graph and temporal learners use free/nudged equilibrium contrasts. Record and
belief models also use explicit adjoints and record writes. State those
mathematics accurately. A record scan is not a joint graph-equilibrium certificate.

## Numerical and behavioral contracts

`Brain.act`, `predict` and `accuracy` check the full state equations through
`NeuralGraph.equilibrate`, including cached activity and optional observers. Refused
`act` calls preserve live state, memory, randomness and pending feedback. If
`step` learns a real outcome before its next action refuses, that learning stays:
retry `act`, not the reward. Finite nudged eligibility and teaching phases retain
their own contracts.

Qualified free solves may use numerical damping within their one declared budget,
then check the undamped model's residual. This does not change the live model or
finite teaching law, and it is not System 2. A numerical qualification does not
prove correctness about the world, reliable recall or a cognitive advantage.

Memory and imagination are implemented capabilities to preserve. Test actual
acquisition, free recall, interference, private-state isolation and saved
continuation when changing them. Finite storage and supplied protection or
salience do not establish general lifelong retention.

## Review and verification

- Prefer fewer concepts, direct interfaces and working defaults. Simplification
  must preserve supported memory, learning, imagination and action behavior.
- Keep signatures, mutation rules, units and timing aligned with code. Provide
  runnable examples and useful refusal/retry behavior.
- Run [contributing checks](CONTRIBUTING.md), executable documentation and local
  links. Test installed wheel/sdist behavior as well as the source checkout.
- Numerical changes need independent reference, derivative or adversarial checks.
  Measure behavior after teaching with answers free, and count query, training,
  replay, planning and refused work. Preserve failed runs and evidence.
- Changes to the population solver need `tests/equilibrium`; default neural
  changes need their own tests. Optional backends must preserve the applicable
  qualification and continuation rules.

Cadence is experimental: backward compatibility is not a design requirement.
Save/load must still preserve the current supported model's complete continuation.
Keep GPL-3.0 and required source attribution. NumPy is required; optional backends
must not become hidden default dependencies.

Use individual GitHub issues for missing or untested capabilities and optimization.
Optional System 2 can ship without a demonstrated task advantage. Release checks
cover correctness, capability preservation, continuation and installed artifacts;
completed general cognition is not a release gate.
