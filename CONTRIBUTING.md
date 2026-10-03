# Contributing

Issues and pull requests are welcome in this repository and in
[cadence-demos](https://github.com/muellerberndt/cadence-demos). This page says
how to set up, what the checks are, and what a change needs.

## Preserve the capable foundation

The goal is a simulated human-like brain built from simplified biological
mechanisms. Memory, plasticity, imagination and continuing interaction are
working parts of the default System 1 foundation. Optional System 2 adds
recursive cortical feedback to that brain, whose base can already be deep and
modular. The optional mechanism can ship without a proven task advantage; do
not turn research on its benefit into a release gate.

A replacement must preserve demonstrated behavior and saved continuation before
removing its predecessor. Keep source-bound evidence, original application
checkpoints and failed comparisons. Smaller code or a newer solver is not a
capability-preservation test. State the actual learning rule and the scope of
its equilibrium guarantee; record scans, temporal repair and graph settlement
must not inherit one another's claims.

## Set up

```bash
git clone git@github.com:muellerberndt/cadence.git
cd cadence
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`[dev]` installs pytest, ruff, mypy, torch, Numba and SciPy. Python 3.11, 3.12 and
3.13 are supported on Linux, macOS and Windows.

## Checks

Every push runs these; run them before opening a pull request:

```bash
ruff check src tests
mypy
pytest -q
```

The test suite includes `tests/test_documentation.py`, which executes every Python block
of the listed guides in order and checks that every local link and anchor in `README.md`
and `docs/` resolves. A change to a guide's code is a change to a test. The minimal-install
job builds the wheel with NumPy alone and runs the quickstart, the record patch, the
temporal, belief, steering and build guides without optional backends.

Formal proofs are maintained in the canonical
[Cadence flagship Lean library](https://github.com/FloatingPragma/oph-meta/blob/main/cadence-flagship/lean/README.md),
with its pinned toolchain and checker. They are not duplicated in this runtime
package; a numerical API must state which theorem assumptions it satisfies.

### Short local feedback

Run the contracts affected by a change while iterating. For teaching, phase
qualification and actual-feedback transactions:

```bash
python -m pytest -q tests/test_qualified_learning.py tests/test_damping_stagnation.py tests/test_learning_reports.py tests/test_feedback_transaction.py
```

For gain selection, bias calibration, backend continuation and motor wiring:

```bash
python -m pytest -q tests/test_learner_calibration.py tests/test_calibration_backends.py tests/test_bias_qualification.py tests/test_composed_lateral.py
```

Run the default foundation and preserved population solver separately to locate
slow cases. These two commands together retain the complete test inventory:

```bash
python -m pytest -q tests --ignore=tests/equilibrium --durations=10
python -m pytest -q tests/equilibrium --durations=10
```

A fresh NumPy-only environment avoids optional accelerator startup and runs
the required-dependency paths. Use a separate environment; installing fewer
packages into an existing development environment does not remove its backends.
The environment below is inside the ignored `.venv/` directory:

```bash
python -m venv .venv/numpy-only
.venv/numpy-only/bin/python -m pip install -e . pytest
.venv/numpy-only/bin/python -m pytest -q tests --ignore=tests/equilibrium -rs --durations=10
```

Focused runs and a NumPy-only run provide partial coverage. Report optional
backend skips explicitly; they do not establish accelerator parity. Before
publication, retain the full suite and installed-package checks in the declared
environments. Run one substantial local QA process at a time on a busy machine.

## What a change needs

- **A test.** A new operation gets a test of its contract, and a numerical claim gets a
  finite-difference or reference check where one exists (`tests/test_equilibrium.py`,
  `tests/test_belief.py` are the pattern).
- **A line in `CHANGELOG.md`** under `Unreleased`, saying what changed and why.
- **A guide.** A public name is described in `docs/api.md`, and a new capability gets a
  section with a runnable snippet in the guide that owns it; add that guide to
  `tests/test_documentation.py` so the snippet keeps running.
- **Contracts kept.** Imagination does not teach or change live activity;
  cost counters may count attempted computation. Invalid calls and failed
  transactions cannot partially install parameters or records. Qualification,
  continuation and device portability follow the owning API's
  [contract](docs/contracts.md); do not transfer a graph guarantee to a finite
  repair or temporal model. The audit tests (`tests/test_audit_*.py`) check
  specific cases and backends.
- **Plain prose.** State what an operation does and what it does not establish, in the
  register of the existing guides.

## Layout

| Path | What lives there |
| --- | --- |
| `src/cadence/` | the numerical library: `brain.py`, `learning.py`, `genome.py`, `regions.py` (the settling brain); `temporal.py`, `planning.py`, `temporal_memory.py`; `record_patch.py`, `records.py`, `record_stack.py`, `record_ports.py`, `ports.py`; `belief.py`, `belief_torch.py`; `steering.py`, `life.py`, `instruments.py` (composition) |
| `tests/` | unit, numerical, integration and documentation contract tests |
| `docs/` | the guides; `docs/index.md` is the map |

## Examples

The archived atlas, browser renderer and local quickstart pages live in
[`cadence-examples/viewer`](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
and [`quickstart`](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart).
Their tests run in that archive's declared environment and library version:
`python -m pytest -q viewer quickstart`. The core package no longer supplies
`cadence.atlas`, `cadence.demo` or the `cadence-demo` executable.

Current applications live in [cadence-demos](https://github.com/muellerberndt/cadence-demos).
Each demo's README names its library version, mechanisms, setup and behavioral scope.
Keep receipts and reproduction checks behind measured claims, and state data rights
and known limits. The separate research archive retains its own example layout and
library pins; use those when reproducing archived results.

The preserved population solver lives in `cadence.experimental.equilibrium`,
with guides in `docs/equilibrium`, examples in `examples/equilibrium` and tests
in `tests/equilibrium`. Test it separately from the default foundation when
changing either API or numerical rule. The default runtime requires NumPy;
optional backends do not make it dependency-free.

Keep [GPL-3.0](LICENSE) as the current package license and preserve historical
attribution and license notices with their source.

## Releases

Releases are tagged `vX.Y.Z`, published to PyPI as `cadence-net`, and listed in
`CHANGELOG.md`. Pin a release or a commit for reproducible work.

Before publication, complete
source-bound foundation, consumer, CI and installed-artifact checks before
publication; record any missing application parity rather than inferring it
from a successful library suite.
