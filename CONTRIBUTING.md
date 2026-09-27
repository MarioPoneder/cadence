# Contributing

Issues and pull requests are welcome in this repository and in
[cadence-examples](https://github.com/muellerberndt/cadence-examples). This page says
how to set up, what the checks are, and what a change needs.

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

The Lean library under `lean/` is checked separately (`python3 lean/check.py` with the
pinned toolchain; see [lean/README.md](lean/README.md)).

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
| `lean/` | the Lean proofs and their audit |

## Examples

The atlas, browser renderer and local quickstart pages live in
[`cadence-examples/viewer`](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
and [`quickstart`](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart).
Their tests run in the examples repository against an installed library:
`python -m pytest -q viewer quickstart`. The core package no longer supplies
`cadence.atlas`, `cadence.demo` or the `cadence-demo` executable.

Worked applications live in [cadence-examples](https://github.com/muellerberndt/cadence-examples),
one directory each with a README, a static page that runs the brain in the browser, the
receipts behind every stated number and a `verify.py` that recomputes them. An example
pins the library release its checks were run against. A new example follows that layout
and opens its README with the card that repository's
[contributing section](https://github.com/muellerberndt/cadence-examples#contributing)
defines: name, author, description, Cadence version, hardware used for the initial
training, the library features it showcases, the problems met while building it, the
hosted URL, its receipts and checks, its data and rights, and what is work in progress.
Half-working examples with a filled card are welcome; every one is data.

## Releases

Releases are tagged `vX.Y.Z`, published to PyPI as `cadence-net`, and listed in
`CHANGELOG.md`. Pin a release or a commit for reproducible work.
