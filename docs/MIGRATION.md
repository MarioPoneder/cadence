# Coming from 0.19 and earlier

0.20.0 is a reset. The library was rebuilt from scratch around one
element (`CorticalColumn`) and one trainable composition (`Cortex`);
everything else - 37 modules, roughly 100 public exports, the old
docs, benchmarks and Lean certificates - was removed in one sweep
rather than deprecated piecemeal.

## Keeping old code running

The last full pre-reset state is tagged:

    git checkout legacy-0.19-final      # the exact pre-reset tree
    git checkout v0.19.0                # the 0.19.0 release commit

Pin an environment to the legacy library with
`pip install "cadence-net @ git+<repo-url>@legacy-0.19-final"`.
Nothing was rewritten in history; every removed module, test,
benchmark and certificate is in these tags.

## What replaces what

| 0.19 surface (examples) | 0.20 |
| --- | --- |
| `BeliefPatch`, `BeliefReadback`, `PatchNet`, `RecordPatchNet`, `Brain`, `GenericBrain`, `Region`, `Connectome` | `Cortex` (hierarchy of column banks; readback is built into every level) |
| `NeuronModel`, `FastSynapses`, `SynapticMemory`, plasticity/temporal modules | `CorticalColumn` evidence algebra + `Cortex` dynamics keywords |
| checkpoints (`save`/`load`) | `snapshot()` / `restore()` on both classes (JSON, validated) |
| `calibrate_bias`, `preflight`, instrument modules | `calibrate` + `wire` (controllability/drive probe) |
| certificates, EP structure, backends (numpy/torch/mlx) | removed; the library is pure stdlib. Exact guarantees live in the element's rejection contracts; qualification evidence lives with the candidate battery in the development workspace |

Old checkpoints and receipts are not convertible: 0.20 state schemas
are `cortical-column-state/1` and `cortex-state/1`, and pre-reset
artifacts keep their meaning only against the legacy tags.

## Why

One element that already carries retention, adaptation, uncertainty
readback, transactional custody and exact federation makes the old
per-capability class zoo redundant, and a smaller surface is easier to
qualify honestly. The candidate battery, its receipts, and the staged
migration inventory that led here live in the development workspace;
this repository carries the minimum that is needed to set up and train
a Cortex, and nothing else.
