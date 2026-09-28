# Choosing context structure and observer height

There are three separate choices: how observations become contexts, how coarse
contexts inform fine ones, and how many uncertainty-observer stages each
column contains. They change model capacity, computation and inductive
assumptions in different ways.

## General sensor and prediction models

Start with `Cortex.from_dimensions` for finite numeric vectors. Bounds and bins
supply a transparent quantized representation. Increase resolution only when
the coarser model merges contexts whose outputs need to differ. Finer grids
allocate more context columns and receive less evidence per context.

Use `BinnedFeatures` or explicit feature maps when you know which coordinates
belong together. Feature maps are part of the model: adding a relative position,
removing a sensor, or selecting coordinates changes what the learner can know.
Report those changes in a comparison.

| Choice | Effect | Useful control |
| --- | --- | --- |
| One context level | Independent predictions for the declared fine contexts. | Flat reference with the same fine map. |
| Additional coarse levels | Coarse predictions supply directed priors to finer contexts. | `coupling=0` or the declared prior-port cut. |
| More context coordinates or bins | More distinctions and potentially many more columns. | Equal memory and observation budgets; report occupied columns. |
| `height=1` | A belief and its reciprocal precision observer. | Default column mechanism. |
| `height>1` | Adds live rate-observer stages with reciprocal feedback inside each column. | Same maps, witness stream and learning configuration at height 1. |
| Learning disabled | Keeps previously retained evidence fixed. | Frozen-memory control; it is not necessarily a uniform-random policy. |

Coarse-to-fine priors are not reciprocal observation across levels. Conversely,
the observer stages controlled by `height` do exchange live messages in both
directions. More of either structure is not automatically better.

## Automatic wiring from an environment

`calibrate` probes a supplied environment with fixed actions and measures
coordinate ranges, action-conditioned mean differences and within-run variation.
`wire` selects a requested number of ranked coordinates and creates fine and
coarse grid contexts over them. It works with finite numeric vectors in declared
or measured units; no special byte format is required.

Depth zero uses only the fine grid. Positive depth includes a global context;
additional `wire` levels coarsen the primary ranked coordinate. In contrast,
`grid` coarsens every declared coordinate and needs no environment probe.
See [Reference](REFERENCE.md) for exact resolution and depth parameters.

Calibration measures association with forced actions and observed variation;
it does not prove causal controllability or identify a biological body.
Calibration consumes interactions and must be included in a resource budget.
Use identical calibration data when comparing model structures.

## A useful comparison

Hold observations, targets, event order and evaluation cases fixed. Compare
height separately from context hierarchy, and include a flat, frozen-memory
and conventional predictor baseline. Record rejected admissions and all
settling work, calibration interactions, context count and storage. Test on
held-out contexts or future observations before attributing improved training
fit to better generalization.
