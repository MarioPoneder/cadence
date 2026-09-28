# Reference

Four public names: `CorticalColumn`, `Cortex`, `calibrate`, `wire`
(plus `settle` and `solve_cluster_forest` for direct element use).

## Cortex

```python
Cortex(n_actions, feature_maps, *, decay=0.99, discount=0.97,
       optimism=0.5, epsilon=0.02, coupling=1.0, target_bound=8.0,
       settle_budget=256, seed=0, learning_enabled=True, height=1)

Cortex.for_environment(env_factory, n_actions=None, *, depth=2, width=1,
                       calibration_steps=400, seed=0, **overrides)
```

`for_environment` probes `action_space.n` when `n_actions` is omitted,
calibrates, wires and constructs; the calibration lands on
`cortex.calibration`. To compare cortices on identical wiring, call
`calibrate` once and pass `wire(calibration, depth=..., width=...)`.

| Parameter | Default | Meaning; when to change |
| --- | --- | --- |
| `n_actions` | required | Discrete actions the body offers. |
| `feature_maps` | required | Context functions, finest first; last usually `lambda o: ()`. From `wire` for byte observations. A map may declare `cells` to enable witness-mass weighting (`wire` does). |
| `depth` | 2 | Hidden levels above the finest: 0 flat, 1 global only, 2 adds a 4-bin coarsening, 3 adds 16- and 4-bin. Explicit `ladder=(...)` overrides. |
| `width` | 1 | Calibrated bytes the finest level reads (top-N controllable + top-N drive). Raise when one byte pair cannot express the task. |
| `decay` | 0.99 | Column leak per admitted unit witness mass; window 1/(1-decay). Lower for fast drift; 1.0 stops adaptation. |
| `discount` | 0.97 | TD horizon 1/(1-discount) steps. |
| `optimism` | 0.5 | Novelty weight: acting scores mean + optimism*sqrt(novelty); targets add it scaled by (1-discount). Too high masks value gradients. |
| `epsilon` | 0.02 | Residual random actions; novelty does the directed exploring. |
| `coupling` | 1.0 | Coarse-as-prior strength (pseudo-evidence units); 0 severs the hierarchy. |
| `target_bound` | 8.0 | Declared value scale; targets clip here. Not a knob. |
| `settle_budget` | 256 | Sweeps per settle (the element's cap); unsettled states are rejected, never used. Tall columns with heavy evidence need the full cap. |
| `seed` | 0 | Only randomness source (tie-breaks, epsilon). |
| `learning_enabled` | True | False = frozen-memory control at the same interface. |
| `height` | 1 | Observer stages per column (vertical microcircuit). Height 1 is the validated learner; taller stacks are mechanism-ready, learning-unproven (docs/VARIANTS.md). |

Methods:

- `act(observation) -> int` - settled optimistic choice.
- `learn(observation, action, reward, next_observation, terminal)` -
  buffers one witnessed transition; a terminal transition flushes.
- `end_episode() -> int` - admits the buffered episode backward, each
  transition exactly once; call it at every episode end.
- `value(observation, action) -> dict` - settled `mean`, fused
  `variance`, `novelty` (per-level evidence-only uncertainty),
  `qualified`, `sweeps`, `residual`.
- `snapshot() -> str` / `restore(text)` - full continuation custody;
  schema `cortex-state/1`; malformed or inadmissible checkpoints raise.
- `stats() -> dict` - counters, columns per level, level weights.

Controls come from the same class: uniform-random is
`Cortex(..., learning_enabled=False, epsilon=1.0)`, frozen-memory is
`Cortex(..., learning_enabled=False)`.

## CorticalColumn

```python
CorticalColumn(height=1)
```

`height` = observer stages on the belief: 1 is the qualified element
exactly (identical numbers, element checkpoints restore); 2 adds a
rate hyper-observer reading the precision observer live; H chains
further anchored rate observers. The frozen candidate class remains
at `cadence.element.CorticalColumn`.

- `observe(event, value, kind='witness', budget=256)` - ordered
  transactional admission; duplicates flagged, conflicts/out-of-order/
  derived/over-capacity rejected with no state change; integer domain
  |value| <= 8, capacity 24, checkpoint <= 4 KiB.
- `query()` - settled `answer`, `variance`, `precision`, `qualified`,
  `residual`; queries acquire no witnesses.
- `snapshot()` / `restore(text)` - schema `cortical-column-state/1`.
- `observer_intervention()` - live readback/feedback perturbation
  traces and the two declared lesions.
- `solve_factors(factors, edges)` - exact Fraction beliefs on certified
  cluster forests; cycles, running-intersection violations, float
  potentials and zero-support problems raise `ValueError`.

## calibrate / wire

```python
calibrate(env_factory, n_actions, *, steps=400, seed=0) -> dict
wire(calibration, *, depth=2, width=1, ladder=None) -> tuple[maps]
```

`calibrate` holds each action in turn and ranks every observation byte
by controllability (moved by the body's own actions) and drive (moved
by the world). `wire` builds the level maps; see docs/VARIANTS.md for
the depth/width menu and the evidence behind the defaults.
