# Changelog

## Unreleased

- Add opt-in qualified graph learning. Free and nudged phases must meet the
  full equation tolerance before a contrast changes parameters; refused lessons
  preserve parameters and optimizer history and report attempted work.
- Allow bounded numerical damping across several integration steps, checking
  the original equations and counting every sweep and residual check. The
  finite teaching configuration remains available as the comparison.
- Make gain calibration honor qualified learning: reject unqualified candidates
  and preserve the graph and optimizer if none pass. Search a bounded grid
  around the current gain by default, retain explicit grids exactly, and expose
  all candidate solve work through `Learner.last_calibration`.
- Expose motor lateral wiring through `Brain.compose(lateral=...)`, retaining
  the default per-pair weight of -0.5. Add opt-in full-equation qualification
  and work reports to `calibrate_bias`, including the final candidate's observed
  means and target gaps. Calibration remains an operating-point heuristic.
- Detect stalled free solves so bounded damping can try a smaller numerical
  step sooner, while every answer still meets the original full equations.
- Keep reward eligibility duration explicit through
  `ActorCriticConfig.eligibility_steps`; the Brain default stays at up to 12 finite
  nudged steps independently of supervised teaching budgets.
- Restore memories and pending feedback when reward bootstrap qualification
  fails, allowing the actual outcome to be retried once.
- Expose supervised phase and presentation costs through `Brain.last_learning`
  under `demonstration_` keys, including refused teaching attempts.
- Score `Brain.fit` epochs through qualified public predictions; a refused
  score preserves the teaching updates already accepted in that epoch.
- Clarify finite and qualified teaching, memory-free graph predictions,
  memory-aware actions and separate retention tests for traces, graph
  plasticity and consolidated associations.
- Add a runnable `Brain.compose` example with actual feedback, demonstrations,
  free recall and saved pending-feedback continuation. Restore the recorded
  0.61.0 release and 0.62.0 development history.
- Distinguish released and development APIs, standalone and composed defaults,
  device learning paths, current application demos and archived examples.

## 0.70.0 — 2026-10-02

- Make System 1 the default continuing brain, with working trace, fast and
  persistent associative memory, plasticity, action and private imagination.
- Provide `Brain.compose` for reciprocal base modules and optional
  System 2 observer regions within the same neural graph.
- Add `Brain.imagine` for private responses to supplied hypothetical
  observations. Learned environmental consequences and action planning use
  the temporal-model API.
- Qualify actions and independent predictions against the full state equations.
  Refusal preserves action state and pending feedback; real outcomes learned
  before a subsequent refusal remain learned.
- Use bounded numerical damping when a free solve needs it, then check the
  original model's residual. The total budget and finite teaching rule stay fixed.
- Provide event records and consolidation, learned temporal paths, continuous
  action planning and finite response protection through advanced APIs.
- Keep exact state-and-error feedback available in the advanced population solver.
- Simplify guides around current usage. This is experimental software; backward
  compatibility is not a design requirement. NumPy is required, with optional
  acceleration backends.

## 0.62.0 development revision — 2026-10-02

This entry records the development sources at
[`1f9daac`](https://github.com/muellerberndt/cadence/commit/1f9daac).
The recovered work became release 0.70.0; 0.62.0 was not published as a
GitHub release or on PyPI. Names below describe that development revision.

- Restore the capable pre-reset foundation from 930ee807: continuing
  `GenericBrain` interaction, `Trace`/`Afterglow`, consolidating
  `SynapticMemory`, record patches and sleep, temporal learning, private
  imagination, action planning and response protection. Preserve subsequent
  numerical, continuation and recursive-wiring hardening.
- Add `GenericBrain.compose` as a direct modular entry with working trace and
  consolidating memory, optional reciprocal observer regions, and the existing
  continuing interaction interface. Add private `GenericBrain.imagine` over
  supplied hypothetical observations; environment prediction remains the
  separate learned temporal-model contract.
- Qualify `GenericBrain.act`, `predict` and `accuracy` against the full state
  equations. Exhausted action repair preserves live state and pending feedback;
  consumed real outcomes stay learned if a following action refuses. Keep finite
  eligibility/training phases distinct from this free-answer qualification.
- Keep cortical observation optional. The foundation can already be deep and
  modular; observer feedback extends the shared graph rather than replacing
  working memory and learning with a narrower model.
- Preserve the newer state-and-error solver under
  `cadence.experimental.equilibrium`, with its own guides, examples and tests.
  Its sparse patch-connectivity checks and same-call stationary-evaluation
  optimization remain available there, without changing the restored APIs.
- Rewrite the entry guides around the biological-brain objective, working
  mechanisms and actual application source identities. The default package
  requires NumPy. Keep current GPL-3.0 licensing and historical attribution.
- Preserve original Amen, Connect Four and Atari checkpoints and browser
  engines. Library recovery, checkpoint parity and native application behavior
  require separate verification; no old receipt is silently promoted.
- Recover the capable foundation before releasing the narrower candidate as the
  default. Its separate numerical, CI and package evidence stays source-bound;
  the restored package requires its own verification.

## 0.61.0 — 2026-10-02

- Restore the principle as an enforced default: patches repair local
  disagreement to reach a coherent brain state, and further repair is driven by
  that state's mismatch with reality. `Cortex.build()` now refuses a layout in
  which a population settles with no other population, and a layout in which a
  group of populations settles apart from the rest. Every population must read
  another population's states or errors, or be read by one, and those reads
  must join all populations into one connected system; an unread sensors-only
  population or a disconnected group raises `ValueError` naming it. The
  smallest brain is two populations.
- Remove the input-only "flat" layout from the README, quickstart, layout and
  design guides, agent guides, examples and test fixtures. The layout example
  defaults to a two-population brain (`small`), with `deep` and the explicit
  `recursive` experiment. The query-cost and temporal-credit examples no longer
  build an input-only arm; their recorded receipts stay as recorded.
- Lead the README and the contributor guide with the main hypothesis and the
  simplicity premise, and contrast settlement with feed-forward backpropagation.
- The repair law, energy, qualification tolerance and admission contract are
  unchanged. `cortex.py` changed, so snapshots bind to this release's sources;
  snapshots saved by `0.60.0` load only in `0.60.0`.
- Error-reading observers remain experimental; this release still claims no
  automatic System 2, retained useful recursive correction or reproduced
  musical quality.

Earlier entries remain in the [source changelog before the guide simplification](https://github.com/muellerberndt/cadence/blob/1f9daac/CHANGELOG.md).
Version numbers 0.63.0–0.69.0 were not published; they are not missing release entries.
