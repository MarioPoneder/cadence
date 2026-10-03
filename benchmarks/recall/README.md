# Vanished-cue recall chamber

This bounded instrument asks whether a continuing `Brain.compose` can answer from
an earlier cue when all current probe inputs are identical. It tests one declared
neural-graph/trace recipe; it does not establish general recall, choose defaults,
or complete [the recall contract](https://github.com/muellerberndt/cadence/issues/84).
Read the [world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

One brain carries its activity and working trace through cue, delay and probe
observations, then straight into the next episode. Each group of rows receives
every cue once, with independently permuted cue assignments per episode and
identical delay/distractor inputs within the group. Row identity and distractor
identity therefore do not supply the answer. The event-time unit is one accepted
observation; these runs do not model variable physical time.

Only the current probe has a teacher. The chamber calls
`Learner.step(brain.stimulus(probe), cue)` with the live trace, followed by a free
`act(..., greedy=True)`. Earlier observations use greedy acts only. No reward is
invented, no actor eligibility is created, and no action awaits reward feedback.
A qualified refused lesson is charged and skipped; the following free action has
its own qualification. A refused action stops that model's continuing life.
There is no automatic reset, repeated reward or silent cold restart.

The selected recipe uses `modules=(32,)`, `lateral=0`, no episodic memory, the
specified working-trace amplitude/decay, and the learner configuration in
`make_brain`. It is **not the complete compose-default configuration**. Qualified
teaching uses 128 nudged sweeps per phase; `--finite` selects the separate finite
12-sweep teaching contract. Labels at the current probe supply no credit through
past observations. Evaluation has no teachers or parameter updates.

## Controls and continuation

Before any model runs, the instrument writes the exact training/test observations,
labels and shuffle permutations. Those arrays are reused across decays and both
models. Test streams use a separate random generator; running or reordering a
control cannot change later inputs.

At each evaluation probe, the continuing vanished-cue brain saves its **complete**
state. Four private branches load that same checkpoint:

- **intact:** reads its retained context normally;
- **erased:** zeros the working trace/last activity and marks rows cold before
  the probe, retaining the neural warm state;
- **shuffled:** transplants all three trace arrays from a row with another cue;
- **reset:** clears both transient neural state and trace, retaining learned
  parameters.

The intact original life alone continues. A separate restored branch must give
an identical answer and byte-equal checkpoint arrays after the probe, including
optimizer and random state. This checks the saved state immediately before the
probe; it is not a general mid-cue/replacement continuation guarantee.

The **appended-history comparator** is a second, independently trained brain
with the same initial parameters, episodes and lesson budget. During both its
training and evaluation, an external history buffer supplies the original cue
in extra probe coordinates. It is a useful acquisition control only to the
extent its own measured free recall succeeds. It is **not a mathematical upper
bound**, and the extra input is never given to the vanished-cue brain. Appending
untrained coordinates to the vanished-cue model would not be a competent control.

## Run and verify

Install the intended checkout first (`python -m pip install -e .` from the repo).
Use an unused output directory; existing attempts are never overwritten. This
small command checks the protocol, not a useful retention horizon:

```sh
python benchmarks/recall/vanished_cue.py --out /tmp/recall-smoke --cues 2 --streams 4 --modules 4 --lessons 2 --test-episodes 2 --seeds 0 --decays 0.8 --delays 1
python benchmarks/recall/vanished_cue.py --verify /tmp/recall-smoke
```

`protocol.json` and compressed input arrays precede all training. Initial,
trained, pre-probe and continued checkpoints retain model custody.
`rows.jsonl` preserves completed conditions if a run is interrupted. A completed
`summary.json` is a `cadence.Receipt`; its `body` holds raw trial labels/answers,
scores, all teacher/free-solve reports, configuration and artifact hashes. Its
source manifest binds copied benchmark and imported library Python sources.
The verifier checks the receipt digest, source bytes and retained artifact hashes;
this is custody verification, not an independent reproduction of learned behavior.
A partial attempt has no completed summary and makes no recall claim.

Work includes accepted and refused teacher phases (free, positive and negative
nudges), every free answer, control fork and continuation check. Sweeps,
row-sweeps, residual checks and trace-update counts are separate. Per-call wall
time excludes checkpoint I/O; run wall time includes it. These measurements are
not physical energy or a cross-machine efficiency comparison. Scores report
attempted, refused and unrun rows; refusals count as wrong among attempted probes.
Stopping before a probe does not fabricate a scored answer.

## What the existing exploration establishes

The [earlier issue comment](https://github.com/muellerberndt/cadence/issues/84#issuecomment-5967884580)
reports an exploratory appended-cue acquisition difference under selected trace
amplitudes, decays and reset interventions. Its numerical table lacks a frozen
reproduction driver and source-bound checkpoint receipt here. Old recall
summaries also exist, but lack source binding and used an earlier control and
accounting protocol. Neither provides a general finding that the default trace
prevents learning; they motivate a controlled comparison.

This corrected instrument has not established a new recall horizon. Freeze
behavioral acceptance before a campaign and retain its failures. Replacement,
partial/noisy cues, finite-capacity competition, irregular event timing,
pre-replacement continuation and an independently checked retention/separation
bound remain outside this initial chamber. The original source-specific records
remain historical evidence; rerunning a corrected protocol produces a new result.
