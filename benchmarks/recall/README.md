# Vanished-cue recall chamber

Roadmap row 02 (issue 84): does a continuing composed brain carry a cue through a delay in
its own state, and for how many steps? One stream of a life shows a cue picture for one step,
then `delay` steps of a blank or distracting picture, then a probe picture that is the same
for every stream. The right action at the probe is the cue the stream saw. The lesson happens
at the probe only, on the drive that includes the living trace (the call `step` makes for a
demonstration), and the life goes on without resets: the next cue follows the last probe, and
the trace carries over as it would in an animal.

Free recall is measured afterwards on fresh episodes of the same life with learning off
(greedy acts only). Three controls run on the same episodes from the same carried trace:

- **erased**: the working trace is cleared before the probe;
- **shuffled**: the trace rows are permuted across streams before the probe;
- **appended**: the cue is appended to the probe input itself, the equal-information upper bound.

```sh
python benchmarks/recall/vanished_cue.py --out /tmp/recall --amplitude 1.0 --decays 0.5 0.8 0.9 --delays 1 2 4 8 16
```

`summary.json` holds recall by decay and delay for the brain and the controls, refusals, and
the sweeps charged per lesson. This is a bounded CPU instrument on a brain of 32 processing
neurons and four cues; it establishes a measured limit of the composed trace, not a retention
law.

## What the chamber found on 0.71.1

Measured on cadence 0.71.1 with a direct probe of the lesson path (the same brain, cue
appended at the probe so that the mapping is trivially learnable; 60 lessons of 16 streams;
agreement of the free answer per block of 15 lessons; chance 0.25):

| Life | Trace amplitude, decay | Agreement by block |
| --- | --- | --- |
| Reset at every episode | 3.0, 0.2 (compose defaults) | 0.69, 0.94, 0.93, 0.88 |
| Continuing, no reset | 3.0, 0.2 (compose defaults) | 0.26, 0.27, 0.26, 0.31 |
| Continuing, trace cleared at each episode, live state kept | 3.0, 0.2 | 0.69, 0.98, 0.90, 0.90 |
| Continuing, live state cleared at each episode, trace kept | 3.0, 0.2 | 0.26, 0.28, 0.30, 0.26 |
| Continuing | 1.0, 0.2 | 0.40, 0.68, 0.72, 0.91 |
| Continuing | 1.0, 0.8 | 0.72, 0.95, 0.97, 0.97 |
| Continuing | 0.3, 0.2 | 0.92, 0.78, 0.47, 0.40 |
| Continuing, no trace | 0.0 | 0.90, 0.99, 0.96, 0.79 |

The carried working trace, not the warm neural start, is what stops a continuing composed
brain from acquiring a new mapping at the default amplitude: the trace enters the association
region at amplitude 3 through the scale-12 prefrontal projection and dominates the probe, so
the brain holds its previous state. At amplitude 1 with decay 0.8 the continuing life learns.

Two further facts about the instrument. A probe taught through `step(teacher=...)` leaves an
action awaiting feedback; the next `step` then trains the actor-critic on a zero reward at its
default rate and undoes the lesson, so this chamber teaches through `Learner.step` on the
trace-bearing drive and advances the life with greedy acts. And the "appended" control is only
an upper bound when the brain was trained with the cue appended; a brain trained on vanished
cues has never read those inputs and answers at chance on them, which the first sweep showed.

Not yet measured: free recall of a vanished cue over delays 1 to 16 in the continuing regime
at amplitude 1, decay 0.8, with and without distractors, against the episodic regime. The sweep
was prepared (`--amplitude 1.0 --decays 0.8 --delays 1 2 4 8 16`) and stopped before it ran.

