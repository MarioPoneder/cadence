"""Vanished-cue recall: how many steps the working trace of a composed brain carries a cue.

The chamber (issue 84, roadmap row 02). A stream shows one of ``cues`` cue patterns for one
step, then ``delay`` steps of a neutral or distracting picture, then a probe picture that is
identical for every stream. The correct action at the probe is the cue the stream saw, so a
brain can only answer from what its own state carried across the delay. Learning happens at
the probe step alone: the lesson ``step`` would run on a demonstration, applied to the drive
that includes the living trace, after which the life goes on with greedy acts (no reward
plasticity enters this chamber). Recall is
then measured on fresh streams with learning off: greedy acts only, which advance the trace
and change nothing else.

Controls on the same trained brain and the same fresh streams: the trace erased before the
probe (``reset`` of the working memory rows), the trace shuffled across streams, and an
equal-information upper bound in which the cue is appended to the probe input itself.

    python benchmarks/recall/vanished_cue.py --out /tmp/recall --delays 1 2 4 8 --decays 0.2 0.5 0.8 0.9

Every run writes ``summary.json`` with recall by decay and delay for the brain and the three
controls, the work charged (sweeps per settle), and the frozen arguments. This is a bounded
CPU instrument; it establishes a measured limit of the trace, not a general retention law.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cadence import Brain, LearnerConfig
from cadence.learning import LearningPhaseError

NEUTRAL = 0  # the blank picture shown through the delay and at the probe
PROBE = 1


def pictures(cues: int, distractors: int) -> tuple[np.ndarray, int]:
    """One-hot pictures: a blank, a probe, ``cues`` cue pictures and ``distractors`` others."""
    n = 2 + cues + distractors
    return np.eye(n), n


def make_brain(inputs: int, cues: int, seed: int, decay: float, amplitude: float, modules: tuple[int, ...],
               qualified: bool) -> Brain:
    learning = LearnerConfig(
        beta=0.1, temperature=0.2, tolerance=3e-3, free_steps=1024, nudged_steps=128 if qualified else 12,
        eta=0.05, eta_bias=0.005, momentum=0.9, normalize=0.99, normalize_floor=1e-4,
        qualified=qualified, damping=3,
    )
    return Brain.compose(
        inputs, cues, modules=modules, lateral=0.0, seed=seed, learning=learning,
        episodic=False, working_memory_decay=decay, working_memory_amplitude=amplitude,
    )


def episode(rng: np.random.Generator, pics: np.ndarray, cues: int, distractors: int, delay: int,
            streams: int) -> tuple[np.ndarray, np.ndarray]:
    """Observations (delay + 2 steps, streams, inputs) and the cue index per stream."""
    cue = rng.integers(0, cues, streams)
    steps = []
    steps.append(pics[2 + cue])
    for _ in range(delay):
        if distractors:
            which = rng.integers(0, distractors, streams)
            steps.append(pics[2 + cues + which])
        else:
            steps.append(np.tile(pics[NEUTRAL], (streams, 1)))
    steps.append(np.tile(pics[PROBE], (streams, 1)))
    return np.stack(steps), cue


def run_stream(brain: Brain, obs: np.ndarray, *, teacher: np.ndarray | None, control: str | None,
               cue_pics: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Drive one episode through the continuing brain (no reset: the trace carries over from the
    episode before, as it would in an animal); returns the probe answers, sweeps and refusals."""
    sweeps = refused = 0
    answers = None
    last = len(obs) - 1
    for t, x in enumerate(obs):
        if t == last and control in ("erased", "shuffled"):
            kept = (brain.working_memory.trace.copy(), brain.working_memory.last.copy())
            if control == "erased":
                brain.working_memory.reset(len(x))
            else:
                perm = rng.permutation(len(x))
                brain.working_memory.trace = brain.working_memory.trace[perm]
                brain.working_memory.last = brain.working_memory.last[perm]
        if t == last and control == "appended" and cue_pics is not None:
            x = np.concatenate([x[:, : -cue_pics.shape[1]], cue_pics], axis=1)
        try:
            if t == last and teacher is not None:
                # The lesson reads the living trace, as a step's demonstration does; the life
                # then goes on with a greedy act, which advances the trace and leaves no pending
                # action, so no reward plasticity enters this chamber.
                brain.learner.step(brain.stimulus(x), np.asarray(teacher, dtype=np.int64))
            answers = brain.act(x, greedy=True)
        except LearningPhaseError:
            refused += 1
            answers = brain.act(x, greedy=True)
        except RuntimeError:
            refused += 1
            answers = np.full(len(x), -1)
            brain.reset()  # a refused settle ends this life's episode; the next starts cold
            break
        report = brain.last_settlement
        if report is not None:
            sweeps += int(report["steps"])
        if t == last and control in ("erased", "shuffled"):
            brain.working_memory.trace, brain.working_memory.last = kept  # the life keeps its own trace
    return answers, sweeps, refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cues", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=0, help="distinct distractor pictures during the delay; 0 = blank delay")
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--decays", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.9])
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--modules", default="32")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--streams", type=int, default=16, help="streams per lesson batch")
    parser.add_argument("--lessons", type=int, default=150, help="lesson batches per delay")
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--finite", action="store_true", help="finite teaching instead of qualified")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    modules = tuple(int(m) for m in args.modules.split(","))
    pics, n_inputs = pictures(args.cues, args.distractors)
    inputs = n_inputs + args.cues  # the last ``cues`` inputs carry the appended cue for the upper bound only
    began = time.time()
    results = []
    for decay in args.decays:
        for delay in args.delays:
            for seed in args.seeds:
                rng = np.random.default_rng(1000 * seed + delay)
                brain = make_brain(inputs, args.cues, seed, decay, args.amplitude, modules, not args.finite)
                brain.reset()  # birth; the life continues from here without further resets
                pad = np.zeros((args.streams, args.cues))
                # bootstrap: lessons at the probe only
                lesson_sweeps = refused_lessons = 0
                for _ in range(args.lessons):
                    obs, cue = episode(rng, pics, args.cues, args.distractors, delay, args.streams)
                    obs = np.concatenate([obs, np.tile(pad, (len(obs), 1, 1))], axis=2)
                    _, sw, rf = run_stream(brain, obs, teacher=cue, control=None, cue_pics=None, rng=rng)
                    lesson_sweeps += sw
                    refused_lessons += rf
                # free recall on fresh streams, learning off, with controls on the same brain
                counts = {k: [0, 0, 0] for k in ("brain", "erased", "shuffled", "appended")}
                for _ in range(args.test_episodes):
                    obs, cue = episode(rng, pics, args.cues, args.distractors, delay, args.streams)
                    obs = np.concatenate([obs, np.tile(pad, (len(obs), 1, 1))], axis=2)
                    cue_pics = np.eye(args.cues)[cue]
                    carried = (brain.working_memory.trace.copy(), brain.working_memory.last.copy(),
                               brain.working_memory.cold.copy())
                    for control in ("brain", "erased", "shuffled", "appended"):
                        if control != "brain":  # each lesion replays the episode from the carried trace
                            brain.working_memory.trace, brain.working_memory.last, brain.working_memory.cold = (
                                carried[0].copy(), carried[1].copy(), carried[2].copy())
                        answers, sw, rf = run_stream(
                            brain, obs, teacher=None, control=None if control == "brain" else control,
                            cue_pics=cue_pics, rng=rng,
                        )
                        counts[control][0] += int(np.sum(answers == cue))
                        counts[control][1] += len(cue)
                        counts[control][2] += rf
                        if control == "brain":
                            lived = (brain.working_memory.trace.copy(), brain.working_memory.last.copy(),
                                     brain.working_memory.cold.copy())
                    brain.working_memory.trace, brain.working_memory.last, brain.working_memory.cold = lived
                row = {
                    "decay": decay, "delay": delay, "seed": seed,
                    "recall": {k: v[0] / v[1] for k, v in counts.items()},
                    "refused_tests": {k: v[2] for k, v in counts.items()},
                    "lesson_batches": args.lessons, "refused_lessons": refused_lessons,
                    "lesson_sweeps": lesson_sweeps,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
    table = {}
    for row in results:
        key = f"decay {row['decay']} delay {row['delay']}"
        table.setdefault(key, []).append(row["recall"])
    mean = {k: {c: float(np.mean([r[c] for r in rows])) for c in rows[0]} for k, rows in table.items()}
    summary = {
        "instrument": "vanished-cue recall (issue 84, roadmap row 02)",
        "arguments": vars(args) | {"out": str(args.out)},
        "chance": 1.0 / args.cues,
        "mean_recall": mean,
        "rows": results,
        "seconds": round(time.time() - began, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nmean recall (brain / erased / shuffled / appended), chance", round(1 / args.cues, 3))
    for k, v in mean.items():
        print(f"  {k}: {v['brain']:.2f} / {v['erased']:.2f} / {v['shuffled']:.2f} / {v['appended']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
