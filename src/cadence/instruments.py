"""The instruments of the orienting response: what a gain does when an event arrives.

A steering patch's gain is a trace over frames; the world's events are stretches of frames of
a kind (a clock's tick, a cry). For each event: the *capture* is the gain's peak during the
event over its level in the quiet frames before it; the *latency* is the first frame from the
onset at which half the capture is reached; the *return* is the number of frames after the
event until the gain is back within a fifth of the capture of its level before. Over the
events of a kind, the *habituation curve* is the capture of the first events against the
last. *Dishabituation* is the capture of one kind's events just before and just after an event
of a consequential kind. The night nursery measured rung 4 with these; every rung that reads a
gain's response to an event reads the same numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["orienting", "dishabituation"]


def _mean(values: Sequence[float]) -> float | None:
    return None if len(values) == 0 else float(np.mean(values))


def orienting(
    gain: np.ndarray,
    events: Sequence[tuple[str, int, int]],
    *,
    pre: int = 4,
    post: int = 12,
    quiet: np.ndarray | None = None,
    bins: int = 10,
) -> dict[str, Any]:
    """Measure one gain trace ``(frames,)`` against ``events``, each ``(kind, start, length)``.
    ``quiet`` ``(frames,)`` says which frames may serve as a baseline (by default every frame
    outside every event); an event with no quiet frame in the ``pre`` frames before it, or
    running past the trace, is skipped. Returns ``rows``, one per measured event (kind,
    start, length, baseline, peak, capture, latency, return), and per kind: the count, the
    capture of the first and of the last ``bins`` events, the curve of captures in
    consecutive groups of ``bins``, the mean capture, latency shares (at the onset, one frame
    after, none) and the mean latency of the events that captured, the mean return and the
    mean baseline."""
    g = np.asarray(gain, dtype=float)
    if g.ndim != 1:
        raise ValueError("gain must be a (frames,) trace")
    if pre < 1 or post < 0 or bins < 1:
        raise ValueError("pre and bins must be positive and post nonnegative")
    T = len(g)
    inside = np.zeros(T, dtype=bool)
    for _, start, length in events:
        inside[max(0, start) : max(0, min(T, start + length))] = True
    usable = ~inside if quiet is None else np.asarray(quiet, dtype=bool)
    if usable.shape != (T,):
        raise ValueError("quiet must have one flag per frame")
    rows: list[dict[str, Any]] = []
    for kind, start, length in events:
        start, length = int(start), int(length)
        if start < 0 or length < 1 or start + length > T:
            continue
        before = np.arange(max(0, start - pre), start)
        before = before[usable[before]]
        if len(before) == 0:
            continue
        base = float(g[before].mean())
        during = g[start : start + length]
        peak = float(during.max())
        capture = peak - base
        latency = int(np.argmax(during - base >= 0.5 * capture)) if capture > 1e-9 else -1
        after = g[start + length : start + length + post]
        if capture > 0.02:
            back = np.flatnonzero(after <= base + 0.2 * capture)
            ret = int(back[0]) if len(back) else post
        else:
            ret = 0
        rows.append({"kind": kind, "start": start, "length": length, "baseline": base, "peak": peak, "capture": capture, "latency": latency, "return": ret})
    kinds: dict[str, Any] = {}
    for kind in sorted({r["kind"] for r in rows}):
        mine = [r for r in rows if r["kind"] == kind]
        caps = [r["capture"] for r in mine]
        lats = [r["latency"] for r in mine]
        groups = max(1, len(caps) // bins)
        curve = [float(np.mean(caps[i * bins : (i + 1) * bins])) for i in range(groups)]
        kinds[kind] = {
            "count": len(mine),
            "capture_first": _mean(caps[:bins]),
            "capture_last": _mean(caps[-bins:]),
            "capture_mean": _mean(caps),
            "curve": curve,
            "latency_zero": float(np.mean([lat == 0 for lat in lats])),
            "latency_one": float(np.mean([lat == 1 for lat in lats])),
            "latency_none": float(np.mean([lat < 0 for lat in lats])),
            "latency_mean": _mean([lat for lat in lats if lat >= 0]),
            "return_mean": _mean([r["return"] for r in mine]),
            "baseline_mean": _mean([r["baseline"] for r in mine]),
            "peak_mean": _mean([r["peak"] for r in mine]),
        }
    return {"rows": rows, "kinds": kinds, "pre": pre, "post": post}


def dishabituation(
    rows: Sequence[dict[str, Any]],
    *,
    consequential: str,
    kind: str,
    window: int = 60,
    count: int = 2,
) -> dict[str, float | None]:
    """Around each event of the ``consequential`` kind, the mean capture of the last ``count``
    events of ``kind`` within ``window`` frames before it and of the first ``count`` after it,
    pooled over the consequential events: a habituated response that returns after a
    consequential event shows as ``after`` above ``before``."""
    before: list[float] = []
    after: list[float] = []
    others = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["start"])
    for c in (r for r in rows if r["kind"] == consequential):
        earlier = [r["capture"] for r in others if c["start"] - window <= r["start"] < c["start"]]
        later = [r["capture"] for r in others if c["start"] < r["start"] <= c["start"] + window]
        before += earlier[-count:]
        after += later[:count]
    return {"before": _mean(before), "after": _mean(after), "events": len(before) + len(after)}
