"""Wiring: calibration chooses what a Cortex reads; depth and width shape it.

No observation byte is hand-picked. ``calibrate`` holds each action in
turn and scores every byte of the (indexable, byte-valued) observation:

- ``controllability``: how far the byte's mean moves between forced-
  action runs against its pooled spread - the body's own efference
  finds the body;
- ``drive``: the pooled spread itself - what the world moves.

``wire`` turns a calibration into fine-to-coarse feature maps for the
Cortex, configured like layer sizes in a small torch model:

- ``width`` (default 1): how many calibrated bytes the finest level
  reads - the top ``width`` controllable bytes (bins 16, then 8 each)
  joined with the top ``width`` drive bytes (bins 8, then 4 each).
  Raise it when one byte pair cannot express the task; the measured
  example is Pong, where width 1 pinned every variant at -21.
- ``depth`` (default 2): hidden levels above the finest.
  0 is the flat ablation (finest level alone), 1 adds only the global
  per-action prior, 2 adds a 4-bin coarsening of the primary
  controllable byte (the receipts' h2), 3 adds 16- and 4-bin
  coarsenings (the receipts' h3).
- ``ladder``: explicit intermediate bin sizes for the primary
  controllable byte, finest first (for example ``(16, 4)``), overriding
  ``depth``'s ladder; the global level is always appended.

Every map declares ``cells`` (its context count) so the Cortex can
weight witness mass per level.
"""
from __future__ import annotations

from statistics import fmean, pstdev

LADDERS = {0: None, 1: (), 2: (4,), 3: (16, 4)}


def quantize(byte: int, bins: int) -> int:
    return byte * bins // 256


def calibrate(env_factory, n_actions: int, *, steps: int = 400, seed: int = 0) -> dict:
    """Score every observation byte under forced single-action policies."""
    traces = []
    length = None
    for action in range(n_actions):
        env = env_factory()
        observation, _ = env.reset(seed=seed + action)
        length = len(observation)
        rows = [list(observation)]
        for _ in range(steps):
            observation, _, terminated, truncated, _ = env.step(action)
            rows.append(list(observation))
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + action)
        close = getattr(env, 'close', None)
        if callable(close):
            close()
        traces.append(rows)
    controllability, drive = [], []
    for byte in range(length):
        series = [[float(row[byte]) for row in rows] for rows in traces]
        means = [fmean(s) for s in series]
        pooled = fmean([pstdev(s) for s in series])
        controllability.append((max(means) - min(means)) / (1.0 + pooled))
        drive.append(pooled)
    ctrl_rank = sorted(range(length), key=lambda b: controllability[b], reverse=True)
    drive_rank = sorted((b for b in range(length) if b not in ctrl_rank[:8]),
                        key=lambda b: drive[b], reverse=True)
    return {'bytes': length, 'probe_steps': steps, 'probe_seed': seed,
            'controllability_rank': [(b, round(controllability[b], 3)) for b in ctrl_rank[:16]],
            'drive_rank': [(b, round(drive[b], 1)) for b in drive_rank[:16]],
            'controllable_byte': ctrl_rank[0],
            'drive_byte': drive_rank[0] if drive_rank else ctrl_rank[-1]}


def wire(calibration: dict, *, depth: int = 2, width: int = 1, ladder=None):
    """Fine-to-coarse feature maps for a Cortex; see the module docstring."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError('width must be an integer of at least 1')
    if ladder is None:
        if depth not in LADDERS:
            raise ValueError('depth must be one of 0, 1, 2, 3; or pass an explicit ladder')
        ladder = LADDERS[depth]
    else:
        ladder = tuple(int(b) for b in ladder)
        if any(b < 2 for b in ladder):
            raise ValueError('ladder bins must be at least 2')
    ctrl = [b for b, _ in calibration['controllability_rank'][:width]]
    drv = [b for b, _ in calibration['drive_rank'][:width]]
    if len(ctrl) < width or len(drv) < width:
        raise ValueError('Calibration offers fewer ranked bytes than the requested width')
    ctrl_bins = [16] + [8] * (width - 1)
    drive_bins = [8] + [4] * (width - 1)

    def fine(observation):
        return tuple([quantize(int(observation[b]), n) for b, n in zip(ctrl, ctrl_bins)]
                     + [quantize(int(observation[b]), n) for b, n in zip(drv, drive_bins)])
    cells = 1
    for n in ctrl_bins + drive_bins:
        cells *= n
    fine.cells = cells

    if ladder is None:  # depth 0: the flat ablation
        return (fine,)

    primary = ctrl[0]
    maps = [fine]
    for bins in ladder:
        def context(observation, _bins=bins, _byte=primary):
            return (quantize(int(observation[_byte]), _bins),)
        context.cells = bins
        maps.append(context)

    def constant(observation):
        return ()
    constant.cells = 1
    maps.append(constant)
    return tuple(maps)
