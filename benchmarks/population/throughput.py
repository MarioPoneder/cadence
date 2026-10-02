#!/usr/bin/env python3
"""Throughput of the population kernel against the NumPy patch on one core.

One moment's work per brain and stream: an imagined reading, a one-moment path from rest,
and an observation with its write. The population runs P instances x B streams in lockstep
on the torch device (cuda, then mps, then cpu); the reference runs the same patch, one brain
in one stream with its own store, on one processor core with the BLAS thread count at one.
The receipt is written beside this file.

    python benchmarks/population/throughput.py
    python benchmarks/population/throughput.py --device cpu --steps 20
"""
from __future__ import annotations

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cadence  # noqa: E402
from cadence import RecordPatchNet  # noqa: E402
from cadence.population import PopulationPatch, device_of  # noqa: E402

HERE = Path(__file__).resolve().parent
PATCH = dict(inputs=32, hidden=32, outputs=8, cells=256, active=8)
SIZES = ((8, 64), (32, 64), (64, 128), (64, 256))  # (instances, streams)
RATE = 0.1


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _processor() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(0)
    return _processor()


def _net(seed: int = 0) -> RecordPatchNet:
    return RecordPatchNet(
        PATCH["inputs"], PATCH["hidden"], PATCH["outputs"], seed=seed,
        cells=PATCH["cells"], active=PATCH["active"], slowest=2.0,
    )


def _world(steps: int, P: int, B: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed random linear world per brain and stream, y = xW + noise, generated before the
    clock starts so that the timed loop is the kernel alone."""
    W = torch.randn(P, B, PATCH["inputs"], PATCH["outputs"], generator=gen) / PATCH["inputs"] ** 0.5
    x = torch.rand(steps, P, B, PATCH["inputs"], generator=gen)
    y = torch.einsum("tpbi,pbio->tpbo", x, W) + 0.05 * torch.randn(
        steps, P, B, PATCH["outputs"], generator=gen
    )
    return x, y


def population_rows(dev: torch.device, steps: int, warm: int, seed: int) -> list[dict]:
    rows = []
    for P, B in SIZES:
        gen = torch.Generator().manual_seed(seed)
        x, y = _world(warm + steps, P, B, gen)
        x, y = x.to(dev), y.to(dev)
        pop = PopulationPatch.from_patch(_net(seed), instances=P, streams=B, device=dev)
        for t in range(warm):
            pop.imagine(x[t])
            pop.observe(x[t], y[t], rate=RATE)
        _sync(dev)
        t0 = time.perf_counter()
        for t in range(warm, warm + steps):
            pop.imagine(x[t])
            pop.observe(x[t], y[t], rate=RATE)
        _sync(dev)
        seconds = time.perf_counter() - t0
        moments = P * B * steps
        rows.append(
            {
                "instances": P,
                "streams": B,
                "moments_per_step": P * B,
                "steps": steps,
                "seconds": round(seconds, 4),
                "moments_per_second": round(moments / seconds, 1),
                "store_bytes": int(pop.memory_bytes()) if hasattr(pop, "memory_bytes") else None,
            }
        )
        print(
            f"population {P} x {B}: {moments / seconds:,.0f} moments/s ({seconds:.2f} s)",
            flush=True,
        )
        del pop, x, y
    return rows


def reference(moments: int, warm: int, seed: int, batch: int = 1) -> dict:
    """The NumPy patch on one core: `batch` streams per call, each moment from rest. With
    batch 1 every stream has its own store, as in the population; with a larger batch the
    rows share one store, which is a different world and is reported only for scale."""
    rng = np.random.default_rng(seed)
    net = _net(seed)
    W = rng.standard_normal((PATCH["inputs"], PATCH["outputs"])) / PATCH["inputs"] ** 0.5
    x = rng.random((warm + moments, batch, 1, PATCH["inputs"]))
    y = x @ W + 0.05 * rng.standard_normal((warm + moments, batch, 1, PATCH["outputs"]))

    def moment(t: int) -> None:
        net.reset()
        net.imagine(x[t])
        net.reset()
        net.observe(x[t], y[t], rate=RATE, backtrack=False, write=True)

    for t in range(warm):
        moment(t)
    t0 = time.perf_counter()
    for t in range(warm, warm + moments):
        moment(t)
    seconds = time.perf_counter() - t0
    per_second = moments * batch / seconds
    print(f"reference batch {batch}: {per_second:,.1f} moments/s ({seconds:.2f} s)", flush=True)
    return {
        "backend": "numpy, one processor core, BLAS threads 1",
        "streams_per_call": batch,
        "own_store_per_stream": batch == 1,
        "moments": moments * batch,
        "seconds": round(seconds, 4),
        "moments_per_second": round(per_second, 1),
    }


def parity(seed: int) -> dict:
    """The twin against the NumPy patch on the cpu in float64: the reading, and the slow
    parameters after one observed moment."""
    net = _net(seed)
    twin = PopulationPatch.from_patch(
        net, instances=1, streams=1, device="cpu", dtype=torch.float64
    )
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(1, 1, PATCH["inputs"]))
    t = rng.normal(size=(1, 1, PATCH["outputs"]))
    out = twin.imagine(torch.as_tensor(x))
    net.reset()
    ref = net.imagine(x[0][:, None, :])
    reading = float(np.max(np.abs(out["output"][0].numpy() - ref.output[:, 0])))
    net.reset()
    net.observe(x, t, rate=0.3, backtrack=False, write=False)
    twin.observe(torch.as_tensor(x), torch.as_tensor(t), rate=0.3, write=False)
    params = net.parameters()
    slow = max(
        float(np.max(np.abs(twin.parameters()[k][0].numpy() - params[n])))
        for n, k in (("G", "G"), ("g", "g"), ("B", "B"), ("b", "b"), ("C", "C"), ("c", "c"))
    )
    return {"max_abs_reading": reading, "max_abs_slow_step": slow}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--device", default=None, help="cuda, mps or cpu; default: the first available"
    )
    ap.add_argument("--steps", type=int, default=40, help="timed steps per population size")
    ap.add_argument("--warm", type=int, default=5, help="untimed steps before the clock")
    ap.add_argument("--reference-moments", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "receipt.json"))
    a = ap.parse_args()
    dev = device_of(a.device)
    started = time.perf_counter()
    rows = population_rows(dev, a.steps, a.warm, a.seed)
    ref = reference(a.reference_moments, a.warm, a.seed, batch=1)
    ref_shared = reference(a.reference_moments, a.warm, a.seed, batch=64)
    par = parity(a.seed)
    best = max(rows, key=lambda r: r["moments_per_second"])
    receipt = {
        "kind": "cadence.population-throughput/1",
        "library": cadence.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": {"type": dev.type, "name": _device_name(dev)},
        "processor": _processor(),
        "date": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d"),
        "patch": dict(PATCH, slowest=2.0, rate=RATE),
        "moment": (
            "one imagined reading from rest and one observation with its write, "
            "per brain and stream"
        ),
        "population": rows,
        "reference": ref,
        "reference_shared_store": ref_shared,
        "ratio_best_over_reference": round(
            best["moments_per_second"] / ref["moments_per_second"], 1
        ),
        "parity_cpu_float64": par,
        "load_average_at_end": (
            [round(v, 1) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None
        ),
        "seconds_total": round(time.perf_counter() - started, 1),
    }
    Path(a.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"ratio: {receipt['ratio_best_over_reference']} x; receipt {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
