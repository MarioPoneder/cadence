"""Record actual CUDA tests from a clean, committed checkout.

Requires the dev tools and a CUDA-enabled PyTorch build. By default, run the
CUDA cases in the tensor math and batch suites; --full runs the entire suite.
The output must be a new path. Historical receipts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CUDA_MODULES = {"tests.test_tensor_math", "tests.test_batch_tensor"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sources():
    paths = git("ls-files", "*.py", "*.md", "pyproject.toml").splitlines()
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root")
    if args.out.exists():
        parser.error("--out must be a new path; preserve previous receipts")
    if git("status", "--porcelain"):
        parser.error("Commit changes before recording evidence; checkout must be clean")

    import pytest
    import torch

    if not torch.cuda.is_available():
        parser.error(
            "CUDA must be available to PyTorch; skipped tests are not evidence"
        )
    torch.set_num_threads(1)
    properties = torch.cuda.get_device_properties(0)
    for dtype in (torch.float32, torch.float64):
        probe = torch.ones(4, device="cuda:0", dtype=dtype)
        if (probe * probe).sum().item() != 4:
            raise RuntimeError(f"CUDA arithmetic preflight failed for {dtype}")
    torch.cuda.synchronize()

    commit, before = git("rev-parse", "HEAD"), sources()
    environment = {
        "python": sys.version,
        "utf8_mode": sys.flags.utf8_mode,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": "cuda:0",
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_device_count": torch.cuda.device_count(),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "pytest": pytest.__version__,
    }
    pytest_args = ["-q"]
    if not args.full:
        pytest_args += [
            "tests/test_tensor_math.py",
            "tests/test_batch_tensor.py",
            "-k",
            "cuda",
        ]
    started_utc = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory(prefix="cadence-cuda-") as directory:
        junit = Path(directory) / "results.xml"
        started = perf_counter()
        exit_code = int(pytest.main([*pytest_args, f"--junitxml={junit}"]))
        elapsed = perf_counter() - started
        suite = ET.parse(junit).getroot().find("testsuite")

    cases, cuda_cases = [], []
    skips = Counter()
    for case in suite.findall("testcase"):
        outcome = next(
            (
                name
                for name in ("failure", "error", "skipped")
                if case.find(name) is not None
            ),
            "passed",
        )
        row = {
            "test": case.get("classname") + "::" + case.get("name"),
            "outcome": outcome,
            "seconds": float(case.get("time")),
        }
        if outcome != "passed":
            detail = case.find(outcome)
            row["message"], row["detail"] = detail.get("message"), detail.text
        is_cuda = (
            case.get("classname") in CUDA_MODULES
            and "cuda" in case.get("name").split("[")[-1]
        )
        if is_cuda:
            cuda_cases.append(row)
        if is_cuda or outcome != "passed":
            cases.append(row)
        if outcome == "skipped":
            skips[row["message"]] += 1

    after = sources()
    unchanged = (
        before == after
        and commit == git("rev-parse", "HEAD")
        and not git("status", "--porcelain")
    )
    both_dtypes = all(
        any(row["test"].endswith(f"cuda{bits}]") for row in cuda_cases)
        for bits in (32, 64)
    )
    passed = (
        exit_code == 0
        and unchanged
        and both_dtypes
        and all(row["outcome"] == "passed" for row in cuda_cases)
    )
    receipt = {
        "issue": "https://github.com/muellerberndt/cadence/issues/64",
        "scope": "Full suite" if args.full else "CUDA subset",
        "started_utc": started_utc,
        "commit": commit,
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_unchanged": bool(unchanged),
        "source_hash_note": "Exact checkout bytes, including line endings; commit identifies the complete Git tree.",
        "environment": environment,
        "dtypes": ["float64", "float32"],
        "pytest_args": pytest_args,
        "pytest_exit_code": exit_code,
        "pytest_seconds": elapsed,
        "timing_scope": "pytest invocation including collection; excludes runner imports and CUDA preflight; not production latency",
        "counts": {
            key: int(suite.get(key))
            for key in ("tests", "failures", "errors", "skipped")
        },
        "cuda_counts": dict(Counter(row["outcome"] for row in cuda_cases)),
        "both_cuda_dtypes_collected": both_dtypes,
        "skips_by_reason": dict(skips),
        "cases": cases,
        "passed": bool(passed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Receipt: {args.out}; passed={bool(passed)}")
    return exit_code or int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
