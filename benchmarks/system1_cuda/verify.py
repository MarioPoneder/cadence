"""Independently verify a campaign and publish its source-bound scalar evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run import write


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(folder):
    campaign = read(folder / "campaign.json")
    outcomes = read(folder / "outcomes.json")
    protocol = campaign["protocol"]
    checks, records, sources = [], [], {}

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    check("job inventory complete", len(outcomes) == len(campaign["jobs"]))
    check(
        "harness remained frozen",
        read(folder / "completion.json")["harness_and_revisions_unchanged"],
    )
    for job in outcomes:
        directory = folder / job["directory"]
        manifest, result = read(directory / "manifest.json"), read(directory / "result.json")
        identity = job["directory"]
        check(identity + " exit", job["exit_code"] == 0)
        check(identity + " completed", result["passed"] and result["source_unchanged"])
        check(identity + " commit", manifest["commit"] == campaign["revisions"][job["revision"]])
        check(
            identity + " protocol", manifest["protocol_sha256_lf"] == campaign["protocol_sha256_lf"]
        )
        script = "run.py" if job["kind"] == "nursery" else job["kind"] + ".py"
        check(
            identity + " harness",
            manifest["harness_sha256_lf"] == campaign["harness_sha256_lf"][script],
        )
        if job["revision"] not in sources:
            sources[job["revision"]] = manifest["source_sha256_lf"]
        check(identity + " sources", sources[job["revision"]] == manifest["source_sha256_lf"])
        record = {k: v for k, v in job.items() if k != "command"}
        record["result"] = result
        if job["kind"] == "nursery":
            check(identity + " clean", not manifest["dirty"])
            check(
                identity + " acquisition",
                result["bootstrap_accuracy"] >= protocol["gate"]["bootstrap_held_out_accuracy"]
                and result["bootstrap_accuracy"] > result["founder_accuracy"],
            )
            check(
                identity + " retention",
                result["retained_accuracy"] >= protocol["gate"]["retained_held_out_accuracy"],
            )
            check(identity + " saved continuation", result["saved_actions_equal"])
            check(
                identity + " live inventory",
                len(result["live"]) == sum(s["calls"] for s in protocol["stages"]),
            )
            check(
                identity + " free admission",
                all(r["settlement"]["qualified"] for r in result["live"]),
            )
            check(
                identity + " no failed calls",
                all(r["failure"] is None for r in result["measurements"]),
            )
            check(
                identity + " dtype",
                result["execution"]["state_dtype"].endswith(
                    "float32" if job["backend"] == "cuda32" else "float64"
                ),
            )
            for name, group in result["latencies"].items():
                rows = [r for r in result["measurements"] if r["stage"] == name]
                durations = [r["seconds"] for r in rows]
                for percentile in (50, 95, 99):
                    check(
                        identity + f" {name} p{percentile}",
                        np.isclose(
                            group[f"p{percentile}_seconds"],
                            np.percentile(durations, percentile),
                            rtol=1e-14,
                            atol=0,
                        ),
                    )
                check(
                    identity + " " + name + " throughput",
                    np.isclose(
                        group["decisions_per_second"],
                        sum(r["decisions"] for r in rows) / sum(durations),
                    ),
                )
        elif job["kind"] == "transport":
            check(
                identity + " cases",
                len(result["cases"])
                == len(protocol["transport_controls"]["layouts"])
                * len(protocol["transport_controls"]["batches"]),
            )
            for case in result["cases"]:
                check(
                    identity + " states hash " + case["states_file"],
                    case["states_sha256"]
                    == hashlib.sha256((directory / case["states_file"]).read_bytes()).hexdigest(),
                )
        else:
            record["python_profile"] = (directory / "python-profile.txt").read_text(
                encoding="utf-8"
            )
            record["operators_table"] = (directory / "operators.txt").read_text(encoding="utf-8")
        records.append(record)

    comparisons = []
    controls = [r for r in records if r["kind"] == "transport"]
    for right in controls:
        if right["revision"] != "candidate":
            continue
        for left in controls:
            pair = left["backend"] == right["backend"] and left["revision"] == "baseline"
            reference = (
                left["backend"] == "cpu"
                and left["revision"] == "candidate"
                and right["backend"] != "cpu"
            )
            if not (pair or reference):
                continue
            tolerance = protocol["transport_controls"][
                "float32_atol_rtol" if right["backend"] == "cuda32" else "float64_atol_rtol"
            ]
            for a, b in zip(left["result"]["cases"], right["result"]["cases"], strict=True):
                with np.load(folder / left["directory"] / a["states_file"]) as archive:
                    expected = archive["potentials"]
                with np.load(folder / right["directory"] / b["states_file"]) as archive:
                    actual = archive["potentials"]
                passed = np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
                label = left["directory"] + " -> " + right["directory"] + " " + a["states_file"]
                check(label, passed)
                comparisons.append(
                    {
                        "left": left["directory"],
                        "right": right["directory"],
                        "case": a["states_file"],
                        "max_abs_deviation": float(np.max(np.abs(actual - expected))),
                        "atol_rtol": tolerance,
                        "passed": bool(passed),
                    }
                )
    return {
        "passed": all(c["passed"] for c in checks),
        "campaign": campaign,
        "source_sha256_lf": sources,
        "records": records,
        "transport_comparisons": comparisons,
        "checks": checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.campaign)
    write(args.out, receipt)
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "checks": len(receipt["checks"]),
                "failures": [c for c in receipt["checks"] if not c["passed"]],
            }
        )
    )
    raise SystemExit(0 if receipt["passed"] else 1)
