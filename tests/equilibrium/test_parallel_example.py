"""Independent process lives preserve learning, ordering and failure visibility."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Import the standalone source as spawn workers do, without a new core export.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "equilibrium"
sys.path.insert(0, str(EXAMPLES))
from parallel_bootstrap import (  # noqa: E402
    MAX_ERROR,
    PointBody,
    experiences,
    run_lives,
)


def stable(life):
    return {
        key: value for key, value in life.items() if key not in ("elapsed_seconds", "process_id")
    }


def test_serial_and_process_lives_have_identical_ordered_continuations():
    seeds = (2, 0, 1)
    serial = run_lives(seeds, workers=1)
    parallel = run_lives(seeds, workers=2)
    assert [life["seed"] for life in parallel] == list(seeds)
    assert [stable(life) for life in serial] == [stable(life) for life in parallel]
    assert all(life["process_id"] == os.getpid() for life in serial)
    # Assert real process isolation without relying on worker scheduling/timing.
    assert all(life["process_id"] != os.getpid() for life in parallel)
    assert 1 <= len({life["process_id"] for life in parallel}) <= 2
    assert len({life["checkpoint_sha256"] for life in parallel}) == len(seeds)
    for life in parallel:
        report = life["bootstrap"]
        assert report["passed"]
        assert report["examples"] == report["checks"] == 4
        assert report["accepted"] > 0
        assert report["history"][-1]["checks"]["max_error"] <= MAX_ERROR
        assert life["admissions"] == report["accepted"] + len(life["live"])
        assert len(life["live"]) == 3
        assert max(abs(x["predicted"] - x["measured"]) for x in life["live"]) < 0.15


def test_simulator_witnesses_are_observed_transitions():
    body = PointBody(0.2)
    assert body.advance(0.4) == pytest.approx(0.3)
    assert body.advance(-0.4) == pytest.approx(0.2)
    assert experiences((0.2,), (0.4,))[0][1]["next_position"][0] == pytest.approx(0.3)


def test_zero_epochs_reports_failure_without_entering_live_phase():
    life = run_lives((0,), epochs=0)[0]
    assert not life["bootstrap"]["passed"]
    assert life["admissions"] == 0
    assert life["live"] == []


def test_worker_exception_propagates_instead_of_dropping_the_life():
    with pytest.raises(ValueError, match="seed"):
        run_lives((0, -1), workers=2)


@pytest.mark.parametrize(
    "seeds,options",
    (((), {}), ((0,), {"workers": 0}), ((0,), {"epochs": 101})),
)
def test_workload_bounds_are_checked(seeds, options):
    with pytest.raises(ValueError):
        run_lives(seeds, **options)


def test_cli_help_and_failed_readiness_are_explicit():
    script = EXAMPLES / "parallel_bootstrap.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--workers" in help_run.stdout
    run = subprocess.run(
        [sys.executable, str(script), "--lives", "1", "--epochs", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stderr
    assert not json.loads(run.stdout)["passed"]
