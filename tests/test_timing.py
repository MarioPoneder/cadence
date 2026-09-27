"""Direct tests for cadence.timing: environment() and latency()."""

import pytest

from cadence.timing import environment, latency

# ------------------------------------------------------------------ environment


def test_environment_returns_dict_with_expected_keys() -> None:
    env = environment()
    for key in ("machine", "system", "cores", "python", "threads", "numpy"):
        assert key in env, f"missing key: {key}"


def test_environment_python_version_is_string() -> None:
    env = environment()
    assert isinstance(env["python"], str) and "." in env["python"]


def test_environment_numpy_version_is_string() -> None:
    env = environment()
    assert isinstance(env["numpy"], str)


def test_environment_cores_is_positive_int_or_none() -> None:
    env = environment()
    assert env["cores"] is None or (isinstance(env["cores"], int) and env["cores"] >= 1)


# ------------------------------------------------------------------ latency


def test_latency_returns_expected_keys() -> None:
    result = latency(lambda: None, repeats=10, warmup=2)
    for key in ("repeats", "p50_us", "p90_us", "p99_us", "max_us", "mean_us"):
        assert key in result, f"missing key: {key}"


def test_latency_percentiles_are_ordered() -> None:
    result = latency(lambda: None, repeats=50, warmup=5)
    assert result["p50_us"] <= result["p90_us"] <= result["p99_us"] <= result["max_us"]


def test_latency_mean_is_positive() -> None:
    result = latency(lambda: None, repeats=20, warmup=2)
    assert result["mean_us"] >= 0.0


def test_latency_repeats_field_matches_argument() -> None:
    result = latency(lambda: None, repeats=17, warmup=0)
    assert result["repeats"] == 17


def test_latency_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="repeats"):
        latency(lambda: None, repeats=0)


def test_latency_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="warmup"):
        latency(lambda: None, repeats=5, warmup=-1)


def test_latency_environment_is_embedded() -> None:
    result = latency(lambda: None, repeats=5, warmup=0)
    assert "environment" in result
    assert "python" in result["environment"]


def test_environment_reports_a_missing_optional_backend_as_none(monkeypatch) -> None:
    import importlib

    real = importlib.import_module

    def absent(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", absent)
    assert environment()["numba"] is None


def test_environment_does_not_swallow_unexpected_errors(monkeypatch) -> None:
    import importlib

    real = importlib.import_module

    def broken(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("a bug, not an absent backend")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(RuntimeError, match="a bug"):
        environment()
