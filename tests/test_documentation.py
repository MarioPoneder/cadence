"""Execute documented examples and keep the public constructor reference current."""

import inspect
import re
import runpy
import time
from pathlib import Path

import pytest

import cadence

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "examples" / "README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


@pytest.mark.parametrize(
    "path", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT))
)
def test_documented_python(path):
    """Each document has one namespace, as if its examples formed a script."""
    text = path.read_text()
    namespace = {"__name__": "__main__", "__file__": str(path)}
    for block in FENCE.finditer(text):
        language = block.group(1).strip()
        if not language.startswith(("python", "py ")) and language != "py":
            continue
        assert language in ("python", "py"), "Put signatures in text fences"
        prefix = text[: block.start()].rstrip()
        marker = re.search(r"<!--\s*not-run\b([^<>]*?)-->$", prefix, re.DOTALL)
        if marker:
            reason = marker.group(1).strip().removeprefix(":").strip()
            assert reason, "Every unexecuted Python example needs an explicit reason"
            continue
        line = text.count("\n", 0, block.start()) + 2
        code = "\n" * (line - 1) + block.group(2)
        exec(compile(code, str(path), "exec"), namespace)


def test_public_exports_are_documented():
    reference = (ROOT / "docs" / "REFERENCE.md").read_text()
    for name in cadence.__all__:
        assert re.search(rf"\b{re.escape(name)}\b", reference), name


@pytest.mark.parametrize("kind", ("small", "deep", "recursive"))
@pytest.mark.parametrize("seed", (0, 2, 7))
def test_layout_learning_example(kind, seed):
    example = runpy.run_path(str(ROOT / "examples" / "layout_learning.py"))
    report = example["learn"](kind, seed=seed)
    assert report["updates"] > 0
    assert report["accepted_examples"] == 4 * report["updates"]
    assert report["max_fresh_error"] < 0.1
    assert report["resume_exact"]


@pytest.mark.parametrize(
    "name",
    [
        "Cortex",
        "bootstrap",
        "History",
        "LearningProgress",
        "Reinforcement",
        "LiveController",
        "slew",
    ],
)
def test_constructor_reference_matches_api(name):
    reference = (ROOT / "docs" / "REFERENCE.md").read_text()
    match = re.search(rf"(?ms)^{name}\(.*?\)", reference)
    assert match, f"Missing complete {name} constructor signature"
    namespace = {"time": time}
    exec(f"def {match.group()}:\n    pass\n", namespace)
    documented = inspect.signature(namespace[name]).parameters
    actual = inspect.signature(getattr(cadence, name)).parameters
    assert list(documented) == list(actual)
    for parameter_name, parameter in actual.items():
        assert documented[parameter_name].kind == parameter.kind
        assert documented[parameter_name].default == parameter.default
        assert f"`{parameter_name}`" in reference


PUBLIC_METHODS = [
    (owner, name)
    for owner in (
        cadence.Cortex,
        cadence.Brain,
        cadence.History,
        cadence.LearningProgress,
        cadence.Reinforcement,
        cadence.LiveController,
    )
    for name, member in inspect.getmembers(owner)
    if not name.startswith("_")
    and (inspect.isfunction(member) or inspect.ismethod(member))
]


@pytest.mark.parametrize(
    ("owner", "name"),
    PUBLIC_METHODS,
    ids=[f"{c.__name__}.{n}" for c, n in PUBLIC_METHODS],
)
def test_public_method_reference_matches_api(owner, name):
    """Examples alone cannot detect misleading optional/required parameters."""
    reference = (ROOT / "docs" / "REFERENCE.md").read_text()
    match = re.search(rf"`(?:{owner.__name__}\.)?{name}(\([^`]*\))`", reference)
    assert match, f"Missing exact {owner.__name__}.{name} signature"
    namespace = {}
    exec(f"def documented{match.group(1)}:\n    pass\n", namespace)
    documented = inspect.signature(namespace["documented"]).parameters
    actual = {
        key: parameter
        for key, parameter in inspect.signature(getattr(owner, name)).parameters.items()
        if key != "self"
    }
    assert list(documented) == list(actual)
    for key, parameter in actual.items():
        assert documented[key].kind == parameter.kind
        assert documented[key].default == parameter.default
