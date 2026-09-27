"""Run the reader's actual introductory snippets and check local documentation links."""

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "page",
    [
        "docs/patchnet.md",
        "docs/temporal.md",
        "docs/architecture.md",
        "docs/temporal-memory.md",
        "docs/quickstart.md",
        "docs/concepts.md",
        "docs/memory.md",
        "docs/continuous.md",
        "docs/tasks.md",
        "docs/reward.md",
        "docs/certificate.md",
        "docs/cortex.md",
        "docs/brain.md",
        "docs/evolution.md",
        "docs/build.md",
        "docs/belief.md",
        "docs/steering.md",
        "docs/recursive-settlement.md",
    ],
)
def test_introductory_python_snippets(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "documentation_example"}
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)```", (ROOT / page).read_text(), re.S)
    ):
        exec(compile(code, f"{page}:python-block-{index + 1}", "exec"), namespace)
    if page == "docs/quickstart.md":
        assert namespace["night"]["updates"] > 200 and namespace["learner"].updates == 80


def test_local_documentation_links_resolve():
    problems = []
    for page in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", page.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = unquote(target).partition("#")
            destination = (page.parent / path).resolve() if path else page
            if not destination.exists():
                problems.append(f"{page.name}: missing {target}")
            elif anchor and destination.suffix == ".md":
                content = destination.read_text()
                headings = re.findall(r"^#+\s+(.+)$", content, re.M)
                slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                if anchor not in slugs and f'id="{anchor}"' not in content:
                    problems.append(f"{page.name}: missing anchor {target}")
    assert not problems, "\n".join(problems)


def test_minimal_install_ci_guides_exist():
    """The installed-wheel smoke must not depend on a retired source guide."""
    workflow = ROOT / ".github/workflows/ci.yml"
    if not workflow.exists():
        pytest.skip("CI workflows are not included in the source distribution")
    pages = re.findall(r'"(docs/[^"\n]+\.md)"', workflow.read_text())
    assert pages, "The minimal-install job must exercise documented examples"
    assert all((ROOT / page).is_file() for page in pages)
