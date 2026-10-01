"""Keep source and distribution identities consistent before building a wheel."""

import tomllib
from pathlib import Path

import cadence


def test_package_version_matches_distribution_metadata():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert cadence.__version__ == metadata["project"]["version"]


def test_unreleased_candidate_has_development_identity():
    root = Path(__file__).resolve().parents[1]
    migration = (root / "docs" / "MIGRATION_060.md").read_text()
    if "**unreleased development candidate**" in migration:
        assert ".dev" in cadence.__version__
        assert f"`{cadence.__version__}`" in migration
