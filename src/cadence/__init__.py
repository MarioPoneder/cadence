"""Self-reading columns and contextual predictors with local settlement."""

from .column import CorticalColumn
from .cortex import Cortex, SettlementError
from .element import Port, settle, solve_cluster_forest
from .wiring import (
    BinnedFeatures,
    ConstantFeatures,
    IdentityFeatures,
    calibrate,
    grid,
    wire,
)

__version__ = "0.20.1"
__all__ = [
    "BinnedFeatures",
    "ConstantFeatures",
    "Cortex",
    "CorticalColumn",
    "IdentityFeatures",
    "Port",
    "SettlementError",
    "__version__",
    "calibrate",
    "grid",
    "settle",
    "solve_cluster_forest",
    "wire",
]
