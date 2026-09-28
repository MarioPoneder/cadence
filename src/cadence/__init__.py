"""Deep Recursive Settlement Networks with jointly settling populations."""

from .drsn import Brain, Cortex, Input, Output, Population, SettlementError

__version__ = "0.43.0"

__all__ = [
    "Brain",
    "Cortex",
    "Input",
    "Output",
    "Population",
    "SettlementError",
    "__version__",
]
