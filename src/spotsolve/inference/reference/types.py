"""Search settings specific to the frozen SciPy reference."""
from dataclasses import dataclass
from ..types import FitOptions as NativeFitOptions

@dataclass(frozen=True)
class FitOptions(NativeFitOptions):
    ftol: float = 1e-11
