"""dynopt -- automated dynamic coupling and resonance optimisation.

Pipeline:  parametric CAD -> modal FEA -> coupling analysis -> GP surrogate
           -> optimiser -> optimised CAD (STEP).
"""

__version__ = "0.1.0"

from .geometry.base import DesignSpace, DesignVariable
from .fea.base import ModalResult, ModalSolver
from .optimize.objectives import ExcitationBand, ResonanceObjective, rpm_bands

__all__ = [
    "DesignSpace", "DesignVariable", "ModalResult", "ModalSolver",
    "ExcitationBand", "ResonanceObjective", "rpm_bands", "__version__",
]
