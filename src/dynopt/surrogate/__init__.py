"""ML surrogates for instant modal prediction."""

from .gp import FrequencySurrogate
from .multifidelity import CORRECTION_MODES, MultiFidelitySurrogate
from .active import expected_improvement, max_variance, propose_batch

__all__ = [
    "FrequencySurrogate", "MultiFidelitySurrogate", "CORRECTION_MODES", "propose_batch", "max_variance", "expected_improvement",
]
