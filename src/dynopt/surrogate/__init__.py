"""ML surrogates for instant modal prediction."""

from .gp import FrequencySurrogate
from .active import expected_improvement, max_variance, propose_batch

__all__ = [
    "FrequencySurrogate", "propose_batch", "max_variance", "expected_improvement",
]
