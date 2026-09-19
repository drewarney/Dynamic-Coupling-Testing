"""Design of experiments."""

from .sampling import discrepancy, full_factorial, lhs_sample, sobol_sample

__all__ = ["sobol_sample", "lhs_sample", "full_factorial", "discrepancy"]
