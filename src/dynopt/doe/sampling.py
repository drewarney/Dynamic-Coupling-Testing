"""Design-of-experiments sampling for surrogate training data.

Sobol sequences beat random sampling badly in 5-10 dimensions: for the same
budget you get lower discrepancy, so the GP sees the corners of the design space
instead of clustering. With ~200 FEA runs at 5 variables that difference is the
difference between a usable surrogate and a useless one.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from ..geometry.base import DesignSpace


def sobol_sample(space: DesignSpace, n: int, seed: int = 0) -> np.ndarray:
    """``n`` Sobol points in physical units, shape ``(n, n_vars)``.

    Sobol is a base-2 sequence: its balance properties hold exactly at powers of
    two, so ``n`` is rounded up to the next power of two and trimmed.
    """
    m = int(np.ceil(np.log2(max(n, 2))))
    u = qmc.Sobol(d=len(space), scramble=True, seed=seed).random_base2(m=m)[:n]
    lo = np.array([v.lower for v in space.variables])
    hi = np.array([v.upper for v in space.variables])
    return qmc.scale(u, lo, hi)


def lhs_sample(space: DesignSpace, n: int, seed: int = 0) -> np.ndarray:
    """Latin hypercube with centred-discrepancy optimisation."""
    u = qmc.LatinHypercube(d=len(space), seed=seed, optimization="random-cd").random(n)
    lo = np.array([v.lower for v in space.variables])
    hi = np.array([v.upper for v in space.variables])
    return qmc.scale(u, lo, hi)


def full_factorial(space: DesignSpace, levels: int = 3) -> np.ndarray:
    """Full factorial grid -- only sane for <=3 variables; here for screening."""
    axes = [np.linspace(v.lower, v.upper, levels) for v in space.variables]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(space), -1).T


def discrepancy(space: DesignSpace, x: np.ndarray) -> float:
    """Centred L2 discrepancy of a sample -- lower is better coverage."""
    lo = np.array([v.lower for v in space.variables])
    hi = np.array([v.upper for v in space.variables])
    return float(qmc.discrepancy((np.asarray(x) - lo) / (hi - lo)))
