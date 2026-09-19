r"""Active learning: spend the FEA budget where the surrogate is worst.

A uniform DOE wastes runs. The surrogate is already accurate in the flat parts
of the design space and bad near mode crossings and near the constraint
boundary -- which is exactly where the optimum lives. Active learning picks the
next design point by maximising an acquisition function over a cheap candidate
pool, so 150 well-chosen runs beat 400 uniform ones.

Two acquisitions are provided:

* :func:`max_variance` (pure exploration) -- drives down global surrogate error,
  best for building a reusable model of the whole design space.
* :func:`expected_improvement` (exploitation-weighted) -- drives down the
  objective, best when you only care about finding one good design.

.. math::
    \mathrm{EI}(x) = (f^* - \mu(x) - \xi)\,\Phi(Z) + \sigma(x)\,\phi(Z),
    \qquad Z = \frac{f^* - \mu(x) - \xi}{\sigma(x)}

where :math:`f^*` is the best objective seen so far and :math:`\xi` the
exploration margin.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..doe.sampling import sobol_sample
from ..geometry.base import DesignSpace


def max_variance(surrogate, candidates: np.ndarray, weights=None) -> np.ndarray:
    """Total predictive std across modes, optionally weighted per mode."""
    _, sd = surrogate.predict(candidates, return_std=True)
    rel = sd / np.clip(surrogate.predict(candidates), 1e-12, None)
    if weights is not None:
        rel = rel * np.asarray(weights)[None, : rel.shape[1]]
    return np.nansum(rel, axis=1)


def expected_improvement(
    mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01
) -> np.ndarray:
    """EI for a *minimisation* problem."""
    sd = np.clip(np.asarray(sd, dtype=float), 1e-12, None)
    imp = best - np.asarray(mu, dtype=float) - xi
    Z = imp / sd
    return imp * norm.cdf(Z) + sd * norm.pdf(Z)


def propose_batch(
    surrogate,
    space: DesignSpace,
    n_batch: int = 8,
    n_candidates: int = 4096,
    seed: int = 0,
    objective=None,
    best: float | None = None,
    min_distance: float = 0.05,
) -> np.ndarray:
    """Propose the next batch of FEA runs.

    Uses greedy selection with a **minimum-distance filter in normalised space**
    so a batch does not collapse onto a single hot spot -- important because you
    will run these FEA jobs in parallel and cannot update the model between
    them.

    Pass ``objective`` (a callable on predicted frequencies/mass) and ``best`` to
    switch from exploration to expected improvement.
    """
    cand = sobol_sample(space, n_candidates, seed=seed)

    if objective is None or best is None:
        score = max_variance(surrogate, cand)
    else:
        mu, sd = surrogate.predict(cand, return_std=True)
        obj_mu = np.array([objective(m) for m in mu])
        obj_sd = np.nanmean(sd, axis=1)
        score = -expected_improvement(obj_mu, obj_sd, best)  # negate: we take max
        score = -score

    order = np.argsort(score)[::-1]
    lo = np.array([v.lower for v in space.variables])
    hi = np.array([v.upper for v in space.variables])
    norm_cand = (cand - lo) / (hi - lo)

    chosen: list[int] = []
    for idx in order:
        if len(chosen) >= n_batch:
            break
        if all(
            np.linalg.norm(norm_cand[idx] - norm_cand[c]) >= min_distance
            for c in chosen
        ):
            chosen.append(int(idx))
    return cand[chosen]
