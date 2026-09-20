r"""Multi-fidelity correction: make a cheap solver trustworthy.

The pipeline has a beam tier that runs in 11 ms and a 3D tier that runs in
~40 s. Cross-solver validation measured the beam tier **31% off** on the design
the optimiser returned, so its numbers are not quotable -- and a DOE on the 3D
tier costs hours. Neither tier alone is both fast and right.

The fix is the standard Kennedy-O'Hagan construction: model the *discrepancy*
between tiers, not each tier separately. Frequencies are positive and
multiplicative, and :class:`FrequencySurrogate` already works in log space, so
the natural form is a log-ratio:

.. math:: \log f_{fine}(x) = \log f_{cheap}(x) + r(x), \qquad r = \log\rho

``r`` is fitted **only on paired designs** -- points where both tiers were run
-- where it is measured directly rather than inferred, so the correction never
learns the cheap surrogate's own error.

Three correction forms are provided, and which one wins is a measurement:

``constant``
    One scalar per column. The null hypothesis, and the one to prefer if it
    ties: if mesh refinement scales every design's frequencies by roughly the
    same factor, a GP is unnecessary machinery.
``multiplicative``
    GP on the log-ratio. Expected best for frequencies.
``additive``
    GP on the raw difference. Right when the discrepancy is an offset rather
    than a scaling.

**Column correspondence is an assumption.** The correction pairs sorted column
*k* of the cheap tier with column *k* of the fine tier. That holds when both
tiers resolve the same modes in the same order; it fails silently the moment
the cheap tier misses a mode the fine tier finds, which misaligns every column
above it and produces a confident wrong answer. :meth:`correspondence_report`
exposes the per-column ratio spread so the assumption can be checked rather
than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..geometry.base import DesignSpace
from .gp import FrequencySurrogate

CORRECTION_MODES = ("constant", "multiplicative", "additive")


@dataclass
class MultiFidelitySurrogate:
    """A cheap-tier surrogate plus a learned correction to the fine tier.

    Usage::

        mf = MultiFidelitySurrogate(space, mode="multiplicative")
        mf.fit(X_cheap, F_cheap, X_pair, F_pair_cheap, F_pair_fine)
        F_hat = mf.predict(X_new)      # fine-tier estimate, cheap-tier cost
    """

    space: DesignSpace
    mode: str = "multiplicative"
    nugget: float = 1e-8
    #: Per-column transform handed to the cheap-tier surrogate.
    transforms: tuple[str, ...] | None = None
    cheap: FrequencySurrogate | None = None
    corrections: list = field(default_factory=list)
    constants: np.ndarray | None = None
    _pair_ratio: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mode not in CORRECTION_MODES:
            raise ValueError(
                f"mode must be one of {CORRECTION_MODES}, got {self.mode!r}"
            )

    # -- internals -----------------------------------------------------------
    def _kernel(self):
        d = len(self.space)
        return ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e4), nu=2.5
        ) + WhiteKernel(self.nugget, noise_level_bounds=(1e-12, 1e-1))

    def _scale(self, X):
        return np.array([self.space.normalize(x) for x in np.atleast_2d(X)])

    # -- fitting -------------------------------------------------------------
    def fit(
        self,
        X_cheap: np.ndarray,
        F_cheap: np.ndarray,
        X_pair: np.ndarray,
        F_pair_cheap: np.ndarray,
        F_pair_fine: np.ndarray,
    ) -> "MultiFidelitySurrogate":
        """Fit the cheap surrogate, then the discrepancy on the paired subset.

        ``X_cheap``/``F_cheap`` are the many cheap-tier samples. The ``_pair``
        arrays are the few designs solved on *both* tiers; they may overlap
        ``X_cheap`` and usually should.
        """
        X_cheap = np.atleast_2d(np.asarray(X_cheap, float))
        F_cheap = np.atleast_2d(np.asarray(F_cheap, float))
        X_pair = np.atleast_2d(np.asarray(X_pair, float))
        F_pair_cheap = np.atleast_2d(np.asarray(F_pair_cheap, float))
        F_pair_fine = np.atleast_2d(np.asarray(F_pair_fine, float))

        if F_pair_cheap.shape[1] != F_pair_fine.shape[1]:
            raise ValueError(
                f"tier column counts differ: cheap has {F_pair_cheap.shape[1]}, "
                f"fine has {F_pair_fine.shape[1]}. Correcting mismatched columns "
                "pairs unrelated modes and returns a confident wrong answer -- "
                "solve both tiers for the same number of modes."
            )
        if F_cheap.shape[1] != F_pair_cheap.shape[1]:
            raise ValueError(
                f"cheap DOE has {F_cheap.shape[1]} columns but the paired cheap "
                f"samples have {F_pair_cheap.shape[1]}"
            )

        self.cheap = FrequencySurrogate(
            self.space, n_modes=F_cheap.shape[1], transforms=self.transforms
        ).fit(X_cheap, F_cheap)

        k = F_pair_cheap.shape[1]
        Xs = self._scale(X_pair)
        self.corrections = [None] * k
        self.constants = np.zeros(k)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log(F_pair_fine) - np.log(F_pair_cheap)
        self._pair_ratio = ratio

        for j in range(k):
            if self.mode == "additive":
                y = F_pair_fine[:, j] - F_pair_cheap[:, j]
            else:
                y = ratio[:, j]
            ok = np.isfinite(y)
            if ok.sum() == 0:
                continue
            self.constants[j] = float(np.mean(y[ok]))
            if self.mode == "constant" or ok.sum() < 4:
                continue
            gp = GaussianProcessRegressor(
                kernel=self._kernel(), normalize_y=True,
                n_restarts_optimizer=4, random_state=0,
            )
            gp.fit(Xs[ok], y[ok])
            self.corrections[j] = gp
        return self

    # -- prediction ----------------------------------------------------------
    def predict(self, X: np.ndarray, return_std: bool = False):
        """Corrected fine-tier estimate at cheap-tier cost."""
        if self.cheap is None:
            raise ValueError("call fit() first")
        base, base_sd = self.cheap.predict(X, return_std=True)
        Xs = self._scale(X)
        n, k = base.shape

        corr = np.zeros((n, k))
        corr_sd = np.zeros((n, k))
        for j in range(k):
            gp = self.corrections[j] if j < len(self.corrections) else None
            if gp is None:
                corr[:, j] = self.constants[j] if self.constants is not None else 0.0
                continue
            m, s = gp.predict(Xs, return_std=True)
            corr[:, j], corr_sd[:, j] = m, s

        if self.mode == "additive":
            mu = base + corr
            # Independent contributions; keeps the calibration story intact.
            sd = np.sqrt(base_sd**2 + corr_sd**2)
        else:
            mu = base * np.exp(corr)
            rel = np.sqrt((base_sd / np.clip(base, 1e-30, None)) ** 2 + corr_sd**2)
            sd = mu * rel
        return (mu, sd) if return_std else mu

    # -- diagnostics ---------------------------------------------------------
    def correspondence_report(self) -> list[dict]:
        r"""Per-column discrepancy statistics, for checking the pairing holds.

        A column whose ratio :math:`\rho` is tight across designs is a column
        where the two tiers are describing the same mode. A wide or bimodal
        spread is the signature of misalignment -- the cheap tier missing a mode
        the fine tier resolves -- and means the correction for that column is
        averaging unrelated physics.
        """
        if self._pair_ratio is None:
            raise ValueError("call fit() first")
        out = []
        for j in range(self._pair_ratio.shape[1]):
            r = self._pair_ratio[:, j]
            r = r[np.isfinite(r)]
            if r.size == 0:
                out.append({"column": j, "n": 0})
                continue
            rho = np.exp(r)
            out.append({
                "column": j,
                "n": int(r.size),
                "ratio_mean": float(rho.mean()),
                "ratio_std": float(rho.std()),
                "ratio_cv": float(rho.std() / max(abs(rho.mean()), 1e-30)),
                "ratio_min": float(rho.min()),
                "ratio_max": float(rho.max()),
            })
        return out

    def constant_is_enough(self, tol: float = 0.05) -> bool:
        """True when every column's ratio varies less than ``tol`` across designs.

        If this holds, the per-column scalar reproduces the GP and the extra
        machinery is not earning its place.
        """
        rep = self.correspondence_report()
        return all(r.get("ratio_cv", np.inf) <= tol for r in rep if r.get("n", 0) > 0)
