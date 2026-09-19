r"""Gaussian Process surrogate for modal frequencies.

Why a GP and not a neural net
-----------------------------
At 100-300 training points in 5-10 dimensions, a GP is the right tool:

* it interpolates exactly through the training data (FEA results are
  deterministic -- there is no observation noise to average out);
* it returns a **calibrated variance**, which is what makes active learning and
  trust-region optimisation possible.  An XGBoost or MLP surrogate gives you a
  number with no honest error bar, so you never know when to re-run FEA;
* it trains in under a second, so you can refit inside the optimisation loop.

Use a GNN/PyTorch surrogate when the *topology* changes between designs (mesh
in, frequencies out).  For a fixed parametric family with a handful of
dimensions, that is strictly more machinery for less accuracy.

Modelling choices that matter
-----------------------------
1. **Predict** :math:`\log f` **, not** :math:`f`.  Frequency scales as
   :math:`\sqrt{K/M}` and is strictly positive and multiplicative in the
   parameters; in log space the response is far closer to linear, which the
   Matern kernel fits with a much shorter length-scale budget.
2. **Anisotropic Matern-5/2 kernel.**  A separate length scale per variable lets
   the GP learn that frequency is very sensitive to ``height`` and weakly
   sensitive to ``plate_thk`` -- that is free sensitivity analysis.  Matern-5/2
   rather than RBF because modal response is twice-differentiable, not analytic.
3. **Mode tracking before fitting** (see :func:`dynopt.dynamics.track_modes`).
   Training on frequency-ordered modes teaches the surrogate the discontinuity
   at every mode crossing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..geometry.base import DesignSpace


@dataclass
class FrequencySurrogate:
    """Independent GP per tracked mode, plus one for mass.

    Mass is included because it is cheap to evaluate exactly -- but having it in
    the same object keeps the optimiser's objective a single call.
    """

    space: DesignSpace
    n_modes: int = 6
    log_target: bool = True
    nugget: float = 1e-8
    normalize_y: bool = True
    models: list = field(default_factory=list)
    mass_model: object | None = None
    _X: np.ndarray | None = None
    _Y: np.ndarray | None = None

    def _kernel(self) -> object:
        d = len(self.space)
        # Bounds are deliberately wide: a length scale that saturates at the
        # upper bound means "this variable barely matters", which is a valid
        # answer we want the optimiser to be able to express.  The WhiteKernel
        # is a numerical nugget, but it also absorbs the small discontinuities
        # left over where mode tracking swapped a mode -- so it needs headroom.
        return ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=np.ones(d), length_scale_bounds=(1e-3, 1e4), nu=2.5
        ) + WhiteKernel(self.nugget, noise_level_bounds=(1e-12, 1e-1))

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """GPs need unit-cube inputs or the length scales are meaningless."""
        return np.array([self.space.normalize(x) for x in np.atleast_2d(X)])

    def fit(self, X: np.ndarray, F: np.ndarray, mass: np.ndarray | None = None):
        """Fit on design points ``X`` (n, d) and tracked frequencies ``F`` (n, k).

        NaNs in ``F`` (a mode that could not be tracked at that design point) are
        dropped per-mode rather than per-row, so one untracked mode does not
        throw away the whole sample.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = np.atleast_2d(np.asarray(F, dtype=float))
        self._X, self._Y = X, F
        Xs = self._scale(X)

        self.models = []
        for j in range(min(self.n_modes, F.shape[1])):
            y = F[:, j]
            ok = np.isfinite(y) & (y > 0)
            if ok.sum() < 3:
                self.models.append(None)
                continue
            t = np.log(y[ok]) if self.log_target else y[ok]
            gp = GaussianProcessRegressor(
                kernel=self._kernel(),
                normalize_y=self.normalize_y,
                n_restarts_optimizer=4,
                random_state=0,
            )
            gp.fit(Xs[ok], t)
            self.models.append(gp)

        if mass is not None:
            mass = np.ravel(np.asarray(mass, dtype=float))
            gp = GaussianProcessRegressor(
                kernel=self._kernel(), normalize_y=True,
                n_restarts_optimizer=4, random_state=0,
            )
            gp.fit(Xs, np.log(mass) if self.log_target else mass)
            self.mass_model = gp
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predicted frequencies (Hz), shape ``(n, n_modes)``.

        The standard deviation is returned in **Hz**, converted out of log space
        by the delta method: :math:`\\sigma_f \\approx f\\,\\sigma_{\\log f}`.
        """
        Xs = self._scale(X)
        n = Xs.shape[0]
        k = len(self.models)
        mu = np.full((n, k), np.nan)
        sd = np.zeros((n, k))
        for j, gp in enumerate(self.models):
            if gp is None:
                continue
            if return_std:
                m, s = gp.predict(Xs, return_std=True)
            else:
                m, s = gp.predict(Xs), np.zeros(n)
            if self.log_target:
                mu[:, j] = np.exp(m)
                sd[:, j] = mu[:, j] * s
            else:
                mu[:, j], sd[:, j] = m, s
        return (mu, sd) if return_std else mu

    def predict_mass(self, X: np.ndarray) -> np.ndarray:
        if self.mass_model is None:
            raise ValueError("surrogate was not fitted with mass data")
        m = self.mass_model.predict(self._scale(X))
        return np.exp(m) if self.log_target else m

    def length_scales(self) -> dict[str, np.ndarray]:
        r"""Learned per-variable length scales, one row per mode.

        A *short* length scale means the frequency changes rapidly along that
        axis -- i.e. that variable is a strong lever.  This is an automatic,
        normalised global sensitivity ranking that costs nothing extra.
        """
        out = {}
        for j, gp in enumerate(self.models):
            if gp is None:
                continue
            for part in [gp.kernel_]:
                ls = _find_length_scale(part)
                if ls is not None:
                    out[f"mode{j+1}"] = np.atleast_1d(ls)
        return out

    def sensitivity_ranking(self, mode: int = 1) -> list[tuple[str, float]]:
        """Variables ranked by influence on ``mode`` (1-based), most first."""
        ls = self.length_scales().get(f"mode{mode}")
        if ls is None:
            return []
        score = 1.0 / np.clip(ls, 1e-12, None)
        score = score / score.sum()
        return sorted(zip(self.space.names, score), key=lambda t: -t[1])

    def loo_error(self) -> np.ndarray:
        """Leave-one-out relative error per mode, in percent.

        Cheap honest validation: a GP's LOO prediction is available in closed
        form, but refitting is fast enough here and avoids kernel-specific
        algebra.  Report this number in your write-up -- it is the headline
        accuracy claim for the surrogate.
        """
        if self._X is None or self._Y is None:
            raise ValueError("call fit() first")
        n = self._X.shape[0]
        errs = np.full((n, len(self.models)), np.nan)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sub = FrequencySurrogate(
                self.space, n_modes=self.n_modes, log_target=self.log_target
            ).fit(self._X[mask], self._Y[mask])
            pred = sub.predict(self._X[i : i + 1])[0]
            truth = self._Y[i, : pred.size]
            ok = np.isfinite(truth) & np.isfinite(pred) & (truth > 0)
            errs[i, : pred.size][ok] = 100.0 * np.abs(pred[ok] - truth[ok]) / truth[ok]
        return np.nanmean(errs, axis=0)


def _find_length_scale(kernel):
    """Walk a composite sklearn kernel and pull out the Matern length scale."""
    if hasattr(kernel, "length_scale"):
        return np.asarray(kernel.length_scale, dtype=float)
    for attr in ("k1", "k2"):
        sub = getattr(kernel, attr, None)
        if sub is not None:
            found = _find_length_scale(sub)
            if found is not None:
                return found
    return None
