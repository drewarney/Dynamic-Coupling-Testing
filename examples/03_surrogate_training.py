#!/usr/bin/env python3
"""Phase 3: train and validate the GP surrogate.

Reproduces the accuracy numbers quoted in the README. Takes ~1 minute.

Run:  PYTHONPATH=src python3 examples/03_surrogate_training.py
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import time
import warnings

import numpy as np

from dynopt.doe import discrepancy, sobol_sample
from dynopt.fea.reference import ReferenceFrameSolver
from dynopt.geometry import MOTOR_MOUNT
from dynopt.optimize.driver import _sample_dataset
from dynopt.surrogate import FrequencySurrogate

warnings.filterwarnings("ignore")

N_TRAIN, N_TEST = 128, 200


def main() -> None:
    part = MOTOR_MOUNT
    space = part.design_space
    solver = ReferenceFrameSolver(part)

    X = sobol_sample(space, N_TRAIN, seed=1)
    print(f"[DOE] {N_TRAIN} Sobol points, centred L2 discrepancy "
          f"{discrepancy(space, X):.2e}")

    t0 = time.perf_counter()
    F, mass, _, cols = _sample_dataset(solver, space, X, 10)
    t_fea = time.perf_counter() - t0
    print(f"[FEA] {N_TRAIN} solves in {t_fea:.1f} s "
          f"({t_fea/N_TRAIN*1e3:.1f} ms each)")
    print(f"[FEA] target availability: "
          f"{dict(zip(cols, np.round(np.isfinite(F).mean(axis=0), 3)))}")

    ok = np.isfinite(mass)
    t0 = time.perf_counter()
    sur = FrequencySurrogate(space, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])
    print(f"[GP ] fitted {F.shape[1]} mode models + mass in "
          f"{time.perf_counter()-t0:.1f} s")

    Xt = sobol_sample(space, N_TEST, seed=99)
    Ft, mt, _, _ = _sample_dataset(solver, space, Xt, 10)
    P, S = sur.predict(Xt, return_std=True)

    print(f"\nHELD-OUT ACCURACY ({N_TRAIN} train / {N_TEST} test)")
    print("  TARGET          MAPE     p95     2-sigma coverage")
    print("  " + "-" * 50)
    for j, c in enumerate(cols):
        m = np.isfinite(Ft[:, j]) & np.isfinite(P[:, j])
        e = 100 * np.abs(P[m, j] - Ft[m, j]) / Ft[m, j]
        cov = 100 * np.mean(np.abs(P[m, j] - Ft[m, j]) <= 2 * S[m, j])
        print(f"  {c:<14s} {e.mean():5.2f}%  {np.percentile(e,95):6.2f}%   {cov:5.0f}%")
    Pm = sur.predict_mass(Xt)
    mm = np.isfinite(mt)
    print(f"  {'mass':<14s} {np.mean(100*np.abs(Pm[mm]-mt[mm])/mt[mm]):5.3f}%")

    t0 = time.perf_counter()
    sur.predict(Xt)
    t_sur = (time.perf_counter() - t0) / N_TEST
    print(f"\nSPEED: surrogate {t_sur*1e3:.3f} ms/design vs FEA "
          f"{t_fea/N_TRAIN*1e3:.1f} ms  ->  {(t_fea/N_TRAIN)/t_sur:.0f}x faster")

    print("\nGLOBAL SENSITIVITY (from learned GP length scales)")
    for j, c in enumerate(cols):
        rank = sur.sensitivity_ranking(j + 1)
        if rank:
            print(f"  {c:<14s} " + "  ".join(f"{n}:{v:.2f}" for n, v in rank[:3]))


if __name__ == "__main__":
    main()
