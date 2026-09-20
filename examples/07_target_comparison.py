#!/usr/bin/env python3
"""Which regression target should the surrogate use? Measured, not assumed.

The pipeline has regressed per-mode frequencies labelled by physics-based mode
families. That works, but families need a payload and a classification origin,
so they are the last part-specific thing in the critical path. Three
alternatives need nothing from the part at all:

    raw       the k lowest frequencies, sorted
    scalars   f_min, resonance risk, worst band separation
    cluster   near-degenerate groups matched to a reference by subspace MAC

Ordered eigenvalues are *continuous* in the design parameters -- a crossing
kinks f_(k)(x), it does not break it -- so the common assumption that sorted
targets are unusable deserves a number rather than an assertion.

**The metric that decides is the end-to-end verified objective**, not held-out
accuracy. A surrogate that ranks designs correctly beats a more accurate one
that does not, because the optimiser only ever uses the ranking.

Run:  python examples/07_target_comparison.py [--quick]
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr

from dynopt.doe import sobol_sample
from dynopt.fea.reference import ReferenceFrameSolver
from dynopt.geometry import CANTILEVER_ARM, MOTOR_MOUNT
from dynopt.optimize import (
    ClusterTargets,
    FamilyTargets,
    RawTargets,
    ResonanceObjective,
    ScalarTargets,
    extract_targets,
    optimize_adaptive,
    rpm_bands,
)
from dynopt.optimize.driver import _sample_dataset
from dynopt.surrogate import FrequencySurrogate

warnings.filterwarnings("ignore")


def cases():
    """Each part with an excitation its real hardware actually sees."""
    yield (
        MOTOR_MOUNT,
        ResonanceObjective(
            bands=rpm_bands(1000, 6000, orders=(1.0, 2.0, 4.0)),
            margin=0.25, w_mass=1.0, w_resonance=10.0,
            mass_ref=0.55, f_min=500.0,
        ),
        12,
    )
    yield (
        CANTILEVER_ARM,
        # Two-blade propeller, 3,000-9,000 RPM: 1X is 50-150 Hz and the arm's
        # first bending pair sits at ~114 Hz, right inside it.
        ResonanceObjective(
            bands=rpm_bands(3000, 9000, orders=(1.0, 2.0)),
            margin=0.25, w_mass=1.0, w_resonance=10.0,
            mass_ref=0.12, f_min=200.0,
        ),
        12,
    )


def target_sets(part, n_modes):
    sets = [("raw", RawTargets(6)), ("scalars", ScalarTargets()),
            ("cluster", ClusterTargets(6))]
    if part.mode_families is not None:
        sets.append(("family", FamilyTargets(part.mode_families)))
    return sets


def accuracy_table(part, objective, n_modes, n_train, n_test):
    """Held-out accuracy and, more importantly, rank correlation of J."""
    solver = ReferenceFrameSolver(part)
    space = part.design_space

    # Solve each DOE exactly once; every target set re-extracts from the same
    # results, so the comparison is on identical physics.
    Xtr = sobol_sample(space, n_train, seed=1)
    Xte = sobol_sample(space, n_test, seed=99)
    _, mtr, rtr, _ = _sample_dataset(solver, space, Xtr, n_modes, targets="raw")
    _, mte, rte, _ = _sample_dataset(solver, space, Xte, n_modes, targets="raw")

    true_J = np.array([
        objective(r.frequencies, m) if r is not None else np.nan
        for r, m in zip(rte, mte)
    ])

    rows = []
    for name, ts in target_sets(part, n_modes):
        Ftr = extract_targets(rtr, ts, objective)
        Fte = extract_targets(rte, ts, objective)
        ok = np.isfinite(mtr)

        t0 = time.perf_counter()
        sur = FrequencySurrogate(
            space, n_modes=Ftr.shape[1], transforms=ts.transforms
        ).fit(Xtr[ok], Ftr[ok], mtr[ok])
        t_fit = time.perf_counter() - t0

        P = sur.predict(Xte)
        Pm = sur.predict_mass(Xte)

        # Normalised RMSE, not MAPE. A column like min_sep crosses zero, so a
        # relative error blows up on samples that are perfectly well predicted
        # -- it measures the denominator, not the model. nRMSE divides by the
        # column's own spread, so every target is scored on the same footing.
        nrmse = []
        for j in range(Fte.shape[1]):
            m = np.isfinite(Fte[:, j]) & np.isfinite(P[:, j])
            if m.sum() < 3:
                continue
            spread = np.std(Fte[m, j])
            if spread <= 0:
                continue
            nrmse.append(np.sqrt(np.mean((P[m, j] - Fte[m, j]) ** 2)) / spread)
        col_err = 100.0 * float(np.mean(nrmse)) if nrmse else np.nan

        pred_J = np.array([
            ts.objective_from(P[i], float(Pm[i]), objective) for i in range(len(Xte))
        ])
        good = np.isfinite(pred_J) & np.isfinite(true_J)
        rho = spearmanr(pred_J[good], true_J[good]).statistic
        jerr = 100.0 * float(
            np.sqrt(np.mean((pred_J[good] - true_J[good]) ** 2)) / np.std(true_J[good])
        )
        cover = 100.0 * np.isfinite(Fte).mean()
        rows.append((name, len(ts.columns), col_err, jerr, rho, cover, t_fit))
    return rows


def run(quick: bool) -> int:
    n_train, n_test = (64, 64) if quick else (128, 128)
    n_init, n_rounds, n_per = (24, 3, 3) if quick else (40, 5, 3)

    print("=" * 78)
    print("  REGRESSION TARGET COMPARISON")
    print("=" * 78)

    verdicts = {}
    for part, objective, n_modes in cases():
        print(f"\n### {part.name}  ({len(part.design_space)} design variables)")
        print(f"\n  Held-out accuracy, {n_train} train / {n_test} test")
        print("  target     cols   col nRMSE   J nRMSE   rank rho   coverage   fit")
        print("  " + "-" * 70)
        for name, k, cerr, jerr, rho, cover, tfit in accuracy_table(
            part, objective, n_modes, n_train, n_test
        ):
            print(f"  {name:<10s} {k:>4d}   {cerr:>8.2f}%  {jerr:>8.2f}%  "
                  f"{rho:>8.4f}   {cover:>7.0f}%  {tfit:>4.1f}s")

        print(f"\n  End-to-end: optimise with each target, verify on the solver")
        print("  target       verified J    mass [g]    f1 [Hz]   violation   FEA   time")
        print("  " + "-" * 72)
        best = {}
        solver = ReferenceFrameSolver(part)
        for name, ts in target_sets(part, n_modes):
            obj = ResonanceObjective(
                bands=objective.bands, margin=objective.margin,
                w_mass=objective.w_mass, w_resonance=objective.w_resonance,
                mass_ref=objective.mass_ref, f_min=objective.f_min,
            )
            t0 = time.perf_counter()
            r = optimize_adaptive(
                solver, part.design_space, obj, n_init=n_init, n_rounds=n_rounds,
                n_per_round=n_per, n_modes=n_modes, targets=ts, verbose=False,
            )
            dt = time.perf_counter() - t0
            viol = obj.constraint_violation(r.frequencies)
            best[name] = r.objective
            print(f"  {name:<10s} {r.objective:>11.4f}  {r.mass*1000:>9.1f}  "
                  f"{r.frequencies[0]:>9.1f}  {viol:>10.4f}  {r.n_fea_calls:>4d}  {dt:>4.0f}s")
        verdicts[part.name] = best

    print("\n" + "=" * 78)
    print("  VERDICT  (lower J is better; within 2% of best counts as a tie)")
    print("=" * 78)
    simplicity = ["raw", "scalars", "cluster", "family"]
    for pname, best in verdicts.items():
        top = min(best.values())
        ties = [n for n in simplicity if n in best and best[n] <= top * 1.02 + 1e-9]
        print(f"  {pname:<16s} best={min(best, key=best.get):<9s} "
              f"J={top:.4f}   within 2%: {', '.join(ties)}")
    common = set.intersection(*[
        {n for n in simplicity if n in b and b[n] <= min(b.values()) * 1.02 + 1e-9}
        for b in verdicts.values()
    ])
    winner = next((n for n in simplicity if n in common), None)
    print(f"\n  Simplest target within 2% on BOTH parts: {winner or 'none -- families win'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller DOE, fewer rounds")
    sys.exit(run(ap.parse_args().quick))
