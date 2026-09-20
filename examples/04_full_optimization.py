#!/usr/bin/env python3
"""Phase 3+4: the full closed loop, with a before/after benchmark report.

This is the script to run for the portfolio result.

Run:  PYTHONPATH=src python3 examples/04_full_optimization.py
"""

import argparse
import json
import warnings

import numpy as np

from dynopt.dynamics import classify_modes, family_frequencies
from dynopt.fea.reference import ReferenceFrameSolver, modal_solve
from dynopt.geometry import MOTOR_MOUNT
from dynopt.optimize import (
    ResonanceObjective,
    evaluate_baseline,
    optimize_adaptive,
    rpm_bands,
)

warnings.filterwarnings("ignore")

RPM_LO, RPM_HI = 1000.0, 6000.0


def describe(params, title):
    part = MOTOR_MOUNT
    res = modal_solve(part.build_frame(params), n_modes=12)
    fams = family_frequencies(
        res, part.classification_origin(res), part.mode_families
    )
    print(f"\n{title}")
    print(f"  mass {res.mass*1000:7.1f} g")
    for k, v in fams.items():
        print(f"  {k:<14s} {v:8.1f} Hz" if np.isfinite(v) else f"  {k:<14s}      --")
    return res, fams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", help="write the optimised design to JSON "
                                   "for examples/06_tier1_signoff.py")
    args = ap.parse_args()

    part = MOTOR_MOUNT
    space = part.design_space
    solver = ReferenceFrameSolver(part)
    bands = rpm_bands(RPM_LO, RPM_HI, orders=(1.0, 2.0, 4.0))

    print("=" * 66)
    print("  RESONANCE OPTIMISATION -- 4-LEG MOTOR MOUNT")
    print("=" * 66)
    print(f"  operating range : {RPM_LO:.0f}-{RPM_HI:.0f} RPM")
    for b in bands:
        print(f"  avoid {b.label:<22s} {b.f_lo:7.1f} - {b.f_hi:7.1f} Hz  "
              f"(weight {b.weight:.2f})")

    obj = ResonanceObjective(
        bands=bands, margin=0.25, w_mass=1.0, w_resonance=10.0,
        w_separation=0.5, mass_ref=0.55, f_min=RPM_HI * 4 / 60 * 1.25,
    )

    base = evaluate_baseline(solver, space, obj)
    describe(base["params"], "BASELINE DESIGN")
    print(f"  objective {base['objective']:.4f}   "
          f"constraint violation {base['violation']:.4f}")

    print("\nOPTIMISING (efficient global optimisation) ...")
    result = optimize_adaptive(
        solver, space, obj, n_init=40, n_rounds=6, n_per_round=4,
        n_modes=12, verbose=True,
    )
    result.baseline = base

    describe(result.params, "OPTIMISED DESIGN")
    print(result.report())

    print("\nVERIFICATION -- every mode vs every excitation band")
    res = modal_solve(part.build_frame(result.params), n_modes=12)
    labels = classify_modes(
        res, part.classification_origin(res), part.mode_families, 0.35
    )
    ok = True
    for f, lab in zip(res.frequencies[:6], labels[:6]):
        worst = min(b.separation(f) for b in bands)
        status = "OK " if worst >= obj.margin else "FAIL"
        ok &= worst >= obj.margin
        print(f"  {status}  {f:8.1f} Hz  ({lab:<11s})  "
              f"min separation {worst*100:6.1f}%  (required {obj.margin*100:.0f}%)")
    print(f"\n  RESULT: {'all modes clear' if ok else 'margin not met'} (Tier 0)")
    print("  NOTE: Tier 0 is a beam idealisation. These numbers are not")
    print("  quotable until verified with examples/06_tier1_signoff.py.")

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(
                {
                    "params": result.params,
                    "tier0_frequencies": result.frequencies.tolist(),
                    "mass": result.mass,
                    "objective": result.objective,
                    "n_fea_calls": result.n_fea_calls,
                    "baseline": {
                        "params": base["params"],
                        "frequencies": base["frequencies"].tolist(),
                        "mass": base["mass"],
                        "objective": base["objective"],
                    },
                },
                fh,
                indent=2,
            )
        print(f"\n  saved {args.save}")


if __name__ == "__main__":
    main()
