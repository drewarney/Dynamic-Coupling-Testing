#!/usr/bin/env python3
"""Phase 1 MVP: parameters -> modal FEA -> classified natural frequencies.

Run:  PYTHONPATH=src python3 examples/01_modal_mvp.py
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dynopt.dynamics import classify_modes, family_frequencies, modal_summary
from dynopt.fea.reference import modal_solve
from dynopt.geometry import MOTOR_MOUNT


def main() -> None:
    part = MOTOR_MOUNT
    space = part.design_space
    params = space.to_dict(space.defaults)

    print("DESIGN PARAMETERS")
    for k, v in params.items():
        print(f"  {k:<12s} {v*1000:7.2f} mm")

    model = part.build_frame(params)
    res = modal_solve(model, n_modes=12)
    origin = part.classification_origin(res)

    print(f"\nMODEL: {model.n_nodes} nodes, {model.n_dof} DOF, "
          f"{len(model.elements)} elements")
    print(f"MASS : {res.mass*1000:.1f} g "
          f"(structure {res.meta['structural_mass']*1000:.1f} g + motor)")

    labels = classify_modes(res, origin, part.mode_families, min_purity=0.35)
    print("\n  MODE    FREQ [Hz]   EFF.MASS-Z   TYPE")
    print("  " + "-" * 48)
    for row, lab in zip(modal_summary(res, direction=2), labels):
        print(f"  {row['mode']:>4d}   {row['f_hz']:9.1f}   "
              f"{row['eff_mass_frac']*100:8.1f}%   {lab}")

    print("\nPHYSICAL MODE FAMILIES (surrogate targets):")
    for k, v in family_frequencies(res, origin, part.mode_families).items():
        print(f"  {k:<14s} {v:8.1f} Hz" if np.isfinite(v) else f"  {k:<14s}   not found")


if __name__ == "__main__":
    main()
