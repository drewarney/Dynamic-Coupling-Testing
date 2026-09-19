#!/usr/bin/env python3
"""Phase 1 MVP: parameters -> modal FEA -> classified natural frequencies.

Run:  PYTHONPATH=src python3 examples/01_modal_mvp.py
"""

import numpy as np

from dynopt.dynamics import classify_modes, family_frequencies, modal_summary
from dynopt.fea.reference import modal_solve
from dynopt.geometry import MOTOR_MOUNT_SPACE, build_frame, motor_node_index


def main() -> None:
    space = MOTOR_MOUNT_SPACE
    params = space.to_dict(space.defaults)

    print("DESIGN PARAMETERS")
    for k, v in params.items():
        print(f"  {k:<12s} {v*1000:7.2f} mm")

    model = build_frame(params)
    res = modal_solve(model, n_modes=12)
    origin = model.nodes[motor_node_index()]

    print(f"\nMODEL: {model.n_nodes} nodes, {model.n_dof} DOF, "
          f"{len(model.elements)} elements")
    print(f"MASS : {res.mass*1000:.1f} g "
          f"(structure {res.meta['structural_mass']*1000:.1f} g + motor)")

    labels = classify_modes(res, origin, min_purity=0.35)
    print("\n  MODE    FREQ [Hz]   EFF.MASS-Z   TYPE")
    print("  " + "-" * 48)
    for row, lab in zip(modal_summary(res, direction=2), labels):
        print(f"  {row['mode']:>4d}   {row['f_hz']:9.1f}   "
              f"{row['eff_mass_frac']*100:8.1f}%   {lab}")

    print("\nPHYSICAL MODE FAMILIES (surrogate targets):")
    for k, v in family_frequencies(res, origin).items():
        print(f"  {k:<14s} {v:8.1f} Hz" if np.isfinite(v) else f"  {k:<14s}   not found")


if __name__ == "__main__":
    main()
