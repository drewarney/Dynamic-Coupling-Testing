#!/usr/bin/env python3
"""Phase 2: cross-modal coupling and harmonic response.

Demonstrates the project's central claim -- that assuming modes decouple gives
a materially different (and optimistic) answer when damping is non-proportional.

Run:  PYTHONPATH=src python3 examples/02_coupling_analysis.py
"""

import numpy as np

from dynopt.dynamics import (
    campbell_diagram,
    coupled_pairs,
    coupling_index,
    discrete_damper_matrix,
    modal_damping_matrix,
    modal_frf,
    rayleigh_damping,
    rms_response,
)
from dynopt.fea.reference import modal_solve
from dynopt.geometry import MOTOR_MOUNT, motor_node_index


def main() -> None:
    part = MOTOR_MOUNT
    space = part.design_space
    model = part.build_frame(space.to_dict(space.defaults))
    res = modal_solve(model, n_modes=10)
    node = motor_node_index()

    print("Natural frequencies [Hz]:", np.round(res.frequencies, 1))

    # --- 1. proportional damping: no cross-modal coupling by construction ----
    C_prop, alpha, beta = rayleigh_damping(res.M, res.K, 0.02, 250.0, 0.02, 2500.0)
    kap_prop = coupling_index(modal_damping_matrix(res, C_prop))
    print(f"\nRayleigh damping: alpha={alpha:.4g}, beta={beta:.4g}")
    print(f"  max off-diagonal coupling kappa = {kap_prop.max():.2e}  (expected ~0)")

    # --- 2. discrete isolator bushings: non-proportional ---------------------
    dampers = [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)]
    C_disc = discrete_damper_matrix(model.n_dof, dampers)
    kap = coupling_index(modal_damping_matrix(res, C_disc))
    print(f"\nDiscrete dampers at the payload (non-proportional):")
    print(f"  max off-diagonal coupling kappa = {kap.max():.3f}")

    pairs = coupled_pairs(res, C_disc, threshold=0.25)
    if pairs:
        print("\n  COUPLED MODE PAIRS (energy-transfer index eta):")
        for p in pairs[:6]:
            print(f"    modes {p['mode_i']}-{p['mode_j']}: "
                  f"{p['f_i']:7.1f} / {p['f_j']:7.1f} Hz   eta={p['eta']:.3f}")
        print("\n  -> These pairs must be separated TOGETHER. Pushing one mode")
        print("     up will drag its partner along.")
    else:
        print("  no strongly coupled pairs above threshold")

    # --- 3. does the coupling actually change the answer? -------------------
    F = np.zeros(model.n_dof)
    F[node * 6 + 0] = 1.0                       # lateral unit force at the motor
    freqs = np.linspace(100.0, 1500.0, 600)
    out = [node * 6 + 0]

    Hc = modal_frf(res, C_disc, freqs, F, dofs_out=out, keep_coupling=True)[0]
    Hu = modal_frf(res, C_disc, freqs, F, dofs_out=out, keep_coupling=False)[0]

    peak_c, peak_u = np.abs(Hc).max(), np.abs(Hu).max()
    rms_c = rms_response(Hc[None, :], freqs)[0]
    rms_u = rms_response(Hu[None, :], freqs)[0]

    print("\nHARMONIC RESPONSE, 100-1500 Hz, lateral drive at the payload")
    print(f"  peak |H|  coupled {peak_c:.4e}   uncoupled {peak_u:.4e}   "
          f"({100*(peak_u-peak_c)/peak_c:+.1f}% error if coupling ignored)")
    print(f"  band RMS  coupled {rms_c:.4e}   uncoupled {rms_u:.4e}   "
          f"({100*(rms_u-rms_c)/rms_c:+.1f}%)")

    # --- 4. Campbell diagram -------------------------------------------------
    cam = campbell_diagram(res.frequencies[:4], (1000.0, 6000.0),
                           orders=(1.0, 2.0, 4.0), gyro_coeff=0.0)
    print("\nCAMPBELL CROSSINGS inside 1,000-6,000 RPM:")
    if cam["crossings"]:
        for c in cam["crossings"]:
            print(f"  {c['branch']:<10s} x {c['order']:<3s} at "
                  f"{c['rpm']:7.0f} RPM ({c['hz']:.1f} Hz)")
    else:
        print("  none -- all modes clear of 1X/2X/4X over the operating range")


if __name__ == "__main__":
    main()
