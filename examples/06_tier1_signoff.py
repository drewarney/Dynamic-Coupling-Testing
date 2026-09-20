#!/usr/bin/env python3
"""Tier 1 sign-off: verify an optimised design on the 3D solid solver.

The optimiser runs on Tier 0 because it needs hundreds of solves. Tier 0 is a
beam idealisation, so its numbers are not quotable on their own no matter how
well calibrated. This script takes the optimiser's answer and re-solves it once
with full 3D quadratic tets in CalculiX, then checks the resonance margin
against that result.

Nothing in the benchmark table is published until this passes.

Usage::

    python examples/04_full_optimization.py --save optimum.json
    python examples/06_tier1_signoff.py optimum.json
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

RPM_LO, RPM_HI = 1000.0, 6000.0
ORDERS = (1.0, 2.0, 4.0)
MARGIN = 0.25
PAYLOAD_TONNE = 3.5e-4          # 350 g in mm/tonne/s
MESH_SIZES = (4.0, 3.0, 2.4)    # convergence check


def main(argv):
    try:
        import cadquery  # noqa: F401
        import gmsh  # noqa: F401
    except ImportError:
        print('needs the OSS extras:  pip install -e ".[oss]"')
        return 1
    if shutil.which("ccx") is None:
        print("CalculiX (ccx) not on PATH:  brew install calculix")
        return 1

    from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver
    from dynopt.fea.reference import modal_solve
    from dynopt.geometry import MOTOR_MOUNT, build_frame
    from dynopt.optimize import rpm_bands

    space = MOTOR_MOUNT.design_space
    if len(argv) > 1:
        params = json.load(open(argv[1]))["params"]
    else:
        params = space.to_dict(space.defaults)
        print("no design file given; signing off the BASELINE design\n")

    bands = rpm_bands(RPM_LO, RPM_HI, orders=ORDERS)
    cad = MOTOR_MOUNT

    print("=" * 70)
    print("  TIER 1 SIGN-OFF -- 3D solid verification")
    print("=" * 70)
    for k, v in params.items():
        print(f"  {k:<14s} {v*1000:8.3f} mm")

    # Material comes from the part; see example 05.
    def cfg(h):
        return CalculiXConfig(
            n_modes=8,
            mesh_size=h, mesh_size_min=min(0.8, h / 4),
            fix_selector="z_plane", fix_plane=0.0, fix_tol=1e-3,
            payload_mass=PAYLOAD_TONNE, payload_radius=0.0,
        )

    # Payload radius follows the plate, so it stays valid as the design moves.
    from dynopt.geometry.motor_mount import plate_geometry

    geom = plate_geometry(params, cad.motor_dia)
    radius = max(geom["hole_radius"] * 1000.0 * 1.6, 8.0)

    print("\n  mesh convergence:")
    result = None
    prev = None
    for h in MESH_SIZES:
        c = cfg(h)
        c.payload_radius = radius
        result = CalculiXSolver(cad, c).solve(params, n_modes=8)
        d = "" if prev is None else f"  (delta {100*(result.frequencies[0]-prev)/prev:+.2f}%)"
        print(f"    h={h:.1f} mm  {result.meta['n_nodes']:6d} nodes  "
              f"f1={result.frequencies[0]:8.1f} Hz{d}")
        prev = result.frequencies[0]

    beam = modal_solve(build_frame(params), n_modes=8)
    print(f"\n  Tier 0 (beam)      f1 = {beam.frequencies[0]:8.1f} Hz")
    print(f"  Tier 1 (3D solid)  f1 = {result.frequencies[0]:8.1f} Hz"
          f"   -> Tier 0 is {100*(beam.frequencies[0]-result.frequencies[0])/result.frequencies[0]:+.1f}%")

    print("\n  RESONANCE CLEARANCE, verified on Tier 1")
    print("    mode    freq [Hz]    min separation    required    status")
    print("    " + "-" * 56)
    ok = True
    for i, f in enumerate(result.frequencies[:6]):
        sep = min(b.separation(f) for b in bands)
        good = sep >= MARGIN
        ok &= good
        print(f"     {i+1:2d}    {f:9.1f}    {sep*100:11.1f}%    {MARGIN*100:7.0f}%    "
              f"{'PASS' if good else 'FAIL'}")

    print("\n  " + ("SIGNED OFF: every mode clears the excitation bands on the "
                    "3D solver." if ok else
                    "NOT SIGNED OFF: a mode violates the required margin."))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
