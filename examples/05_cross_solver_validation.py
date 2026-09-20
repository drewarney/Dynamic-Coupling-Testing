#!/usr/bin/env python3
"""Cross-solver validation: beam idealisation (Tier 0) vs 3D solid FEA (Tier 1).

Every surrogate and every optimisation result in this project is computed on
Tier 0. If Tier 0 is wrong, all of it is wrong. This script checks it against an
independent discretisation -- a full 3D quadratic-tet solve in CalculiX -- which
shares no code, no element formulation and no mesh with the beam model.

Requires the open-source stack::

    pip install -e ".[oss]"
    brew install calculix        # or: apt install calculix-ccx

Run:  python examples/05_cross_solver_validation.py
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MESH_SIZES = (5.0, 4.0, 3.0)


def main() -> int:
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

    space = MOTOR_MOUNT.design_space
    params = space.to_dict(space.defaults)
    cad = MOTOR_MOUNT

    # CadQuery exports millimetres, so the solver runs in mm / tonne / s:
    # No E/nu/rho here: the part supplies them, so the two tiers cannot end up
    # modelling different materials. (They are AL6061 either way, in MPa and
    # tonne/mm^3 -- feeding a mm model SI values scales every frequency ~31x.)
    def cfg(h, payload=0.0):
        return CalculiXConfig(
            n_modes=8,
            mesh_size=h, mesh_size_min=min(0.8, h / 4),
            fix_selector="z_plane", fix_plane=0.0, fix_tol=1e-3,
            payload_mass=payload, payload_radius=24.0,
        )

    print("=" * 68)
    print("  CROSS-SOLVER VALIDATION -- beam idealisation vs 3D solid")
    print("=" * 68)

    solid_vol = cad.build(params).val().Volume()
    solid_mass = solid_vol * 1e-9 * 2700.0
    beam_bare = modal_solve(
        build_frame(params, motor_mass=0.0, motor_inertia=(0.0, 0.0, 0.0)), n_modes=8
    )
    print(f"\n  structural mass:  beam {beam_bare.mass*1000:6.1f} g   "
          f"3D solid {solid_mass*1000:6.1f} g   "
          f"(beam/3D = {beam_bare.mass/solid_mass:.2f}x)")

    print("\n  3D mesh convergence (bare structure):")
    f3d = None
    for h in MESH_SIZES:
        r = CalculiXSolver(cad, cfg(h)).solve(params, n_modes=6)
        d = "" if f3d is None else f"   (delta {100*(r.frequencies[0]-f3d)/f3d:+.2f}%)"
        print(f"    h={h:.1f} mm  {r.meta['n_nodes']:6d} nodes  "
              f"f1={r.frequencies[0]:8.1f} Hz{d}")
        f3d = r.frequencies[0]

    bare_err = 100 * (beam_bare.frequencies[0] - f3d) / f3d
    print(f"\n  BARE STRUCTURE  (payload_mass=0.0, explicitly)")
    print(f"    beam f1      {beam_bare.frequencies[0]:8.1f} Hz")
    print(f"    3D   f1      {f3d:8.1f} Hz     beam is {bare_err:+.1f}%")
    corrected = beam_bare.frequencies[0] * np.sqrt(beam_bare.mass / solid_mass)
    rescaled_err = 100 * (corrected - f3d) / f3d
    print(f"    beam f1 rescaled by sqrt(m_beam/m_3d) = {corrected:8.1f} Hz     "
          f"residual {rescaled_err:+.1f}%")
    if abs(bare_err) < abs(rescaled_err):
        print("    -> unloaded, the two tiers already agree; the mass difference"
              "\n       is not what drives the error, and correcting for it hurts.")

    print("\n  LOADED (350 g payload)")
    loaded = CalculiXSolver(cad, cfg(3.0, payload=3.5e-4)).solve(params, n_modes=8)
    beam = modal_solve(build_frame(params), n_modes=8)
    print("    mode      beam [Hz]     3D [Hz]      error")
    for i in range(min(4, loaded.n_modes)):
        e = 100 * (beam.frequencies[i] - loaded.frequencies[i]) / loaded.frequencies[i]
        print(f"      {i+1}     {beam.frequencies[i]:9.1f}   {loaded.frequencies[i]:9.1f}   {e:+8.1f}%")

    # The diagnostic this example exists for: not "how wrong is the beam tier"
    # but *where* it goes wrong. Both tiers are solving the same structure, so
    # if they agree bare and disagree loaded, the error is in how each carries
    # the payload -- not in the structural idealisation.
    drop_beam = beam_bare.frequencies[0] / beam.frequencies[0]
    drop_3d = f3d / loaded.frequencies[0]
    loaded_err = 100 * (beam.frequencies[0] - loaded.frequencies[0]) / loaded.frequencies[0]
    print(f"\n  WHERE THE ERROR LIVES")
    print(f"    unloaded, beam vs 3D          {bare_err:+8.1f}%")
    print(f"    loaded,   beam vs 3D          {loaded_err:+8.1f}%")
    print(f"    adding the payload divides f1 by {drop_beam:5.2f}x (beam) "
          f"vs {drop_3d:5.2f}x (3D)")
    if abs(bare_err) < 0.25 * abs(loaded_err):
        print("    -> the structural idealisation is sound; the beam tier is far")
        print("       more sensitive to the payload than the solid is. A point")
        print("       mass on a 4-edge-beam plate has no membrane action to")
        print("       resist it, so the plate is locally far too soft.")

    print("\n  VERDICT: the beam tier under-predicts loaded frequency, i.e. it is")
    print("  conservative for resonance avoidance but not quantitatively")
    print("  accurate. Use it for DOE and surrogate training; re-verify any")
    print("  final design on Tier 1 before quoting numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
