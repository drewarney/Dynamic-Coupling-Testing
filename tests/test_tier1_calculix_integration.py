"""End-to-end Tier 1 integration: CadQuery -> STEP -> Gmsh -> CalculiX.

Every bug this file guards against produced a *confident wrong answer* rather
than an exception. The backend ran, the solver converged, and the numbers were
physically meaningless:

* legs protruding below the mounting plane, so a geometric face selector
  clamped four isolated leg tips instead of the bolt-down faces;
* a STEP carrying five disjoint volumes because the CAD union never fused,
  meshed independently and therefore mechanically disconnected;
* ``ELSET=EALL`` in the deck, a convention Gmsh never emits;
* Gmsh's 1D/2D elements handed to a solid solver;
* an empty node selection solving happily instead of raising.

Unit tests on the beam tier cannot catch any of these. Only actually invoking
the toolchain can, which is what these tests do.

Skipped automatically when the optional stack is absent; run them with::

    pip install -e ".[oss]"     # CadQuery + Gmsh
    brew install calculix       # or: apt install calculix-ccx
"""

import shutil

import numpy as np
import pytest

cq = pytest.importorskip("cadquery", reason="CadQuery not installed")
gmsh = pytest.importorskip("gmsh", reason="Gmsh not installed")
pytestmark = pytest.mark.skipif(
    shutil.which("ccx") is None, reason="CalculiX (ccx) not on PATH"
)

from dynopt.fea.calculix import (  # noqa: E402
    CalculiXConfig,
    CalculiXSolver,
    _select_fixed_nodes,
    _split_mesh_inp,
    count_volumes,
    mesh_step,
)
from dynopt.geometry import MOTOR_MOUNT, MOTOR_MOUNT_SPACE  # noqa: E402

SPACE = MOTOR_MOUNT_SPACE


def _cfg(**kw):
    """mm / tonne / s. Material is left to the part, which owns it."""
    base = dict(
        n_modes=6,
        mesh_size=5.0, mesh_size_min=1.2,
        fix_selector="z_plane", fix_plane=0.0, fix_tol=1e-3,
    )
    base.update(kw)
    return CalculiXConfig(**base)


@pytest.fixture(scope="module")
def step_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("tier1") / "mount.step"
    MOTOR_MOUNT.export_step(SPACE.to_dict(SPACE.defaults), str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_cad_produces_single_contiguous_volume(step_file):
    """The STEP must contain exactly one solid.

    Legs that merely touch the plate face produce no boolean intersection in
    OpenCASCADE, so `union` silently returns a compound. Meshed, the pieces
    share no nodes and the assembly falls apart.
    """
    n_vol = count_volumes(step_file)
    assert n_vol == 1, f"STEP has {n_vol} disjoint volumes; the CAD union did not fuse"


def test_part_has_flat_mounting_face_at_origin(step_file, tmp_path):
    """Nothing may hang below z = 0, and real area must sit on that plane."""
    solid = MOTOR_MOUNT.build(SPACE.to_dict(SPACE.defaults))
    bb = solid.val().BoundingBox()
    assert bb.zmin == pytest.approx(0.0, abs=1e-6), (
        f"geometry extends to z={bb.zmin:.3f}; there is no flat bolt-down face"
    )

    coords, _, _ = mesh_step(step_file, str(tmp_path / "flat.inp"), _cfg())
    on_plane = np.abs(coords[:, 2]) <= 1e-3
    assert on_plane.sum() > 50, (
        f"only {on_plane.sum()} nodes on z=0; the mounting face is not planar"
    )


# ---------------------------------------------------------------------------
# Meshing and deck generation
# ---------------------------------------------------------------------------
def test_mesh_is_second_order_and_solid_only(step_file, tmp_path):
    """Gmsh also meshes curves and surfaces; only C3D10 may reach the solver.

    Linear tets are additionally rejected by design -- they are over-stiff and
    report modal frequencies 10-25% high.
    """
    inp = tmp_path / "mesh.inp"
    mesh_step(step_file, str(inp), _cfg())
    nodes, elems = _split_mesh_inp(str(inp), "C3D10")
    assert nodes and elems
    assert all(len(e.split(",")) == 11 for e in elems[:50]), "expected 10-node tets"

    raw = inp.read_text().upper()
    assert "T3D3" in raw or "CPS6" in raw, "fixture no longer exercises the filter"
    assert "T3D3" not in "".join(elems) and "CPS6" not in "".join(elems)


def test_deck_defines_the_element_set_it_references(step_file, tmp_path):
    """Gmsh never emits EALL; referencing it aborts ccx before it solves."""
    from dynopt.fea.calculix import write_modal_deck

    inp = tmp_path / "mesh.inp"
    coords, tags, _ = mesh_step(step_file, str(inp), _cfg())
    cfg = _cfg()
    fixed = _select_fixed_nodes(coords, tags, cfg)
    deck = tmp_path / "job.inp"
    write_modal_deck(str(inp), str(deck), fixed, cfg)

    text = deck.read_text()
    assert "ELSET=EALL" not in text.upper()
    referenced = [
        ln.split("ELSET=")[1].split(",")[0].strip()
        for ln in text.splitlines()
        if ln.upper().startswith("*SOLID SECTION")
    ]
    assert referenced, "no *SOLID SECTION in deck"
    for name in referenced:
        assert f"ELSET={name}" in text.replace(" ", ""), f"{name} never defined"


def test_empty_node_selection_raises(step_file, tmp_path):
    """A selector that matches nothing must fail loudly, not solve nonsense."""
    coords, tags, _ = mesh_step(step_file, str(tmp_path / "m.inp"), _cfg())
    bad = _cfg(fix_selector="z_plane", fix_plane=-999.0, fix_tol=1e-6)
    with pytest.raises(ValueError, match="matched no nodes"):
        _select_fixed_nodes(coords, tags, bad)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tier1_result(step_file):
    return CalculiXSolver(MOTOR_MOUNT, _cfg()).solve(
        SPACE.to_dict(SPACE.defaults), n_modes=6
    )


def test_no_rigid_body_modes(tier1_result):
    """Zero-frequency modes mean part of the structure is unconstrained.

    This is the signature of the disjoint-volume bug: the feet were clamped and
    the platform floated free, giving six rigid-body modes and a solve that
    "succeeded".
    """
    f = tier1_result.frequencies
    n_rigid = int((f < 1.0).sum())
    assert n_rigid == 0, (
        f"{n_rigid} rigid-body modes: {np.round(f, 2)} -- the constrained "
        "material is not connected to the rest of the part"
    )


def test_frequencies_are_physical(tier1_result):
    f = tier1_result.frequencies
    assert f.size >= 4
    assert np.all(np.diff(f) >= -1e-6), "eigenvalues must be ascending"
    assert 100.0 < f[0] < 20000.0, f"f1={f[0]:.1f} Hz is not a credible frequency"


def test_units_are_consistent(tier1_result):
    """A mm/SI unit mix scales every frequency by ~31x -- catch it explicitly."""
    f1 = tier1_result.frequencies[0]
    assert 200.0 < f1 < 5000.0, (
        f"f1={f1:.1f} Hz suggests a unit error: with mm geometry, E must be in "
        "MPa and rho in tonne/mm^3"
    )


# ---------------------------------------------------------------------------
# Design-space robustness
# ---------------------------------------------------------------------------
# Geometry that is valid only at the default design is useless to an optimiser,
# which spends all its time elsewhere. The first version of these fixes passed
# at the default point and still produced disjoint volumes at 3 of 6 sampled
# designs: a corner fillet scaled with the plate and ate the leg landing zone,
# and a fixed-diameter motor bore collided with the legs on small spans.
#
# This test is cheap (CAD only, no meshing) and is the one that matters most.


@pytest.mark.parametrize("seed", [3, 11])
def test_cad_valid_across_design_space(seed, tmp_path):
    """Every sampled design must yield ONE solid with a flat face at z = 0."""
    from dynopt.doe import sobol_sample

    failures = []
    for i, x in enumerate(sobol_sample(SPACE, 8, seed=seed)):
        params = SPACE.to_dict(x)
        path = tmp_path / f"d{seed}_{i}.step"
        try:
            solid = MOTOR_MOUNT.build(params)
            bb = solid.val().BoundingBox()
            cq.exporters.export(solid, str(path))
            n_vol = count_volumes(str(path))

            if n_vol != 1:
                failures.append(f"design {i}: {n_vol} disjoint volumes")
            elif abs(bb.zmin) > 1e-6:
                failures.append(f"design {i}: zmin={bb.zmin:.4f}, no flat face")
        except Exception as exc:  # a build that throws is also a failure
            failures.append(f"design {i}: {type(exc).__name__}: {exc}")

    assert not failures, "CAD invalid at:\n  " + "\n  ".join(failures)


def test_extreme_designs_keep_bore_clear_of_legs():
    """The motor bore must never intersect a leg, however small the span."""
    from dynopt.geometry.motor_mount import plate_geometry

    for span in (SPACE["span"].lower, 0.05, SPACE["span"].upper):
        for dia in (SPACE["leg_dia"].lower, SPACE["leg_dia"].upper):
            p = SPACE.to_dict(SPACE.defaults)
            p["span"], p["leg_dia"] = span, dia
            g = plate_geometry(p)
            clearance = g["top_span"] - 0.5 * dia - g["hole_radius"]
            assert clearance > 0.0, (
                f"span={span*1e3:.0f}mm dia={dia*1e3:.0f}mm: bore overlaps leg "
                f"by {-clearance*1e3:.1f}mm"
            )
            assert g["half_width"] > g["top_span"] + 0.5 * dia, "plate must overhang legs"


def test_designs_are_meshable_not_merely_valid(tmp_path):
    """A single valid volume is not enough -- it must also mesh.

    Volume count passed at a design whose leg wall (5.18 mm) exceeded its radius
    (4.51 mm). The CAD clamped the bore to 0.1 mm rather than failing, producing
    a sliver annulus that Gmsh rejected with "invalid boundary mesh (overlapping
    facets)". Meshing is the real gate, so this test pays for the extra seconds.
    """
    from dynopt.doe import sobol_sample

    failures = []
    for i, x in enumerate(sobol_sample(SPACE, 4, seed=7)):
        params = SPACE.to_dict(x)
        path = tmp_path / f"m{i}.step"
        try:
            MOTOR_MOUNT.export_step(params, str(path))
            coords, _, _ = mesh_step(str(path), str(tmp_path / f"m{i}.inp"), _cfg())
            if coords.shape[0] < 500:
                failures.append(f"design {i}: only {coords.shape[0]} nodes")
        except Exception as exc:
            failures.append(f"design {i}: {type(exc).__name__}: {exc}")
    assert not failures, "unmeshable designs:\n  " + "\n  ".join(failures)


def test_degenerate_leg_wall_becomes_solid():
    """A wall thicker than the radius must yield a solid leg, not a sliver bore."""
    from dynopt.geometry.motor_mount import MIN_BORE, leg_bore_radius

    assert leg_bore_radius(0.00902, 0.00518) == 0.0, "negative bore must read solid"
    assert leg_bore_radius(0.014, 0.002) == pytest.approx(0.005)
    # Anything the design box permits must be either solid or properly hollow.
    for dia in np.linspace(SPACE["leg_dia"].lower, SPACE["leg_dia"].upper, 7):
        for thk in np.linspace(SPACE["leg_thk"].lower, SPACE["leg_thk"].upper, 7):
            b = leg_bore_radius(dia, thk)
            assert b == 0.0 or b >= MIN_BORE, f"sliver bore {b*1e3:.4f} mm"


def test_beam_and_solid_agree_on_leg_hollowness():
    """Both tiers must make the same solid/hollow call, or they model different parts."""
    from dynopt.geometry import build_frame
    from dynopt.geometry.motor_mount import leg_bore_radius
    from dynopt.doe import sobol_sample

    for x in sobol_sample(SPACE, 12, seed=5):
        p = SPACE.to_dict(x)
        bore = leg_bore_radius(p["leg_dia"], p["leg_thk"])
        leg_area = build_frame(p).sections[0].A
        solid_area = np.pi * (p["leg_dia"] / 2) ** 2
        if bore == 0.0:
            assert leg_area == pytest.approx(solid_area, rel=1e-9)
        else:
            assert leg_area < solid_area



def test_repeated_meshing_does_not_abort_the_process(step_file, tmp_path):
    """Many meshes in one process must not kill the interpreter.

    Gmsh aborts -- ``Fatal Python error: Aborted``, no catchable exception --
    after enough initialize/finalize cycles. It showed up here first, but it is
    a production bug: a Tier 1 DOE meshes hundreds of designs in one process.
    The fix is one session for the process lifetime with the model cleared
    between designs; this test pins that behaviour.
    """
    for i in range(12):
        coords, tags, _ = mesh_step(step_file, str(tmp_path / f"r{i}.inp"), _cfg())
        assert coords.shape[0] > 500
        assert tags.size == coords.shape[0]
