"""Every backend must supply what the layers above it read.

`ModalResult.meta` carries an implicit contract: `nodes`, `payload_nodes` and
`n_dof`, plus a finite `mass`. The reference beam backend supplied all of them
from the start. The CalculiX backend supplied **none** of them, and because
every consumer reads them with a silent fallback, nothing failed:

* `cli.py` damped `payload_nodes[0]`, falling back to node 0 -- which on the
  motor mount is a clamped base corner. `dynopt coupling --backend calculix`
  reported coupling for a dashpot on a constrained node.
* `PartSpec.classification_origin` needs `nodes`; without it the origin is
  `None` and mode classification quietly does nothing on the 3D tier.
* `ResonanceObjective` divides by `mass`, which was `nan`. Every objective in
  a 3D run was `nan`, and an optimiser comparing `nan` to `nan` neither fails
  nor converges.

A conformance test is the only thing that keeps a new backend from repeating
this, so it is parameterised over backends rather than written once per solver.
"""

import shutil

import numpy as np
import pytest

from dynopt.fea.base import CONTRACT_META, ModalResult
from dynopt.fea.reference import ReferenceFrameSolver
from dynopt.geometry import CANTILEVER_ARM, MOTOR_MOUNT

PARTS = [MOTOR_MOUNT, CANTILEVER_ARM]


def _calculix(part):
    from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver

    return CalculiXSolver(
        part, CalculiXConfig(n_modes=8, mesh_size=4.0, mesh_size_min=1.0)
    )


# ---------------------------------------------------------------------------
# The contract checker itself
# ---------------------------------------------------------------------------
def test_missing_contract_names_every_gap():
    """A result with nothing filled in must complain about everything."""
    bare = ModalResult(frequencies=np.array([10.0, 20.0]))
    gaps = bare.missing_contract()
    assert len(gaps) == len(CONTRACT_META) + 1, gaps          # + the nan mass
    joined = " ".join(gaps)
    for key in CONTRACT_META:
        assert key in joined
    assert "mass" in joined


def test_missing_contract_is_quiet_when_satisfied():
    ok = ModalResult(
        frequencies=np.array([10.0]),
        modes=np.zeros((6, 1)),
        mass=0.5,
        meta={"nodes": np.zeros((2, 3)), "payload_nodes": [1], "n_dof": 6},
    )
    assert ok.missing_contract() == []


def test_an_empty_payload_is_an_answer_but_a_missing_key_is_not():
    """A bare run legitimately has no payload; a forgetful backend does not."""
    common = dict(frequencies=np.array([1.0]), mass=1.0,
                  meta={"nodes": np.zeros((1, 3)), "n_dof": 3})
    forgot = ModalResult(**common)
    assert any("payload_nodes" in g for g in forgot.missing_contract())

    bare = ModalResult(frequencies=np.array([1.0]), mass=1.0,
                       meta={"nodes": np.zeros((1, 3)), "n_dof": 3,
                             "payload_nodes": []})
    assert bare.missing_contract() == []


def test_mode_rows_must_map_onto_nodes():
    """Mode shapes with a row count that is not a multiple of the node count."""
    bad = ModalResult(
        frequencies=np.array([1.0]),
        modes=np.zeros((7, 1)),
        mass=1.0,
        meta={"nodes": np.zeros((2, 3)), "payload_nodes": [0], "n_dof": 7},
    )
    assert any("cannot be mapped back onto the geometry" in g
               for g in bad.missing_contract())


# ---------------------------------------------------------------------------
# Every real backend, against the real contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.name)
def test_reference_backend_conforms(part):
    res = ReferenceFrameSolver(part).solve(
        part.design_space.to_dict(part.design_space.defaults), n_modes=6
    )
    assert res.missing_contract() == []


@pytest.mark.slow
@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.name)
def test_calculix_backend_conforms(part):
    pytest.importorskip("gmsh", reason="Gmsh not installed")
    pytest.importorskip("cadquery", reason="CadQuery not installed")
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX (ccx) not on PATH")

    d = part.design_space.to_dict(part.design_space.defaults)
    res = _calculix(part).solve(d, n_modes=6)
    assert res.missing_contract() == []

    # The payload must land on nodes that exist and move.
    nodes = np.asarray(res.meta["nodes"])
    payload = res.meta["payload_nodes"]
    assert len(payload) > 0
    assert max(payload) < len(nodes)
    assert res.meta["n_dof"] == 3 * len(nodes)


@pytest.mark.slow
def test_solved_mass_matches_the_cad_volume():
    """The mass a backend reports must be the mass it actually solved.

    Measured from the mesh, not the CAD, so a disagreement means the mesh is
    not the part -- which is worth knowing. Straight-sided tets under-fill a
    curved solid slightly, hence the one-sided tolerance.
    """
    pytest.importorskip("gmsh", reason="Gmsh not installed")
    cq = pytest.importorskip("cadquery", reason="CadQuery not installed")
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX (ccx) not on PATH")

    part = MOTOR_MOUNT
    d = part.design_space.to_dict(part.design_space.defaults)
    res = _calculix(part).solve(d, n_modes=6)

    cad_mm3 = part.build_cad(d).val().Volume()
    cad_kg = cad_mm3 * 1e-9 * 2700.0        # AL6061, the part's own material
    assert res.mass == pytest.approx(cad_kg, rel=0.05), (
        f"solved {res.mass*1000:.1f} g against a CAD volume of {cad_kg*1000:.1f} g"
    )
    assert res.mass < cad_kg * 1.001, "straight-sided tets cannot exceed the CAD volume"


@pytest.mark.slow
def test_a_bare_run_reports_no_payload_rather_than_a_wrong_one():
    pytest.importorskip("gmsh", reason="Gmsh not installed")
    pytest.importorskip("cadquery", reason="CadQuery not installed")
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX (ccx) not on PATH")

    from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver

    part = MOTOR_MOUNT
    d = part.design_space.to_dict(part.design_space.defaults)
    cfg = CalculiXConfig(n_modes=6, mesh_size=4.0, mesh_size_min=1.0,
                         payload_mass=0.0)
    res = CalculiXSolver(part, cfg).solve(d, n_modes=6)
    assert res.meta["payload_nodes"] == []
    assert res.missing_contract() == []


def test_the_objective_refuses_a_nan_mass():
    """It used to return nan, which an optimiser cannot distinguish from a value."""
    from dynopt.optimize import ResonanceObjective, rpm_bands

    obj = ResonanceObjective(bands=rpm_bands(1000, 6000), mass_ref=0.5)
    assert np.isfinite(obj(np.array([800.0, 900.0]), mass=0.4))
    with pytest.raises(ValueError, match="nan"):
        obj(np.array([800.0, 900.0]), mass=float("nan"))
