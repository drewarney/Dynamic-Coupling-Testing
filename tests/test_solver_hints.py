"""Per-design solver settings, and recovery from a failed mesh.

Three defects sit behind this file, all of the kind the project keeps running
into: the toolchain kept working and the numbers stayed plausible.

* **Stale hints.** ``_apply_part_hints`` only wrote a field that still held its
  dataclass default. The first solve filled those fields in, so from the second
  design onward every design-dependent hint -- the payload smear radius, the
  mesh cap -- silently kept the *first* design's value. A DOE driven by one
  solver instance smeared a 28 mm motor over a 6 mm boss and never said so.
* **No feature cap.** A fixed element size is not portable across a design
  space. One sampled arm (11 mm tube, 5.5 mm bore) cannot be wrapped by 6 mm
  tets at all; Gmsh rejected it and the design vanished from the DOE.
* **A poisoned session.** After one ``mesh.generate`` threw, the shared Gmsh
  session kept accepting calls and returned *empty* meshes forever after. In a
  24-design run, design 4 failed legitimately and designs 5-23 plus an entire
  second part were lost to it -- 4 of 48 designs survived.

Only the last needs Gmsh installed.
"""

import numpy as np
import pytest

from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver
from dynopt.geometry import CANTILEVER_ARM, MOTOR_MOUNT


def _solver(part, **cfg_kw):
    return CalculiXSolver(part, CalculiXConfig(**cfg_kw))


def _at(part, frac):
    """Design at a fraction of the way across every bound."""
    sp = part.design_space
    return sp.to_dict(sp.denormalize(np.full(len(sp.variables), float(frac))))


@pytest.mark.parametrize("part", [MOTOR_MOUNT, CANTILEVER_ARM])
def test_hints_track_the_design_not_the_first_call(part):
    """Two different designs must produce two different configs."""
    lo, hi = _at(part, 0.0), _at(part, 1.0)

    s = _solver(part)
    s._apply_part_hints(lo)
    first = (s.cfg.payload_radius, s.cfg.min_feature)
    s._apply_part_hints(hi)
    second = (s.cfg.payload_radius, s.cfg.min_feature)

    fresh = _solver(part)
    fresh._apply_part_hints(hi)
    assert second == (fresh.cfg.payload_radius, fresh.cfg.min_feature), (
        "a reused solver must give the same config as a fresh one"
    )
    assert first != second, "these two designs differ; their hints must too"


def test_caller_pinned_fields_survive_every_design():
    part = MOTOR_MOUNT
    s = _solver(part, payload_radius=12.345, fix_tol=0.5)
    for frac in (0.0, 0.5, 1.0):
        s._apply_part_hints(_at(part, frac))
        assert s.cfg.payload_radius == 12.345
        assert s.cfg.fix_tol == 0.5


@pytest.mark.parametrize("part", [MOTOR_MOUNT, CANTILEVER_ARM])
def test_min_feature_is_positive_everywhere_in_the_box(part):
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert part.solver_hints(_at(part, frac))["min_feature"] > 0.0


def test_min_feature_lowers_the_floor_not_the_target():
    """The cap is on ``mesh_size_min``, so curvature refinement has room."""
    part = CANTILEVER_ARM
    sp = part.design_space
    # Long, slim, thinnest wall: the design that loses at a fixed element size.
    thin = sp.to_dict(sp.denormalize(np.array([1.0, 0.0, 0.0])))

    coarse = _solver(part, mesh_size=6.0, mesh_size_min=1.5)
    coarse._apply_part_hints(thin)
    h, hmin = coarse._mesh_sizes()
    assert h == 6.0, "the target element size is the caller's to set"
    assert hmin <= coarse.cfg.min_feature / 3.0, (
        "the floor must sit below the smallest feature or refinement is clamped"
    )

    fine = _solver(part, mesh_size=0.2, mesh_size_min=0.05)
    fine._apply_part_hints(thin)
    assert fine._mesh_sizes() == (0.2, 0.05), "a floor must never coarsen a fine mesh"


def test_unknown_min_feature_leaves_the_mesh_alone():
    s = CalculiXSolver(object(), CalculiXConfig(mesh_size=4.0, mesh_size_min=1.0))
    assert s.cfg.min_feature == 0.0
    assert s._mesh_sizes() == (4.0, 1.0)


def test_one_failed_mesh_does_not_poison_the_next():
    """The cascade that cost 44 of 48 designs in a DOE."""
    gmsh = pytest.importorskip("gmsh", reason="Gmsh not installed")
    from dynopt.fea.calculix import gmsh_model

    def cube_nodes():
        with gmsh_model("cube") as g:
            g.model.occ.addBox(0, 0, 0, 10, 10, 10)
            g.model.occ.synchronize()
            g.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
            g.model.mesh.generate(3)
            return len(g.model.mesh.getNodes()[0])

    before = cube_nodes()
    assert before > 0

    with pytest.raises(RuntimeError, match="deliberate"):
        with gmsh_model("doomed") as g:
            g.model.occ.addBox(0, 0, 0, 1, 1, 1)
            g.model.occ.synchronize()
            raise RuntimeError("deliberate meshing failure")

    assert cube_nodes() == before, "the session did not recover from a failure"


@pytest.mark.parametrize("part", [MOTOR_MOUNT, CANTILEVER_ARM])
def test_both_tiers_model_the_same_material(part):
    """A carbon arm solved as aluminium reads 31% low and nothing objects.

    The beam tier takes its material from the part. Until the 3D tier did too,
    a DOE script had to repeat the properties by hand -- and got them from
    whichever part it was written for. The cross-tier discrepancy that a
    multi-fidelity correction then "learns" is partly a material mismatch.
    """
    from dynopt.fea.reference import MATERIALS

    beam_mat = MATERIALS[part.material]
    s = _solver(part)
    s._apply_part_hints(_at(part, 0.5))
    # mm / tonne / s against SI
    assert s.cfg.E == pytest.approx(beam_mat.E / 1e6)
    assert s.cfg.rho == pytest.approx(beam_mat.rho * 1e-12)
    assert s.cfg.nu == pytest.approx(beam_mat.nu)


@pytest.mark.slow
@pytest.mark.parametrize("part", [MOTOR_MOUNT, CANTILEVER_ARM])
def test_every_design_in_the_box_meshes(part):
    """The shipped defaults must not lose designs from a DOE.

    Owns the claim in ``CalculiXConfig.mesh_from_curvature``. Measured on the
    same 24 Sobol designs per part, at the coarse tier's 6 mm target: a fixed
    element size meshed 39 of 48 (18/24 mount, 21/24 arm) and every failure was
    an element larger than a bore it had to wrap. Curvature-driven refinement
    meshes 48 of 48.

    Meshing only -- no CalculiX -- so this costs about a minute, not an hour.
    """
    pytest.importorskip("gmsh", reason="Gmsh not installed")
    pytest.importorskip("cadquery", reason="CadQuery not installed")
    import os
    import shutil as _shutil
    import tempfile

    from dynopt.doe import sobol_sample
    from dynopt.fea.calculix import mesh_step

    sp = part.design_space
    failures = []
    for i, x in enumerate(sobol_sample(sp, 24, seed=21)):
        d = sp.to_dict(x)
        s = _solver(part, mesh_size=6.0, mesh_size_min=1.5)
        s._apply_part_hints(d)
        h, hmin = s._mesh_sizes()
        cfg = CalculiXConfig(
            mesh_size=h, mesh_size_min=hmin,
            mesh_from_curvature=s.cfg.mesh_from_curvature,
        )
        tmp = tempfile.mkdtemp()
        try:
            part.export_step(d, os.path.join(tmp, "p.step"))
            coords, _ = mesh_step(
                os.path.join(tmp, "p.step"), os.path.join(tmp, "m.inp"), cfg
            )
            assert coords.shape[0] > 0
        except Exception as exc:
            failures.append((i, str(exc)[:80]))
        finally:
            _shutil.rmtree(tmp, ignore_errors=True)

    assert not failures, f"{len(failures)}/24 designs lost: {failures}"
