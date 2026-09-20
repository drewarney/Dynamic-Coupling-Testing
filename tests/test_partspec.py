"""The PartSpec contract, exercised by a part the core has never seen.

This file is the evidence that the plugin boundary is real. It defines a
completely different structure -- a cantilever tube with a tip mass, sharing no
code with the benchmark mount -- and drives it through the whole pipeline:
solver, mode classification, DOE, surrogate targets. If any of that required
editing the core, this test could not be written.

The first bending frequency is checked against the closed-form Rayleigh
estimate, so the part is verified rather than merely runnable.
"""

import numpy as np
import pytest

from dynopt.doe import sobol_sample
from dynopt.dynamics.modal import ModeFamilySpec
from dynopt.fea.reference import MATERIALS, FrameModel, ReferenceFrameSolver, Section
from dynopt.geometry.base import DesignSpace, DesignVariable, PartSpec
from dynopt.optimize.driver import _sample_dataset

TIP_MASS = 0.25  # kg


class TipMassBeam(PartSpec):
    """A cantilever tube carrying a mass at its free end.

    Deliberately minimal: it implements only ``design_space``, ``build_frame``
    and ``mode_families``. No CAD tier, no solver hints, no origin override --
    the defaults cover the rest.
    """

    name = "tip_mass_beam"

    #: The beam runs along +x, so about the tip: bending couples the transverse
    #: translations with the in-plane rotations, axial is ux, torsion is rx.
    FAMILIES = ModeFamilySpec(
        dofs={"bending": (1, 2, 4, 5), "axial": (0,), "torsion": (3,)},
        branches={"bending": 2},
    )

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace([
            DesignVariable("length", 0.10, 0.40, 0.25, "m", "cantilever length"),
            DesignVariable("dia", 0.008, 0.030, 0.016, "m", "tube outer diameter"),
            DesignVariable("thk", 0.0005, 0.004, 0.0015, "m", "wall thickness"),
        ])

    @property
    def mode_families(self) -> ModeFamilySpec:
        return self.FAMILIES

    def build_frame(self, params, n_elem: int = 12) -> FrameModel:
        L = float(params["length"])
        sec = Section.tube(float(params["dia"]), float(params["thk"]))
        nodes = np.zeros((n_elem + 1, 3))
        nodes[:, 0] = np.linspace(0.0, L, n_elem + 1)
        return FrameModel(
            nodes=nodes,
            elements=[(i, i + 1) for i in range(n_elem)],
            sections=[sec] * n_elem,
            material=MATERIALS["AL6061"],
            fixed_nodes=[0],
            point_masses={n_elem: TIP_MASS},
        )


PART = TipMassBeam()


def test_minimal_part_satisfies_the_contract():
    """A part may implement one tier and inherit everything else."""
    assert PART.has_frame
    assert not PART.has_cad
    assert len(PART.design_space) == 3
    with pytest.raises(NotImplementedError):
        PART.build_cad(PART.design_space.to_dict(PART.design_space.defaults))


def test_new_part_runs_through_the_solver_untouched():
    """No core change was needed to solve a structure the core has never seen."""
    solver = ReferenceFrameSolver(PART)
    space = PART.design_space
    res = solver.solve(space.to_dict(space.defaults), n_modes=10)
    assert res.meta["part"] is PART
    assert np.all(np.diff(res.frequencies) >= -1e-6)
    assert res.frequencies[0] > 1.0


def test_first_bending_matches_rayleigh_estimate():
    r"""Verified, not just runnable.

    Cantilever with a tip mass, Rayleigh approximation including the beam's own
    inertia:

    .. math:: f_1 \approx \frac{1}{2\pi}\sqrt{\frac{3EI}{L^3 (M + 0.24 m)}}
    """
    space = PART.design_space
    params = space.to_dict(space.defaults)
    model = PART.build_frame(params)
    res = ReferenceFrameSolver(PART).solve(params, n_modes=6)

    sec = model.sections[0]
    mat = model.material
    L = float(params["length"])
    beam_mass = mat.rho * sec.A * L
    f_est = np.sqrt(
        3 * mat.E * sec.Iy / (L**3 * (TIP_MASS + 0.24 * beam_mass))
    ) / (2 * np.pi)

    assert abs(res.frequencies[0] - f_est) / f_est < 0.05, (
        f"FE {res.frequencies[0]:.2f} Hz vs Rayleigh {f_est:.2f} Hz"
    )


def test_classification_uses_the_parts_own_families():
    """Families are the part's vocabulary, not a fixed list in the core."""
    from dynopt.dynamics import classify_modes, family_frequencies

    space = PART.design_space
    solver = ReferenceFrameSolver(PART)
    res = solver.solve(space.to_dict(space.defaults), n_modes=12)
    origin = PART.classification_origin(res)
    assert origin is not None

    labels = classify_modes(res, origin, PART.mode_families, min_purity=0.3)
    assert "bending" in labels, labels

    fams = family_frequencies(res, origin, PART.mode_families)
    assert set(fams) == set(PART.mode_families.columns)
    assert np.isfinite(fams["bending_1"])
    if np.isfinite(fams["bending_2"]):
        assert fams["bending_2"] >= fams["bending_1"]


def test_surrogate_targets_build_for_an_unseen_part():
    """The DOE path resolves families from the part with no argument passed."""
    solver = ReferenceFrameSolver(PART)
    space = PART.design_space
    X = sobol_sample(space, 8, seed=2)
    F, mass, results, cols = _sample_dataset(solver, space, X, 12, targets="family")

    assert cols == list(PART.mode_families.columns)
    assert np.isfinite(mass).all()
    assert np.isfinite(F[:, cols.index("bending_1")]).all()


def test_two_parts_coexist_in_one_process():
    """Two parts with different families must not contaminate each other."""
    from dynopt.geometry import MOTOR_MOUNT

    beam = ReferenceFrameSolver(PART)
    mount = ReferenceFrameSolver(MOTOR_MOUNT)

    rb = beam.solve(PART.design_space.to_dict(PART.design_space.defaults), n_modes=8)
    rm = mount.solve(
        MOTOR_MOUNT.design_space.to_dict(MOTOR_MOUNT.design_space.defaults), n_modes=8
    )
    assert rb.meta["part"].mode_families.columns != rm.meta["part"].mode_families.columns
    assert "bending_1" in rb.meta["part"].mode_families.columns
    assert "bounce_z" in rm.meta["part"].mode_families.columns
