"""Verification against closed-form solutions.

These are the tests that make the whole pipeline defensible.  An optimiser
built on an unverified solver is a random number generator with extra steps --
and "I validated the solver against analytical solutions to <0.1%" is a far
stronger portfolio claim than "I used Ansys".
"""

import numpy as np
import pytest

from dynopt.fea.reference import MATERIALS, FrameModel, Section, modal_solve

#: Roots of cos(b)cosh(b) + 1 = 0 -- the cantilever characteristic equation.
BETA_CANTILEVER = [1.87510407, 4.69409113, 7.85475744, 10.99554073]
#: Roots for a fixed-fixed beam: cos(b)cosh(b) - 1 = 0.
BETA_FIXED_FIXED = [4.73004074, 7.85320462, 10.99560784]


def _straight_beam(L, sec, mat, n_elem=20, fixed_nodes=(0,)):
    nodes = np.zeros((n_elem + 1, 3))
    nodes[:, 0] = np.linspace(0.0, L, n_elem + 1)
    return FrameModel(
        nodes=nodes,
        elements=[(i, i + 1) for i in range(n_elem)],
        sections=[sec] * n_elem,
        material=mat,
        fixed_nodes=list(fixed_nodes),
    )


def _unique(freqs, rtol=1e-3):
    """Collapse degenerate pairs (a circular section bends identically in y, z)."""
    out = []
    for f in freqs:
        if not out or abs(f - out[-1]) / max(f, 1e-12) > rtol:
            out.append(f)
    return np.array(out)


def test_cantilever_bending_matches_euler_bernoulli():
    r"""f_n = (beta_n^2 / 2*pi*L^2) * sqrt(EI / rho*A), to better than 0.1%."""
    L, d = 1.0, 0.02
    mat, sec = MATERIALS["STEEL"], Section.circle(d)
    res = modal_solve(_straight_beam(L, sec, mat), n_modes=10, shear=False)

    analytic = [
        (b**2 / (2 * np.pi * L**2)) * np.sqrt(mat.E * sec.Iy / (mat.rho * sec.A))
        for b in BETA_CANTILEVER[:3]
    ]
    got = _unique(res.frequencies)[:3]
    for fe, fa in zip(got, analytic):
        assert abs(fe - fa) / fa < 1e-3, f"FE {fe:.4f} vs analytic {fa:.4f}"


def test_fixed_fixed_bending_matches_theory():
    """Different boundary conditions must also be right, not just cantilever."""
    L, d = 0.8, 0.015
    mat, sec = MATERIALS["AL6061"], Section.circle(d)
    model = _straight_beam(L, sec, mat, n_elem=24, fixed_nodes=(0, 24))
    res = modal_solve(model, n_modes=8, shear=False)

    analytic = [
        (b**2 / (2 * np.pi * L**2)) * np.sqrt(mat.E * sec.Iy / (mat.rho * sec.A))
        for b in BETA_FIXED_FIXED[:2]
    ]
    got = _unique(res.frequencies)[:2]
    for fe, fa in zip(got, analytic):
        assert abs(fe - fa) / fa < 2e-3


def test_axial_and_torsional_modes():
    """Rod axial f1 = c/4L with c = sqrt(E/rho); torsional uses sqrt(G/rho)."""
    L, d = 1.0, 0.02
    mat, sec = MATERIALS["STEEL"], Section.circle(d)
    res = modal_solve(_straight_beam(L, sec, mat, n_elem=30), n_modes=30, shear=False)

    f_axial = np.sqrt(mat.E / mat.rho) / (4 * L)
    f_torsion = np.sqrt(mat.G / mat.rho) / (4 * L)
    for target in (f_axial, f_torsion):
        assert np.min(np.abs(res.frequencies - target)) / target < 5e-3, (
            f"no mode near {target:.1f} Hz in {np.round(res.frequencies, 1)}"
        )


def test_mass_matrix_conserves_total_mass():
    r"""Sum of translational mass terms must equal rho*A*L exactly."""
    L = 1.0
    sec, mat = Section.rectangle(0.02, 0.01), MATERIALS["AL6061"]
    model = _straight_beam(L, sec, mat, n_elem=10, fixed_nodes=())
    from dynopt.fea.reference import assemble

    _, M = assemble(model)
    Md = np.asarray(M.todense())
    # A rigid translation r has generalised mass r^T M r = total mass.
    r = np.zeros(model.n_dof)
    r[0::6] = 1.0
    assert abs(float(r @ Md @ r) - mat.rho * sec.A * L) / (mat.rho * sec.A * L) < 1e-10


def test_stiffness_scaling_law():
    """Cantilever frequency must scale as 1/L^2 -- catches unit/algebra errors."""
    sec, mat = Section.circle(0.01), MATERIALS["STEEL"]
    f = [
        modal_solve(_straight_beam(L, sec, mat), n_modes=2, shear=False).frequencies[0]
        for L in (1.0, 2.0)
    ]
    assert abs(f[0] / f[1] - 4.0) / 4.0 < 1e-3


def test_shear_deformation_lowers_frequency():
    """Timoshenko shear flexibility must soften a stubby beam, never stiffen it."""
    L, sec, mat = 0.1, Section.circle(0.03), MATERIALS["STEEL"]  # L/d ~ 3.3
    eb = modal_solve(_straight_beam(L, sec, mat), n_modes=2, shear=False).frequencies[0]
    ti = modal_solve(_straight_beam(L, sec, mat), n_modes=2, shear=True).frequencies[0]
    assert ti < eb
    assert (eb - ti) / eb > 0.02, "stubby beam should show a few % shear softening"


def test_mesh_convergence():
    """Refining the discretisation must converge monotonically from above."""
    L, sec, mat = 1.0, Section.circle(0.02), MATERIALS["STEEL"]
    f = [
        modal_solve(_straight_beam(L, sec, mat, n_elem=n), n_modes=2, shear=False).frequencies[0]
        for n in (2, 4, 8, 16)
    ]
    assert all(f[i] >= f[i + 1] - 1e-9 for i in range(len(f) - 1)), f
    assert abs(f[-1] - f[-2]) / f[-1] < 1e-4


@pytest.mark.parametrize("name", ["AL6061", "STEEL", "TI64"])
def test_specific_stiffness_ordering(name):
    r"""Frequency tracks sqrt(E/rho), so material swaps must move it predictably."""
    L, sec = 1.0, Section.circle(0.02)
    mat = MATERIALS[name]
    f = modal_solve(_straight_beam(L, sec, mat), n_modes=2, shear=False).frequencies[0]
    expected = (
        (BETA_CANTILEVER[0] ** 2 / (2 * np.pi * L**2))
        * np.sqrt(mat.E * sec.Iy / (mat.rho * sec.A))
    )
    assert abs(f - expected) / expected < 1e-3
