"""Tests for modal post-processing, coupling metrics and harmonic response."""

import numpy as np
import pytest

from dynopt.dynamics import (
    canonicalize_degenerate_basis,
    classify_modes,
    cluster_modes,
    coupling_index,
    direct_frf,
    discrete_damper_matrix,
    effective_mass_fraction,
    energy_transfer_index,
    family_frequencies,
    mac,
    modal_damping_matrix,
    modal_frf,
    rayleigh_damping,
    subspace_mac,
)
from dynopt.fea.reference import MATERIALS, FrameModel, Section, modal_solve
from dynopt.geometry import (  # noqa: E402
    MODE_FAMILIES,
    MOTOR_MOUNT,
    MOTOR_MOUNT_FAMILIES,
    MOTOR_MOUNT_SPACE,
    build_frame,
    motor_node_index,
)


@pytest.fixture(scope="module")
def mount():
    params = MOTOR_MOUNT_SPACE.to_dict(MOTOR_MOUNT_SPACE.defaults)
    model = build_frame(params)
    res = modal_solve(model, n_modes=12)
    return model, res, model.nodes[motor_node_index()]


def test_effective_mass_sums_toward_total(mount):
    """Effective modal masses must sum to no more than the total mass."""
    _, res, _ = mount
    frac = effective_mass_fraction(res, 2)
    assert (frac >= -1e-9).all()
    assert frac.sum() <= 1.0 + 1e-6
    assert frac.max() > 0.3, "some mode must carry real vertical mass"


def test_mac_identity_and_orthogonality(mount):
    """MAC of a mode set against itself is the identity for distinct modes."""
    _, res, _ = mount
    m = mac(res.modes[:, :4], res.modes[:, :4])
    assert np.allclose(np.diag(m), 1.0, atol=1e-8)


def test_subspace_mac_is_rotation_invariant(mount):
    """Rotating within a subspace must not change its subspace MAC."""
    _, res, _ = mount
    A = res.modes[:, 1:3]
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert subspace_mac(A, A) == pytest.approx(1.0, abs=1e-8)
    assert subspace_mac(A, A @ R) == pytest.approx(1.0, abs=1e-8)


def test_degenerate_modes_are_clustered(mount):
    """The 4-fold symmetric mount must produce repeated eigenvalues."""
    _, res, _ = mount
    clusters = cluster_modes(res.frequencies, rtol=0.02)
    assert any(len(c) > 1 for c in clusters), "expected degenerate pairs"


def test_mode_classification_covers_all_families(mount):
    """Every physical family must be identified on the benchmark part."""
    _, res, origin = mount
    fams = family_frequencies(res, origin, MOTOR_MOUNT_FAMILIES)
    for key in MODE_FAMILIES:
        assert np.isfinite(fams[key]), f"{key} not found in {fams}"
    assert fams["sway_rock_1"] <= fams["sway_rock_2"]


def test_classification_is_stable_across_designs():
    """Family frequencies must stay finite and ordered across the design space."""
    from dynopt.doe import sobol_sample

    for x in sobol_sample(MOTOR_MOUNT_SPACE, 16, seed=3):
        model = build_frame(MOTOR_MOUNT_SPACE.to_dict(x))
        res = modal_solve(model, n_modes=12)
        fams = family_frequencies(
            res, model.nodes[motor_node_index()], MOTOR_MOUNT_FAMILIES
        )
        assert np.isfinite(fams["bounce_z"])
        assert np.isfinite(fams["sway_rock_1"])


def test_rayleigh_damping_is_proportional(mount):
    """Rayleigh damping must diagonalise in the modal basis -- zero coupling."""
    _, res, _ = mount
    C, alpha, beta = rayleigh_damping(res.M, res.K, 0.02, 200.0, 0.02, 2000.0)
    Ctil = modal_damping_matrix(res, C)
    off = coupling_index(Ctil)
    assert off.max() < 1e-6, f"proportional damping leaked coupling: {off.max():.2e}"
    assert alpha > 0 and beta > 0


def test_discrete_damper_creates_coupling(mount):
    """A single grounded dashpot is non-proportional and must couple modes."""
    model, res, _ = mount
    dof = motor_node_index() * 6 + 0          # x-translation at the motor node
    C = discrete_damper_matrix(model.n_dof, [(dof, 50.0)])
    off = coupling_index(modal_damping_matrix(res, C))
    assert off.max() > 1e-3, "local damper should produce off-diagonal terms"
    eta = energy_transfer_index(res, C)
    assert eta.shape == (res.n_modes, res.n_modes)
    assert np.allclose(np.diag(eta), 0.0)
    assert (eta >= -1e-12).all() and (eta <= 1.0 + 1e-9).all()


def test_modal_frf_matches_direct_frf(mount):
    """Modal superposition must reproduce the direct solution near resonance.

    This is the correctness check for the fast path: if the retained modal basis
    is adequate, the two curves agree.  Disagreement means modal truncation.
    """
    model, res, _ = mount
    free = res.meta["free_dofs"]
    C, _, _ = rayleigh_damping(res.M, res.K, 0.02, 200.0, 0.02, 3000.0)

    F = np.zeros(model.n_dof)
    F[motor_node_index() * 6 + 2] = 1.0

    freqs = np.linspace(50.0, 400.0, 40)
    Kr = res.K.tocsr()[free][:, free]
    Mr = res.M.tocsr()[free][:, free]
    Cr = C.tocsr()[free][:, free]

    out_dof = motor_node_index() * 6 + 2
    j = int(np.where(free == out_dof)[0][0])

    H_dir = direct_frf(Kr, Mr, Cr, freqs, F[free])[j]
    H_mod = modal_frf(res, C, freqs, F, dofs_out=[out_dof])[0]

    rel = np.abs(np.abs(H_mod) - np.abs(H_dir)) / np.abs(H_dir)
    assert np.median(rel) < 0.05, f"median FRF error {np.median(rel):.3f}"


def test_frf_peaks_at_natural_frequencies(mount):
    """The FRF magnitude must peak at a resonance carrying vertical mass."""
    model, res, _ = mount
    C, _, _ = rayleigh_damping(res.M, res.K, 0.01, 200.0, 0.01, 3000.0)
    F = np.zeros(model.n_dof)
    F[motor_node_index() * 6 + 2] = 1.0

    f1 = res.frequencies[int(np.argmax(effective_mass_fraction(res, 2)))]
    freqs = np.linspace(0.5 * f1, 1.5 * f1, 201)
    H = np.abs(modal_frf(res, C, freqs, F, dofs_out=[motor_node_index() * 6 + 2])[0])
    assert abs(freqs[int(np.argmax(H))] - f1) / f1 < 0.02


def test_coupling_changes_frf():
    """Dropping cross-modal terms must change the answer when coupling is real.

    If this test ever passes trivially, the coupling analysis is not earning its
    keep -- so it doubles as a check that the benchmark case is interesting.
    """
    params = MOTOR_MOUNT_SPACE.to_dict(MOTOR_MOUNT_SPACE.defaults)
    model = build_frame(params)
    res = modal_solve(model, n_modes=10)
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 200.0), (node * 6 + 4, 5.0)])

    F = np.zeros(model.n_dof)
    F[node * 6 + 0] = 1.0
    freqs = np.linspace(100.0, 1200.0, 120)
    Hc = modal_frf(res, C, freqs, F, dofs_out=[node * 6 + 0], keep_coupling=True)[0]
    Hu = modal_frf(res, C, freqs, F, dofs_out=[node * 6 + 0], keep_coupling=False)[0]
    assert np.max(np.abs(np.abs(Hc) - np.abs(Hu)) / np.abs(Hc)) > 1e-3


# ---------------------------------------------------------------------------
# Basis-invariance regression tests
# ---------------------------------------------------------------------------
# These exist because the original coupling analysis silently produced
# different answers on macOS (Accelerate) and Linux (OpenBLAS) for the same
# model -- a reported "-17.8% error from ignoring coupling" on one machine and
# "-2.3%" on the other. Both were valid outputs of the same deterministic code,
# because eigenvectors of a repeated eigenvalue are defined only up to a
# rotation within their eigenspace and the two LAPACK builds chose differently.
#
# Any physical quantity must be invariant to that rotation. These tests assert
# it directly, by re-running with a deliberately rotated (but equally valid)
# eigenbasis.


def _rotate_degenerate(res, pairs, theta):
    """Rotate within degenerate subspaces -- still a valid eigenbasis."""
    import copy

    phi = res.modes.copy()
    c, s = np.cos(theta), np.sin(theta)
    for i, j in pairs:
        a, b = phi[:, i].copy(), phi[:, j].copy()
        phi[:, i], phi[:, j] = c * a - s * b, s * a + c * b
    out = copy.copy(res)
    out.modes = phi
    return out


def _degenerate_pairs(res, rtol=0.02):
    return [
        (c[0], c[1]) for c in cluster_modes(res.frequencies, rtol=rtol) if len(c) > 1
    ]


def test_coupled_frf_is_basis_invariant(mount):
    """The full coupled FRF must not depend on the degenerate basis at all.

    This one held even before the fix -- solving the full modal system is
    basis-independent by construction. It is the control for the test below.
    """
    model, res, _ = mount
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)])
    F = np.zeros(model.n_dof)
    F[node * 6 + 0] = 1.0
    freqs = np.linspace(100.0, 1500.0, 200)

    pairs = _degenerate_pairs(res)
    assert pairs, "benchmark part must have degenerate pairs for this test"

    ref = None
    for theta in (0.0, 0.4, 1.1):
        H = modal_frf(
            _rotate_degenerate(res, pairs, theta), C, freqs, F,
            dofs_out=[node * 6 + 0], keep_coupling=True,
        )[0]
        val = float(np.sqrt(np.trapezoid(np.abs(H) ** 2, freqs)))
        if ref is None:
            ref = val
        assert abs(val - ref) / ref < 1e-9


def test_uncoupled_frf_is_basis_invariant(mount):
    """The uncoupled approximation must be basis-invariant too.

    Before canonicalisation this varied by ~26 percentage points across valid
    bases of the same eigensolution, which is what made the headline coupling
    number irreproducible between machines.
    """
    model, res, _ = mount
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)])
    F = np.zeros(model.n_dof)
    F[node * 6 + 0] = 1.0
    freqs = np.linspace(100.0, 1500.0, 200)

    pairs = _degenerate_pairs(res)
    errors = []
    for theta in (0.0, 0.2, 0.5, np.pi / 4, 1.0, 1.3, 2.1):
        r = _rotate_degenerate(res, pairs, theta)
        out = [node * 6 + 0]
        Hc = modal_frf(r, C, freqs, F, dofs_out=out, keep_coupling=True)[0]
        Hu = modal_frf(r, C, freqs, F, dofs_out=out, keep_coupling=False)[0]
        rc = float(np.sqrt(np.trapezoid(np.abs(Hc) ** 2, freqs)))
        ru = float(np.sqrt(np.trapezoid(np.abs(Hu) ** 2, freqs)))
        errors.append(100.0 * (ru - rc) / rc)

    spread = max(errors) - min(errors)
    assert spread < 1e-6, (
        f"uncoupled FRF varies by {spread:.2f} percentage points across "
        f"equivalent eigenbases: {np.round(errors, 2)}"
    )


def test_degenerate_pairs_report_no_spurious_coupling(mount):
    """A degenerate pair must not be reported as maximally coupled.

    Pre-fix this returned eta = 1.000 for every degenerate pair, which looked
    like a dramatic physical finding and was entirely an artifact.
    """
    model, res, _ = mount
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)])

    kap = coupling_index(modal_damping_matrix(res, C))
    for i, j in _degenerate_pairs(res):
        assert kap[i, j] < 0.05, (
            f"degenerate modes {i+1},{j+1} report kappa={kap[i, j]:.3f}; "
            "within a repeated eigenvalue this is basis-dependent, not physical"
        )
    # Genuine coupling between distinct frequencies must survive the fix.
    assert kap.max() > 0.1, "canonicalisation must not erase real coupling"


def test_coupling_index_is_basis_invariant(mount):
    """kappa between distinct modes must not depend on the degenerate basis."""
    model, res, _ = mount
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)])

    pairs = _degenerate_pairs(res)
    ref = None
    for theta in (0.0, 0.6, 1.4):
        kap = coupling_index(
            modal_damping_matrix(_rotate_degenerate(res, pairs, theta), C)
        )
        val = np.sort(kap, axis=None)
        if ref is None:
            ref = val
        assert np.allclose(val, ref, atol=1e-8)


def test_published_coupling_figure_is_current(mount):
    """Pin the headline coupling number to what the code actually produces.

    This figure went stale once already: it was measured, published, and then
    invalidated two commits later by a plate-mass recalibration that moved a
    mode from 555.7 to 600.1 Hz. Nothing failed, because no test owned the
    number. This test owns it -- if the physics moves, it fails here rather
    than quietly contradicting the README.

    Update the bound and the docs together, or not at all.
    """
    model, res, _ = mount
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, 180.0), (node * 6 + 4, 6.0)])
    F = np.zeros(model.n_dof)
    F[node * 6 + 0] = 1.0
    freqs = np.linspace(100.0, 1500.0, 600)
    out = [node * 6 + 0]

    Hc = modal_frf(res, C, freqs, F, dofs_out=out, keep_coupling=True)[0]
    Hu = modal_frf(res, C, freqs, F, dofs_out=out, keep_coupling=False)[0]
    rms_c = float(np.sqrt(np.trapezoid(np.abs(Hc) ** 2, freqs)))
    rms_u = float(np.sqrt(np.trapezoid(np.abs(Hu) ** 2, freqs)))
    err = 100.0 * (rms_u - rms_c) / rms_c

    assert -24.0 < err < -21.0, (
        f"band RMS coupling error is {err:.1f}%, but the docs publish -22.6%. "
        "Re-measure and update README.md and docs/ in the same commit."
    )
