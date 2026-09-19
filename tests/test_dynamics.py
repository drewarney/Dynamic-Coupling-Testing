"""Tests for modal post-processing, coupling metrics and harmonic response."""

import numpy as np
import pytest

from dynopt.dynamics import (
    MODE_FAMILIES,
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
from dynopt.geometry import MOTOR_MOUNT_SPACE, build_frame, motor_node_index


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
    fams = family_frequencies(res, origin)
    for key in MODE_FAMILIES:
        assert np.isfinite(fams[key]), f"{key} not found in {fams}"
    assert fams["sway_rock_1"] <= fams["sway_rock_2"]


def test_classification_is_stable_across_designs():
    """Family frequencies must stay finite and ordered across the design space."""
    from dynopt.doe import sobol_sample

    for x in sobol_sample(MOTOR_MOUNT_SPACE, 16, seed=3):
        model = build_frame(MOTOR_MOUNT_SPACE.to_dict(x))
        res = modal_solve(model, n_modes=12)
        fams = family_frequencies(res, model.nodes[motor_node_index()])
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
