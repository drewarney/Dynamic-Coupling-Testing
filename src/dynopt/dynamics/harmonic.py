r"""Harmonic (frequency) response.

The steady-state response to :math:`f(t) = F e^{i\omega t}` is
:math:`u(t) = U(\omega)e^{i\omega t}` with

.. math:: U(\omega) = \underbrace{\left[K + i\omega C - \omega^2 M\right]^{-1}}_{H(\omega)} F

Two routes are provided:

* :func:`direct_frf` -- factorise the full complex dynamic stiffness at every
  frequency.  Exact, and the only correct option when damping is strongly
  non-proportional, but O(n^3) per frequency line.
* :func:`modal_frf` -- project onto the retained modes first:

  .. math:: \tilde{H}(\omega) = \Phi\left[\tilde{K} + i\omega\tilde{C}
            - \omega^2 \tilde{M}\right]^{-1}\Phi^T

  With mass-normalised modes :math:`\tilde{M}=I`, :math:`\tilde{K}=\Omega^2`,
  and :math:`\tilde{C}=\Phi^T C \Phi`.  Crucially we **keep the off-diagonal
  terms of** :math:`\tilde{C}` and solve the small coupled system rather than
  assuming SDOF modes -- that is what captures cross-modal coupling.  The
  reduced system is typically 10-30 DOF, giving 3-4 orders of magnitude speedup
  over the direct route.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ..fea.base import ModalResult
from .modal import _as_array, canonicalize_degenerate_basis, mass_normalize


def direct_frf(K, M, C, freqs_hz, F, dofs_out=None) -> np.ndarray:
    """Exact FRF by direct inversion.  Returns ``(n_out, n_freq)`` complex."""
    K, M, C = sp.csc_matrix(K), sp.csc_matrix(M), sp.csc_matrix(C)
    F = np.asarray(F, dtype=complex).ravel()
    freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    n_out = F.size if dofs_out is None else len(dofs_out)
    out = np.zeros((n_out, freqs.size), dtype=complex)
    for k, f in enumerate(freqs):
        w = 2.0 * np.pi * f
        Z = K + 1j * w * C - (w**2) * M
        u = spsolve(Z.tocsc(), F)
        out[:, k] = u if dofs_out is None else u[dofs_out]
    return out


def modal_frf(
    res: ModalResult,
    C,
    freqs_hz,
    F,
    dofs_out=None,
    keep_coupling: bool = True,
    canonical: bool = True,
) -> np.ndarray:
    r"""FRF via modal superposition, retaining cross-modal damping coupling.

    Set ``keep_coupling=False`` to zero the off-diagonal terms of
    :math:`\tilde{C}` -- the textbook uncoupled-SDOF assumption.  Comparing the
    two curves is the cleanest way to *show* how much coupling matters for a
    given design, and makes a compelling plot in a project report.

    ``canonical=True`` (default) canonicalises the basis inside degenerate
    subspaces before dropping those terms. Do not turn it off: without it the
    comparison is basis-dependent and varies by tens of percent between
    platforms for the same model. See
    :func:`dynopt.dynamics.modal.canonicalize_degenerate_basis`.
    """
    if res.modes is None or res.M is None:
        raise ValueError("modal_frf needs modes and M")

    # Dropping off-diagonal damping terms is only well posed once the basis
    # inside each degenerate subspace is pinned down; otherwise the result
    # depends on which rotation LAPACK returned and is not reproducible.
    if canonical and not keep_coupling:
        res = canonicalize_degenerate_basis(res, C)

    phi = mass_normalize(res.modes, res.M)
    free = res.meta.get("free_dofs")
    if free is not None:
        phi_r = phi[free, :]
        F_full = np.asarray(F, dtype=complex).ravel()
        Fr = F_full[free]
        Cr = _as_array(C)
        if Cr.shape[0] == phi.shape[0]:
            Cr = Cr[np.ix_(free, free)]
    else:
        phi_r, Fr, Cr = phi, np.asarray(F, dtype=complex).ravel(), _as_array(C)

    Ctil = phi_r.T @ Cr @ phi_r
    if not keep_coupling:
        Ctil = np.diag(np.diag(Ctil))

    w_n = res.omega()
    Ktil = np.diag(w_n**2)
    Mtil = np.eye(phi_r.shape[1])
    Ftil = phi_r.T @ Fr

    freqs = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    q = np.zeros((phi_r.shape[1], freqs.size), dtype=complex)
    for k, f in enumerate(freqs):
        w = 2.0 * np.pi * f
        q[:, k] = np.linalg.solve(Ktil + 1j * w * Ctil - (w**2) * Mtil, Ftil)

    u = phi_r @ q
    if dofs_out is None:
        full = np.zeros((phi.shape[0], freqs.size), dtype=complex)
        if free is not None:
            full[free, :] = u
        else:
            full = u
        return full
    sel = dofs_out if free is None else [int(np.where(free == d)[0][0]) for d in dofs_out]
    return u[sel, :]


def modal_damping_from_zeta(res: ModalResult, zeta) -> np.ndarray:
    r"""Build a *proportional* modal damping matrix from per-mode ratios.

    :math:`\tilde{C} = \mathrm{diag}(2\zeta_j\omega_j)`.  Handy when you have
    measured damping ratios from a hammer test but no physical C matrix.
    """
    z = np.broadcast_to(np.atleast_1d(np.asarray(zeta, dtype=float)), (res.n_modes,))
    return np.diag(2.0 * z * res.omega())


def rms_response(H: np.ndarray, freqs_hz: np.ndarray, psd=None) -> np.ndarray:
    r"""Band-limited RMS of an FRF, optionally weighted by an input PSD.

    .. math:: u_{rms} = \sqrt{\int |H(f)|^2 S_{ff}(f)\, df}

    With ``psd=None`` this is a flat-input RMS -- a scalar "how loud is this
    design over the operating band" objective the optimiser can minimise.
    """
    H = np.atleast_2d(H)
    f = np.asarray(freqs_hz, dtype=float)
    S = np.ones_like(f) if psd is None else np.asarray(psd, dtype=float)
    return np.sqrt(np.trapezoid(np.abs(H) ** 2 * S, f, axis=-1))


def transmissibility(H_out: np.ndarray, H_in: np.ndarray) -> np.ndarray:
    """Ratio of response to input motion -- the isolator figure of merit."""
    return np.abs(H_out) / np.clip(np.abs(H_in), 1e-30, None)
