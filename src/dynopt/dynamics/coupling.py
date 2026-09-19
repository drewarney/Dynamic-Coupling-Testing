r"""Cross-modal dynamic coupling.

Classical modal analysis assumes the modes decouple.  They do *only* when
damping is proportional (Rayleigh / Caughey) and the structure is non-rotating.
Real hardware violates both: bolted joints and elastomer isolators are local
dampers, and anything on a motor carries gyroscopic terms.  The result is energy
transfer between modes -- a torsion mode getting excited by a purely vertical
force because it is coupled to a bending mode that the force does reach.

Three coupling mechanisms are quantified here.

**1. Non-proportional damping.**  Project the physical damping matrix onto the
modal basis:

.. math:: \tilde{C} = \Phi^T C \Phi

If :math:`\tilde{C}` is diagonal the modes are decoupled.  Otherwise define a
normalised coupling index

.. math:: \kappa_{ij} = \frac{|\tilde{C}_{ij}|}{\sqrt{\tilde{C}_{ii}\tilde{C}_{jj}}}

**2. Frequency proximity (modal veering).**  Two modes with close frequencies
exchange energy readily.  The classic two-DOF perturbation result gives the
energy-transfer sensitivity as

.. math:: \eta_{ij} = \frac{\kappa_{ij}}{\sqrt{\kappa_{ij}^2 + (\Delta_{ij}/2)^2}},
          \qquad \Delta_{ij} = \frac{|\omega_i - \omega_j|}{(\omega_i+\omega_j)/2}

so coupling matters when the coupling strength is comparable to the fractional
frequency separation.  Two well-separated modes are effectively independent even
with strong :math:`\kappa`.

**3. Gyroscopic coupling.**  For a spinning rotor of speed :math:`\Omega` the
equation of motion gains a skew-symmetric term,
:math:`M\ddot{u} + (C+\Omega G)\dot{u} + Ku = f`, which splits each bending mode
into forward- and backward-whirl branches.  Their intersection with the
excitation orders is what a Campbell diagram shows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..fea.base import ModalResult
from .modal import (
    _as_array,
    canonicalize_degenerate_basis,
    mass_normalize,
    modal_damping_matrix_raw,
)


def rayleigh_damping(M, K, zeta1: float, f1: float, zeta2: float, f2: float):
    r"""Classical Rayleigh damping :math:`C = \alpha M + \beta K`.

    Solves for :math:`\alpha,\beta` that hit target damping ratios at two
    frequencies, from :math:`\zeta_i = \frac{\alpha}{2\omega_i}+\frac{\beta\omega_i}{2}`.
    This damping is *proportional* by construction, so it produces no cross-modal
    coupling -- use it as the baseline to compare against.
    """
    w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
    A = np.array([[1.0 / (2 * w1), w1 / 2.0], [1.0 / (2 * w2), w2 / 2.0]])
    alpha, beta = np.linalg.solve(A, np.array([zeta1, zeta2]))
    return alpha * M + beta * K, float(alpha), float(beta)


def discrete_damper_matrix(n_dof: int, dampers: list[tuple[int, float]]):
    r"""Assemble C for grounded discrete dashpots ``(dof, c)`` in N.s/m.

    Local dampers -- isolator bushings, bearing films, bonded joints -- are the
    usual physical source of *non*-proportional damping.
    """
    C = sp.lil_matrix((n_dof, n_dof))
    for dof, c in dampers:
        C[dof, dof] += c
    return C.tocsr()


def modal_damping_matrix(
    res: ModalResult, C, canonical: bool = True, rtol: float = 0.02
) -> np.ndarray:
    r""":math:`\tilde{C} = \Phi^T C \Phi` in the mass-normalised modal basis.

    With ``canonical=True`` (the default) the eigenbasis is first canonicalised
    inside every repeated-eigenvalue subspace -- see
    :func:`dynopt.dynamics.modal.canonicalize_degenerate_basis`. Without it, the
    off-diagonal terms involving degenerate modes are an artifact of whichever
    rotation LAPACK returned, and differ between platforms.
    """
    if res.modes is None or res.M is None:
        raise ValueError("need modes and M")
    if canonical:
        res = canonicalize_degenerate_basis(res, C, rtol=rtol)
    return modal_damping_matrix_raw(res, C)


def coupling_index(Ctil: np.ndarray, rel_tol: float = 1e-9) -> np.ndarray:
    r"""Normalised damping coupling matrix :math:`\kappa_{ij}` (diagonal zeroed).

    .. math:: \kappa_{ij} = \frac{|\tilde{C}_{ij}|}
                                  {\sqrt{\tilde{C}_{ii}\tilde{C}_{jj}}}

    **Undamped modes.** A mode can have :math:`\tilde{C}_{ii} = 0` exactly --
    it happens whenever a discrete damper sits on a nodal point of that mode
    shape, which is common on symmetric structures. The ratio is then 0/0.

    Physical damping matrices are positive semi-definite, so Cauchy-Schwarz
    gives :math:`|\tilde{C}_{ij}|^2 \le \tilde{C}_{ii}\tilde{C}_{jj}`: a mode
    with zero modal damping has :math:`\tilde{C}_{ij} = 0` for every ``j`` and
    genuinely cannot exchange energy through damping. Its coupling is therefore
    set to exactly zero rather than left to float on rounding noise -- which
    otherwise amplifies 1e-14 perturbations into differences of ~1e-3 and makes
    the metric irreproducible across platforms.

    ``rel_tol`` is applied relative to the largest diagonal entry.
    """
    d = np.diag(Ctil).astype(float)
    scale = float(np.max(np.abs(d))) if d.size else 0.0
    live = d > rel_tol * scale if scale > 0 else np.zeros_like(d, dtype=bool)

    K = np.zeros_like(Ctil, dtype=float)
    if live.sum() >= 2:
        idx = np.where(live)[0]
        sub = np.sqrt(d[idx])
        K[np.ix_(idx, idx)] = np.abs(Ctil[np.ix_(idx, idx)]) / np.outer(sub, sub)
    np.fill_diagonal(K, 0.0)
    return K


def frequency_separation(freqs: np.ndarray) -> np.ndarray:
    r"""Fractional frequency separation :math:`\Delta_{ij}` between every pair."""
    f = np.asarray(freqs, dtype=float)
    diff = np.abs(f[:, None] - f[None, :])
    mean = (f[:, None] + f[None, :]) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(mean > 0, diff / mean, np.inf)
    return d


def energy_transfer_index(res: ModalResult, C, canonical: bool = True) -> np.ndarray:
    r"""Combined coupling metric :math:`\eta_{ij} \in [0,1]`.

    Folds damping coupling strength together with frequency proximity.  Values
    above ~0.3 flag mode pairs that will exchange energy; those pairs must be
    separated *together*, not individually, or the optimiser will push one into
    the other.
    """
    kap = coupling_index(modal_damping_matrix(res, C, canonical=canonical))
    delta = frequency_separation(res.frequencies)
    eta = kap / np.sqrt(kap**2 + (delta / 2.0) ** 2 + 1e-30)
    np.fill_diagonal(eta, 0.0)
    return eta


def coupled_pairs(
    res: ModalResult, C, threshold: float = 0.3, canonical: bool = True
) -> list[dict]:
    """List mode pairs whose energy-transfer index exceeds ``threshold``.

    With ``canonical=True`` degenerate pairs correctly report near-zero coupling
    instead of a spurious ``eta = 1.0``, so what remains is genuine coupling
    between distinct frequencies.
    """
    eta = energy_transfer_index(res, C, canonical=canonical)
    out = []
    for i in range(eta.shape[0]):
        for j in range(i + 1, eta.shape[1]):
            if eta[i, j] >= threshold:
                out.append(
                    {
                        "mode_i": i + 1,
                        "mode_j": j + 1,
                        "f_i": float(res.frequencies[i]),
                        "f_j": float(res.frequencies[j]),
                        "eta": float(eta[i, j]),
                    }
                )
    return sorted(out, key=lambda d: -d["eta"])


def campbell_diagram(
    base_freqs: np.ndarray,
    rpm_range: tuple[float, float],
    orders: tuple[float, ...] = (1.0, 2.0, 4.0),
    n_rpm: int = 200,
    gyro_coeff: float = 0.0,
) -> dict:
    r"""Campbell data: whirl branches vs engine orders.

    Forward/backward whirl split is approximated as
    :math:`f_{\pm}(\Omega) = f_0 \pm g\,\Omega/60`, with ``gyro_coeff`` = ``g``
    the non-dimensional gyroscopic sensitivity (0 for a non-rotating part; for a
    thin disc on a slender shaft it approaches 1).  An intersection of a whirl
    branch with an order line is a critical speed.
    """
    rpm = np.linspace(rpm_range[0], rpm_range[1], n_rpm)
    rot_hz = rpm / 60.0
    branches = {}
    for i, f0 in enumerate(np.atleast_1d(base_freqs)):
        if gyro_coeff:
            branches[f"mode{i+1}_fw"] = f0 + gyro_coeff * rot_hz
            branches[f"mode{i+1}_bw"] = np.clip(f0 - gyro_coeff * rot_hz, 0.0, None)
        else:
            branches[f"mode{i+1}"] = np.full_like(rot_hz, f0)

    order_lines = {f"{o}X": o * rot_hz for o in orders}

    crossings = []
    for bname, bvals in branches.items():
        for oname, ovals in order_lines.items():
            s = np.sign(bvals - ovals)
            idx = np.where(np.diff(s) != 0)[0]
            for k in idx:
                # linear interpolation onto the zero crossing
                d0, d1 = (bvals - ovals)[k], (bvals - ovals)[k + 1]
                t = d0 / (d0 - d1) if d1 != d0 else 0.0
                crossings.append(
                    {
                        "branch": bname,
                        "order": oname,
                        "rpm": float(rpm[k] + t * (rpm[k + 1] - rpm[k])),
                        "hz": float(bvals[k] + t * (bvals[k + 1] - bvals[k])),
                    }
                )
    return {"rpm": rpm, "branches": branches, "orders": order_lines,
            "crossings": sorted(crossings, key=lambda c: c["rpm"])}
