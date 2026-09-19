"""Modal post-processing: participation, effective mass, MAC, mode tracking.

Natural frequencies alone are not enough to drive an optimiser.  A mode at
3,200 Hz that no excitation can reach is irrelevant; a mode at 3,200 Hz with 60%
of the effective mass in the excitation direction will destroy the part.  The
quantities here turn a raw eigensolution into *actionable risk*.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..fea.base import ModalResult


def _as_array(A):
    return np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)


def mass_normalize(phi: np.ndarray, M) -> np.ndarray:
    """Scale each column so that :math:`\\phi^T M \\phi = 1`."""
    M = _as_array(M)
    out = np.array(phi, dtype=float, copy=True)
    for j in range(out.shape[1]):
        g = float(out[:, j] @ M @ out[:, j])
        if g > 0:
            out[:, j] /= np.sqrt(g)
    return out


def rigid_body_vector(dof_map: np.ndarray, direction: int) -> np.ndarray:
    """Influence vector :math:`r_d`: unit rigid-body translation along an axis.

    ``direction`` is 0/1/2 for global X/Y/Z.  Used for base-excitation
    participation factors -- exactly what a motor mount or a shaker table sees.
    """
    r = np.zeros(dof_map.shape[0])
    r[dof_map[:, 1] == direction] = 1.0
    return r


def participation_factors(res: ModalResult, direction: int) -> np.ndarray:
    r"""Modal participation factors.

    .. math:: \Gamma_j = \frac{\phi_j^T M r_d}{\phi_j^T M \phi_j}

    With mass-normalised modes the denominator is 1, so
    :math:`\Gamma_j = \phi_j^T M r_d`.
    """
    if res.modes is None or res.M is None or res.dof_map is None:
        raise ValueError("participation factors need modes, M and dof_map")
    M = _as_array(res.M)
    phi = mass_normalize(res.modes, M)
    r = rigid_body_vector(res.dof_map, direction)
    return phi.T @ (M @ r)


def effective_modal_mass(res: ModalResult, direction: int) -> np.ndarray:
    r"""Effective modal mass :math:`M_{eff,j} = \Gamma_j^2` [kg] (mass-normalised).

    The sum over all modes tends to the total unconstrained mass in that
    direction; the fraction captured by the retained modes tells you whether the
    modal truncation is trustworthy.
    """
    return participation_factors(res, direction) ** 2


def effective_mass_fraction(res: ModalResult, direction: int) -> np.ndarray:
    """Effective modal mass as a fraction of total model mass."""
    meff = effective_modal_mass(res, direction)
    total = res.mass if np.isfinite(res.mass) and res.mass > 0 else meff.sum()
    return meff / total if total > 0 else meff


def mac(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    r"""Modal Assurance Criterion matrix.

    .. math::
        \mathrm{MAC}_{ij} = \frac{|\phi_{a,i}^T \phi_{b,j}|^2}
                                 {(\phi_{a,i}^T\phi_{a,i})(\phi_{b,j}^T\phi_{b,j})}

    1.0 means the two shapes are collinear, 0.0 orthogonal.  This is how you
    compare FEA against a hammer test -- and how you keep mode *identity*
    straight across design iterations.
    """
    a = np.asarray(phi_a, dtype=float)
    b = np.asarray(phi_b, dtype=float)
    num = np.abs(a.T @ b) ** 2
    na = np.einsum("ij,ij->j", a, a)
    nb = np.einsum("ij,ij->j", b, b)
    return num / np.outer(na, nb)


def track_modes(
    phi_ref: np.ndarray, phi_new: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Match new modes to reference modes by MAC (greedy, one-to-one).

    **This is the single most important utility for surrogate modelling.**  As
    geometry changes, modes cross over in frequency order: "mode 3" at design A
    may be a torsion mode and at design B a bending mode.  Regressing frequency
    against parameters without tracking teaches the surrogate a discontinuous
    function and it will never converge.  Sorting by MAC instead of by frequency
    keeps each regression target physically continuous.

    Returns ``idx`` of length ``n_ref`` where ``idx[i]`` is the column of
    ``phi_new`` matching reference mode ``i``, or -1 if nothing cleared
    ``threshold``.
    """
    m = mac(phi_ref, phi_new)
    n_ref, n_new = m.shape
    out = np.full(n_ref, -1, dtype=int)
    taken = set()
    # Greedy over globally strongest pairs; ties resolved by descending MAC.
    order = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0]
    for i, j in order:
        if out[i] == -1 and j not in taken and m[i, j] >= threshold:
            out[i] = j
            taken.add(int(j))
        if (out != -1).all():
            break
    return out


def modal_summary(res: ModalResult, direction: int = 2) -> list[dict]:
    """Per-mode table: frequency, participation, effective mass fraction."""
    gamma = participation_factors(res, direction)
    frac = effective_mass_fraction(res, direction)
    return [
        {
            "mode": i + 1,
            "f_hz": float(res.frequencies[i]),
            "gamma": float(gamma[i]),
            "eff_mass_frac": float(frac[i]),
        }
        for i in range(res.n_modes)
    ]


# ---------------------------------------------------------------------------
# Degenerate mode handling
# ---------------------------------------------------------------------------
# Symmetric structures (a 4-leg mount, a 3-arm drone frame, any part with a
# rotational symmetry group) have *repeated eigenvalues*.  The eigenvectors of a
# repeated eigenvalue are not unique: any orthonormal rotation within that
# eigenspace is an equally valid answer, and LAPACK will return a different
# rotation for each design point.
#
# Plain MAC tracking therefore fails on exactly these modes -- it tries to match
# vectors that carry no individual physical identity.  Measured naively, a
# surrogate fitted on a symmetric part shows ~30% error on the degenerate modes
# and ~0.5% on the rest, which looks like a modelling failure but is really a
# tracking failure.
#
# The fix is to track the *subspace*, not the vector: group modes into clusters
# of near-equal frequency and match clusters by principal angles.  The cluster
# frequency is then a continuous, well-defined function of the design
# parameters -- which is what the GP needs.


def cluster_modes(freqs: np.ndarray, rtol: float = 0.02) -> list[list[int]]:
    """Group near-degenerate modes into clusters of relative width ``rtol``.

    ``rtol=0.02`` groups modes within 2% of each other.  Too tight and true
    degenerate pairs get split by numerical noise; too loose and physically
    distinct modes get merged.  1-3% suits most metal parts.
    """
    f = np.asarray(freqs, dtype=float)
    if f.size == 0:
        return []
    clusters, current = [], [0]
    for i in range(1, f.size):
        ref = f[current[0]]
        if ref > 0 and abs(f[i] - ref) / ref <= rtol:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)
    return clusters


def subspace_mac(A: np.ndarray, B: np.ndarray) -> float:
    r"""Rotation-invariant MAC between two mode *subspaces*.

    Orthonormalises each set of shapes and returns
    :math:`\|Q_A^T Q_B\|_F^2 / \min(p, q)`, i.e. the mean squared cosine of the
    principal angles.  Unlike a vector MAC this is invariant to any rotation
    within either subspace, which is exactly the ambiguity that repeated
    eigenvalues introduce.  Returns 1.0 for identical subspaces.
    """
    Qa, _ = np.linalg.qr(np.atleast_2d(np.asarray(A, dtype=float)))
    Qb, _ = np.linalg.qr(np.atleast_2d(np.asarray(B, dtype=float)))
    p, q = Qa.shape[1], Qb.shape[1]
    return float(np.linalg.norm(Qa.T @ Qb, "fro") ** 2 / max(min(p, q), 1))


def track_clusters(
    phi_ref: np.ndarray,
    clusters_ref: list[list[int]],
    phi_new: np.ndarray,
    clusters_new: list[list[int]],
    threshold: float = 0.5,
) -> np.ndarray:
    """Greedily match reference clusters to new clusters by :func:`subspace_mac`.

    Returns ``idx`` where ``idx[i]`` is the index into ``clusters_new`` matching
    reference cluster ``i``, or -1 if nothing cleared ``threshold``.
    """
    n_ref, n_new = len(clusters_ref), len(clusters_new)
    S = np.zeros((n_ref, n_new))
    for i, ci in enumerate(clusters_ref):
        for j, cj in enumerate(clusters_new):
            S[i, j] = subspace_mac(phi_ref[:, ci], phi_new[:, cj])

    out = np.full(n_ref, -1, dtype=int)
    taken: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(S, axis=None)[::-1], S.shape))[0]
    for i, j in order:
        if out[i] == -1 and j not in taken and S[i, j] >= threshold:
            out[i] = j
            taken.add(int(j))
        if (out != -1).all():
            break
    return out


def cluster_frequencies(freqs: np.ndarray, clusters: list[list[int]]) -> np.ndarray:
    """Representative (mean) frequency of each cluster -- the regression target."""
    f = np.asarray(freqs, dtype=float)
    return np.array([float(f[c].mean()) for c in clusters])


def track_modes_robust(
    phi_ref: np.ndarray,
    freqs_ref: np.ndarray,
    phi_new: np.ndarray,
    freqs_new: np.ndarray,
    rtol: float = 0.02,
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[list[int]]]:
    """Degenerate-safe mode tracking.

    Returns ``(matched_freqs, ref_clusters)`` where ``matched_freqs[i]`` is the
    frequency of the new cluster matching reference cluster ``i`` (NaN if
    unmatched).  Use this instead of :func:`track_modes` on any part with a
    symmetry -- which is most real hardware.
    """
    cref = cluster_modes(freqs_ref, rtol=rtol)
    cnew = cluster_modes(freqs_new, rtol=rtol)
    idx = track_clusters(phi_ref, cref, phi_new, cnew, threshold=threshold)
    fnew = cluster_frequencies(freqs_new, cnew)
    out = np.full(len(cref), np.nan)
    for i, j in enumerate(idx):
        if j >= 0:
            out[i] = fnew[j]
    return out, cref


# ---------------------------------------------------------------------------
# Physics-based mode classification
# ---------------------------------------------------------------------------
# Shape tracking against a fixed reference design breaks down over a *wide*
# design space.  When the base half-span ranges 40-120 mm and the height
# 30-100 mm, the mode shapes themselves change so much that MAC against the
# reference geometry falls to 0.5-0.8 and the ordering of the physical mode
# families flips: at one design the vertical bounce mode is lowest, at another a
# lateral pair is.  Continuation tracking fixes this along a *path*, but a
# scattered DOE has no path.
#
# The robust alternative is to stop matching shapes and start asking what each
# mode physically *does*.  Project every mode onto the six rigid-body motions of
# the payload and label it by which motion dominates:
#
#     bounce_z   - the payload translates vertically  (the classic mount mode)
#     lateral_xy - the payload translates sideways    (degenerate pair)
#     rocking_xy - the payload pitches/rolls          (degenerate pair)
#     torsion_z  - the payload yaws about the axis
#
# "The lowest bounce_z frequency" is a continuous function of the design
# parameters no matter how the eigenvalue ordering shuffles, so it is a target a
# GP can actually learn.  It is also the quantity an engineer cares about: you
# do not want to move "mode 3", you want to move the bounce mode.

#: Rigid-body DOF groups that define each physical mode family.
#:
#: ``sway_rock`` deliberately merges the two lateral translations with the two
#: rocking rotations.  On a payload carried by compliant legs these are **not**
#: separable: the lateral and rocking motions are inertially coupled through the
#: payload's centre of gravity offset, producing two mixed branches whose
#: character swaps as the mount's height/span ratio changes.  A mode signature
#: of ``[0.66 ux, 0.66 ry]`` is a real mixed sway-rock mode, not a
#: classification failure -- and deciding its label by argmax over two tied
#: components is a coin flip that injects ~50% noise into the surrogate target.
#:
#: Scoring by *group energy* instead sums the ambiguous components rather than
#: letting them compete, then orders the branches within the family by
#: frequency.  Both branches are then continuous in the design parameters.
FAMILY_DOFS: dict[str, tuple[int, ...]] = {
    "bounce_z": (2,),
    "sway_rock": (0, 1, 3, 4),
    "torsion_z": (5,),
}

#: How many frequency-ordered branches to extract per family.
FAMILY_BRANCHES: dict[str, int] = {"bounce_z": 1, "sway_rock": 2, "torsion_z": 1}

#: Target column names, in order -- what the surrogate regresses against.
MODE_FAMILIES = ("bounce_z", "sway_rock_1", "sway_rock_2", "torsion_z")


def rigid_body_basis(
    nodes: np.ndarray, dof_map: np.ndarray, origin: np.ndarray | None = None
) -> np.ndarray:
    r"""Six rigid-body vectors (3 translations + 3 rotations) as ``(n_dof, 6)``.

    A rotation of magnitude 1 about axis :math:`a` through ``origin`` moves node
    :math:`i` by :math:`a \times (x_i - o)` and rotates it by :math:`a`.
    """
    nodes = np.asarray(nodes, dtype=float)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    n_dof = dof_map.shape[0]
    R = np.zeros((n_dof, 6))

    node_id = dof_map[:, 0].astype(int)
    local = dof_map[:, 1].astype(int)

    for d in range(3):  # translations
        R[(local == d), d] = 1.0

    for a in range(3):  # rotations about x, y, z
        axis = np.zeros(3)
        axis[a] = 1.0
        rel = nodes[node_id] - o
        disp = np.cross(np.broadcast_to(axis, rel.shape), rel)
        for d in range(3):
            sel = local == d
            R[sel, 3 + a] = disp[sel, d]
        R[(local == 3 + a), 3 + a] = 1.0
    return R


def mode_signature(
    res: ModalResult, origin: np.ndarray | None = None
) -> np.ndarray:
    r"""Normalised rigid-body participation of each mode, shape ``(n_modes, 6)``.

    Each rigid-body vector is scaled by its own generalised mass
    :math:`\sqrt{r_d^T M r_d}` so translations (metres) and rotations (radians)
    are comparable, then each mode's 6-vector is normalised to unit length.
    Row ``j`` reads as "how much of mode j is vertical / lateral / torsional".
    """
    if res.modes is None or res.M is None or res.dof_map is None:
        raise ValueError("mode_signature needs modes, M and dof_map")
    nodes = res.meta.get("nodes")
    if nodes is None:
        raise ValueError("ModalResult.meta['nodes'] is required for classification")

    M = _as_array(res.M)
    phi = mass_normalize(res.modes, M)
    R = rigid_body_basis(nodes, res.dof_map, origin)

    gen = np.sqrt(np.clip(np.einsum("ij,jk,ki->i", R.T, M, R), 1e-30, None))
    G = np.abs(phi.T @ (M @ R)) / gen[None, :]
    norm = np.linalg.norm(G, axis=1, keepdims=True)
    return G / np.clip(norm, 1e-30, None)


def family_scores(S: np.ndarray) -> dict[str, np.ndarray]:
    r"""Energy each mode puts into each family's DOF group.

    .. math:: E_{j,\mathcal{F}} = \sqrt{\sum_{d \in \mathcal{F}} S_{jd}^2}

    Summing within the group is what makes the assignment stable for mixed
    modes: a mode split evenly between ``ux`` and ``ry`` scores high on
    ``sway_rock`` either way, instead of flipping between two labels.
    """
    return {
        fam: np.sqrt(np.sum(S[:, list(dofs)] ** 2, axis=1))
        for fam, dofs in FAMILY_DOFS.items()
    }


def classify_modes(
    res: ModalResult, origin: np.ndarray | None = None, min_purity: float = 0.0
) -> list[str]:
    """Label each mode with the family holding most of its rigid-body energy.

    Modes whose best family score is below ``min_purity`` -- local panel or
    higher-order modes that barely move the payload -- are labelled ``"local"``.
    They are real modes, but they carry little effective mass and are not what
    the optimiser should chase.
    """
    S = mode_signature(res, origin)
    E = family_scores(S)
    names = list(FAMILY_DOFS)
    scores = np.column_stack([E[n] for n in names])
    out = []
    for row in scores:
        k = int(np.argmax(row))
        out.append(names[k] if row[k] >= min_purity else "local")
    return out


def family_frequencies(
    res: ModalResult,
    origin: np.ndarray | None = None,
    families: tuple[str, ...] = MODE_FAMILIES,
    min_purity: float = 0.35,
) -> dict[str, float]:
    """Frequency of each physical mode branch (NaN if absent).

    This is the surrogate's regression target: a fixed-length, physically
    labelled, continuous vector that survives eigenvalue reordering *and*
    degenerate eigenspaces.  Multi-branch families (``sway_rock``) yield
    ``sway_rock_1``, ``sway_rock_2``, ... ordered by frequency.
    """
    labels = classify_modes(res, origin, min_purity=min_purity)
    found: dict[str, list[float]] = {fam: [] for fam in FAMILY_DOFS}
    seen: dict[str, float] = {}

    for f, lab in zip(res.frequencies, labels):
        if lab not in found:
            continue
        # Collapse degenerate repeats: a pair at the same frequency is one branch.
        prev = seen.get(lab)
        if prev is not None and prev > 0 and abs(f - prev) / prev < 0.02:
            continue
        found[lab].append(float(f))
        seen[lab] = float(f)

    out = {fam: np.nan for fam in families}
    for fam, n in FAMILY_BRANCHES.items():
        vals = sorted(found.get(fam, []))
        for b in range(n):
            key = fam if n == 1 else f"{fam}_{b+1}"
            if key in out and b < len(vals):
                out[key] = vals[b]
    return out


def canonicalize_degenerate_basis(res: ModalResult, C, rtol: float = 0.02) -> ModalResult:
    r"""Fix the arbitrary eigenbasis inside repeated-eigenvalue subspaces.

    **Why this is necessary.** Eigenvectors of a repeated eigenvalue are defined
    only up to an arbitrary orthonormal rotation within their eigenspace, and
    different LAPACK builds return different rotations for the same input
    (macOS Accelerate and Linux OpenBLAS demonstrably disagree). Any quantity
    computed from the *individual* vectors of a degenerate set is therefore not
    a physical quantity at all.

    That invalidates the two most natural things to ask about modal coupling:

    * :math:`\kappa_{ij}` between two degenerate modes is whatever the rotation
      happened to make it -- reporting "modes 2 and 3 are coupled, eta = 1.0"
      for a degenerate pair is an artifact, not physics.
    * Zeroing the off-diagonals of :math:`\tilde{C}` -- the textbook uncoupled
      SDOF approximation -- is not invariant under that rotation. Measured on
      the benchmark mount, the "error from ignoring coupling" swings between
      -1.5% and -29.9% across valid bases of the same eigensolution.

    **The fix.** Within each degenerate cluster, exactly one rotation
    diagonalises the damping block restricted to that subspace. Pick it. Since
    every rotation in the subspace is a valid eigenbasis, this costs nothing and
    is uniquely determined:

    .. math:: \tilde{C}_{\mathcal{D}\mathcal{D}} = V \Lambda V^T,
              \qquad \Phi_{\mathcal{D}} \leftarrow \Phi_{\mathcal{D}} V

    Afterwards "ignore the off-diagonal terms" is a well-posed operation, the
    residual off-diagonals represent genuine coupling between *distinct*
    frequencies, and results are reproducible across machines.

    Returns a shallow copy with rotated ``modes``; the input is not mutated.
    """
    import copy

    if res.modes is None or res.M is None:
        return res

    Ct = modal_damping_matrix_raw(res, C)
    phi = res.modes.copy()
    for cl in cluster_modes(res.frequencies, rtol=rtol):
        if len(cl) < 2:
            continue
        blk = Ct[np.ix_(cl, cl)]
        _, V = np.linalg.eigh(0.5 * (blk + blk.T))   # symmetric => orthogonal V
        phi[:, cl] = phi[:, cl] @ V

    out = copy.copy(res)
    out.modes = phi
    return out


def modal_damping_matrix_raw(res: ModalResult, C) -> np.ndarray:
    r"""Raw projection :math:`\Phi^T C \Phi` with no basis canonicalisation.

    Separated out so :func:`canonicalize_degenerate_basis` can call it without
    recursing. Prefer :func:`dynopt.dynamics.coupling.modal_damping_matrix`,
    which canonicalises first.
    """
    if res.modes is None or res.M is None:
        raise ValueError("need modes and M")
    phi = mass_normalize(res.modes, res.M)
    return phi.T @ _as_array(C) @ phi
