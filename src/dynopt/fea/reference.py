"""Pure-NumPy 3D frame finite element solver: the *reference tier*.

Why this exists
---------------
A resonance optimiser needs hundreds of modal solves.  Before wiring up a
licensed solver you want a backend that is:

* dependency-free (numpy/scipy only) so CI can run it,
* fast enough (<10 ms/solve) to smoke-test the optimiser loop, and
* **verifiable against closed-form solutions**, which is what makes the rest of
  the pipeline trustworthy.  See ``tests/test_beam_validation.py``.

It implements the standard 12-DOF two-node space frame element with Timoshenko
shear flexibility and a consistent mass matrix.

Element stiffness (local axes, bending in the x-y plane about z):

.. math::
    k_{xy} = \\frac{EI_z}{(1+\\Phi_y)L^3}
    \\begin{bmatrix}
    12 & 6L & -12 & 6L \\\\
    6L & (4+\\Phi_y)L^2 & -6L & (2-\\Phi_y)L^2 \\\\
    -12 & -6L & 12 & -6L \\\\
    6L & (2-\\Phi_y)L^2 & -6L & (4+\\Phi_y)L^2
    \\end{bmatrix},
    \\qquad \\Phi_y = \\frac{12EI_z}{\\kappa G A L^2}

with :math:`\\Phi \\to 0` recovering Euler-Bernoulli.  The generalised
eigenproblem solved is :math:`(K - \\omega^2 M)\\phi = 0`.

DOF ordering is ``[ux, uy, uz, rx, ry, rz]`` per node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from .base import ModalResult, ModalSolver

DOF_PER_NODE = 6


@dataclass
class Section:
    """Beam cross-section properties.

    ``J`` is the St. Venant torsion constant (drives torsional *stiffness*);
    ``Ip`` is the polar second moment of area (drives rotary *inertia*).  They
    coincide only for circular sections -- conflating them is a classic source
    of wrong torsional frequencies.
    """

    A: float
    Iy: float
    Iz: float
    J: float
    Ip: float | None = None
    kappa: float = 5.0 / 6.0  # Timoshenko shear coefficient
    #: Multiplier on the element's *inertia* only, leaving stiffness untouched.
    #:
    #: A beam section sets mass and stiffness from one area, which is wrong
    #: whenever a 1D element stands in for a 2D feature: overlapping members
    #: that share material double-count it. Equivalent to a negative
    #: non-structural mass on a NASTRAN PBEAM, and the only correct way to fix
    #: an idealisation's mass without softening it.
    mass_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.Ip is None:
            self.Ip = self.Iy + self.Iz

    @classmethod
    def rectangle(cls, b: float, h: float, mass_factor: float = 1.0) -> "Section":
        """Solid rectangle, ``b`` wide (local y) by ``h`` deep (local z)."""
        a, c = max(b, h) / 2.0, min(b, h) / 2.0
        # Roark's approximation for the torsion constant of a solid rectangle.
        J = a * c**3 * (16.0 / 3.0 - 3.36 * (c / a) * (1.0 - c**4 / (12.0 * a**4)))
        return cls(A=b * h, Iy=b * h**3 / 12.0, Iz=h * b**3 / 12.0, J=J,
                   mass_factor=mass_factor)

    @classmethod
    def circle(cls, d: float, mass_factor: float = 1.0) -> "Section":
        I = np.pi * d**4 / 64.0
        return cls(A=np.pi * d**2 / 4.0, Iy=I, Iz=I, J=2 * I, Ip=2 * I, kappa=0.9,
                   mass_factor=mass_factor)

    @classmethod
    def tube(cls, d_out: float, t: float, mass_factor: float = 1.0) -> "Section":
        """Thin/thick-walled circular tube -- the drone-arm workhorse section."""
        d_in = max(d_out - 2.0 * t, 0.0)
        I = np.pi * (d_out**4 - d_in**4) / 64.0
        A = np.pi * (d_out**2 - d_in**2) / 4.0
        return cls(A=A, Iy=I, Iz=I, J=2 * I, Ip=2 * I, kappa=0.53,
                   mass_factor=mass_factor)

    @classmethod
    def box(cls, b: float, h: float, t: float, mass_factor: float = 1.0) -> "Section":
        """Rectangular hollow section (thin wall), b x h outer, wall t."""
        bi, hi = b - 2 * t, h - 2 * t
        if bi <= 0 or hi <= 0:
            return cls.rectangle(b, h, mass_factor=mass_factor)
        A = b * h - bi * hi
        Iy = (b * h**3 - bi * hi**3) / 12.0
        Iz = (h * b**3 - hi * bi**3) / 12.0
        # Bredt-Batho for a closed thin-walled section.
        am, bm = b - t, h - t
        J = 2.0 * t * (am * bm) ** 2 / (am + bm)
        return cls(A=A, Iy=Iy, Iz=Iz, J=J, Ip=Iy + Iz, mass_factor=mass_factor)


@dataclass
class Material:
    E: float          # Young's modulus [Pa]
    rho: float        # density [kg/m^3]
    nu: float = 0.33  # Poisson ratio

    @property
    def G(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))


#: A few stock materials, SI units.
MATERIALS = {
    "AL6061": Material(E=68.9e9, rho=2700.0, nu=0.33),
    "AL7075": Material(E=71.7e9, rho=2810.0, nu=0.33),
    "STEEL": Material(E=200e9, rho=7850.0, nu=0.30),
    "TI64": Material(E=113.8e9, rho=4430.0, nu=0.342),
    "CFRP_QI": Material(E=70e9, rho=1600.0, nu=0.30),  # quasi-isotropic laminate
}


@dataclass
class FrameModel:
    """A parametric space-frame: nodes, elements, supports, lumped masses."""

    nodes: np.ndarray                      # (n_nodes, 3) coordinates [m]
    elements: list[tuple[int, int]]        # node index pairs
    sections: list[Section]                # one per element
    material: Material
    fixed_nodes: list[int] = field(default_factory=list)
    fixed_dofs: list[tuple[int, int]] = field(default_factory=list)
    point_masses: dict[int, float] = field(default_factory=dict)
    point_inertias: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    roll: list[float] = field(default_factory=list)  # per-element roll angle [rad]

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        if len(self.sections) != len(self.elements):
            raise ValueError("need one Section per element")
        if not self.roll:
            self.roll = [0.0] * len(self.elements)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dof(self) -> int:
        return self.n_nodes * DOF_PER_NODE

    def element_length(self, e: int) -> float:
        i, j = self.elements[e]
        return float(np.linalg.norm(self.nodes[j] - self.nodes[i]))

    def structural_mass(self) -> float:
        """Mass of the beams only (point masses are non-structural payload)."""
        return float(
            sum(
                self.material.rho * s.A * s.mass_factor * self.element_length(e)
                for e, s in enumerate(self.sections)
            )
        )

    def total_mass(self) -> float:
        return self.structural_mass() + float(sum(self.point_masses.values()))

    def constrained_dofs(self) -> np.ndarray:
        dofs = set()
        for n in self.fixed_nodes:
            dofs.update(range(n * DOF_PER_NODE, (n + 1) * DOF_PER_NODE))
        for n, d in self.fixed_dofs:
            dofs.add(n * DOF_PER_NODE + d)
        return np.array(sorted(dofs), dtype=int)


def _transform(p1: np.ndarray, p2: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """12x12 local->global rotation for a frame element.

    The local x axis runs from ``p1`` to ``p2``.  The local z axis is chosen to
    lie in the global vertical plane, with the degenerate vertical-member case
    handled explicitly; ``roll`` then spins the section about the beam axis.
    """
    dx = p2 - p1
    L = np.linalg.norm(dx)
    if L <= 0:
        raise ValueError("zero-length element")
    ex = dx / L

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(ex @ up)) > 0.999:  # member is (near) vertical
        up = np.array([0.0, 1.0, 0.0])
    ey = np.cross(up, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    if roll:
        c, s = np.cos(roll), np.sin(roll)
        ey, ez = c * ey + s * ez, -s * ey + c * ez

    R = np.vstack([ex, ey, ez])           # rows are local axes in global coords
    T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b : 3 * b + 3, 3 * b : 3 * b + 3] = R
    return T


def element_matrices(
    mat: Material, sec: Section, L: float, shear: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Local 12x12 stiffness and consistent mass matrices."""
    E, G, rho = mat.E, mat.G, mat.rho
    A, Iy, Iz, J, Ip = sec.A, sec.Iy, sec.Iz, sec.J, sec.Ip

    if shear:
        phi_y = 12.0 * E * Iz / (sec.kappa * G * A * L**2)  # bending in x-y
        phi_z = 12.0 * E * Iy / (sec.kappa * G * A * L**2)  # bending in x-z
    else:
        phi_y = phi_z = 0.0

    k = np.zeros((12, 12))

    # --- axial ---------------------------------------------------------------
    ka = E * A / L
    k[0, 0] = k[6, 6] = ka
    k[0, 6] = k[6, 0] = -ka

    # --- torsion -------------------------------------------------------------
    kt = G * J / L
    k[3, 3] = k[9, 9] = kt
    k[3, 9] = k[9, 3] = -kt

    # --- bending in x-y plane (uy, rz) about local z -------------------------
    c = E * Iz / ((1.0 + phi_y) * L**3)
    idx = [1, 5, 7, 11]
    kb = c * np.array(
        [
            [12.0, 6 * L, -12.0, 6 * L],
            [6 * L, (4 + phi_y) * L**2, -6 * L, (2 - phi_y) * L**2],
            [-12.0, -6 * L, 12.0, -6 * L],
            [6 * L, (2 - phi_y) * L**2, -6 * L, (4 + phi_y) * L**2],
        ]
    )
    k[np.ix_(idx, idx)] += kb

    # --- bending in x-z plane (uz, ry) about local y -------------------------
    c = E * Iy / ((1.0 + phi_z) * L**3)
    idx = [2, 4, 8, 10]
    kb = c * np.array(
        [
            [12.0, -6 * L, -12.0, -6 * L],
            [-6 * L, (4 + phi_z) * L**2, 6 * L, (2 - phi_z) * L**2],
            [-12.0, 6 * L, 12.0, 6 * L],
            [-6 * L, (2 - phi_z) * L**2, 6 * L, (4 + phi_z) * L**2],
        ]
    )
    k[np.ix_(idx, idx)] += kb

    # --- consistent mass -----------------------------------------------------
    # mass_factor scales inertia only; the stiffness block above is untouched.
    m = np.zeros((12, 12))
    rho = rho * sec.mass_factor
    ma = rho * A * L / 6.0
    for a, b in ((0, 6), (3, 9)):
        scale = ma if a == 0 else rho * Ip * L / 6.0
        m[a, a] = m[b, b] = 2.0 * scale
        m[a, b] = m[b, a] = scale

    mc = rho * A * L / 420.0
    idx = [1, 5, 7, 11]
    mb = mc * np.array(
        [
            [156.0, 22 * L, 54.0, -13 * L],
            [22 * L, 4 * L**2, 13 * L, -3 * L**2],
            [54.0, 13 * L, 156.0, -22 * L],
            [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
        ]
    )
    m[np.ix_(idx, idx)] += mb

    idx = [2, 4, 8, 10]
    mb = mc * np.array(
        [
            [156.0, -22 * L, 54.0, 13 * L],
            [-22 * L, 4 * L**2, -13 * L, -3 * L**2],
            [54.0, -13 * L, 156.0, 22 * L],
            [13 * L, -3 * L**2, 22 * L, 4 * L**2],
        ]
    )
    m[np.ix_(idx, idx)] += mb

    return k, m


def assemble(model: FrameModel, shear: bool = True) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Assemble global K and M (unconstrained, size ``n_dof``)."""
    n = model.n_dof
    rows, cols, kvals, mvals = [], [], [], []

    for e, (i, j) in enumerate(model.elements):
        L = model.element_length(e)
        ke, me = element_matrices(model.material, model.sections[e], L, shear=shear)
        T = _transform(model.nodes[i], model.nodes[j], model.roll[e])
        ke = T.T @ ke @ T
        me = T.T @ me @ T

        dofs = np.r_[
            np.arange(i * DOF_PER_NODE, (i + 1) * DOF_PER_NODE),
            np.arange(j * DOF_PER_NODE, (j + 1) * DOF_PER_NODE),
        ]
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.append(rr.ravel())
        cols.append(cc.ravel())
        kvals.append(ke.ravel())
        mvals.append(me.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kvals), (rows, cols)), shape=(n, n)).tocsr()
    M = sp.coo_matrix((np.concatenate(mvals), (rows, cols)), shape=(n, n)).tocsr()

    if model.point_masses or model.point_inertias:
        diag = np.zeros(n)
        for node, mass in model.point_masses.items():
            diag[node * DOF_PER_NODE : node * DOF_PER_NODE + 3] += mass
        for node, (ixx, iyy, izz) in model.point_inertias.items():
            base = node * DOF_PER_NODE + 3
            diag[base : base + 3] += np.array([ixx, iyy, izz])
        M = M + sp.diags(diag)

    return K.tocsr(), M.tocsr()


def apply_bcs(
    K: sp.spmatrix, M: sp.spmatrix, fixed: np.ndarray, n_dof: int
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Row/column elimination.  Returns reduced K, M and the kept-DOF indices."""
    keep = np.setdiff1d(np.arange(n_dof), fixed)
    return K.tocsr()[keep][:, keep], M.tocsr()[keep][:, keep], keep


def modal_solve(
    model: FrameModel, n_modes: int = 10, shear: bool = True, rigid_tol: float = 1e-3
) -> ModalResult:
    """Assemble, constrain, and solve :math:`(K-\\omega^2 M)\\phi=0`.

    Mode shapes come back mass-normalised and expanded to full (unconstrained)
    DOF numbering, so downstream code can index them by physical node.
    Rigid-body modes (f < ``rigid_tol`` Hz) are dropped from a free-free model.
    """
    K, M = assemble(model, shear=shear)
    fixed = model.constrained_dofs()
    Kr, Mr, keep = apply_bcs(K, M, fixed, model.n_dof)

    # Dense symmetric solver: exact and robust at the sizes this tier targets
    # (< ~3000 DOF).  Large meshes belong in the CalculiX / MAPDL backends,
    # which use Lanczos internally.
    Kd = np.asarray(Kr.todense())
    Md = np.asarray(Mr.todense())
    Kd = 0.5 * (Kd + Kd.T)
    Md = 0.5 * (Md + Md.T)

    n_req = min(n_modes + 6, Kd.shape[0])
    lam, vec = eigh(Kd, Md, subset_by_index=[0, n_req - 1])

    lam = np.clip(lam, 0.0, None)
    freqs = np.sqrt(lam) / (2.0 * np.pi)

    live = freqs > rigid_tol
    freqs, vec = freqs[live][:n_modes], vec[:, live][:, :n_modes]

    phi = np.zeros((model.n_dof, freqs.size))
    phi[keep, :] = vec

    node_ids = np.repeat(np.arange(model.n_nodes), DOF_PER_NODE)
    local = np.tile(np.arange(DOF_PER_NODE), model.n_nodes)

    return ModalResult(
        frequencies=freqs,
        modes=phi,
        mass=model.total_mass(),
        K=K,
        M=M,
        dof_map=np.column_stack([node_ids, local]),
        meta={
            "backend": "reference-frame",
            "free_dofs": keep,
            "n_dof": model.n_dof,
            "structural_mass": model.structural_mass(),
            # Node coordinates and the payload location are needed by the
            # physics-based mode classifier in dynopt.dynamics.modal.
            "nodes": model.nodes,
            "payload_nodes": sorted(model.point_masses),
        },
    )


class ReferenceFrameSolver(ModalSolver):
    """Adapts a parametric :class:`FrameModel` builder to the ModalSolver API."""

    name = "reference-frame"

    def __init__(self, builder, shear: bool = True):
        self.builder = builder  # callable: dict[str, float] -> FrameModel
        self.shear = shear

    def solve(self, params: dict[str, float], n_modes: int = 10) -> ModalResult:
        model = self.builder(params)
        res = modal_solve(model, n_modes=n_modes, shear=self.shear)
        res.meta["params"] = dict(params)
        return res
