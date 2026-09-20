"""Solver-agnostic contract for modal extraction.

Every backend (reference beam FE, Gmsh+CalculiX, PyAnsys/MAPDL) returns the same
:class:`ModalResult`, so the dynamics / surrogate / optimizer layers never learn
which solver produced the numbers.  Swapping CalculiX for MAPDL is a one-line
change in the driver config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


#: Meta keys the layers above a backend read, with what each is for.
#:
#: This is a **contract**, not a convention. Every consumer below reads these
#: with a silent fallback, so a backend that omits one does not fail -- it
#: produces a plausible wrong answer:
#:
#: * ``cli.py`` damps ``payload_nodes[0]``; absent, it damped node 0, which on
#:   the motor mount is a *clamped* base corner, and reported coupling for it;
#: * ``PartSpec.classification_origin`` needs ``nodes``; absent, it returns
#:   ``None`` and mode classification silently degrades to nothing;
#: * ``ResonanceObjective`` divides by ``mass``; absent (``nan``), every
#:   objective in the run is ``nan`` and the optimiser compares ``nan`` to
#:   ``nan``.
#:
#: :meth:`ModalResult.missing_contract` reports what a result lacks, and
#: ``tests/test_backend_contract.py`` runs it over every backend.
CONTRACT_META = {
    "nodes": "(n_nodes, 3) node coordinates, in the same order as the rows of "
             "`modes` -- three rows (ux, uy, uz) or six (plus rotations) per node",
    "payload_nodes": "indices into `nodes` carrying the payload; [] when a run "
                     "is deliberately bare",
    "n_dof": "number of degrees of freedom in the solved system",
}


@dataclass
class ModalResult:
    """Output of a modal (eigenvalue) extraction.

    Attributes
    ----------
    frequencies:
        Natural frequencies in Hz, ascending, shape ``(n_modes,)``.
    modes:
        Mass-normalised mode shapes, shape ``(n_dof, n_modes)``.  Column ``j`` is
        :math:`\\phi_j` with :math:`\\phi_j^T M \\phi_j = 1`.
    mass:
        Total structural mass in kg.
    K, M:
        Global stiffness / mass matrices with boundary conditions applied.  These
        are optional: CalculiX and MAPDL can both export them, but a cheap run
        may skip them.  Cross-modal coupling analysis needs them.
    dof_map:
        ``(n_dof, 2)`` integer array of ``(node_id, local_dof)`` so mode shapes can
        be projected back onto physical directions.
    """

    frequencies: np.ndarray
    modes: np.ndarray | None = None
    mass: float = float("nan")
    K: sp.spmatrix | np.ndarray | None = None
    M: sp.spmatrix | np.ndarray | None = None
    dof_map: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        if self.modes is not None:
            self.modes = np.asarray(self.modes, dtype=float)
            if self.modes.shape[1] != self.frequencies.size:
                raise ValueError(
                    f"modes has {self.modes.shape[1]} columns but "
                    f"{self.frequencies.size} frequencies were given"
                )

    @property
    def n_modes(self) -> int:
        return int(self.frequencies.size)

    def missing_contract(self) -> list[str]:
        """What this result fails to supply, as human-readable complaints.

        Empty means the result is usable by every layer above the solver.
        """
        gaps = []
        for key, why in CONTRACT_META.items():
            if key == "payload_nodes":
                # An empty list is a legitimate answer ("no payload"); the key
                # being absent entirely is not, because a consumer cannot tell
                # "bare" from "this backend forgot".
                if key not in self.meta:
                    gaps.append(f"meta[{key!r}] absent -- {why}")
                continue
            if self.meta.get(key) is None:
                gaps.append(f"meta[{key!r}] missing -- {why}")

        if not np.isfinite(self.mass):
            gaps.append(
                "mass is not finite -- ResonanceObjective divides by it, so "
                "every objective in the run would be nan"
            )

        nodes = self.meta.get("nodes")
        if nodes is not None and self.modes is not None:
            n_nodes = np.asarray(nodes).shape[0]
            rows = self.modes.shape[0]
            if rows % max(n_nodes, 1) != 0:
                gaps.append(
                    f"modes has {rows} rows against {n_nodes} nodes, which is "
                    "not a whole number of DOF per node -- the rows cannot be "
                    "mapped back onto the geometry"
                )
        return gaps

    def omega(self) -> np.ndarray:
        """Circular natural frequencies [rad/s]."""
        return 2.0 * np.pi * self.frequencies


class ModalSolver(ABC):
    """Base class for anything that turns design variables into a ModalResult."""

    #: Human-readable backend name, used in provenance metadata.
    name: str = "abstract"

    @abstractmethod
    def solve(self, params: dict[str, float], n_modes: int = 10) -> ModalResult:
        """Run a modal analysis for one design point."""

    def __call__(self, params: dict[str, float], n_modes: int = 10) -> ModalResult:
        return self.solve(params, n_modes=n_modes)
