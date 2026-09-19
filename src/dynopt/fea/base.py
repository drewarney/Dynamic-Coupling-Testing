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
