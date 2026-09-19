"""FEA backends.  Heavy/optional backends are imported lazily by name."""

from .base import ModalResult, ModalSolver
from .reference import (
    MATERIALS,
    FrameModel,
    Material,
    ReferenceFrameSolver,
    Section,
    assemble,
    modal_solve,
)

__all__ = [
    "ModalResult", "ModalSolver", "ReferenceFrameSolver", "modal_solve",
    "FrameModel", "Section", "Material", "MATERIALS", "assemble",
]


def get_solver(name: str, *args, **kwargs):
    """Factory so the driver never imports a backend it will not use."""
    if name in ("reference", "reference-frame"):
        return ReferenceFrameSolver(*args, **kwargs)
    if name in ("calculix", "ccx", "gmsh-calculix"):
        from .calculix import CalculiXSolver

        return CalculiXSolver(*args, **kwargs)
    if name in ("ansys", "mapdl", "pyansys"):
        from .pyansys import MapdlSolver

        return MapdlSolver(*args, **kwargs)
    raise ValueError(f"unknown solver backend {name!r}")
