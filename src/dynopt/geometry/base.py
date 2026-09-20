"""Design-variable plumbing shared by every geometry backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DesignVariable:
    """One continuous CAD parameter with bounds, in SI units."""

    name: str
    lower: float
    upper: float
    default: float
    units: str = "m"
    description: str = ""

    def clip(self, x: float) -> float:
        return float(np.clip(x, self.lower, self.upper))

    def normalize(self, x: float) -> float:
        """Map to [0, 1] -- surrogates and optimisers want unit-scaled inputs."""
        return (x - self.lower) / (self.upper - self.lower)

    def denormalize(self, u: float) -> float:
        return self.lower + u * (self.upper - self.lower)


class DesignSpace:
    """An ordered collection of design variables with vector<->dict conversion."""

    def __init__(self, variables: list[DesignVariable]):
        self.variables = list(variables)
        self.names = [v.name for v in self.variables]
        self._by_name = {v.name: v for v in self.variables}

    def __len__(self) -> int:
        return len(self.variables)

    def __getitem__(self, key):
        return self._by_name[key] if isinstance(key, str) else self.variables[key]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """``[(lo, hi), ...]`` in the layout scipy.optimize expects."""
        return [(v.lower, v.upper) for v in self.variables]

    @property
    def defaults(self) -> np.ndarray:
        return np.array([v.default for v in self.variables])

    def to_dict(self, x: np.ndarray) -> dict[str, float]:
        return {v.name: float(xi) for v, xi in zip(self.variables, np.ravel(x))}

    def to_vector(self, params: dict[str, float]) -> np.ndarray:
        return np.array([float(params[v.name]) for v in self.variables])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.array([v.normalize(xi) for v, xi in zip(self.variables, np.ravel(x))])

    def denormalize(self, u: np.ndarray) -> np.ndarray:
        return np.array([v.denormalize(ui) for v, ui in zip(self.variables, np.ravel(u))])

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.array([v.clip(xi) for v, xi in zip(self.variables, np.ravel(x))])


def material_hints(material) -> dict[str, float]:
    """``E``/``nu``/``rho`` in the mm-tonne-s units the CAD tiers work in.

    :mod:`dynopt.fea.reference` stores materials in SI; CadQuery exports
    millimetres, and a 3D deck written in SI against mm geometry is off by nine
    orders of magnitude in density. Convert in exactly one place.
    """
    from ..fea.reference import MATERIALS

    mat = MATERIALS[material] if isinstance(material, str) else material
    return {"E": mat.E / 1e6, "nu": mat.nu, "rho": mat.rho * 1e-12}


class ParametricModel(ABC):
    """Contract for a CAD backend: parameters in, solid + STEP file out."""

    @property
    @abstractmethod
    def design_space(self) -> DesignSpace: ...

    @abstractmethod
    def build(self, params: dict[str, float]):
        """Return a backend-native solid/assembly."""

    @abstractmethod
    def export_step(self, params: dict[str, float], path: str) -> str:
        """Write a STEP file for this design point; return the path."""


class PartSpec(ABC):
    """Everything the pipeline needs to know about one part.

    Adding a part means writing one of these. The core -- FEA backends, mode
    classification, DOE, surrogate, optimiser -- holds no knowledge of any
    specific geometry, so nothing outside this class changes.

    Only :attr:`design_space` and at least one builder are required. A part that
    supplies :meth:`build_frame` gets the fast beam tier; one that supplies
    :meth:`build_cad` gets the 3D tier; supplying both enables cross-solver
    validation and the tiered workflow the benchmark uses.

    :attr:`mode_families` is optional but strongly recommended: without it the
    surrogate falls back to tracking modes by shape, which degrades over a wide
    design space (see :mod:`dynopt.dynamics.modal`).
    """

    #: Short identifier, used in reports and the CLI registry.
    name: str = "part"

    # -- required ------------------------------------------------------------
    @property
    @abstractmethod
    def design_space(self) -> DesignSpace:
        """The parameters the optimiser may vary, with bounds."""

    # -- tier 0: beam / frame idealisation -----------------------------------
    def build_frame(self, params: dict[str, float]):
        """Return a :class:`dynopt.fea.reference.FrameModel` for ``params``.

        Optional. Implement it when the part is frame-like (struts, arms,
        brackets built from slender members); a beam idealisation runs in
        milliseconds and makes large DOEs affordable. Do not implement it for
        bulky solids -- a beam model of those is not merely approximate, it is
        wrong, and a validity envelope cannot rescue it.
        """
        raise NotImplementedError(f"{type(self).__name__} has no frame tier")

    @property
    def has_frame(self) -> bool:
        return type(self).build_frame is not PartSpec.build_frame

    # -- tier 1: solid CAD ---------------------------------------------------
    def build_cad(self, params: dict[str, float]):
        """Return a backend-native solid (e.g. a CadQuery Workplane)."""
        raise NotImplementedError(f"{type(self).__name__} has no CAD tier")

    def export_step(self, params: dict[str, float], path: str) -> str:
        """Write a STEP file for this design point; return the path."""
        import cadquery as cq

        cq.exporters.export(self.build_cad(params), path)
        return path

    @property
    def has_cad(self) -> bool:
        return type(self).build_cad is not PartSpec.build_cad

    # -- mode classification -------------------------------------------------
    @property
    def mode_families(self):
        """A :class:`dynopt.dynamics.modal.ModeFamilySpec`, or ``None``.

        Declares which rigid-body motions of :meth:`classification_origin`
        constitute a physical mode family for this part, and how many branches
        each family has.
        """
        return None

    def classification_origin(self, result) -> np.ndarray | None:
        """Point about which rigid-body motions are measured.

        Defaults to the centroid of any payload (point-mass) nodes the solver
        recorded, falling back to the model centroid. Override when the response
        of interest is somewhere else -- a sensor mount, a bearing seat, an
        optical bench.
        """
        nodes = result.meta.get("nodes")
        if nodes is None:
            return None
        payload = result.meta.get("payload_nodes")
        nodes = np.asarray(nodes, dtype=float)
        if payload:
            return nodes[list(payload)].mean(axis=0)
        return nodes.mean(axis=0)

    # -- tier 1 solver configuration -----------------------------------------
    def solver_hints(self, params: dict[str, float]) -> dict:
        """Geometry-dependent settings the 3D backend cannot infer itself.

        Boundary conditions are the part's business, not the solver's. Returning
        them here keeps a selector like "clamp the lowest face" from silently
        picking the wrong face when a design variable moves geometry around --
        a failure that solves happily and reports confident nonsense.

        Recognised keys: ``fix_selector``, ``fix_plane``, ``fix_tol``,
        ``payload_mass``, ``payload_radius``, ``min_feature``, ``E``, ``nu``,
        ``rho``.

        The material belongs here for the same reason the supports do: the beam
        tier reads it from the part, so if the 3D tier does not, the two tiers
        silently model different materials. A carbon arm solved as aluminium
        reports frequencies 31% low, and nothing in the pipeline objects --
        the discrepancy just looks like a fidelity effect and gets "corrected".
        Use :func:`material_hints` rather than writing the conversion out.

        ``min_feature`` is the smallest wall or bore in *this* design, in the
        model units the CAD exports (mm here). The 3D backend caps its element
        size by it, because a fixed element size is not portable across a design
        space: too coarse for the thinnest design in the box and the mesher
        rejects the part outright, losing it from the DOE.

        Hints are re-read for every design, so anything derived from ``params``
        must be computed here rather than cached.
        """
        return {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tiers = [t for t, ok in (("frame", self.has_frame), ("cad", self.has_cad)) if ok]
        return f"<{type(self).__name__} {self.name!r} tiers={tiers or ['none']}>"
