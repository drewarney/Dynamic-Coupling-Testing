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
