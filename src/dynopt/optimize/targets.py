r"""What the surrogate actually regresses.

Turning an eigensolution into a fixed-length, continuous regression target is
the most consequential choice in the pipeline, and there is more than one
defensible answer. This module makes the choice explicit and swappable instead
of a string switch buried in the driver.

The tension is generality against physical meaning:

* **Mode families** carry meaning -- "the bounce mode" is a thing an engineer
  can act on -- but need a payload and a classification origin, so they are a
  property of the part rather than of the pipeline.
* **Sorted frequencies** need nothing at all. The :math:`k`-th smallest
  eigenvalue of a symmetric pencil is a *continuous* function of the design
  parameters: mode crossings kink it, they do not break it. It carries no
  physical label, but a GP does not need one.
* **Order statistics** of the objective -- lowest frequency, resonance risk,
  worst separation -- are likewise continuous and label-free, and are exactly
  the quantities :meth:`ResonanceObjective.__call__` consumes.

Which is best is a measurement, not an opinion; see
``examples/07_target_comparison.py``.

Re-targeting is cheap. ``_sample_dataset`` returns the solved
:class:`ModalResult` objects, so switching target type -- or changing the
excitation bands -- costs a re-extract, never a re-solve. Do not re-run a DOE
to try a different target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..dynamics.modal import (
    ModeFamilySpec,
    cluster_frequencies,
    cluster_modes,
    family_frequencies,
    track_clusters,
)
from ..fea.base import ModalResult


class TargetSet(ABC):
    """A fixed-length regression target extracted from an eigensolution."""

    #: Column names, in the order :meth:`extract` returns them.
    columns: tuple[str, ...] = ()
    #: Per-column surrogate transform: ``"log"``, ``"log1p"`` or ``"linear"``.
    transforms: tuple[str, ...] = ()
    #: True when this target needs the part to declare mode families.
    needs_families: bool = False

    def reset(self) -> None:
        """Clear any cross-sample state before a new DOE (see :class:`ClusterTargets`)."""

    @abstractmethod
    def extract(self, res: ModalResult, objective=None) -> list[float]:
        """One row of targets for a solved design. Missing values are NaN."""

    @abstractmethod
    def objective_from(self, row, mass: float, objective) -> float:
        """Rebuild the scalar objective from a *predicted* row."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {list(self.columns)}>"


def _origin_for(res: ModalResult):
    part = res.meta.get("part")
    if part is not None:
        return part.classification_origin(res)
    nodes = res.meta.get("nodes")
    if nodes is None:
        return None
    payload = res.meta.get("payload_nodes")
    nodes = np.asarray(nodes, dtype=float)
    return nodes[list(payload)].mean(axis=0) if payload else nodes.mean(axis=0)


class FamilyTargets(TargetSet):
    """Lowest frequency of each physical mode family.

    Immune to eigenvalue reordering and to the arbitrary basis inside degenerate
    eigenspaces, and the columns mean something. The cost is that the part must
    declare :class:`ModeFamilySpec` and a classification origin.
    """

    needs_families = True

    def __init__(self, spec: ModeFamilySpec, min_purity: float = 0.35):
        self.spec = spec
        self.min_purity = min_purity
        self.columns = tuple(spec.columns)
        self.transforms = ("log",) * len(self.columns)

    def extract(self, res: ModalResult, objective=None) -> list[float]:
        fam = family_frequencies(
            res, _origin_for(res), self.spec, min_purity=self.min_purity
        )
        return [fam[c] for c in self.columns]

    def objective_from(self, row, mass: float, objective) -> float:
        f = np.asarray(row, dtype=float)
        f = f[np.isfinite(f)]
        return objective(f, mass)


class RawTargets(TargetSet):
    r"""The ``k`` lowest natural frequencies, in ascending order.

    Needs nothing from the part -- no payload, no families, no classification.

    Ordered eigenvalues are continuous in the design parameters, so this is not
    the pathological choice it is often assumed to be: a crossing puts a kink in
    :math:`f_{(k)}(x)`, not a jump. The one real discontinuity is at the top of
    the window, where the :math:`(k{+}1)`-th mode can drop below the
    :math:`k`-th and shift the list; request a couple more modes than you
    regress to keep that away from the columns you use.
    """

    def __init__(self, k: int = 6):
        self.k = int(k)
        self.columns = tuple(f"f_{i+1}" for i in range(self.k))
        self.transforms = ("log",) * self.k

    def extract(self, res: ModalResult, objective=None) -> list[float]:
        row = [np.nan] * self.k
        f = np.sort(np.asarray(res.frequencies, dtype=float))
        row[: min(f.size, self.k)] = f[: self.k]
        return row

    def objective_from(self, row, mass: float, objective) -> float:
        f = np.asarray(row, dtype=float)
        f = f[np.isfinite(f)]
        return objective(f, mass)


class ScalarTargets(TargetSet):
    r"""The three order statistics the objective actually consumes.

    ``f_min`` (lowest natural frequency), ``risk`` (excitation-weighted
    resonance risk) and ``min_sep`` (worst normalised separation from any
    excitation band). All three are continuous under mode crossing -- each is a
    min or a sum over modes, so no mode needs an identity -- and none requires a
    payload or a classification.

    Three columns instead of one per mode, which is the point: the surrogate
    learns the decision surface rather than the spectrum.

    Two of the three depend on the objective's bands, so changing bands means
    re-extracting. That is a re-extract from stored results, not a re-solve.
    """

    def __init__(self) -> None:
        self.columns = ("f_min", "risk", "min_sep")
        # risk is >= 0 and is exactly 0 for the designs you want; min_sep goes
        # negative inside a band. Neither survives a log.
        self.transforms = ("log", "log1p", "linear")

    def extract(self, res: ModalResult, objective=None) -> list[float]:
        if objective is None:
            raise ValueError("ScalarTargets.extract needs the objective")
        f = np.asarray(res.frequencies, dtype=float)
        f = f[np.isfinite(f)]
        if f.size == 0:
            return [np.nan] * 3
        return [float(f.min()), float(objective.risk(f)), float(objective.min_separation(f))]

    def objective_from(self, row, mass: float, objective) -> float:
        """Mirror of :meth:`ResonanceObjective.__call__`, from predicted parts."""
        f_min, risk, sep = (float(v) for v in row[:3])
        if not np.isfinite(f_min):
            return float("inf")

        j_mass = objective.w_mass * mass / max(objective.mass_ref, 1e-12)
        j_res = objective.w_resonance * max(risk, 0.0)
        j_sep = -objective.w_separation * np.tanh(sep / max(objective.margin, 1e-9))

        j_floor = 0.0
        if objective.f_min is not None and f_min < objective.f_min:
            j_floor = objective.f_min_penalty * (1.0 - f_min / objective.f_min) ** 2
        return float(j_mass + j_res + j_sep + j_floor)


class ClusterTargets(TargetSet):
    """Near-degenerate modes grouped and matched to a reference by subspace MAC.

    Stateful: the first design it sees becomes the reference. Correct over a
    narrow design space or a continuation sweep, and it degrades once the shapes
    drift far from that reference -- kept for comparison, not recommended.
    """

    def __init__(self, k: int = 6, rtol: float = 0.02):
        self.k = int(k)
        self.rtol = rtol
        self.columns = tuple(f"cluster_{i+1}" for i in range(self.k))
        self.transforms = ("log",) * self.k
        self._phi_ref = None
        self._clusters_ref = None

    def reset(self) -> None:
        self._phi_ref = None
        self._clusters_ref = None

    def extract(self, res: ModalResult, objective=None) -> list[float]:
        row = [np.nan] * self.k
        if res.modes is None:
            return row
        if self._phi_ref is None:
            self._phi_ref = res.modes
            self._clusters_ref = cluster_modes(res.frequencies, rtol=self.rtol)
            fc = cluster_frequencies(res.frequencies, self._clusters_ref)
            row[: min(fc.size, self.k)] = fc[: self.k]
            return row
        cnew = cluster_modes(res.frequencies, rtol=self.rtol)
        idx = track_clusters(self._phi_ref, self._clusters_ref, res.modes, cnew)
        fc = cluster_frequencies(res.frequencies, cnew)
        for j, jj in enumerate(idx[: self.k]):
            if jj >= 0:
                row[j] = fc[jj]
        return row

    def objective_from(self, row, mass: float, objective) -> float:
        f = np.asarray(row, dtype=float)
        f = f[np.isfinite(f)]
        return objective(f, mass)


def make_target_set(spec_or_name, n_modes: int = 6, part=None) -> TargetSet:
    """Resolve a :class:`TargetSet`, a name, or ``None`` into a target set.

    Names map to ``"family"``, ``"raw"``, ``"scalars"`` and ``"cluster"`` so
    existing callers keep working.
    """
    if isinstance(spec_or_name, TargetSet):
        return spec_or_name
    name = spec_or_name or "family"
    if name == "raw":
        return RawTargets(n_modes)
    if name in ("scalars", "scalar", "risk"):
        return ScalarTargets()
    if name == "cluster":
        return ClusterTargets(n_modes)
    if name == "family":
        families = getattr(part, "mode_families", None) if part is not None else None
        if families is None:
            raise ValueError(
                "targets='family' needs a part declaring mode_families; pass a "
                "TargetSet explicitly, or use 'raw'/'scalars' which need neither"
            )
        return FamilyTargets(families)
    raise ValueError(f"unknown target set {name!r}")


def extract_targets(results, target_set: TargetSet, objective=None) -> np.ndarray:
    """Re-extract targets from already-solved results, with no re-solving.

    The expensive thing is the eigensolution, not the extraction. Comparing
    target sets, or changing the excitation bands, should therefore cost
    seconds, not another DOE. Rows for failed designs come back as NaN.
    """
    target_set.reset()
    n, k = len(results), len(target_set.columns)
    F = np.full((n, k), np.nan)
    for i, res in enumerate(results):
        if res is None:
            continue
        row = target_set.extract(res, objective)
        F[i, : len(row)] = row
    return F
