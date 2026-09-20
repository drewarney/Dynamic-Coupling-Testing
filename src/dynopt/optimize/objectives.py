r"""Resonance-avoidance objectives.

The formulation
---------------
Given excitation bands :math:`\mathcal{B}_k = [f_k^{lo}, f_k^{hi}]` (e.g. 1X and
4X of a 1,000-6,000 RPM motor) and predicted natural frequencies :math:`f_j`,
minimise

.. math::
    J(x) = w_m \frac{m(x)}{m_0}
         + w_r \sum_j \alpha_j\, R_j(x)
         - w_s\, \min_j d_j(x)

where:

* :math:`R_j` is a **smooth resonance risk** for mode :math:`j`,
* :math:`\alpha_j` is the mode's effective-mass fraction in the excitation
  direction -- a mode the excitation cannot reach gets no weight,
* :math:`d_j` is the normalised separation from the nearest excitation band.

Why a *smooth* penalty
----------------------
The naive objective "maximise :math:`\min_j |f_j - f_{exc}|`" is non-smooth and
has a pathological failure mode: pushing a mode up through an excitation band is
penalised at every intermediate step, so a gradient optimiser will never cross,
even when the far side is much better. The Gaussian risk kernel

.. math:: R_j = \exp\!\left(-\tfrac{1}{2}\left(\frac{f_j - f_c}{\beta f_c}\right)^2\right)

is smooth, bounded, and its gradient vanishes once a mode is well clear, so the
optimiser stops spending effort on modes that are already safe and moves on to
minimising mass. This is the single most important design decision in the
whole optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExcitationBand:
    """A forcing source to avoid.

    ``from_rpm`` builds the band from a speed range and an engine order: a motor
    running 1,000-6,000 RPM excites 1X at 16.7-100 Hz and 4X (4-blade prop) at
    66.7-400 Hz.
    """

    f_lo: float
    f_hi: float
    order: float = 1.0
    weight: float = 1.0
    label: str = ""

    @classmethod
    def from_rpm(
        cls, rpm_lo: float, rpm_hi: float, order: float = 1.0,
        weight: float = 1.0, label: str = "",
    ) -> "ExcitationBand":
        return cls(
            f_lo=rpm_lo * order / 60.0,
            f_hi=rpm_hi * order / 60.0,
            order=order,
            weight=weight,
            label=label or f"{order:g}X {rpm_lo:g}-{rpm_hi:g} RPM",
        )

    @property
    def center(self) -> float:
        return 0.5 * (self.f_lo + self.f_hi)

    def contains(self, f: float) -> bool:
        return self.f_lo <= f <= self.f_hi

    def separation(self, f: float) -> float:
        """Normalised distance from the band edge; negative means inside."""
        if f < self.f_lo:
            return (self.f_lo - f) / self.f_lo
        if f > self.f_hi:
            return (f - self.f_hi) / self.f_hi
        return -min(f - self.f_lo, self.f_hi - f) / max(self.center, 1e-9)


@dataclass
class ResonanceObjective:
    """Weighted mass + resonance-risk objective, evaluated on predicted modes.

    Parameters
    ----------
    bands:
        Excitation bands to avoid.
    margin:
        Required fractional separation :math:`\\beta`. 0.20 means every mode
        must sit at least 20% away in frequency -- a standard aerospace
        separation rule; automotive NVH practice often uses 0.10-0.15.
    w_mass, w_resonance, w_separation:
        Objective weights. ``w_mass`` is applied to mass normalised by
        ``mass_ref``, so the weights are directly comparable.
    mode_weights:
        Per-mode importance, normally the effective-mass fraction in the
        excitation direction. Defaults to all-ones.
    """

    bands: list[ExcitationBand]
    margin: float = 0.20
    w_mass: float = 1.0
    w_resonance: float = 10.0
    w_separation: float = 0.5
    mass_ref: float = 1.0
    mode_weights: np.ndarray | None = None
    f_min: float | None = None
    f_min_penalty: float = 50.0
    history: list[dict] = field(default_factory=list)

    def risk(self, freqs: np.ndarray) -> float:
        r"""Smooth aggregate resonance risk :math:`\sum_j \alpha_j R_j \in [0,\infty)`."""
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        w = (
            np.ones_like(f)
            if self.mode_weights is None
            else np.broadcast_to(np.asarray(self.mode_weights, dtype=float), f.shape)
        )
        total = 0.0
        for band in self.bands:
            fc = band.center
            half = max(0.5 * (band.f_hi - band.f_lo), 1e-9)
            # Effective sigma covers the band half-width plus the margin.
            sigma = max(self.margin * fc, half)
            total += band.weight * float(
                np.sum(w * np.exp(-0.5 * ((f - fc) / sigma) ** 2))
            )
        return total

    def min_separation(self, freqs: np.ndarray) -> float:
        """Smallest normalised separation over all (mode, band) pairs."""
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        return float(min(b.separation(fi) for fi in f for b in self.bands))

    def constraint_violation(self, freqs: np.ndarray) -> float:
        """Total shortfall against the ``margin`` rule; 0.0 means feasible."""
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        v = 0.0
        for fi in f:
            for b in self.bands:
                s = b.separation(fi)
                if s < self.margin:
                    v += (self.margin - s) * b.weight
        if self.f_min is not None and f.size and f.min() < self.f_min:
            v += (self.f_min - f.min()) / self.f_min
        return float(v)

    def __call__(self, freqs: np.ndarray, mass: float = 0.0, record: bool = False) -> float:
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        # A nan mass produces a nan objective, and an optimiser comparing nan to
        # nan neither fails nor converges -- it just returns whatever it started
        # with. The 3D backend reported nan mass for its whole life, so
        # `dynopt optimize --backend calculix` did exactly that. Refuse it here,
        # where the cause is still visible.
        if not np.isfinite(mass):
            raise ValueError(
                f"mass is {mass!r}: the objective divides by it, so every value "
                "in the run would be nan. The solver that produced this result "
                "does not report mass -- see ModalResult.missing_contract()."
            )
        j_mass = self.w_mass * mass / max(self.mass_ref, 1e-12)
        j_res = self.w_resonance * self.risk(f)
        sep = self.min_separation(f)
        j_sep = -self.w_separation * np.tanh(sep / max(self.margin, 1e-9))

        j_floor = 0.0
        if self.f_min is not None and f.size and f.min() < self.f_min:
            j_floor = self.f_min_penalty * (1.0 - f.min() / self.f_min) ** 2

        total = float(j_mass + j_res + j_sep + j_floor)
        if record:
            self.history.append(
                {
                    "J": total, "mass": float(mass), "f1": float(f.min()) if f.size else np.nan,
                    "risk": float(self.risk(f)), "min_sep": sep,
                    "violation": self.constraint_violation(f),
                }
            )
        return total


def rpm_bands(
    rpm_lo: float, rpm_hi: float, orders=(1.0, 2.0, 4.0), weights=None
) -> list[ExcitationBand]:
    """Convenience: build one band per engine order over an RPM range.

    Weights default to :math:`1/o` -- higher orders carry less energy in most
    rotating machinery, so a 4X crossing matters less than a 1X crossing.
    """
    w = weights if weights is not None else [1.0 / o for o in orders]
    return [
        ExcitationBand.from_rpm(rpm_lo, rpm_hi, order=o, weight=wi)
        for o, wi in zip(orders, w)
    ]
