"""The objective every optimisation run is scored against, which had no tests.

`ResonanceObjective` decides what "better" means for the whole pipeline. A
defect here does not crash anything -- it silently changes which design the
optimiser returns, and every number downstream inherits it.
"""

import numpy as np
import pytest

from dynopt.optimize.objectives import ExcitationBand, ResonanceObjective, rpm_bands


# ---------------------------------------------------------------------------
# ExcitationBand
# ---------------------------------------------------------------------------
def test_from_rpm_converts_speed_and_order_to_frequency():
    """1,000-6,000 RPM at 4X is 66.7-400 Hz."""
    b = ExcitationBand.from_rpm(1000, 6000, order=4)
    assert b.f_lo == pytest.approx(1000 * 4 / 60)
    assert b.f_hi == pytest.approx(6000 * 4 / 60)
    assert b.center == pytest.approx(0.5 * (b.f_lo + b.f_hi))


def test_separation_is_continuous_across_both_band_edges():
    """A kink or jump at an edge would put a false optimum on the boundary."""
    b = ExcitationBand(100.0, 400.0)
    eps = 1e-7
    assert b.separation(100.0 - eps) == pytest.approx(0.0, abs=1e-8)
    assert b.separation(100.0 + eps) == pytest.approx(0.0, abs=1e-8)
    assert b.separation(400.0 - eps) == pytest.approx(0.0, abs=1e-8)
    assert b.separation(400.0 + eps) == pytest.approx(0.0, abs=1e-8)


def test_separation_is_negative_inside_and_positive_outside():
    b = ExcitationBand(100.0, 400.0)
    assert b.separation(250.0) < 0.0
    assert b.separation(50.0) > 0.0
    assert b.separation(800.0) > 0.0
    assert b.contains(250.0) and not b.contains(50.0)


def test_separation_is_measured_against_the_near_edge():
    """Fractional separation from the edge you are near, not from the centre."""
    b = ExcitationBand(100.0, 400.0)
    assert b.separation(50.0) == pytest.approx((100.0 - 50.0) / 100.0)
    assert b.separation(500.0) == pytest.approx((500.0 - 400.0) / 400.0)


# ---------------------------------------------------------------------------
# The risk kernel: smoothness is the whole design rationale
# ---------------------------------------------------------------------------
def _obj(**kw):
    return ResonanceObjective(bands=[ExcitationBand(100.0, 400.0)], **kw)


def test_risk_peaks_inside_the_band_and_decays_outside():
    o = _obj()
    inside = o.risk(np.array([250.0]))
    near = o.risk(np.array([600.0]))
    far = o.risk(np.array([5000.0]))
    assert inside > near > far
    assert far == pytest.approx(0.0, abs=1e-6)


def test_risk_is_smooth_through_the_band():
    """The reason for a Gaussian kernel: an optimiser must be able to cross.

    With a non-smooth penalty, pushing a mode up through a band is penalised at
    every intermediate step and a gradient method never crosses, even when the
    far side is much better.
    """
    o = _obj()
    f = np.linspace(50.0, 900.0, 400)
    r = np.array([o.risk(np.array([fi])) for fi in f])
    d2 = np.diff(r, 2)
    assert np.all(np.abs(d2) < 1e-2), "risk kernel has a kink"
    assert r.argmax() == np.abs(f - 250.0).argmin()


def test_mode_weights_silence_unreachable_modes():
    o = _obj()
    both = o.risk(np.array([250.0, 250.0]))
    o.mode_weights = np.array([1.0, 0.0])
    weighted = o.risk(np.array([250.0, 250.0]))
    assert weighted == pytest.approx(both / 2)


# ---------------------------------------------------------------------------
# Feasibility and the scalar objective
# ---------------------------------------------------------------------------
def test_constraint_violation_is_zero_exactly_when_the_margin_is_met():
    o = _obj(margin=0.25)
    assert o.constraint_violation(np.array([600.0])) == 0.0   # 50% clear
    assert o.constraint_violation(np.array([250.0])) > 0.0    # inside
    assert o.constraint_violation(np.array([440.0])) > 0.0    # 10% clear


def test_f_min_floor_penalises_designs_below_it():
    o = _obj(f_min=500.0)
    assert o.constraint_violation(np.array([600.0])) == 0.0
    assert o.constraint_violation(np.array([400.0])) > 0.0


def test_objective_prefers_light_and_clear_over_heavy_and_resonant():
    o = _obj(margin=0.25, mass_ref=0.5)
    good = o(np.array([900.0]), mass=0.40)
    heavy = o(np.array([900.0]), mass=0.60)
    resonant = o(np.array([250.0]), mass=0.40)
    assert good < heavy
    assert good < resonant


def test_a_nan_mass_raises_rather_than_returning_nan():
    """An optimiser cannot tell a nan objective from a value; it just stalls."""
    o = _obj(mass_ref=0.5)
    with pytest.raises(ValueError, match="nan"):
        o(np.array([900.0]), mass=float("nan"))


def test_record_captures_what_the_report_prints():
    o = _obj(mass_ref=0.5)
    o(np.array([900.0, 1200.0]), mass=0.4, record=True)
    assert len(o.history) == 1
    row = o.history[0]
    assert set(row) >= {"J", "mass", "f1", "risk", "min_sep", "violation"}
    assert row["f1"] == pytest.approx(900.0)


def test_rpm_bands_weight_higher_orders_less():
    bands = rpm_bands(1000, 6000, orders=(1.0, 2.0, 4.0))
    assert [b.order for b in bands] == [1.0, 2.0, 4.0]
    assert bands[0].weight > bands[1].weight > bands[2].weight
