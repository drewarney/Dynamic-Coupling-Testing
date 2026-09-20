"""Tests for the objective function, surrogate and optimisation drivers."""

import numpy as np
import pytest

from dynopt.doe import lhs_sample, sobol_sample
from dynopt.fea.reference import ReferenceFrameSolver
from dynopt.geometry import MOTOR_MOUNT, MOTOR_MOUNT_SPACE
from dynopt.optimize import (
    ExcitationBand,
    ResonanceObjective,
    evaluate_baseline,
    optimize_adaptive,
    rpm_bands,
)
from dynopt.optimize.driver import _sample_dataset
from dynopt.surrogate import FrequencySurrogate, propose_batch

SPACE = MOTOR_MOUNT_SPACE


@pytest.fixture(scope="module")
def solver():
    return ReferenceFrameSolver(MOTOR_MOUNT)


# --------------------------------------------------------------------------
# Excitation bands and objective
# --------------------------------------------------------------------------
def test_rpm_to_frequency_conversion():
    """6,000 RPM at 1X is 100 Hz; at 4X it is 400 Hz."""
    b = ExcitationBand.from_rpm(1000, 6000, order=1.0)
    assert b.f_lo == pytest.approx(1000 / 60)
    assert b.f_hi == pytest.approx(100.0)
    b4 = ExcitationBand.from_rpm(1000, 6000, order=4.0)
    assert b4.f_hi == pytest.approx(400.0)


def test_separation_sign_convention():
    """Separation is negative inside the band, positive outside."""
    b = ExcitationBand(100.0, 200.0)
    assert b.separation(150.0) < 0
    assert b.separation(250.0) > 0
    assert b.separation(50.0) > 0
    assert b.contains(150.0) and not b.contains(250.0)


def test_risk_is_smooth_and_decays():
    """Risk must fall monotonically as a mode moves away from the band."""
    obj = ResonanceObjective(bands=[ExcitationBand(90.0, 110.0)], margin=0.2)
    risks = [obj.risk(np.array([f])) for f in (100.0, 150.0, 300.0, 1000.0)]
    assert all(risks[i] > risks[i + 1] for i in range(len(risks) - 1))
    assert risks[-1] < 1e-6


def test_objective_prefers_clear_of_band():
    """A design clear of the excitation must score better than one inside it."""
    obj = ResonanceObjective(bands=rpm_bands(1000, 6000), margin=0.25, mass_ref=0.5)
    inside = obj(np.array([80.0, 300.0]), mass=0.5)
    outside = obj(np.array([800.0, 1600.0]), mass=0.5)
    assert outside < inside


def test_objective_trades_mass_against_resonance():
    """With equal resonance risk, the lighter design must win."""
    obj = ResonanceObjective(bands=rpm_bands(1000, 6000), margin=0.25, mass_ref=0.5)
    f = np.array([900.0, 1500.0])
    assert obj(f, mass=0.3) < obj(f, mass=0.6)


def test_constraint_violation_zero_when_feasible():
    obj = ResonanceObjective(bands=[ExcitationBand(90.0, 110.0)], margin=0.2)
    assert obj.constraint_violation(np.array([1000.0])) == 0.0
    assert obj.constraint_violation(np.array([100.0])) > 0.0


# --------------------------------------------------------------------------
# DOE
# --------------------------------------------------------------------------
def test_doe_respects_bounds():
    for sampler in (sobol_sample, lhs_sample):
        X = sampler(SPACE, 32, seed=0)
        assert X.shape == (32, len(SPACE))
        for j, v in enumerate(SPACE.variables):
            assert X[:, j].min() >= v.lower - 1e-12
            assert X[:, j].max() <= v.upper + 1e-12


def test_sobol_beats_random_coverage():
    """Low-discrepancy sampling must cover better than uniform random."""
    from dynopt.doe import discrepancy

    rng = np.random.default_rng(0)
    lo = np.array([v.lower for v in SPACE.variables])
    hi = np.array([v.upper for v in SPACE.variables])
    rand = lo + rng.random((64, len(SPACE))) * (hi - lo)
    assert discrepancy(SPACE, sobol_sample(SPACE, 64, seed=0)) < discrepancy(SPACE, rand)


# --------------------------------------------------------------------------
# Surrogate
# --------------------------------------------------------------------------
def test_surrogate_interpolates_training_data(solver):
    """A noise-free GP must reproduce its own training points closely."""
    X = sobol_sample(SPACE, 24, seed=0)
    F, mass, _, _ = _sample_dataset(solver, SPACE, X, 10)
    ok = np.isfinite(mass)
    sur = FrequencySurrogate(SPACE, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])
    P = sur.predict(X[ok])
    m = np.isfinite(F[ok]) & np.isfinite(P)
    assert np.median(np.abs(P[m] - F[ok][m]) / F[ok][m]) < 0.02


def test_surrogate_accuracy_on_held_out_data(solver):
    """Headline accuracy claim: <8% MAPE per family from 64 training runs.

    The production numbers (~0.5-2% from 128 runs) are better; this bound is
    loose so the test is not flaky, while still failing loudly if the mode
    classification or tracking regresses.
    """
    X = sobol_sample(SPACE, 64, seed=1)
    F, mass, _, cols = _sample_dataset(solver, SPACE, X, 10)
    ok = np.isfinite(mass)
    sur = FrequencySurrogate(SPACE, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])

    Xt = sobol_sample(SPACE, 48, seed=99)
    Ft, mt, _, _ = _sample_dataset(solver, SPACE, Xt, 10)
    P = sur.predict(Xt)
    for j, name in enumerate(cols):
        m = np.isfinite(Ft[:, j]) & np.isfinite(P[:, j])
        if m.sum() < 10:
            continue
        mape = np.mean(np.abs(P[m, j] - Ft[m, j]) / Ft[m, j])
        assert mape < 0.08, f"{name}: MAPE {mape:.1%}"


def test_surrogate_is_orders_of_magnitude_faster(solver):
    """The whole point: prediction must be far cheaper than a solve."""
    import time

    X = sobol_sample(SPACE, 24, seed=0)
    F, mass, _, _ = _sample_dataset(solver, SPACE, X, 10)
    ok = np.isfinite(mass)
    sur = FrequencySurrogate(SPACE, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])

    Xt = sobol_sample(SPACE, 100, seed=5)
    t0 = time.perf_counter()
    sur.predict(Xt)
    t_sur = (time.perf_counter() - t0) / 100

    t0 = time.perf_counter()
    solver.solve(SPACE.to_dict(Xt[0]), n_modes=10)
    t_fea = time.perf_counter() - t0

    assert t_sur < t_fea / 50, f"surrogate {t_sur*1e3:.3f} ms vs FEA {t_fea*1e3:.1f} ms"


def test_sensitivity_ranking_is_normalised(solver):
    X = sobol_sample(SPACE, 32, seed=0)
    F, mass, _, _ = _sample_dataset(solver, SPACE, X, 10)
    ok = np.isfinite(mass)
    sur = FrequencySurrogate(SPACE, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])
    rank = sur.sensitivity_ranking(1)
    assert len(rank) == len(SPACE)
    assert sum(v for _, v in rank) == pytest.approx(1.0, abs=1e-6)
    assert rank[0][1] >= rank[-1][1]


def test_active_learning_batch_is_diverse(solver):
    """Proposed points must be spread out, not piled on one hot spot."""
    X = sobol_sample(SPACE, 24, seed=0)
    F, mass, _, _ = _sample_dataset(solver, SPACE, X, 10)
    ok = np.isfinite(mass)
    sur = FrequencySurrogate(SPACE, n_modes=F.shape[1]).fit(X[ok], F[ok], mass[ok])

    batch = propose_batch(sur, SPACE, n_batch=6, n_candidates=512, min_distance=0.1)
    assert batch.shape[0] >= 2
    lo = np.array([v.lower for v in SPACE.variables])
    hi = np.array([v.upper for v in SPACE.variables])
    nb = (batch - lo) / (hi - lo)
    for i in range(len(nb)):
        for j in range(i + 1, len(nb)):
            assert np.linalg.norm(nb[i] - nb[j]) >= 0.1 - 1e-9


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------
def test_adaptive_optimisation_improves_on_baseline(solver):
    """The headline claim: the optimiser must beat the starting design.

    Asserted on the FEA-verified objective, never on a surrogate prediction.
    """
    obj = ResonanceObjective(
        bands=rpm_bands(1000, 6000), margin=0.25, mass_ref=0.55,
        w_mass=1.0, w_resonance=10.0, f_min=450.0,
    )
    base = evaluate_baseline(solver, SPACE, obj)
    res = optimize_adaptive(
        solver, SPACE, obj, n_init=24, n_rounds=3, n_per_round=3,
        n_modes=10, verbose=False,
    )

    assert res.objective < base["objective"]
    assert obj.constraint_violation(res.frequencies) <= base["violation"]
    assert np.all(np.isfinite(res.frequencies))
    for name, val in res.params.items():
        v = SPACE[name]
        assert v.lower - 1e-9 <= val <= v.upper + 1e-9


def test_optimum_clears_the_excitation_bands(solver):
    """A feasible result must genuinely sit outside every excitation band."""
    bands = rpm_bands(1000, 6000, orders=(1.0, 2.0, 4.0))
    obj = ResonanceObjective(
        bands=bands, margin=0.25, mass_ref=0.55, w_mass=1.0,
        w_resonance=20.0, f_min=450.0,
    )
    res = optimize_adaptive(
        solver, SPACE, obj, n_init=24, n_rounds=3, n_per_round=3,
        n_modes=10, verbose=False,
    )
    for f in res.frequencies[:4]:
        for b in bands:
            assert not b.contains(f), f"mode {f:.1f} Hz sits inside {b.label}"
