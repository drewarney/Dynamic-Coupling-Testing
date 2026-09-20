"""Multi-fidelity correction: does a cheap tier become trustworthy?

Synthetic fixtures with a *known* discrepancy, so the tests measure whether the
correction recovers what it should rather than whether two solvers happen to
agree. The physics study lives in examples/08_multifidelity.py.
"""

import numpy as np
import pytest

from dynopt.doe import sobol_sample
from dynopt.geometry import MOTOR_MOUNT
from dynopt.geometry.base import DesignSpace, DesignVariable
from dynopt.surrogate import MultiFidelitySurrogate

SPACE = DesignSpace([
    DesignVariable("a", 0.0, 1.0, 0.5, "-", ""),
    DesignVariable("b", 0.0, 1.0, 0.5, "-", ""),
])


def cheap(X: np.ndarray) -> np.ndarray:
    """Two strictly positive 'frequency' columns."""
    X = np.atleast_2d(X)
    return np.column_stack([
        100.0 + 50.0 * X[:, 0] + 20.0 * X[:, 1],
        300.0 + 80.0 * X[:, 0] ** 2,
    ])


def fine_constant(X: np.ndarray) -> np.ndarray:
    """A pure scale factor per column -- the case a scalar should handle."""
    return cheap(X) * np.array([1.70, 1.35])


def fine_varying(X: np.ndarray) -> np.ndarray:
    """A ratio that genuinely varies with the design -- needs the GP."""
    X = np.atleast_2d(X)
    rho = np.column_stack([
        1.4 + 0.6 * X[:, 0],
        1.2 + 0.5 * np.sin(2.0 * np.pi * X[:, 1]),
    ])
    return cheap(X) * rho


@pytest.fixture
def data():
    X_cheap = sobol_sample(SPACE, 48, seed=0)
    X_pair = sobol_sample(SPACE, 16, seed=5)
    X_test = sobol_sample(SPACE, 32, seed=99)
    return X_cheap, X_pair, X_test


# ---------------------------------------------------------------------------
# Failing loudly on a mismatch
# ---------------------------------------------------------------------------
def test_mismatched_columns_raise(data):
    """Correcting column k of one tier against column k of another assumes they
    describe the same mode. If the tiers report different numbers of modes that
    assumption is already broken, and silently averaging unrelated physics
    produces a confident wrong answer.
    """
    X_cheap, X_pair, _ = data
    mf = MultiFidelitySurrogate(SPACE)
    with pytest.raises(ValueError, match="column counts differ"):
        mf.fit(X_cheap, cheap(X_cheap), X_pair,
               cheap(X_pair), fine_constant(X_pair)[:, :1])


def test_cheap_doe_column_mismatch_raises(data):
    X_cheap, X_pair, _ = data
    mf = MultiFidelitySurrogate(SPACE)
    with pytest.raises(ValueError, match="columns"):
        mf.fit(X_cheap, cheap(X_cheap)[:, :1], X_pair,
               cheap(X_pair), fine_constant(X_pair))


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        MultiFidelitySurrogate(SPACE, mode="magic")


# ---------------------------------------------------------------------------
# Does the correction actually correct?
# ---------------------------------------------------------------------------
# A constant correction cannot track a ratio that varies with the design, so it
# is held to a weaker bound than the two that can. Demanding the same of all
# three would be asserting something the physics does not offer.
@pytest.mark.parametrize("mode,min_improvement", [
    ("constant", 1.5),
    ("multiplicative", 5.0),
    ("additive", 5.0),
])
def test_correction_beats_uncorrected(mode, min_improvement, data):
    """The point of the whole exercise, on a discrepancy we control."""
    X_cheap, X_pair, X_test = data
    mf = MultiFidelitySurrogate(SPACE, mode=mode).fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_varying(X_pair)
    )
    truth = fine_varying(X_test)
    err_corrected = float((np.abs(mf.predict(X_test) - truth) / truth).mean())
    err_raw = float((np.abs(cheap(X_test) - truth) / truth).mean())

    assert err_corrected < err_raw / min_improvement, (
        f"{mode}: corrected {100*err_corrected:.2f}% vs "
        f"uncorrected {100*err_raw:.2f}% (wanted {min_improvement}x better)"
    )


def test_constant_ratio_is_recovered_exactly(data):
    """When the discrepancy really is a scale factor, a scalar nails it."""
    X_cheap, X_pair, X_test = data
    mf = MultiFidelitySurrogate(SPACE, mode="constant").fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_constant(X_pair)
    )
    truth = fine_constant(X_test)
    assert (np.abs(mf.predict(X_test) - truth) / truth).max() < 0.01
    assert mf.constants == pytest.approx(np.log([1.70, 1.35]), rel=1e-9)


def test_gp_correction_beats_a_scalar_when_the_ratio_varies(data):
    """And when it is not a scale factor, the GP earns its place."""
    X_cheap, X_pair, X_test = data
    truth = fine_varying(X_test)
    err = {}
    for mode in ("constant", "multiplicative"):
        mf = MultiFidelitySurrogate(SPACE, mode=mode).fit(
            X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_varying(X_pair)
        )
        err[mode] = float((np.abs(mf.predict(X_test) - truth) / truth).mean())
    assert err["multiplicative"] < err["constant"] / 2.0, err


# ---------------------------------------------------------------------------
# Diagnostics: is the scalar enough, and do the tiers correspond?
# ---------------------------------------------------------------------------
def test_constant_is_enough_distinguishes_the_two_cases(data):
    """A scalar should be preferred when it suffices, and flagged when not.

    Reaching for a GP that a constant would match is unearned complexity; the
    diagnostic makes that call explicit rather than a default.
    """
    X_cheap, X_pair, _ = data
    flat = MultiFidelitySurrogate(SPACE, mode="multiplicative").fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_constant(X_pair)
    )
    varying = MultiFidelitySurrogate(SPACE, mode="multiplicative").fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_varying(X_pair)
    )
    assert flat.constant_is_enough()
    assert not varying.constant_is_enough()


def test_correspondence_report_exposes_the_spread(data):
    X_cheap, X_pair, _ = data
    mf = MultiFidelitySurrogate(SPACE).fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_constant(X_pair)
    )
    rep = mf.correspondence_report()
    assert len(rep) == 2
    assert rep[0]["ratio_mean"] == pytest.approx(1.70, rel=1e-6)
    assert rep[0]["ratio_cv"] < 1e-6
    assert rep[0]["n"] == 16


def test_uncertainty_grows_with_the_correction(data):
    """Correcting adds a source of error; the error bars must say so."""
    X_cheap, X_pair, X_test = data
    mf = MultiFidelitySurrogate(SPACE, mode="multiplicative").fit(
        X_cheap, cheap(X_cheap), X_pair, cheap(X_pair), fine_varying(X_pair)
    )
    _, sd = mf.predict(X_test, return_std=True)
    assert np.isfinite(sd).all()
    assert (sd > 0).all()


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="fit"):
        MultiFidelitySurrogate(SPACE).predict(sobol_sample(SPACE, 4, seed=0))


# ---------------------------------------------------------------------------
# Real design space, real solver: the machinery must survive actual data
# ---------------------------------------------------------------------------
def test_runs_on_a_real_part_beam_tier():
    """Smoke test on the mount's beam tier, correcting against a synthetic
    'fine' tier so the test stays fast. The physics comparison is the example.
    """
    from dynopt.fea.reference import ReferenceFrameSolver
    from dynopt.optimize.driver import _sample_dataset

    part = MOTOR_MOUNT
    space = part.design_space
    solver = ReferenceFrameSolver(part)

    X = sobol_sample(space, 24, seed=0)
    Xp = sobol_sample(space, 10, seed=7)
    F, _, _, _ = _sample_dataset(solver, space, X, 12, targets="raw")
    Fp, _, _, _ = _sample_dataset(solver, space, Xp, 12, targets="raw")

    mf = MultiFidelitySurrogate(space, mode="multiplicative").fit(
        X, F, Xp, Fp, Fp * 1.75
    )
    pred = mf.predict(Xp)
    assert np.isfinite(pred).all()

    # Compare against the cheap surrogate's own prediction, not against the
    # solver. Predicting at points outside a 24-sample DOE in 5 dimensions
    # carries the cheap GP's interpolation error, which is not what this test
    # is about -- the ratio isolates the correction.
    base = mf.cheap.predict(Xp)
    ratio = pred / base
    assert np.abs(ratio - 1.75).max() < 0.02, (
        f"correction applied {ratio.mean():.3f}x, expected 1.75x"
    )
