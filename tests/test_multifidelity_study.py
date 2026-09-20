"""The published multi-fidelity numbers, owned by a test.

`docs/06_MULTIFIDELITY.md` quotes held-out errors from a 48-design, three-tier
solver run that takes about an hour. Re-solving it in CI is not an option, so
the dataset itself is checked in (`data/tier_pairs.json`, produced by
`examples/generate_tier_pairs.py`) and these tests re-derive the published
figures from it. If a change to the surrogate or the correction moves a number
that appears in the docs, this fails.

Tolerances are loose enough to absorb BLAS and scikit-learn version drift in
the GP fit, and tight enough that a real regression cannot hide: the
conclusions here are separations of 5 to 20 percentage points.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dynopt.geometry import get_part
from dynopt.surrogate import MultiFidelitySurrogate

DATA = Path(__file__).resolve().parents[1] / "data" / "tier_pairs.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="paired dataset absent")


@pytest.fixture(scope="module")
def pairs():
    return json.load(open(DATA))


def _arrays(rows, cheap):
    keep = [r for r in rows if cheap in r and "fine" in r]
    return (np.array([r["x"] for r in keep]),
            np.array([r[cheap] for r in keep]),
            np.array([r["fine"] for r in keep]))


def _held_out_error(space, X, C, F, mode, n_train=14):
    """Correction applied to *measured* cheap values -- isolates the model."""
    tr, te = np.arange(n_train), np.arange(n_train, len(X))
    mf = MultiFidelitySurrogate(space, mode=mode).fit(X[tr], C[tr], X[tr], C[tr], F[tr])
    Xn = np.array([space.normalize(x) for x in X[te]])
    out = np.zeros_like(C[te])
    for j in range(C.shape[1]):
        gp = mf.corrections[j]
        c = gp.predict(Xn) if gp is not None else np.full(te.size, mf.constants[j])
        out[:, j] = (C[te][:, j] + c) if mode == "additive" else (C[te][:, j] * np.exp(c))
    return 100 * np.mean(np.abs(out - F[te]) / F[te])


def test_dataset_is_complete(pairs):
    """48 designs, every tier. The run this replaced kept 4."""
    assert set(pairs) == {"motor_mount", "cantilever_arm"}
    for name, rows in pairs.items():
        assert len(rows) == 24, name
        for tier in ("beam", "coarse", "fine"):
            have = sum(tier in r for r in rows)
            assert have == 24, f"{name}: {tier} on {have}/24"


@pytest.mark.parametrize("name", ["motor_mount", "cantilever_arm"])
def test_correcting_the_coarse_tier_does_not_help(pairs, name):
    """docs/06 S6.2. Every correction makes the mount's coarse tier worse.

    The claim that matters is that there is no *useful* gain, not the sign of a
    fraction of a point, so this asserts the discrepancy is negligible to begin
    with and that no correction produces a meaningful improvement.
    """
    space = get_part(name).design_space
    X, C, F = _arrays(pairs[name], "coarse")
    te = np.arange(14, len(X))
    raw = 100 * np.mean(np.abs(C[te] - F[te]) / F[te])
    assert raw < 2.0, f"{name}: coarse tier is {raw:.2f}% off, expected under 2%"
    for mode in ("constant", "multiplicative", "additive"):
        got = _held_out_error(space, X, C, F, mode)
        assert got > raw - 1.0, (
            f"{name}: {mode} reached {got:.2f}% against {raw:.2f}% uncorrected -- "
            "if a correction now buys a real gain here, docs/06 S6.2 is stale"
        )


@pytest.mark.parametrize(
    "name,raw_lo,gp_hi",
    [("motor_mount", 18.0, 20.0), ("cantilever_arm", 15.0, 8.0)],
)
def test_the_gp_correction_beats_the_scalar_on_the_beam_tier(pairs, name, raw_lo, gp_hi):
    """docs/06 S6.3. Published: mount 24.42 -> 20.37 -> 16.96, arm 20.97 -> 13.21 -> 4.91.

    The mount's figures moved when the plate spokes were calibrated: its
    uncorrected error fell from 40.23% to 24.42%, and the correction's share of
    the remaining gap fell with it. Fixing the physics beat learning the
    discrepancy, which is the point docs/06 S6.6 now makes.
    """
    space = get_part(name).design_space
    X, C, F = _arrays(pairs[name], "beam")
    te = np.arange(14, len(X))
    raw = 100 * np.mean(np.abs(C[te] - F[te]) / F[te])
    const = _held_out_error(space, X, C, F, "constant")
    gp = _held_out_error(space, X, C, F, "multiplicative")

    assert raw > raw_lo, f"{name}: uncorrected {raw:.2f}%, expected above {raw_lo}%"
    assert const < raw, f"{name}: a scalar must at least help ({const:.2f}% vs {raw:.2f}%)"
    assert gp < const, (
        f"{name}: the GP reached {gp:.2f}% against the scalar's {const:.2f}%. "
        "docs/06 S6.3 claims a design-dependent correction beats a scalar."
    )
    assert gp < gp_hi, f"{name}: GP correction {gp:.2f}%, expected under {gp_hi}%"


@pytest.mark.parametrize(
    "name,lo,hi",
    [("motor_mount", 0.904, 2.630), ("cantilever_arm", 0.984, 1.173)],
)
def test_published_tier_ratio_ranges(pairs, name, lo, hi):
    """docs/06 S6.6 and architecture S1.5.3 both quote these ranges."""
    _, C, F = _arrays(pairs[name], "beam")
    ratio = F[:, 0] / C[:, 0]
    assert ratio.min() == pytest.approx(lo, abs=0.002)
    assert ratio.max() == pytest.approx(hi, abs=0.002)
