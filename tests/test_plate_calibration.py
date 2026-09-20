"""The plate spoke calibration, owned by the data it was calibrated on.

The beam tier's loaded error was one missing piece of stiffness, not a failure
of beam theory: four narrow spokes reproduce a fraction of a plate's
two-dimensional stiffness, and the payload sits exactly where that matters. The
idealisation was 11.4x too soft at the plate centre.

`data/bare_refs.json` holds 24 mesh-converged *unloaded* 3D solves and
`data/tier_pairs.json` the matching loaded ones, so the calibration is
reproducible in seconds without touching CalculiX.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dynopt.fea.reference import modal_solve
from dynopt.geometry import MOTOR_MOUNT
from dynopt.geometry import motor_mount as mm

ROOT = Path(__file__).resolve().parents[1]
PAIRS, BARE = ROOT / "data" / "tier_pairs.json", ROOT / "data" / "bare_refs.json"
pytestmark = pytest.mark.skipif(
    not (PAIRS.exists() and BARE.exists()), reason="reference datasets absent"
)


@pytest.fixture(scope="module")
def refs():
    loaded = json.load(open(PAIRS))["motor_mount"]
    bare = {r["i"]: r for r in json.load(open(BARE)) if "bare" in r}
    sp = MOTOR_MOUNT.design_space
    return (
        [sp.to_dict(np.array(r["x"])) for r in loaded],
        np.array([r["fine"][0] for r in loaded]),
        np.array([bare[r["i"]]["bare"][0] for r in loaded]),
    )


def _f1(d, beta, loaded):
    kw = {} if loaded else dict(motor_mass=0.0, motor_inertia=(0.0, 0.0, 0.0))
    return modal_solve(
        mm.build_frame(d, plate_spoke_factor=beta, **kw), n_modes=6
    ).frequencies[0]


def _err(X, truth, beta, loaded, which):
    return 100 * float(np.mean([
        abs(_f1(X[i], beta, loaded) / truth[i] - 1.0) for i in which
    ]))


def test_the_calibration_improves_both_targets_out_of_sample(refs):
    """docs/06 S6.6: held out, bare 10.1 -> 7.5% and loaded 40.8 -> 13.7%.

    Both, not one at the other's expense. An earlier reading of this said the
    fix traded bare accuracy for loaded accuracy -- that came from judging the
    bare case at a single design, where the old model happened to be unusually
    good.
    """
    X, loaded_truth, bare_truth = refs
    held = np.arange(len(X))[1::2]

    old_bare = _err(X, bare_truth, 1.0, False, held)
    new_bare = _err(X, bare_truth, mm.PLATE_SPOKE_FACTOR, False, held)
    old_load = _err(X, loaded_truth, 1.0, True, held)
    new_load = _err(X, loaded_truth, mm.PLATE_SPOKE_FACTOR, True, held)

    assert new_load < 0.5 * old_load, (
        f"loaded held-out error {new_load:.1f}% against {old_load:.1f}% before; "
        "docs/06 claims it roughly a third"
    )
    assert new_bare <= old_bare, (
        f"bare held-out error {new_bare:.1f}% against {old_bare:.1f}% before; "
        "the calibration must not buy loaded accuracy with bare accuracy"
    )
    assert new_load == pytest.approx(13.7, abs=2.0)
    assert new_bare == pytest.approx(7.5, abs=2.0)


def test_the_shipped_factor_is_the_one_that_was_calibrated(refs):
    """Guards against the constant drifting away from its own calibration."""
    X, loaded_truth, _ = refs
    fit = np.arange(len(X))[0::2]
    shipped = _err(X, loaded_truth, mm.PLATE_SPOKE_FACTOR, True, fit)
    for other in (1.0, 1.5, 4.0, 6.0):
        assert shipped < _err(X, loaded_truth, other, True, fit), (
            f"spoke factor {other} beats the shipped "
            f"{mm.PLATE_SPOKE_FACTOR} on the fitting half"
        )


def test_the_spoke_idealisation_was_softer_than_plate_theory():
    """The 11.4x quoted in PLATE_SPOKE_FACTOR's derivation."""
    d = MOTOR_MOUNT.design_space.to_dict(MOTOR_MOUNT.design_space.defaults)
    top_span, t = 0.55 * d["span"], d["plate_thk"]
    E, nu = 68.9e9, 0.33

    w = 0.5 * 1.0 * top_span                       # the old effective width
    k_spokes = 4 * 3 * E * (w * t**3 / 12) / (top_span * np.sqrt(2)) ** 3
    D = E * t**3 / (12 * (1 - nu**2))
    k_plate = D / (0.0116 * (2 * top_span) ** 2)   # square plate, point load

    assert k_plate / k_spokes == pytest.approx(11.4, rel=0.1)
    # And the shipped factor closes most, but not all, of that gap -- four point
    # supports really are softer than a continuous edge.
    assert 1.0 < mm.PLATE_SPOKE_FACTOR < 5.7


def test_mass_is_untouched_by_the_stiffness_calibration(refs):
    """Effective width sets stiffness; the plate's true mass is renormalised."""
    X, _, _ = refs
    for d in X[:6]:
        m = [modal_solve(mm.build_frame(d, plate_spoke_factor=b), n_modes=3).mass
             for b in (1.0, 2.5, 4.0)]
        assert m[0] == pytest.approx(m[1], rel=1e-9)
        assert m[0] == pytest.approx(m[2], rel=1e-9)
