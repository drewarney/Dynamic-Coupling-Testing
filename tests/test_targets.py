"""Regression targets: what the surrogate learns, and what it must not drop."""

import numpy as np
import pytest

from dynopt.doe import sobol_sample
from dynopt.fea.reference import MATERIALS, FrameModel, ReferenceFrameSolver, Section
from dynopt.geometry import CANTILEVER_ARM, MOTOR_MOUNT
from dynopt.geometry.base import DesignSpace, DesignVariable, PartSpec
from dynopt.optimize import (
    FamilyTargets,
    RawTargets,
    ResonanceObjective,
    ScalarTargets,
    extract_targets,
    make_target_set,
    rpm_bands,
)
from dynopt.optimize.driver import _sample_dataset
from dynopt.surrogate import FrequencySurrogate


@pytest.fixture(scope="module")
def objective():
    return ResonanceObjective(
        bands=rpm_bands(1000, 6000, orders=(1.0, 2.0, 4.0)),
        margin=0.25, w_mass=1.0, w_resonance=10.0, mass_ref=0.55, f_min=500.0,
    )


class Unlabelled(PartSpec):
    """A part that declares no mode families at all.

    Exists to prove the tracking-free targets need nothing from the part. If
    they ever start requiring a classification, this fixture stops working.
    """

    name = "unlabelled"

    @property
    def design_space(self) -> DesignSpace:
        return DesignSpace([
            DesignVariable("length", 0.10, 0.40, 0.25, "m", ""),
            DesignVariable("dia", 0.008, 0.030, 0.016, "m", ""),
        ])

    def build_frame(self, params, n_elem: int = 10) -> FrameModel:
        L = float(params["length"])
        nodes = np.zeros((n_elem + 1, 3))
        nodes[:, 0] = np.linspace(0.0, L, n_elem + 1)
        return FrameModel(
            nodes=nodes,
            elements=[(i, i + 1) for i in range(n_elem)],
            sections=[Section.circle(float(params["dia"]))] * n_elem,
            material=MATERIALS["AL6061"],
            fixed_nodes=[0],
        )


# ---------------------------------------------------------------------------
# The trap: a filter that silently discards the best designs
# ---------------------------------------------------------------------------
def test_surrogate_keeps_zero_and_negative_targets():
    """A blanket ``y > 0`` filter would drop exactly the samples that matter.

    Resonance risk is zero for a design that clears every band -- the ones you
    are searching for -- and separation margin goes negative inside a band.
    Under the old filter those rows vanished from training with no warning, so
    the surrogate would have been fitted on the designs it least needed to
    understand.
    """
    space = MOTOR_MOUNT.design_space
    X = sobol_sample(space, 24, seed=0)
    Y = np.column_stack([np.linspace(0.0, 3.0, 24), np.linspace(-1.5, 2.0, 24)])

    kept_by_old_filter = int((np.isfinite(Y) & (Y > 0)).all(axis=1).sum())
    assert kept_by_old_filter < 24, "fixture must contain zero/negative targets"

    sur = FrequencySurrogate(
        space, n_modes=2, transforms=("log1p", "linear")
    ).fit(X, Y)
    assert all(m is not None for m in sur.models), "a column was silently dropped"
    assert np.abs(sur.predict(X) - Y).max() < 1e-5


@pytest.mark.parametrize("kind,lo,hi", [
    ("log1p", 0.0, 5.0),     # risk: non-negative, can be vanishingly small
    ("linear", -3.0, 3.0),   # separation margin: negative inside a band
    ("log", 1.0, 50.0),      # frequencies and mass: strictly positive
])
def test_each_transform_round_trips(kind, lo, hi):
    """Each transform must survive a fit-and-predict cycle in its own domain.

    The target is a smooth function *of the design variables*, not an arbitrary
    vector: a GP asked to interpolate values unrelated to its inputs is being
    tested on noise, which says nothing about the transform.
    """
    space = MOTOR_MOUNT.design_space
    X = sobol_sample(space, 40, seed=3)
    u = space.normalize(X[0]) * 0  # shape helper
    t = np.array([space.normalize(x)[0] for x in X])
    y = lo + (hi - lo) * (0.5 + 0.5 * np.sin(2.0 * np.pi * t))

    sur = FrequencySurrogate(space, n_modes=1, transforms=(kind,)).fit(X, y[:, None])
    err = np.abs(sur.predict(X)[:, 0] - y)
    scale = max(abs(hi - lo), 1.0)
    assert err.max() / scale < 1e-3, f"{kind}: max round-trip error {err.max():.3e}"


# ---------------------------------------------------------------------------
# Tracking-free targets need nothing from the part
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ts", [RawTargets(5), ScalarTargets()])
def test_tracking_free_targets_run_without_families(ts, objective):
    part = Unlabelled()
    assert part.mode_families is None

    solver = ReferenceFrameSolver(part)
    X = sobol_sample(part.design_space, 6, seed=0)
    F, mass, results, cols = _sample_dataset(
        solver, part.design_space, X, 10, targets=ts, objective=objective
    )
    assert cols == list(ts.columns)
    assert np.isfinite(F).all()
    assert np.isfinite(mass).all()


def test_family_targets_refuse_an_unlabelled_part():
    """Failing loudly beats regressing something meaningless."""
    with pytest.raises(ValueError, match="mode_families"):
        make_target_set("family", n_modes=6, part=Unlabelled())


# ---------------------------------------------------------------------------
# The premise: ordered eigenvalues are continuous in the design parameters
# ---------------------------------------------------------------------------
def test_sorted_frequencies_are_continuous_along_a_design_path():
    r"""Sorted targets were dismissed as divergent; the maths says otherwise.

    The k-th smallest eigenvalue of a symmetric pencil is a continuous function
    of the design parameters. A mode crossing puts a *kink* in :math:`f_{(k)}(x)`,
    not a jump -- which a GP handles. This walks a straight line across the
    design space and bounds the step-to-step change.
    """
    part = MOTOR_MOUNT
    space = part.design_space
    solver = ReferenceFrameSolver(part)
    lo = np.array([v.lower for v in space.variables])
    hi = np.array([v.upper for v in space.variables])

    n_steps = 40
    F = []
    for t in np.linspace(0.0, 1.0, n_steps):
        res = solver.solve(space.to_dict(lo + t * (hi - lo)), n_modes=8)
        F.append(np.sort(res.frequencies)[:5])
    F = np.array(F)

    rel_step = np.abs(np.diff(F, axis=0)) / F[:-1]
    assert np.isfinite(F).all()
    # 1/40th of the full diagonal of the design box must not move any sorted
    # frequency by more than a modest fraction: continuous, not jumpy.
    assert rel_step.max() < 0.25, (
        f"largest sorted-frequency step is {100*rel_step.max():.1f}% over "
        f"1/{n_steps} of the design space -- that would be a discontinuity"
    )


# ---------------------------------------------------------------------------
# ScalarTargets must reproduce the objective it stands in for
# ---------------------------------------------------------------------------
def test_scalar_targets_reconstruct_the_objective_exactly(objective):
    """Given the true scalars, the reconstruction must equal the real objective.

    If this drifts, the optimiser is minimising something other than what the
    sign-off will score.
    """
    part = MOTOR_MOUNT
    solver = ReferenceFrameSolver(part)
    ts = ScalarTargets()
    for x in sobol_sample(part.design_space, 12, seed=7):
        res = solver.solve(part.design_space.to_dict(x), n_modes=12)
        row = ts.extract(res, objective)
        assert ts.objective_from(row, res.mass, objective) == pytest.approx(
            objective(res.frequencies, res.mass), rel=1e-12
        )


def test_scalar_columns_have_the_transforms_their_ranges_need(objective):
    """The transforms must match the ranges these quantities actually take.

    ``min_sep`` goes genuinely negative inside an excitation band, so it cannot
    be logged at all. ``risk`` is a sum of Gaussians, so it is strictly positive
    but ranges over four decades and gets arbitrarily close to zero for exactly
    the designs worth finding; ``log`` would stretch that irrelevant tail across
    most of the GP's dynamic range, while ``log1p`` leaves it compact.
    """
    ts = ScalarTargets()
    assert dict(zip(ts.columns, ts.transforms)) == {
        "f_min": "log", "risk": "log1p", "min_sep": "linear"
    }

    solver = ReferenceFrameSolver(MOTOR_MOUNT)
    space = MOTOR_MOUNT.design_space
    rows = np.array([
        ts.extract(solver.solve(space.to_dict(x), n_modes=12), objective)
        for x in sobol_sample(space, 24, seed=11)
    ])
    risk, sep = rows[:, 1], rows[:, 2]
    assert risk.min() > 0.0, "risk is a sum of Gaussians; it cannot be negative"
    assert risk.min() < 1e-3, "no near-zero risk sampled; fixture is weak"
    assert risk.max() / risk.min() > 1e3, "risk should span decades"
    assert sep.min() < 0.0, "no in-band design sampled; fixture is weak"
    assert np.isfinite(rows[:, 0]).all() and (rows[:, 0] > 0).all()


# ---------------------------------------------------------------------------
# Re-extraction is free
# ---------------------------------------------------------------------------
def test_retargeting_needs_no_resolve(objective):
    """Changing target type must not cost another DOE."""
    part = MOTOR_MOUNT
    solver = ReferenceFrameSolver(part)
    X = sobol_sample(part.design_space, 16, seed=0)
    _, _, results, _ = _sample_dataset(solver, part.design_space, X, 12, targets="raw")

    for ts in (RawTargets(6), ScalarTargets(), FamilyTargets(part.mode_families)):
        F = extract_targets(results, ts, objective)
        assert F.shape == (16, len(ts.columns))
        assert np.isfinite(F).mean() > 0.9


def test_both_parts_support_every_tracking_free_target(objective):
    """Generality claim: the same targets work on two unrelated structures."""
    for part in (MOTOR_MOUNT, CANTILEVER_ARM):
        solver = ReferenceFrameSolver(part)
        X = sobol_sample(part.design_space, 6, seed=1)
        for ts in (RawTargets(5), ScalarTargets()):
            F, mass, _, cols = _sample_dataset(
                solver, part.design_space, X, 10, targets=ts, objective=objective
            )
            assert np.isfinite(F).all(), f"{part.name} / {ts}"


# ---------------------------------------------------------------------------
# The measured verdict: sorted frequencies beat labelled ones
# ---------------------------------------------------------------------------
def test_default_target_is_raw():
    """The default is a measured choice; changing it needs new measurements.

    examples/07_target_comparison.py scored every target on the FEA-verified
    objective of the design it produces, on two unrelated parts. Sorted
    frequencies won both.
    """
    from dynopt.optimize.targets import RawTargets, make_target_set

    assert isinstance(make_target_set(None, n_modes=6), RawTargets)


def test_family_targets_discard_modes_the_objective_still_sees(objective):
    """The mechanism behind the verdict, pinned cheaply.

    Resonance risk sums over every mode and the frequency floor is a minimum
    over every mode, so the objective is a functional of the whole spectrum.
    Family columns are a *subset*: modes the classifier labels ``local`` never
    reach the surrogate but are fully present in the truth. That gap is why
    families predict their own columns better and the objective far worse.
    """
    part = MOTOR_MOUNT
    solver = ReferenceFrameSolver(part)
    space = part.design_space
    fam = FamilyTargets(part.mode_families)

    gaps = []
    for x in sobol_sample(space, 12, seed=5):
        res = solver.solve(space.to_dict(x), n_modes=12)
        all_f = np.asarray(res.frequencies, dtype=float)
        fam_f = np.array([v for v in fam.extract(res) if np.isfinite(v)])

        assert fam_f.size < all_f.size, "fixture must have unlabelled modes"
        gaps.append(abs(objective(fam_f, res.mass) - objective(all_f, res.mass)))

    assert max(gaps) > 1e-6, (
        "family columns reproduced the objective exactly -- if this ever holds, "
        "the discarded modes carry no objective weight and the verdict may flip"
    )


def test_raw_predicts_the_objective_better_than_families(objective):
    """Headline claim, owned: sorted targets beat labelled ones on the objective.

    Not on per-column accuracy -- families win there, and that is the point.
    Scored on the quantity the optimiser actually consumes.
    """
    part = MOTOR_MOUNT
    solver = ReferenceFrameSolver(part)
    space = part.design_space

    Xtr = sobol_sample(space, 64, seed=1)
    Xte = sobol_sample(space, 48, seed=99)
    _, mtr, rtr, _ = _sample_dataset(solver, space, Xtr, 12, targets="raw")
    _, mte, rte, _ = _sample_dataset(solver, space, Xte, 12, targets="raw")
    true_J = np.array([objective(r.frequencies, m) for r, m in zip(rte, mte)])

    err = {}
    for name, ts in (("raw", RawTargets(6)), ("family", FamilyTargets(part.mode_families))):
        F = extract_targets(rtr, ts, objective)
        ok = np.isfinite(mtr)
        sur = FrequencySurrogate(
            space, n_modes=F.shape[1], transforms=ts.transforms
        ).fit(Xtr[ok], F[ok], mtr[ok])
        P, Pm = sur.predict(Xte), sur.predict_mass(Xte)
        pred_J = np.array([
            ts.objective_from(P[i], float(Pm[i]), objective) for i in range(len(Xte))
        ])
        good = np.isfinite(pred_J) & np.isfinite(true_J)
        err[name] = float(
            np.sqrt(np.mean((pred_J[good] - true_J[good]) ** 2)) / np.std(true_J[good])
        )

    assert err["raw"] < err["family"], (
        f"objective nRMSE: raw {err['raw']:.3f} vs family {err['family']:.3f}. "
        "The README and docs/01_THEORY.md §2.3.1 publish the opposite; "
        "re-measure with examples/07_target_comparison.py and update both."
    )
