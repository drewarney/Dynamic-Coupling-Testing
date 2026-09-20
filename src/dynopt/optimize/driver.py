r"""The orchestration loop: CAD -> FEA -> surrogate -> optimiser -> CAD.

Three strategies, in increasing sophistication:

1. :func:`optimize_direct` -- call FEA inside the optimiser.  Simple, exact, and
   too slow for anything but the reference tier (SLSQP needs ~(n+1) evaluations
   per gradient, so 5 variables x 60 iterations x 30 s/solve = 2.5 hours).
2. :func:`optimize_surrogate` -- train once on a DOE, optimise the surrogate,
   verify the winner with one real FEA run.  Fast, but only as good as the
   surrogate near the optimum.
3. :func:`optimize_adaptive` -- **the one to use.**  Alternates surrogate
   optimisation with targeted FEA runs at the proposed optimum, refitting each
   round.  This is efficient global optimisation (EGO): the surrogate gets
   accurate exactly where it matters and the FEA budget goes where it pays.
   Typically converges in 40-80 total FEA runs versus ~600 for direct SLSQP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution, minimize

from ..doe.sampling import sobol_sample
from ..dynamics.modal import (
    ModeFamilySpec,
    cluster_frequencies,
    cluster_modes,
    effective_mass_fraction,
    family_frequencies,
    track_clusters,
)
from ..fea.base import ModalResult, ModalSolver
from ..geometry.base import DesignSpace
from ..surrogate.gp import FrequencySurrogate
from .objectives import ResonanceObjective
from .targets import ClusterTargets, FamilyTargets, TargetSet, make_target_set


@dataclass
class OptimizationResult:
    x: np.ndarray
    params: dict[str, float]
    objective: float
    frequencies: np.ndarray
    mass: float
    n_fea_calls: int
    wall_time: float
    history: list[dict] = field(default_factory=list)
    surrogate: FrequencySurrogate | None = None
    baseline: dict | None = None

    def report(self) -> str:
        lines = [
            "=" * 66,
            "  RESONANCE OPTIMISATION RESULT",
            "=" * 66,
            f"  objective        : {self.objective:.4f}",
            f"  mass             : {self.mass*1000:.1f} g",
            f"  FEA evaluations  : {self.n_fea_calls}",
            f"  wall time        : {self.wall_time:.1f} s",
            "  frequencies [Hz] : "
            + ", ".join(f"{f:.1f}" for f in self.frequencies[:8]),
            "-" * 66,
        ]
        for k, v in self.params.items():
            lines.append(f"  {k:<16s} : {v*1000:8.3f} mm")
        if self.baseline:
            b = self.baseline
            dm = 100.0 * (self.mass - b["mass"]) / b["mass"]
            df = 100.0 * (self.frequencies[0] - b["frequencies"][0]) / b["frequencies"][0]
            lines += [
                "-" * 66,
                f"  vs baseline      : mass {dm:+.1f}%, f1 {df:+.1f}%",
            ]
        lines.append("=" * 66)
        return "\n".join(lines)


def _part_of(res: ModalResult):
    """The :class:`PartSpec` the solver recorded on this result, if any."""
    return res.meta.get("part")


def _origin_of(res: ModalResult) -> np.ndarray | None:
    """Classification origin: ask the part, else fall back to the payload."""
    part = _part_of(res)
    if part is not None:
        return part.classification_origin(res)
    nodes = res.meta.get("nodes")
    if nodes is None:
        return None
    payload = res.meta.get("payload_nodes")
    nodes = np.asarray(nodes, dtype=float)
    return nodes[list(payload)].mean(axis=0) if payload else nodes.mean(axis=0)


def _families_of(res: ModalResult) -> ModeFamilySpec | None:
    part = _part_of(res)
    return part.mode_families if part is not None else None


def resolve_targets(solver, targets, n_modes: int) -> TargetSet:
    """Turn a name or TargetSet into a TargetSet, using the solver's part.

    Resolved before sampling rather than on the first result, so a
    misconfigured run fails immediately instead of after the DOE.
    """
    ts = make_target_set(targets, n_modes=n_modes, part=getattr(solver, "part", None))
    ts.reset()
    return ts


def _sample_dataset(
    solver: ModalSolver,
    space: DesignSpace,
    X: np.ndarray,
    n_modes: int,
    targets: str | TargetSet = "raw",
    verbose: bool = False,
    rtol: float = 0.02,
    min_purity: float = 0.35,
    families: ModeFamilySpec | None = None,
    objective: ResonanceObjective | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ModalResult], list[str]]:
    """Run FEA over a design matrix and build continuous regression targets.

    ``targets`` is a :class:`dynopt.optimize.targets.TargetSet`, or one of the
    names ``"raw"`` (default), ``"family"``, ``"scalars"``, ``"cluster"``.

    ``"raw"`` is the default on measured evidence, not preference: it produced
    the best FEA-verified objective on both benchmark parts. ``"family"`` gives
    physically named columns and is better for *reporting* a design, but it
    discards modes the objective still sees. See
    ``examples/07_target_comparison.py``.

    The solved results come back alongside the targets, so switching target type
    or changing the excitation bands is a re-extract, never a re-solve.

    Returns ``(F, mass, results, column_names)``.
    """
    ts = (
        FamilyTargets(families, min_purity=min_purity)
        if families is not None
        else resolve_targets(solver, targets, n_modes)
    )
    if isinstance(ts, ClusterTargets):
        ts.rtol = rtol

    n = X.shape[0]
    mass = np.full(n, np.nan)
    results: list[ModalResult] = []
    rows: list[list[float] | None] = [None] * n

    for i, x in enumerate(X):
        try:
            res = solver.solve(space.to_dict(x), n_modes=n_modes)
        except Exception as exc:  # a degenerate geometry is a data point, not a crash
            if verbose:
                print(f"  [{i+1}/{n}] FAILED: {exc}")
            results.append(None)
            continue

        results.append(res)
        mass[i] = res.mass
        rows[i] = ts.extract(res, objective)

        if verbose and (i + 1) % 10 == 0:
            vals = np.asarray(rows[i], dtype=float)
            f1 = np.nanmin(vals) if np.isfinite(vals).any() else np.nan
            print(f"  [{i+1}/{n}] {ts.columns[0]}={f1:.1f}  m={mass[i]*1000:.1f} g")

    cols = list(ts.columns)
    F = np.full((n, len(cols)), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            F[i, : len(row)] = row
    return F, mass, results, cols


def optimize_direct(
    solver: ModalSolver,
    space: DesignSpace,
    objective: ResonanceObjective,
    x0: np.ndarray | None = None,
    n_modes: int = 6,
    method: str = "SLSQP",
    maxiter: int = 60,
    verbose: bool = True,
) -> OptimizationResult:
    """Gradient-based optimisation calling FEA directly.  Exact but expensive."""
    t0 = time.perf_counter()
    calls = {"n": 0}
    x0 = space.defaults if x0 is None else np.asarray(x0, dtype=float)

    def f(x):
        calls["n"] += 1
        res = solver.solve(space.to_dict(space.clip(x)), n_modes=n_modes)
        J = objective(res.frequencies, res.mass, record=True)
        if verbose and calls["n"] % 10 == 0:
            print(f"  eval {calls['n']:4d}  J={J:8.4f}  f1={res.frequencies[0]:7.1f} Hz")
        return J

    opt = minimize(
        f, x0, method=method, bounds=space.bounds,
        options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1e-5},
    )

    best = solver.solve(space.to_dict(space.clip(opt.x)), n_modes=n_modes)
    return OptimizationResult(
        x=space.clip(opt.x),
        params=space.to_dict(space.clip(opt.x)),
        objective=float(opt.fun),
        frequencies=best.frequencies,
        mass=best.mass,
        n_fea_calls=calls["n"] + 1,
        wall_time=time.perf_counter() - t0,
        history=list(objective.history),
    )


def optimize_surrogate(
    solver: ModalSolver,
    space: DesignSpace,
    objective: ResonanceObjective,
    n_train: int = 200,
    n_modes: int = 6,
    targets: str = "raw",
    seed: int = 0,
    verbose: bool = True,
) -> OptimizationResult:
    """Train a GP on a one-shot DOE, then optimise the surrogate globally."""
    t0 = time.perf_counter()
    ts = resolve_targets(solver, targets, n_modes)
    X = sobol_sample(space, n_train, seed=seed)
    if verbose:
        print(f"[DOE] running {n_train} FEA solves ({ts}) ...")
    F, mass, _, cols = _sample_dataset(
        solver, space, X, n_modes, targets=ts, verbose=verbose, objective=objective
    )

    ok = np.isfinite(mass)
    sur = FrequencySurrogate(
        space, n_modes=F.shape[1], transforms=ts.transforms
    ).fit(X[ok], F[ok], mass[ok])
    if verbose:
        print("[GP] fitted. mean LOO error per mode will be reported separately.")

    def f(x):
        xx = np.atleast_2d(space.clip(x))
        return ts.objective_from(
            sur.predict(xx)[0], float(sur.predict_mass(xx)[0]), objective
        )

    opt = differential_evolution(
        f, space.bounds, seed=seed, maxiter=200, tol=1e-8, polish=True
    )

    verify = solver.solve(space.to_dict(space.clip(opt.x)), n_modes=n_modes)
    return OptimizationResult(
        x=space.clip(opt.x),
        params=space.to_dict(space.clip(opt.x)),
        objective=objective(verify.frequencies, verify.mass),
        frequencies=verify.frequencies,
        mass=verify.mass,
        n_fea_calls=int(ok.sum()) + 1,
        wall_time=time.perf_counter() - t0,
        surrogate=sur,
    )


def optimize_adaptive(
    solver: ModalSolver,
    space: DesignSpace,
    objective: ResonanceObjective,
    n_init: int = 40,
    n_rounds: int = 8,
    n_per_round: int = 4,
    n_modes: int = 6,
    targets: str = "raw",
    seed: int = 0,
    verbose: bool = True,
) -> OptimizationResult:
    """Efficient global optimisation: alternate GP fitting and targeted FEA.

    Each round optimises the current surrogate, then spends ``n_per_round`` FEA
    runs -- one at the surrogate optimum (exploitation) and the rest perturbed
    around it (local exploration, keeping the GP honest nearby).  The returned
    design is the best *FEA-verified* point, never an unverified surrogate
    prediction.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    ts = resolve_targets(solver, targets, n_modes)

    X = sobol_sample(space, n_init, seed=seed)
    if verbose:
        print(f"[init] {n_init} FEA solves ({ts}) ...")
    F, mass, _, cols = _sample_dataset(
        solver, space, X, n_modes, targets=ts, verbose=verbose, objective=objective
    )

    history = []
    best_J, best_x, best_res = np.inf, None, None

    for r in range(n_rounds):
        ok = np.isfinite(mass)
        sur = FrequencySurrogate(
        space, n_modes=F.shape[1], transforms=ts.transforms
    ).fit(X[ok], F[ok], mass[ok])

        def f(x):
            xx = np.atleast_2d(space.clip(x))
            return ts.objective_from(
                sur.predict(xx)[0], float(sur.predict_mass(xx)[0]), objective
            )

        opt = differential_evolution(
            f, space.bounds, seed=seed + r, maxiter=120, tol=1e-8, polish=True
        )

        # One exploitation point + perturbed neighbours, all clipped to bounds.
        lo = np.array([v.lower for v in space.variables])
        hi = np.array([v.upper for v in space.variables])
        batch = [space.clip(opt.x)]
        for _ in range(max(n_per_round - 1, 0)):
            step = 0.08 * (hi - lo) * rng.standard_normal(len(space))
            batch.append(space.clip(opt.x + step))
        batch = np.array(batch)

        Fb, mb, resb, _ = _sample_dataset(
            solver, space, batch, n_modes, targets=ts, verbose=False,
            objective=objective,
        )
        X = np.vstack([X, batch])
        F = np.vstack([F, Fb])
        mass = np.concatenate([mass, mb])

        for xi, fi, mi, ri in zip(batch, Fb, mb, resb):
            if ri is None or not np.isfinite(mi):
                continue
            J = objective(ri.frequencies, mi)
            if J < best_J:
                best_J, best_x, best_res = J, xi, ri

        history.append(
            {
                "round": r + 1,
                "surrogate_J": float(opt.fun),
                "best_J": float(best_J),
                "n_fea": int(X.shape[0]),
                "f1": float(best_res.frequencies[0]) if best_res else np.nan,
                "mass": float(best_res.mass) if best_res else np.nan,
            }
        )
        if verbose:
            h = history[-1]
            print(
                f"[round {r+1}/{n_rounds}] surrogate J={h['surrogate_J']:.4f}  "
                f"verified best J={h['best_J']:.4f}  f1={h['f1']:.1f} Hz  "
                f"m={h['mass']*1000:.1f} g  (FEA calls: {h['n_fea']})"
            )

    ok = np.isfinite(mass)
    sur = FrequencySurrogate(
        space, n_modes=F.shape[1], transforms=ts.transforms
    ).fit(X[ok], F[ok], mass[ok])

    return OptimizationResult(
        x=best_x,
        params=space.to_dict(best_x),
        objective=float(best_J),
        frequencies=best_res.frequencies,
        mass=best_res.mass,
        n_fea_calls=int(X.shape[0]),
        wall_time=time.perf_counter() - t0,
        history=history,
        surrogate=sur,
    )


def evaluate_baseline(
    solver: ModalSolver, space: DesignSpace, objective: ResonanceObjective,
    x: np.ndarray | None = None, n_modes: int = 6,
) -> dict:
    """Evaluate the starting design so improvements can be quoted against it."""
    x = space.defaults if x is None else x
    res = solver.solve(space.to_dict(x), n_modes=n_modes)
    return {
        "x": x,
        "params": space.to_dict(x),
        "frequencies": res.frequencies,
        "mass": res.mass,
        "objective": objective(res.frequencies, res.mass),
        "violation": objective.constraint_violation(res.frequencies),
        "eff_mass": effective_mass_fraction(res, 2) if res.modes is not None else None,
    }
