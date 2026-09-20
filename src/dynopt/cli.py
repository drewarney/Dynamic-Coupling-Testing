"""Command-line interface:  dynopt {modal,doe,optimize,coupling}."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np


def get_part(name: str):
    """Resolve a part from the registry, as a CLI-friendly error."""
    from dynopt.geometry import get_part as _get

    try:
        return _get(name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from None


def _solver_for(args, part):
    """Build the requested backend for a part."""
    if args.backend == "reference":
        from dynopt.fea.reference import ReferenceFrameSolver

        if not part.has_frame:
            raise SystemExit(f"part {part.name!r} has no frame tier; try --backend calculix")
        return ReferenceFrameSolver(part)
    if args.backend == "calculix":
        from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver

        return CalculiXSolver(part, CalculiXConfig())
    if args.backend == "ansys":
        from dynopt.fea.pyansys import MapdlConfig, MapdlSolver

        return MapdlSolver(part, MapdlConfig())
    raise SystemExit(f"unknown backend {args.backend!r}")


def _payload_dof(res, direction: int = 0) -> int:
    """A global DOF at the payload -- where a CLI demo should poke the structure.

    No fallback. This used to default to node 0 when a backend did not report
    its payload nodes, which is every 3D result: on the motor mount node 0 is a
    *clamped* base corner, so `dynopt coupling --backend calculix` hung its
    dashpot on a constrained node and printed coupling numbers for it.
    """
    payload = res.meta.get("payload_nodes")
    if not payload:
        gaps = "; ".join(res.missing_contract()) or "the run may be deliberately bare"
        raise SystemExit(
            f"backend {res.meta.get('backend', '?')} reported no payload nodes, so "
            f"there is nowhere meaningful to attach the damper ({gaps})"
        )
    per_node = int(res.meta.get("n_dof", 0)) // max(len(res.meta.get("nodes", [])), 1) or 6
    return int(payload[0]) * per_node + direction


def cmd_modal(args) -> int:
    from dynopt.dynamics import classify_modes, family_frequencies

    part = get_part(args.part)
    solver = _solver_for(args, part)
    params = part.design_space.to_dict(part.design_space.defaults)
    if args.params:
        params.update(json.loads(args.params))

    res = solver.solve(params, n_modes=args.modes)
    print(f"part: {part.name}    mass: {res.mass*1000:.1f} g")

    spec = part.mode_families
    origin = part.classification_origin(res)
    labels = (
        classify_modes(res, origin, spec, 0.35)
        if spec is not None and res.modes is not None
        else [""] * res.n_modes
    )
    for i, (f, lab) in enumerate(zip(res.frequencies, labels), start=1):
        print(f"  mode {i:2d}  {f:9.2f} Hz  {lab}")

    if spec is not None and res.modes is not None:
        print("\nfamilies:")
        for k, v in family_frequencies(res, origin, spec).items():
            print(f"  {k:<14s} {v:9.2f} Hz" if np.isfinite(v) else f"  {k:<14s}     --")
    return 0


def cmd_doe(args) -> int:
    from dynopt.doe import sobol_sample
    from dynopt.optimize.driver import _sample_dataset

    part = get_part(args.part)
    solver, space = _solver_for(args, part), part.design_space
    X = sobol_sample(space, args.n, seed=args.seed)
    F, mass, _, cols = _sample_dataset(solver, space, X, args.modes, verbose=True)

    if args.out:
        np.savez(args.out, X=X, F=F, mass=mass, cols=np.array(cols),
                 names=np.array(space.names))
        print(f"saved {args.out}")
    print(f"availability: {dict(zip(cols, np.round(np.isfinite(F).mean(axis=0), 3)))}")
    return 0


def cmd_coupling(args) -> int:
    from dynopt.dynamics import coupled_pairs, discrete_damper_matrix

    part = get_part(args.part)
    solver = _solver_for(args, part)
    space = part.design_space
    res = solver.solve(space.to_dict(space.defaults), n_modes=args.modes)
    n_dof = res.meta.get("n_dof") or (res.modes.shape[0] if res.modes is not None else 0)
    C = discrete_damper_matrix(n_dof, [(_payload_dof(res, 0), args.damping)])

    pairs = coupled_pairs(res, C, threshold=args.threshold)
    print(f"{len(pairs)} coupled pairs above eta={args.threshold}")
    for p in pairs[:20]:
        print(f"  modes {p['mode_i']:2d}-{p['mode_j']:2d}  "
              f"{p['f_i']:8.1f}/{p['f_j']:8.1f} Hz  eta={p['eta']:.3f}")
    return 0


def cmd_optimize(args) -> int:
    from dynopt.optimize import (
        ResonanceObjective,
        evaluate_baseline,
        optimize_adaptive,
        rpm_bands,
    )

    part = get_part(args.part)
    solver, space = _solver_for(args, part), part.design_space
    bands = rpm_bands(args.rpm_lo, args.rpm_hi, orders=tuple(args.orders))
    obj = ResonanceObjective(
        bands=bands, margin=args.margin, w_mass=args.w_mass,
        w_resonance=args.w_resonance, mass_ref=args.mass_ref,
        f_min=args.f_min,
    )
    base = evaluate_baseline(solver, space, obj)
    res = optimize_adaptive(
        solver, space, obj, n_init=args.init, n_rounds=args.rounds,
        n_per_round=args.per_round, n_modes=args.modes, verbose=True,
    )
    res.baseline = base
    print(res.report())

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "params": res.params,
                    "frequencies": res.frequencies.tolist(),
                    "mass": res.mass,
                    "objective": res.objective,
                    "n_fea_calls": res.n_fea_calls,
                    "baseline": {
                        "params": base["params"],
                        "frequencies": base["frequencies"].tolist(),
                        "mass": base["mass"],
                        "objective": base["objective"],
                    },
                },
                fh,
                indent=2,
            )
        print(f"saved {args.out}")

    if args.step:
        try:
            part.export_step(res.params, args.step)
            print(f"exported {args.step}")
        except (ImportError, NotImplementedError) as exc:
            print(f"STEP export skipped: {exc}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    # Common options live on a parent parser so they are accepted both before
    # and after the subcommand -- `dynopt modal --modes 6` is what people type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="reference",
                        choices=["reference", "calculix", "ansys"])
    common.add_argument("--modes", type=int, default=10)
    from dynopt.geometry import PART_NAMES

    common.add_argument("--part", default=PART_NAMES[0],
                        help=f"which part to work on ({', '.join(PART_NAMES)})")

    p = argparse.ArgumentParser(prog="dynopt", description=__doc__, parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("modal", help="run one modal analysis", parents=[common])
    m.add_argument("--params", help="JSON dict overriding design parameters")
    m.set_defaults(func=cmd_modal)

    d = sub.add_parser("doe", parents=[common], help="run a design of experiments")
    d.add_argument("-n", type=int, default=128)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", help="write results to an .npz file")
    d.set_defaults(func=cmd_doe)

    c = sub.add_parser("coupling", parents=[common], help="report cross-modal coupling")
    c.add_argument("--damping", type=float, default=150.0, help="dashpot rate [N.s/m]")
    c.add_argument("--threshold", type=float, default=0.3)
    c.set_defaults(func=cmd_coupling)

    o = sub.add_parser("optimize", parents=[common], help="run the full optimisation loop")
    o.add_argument("--rpm-lo", type=float, default=1000.0)
    o.add_argument("--rpm-hi", type=float, default=6000.0)
    o.add_argument("--orders", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    o.add_argument("--margin", type=float, default=0.25)
    o.add_argument("--w-mass", type=float, default=1.0)
    o.add_argument("--w-resonance", type=float, default=10.0)
    o.add_argument("--mass-ref", type=float, default=0.55)
    o.add_argument("--f-min", type=float, default=500.0)
    o.add_argument("--init", type=int, default=40)
    o.add_argument("--rounds", type=int, default=6)
    o.add_argument("--per-round", type=int, default=4)
    o.add_argument("--out", help="write a JSON result summary")
    o.add_argument("--step", help="export the optimised STEP file here")
    o.set_defaults(func=cmd_optimize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
