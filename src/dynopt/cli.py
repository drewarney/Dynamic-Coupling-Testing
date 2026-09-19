"""Command-line interface:  dynopt {modal,doe,optimize,coupling}."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np


def _solver_and_space(args):
    from dynopt.fea.reference import ReferenceFrameSolver
    from dynopt.geometry import MOTOR_MOUNT_SPACE, build_frame

    if args.backend == "reference":
        return ReferenceFrameSolver(build_frame), MOTOR_MOUNT_SPACE
    if args.backend == "calculix":
        from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver
        from dynopt.geometry import MotorMountCAD

        return CalculiXSolver(MotorMountCAD(), CalculiXConfig()), MOTOR_MOUNT_SPACE
    if args.backend == "ansys":
        from dynopt.fea.pyansys import MapdlConfig, MapdlSolver
        from dynopt.geometry import MotorMountCAD

        return MapdlSolver(MotorMountCAD(), MapdlConfig()), MOTOR_MOUNT_SPACE
    raise SystemExit(f"unknown backend {args.backend!r}")


def cmd_modal(args) -> int:
    from dynopt.dynamics import classify_modes, family_frequencies
    from dynopt.fea.reference import modal_solve
    from dynopt.geometry import build_frame, motor_node_index

    _, space = _solver_and_space(args)
    params = space.to_dict(space.defaults)
    if args.params:
        params.update(json.loads(args.params))

    model = build_frame(params)
    res = modal_solve(model, n_modes=args.modes)
    origin = model.nodes[motor_node_index()]
    labels = classify_modes(res, origin, 0.35)

    print(f"mass: {res.mass*1000:.1f} g")
    for i, (f, lab) in enumerate(zip(res.frequencies, labels), start=1):
        print(f"  mode {i:2d}  {f:9.2f} Hz  {lab}")
    print("\nfamilies:")
    for k, v in family_frequencies(res, origin).items():
        print(f"  {k:<14s} {v:9.2f} Hz" if np.isfinite(v) else f"  {k:<14s}     --")
    return 0


def cmd_doe(args) -> int:
    from dynopt.doe import sobol_sample
    from dynopt.optimize.driver import _sample_dataset

    solver, space = _solver_and_space(args)
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
    from dynopt.fea.reference import modal_solve
    from dynopt.geometry import build_frame, motor_node_index

    _, space = _solver_and_space(args)
    model = build_frame(space.to_dict(space.defaults))
    res = modal_solve(model, n_modes=args.modes)
    node = motor_node_index()
    C = discrete_damper_matrix(model.n_dof, [(node * 6 + 0, args.damping)])

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

    solver, space = _solver_and_space(args)
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
        from dynopt.geometry import MotorMountCAD

        try:
            MotorMountCAD().export_step(res.params, args.step)
            print(f"exported {args.step}")
        except ImportError as exc:
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
