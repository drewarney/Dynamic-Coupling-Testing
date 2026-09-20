"""Solve the same designs on every tier and store the result.

The expensive part of a multi-fidelity study is the fine-tier solves. Doing them
once and saving means every correction variant is analysed from the same
physics, and re-analysis costs seconds.

Two rules learned the hard way from the first attempt, which kept 4 of 48:

* a tier that fails is recorded as missing, never fatal to the row -- the beam
  and fine tiers are the study, the coarse tier is a question being asked;
* every failure is written down with its design, so "kept 19/24" can be
  explained instead of guessed at.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from dynopt.doe import sobol_sample
from dynopt.fea.calculix import CalculiXConfig, CalculiXSolver
from dynopt.fea.reference import ReferenceFrameSolver
from dynopt.geometry import CANTILEVER_ARM, MOTOR_MOUNT

N_PAIR, K = int(os.environ.get("N_PAIR", 24)), 6
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/tier_pairs.json"
CURV = int(os.environ.get("CURV", 12))
TIERS = (("coarse", float(os.environ.get("H_COARSE", 7.0))),
         ("fine", float(os.environ.get("H_FINE", 3.0))))


def ccx(part, h):
    # Deliberately no E/nu/rho: the part supplies them, so the 3D tier and the
    # beam tier cannot disagree about the material. Passing them here is what
    # made the carbon arm solve as aluminium.
    cfg = CalculiXConfig(n_modes=10, mesh_size=h, mesh_size_min=h / 4,
                         mesh_from_curvature=CURV)
    return CalculiXSolver(part, cfg)


data = {}
for part in (MOTOR_MOUNT, CANTILEVER_ARM):
    sp = part.design_space
    X = sobol_sample(sp, N_PAIR, seed=21)
    beam = ReferenceFrameSolver(part)
    rows = []
    print(f"=== {part.name} ===", flush=True)
    for i, x in enumerate(X):
        d = sp.to_dict(x)
        rec = {"x": x.tolist(), "i": i}
        try:
            t = time.perf_counter()
            r = beam.solve(d, n_modes=12)
            rec["beam"] = np.sort(r.frequencies)[:K].tolist()
            rec["beam_mass"] = float(r.mass)
            rec["beam_t"] = time.perf_counter() - t
        except Exception as e:
            print(f"  [{i}] beam FAILED {str(e)[:70]}", flush=True)
        for name, h in TIERS:
            try:
                t = time.perf_counter()
                r = ccx(part, h).solve(d, n_modes=10)
                if r.frequencies[0] < 1.0:
                    raise ValueError("rigid-body modes: model is disconnected")
                rec[name] = np.sort(r.frequencies)[:K].tolist()
                rec[name + "_t"] = time.perf_counter() - t
                rec[name + "_nodes"] = int(r.meta.get("n_nodes", 0))
                rec[name + "_h"] = float(r.meta.get("mesh_size", h))
            except Exception as e:
                rec.setdefault("failed", {})[name] = str(e)[:120]
                print(f"  [{i}] {name} FAILED: {str(e)[:70]}", flush=True)
        rows.append(rec)
        have = [t for t, _ in TIERS if t in rec]
        print(f"  [{i + 1}/{N_PAIR}] beam {rec.get('beam', [float('nan')])[0]:7.1f}"
              + "".join(f" | {t} {rec[t][0]:7.1f} ({rec[t + '_t']:4.1f}s,"
                        f" {rec[t + '_nodes']:6d}n, h={rec[t + '_h']:.2f})"
                        for t in have), flush=True)
    data[part.name] = rows
    full = sum(1 for r in rows if "beam" in r and "fine" in r)
    print(f"  beam+fine on {full}/{N_PAIR}; "
          f"coarse on {sum(1 for r in rows if 'coarse' in r)}/{N_PAIR}", flush=True)

json.dump(data, open(OUT, "w"))
print("saved", OUT, flush=True)
