#!/usr/bin/env python3
"""Can a learned correction make the cheap tier trustworthy?

The beam tier runs in ~10 ms and measured 31% off against 3D on the design the
optimiser returned, so its numbers are fast and unquotable. The 3D tier is right
and costs a minute a solve. Multi-fidelity asks whether a discrepancy model,
fitted on a handful of designs solved on both tiers, buys the accuracy of the
second at the price of the first.

Three questions, kept separate because they measure different things:

  (A) **correction quality** -- given a *measured* cheap-tier result, how close
      does the correction get to fine-tier truth? Isolates the discrepancy
      model from the cheap surrogate's own error.
  (B) **equal budget** -- spend the same solver-seconds on fine solves either
      way. Does (cheap DOE + correction from n pairs) beat (a GP fitted on n
      fine solves)? This is the only question that decides whether any of this
      is worth having.
  (C) **how many pairs** before the correction stops improving.

Needs the paired dataset from the generator; pass its path.

Run:  python examples/08_multifidelity.py pairs.json
"""

import sys as _sys
from pathlib import Path as _Path

# Run straight from a checkout, with or without an editable install.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import json
import sys
import warnings

import numpy as np

from dynopt.geometry import get_part
from dynopt.surrogate import MultiFidelitySurrogate
from dynopt.surrogate.gp import FrequencySurrogate

warnings.filterwarnings("ignore")

MODES = ("constant", "multiplicative", "additive")
TIERS = ("beam", "coarse", "fine")


def load(path):
    """Rows keep whatever tiers succeeded, so every pairing gets its own mask."""
    out = {}
    for name, rows in json.load(open(path)).items():
        if not rows:
            continue
        d = {"X": np.array([r["x"] for r in rows])}
        for tier in TIERS:
            have = np.array([tier in r for r in rows])
            d[tier] = {
                "have": have,
                "F": np.array([r[tier] for r in rows if tier in r]),
                "t": float(np.mean([r[f"{tier}_t"] for r in rows if tier in r]))
                if have.any() else float("nan"),
                "nodes": float(np.mean([r.get(f"{tier}_nodes", 0)
                                        for r in rows if tier in r]))
                if have.any() else float("nan"),
            }
        out[name] = d
    return out


def paired(d, cheap_key):
    """Designs where both the cheap tier and the fine tier succeeded."""
    both = d[cheap_key]["have"] & d["fine"]["have"]
    idx_c = np.cumsum(d[cheap_key]["have"]) - 1
    idx_f = np.cumsum(d["fine"]["have"]) - 1
    return (d["X"][both],
            d[cheap_key]["F"][idx_c[both]],
            d["fine"]["F"][idx_f[both]])


def rel_err(pred, truth):
    m = np.isfinite(pred) & np.isfinite(truth) & (truth > 0)
    return float(np.mean(np.abs(pred[m] - truth[m]) / truth[m])) if m.any() else np.nan


def _apply(mf, space, Fc, X, mode):
    """Correction applied to *measured* cheap values -- no cheap surrogate."""
    Xs = np.array([space.normalize(x) for x in X])
    out = np.zeros_like(Fc)
    for j in range(Fc.shape[1]):
        gp = mf.corrections[j]
        c = gp.predict(Xs) if gp is not None else np.full(len(X), mf.constants[j])
        out[:, j] = (Fc[:, j] + c) if mode == "additive" else (Fc[:, j] * np.exp(c))
    return out


def study_correction(name, X, Fc, Ff, cheap_key, n_train):
    space = get_part(name).design_space
    tr, te = np.arange(n_train), np.arange(n_train, len(X))
    if te.size < 4:
        print(f"\n  {cheap_key} -> fine: only {len(X)} pairs, not enough to hold out")
        return
    print(f"\n  (A) {cheap_key} -> fine   "
          f"[{n_train} pairs fitted, {te.size} held out]")
    print(f"      uncorrected                     {100*rel_err(Fc[te], Ff[te]):7.2f}%")
    mf = None
    for mode in MODES:
        mf = MultiFidelitySurrogate(space, mode=mode)
        mf.fit(X[tr], Fc[tr], X[tr], Fc[tr], Ff[tr])
        e = rel_err(_apply(mf, space, Fc[te], X[te], mode), Ff[te])
        print(f"      {mode:<16s}              {100*e:7.2f}%")

    rep = mf.correspondence_report()
    cv = [r.get("ratio_cv", float("nan")) for r in rep]
    print(f"      per-column ratio  {np.round([r.get('ratio_mean', np.nan) for r in rep], 3).tolist()}")
    print(f"      ratio spread      cv = {np.round(cv, 3).tolist()}")
    print(f"                        -> {'a scalar suffices' if np.nanmax(cv) < 0.05 else 'varies with design: a GP has something to learn'}")


def study_budget(name, d, X, Fc, Ff, cheap_key, t_cheap, t_fine):
    """(B) The deciding comparison: same fine-solve budget, two ways to spend it."""
    space = get_part(name).design_space
    n = len(X)
    n_hold = max(6, n // 3)
    hold = np.arange(n - n_hold, n)
    pool = np.arange(n - n_hold)
    print(f"\n  (B) equal budget, {cheap_key} -> fine   "
          f"[{n_hold} held out; fine {t_fine:.0f}s, {cheap_key} "
          f"{t_cheap:.3g}s per solve]")
    print("      n_fine   budget    fine-only GP   cheap+correction")
    for k in (4, 6, 8, 12, 16):
        if k > pool.size:
            break
        tr = pool[:k]
        budget = k * t_fine + pool.size * t_cheap

        fine_only = FrequencySurrogate(space, n_modes=Ff.shape[1]).fit(X[tr], Ff[tr])
        e_fine = rel_err(np.atleast_2d(fine_only.predict(X[hold])), Ff[hold])

        mf = MultiFidelitySurrogate(space, mode="multiplicative")
        mf.fit(X[pool], Fc[pool], X[tr], Fc[tr], Ff[tr])
        e_mf = rel_err(np.atleast_2d(mf.predict(X[hold])), Ff[hold])

        print(f"      {k:>6d}   {budget:6.0f}s   {100*e_fine:11.2f}%   {100*e_mf:14.2f}%")


def study_pairs_needed(name, X, Fc, Ff, cheap_key):
    """(C) Where the learning curve flattens."""
    space = get_part(name).design_space
    n = len(X)
    n_hold = max(6, n // 3)
    hold = np.arange(n - n_hold, n)
    pool = np.arange(n - n_hold)
    print(f"\n  (C) pairs needed, {cheap_key} -> fine   [{n_hold} held out]")
    print("      n_pairs   constant   multiplicative")
    for k in (3, 4, 6, 8, 12, 16):
        if k > pool.size:
            break
        tr = pool[:k]
        line = f"      {k:>7d}"
        for mode in ("constant", "multiplicative"):
            mf = MultiFidelitySurrogate(space, mode=mode)
            mf.fit(X[tr], Fc[tr], X[tr], Fc[tr], Ff[tr])
            line += f"   {100*rel_err(_apply(mf, space, Fc[hold], X[hold], mode), Ff[hold]):7.2f}%"
        print(line)


def main(path) -> int:
    data = load(path)
    print("=" * 76)
    print("  MULTI-FIDELITY CORRECTION")
    print("=" * 76)

    for name, d in data.items():
        n = len(d["X"])
        print(f"\n### {name}   ({n} designs sampled)")
        for tier in TIERS:
            t = d[tier]
            if not t["have"].any():
                print(f"  {tier:7s} no successful solves")
                continue
            cost = (f"{t['t']*1000:7.0f} ms" if t["t"] < 1.0
                    else f"{t['t']:7.1f} s ")
            nodes = f"{t['nodes']:8.0f} nodes" if t["nodes"] else "   (no mesh)"
            print(f"  {tier:7s} {t['have'].sum():2d}/{n} solved | {cost} | {nodes}")

        for cheap_key in ("beam", "coarse"):
            X, Fc, Ff = paired(d, cheap_key)
            if len(X) < 10:
                print(f"\n  {cheap_key} -> fine: {len(X)} usable pairs, skipped")
                continue
            study_correction(name, X, Fc, Ff, cheap_key,
                             n_train=max(8, int(0.6 * len(X))))
            study_budget(name, d, X, Fc, Ff, cheap_key,
                         d[cheap_key]["t"], d["fine"]["t"])
            study_pairs_needed(name, X, Fc, Ff, cheap_key)

    print("\n" + "=" * 76)
    print("  (A) says whether the discrepancy is learnable at all.")
    print("  (B) is the one that decides: if the fine-only GP wins at equal")
    print("      budget, the correction is machinery that costs accuracy.")
    print("  If a scalar matches the GP in (C), ship the scalar -- it cannot")
    print("  overfit a dozen points and it extrapolates predictably.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
