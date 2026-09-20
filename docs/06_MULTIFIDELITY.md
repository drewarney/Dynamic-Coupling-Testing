# 6. Multi-fidelity correction: what it buys, and what it cannot

A cheap solver you cannot quote is only half useful. The beam tier runs in
~10 ms and the 3D tier in ~70 s, so the whole pipeline's economics turn on
whether a learned discrepancy model can make the cheap tier trustworthy.

This is the measurement. 24 Sobol designs per part, every design solved on all
three tiers, held-out evaluation against 3D truth. Reproduce with:

```bash
python examples/08_multifidelity.py pairs.json
```

## 6.1 The construction

Kennedy-O'Hagan, in the form the existing GP already wants. Frequencies are
positive and multiplicative and `FrequencySurrogate` already works in log
space, so the natural discrepancy is a log-ratio:

$$\log f_{\text{fine}}(x) = \log f_{\text{cheap}}(x) + r(x), \qquad r = \log\rho$$

`r` is fitted **only on paired designs**, where it is measured directly, so the
correction never learns the cheap surrogate's own interpolation error. Three
forms are implemented and which one wins is a measurement, not a preference:
`constant` (one scalar per column — the null hypothesis), `multiplicative` (GP
on the log-ratio), `additive` (GP on the difference).

## 6.2 Result 1 — do not correct the coarse 3D tier

| motor_mount, coarse → fine | error |
|---|---|
| uncorrected | **1.07%** |
| `constant` | 1.17% |
| `multiplicative` | 1.27% |
| `additive` | 1.57% |

**Every correction makes it worse.** The discrepancy is smaller than the noise
the correction introduces, so fitting one costs accuracy. The per-column ratio
is 0.982–1.000 with cv ≤ 0.021 — there is nothing there to learn.

The arm agrees in direction, from a slightly larger starting error (1.24% →
0.60% with a GP), but a 0.6-point gain for a tier that costs 7.1 s against the
fine tier's 16.1 s is not a trade worth making.

This is the same conclusion §1.5.2 reached from mesh economics, arrived at
independently: **coarse 3D is not a fidelity tier.** It is the same solve with
a worse answer.

## 6.3 Result 2 — the beam tier is learnable, and the GP earns its place

Held out, 14 pairs fitted, 10 designs tested:

| | uncorrected | `constant` | `multiplicative` | `additive` |
|---|---|---|---|---|
| `motor_mount` | 24.42% | 20.37% | 16.96% | **16.52%** |
| `cantilever_arm` | 20.97% | 13.21% | **4.91%** | 7.36% |

The GP beats the scalar decisively on both parts, which settles a claim this
repository has carried since the first cross-solver validation: *"the error is
design-dependent with both signs, so no scalar correction removes it."* That
was right about the scalar and wrong to stop there. A **design-dependent**
correction removes most of it.

## 6.4 Result 3 — equal budget, which is the only question that matters

Same solver-seconds either way: a GP fitted on *n* fine solves, against a beam
DOE corrected from *n* pairs. Held out on 8 designs.

**`motor_mount`** (fine 73 s/solve, beam 0.017 s):

| n fine | budget | fine-only GP | beam + correction |
|---|---|---|---|
| 4 | 292 s | 44.65% | **32.58%** |
| 8 | 583 s | 44.99% | **26.49%** |
| 12 | 874 s | 48.14% | **22.12%** |
| 16 | 1165 s | 20.57% | **17.69%** |

**`cantilever_arm`** (fine 16 s/solve, beam 0.008 s):

| n fine | budget | fine-only GP | beam + correction |
|---|---|---|---|
| 4 | 65 s | 46.00% | **16.05%** |
| 8 | 129 s | 10.63% | **7.83%** |
| 12 | 194 s | **6.41%** | 7.46% |
| 16 | 258 s | **7.39%** | 7.35% |

The correction wins where data is scarce and stops mattering once it is not.
On the arm the fine-only GP overtakes at 12 fine solves; on the mount it has
not overtaken by 16. **The correction buys roughly the first ten fine solves'
worth of accuracy, and nothing after that.**

That is a real result and a narrow one. It is most valuable exactly where you
would want it — early exploration, before you have spent an hour of solver
time — and it is not a substitute for fine-tier data once you have some.

## 6.5 Result 4 — how many pairs

`motor_mount`: `constant` is flat at ~22–28% from 3 pairs onward; the GP needs
8 pairs to beat it and plateaus at 12 (18.6% → 18.1% from 12 to 16).

`cantilever_arm`: `constant` is flat at ~14%; the GP separates at 8 pairs
(5.23%) and is still improving at 16 (4.08%).

Both plateau within the budget a real project would spend. The scalar's
flatness is the point: it converges immediately to an answer that is simply
not very good.

> **These mount numbers were re-measured.** The first run of this study gave
> 40.23% / 25.31% / 18.27%. That was against a beam tier with a stiffness
> defect, since fixed (§6.6), and re-running on the corrected tier moved every
> mount figure. The arm's are unchanged; its beam tier never had the defect.

## 6.6 What this does not fix

**The mount's corrected error is 16.5%. That is not quotable**, and the
correction does not change the rule that anything published goes through a 3D
sign-off (`examples/06_tier1_signoff.py`).

The two parts still differ by more than a factor of three in corrected error.
Across its design box the mount's beam-tier ratio runs **0.904 to 2.630** after
the fix below; the arm's runs **0.984 to 1.173** and never needed one.

**The mount's frame idealisation was not bad — one term in it was missing**,
and finding it was worth more than the entire correction.

The clue was that the mount agreed far better unloaded than loaded. Four
spokes carry the payload into the plate, and four narrow strips in bending
reproduce only a fraction of a plate's two-dimensional stiffness: measured
against plate theory, the idealisation was **11.4× too soft at the plate
centre**, which is exactly where the motor sits.

Equating the four-spoke central stiffness to plate theory gives the *form* —
effective width proportional to the platform span, which the code already
assumed — so only the coefficient was wrong. Simply-supported theory gives 5.7;
four discrete leg tops are softer than a continuous edge, so the coefficient is
calibrated against 24 mesh-converged 3D designs, bare and loaded, fitted on half
and reported on the other half:

| held-out first frequency | spoke factor 1.0 | **2.5** |
|---|---|---|
| bare structure | 10.1% | **7.5%** |
| loaded | 40.8% | **13.7%** |
| default design, loaded | −44.1% | **−14.9%** |
| 3D/beam ratio, median over the box | 1.532 | **1.057** |
| 3D/beam ratio, range | 1.029–4.170 | 0.904–2.630 |

One calibrated constant; `PLATE_WIDTH_FACTOR` stays at 1.0 because fitting it
as well bought no held-out accuracy, and a second fitted constant has to earn
its place.

### What that does to the case for a correction

| mount, beam → fine | before the plate fix | after |
|---|---|---|
| uncorrected | 40.23% | 24.42% |
| best correction | 18.27% | 16.52% |
| what the correction buys | 22.0 points | 7.9 points |

**Fixing the physics beat learning the discrepancy, and it shrank the
correction's job by two-thirds.** A learned discrepancy is worth having, but it
is second in line: it absorbs whatever a cheap model gets wrong, including
things that a few hours of modelling would remove outright. Before fitting a
correction, look for the missing term.

`examples/05_cross_solver_validation.py` prints the diagnostic that found this,
`tests/test_plate_calibration.py` owns the calibration, and `data/bare_refs.json`
holds the unloaded references it needs.

**An open question, stated rather than assumed.** Within a single mount design
the per-column ratios are not consistent. On the *pre-fix* tier, design 8 gave
`[4.17, 5.79, 4.12, 2.08, 1.40, 1.40]` and design 4 gave
`[1.03, 1.03, 1.26, 1.47, 2.77, 2.79]`. A smooth fidelity discrepancy should
not jump like that between adjacent columns. The likely cause is that sorted
column *k* of one tier is not the same mode as column *k* of the other, which
is exactly the hazard `MultiFidelitySurrogate.correspondence_report()` exists
to expose; the per-column cv is still 0.25–0.50 after the fix.

It is **not yet demonstrated**, but the plate fix added evidence for it. That
fix is a pure stiffness correction, so it should lift every column roughly
together. Held out, per column, it did not:

| column | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| improvement (points) | −29 | −19 | −19 | −14 | −10 | −10 |

Column 0 — where both tiers unambiguously mean "the lowest mode" — improved
three times as much as column 4. A stiffness error does not behave that way; a
column that is not comparing the same mode in both tiers does.

Settling it still needs 3D mode shapes stored alongside the frequencies and
MAC-based matching instead of sorted order. That is now cheaper than it was:
the CalculiX backend records `nodes`, `payload_nodes` and a `dof_map` as of the
backend-contract fix, so its mode shapes can finally be mapped back onto the
geometry at all.
