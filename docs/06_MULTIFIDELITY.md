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
| `motor_mount` | 40.23% | 25.31% | **18.27%** | 21.23% |
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

## 6.6 What this does not fix

**The mount's corrected error is 18%. That is not quotable**, and the correction
does not change the rule that anything published goes through a 3D sign-off
(`examples/06_tier1_signoff.py`).

The two parts differ by a factor of four in corrected error. Across its design
box the mount's beam-tier ratio runs **1.029 to 4.170**; the arm's runs **0.984
to 1.173**.

**It is not that the mount's frame idealisation is bad.** Running the mount
bare — no payload — the two tiers agree to **0.8%** (beam 1262.2 Hz, 3D
1252.0 Hz). The frame, the legs and the plate are all represented well. The
error appears only when the motor goes on:

| motor_mount, default design | beam | 3D | |
|---|---|---|---|
| bare structure | 1262.2 Hz | 1252.0 Hz | **+0.8%** |
| with 350 g payload | 244.6 Hz | 437.9 Hz | **−44.1%** |
| effect of adding the payload | ÷5.16 | ÷2.86 | |

The beam tier is nearly twice as sensitive to the payload as the solid is,
because a point mass at the plate centre sits on four edge beams with no
membrane action to resist it. The plate is locally far too soft, and that
softness scales with plate thickness and span — which is why the earlier
calibration found the error correlating with plate aspect ratio at r = −0.780.
That correlation now has a mechanism.

So the honest version of the rule is narrower and more useful than "the
idealisation is wrong":

> A learned discrepancy sharpens a cheap model that is already structurally
> right. Where it struggles, look for a *specific* missing physics before
> concluding the tier is unusable — the mount's 18% residual is one modelling
> gap, not a general failure of beam theory.

`examples/05_cross_solver_validation.py` prints this diagnostic, and
`tests/test_solver_hints.py::test_the_beam_tier_is_sound_until_it_is_loaded`
owns the numbers. Closing the gap — an effective plate stiffness under a
central mass, or a shell element for the plate — is the highest-value follow-up
this study produced, and it is worth more than any further work on the
correction itself.

**An open question, stated rather than assumed.** Within a single mount design
the per-column ratios are not consistent — design 8 gives
`[4.17, 5.79, 4.12, 2.08, 1.40, 1.40]`, design 4 gives
`[1.03, 1.03, 1.26, 1.47, 2.77, 2.79]`. A smooth fidelity discrepancy should
not jump like that between adjacent columns. The likely cause is that sorted
column *k* of one tier is not the same mode as column *k* of the other, which
is exactly the hazard `MultiFidelitySurrogate.correspondence_report()` exists
to expose, and the mount's per-column cv of up to 0.64 is consistent with it.

It is **not yet demonstrated**. Splitting the designs by `tier0_validity()` did
not separate the effect (inside the envelope the corrected error is 19.5–21.1%
against 18.1% for all designs), but with 11 and 13 designs per group that test
is underpowered rather than negative. Settling it needs 3D mode shapes stored
alongside the frequencies and MAC-based matching instead of sorted order —
which is the natural next piece of work, and would say whether the mount's
residual 18% is idealisation error or bookkeeping.
