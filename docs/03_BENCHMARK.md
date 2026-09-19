# 4. Real-World Benchmark: Proof of Concept

## 4.1 The test case

**A four-leg motor mount carrying a 350 g BLDC motor, operating 1,000–6,000 RPM.**

Chosen deliberately over an engine bracket or a drone arm because it is the
smallest problem that is still *genuinely hard*:

- **The excitation sweeps.** 1,000–6,000 RPM at orders 1X/2X/4X spans
  16.7–400 Hz — a wide continuous band, not a single frequency. You cannot
  dodge it by tuning to one number.
- **It has real modal structure.** Four families (bounce, two sway–rock
  branches, torsion), with genuine degenerate pairs from the 4-fold symmetry.
- **The objectives genuinely conflict.** Stiffening to raise frequency adds
  mass; lightening drops modes into the excitation band.
- **It is physically testable for under $100** — which is the part most
  portfolio projects skip, and the part that makes yours credible.

### Specification

| Parameter | Value |
|---|---|
| Payload | 350 g motor, $I = (2.0, 2.0, 3.5)\times10^{-4}$ kg·m² |
| Material | AL6061-T6 ($E$ = 68.9 GPa, $\rho$ = 2700 kg/m³, $\nu$ = 0.33) |
| Operating range | 1,000–6,000 RPM |
| Excitation orders | 1X, 2X, 4X (4-blade propeller) |
| Avoid bands | 16.7–100 / 33.3–200 / 66.7–400 Hz |
| Required separation | 25% |
| Design variables | leg thickness, leg diameter, base span, height, plate thickness |
| Objective | minimise mass subject to resonance clearance |

## 4.2 Baseline: why this design fails

**Tier 1 (3D solid, mesh-converged to −0.02%):** 502.9 g, first mode 423.7 Hz.

| mode | freq | min separation | required | inside a band? |
|---|---|---|---|---|
| 1 | 423.7 Hz | **5.9%** | 25% | no |
| 2 | 570.1 Hz | 42.5% | 25% | no |
| 3 | 570.5 Hz | 42.6% | 25% | no |
| 4 | 996.1 Hz | 149.0% | 25% | no |
| 5 | 1001.8 Hz | 150.4% | 25% | no |
| 6 | 1544.7 Hz | 286.2% | 25% | no |

The baseline **fails the margin requirement without actually resonating.** Its
first mode sits 5.9% above the 4X band top (400 Hz at 6,000 RPM) — no mode lies
inside any excitation band. Six percent is not engineering margin: a tolerance
stack, a temperature change or a bolt-preload shift moves it in. Rotating
assemblies are specified at 20–25% for exactly this reason.

> **Correction.** An earlier version of this document reported a 4X Campbell
> crossing at 3,636 RPM and described the baseline as resonating mid-range.
> That was a Tier 0 artifact: the beam tier put the first mode at 242.4 Hz,
> which *does* cross 4X. The 3D solver puts it at 423.7 Hz, which does not.
> The baseline is marginal, not resonant. The engineering conclusion —
> this design needs fixing — is unchanged; the reason is different.

## 4.3 Result

```
python examples/04_full_optimization.py --save optimum.json   # Tier 0, explores
python examples/06_tier1_signoff.py optimum.json              # Tier 1, verifies
```

**Quotable figures, 3D solid solver, straight-sided quadratic tets,
mesh-converged to ±0.02%:**

| | Baseline | Optimised | Change |
|---|---|---|---|
| Mass | 502.9 g | **420.4 g** | **−16.4%** |
| Structural mass | 152.9 g | 70.4 g | −54.0% |
| f₁ | 423.7 Hz | **1191.5 Hz** | **+181.2%** |
| Min band separation | 5.9% *(fail)* | **197.9%** | requirement 25% |

Per-mode verification on the 3D solver:

| mode | freq | separation | status |
|---|---|---|---|
| 1 | 1191.5 Hz | 197.9% | PASS |
| 2 | 1191.8 Hz | 198.0% | PASS |
| 3 | 1812.2 Hz | 353.1% | PASS |
| 4 | 1933.2 Hz | 383.3% | PASS |
| 5 | 3129.3 Hz | 682.3% | PASS |
| 6 | 3137.0 Hz | 684.3% | PASS |

**Lighter *and* dramatically safer** — the objectives were not truly in conflict
once the geometry could redistribute material.

### Mesh convergence

Optimised design:

| h | nodes | f₁ | Δ |
|---|---|---|---|
| 3.0 mm | 24,450 | 1192.0 Hz | — |
| 2.4 mm | 36,943 | 1191.4 Hz | −0.05% |
| 1.9 mm | 57,364 | 1191.5 Hz | +0.01% |

Baseline:

| h | nodes | f₁ | Δ |
|---|---|---|---|
| 3.0 mm | 47,891 | 424.3 Hz | — |
| 2.4 mm | 71,775 | 423.8 Hz | −0.12% |
| 1.9 mm | 111,347 | 423.7 Hz | −0.02% |

### Cost

| Metric | Value |
|---|---|
| Tier 0 evaluations (exploration) | 64 |
| Tier 1 solves (verification) | 6 |
| Tier 0 wall time | 93 s |
| Tier 1 wall time | ~3 min |

### Optimised geometry

| Variable | Baseline | Optimised |
|---|---|---|
| `leg_thk` | 2.000 mm | 1.000 mm ← at bound |
| `leg_dia` | 14.000 mm | 9.838 mm |
| `span` | 70.000 mm | 40.000 mm ← at bound |
| `height` | 60.000 mm | 30.000 mm ← at bound |
| `plate_thk` | 3.000 mm | 7.687 mm |

Shorten and squat the legs (bending frequency ∝ 1/L²), thin the walls to
recover mass, thicken the plate — which the GP sensitivity analysis
independently identified as the dominant lever.

> **Three bounds are active** (`leg_thk`, `span`, `height`). The true optimum
> lies outside the stated design space, so this is the right answer to the
> stated problem, not an interior optimum. Widening the bounds is the obvious
> next study. In a real programme those bounds come from packaging and the
> bolt pattern.

> **The optimum sits outside Tier 0's validated envelope** (L/d = 4.00 against a
> 4.5 floor), which is why the 3D sign-off is mandatory rather than advisory.
> Tier 0 under-predicted its first mode by 31.2%.

## 4.4 Physical validation protocol

**This is the step that separates a portfolio project from a homework
assignment.** Budget one weekend and ~$100.

### Bill of materials

| Item | Cost |
|---|---|
| 3D-printed or CNC mount (both designs) | $20–60 |
| ADXL1002 breakout accelerometer (±50 g, 11 kHz) | $30 |
| Modal hammer — a small instrumented hammer, or a nylon-tipped hobby hammer | $0–250 |
| USB DAQ / any ≥ 20 kS/s audio interface | $25 |

Print in aluminium-filled PLA or CNC in 6061 if you have shop access. If you
print, **measure the actual $E$ and $\rho$ of your print** with a simple
cantilever test and feed those into the model — print anisotropy is large and
you will otherwise spend a day chasing a material error.

### Procedure

1. **Bolt the mount to a rigid base.** The base must be ≫ stiffer than the part;
   a granite surface plate or a thick steel block. A bench top is not rigid.
2. **Mount the accelerometer at the motor location**, sensing Z. Record its mass
   and add it as a point mass in the model — a 5 g sensor on a 350 g payload
   shifts $f_1$ by ~1%, but on a bare 190 g structure it shifts it by ~4%.
3. **Roving-hammer impact test.** Fix the response point, strike at 8–12 points,
   average 5 impacts each. Reject double-hits.
4. **Compute the FRF** and extract peaks:
   ```python
   from scipy.signal import csd, welch
   f, Pxy = csd(force, accel, fs=25600, nperseg=8192)
   _, Pxx = welch(force, fs=25600, nperseg=8192)
   H1 = Pxy / Pxx                      # H1 estimator, best with output noise
   ```
5. **Damping from half-power bandwidth:**
   $\zeta = \dfrac{f_2 - f_1}{2f_n}$ at the −3 dB points. Feed the measured
   $\zeta$ back as the modal damping for your FRF predictions.
6. **Correlate** predicted vs measured with MAC if you measure enough points.

### Acceptance criteria

| Metric | Target | Interpretation |
|---|---|---|
| $f_1$ error | < 10% | Good for a bolted assembly |
| $f_1$ error | < 5% | Excellent; your BCs are right |
| $f_{2,3}$ error | < 15% | Higher modes are BC-sensitive |
| MAC (if measured) | > 0.8 | Shapes agree, not just frequencies |

### When it disagrees — diagnose, don't fudge

| Symptom | Likely cause |
|---|---|
| Measured **lower** than predicted | Joint compliance; bolted joints are softer than bonded. Add a joint stiffness or fix a smaller footprint. |
| Measured **higher** than predicted | Bolt preload stiffening. Run prestressed modal (`PSTRES,ON`). |
| Split peak where one mode expected | Real degeneracy broken by manufacturing asymmetry — physically interesting, worth a paragraph. |
| All frequencies ~30× off | Unit error. mm/tonne/s vs SI. |
| Everything 10–25% high | Linear tets. Switch to order 2. |

**Write up whatever you actually get.** A documented 12% disagreement with a
correct diagnosis is a far stronger portfolio signal than an unexplained 2%
match — the first shows engineering judgement, the second looks like luck or
tuning.

## 4.5 What to put in the report

1. Problem statement with the Campbell diagram showing the baseline crossings.
2. Method: the three-tier architecture and *why*.
3. Verification: the analytical test table (< 0.1%).
4. Coupling analysis: the 29.9% RMS result, with the basis-invariance check
   that makes it trustworthy.
5. Surrogate validation: MAPE, p95, **and coverage**.
6. Optimisation result: the before/after table.
7. Physical validation: measured vs predicted, with diagnosis.
8. **Limitations**, stated plainly: active bounds, beam idealisation at Tier 0,
   linear-elastic small-deflection assumption, no thermal effects, joints
   modelled as rigid.

Section 8 is the one that makes experienced engineers take you seriously.
