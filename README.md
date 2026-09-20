# dynopt — Automated Dynamic Coupling & Resonance Optimization

An end-to-end pipeline that takes parametric CAD, runs modal FEA, learns a
calibrated ML surrogate of the structure's dynamic response, and optimises the
geometry to move resonances out of an excitation band while minimising mass.

**Benchmark: a 4-leg motor mount for a 350 g BLDC motor running 1,000–6,000 RPM.**

All figures below are from the **3D solid solver** (Gmsh + CalculiX, quadratic
tets, straight-sided, mesh-converged to ±0.02%). The beam tier is used for exploration only;
see [Two tiers, and why the table is 3D](#two-tiers-and-why-the-table-is-3d).

| | Baseline | Optimised | Change |
|---|---|---|---|
| Mass | 502.9 g | **420.4 g** | **−16.4%** |
| Lowest natural frequency | 423.7 Hz | **1191.5 Hz** | **+181.2%** |
| Minimum band separation | 5.9% *(fails)* | **197.9%** | requirement: 25% |

The baseline's first mode sits 5.9% above the 4X excitation band — it does not
resonate, but it has nowhere near the 25% separation a rotating assembly needs,
so any tolerance stack or thermal shift pushes it in. The optimised design is
**16.4% lighter and clears every band by ≥197.9%**, because the two objectives
stopped conflicting once material was free to redistribute.

Cost: **64 beam-tier evaluations** to explore, **4 3D solves** to verify.

### Two tiers, and why the table is 3D

The beam tier runs in ~11 ms and makes the design-space exploration possible.
It is *not* quotable. Calibrated against seven mesh-converged 3D reference
points, its error is design-dependent with **both signs**, from −27.1% to
+4.8%, so no scalar correction removes it — a **design-dependent** one removes
most of it, which is a different claim and is measured in
[docs/06_MULTIFIDELITY.md](docs/06_MULTIFIDELITY.md):

| | Tier 0 (beam) | Tier 1 (3D solid) | Tier 0 error |
|---|---|---|---|
| Baseline f₁ | 372.6 Hz | 437.9 Hz | **−14.9%** |

(Both tier-0 figures above are after the plate-stiffness calibration; before it
the baseline read 244.6 Hz, −44.1%.)

Two physical causes, both measured: leg slenderness (error correlates with
`leg_thk` at r = −0.874; below L/d ≈ 4.5 beam theory stops representing a
member) and plate aspect ratio (r = −0.780).

**The plate correlation had a mechanism, and it has since been fixed.** The
mount agreed far better unloaded than loaded, which pointed at how the payload
was carried rather than at the frame. Four spokes reproduce a fraction of a
plate's two-dimensional stiffness — measured against plate theory the
idealisation was **11.4x too soft at the plate centre**, exactly where the motor
sits. Calibrating one effective width against 24 mesh-converged 3D designs took
the held-out loaded error from 40.8% to **13.7%** and the bare error from 10.1%
to **7.5%**; the table above is *after* that fix. See
[docs/06_MULTIFIDELITY.md](docs/06_MULTIFIDELITY.md) §6.6.

**The optimiser converged to L/d = 4.00 — outside the beam tier's validated
envelope.** That is not bad luck: raising frequency favours short stubby legs,
which is exactly where beam theory fails. An optimiser reliably walks into its
solver's blind spot, because the blind spot looks like cheap objective gain.
`tier0_validity()` flags such designs, and `examples/06_tier1_signoff.py`
re-solves them in 3D before any number is published.

Here the beam tier erred *conservatively*, so the design is safe. Had it erred
the other way, the beam-tier table would have shown a comfortable margin on a
design that genuinely violated it — and nothing in that tier would have
objected.

### Can a learned correction make the beam tier quotable?

Partly, and the honest answer is measured rather than asserted — 24 designs per
part, all tiers, held out against 3D truth:

| beam → 3D, held out | uncorrected | scalar | best correction |
|---|---|---|---|
| motor mount | 24.4% | 20.4% | **16.5%** |
| cantilever arm | 21.0% | 13.2% | **4.9%** |

At equal solver-seconds the correction wins while fine-tier data is scarce and
stops mattering once it is not: the arm's fine-only GP overtakes it at 12 fine
solves. Correcting the *coarse 3D* tier makes it worse — the discrepancy there
(1.07%) is smaller than the noise a correction introduces.

**Still not quotable**, so the 3D sign-off rule stands unchanged.

The more useful result came out of trying: the mount's beam tier had one
missing stiffness term, and fixing it cut the uncorrected error from 40.2% to
24.4% — shrinking what the correction had to do by two-thirds. *Fixing the
physics beat learning the discrepancy.* Full results, method and the open
questions: [docs/06_MULTIFIDELITY.md](docs/06_MULTIFIDELITY.md).

---

## Quick start

No licences, no commercial software, ~90 seconds:

```bash
pip install -e .
pytest -q                                   # 36 tests
python examples/01_modal_mvp.py             # modal analysis + mode classification
python examples/02_coupling_analysis.py     # cross-modal coupling + FRF
python examples/03_surrogate_training.py    # GP surrogate, held-out accuracy
python examples/04_full_optimization.py     # the full loop + before/after report
```

Or via the CLI:

```bash
dynopt modal --modes 8
dynopt coupling --damping 150 --threshold 0.3
dynopt optimize --rpm-lo 1000 --rpm-hi 6000 --orders 1 2 4 --margin 0.25
```

---

## Verification

The solver is checked against closed-form solutions before anything is built on
top of it (`tests/test_beam_validation.py`):

| Case | Closed form | Agreement |
|---|---|---|
| Cantilever bending, modes 1–3 | $f_n = \frac{\beta_n^2}{2\pi L^2}\sqrt{EI/\rho A}$ | **< 0.1%** |
| Fixed–fixed bending | $\beta = 4.7300,\ 7.8532$ | < 0.2% |
| Rod axial | $f_1 = \frac{1}{4L}\sqrt{E/\rho}$ | < 0.5% |
| Shaft torsion | $f_1 = \frac{1}{4L}\sqrt{G/\rho}$ | < 0.5% |
| Mass conservation | $r^T M r = \rho A L$ | < 1e-10 |
| Mesh convergence | monotone from above | ✓ |

---

## The core technical result

**The metric you validate a surrogate against decides which surrogate you build.**

The obvious target is per-mode frequency, labelled so the columns keep their
physical identity as geometry changes. That takes real machinery: modes cross,
symmetric structures have degenerate eigenvalues whose eigenvectors are an
arbitrary rotation within the eigenspace, and over a wide design space the
shapes drift so far that MAC against a reference design breaks down. Classifying
modes by rigid-body participation solves all of it and gets per-column error
down to ~0.5%.

It is also unnecessary. Measured against the plainest possible alternative —
the *k* lowest frequencies in sorted order, no labels, no tracking — on two
unrelated parts, same DOE, same GP, scored on the **FEA-verified objective of
the design each one actually produces**:

| Target | Motor mount | Cantilever arm |
|---|---|---|
| **sorted frequencies** | **0.2628** | **0.2629** |
| mode families | 0.2645 | 0.2632 |
| subspace-MAC clusters | 0.2647 | 0.2632 |
| objective scalars | 0.2703 | 0.2802 |

Sorted order wins on both. The reason is not accuracy — families predict *their
own columns* 2–3× better:

| Target | per-column nRMSE | **objective nRMSE** |
|---|---|---|
| sorted frequencies (arm) | 1.04% | **0.52%** |
| mode families (arm) | **0.43%** | **48.01%** |

**The ranking inverts.** On the arm, families are 2.4× more accurate per column
and 92× worse at predicting the objective. The objective reads the *whole*
spectrum — resonance risk sums over every mode, and the frequency floor is a
minimum over every mode — so a mode the classifier discards as `local` is
invisible to the surrogate and fully visible to the truth. Describing every mode
crudely beats naming a few precisely.

> **Correction.** An earlier version of this table claimed sorted-frequency
> targets "diverge". That row was never measured. Ordered eigenvalues of a
> symmetric pencil are *continuous* in the design parameters — a crossing kinks
> `f₍ₖ₎(x)`, it does not break it — and a GP fits that without difficulty. The
> 33% figure cited as justification came from vector-MAC tracking, which is
> genuinely poor; sorted order was never in that comparison.

Sorted frequencies are now the default. Mode classification is retained for
**reporting** a finished design, where naming the offending mode is exactly what
an engineer wants. Reproduce with
[`examples/07_target_comparison.py`](examples/07_target_comparison.py).

### Surrogate performance

Mode-family columns, 128 train / 200 test — still the most accurate *per-column*
predictions, and what the reporting path uses:

| Target | MAPE | p95 | 2σ coverage |
|---|---|---|---|
| `bounce_z` | 0.52% | 1.45% | 92% |
| `sway_rock_1` | 1.90% | 5.04% | 94% |
| `sway_rock_2` | 1.97% | 6.33% | 94% |
| `torsion_z` | 1.67% | 5.47% | 94% |
| mass | 0.153% | — | — |

Coverage near the nominal 95% means the uncertainty is **calibrated** — the
model knows what it doesn't know, which is what makes active learning valid.
Prediction runs at **0.018 ms vs 11 ms for FEA (~600×)**.

### Cross-modal coupling

Modal decoupling holds only for proportional damping. Real assemblies have local
dampers — bushings, bolted joints, bearing films — which are not proportional:

| Damping model | max $\kappa_{ij}$ | Interpretation |
|---|---|---|
| Rayleigh $C = \alpha M + \beta K$ | $4\times10^{-13}$ | decoupled to machine precision ✓ |
| Discrete dashpots at the payload | 0.62 | strongly coupled |

**Ignoring cross-modal coupling understates band-limited RMS response by 58.8%**
(and peak response by 80.4%) on the benchmark part.

This number is basis-invariant by construction. Getting it right required
canonicalising the eigenbasis inside degenerate subspaces first — without that
step the same code returned anywhere from −1.5% to −22.6% depending on which
LAPACK build ran it. See [`docs/01_THEORY.md`](docs/01_THEORY.md) §2.4.

*It read 22.6% until the plate-stiffness calibration
([docs/06](docs/06_MULTIFIDELITY.md) §6.6) moved the beam tier's modes from
244.6/264.9/600.1 Hz to 372.6/405.0/661.0 Hz. The effect is a property of this
part with this dashpot configuration (180 N·s/m and 6 N·m·s/rad at the
payload), not a universal constant —
`tests/test_dynamics.py::test_published_coupling_figure_is_current` owns it and
failed the moment the modes moved, which is what it is for.*

---

## Architecture

Three interchangeable fidelity tiers behind one `ModalSolver` interface:

```
Tier 0   reference beam FE (numpy)     ~11 ms    DOE, surrogate training, CI
Tier 1   Gmsh + CalculiX  (GPL)      ~10-60 s    shape verification, shippable
Tier 2   PyAnsys / MAPDL  (licensed) ~30-300 s   prestress, contact, sign-off
```

```python
solver = get_solver("reference", build_frame)       # no licence, runs in CI
solver = get_solver("calculix",  MotorMountCAD())   # full 3D, GPL only
solver = get_solver("ansys",     MotorMountCAD())   # prestressed, contact
```

```
params → CadQuery → STEP → Gmsh (C3D10) → CalculiX → ModalResult
                                                          ↓
                                          rigid-body mode classification
                                                          ↓
                                       GP surrogate (ARD Matérn-5/2, log-f)
                                                          ↓
                            differential evolution + FEA verification each round
                                                          ↓
                                                  optimised STEP
```

A MATLAB implementation of the same loop is in [`matlab/`](matlab/).

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/00_ARCHITECTURE.md`](docs/00_ARCHITECTURE.md) | Python vs MATLAB stack comparison; exact API hooks; licensing |
| [`docs/01_THEORY.md`](docs/01_THEORY.md) | Element matrices, participation, MAC, coupling, objective formulation; §2.3.1 on why the surrogate does not need mode labels |
| [`docs/02_ROADMAP.md`](docs/02_ROADMAP.md) | Phase-by-phase plan with acceptance tests and failure modes |
| [`docs/03_BENCHMARK.md`](docs/03_BENCHMARK.md) | Benchmark spec, results, **physical validation protocol** |
| [`docs/04_COMMERCIALIZATION.md`](docs/04_COMMERCIALIZATION.md) | Resume framing; honest SaaS assessment |
| [`docs/05_PRODUCT_ROADMAP.md`](docs/05_PRODUCT_ROADMAP.md) | What blocks generality (measured); the qual-testing end goal; next-session plan |
| [`docs/06_MULTIFIDELITY.md`](docs/06_MULTIFIDELITY.md) | Can a learned correction make the cheap tier quotable? Measured, both parts, including where the answer is no |

---

## Installation

```bash
pip install -e .                # core: numpy, scipy, scikit-learn
pip install -e ".[oss]"         # + CadQuery, Gmsh   (needs `ccx` on PATH)
pip install -e ".[ansys]"       # + PyAnsys          (needs an MAPDL licence)
pip install -e ".[dev,viz]"     # + pytest, matplotlib, pyvista
```

CalculiX is a system binary: `apt install calculix-ccx`, `brew install calculix`,
or the `ccx` executable from calculix.de.

---

## Limitations

Stated plainly, because they matter:

- **The benchmark optimum sits on the `span` and `height` lower bounds.** The
  constraint is active, so the unconstrained optimum lies outside the stated
  design space. Real bounds come from packaging; widening them is the obvious
  next study.
- **Tier 0 is a beam idealisation.** Good for slender frames, wrong for bulky
  solids and stress concentrations. Use Tier 1/2 for shape verification.
- **Linear elastic, small deflection.** No contact, plasticity, or geometric
  nonlinearity.
- **Joints are rigid.** Bolted joints are softer in reality; expect measured
  frequencies below predicted until you calibrate joint stiffness.
- **No manufacturing or tolerance constraints.** The optimiser can return a
  design that is a knife edge away from a resonance. Robust optimisation over a
  tolerance band is the most valuable next feature.
- **No stress or fatigue constraint** — a modal-optimal design is not
  automatically strong enough.
- **Ansys academic licences forbid commercial use.** The shippable path is
  Tier 0 + Tier 1 (all GPL/BSD).

---

## License

MIT for this project's own code. Optional backends carry their own licences —
Gmsh and CalculiX are GPL and are invoked as subprocesses, which keeps the
copyleft boundary clean. See [`docs/00_ARCHITECTURE.md`](docs/00_ARCHITECTURE.md) §1.4.
