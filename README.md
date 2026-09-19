# dynopt — Automated Dynamic Coupling & Resonance Optimization

An end-to-end pipeline that takes parametric CAD, runs modal FEA, learns a
calibrated ML surrogate of the structure's dynamic response, and optimises the
geometry to move resonances out of an excitation band while minimising mass.

**Benchmark: a 4-leg motor mount for a 350 g BLDC motor running 1,000–6,000 RPM.**

| | Baseline | Optimised | Change |
|---|---|---|---|
| Mass | 540.9 g | **465.7 g** | **−13.9%** |
| Lowest natural frequency | 242.4 Hz | **798.7 Hz** | **+229.5%** |
| Minimum band separation | *violated* | **99.7%** | requirement: 25% |
| Constraint violation | 1.127 | **0.000** | feasible |

Achieved in **64 FEA evaluations / 93 seconds** — roughly 9× fewer solver calls
than direct gradient optimisation on the same problem.

The baseline mount resonates at 3,636 RPM (4X crossing of a 242 Hz mode carrying
69% of the vertical effective mass) — squarely mid-operating-range. The
optimised design is **lighter *and* clear of every excitation band**, because
the two objectives stopped conflicting once material was allowed to redistribute.

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

**Defining the surrogate's regression target is worth more than the choice of ML model.**

Naively regressing frequencies in sorted eigenvalue order fails, for three
compounding reasons: modes cross as geometry changes; symmetric structures have
*degenerate* eigenvalues whose eigenvectors are an arbitrary rotation within the
eigenspace; and over a wide design space the shapes drift so far that MAC
against a reference design breaks down entirely.

The fix is to stop matching shapes and classify each mode by what it physically
**does** — projecting onto the six rigid-body motions of the payload and scoring
by *DOF group* rather than by individual component:

| Target definition | Held-out MAPE |
|---|---|
| Sorted frequency order | diverges |
| Vector MAC tracking | 33% |
| Subspace-MAC cluster tracking | 31–103% |
| Argmax rigid-body classification | 0.5–50% |
| **Group-energy classification** | **0.52–1.97%** |

Same solver, same GP, same DOE. Detail in [`docs/01_THEORY.md`](docs/01_THEORY.md) §2.3.

### Surrogate performance (128 train / 200 test)

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
| Discrete dashpots at the payload | 1.00 | strongly coupled |

**Ignoring cross-modal coupling understates band-limited RMS response by 17.8%**
on the benchmark part.

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
| [`docs/01_THEORY.md`](docs/01_THEORY.md) | Element matrices, participation, MAC, coupling, objective formulation |
| [`docs/02_ROADMAP.md`](docs/02_ROADMAP.md) | Phase-by-phase plan with acceptance tests and failure modes |
| [`docs/03_BENCHMARK.md`](docs/03_BENCHMARK.md) | Benchmark spec, results, **physical validation protocol** |
| [`docs/04_COMMERCIALIZATION.md`](docs/04_COMMERCIALIZATION.md) | Resume framing; honest SaaS assessment |

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
