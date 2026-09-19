# 2. Step-by-Step Roadmap: MVP → Business

Each phase has a **deliverable**, an **acceptance test**, and a **failure mode
to watch for**. Do not advance until the acceptance test passes — the failure
modes compound viciously.

---

## Phase 1 — Core Pipeline MVP  *(week 1–2)*

**Goal:** parameters in, verified natural frequencies out.

### Steps

1. **Define the design space before touching CAD.** 5–8 continuous variables
   with physical bounds. More than ~10 and the DOE cost explodes; fewer than 4
   and there is nothing to optimise.

   ```python
   MOTOR_MOUNT_SPACE = DesignSpace([
       DesignVariable("leg_thk",  0.0010, 0.0060, 0.0020, "m", "leg wall thickness"),
       DesignVariable("leg_dia",  0.0080, 0.0250, 0.0140, "m", "leg outer diameter"),
       ...
   ])
   ```

2. **Build the Tier-0 reference solver first.** Before any CAD or mesher. It
   must be verifiable in closed form. This feels like a detour; it is the thing
   that makes every later number trustworthy.

3. **Verify against analytical solutions.** Cantilever, fixed–fixed, axial,
   torsional, mesh convergence, mass conservation. Target < 0.1%.

4. **Then** add CAD (CadQuery → STEP) and Tier 1 (Gmsh → CalculiX) behind the
   same `ModalSolver` interface.

5. **Cross-validate the tiers.** Tier 0 and Tier 1 should agree within ~5% on
   the first few modes. Systematic disagreement means a boundary condition or
   unit mismatch, not "beam theory is approximate".

### Acceptance test
```bash
pytest tests/test_beam_validation.py -q     # 10 passed
python examples/01_modal_mvp.py             # frequencies + effective mass
```

### Failure modes
- **Linear tets.** C3D4/SOLID185 report frequencies 10–25% high. Always order 2.
- **Unit mismatch.** mm geometry with SI density → wrong by √1000 ≈ 31×.
- **Over-constrained BCs.** Clamping a whole face instead of the bolt footprint
  can raise $f_1$ by 20–40%.
- **Single element per member.** One beam element per leg over-stiffens by
  several percent. Use ≥ 4.

---

## Phase 2 — Dynamic Coupling Analysis  *(week 3–4)*

**Goal:** go beyond "here are the frequencies" to "here is which modes talk to
each other, and what the structure actually does under excitation."

This phase is the project's **differentiator**. Anyone can run a modal analysis
in Workbench. Programmatic cross-modal coupling is what makes it a tool.

### Steps

1. **Extract M and K**, not just eigenvalues. CalculiX `*MATRIX OUTPUT`;
   MAPDL `HBMAT` in `/AUX2`. Without these there is no coupling analysis.

2. **Build a realistic damping matrix.** Rayleigh first as the decoupled
   baseline, then discrete dashpots at the isolator/bolt locations — that is
   where non-proportionality physically comes from.

3. **Compute the coupling metrics:** $\tilde{C} = \Phi^T C \Phi$, then
   $\kappa_{ij}$ and $\eta_{ij}$ (see `docs/01_THEORY.md` §2.4).

4. **Prove the coupling matters** by running the FRF both ways —
   `keep_coupling=True/False`. If the two curves overlay, say so; if they do
   not, you have a quantified result.

5. **Campbell diagram** against the engine orders over the RPM range.

### Acceptance test
```bash
python examples/02_coupling_analysis.py
```
Expect: Rayleigh $\kappa \approx 10^{-13}$ (decoupled, as theory says),
discrete dampers $\kappa \approx 1.0$, and a quantified FRF difference.

Benchmark result: **ignoring cross-modal coupling understates band RMS
response by 29.9%, and peak response by 49.4%.** That one sentence is the core
of your project report.

**Before you quote any coupling number, check it is basis-invariant.** Re-run
with the degenerate mode shapes rotated within their own eigenspace (still a
valid eigensolution). If the answer moves, you are measuring your LAPACK build,
not your structure — see `tests/test_dynamics.py::test_uncoupled_frf_is_basis_invariant`.

### Failure modes
- **Assuming modal damping.** `zeta=0.02` everywhere hides exactly the effect
  you are trying to measure.
- **Too few retained modes.** Check effective mass sums to ≥ 90% in the
  excitation direction before trusting a mode-superposition FRF.
- **Reading $\kappa$ without $\Delta$.** Two modes 3 kHz apart with $\kappa=0.9$
  do not meaningfully interact. Always use $\eta$.

---

## Phase 3 — The AI / Optimisation Layer  *(week 5–8)*

**Goal:** frequency prediction in < 1 ms, and a closed optimisation loop.

### Steps

1. **Sobol DOE, 100–300 points.** Sobol over uniform random: lower discrepancy
   means the GP sees the corners instead of clustering.

2. **Define the regression target correctly. This is the whole ball game.**
   Not `frequencies[j]`. Classify modes by rigid-body family and regress
   `family_frequencies()`. See §2.3 of the theory doc — this choice is worth
   **50% → 1.9% MAPE**, same solver, same GP, same DOE.

3. **Fit GPs on $\log f$ with an ARD Matérn-5/2 kernel**, unit-cube inputs.

4. **Report held-out accuracy honestly.** MAPE, p95 *and* the 2σ coverage. A
   surrogate whose error bars are wrong is worse than no surrogate, because
   the optimiser will trust it.

5. **Optimise with EGO, not one-shot.** Alternate global search on the
   surrogate with targeted FEA at the proposed optimum, refitting each round.
   **Always return the best FEA-verified point, never a surrogate prediction.**

6. **Use the learned length scales** as a free sensitivity analysis. On the
   benchmark: `plate_thk` 0.49, `span` 0.21, `leg_dia` 0.17, `leg_thk` 0.08,
   `height` 0.05 — i.e. plate thickness dominates the bounce mode.

### Acceptance test
```bash
python examples/03_surrogate_training.py              # < 2% MAPE, ~90% coverage
python examples/04_full_optimization.py --save o.json # beats baseline
python examples/06_tier1_signoff.py o.json            # MUST pass before quoting
```

**A cheap solver's optimum is not a result.** The optimiser will find the
region where your fast model is most wrong, because that is where the objective
looks cheapest to improve. On this benchmark it converged to a leg slenderness
of L/d = 4.00, below the beam tier's validated floor of 4.5, and the beam tier
under-predicted that design's first mode by 31.2%. Verify every published
design on the highest-fidelity solver you have.

### Why a GP and not a neural net

At 100–300 points in 5–10 dimensions, a GP is simply correct: it interpolates
exactly (FEA is deterministic — there is no observation noise to average out),
it trains in seconds, and it gives a **calibrated variance**. Reach for a
PyTorch GNN only when the *topology* changes between designs (mesh in,
frequencies out) — for a fixed parametric family that is more machinery for
less accuracy. XGBoost is a reasonable third option if you have > 2,000 samples
and no need for uncertainty.

### Failure modes
- **Regressing raw sorted frequencies.** The single most common mistake in
  surrogate-based structural optimisation. Guarantees failure and looks like
  "ML doesn't work here."
- **Trusting the surrogate optimum without verification.** The optimum sits
  where the GP is least certain almost by construction.
- **Optimising into a knife edge.** A design 0.1 mm from a resonance is not
  manufacturable. Enforce a margin (20–25%) and sanity-check against tolerances.

---

## Phase 4 — Report, Portfolio, Commercialisation  *(week 9–12)*

**Goal:** turn working code into evidence of capability, then into an offer.

### Steps

1. **Run the benchmark end to end** (`docs/03_BENCHMARK.md`) and write the
   report: problem → method → verification → result → limitations.

2. **Physical validation if you can.** A $40 accelerometer and a modal hammer
   turns "I simulated it" into "I predicted it and then measured it." This is
   the single highest-leverage weekend of the whole project. Target < 10%
   agreement on $f_1$; write up honestly whatever you get, and diagnose the gap
   (bolt preload, added sensor mass, joint compliance).

3. **Make the repo legible in 60 seconds.** README with the result table up top,
   one-command demo, CI badge.

4. **Package the offering** (see `docs/04_COMMERCIALIZATION.md`).

### Acceptance test
A stranger clones the repo and reproduces your headline number in under two
minutes without a licence.

---

## Realistic schedule

| Weeks | Phase | Output |
|---|---|---|
| 1–2 | Core pipeline + verification | Frequencies you can defend |
| 3–4 | Coupling + harmonic | The differentiating analysis |
| 5–6 | DOE + surrogate | < 2% MAPE, 600× speedup |
| 7–8 | Optimisation loop | Verified before/after result |
| 9–10 | Benchmark + report | Portfolio artifact |
| 11–12 | Physical validation + packaging | Credibility + offer |

Compress by cutting Tier 2 (Ansys) entirely — it adds credibility with some
audiences but zero capability.
