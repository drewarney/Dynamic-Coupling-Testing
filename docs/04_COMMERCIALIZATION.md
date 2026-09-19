# 5. Portfolio & Commercialisation Strategy

## 5.1 Positioning: what this actually is

Call it a **"Predictive Dynamics & Generative Design Engine"** if you like the
phrase — but understand what is load-bearing underneath it, because that is what
gets asked about in an interview:

> An automated pipeline that takes parametric CAD, runs modal FEA, learns a
> calibrated surrogate of the structure's dynamic response, and optimises
> geometry to move resonances out of an excitation band while minimising mass.

Three claims are defensible and specific. Lead with these:

1. **Verified, not just run.** Solver agrees with closed-form solutions to
   < 0.1%.
2. **A real ML insight, measured.** Physics-based mode classification took
   surrogate error from 50% → 1.9% — same solver, same GP, same DOE.
3. **A quantified engineering result.** 13.9% lighter and 229% higher first
   mode, all modes clear by ≥ 99.7% against a 25% requirement, in 64 FEA calls.

Avoid "AI-powered" as the headline. Every structural dynamicist who might hire
you has seen a dozen of those. "I found and fixed the mode-tracking failure that
makes naive surrogates useless" is the sentence that gets you the interview.

## 5.2 Resume lines

Pick two or three. Each is one line, leads with the quantified outcome, and
names a specific technique.

> **Automated Resonance Optimisation Pipeline** — Built an end-to-end
> CAD→FEA→ML→CAD loop (CadQuery, Gmsh, CalculiX, scikit-learn) that cut motor
> mount mass 13.9% while raising the first natural frequency 229% and clearing
> all modes from a 1,000–6,000 RPM excitation band by ≥ 99.7% margin.

> **Surrogate Modelling for Structural Dynamics** — Diagnosed and fixed a mode
> classification failure that caused 50% surrogate error on symmetric
> structures; physics-based rigid-body classification of degenerate mode
> families reduced held-out error to 1.9% MAPE with calibrated 94% uncertainty
> coverage, enabling 600× faster design evaluation.

> **Cross-Modal Coupling Analysis** — Implemented programmatic non-proportional
> damping analysis (Φᵀ C Φ projection, energy-transfer indexing) demonstrating
> that the standard uncoupled-mode assumption understates band-limited RMS
> response by 29.9% for discretely damped assemblies; identified and fixed a
> degenerate-eigenspace basis dependence that made the result vary by 28
> percentage points across platforms.

> **FEA Verification & Validation** — Developed a dependency-free 3D Timoshenko
> frame solver validated against closed-form Euler–Bernoulli, fixed–fixed,
> axial and torsional solutions to < 0.1%, enabling the full optimisation
> pipeline to run in CI without commercial licences.

**What to avoid:** "Leveraged AI/ML for generative design" — unfalsifiable and
says nothing. "Used Ansys" — so has everyone.

## 5.3 GitHub presentation

The README must let a reviewer reach your headline number in **60 seconds**:

1. One-sentence description, then the **before/after result table**. Above the
   fold, before any installation instructions.
2. A Campbell or FRF figure showing baseline crossings vs optimised clearance.
3. One-command demo that runs licence-free in under two minutes.
4. The verification table — this is your credibility.
5. Architecture diagram, then the docs links.
6. CI badge with tests actually passing.
7. An explicit **Limitations** section. Its presence is a positive signal.

Add the figures — generate a Campbell diagram, an FRF coupled-vs-uncoupled
overlay, and a surrogate parity plot. `docs/` is text-only right now; three
plots would materially improve it.

## 5.4 Is there a business here? An honest assessment

**Short answer: it is a strong consulting wedge and a weak standalone SaaS.**

### Why SaaS is hard

- **Ansys, Altair (OptiStruct/HyperStudy), Siemens (Simcenter) and Dassault all
  ship optimisation modules already.** You are not filling an empty category.
- **Geometry is the moat and the trap.** Your pipeline works on *parametric*
  models with a defined design space. Real customer CAD arrives as dumb STEP
  solids with no parameters. Extracting a usable design space from arbitrary
  geometry is the actual hard problem, and it is unsolved in general.
- **Engineering software sales cycles run 6–18 months** and route through
  procurement and IT security. This is not a self-serve credit-card market.
- **Validation liability.** If a bracket cracks in the field and your tool
  approved the design, that is a conversation you need insurance for.

### Where the real opportunity is

**Mid-size suppliers who have Ansys licences but no optimisation specialists.**
A 200-person automotive Tier-2 has a CAE engineer running one modal analysis at
a time by hand in Workbench. They do not have someone who can set up a DOE,
train a surrogate, and defend it. That gap is real.

### Three routes, in order of realism

**1. Consulting / fractional CAE (start here).** $8–25k per engagement:
resonance study, optimisation, report. No product risk, immediate revenue,
teaches you which parts customers actually pay for. Your portfolio project *is*
the sales collateral. This is how essentially every successful engineering
software company started.

**2. Vertical tool.** Pick one narrow part family where the parameterisation is
stable and known — e-motor mounts, EV battery module brackets, drone arms,
compressor manifolds. Sell a configured tool plus setup service, $15–40k/year.
Narrow enough that "extract the design space" is solved once, by you, for that
family.

**3. Open-core.** MIT-license the pipeline, sell the Ansys/NX connectors,
managed cloud DOE runs, and support. Slowest to revenue, best distribution,
and the licensing story is clean if you stay on the GPL-subprocess boundary.

### Who to talk to first

E-mobility and drone suppliers. NVH is a first-order product requirement (no
engine noise to mask it), development cycles are short, and the companies are
young enough not to have a 15-year Altair contract.

### Validate before you build

Before writing a line of product code: **show the benchmark result to five CAE
engineers at target companies.** Ask what they currently do about resonance
avoidance and what it costs them. If fewer than three say "we do this by hand
and it takes weeks," the market is not there and you have a great portfolio
project instead — which is a perfectly good outcome.

## 5.5 Technical gaps before any commercial claim

Be clear-eyed about what is not done:

| Gap | Effort |
|---|---|
| Design-space extraction from non-parametric STEP | Hard — this is the real moat |
| Manufacturing constraints (draft, min wall, tool access) | Medium |
| Stress/fatigue constraints alongside modal | Medium — needs a static solve in the loop |
| Bolted-joint modelling | Medium — needs test correlation |
| Thermal dependence of $E$ | Low |
| Multi-body / assembly-level modes | Hard |
| Robustness to manufacturing tolerance | Medium — the optimum must not be a knife edge |

The tolerance item deserves emphasis: **an optimum that sits 0.1 mm from a
resonance is worthless.** Robust optimisation (optimise the worst case over a
tolerance band, or add a variance penalty from the GP) is the most valuable next
feature both technically and commercially.
