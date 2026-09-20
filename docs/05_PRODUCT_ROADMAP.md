# 6. Product Roadmap: from one part to a practice

> Strategy note, written at the end of the verification/hardening session that
> produced commits `8d2af8a`..`016ef32`. Private repo; contains commercial
> positioning as well as engineering plan.

## Context

`dynopt` currently optimises one part: a 4-leg motor mount. The goal is a tool
that works on arbitrary geometry and supports a consulting practice with a
credible end goal. This plan states what actually blocks generality (measured,
not guessed), what is monetizable, and gives a ready-to-paste prompt for the
next session.

Inputs: ~10 hrs/week with AI leverage; industry access in drone/eVTOL/robotics
and aerospace/defence; repo private by default, selectively open-sourced for
trust.

## Where the code actually stands

Measured across 3,943 LOC:

| | LOC | Part-specific? |
|---|---|---|
| Core (fea, dynamics, doe, surrogate, optimize) | ~2,400 | No |
| `geometry/motor_mount.py` | 414 | Yes — by design, it is the plugin |
| Mode-family block in `dynamics/modal.py` (~L300–430) | ~130 | **Yes — the real blocker** |
| `cli.py` demo wiring | 194 | Trivially |

**~79% is already part-agnostic.** The architecture generalises; one *concept*
does not. `FAMILY_DOFS` / `MODE_FAMILIES` (`bounce_z`, `sway_rock`,
`torsion_z`) assume "payload on compliant supports". No payload, no families.

## The key insight: dissolve mode tracking, don't generalise it

Today: regress mode frequencies → compute band risk from them. Mode identity is
the hard part and it does not generalise.

Instead: **regress the band-risk scalar directly.** `ResonanceObjective.risk()`
in `optimize/objectives.py` already computes it. As a regression target it is

- continuous in the design variables (summing over modes smooths the crossings
  that forced all the tracking work),
- defined for *any* structure — no payload, no classification, no MAC,
- exactly what the optimiser minimises.

Mode tracking survives as a **reporting** layer for the final design ("which
mode is the problem"), not a training target. This removes the single hardest
piece of the current design.

## The end goal: first-time-pass vibration qualification

Both target industries converge on one expensive, well-defined pain:
**MIL-STD-810 / DO-160 random-vibration qualification.** A failed shaker test
costs weeks of schedule and a redesign loop. Drone/eVTOL adds propeller-order
excitation on top.

That is the sellable promise: *predict and fix resonance and random-vibration
failures before the shaker table.* It has a number attached (cost of a failed
qual), a named buyer (the engineer who owns qual), and a natural repeat cycle.

This requires one capability the tool lacks: random-vibration response. Current
`ExcitationBand` is sinusoidal-order based. Needed: input PSD → response PSD via
`|H(f)|^2 S_in(f)` → RMS g, 3-sigma, and Miner's-rule fatigue. `dynamics/harmonic.py`
already has `rms_response(H, freqs, psd=...)`, so the hook exists.

## Monetization, honestly

**Now — consulting.** $8–25k per engagement (resonance study, optimisation,
report). The tool is leverage, not the product; each engagement costs 1–3 days
of parameterisation. 6–10 engagements/year is a real income but it is a job.
Correct first step: it funds everything and tells you which part family repeats.

**6–12 months — vertical tool.** One part family (drone arms / motor mounts /
avionics trays), configured tool plus setup. $15–40k/yr. Needs a reference
customer, which consulting supplies.

**Not worth chasing — generic optimisation SaaS.** Ansys DesignXplorer, Altair
HyperStudy, Siemens HEEDS and Dassault Tosca are mature and integrated. Do not
compete head-on.

**The real differentiator is not optimisation — it is trustworthy automation.**
This session found eight silent failures in a pipeline that appeared to work:
geometry meshing as five disjoint solids, a 2.71x mass error, a coupling result
that differed between macOS and Linux, two uncatchable Gmsh aborts. Every group
attempting CAE automation hits these. Nobody sells the discipline that catches
them: solvers validated against closed form, tier-validity envelopes where the
model declares its own range, mandatory high-fidelity sign-off, reproducibility
checks. Pitch: **"automated FEA you can put your name on."**

**Visibility strategy that fits "private, selectively open":** keep the
framework and vertical configs private; open-source a small *verification
kernel* — the analytically-validated beam solver plus the V&V test suite. It is
the least commercially sensitive part and the most credibility-generating,
because trust has to be inspectable.

## Roadmap (recalibrated to ~10 hrs/wk with AI leverage)

| # | Work | Est. |
|---|---|---|
| 1 | `PartSpec` plugin boundary; port the mount onto it | 1 session |
| 2 | Band-risk regression target; demote mode tracking to reporting | 1 session |
| 3 | Second part (drone arm) — proves generality empirically | 1 session |
| 4 | ~~Coarse-3D Tier 0~~ + multi-fidelity GP (arbitrary geometry) | 2–3 sessions |
| 5 | Random-vibration/PSD objective — the qual wedge | 2 sessions |
| 6 | Mesh morphing / FFD for non-parametric STEP | 3–4 sessions |
| 7 | Physical hammer test on the mount | 1 weekend + hardware |

Items 1–3 are the generalisation proof. Item 5 is what makes it sellable.
Item 7 is what makes anyone believe it.

**On item 4 — coarse-3D is struck out, and that is a result.** The plan was a
coarse mesh as the cheap tier for parts too bulky for a beam. Measured on both
parts, it does not exist: once the mesh resolves the thin curved features it
has to resolve at all, the global target size barely moves the node count. A
7 mm target and a converged 3 mm target differ by 1.6x in cost and under 1% in
answer (`docs/00_ARCHITECTURE.md` §1.5.2). There is no discrepancy worth
learning and no speed worth having. The multi-fidelity work therefore stands or
falls on the **beam** tier, which is ~2500x cheaper and genuinely wrong — 1.735x
at the mount's default design.

That leaves the original gap open: a bulky solid still has no cheap tier. The
honest options are a reduced-order model (Craig-Bampton on a fixed mesh) or
accepting 3D cost for such parts. Neither is a coarse mesh.

**On item 4 — the beam half, measured.** A GP on the log-ratio cuts held-out
error from 40.2% to 18.3% on the mount and from 21.0% to 4.9% on the arm, and a
per-column scalar does not (25.3% / 13.2%). At equal solver-seconds it wins
while fine-tier data is scarce and stops mattering once it is not — the arm's
fine-only GP overtakes at 12 fine solves. So the correction is worth shipping
for early exploration and is not a route to quotable numbers: 18% still needs a
3D sign-off. Full write-up in `docs/06_MULTIFIDELITY.md`.

**What the study actually surfaced as the next move.** The mount's beam tier
agrees with 3D to 0.8% *unloaded* and errs 44% *loaded*. The gap is one missing
piece of physics — a plate carrying a central point mass on four edge beams has
no membrane action — not a failure of beam theory. Closing that (an effective
plate stiffness, or a shell element for the plate) is worth more than any
further work on the correction, and it would move the mount toward the arm's
5%. That is the highest-value item this study produced and it was not on the
roadmap before.

**On item 6 — the honest caveat.** Morphing optimises arbitrary geometry without
parameters, which is the generality unlock. But it returns a deformed *mesh*,
not manufacturable CAD. Either a reconstruction step or customer redraw is
required. Do not promise "upload any STEP, get optimised CAD back" until this
is solved.

## Next-session prompt (paste verbatim)

> **Superseded — items 1-4 are done.** Kept for the record; the current next
> step is the plate-stiffness gap described under item 4 above, then item 5.
>
> Continue `dynopt` on branch `claude/vibrant-johnson-ocoqui`. Goal this
> session: break the pipeline's dependence on the motor mount so a new part is
> one file, and remove mode tracking from the surrogate's critical path.
>
> **Task 1 — `PartSpec` plugin boundary.** Define a protocol in
> `src/dynopt/geometry/base.py` covering everything the core needs from a part:
> design space, frame builder (optional), CAD builder, BC selection strategy,
> payload definition, and the reporting mode families (optional). Port
> `motor_mount.py` onto it unchanged in behaviour. Remove the mount references
> from `cli.py` and `dynamics/modal.py` — the families belong to the part, not
> the dynamics core. Acceptance: the existing 60 tests still pass, and
> `grep -rE "motor_mount|bounce_z|sway_rock" src/dynopt --include="*.py"` returns
> hits only under `geometry/`.
>
> **Task 2 — band-risk as the regression target.** Add a `targets="risk"` mode
> to `_sample_dataset` in `optimize/driver.py` that regresses
> `ResonanceObjective.risk()` plus mass, instead of per-mode frequencies. Keep
> `targets="family"` working. Measure both on the same DOE and report held-out
> MAPE side by side — I want to know what accuracy the generality costs, with
> numbers, before committing to it. Keep mode classification available for
> reporting the final design.
>
> **Task 3 — second part, if time allows.** Add a drone arm (cantilever tube,
> motor mass at the tip, propeller-order excitation) as a second `PartSpec`.
> This is the empirical proof that tasks 1 and 2 worked.
>
> Constraints, unchanged from last session: validate against closed-form
> solutions before building on anything; no published number that the shipped
> code cannot reproduce; report negative results rather than tuning tolerances
> to pass. If a design-space sweep or a repeated-meshing loop can break it,
> write that test.

## Verification

- [ ] `pytest -q` still passes (60, or 45 + 15 skipped without CalculiX)
- [ ] Mount references confined to `src/dynopt/geometry/`
- [ ] Risk-target vs family-target held-out MAPE reported with numbers
- [ ] Drone arm runs end to end through the same driver with no core changes
