# dynopt — working notes for Claude

Parametric CAD → modal FEA → GP surrogate → geometry optimised to move
resonances out of an excitation band while minimising mass.

## Commands

```bash
pytest -q                          # 162 tests; fewer if ccx is absent
pytest -q -m "not slow"            # skips the multi-minute meshing sweeps
dynopt modal --part motor_mount    # also: doe, coupling, optimize
dynopt --help
```

`pip install -e ".[oss]"` gives CadQuery + gmsh. **CalculiX (`ccx`) is a system
binary, not a wheel** — without it on PATH the 3D tests skip and only the beam
tier runs. `pythonpath = ["src"]` is set, so pytest works even if the editable
install breaks.

## Architecture

Three tiers behind one `ModalResult` interface:

| Tier | Backend | Cost | Use |
|---|---|---|---|
| 0 | beam FE (numpy) | ~10 ms | DOE, surrogate, CI |
| 1 | Gmsh + CalculiX | 10–70 s | verification, shippable |
| 2 | PyAnsys / MAPDL | — | **refuses to run** — never executed, 6 known defects |

**Adding a part = one file.** Implement `PartSpec` (`geometry/base.py`) and
register it in `geometry/registry.py`. The core holds no knowledge of any
specific geometry; `motor_mount.py` and `cantilever_arm.py` are the two
examples. The arm exists to prove the core is not secretly mount-shaped, and it
has earned its place by catching two real bugs through being different.

## Standing rules

These come from defects that all produced *confident wrong answers* rather than
errors. Full detail in `docs/00_ARCHITECTURE.md` §1.5.

1. **Validate against closed-form or a second solver** before building on
   anything.
2. **No published number the shipped code cannot reproduce.** Every figure in
   the README and docs is owned by a test — see
   `test_published_coupling_figure_is_current`, `test_multifidelity_study.py`,
   `test_plate_calibration.py`. Update the number and the docs in one commit,
   or not at all.
3. **Report negative results.** Never tune a tolerance to pass. Correcting the
   coarse-3D tier makes it *worse*; that is written down as a result, not
   hidden.
4. **Measure over a Sobol sample, never one design** (§1.5 rule 9). This
   produced three wrong published claims before it became a rule.
5. **A failure must not outlive its design.** Shared resources — the Gmsh
   session above all — get reset when anything using them throws.
6. **Anything that reads "the top face" is a part's opinion, not a law.**
   Supports, payload face, material and `min_feature` all come from
   `PartSpec.solver_hints`, recomputed per design.
7. If a design-space sweep or a repeated loop can break it, write that test.

## Gotchas that have actually bitten

- **Units are mm / tonne / s** everywhere the CAD touches, because CadQuery
  exports millimetres. SI values against mm geometry are wrong by 1e9 in
  density and still converge happily.
- **Gmsh options are global and survive `clear()`.** `gmsh_model()` restores
  defaults on entry; without it, mesh density depended on what ran earlier in
  the process.
- **`ModalResult.meta` is a contract, not a convention.** Call
  `missing_contract()`; `tests/test_backend_contract.py` enforces it per
  backend. A backend that omits a key does not fail — it degrades silently.
- **`payload_mass=None` means "ask the part"; `0.0` means "explicitly bare."**
  They are different, and conflating them made a bare run carry a 350 g motor
  while reporting the same frequency as the loaded one.
- Mode shapes parsed from `.frd` are ordered by that file's own sorted node
  tags, not the mesh's node order.

## State and next step

Branch `claude/vibrant-johnson-ocoqui`. 162 tests green.

**The goal is: user's STEP file in → resonance analysis → optimised geometry
out.** The pipeline optimises *design variables* and a STEP file has none, so
the missing piece is reconstruction — read the STEP, interrogate its features,
write a `PartSpec` that reproduces it, **verify the reconstruction against the
original** (volume, bounding box, inertia tensor, first six modal frequencies),
and then the existing pipeline runs unchanged.

There is no STEP *import* path yet: every `export_step` writes outward. That is
the next piece of work.

Known stale: the headline benchmark in the README and `docs/03_BENCHMARK.md`
(502.9 g → 420.4 g, 423.7 → 1191.5 Hz) was optimised on the *old* beam tier,
before the plate calibration, and no test owns it.
