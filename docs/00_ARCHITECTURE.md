# 1. Architecture & Toolstack Selection

## 1.1 The decision, up front

**Build on the pure-Python stack. Use Ansys as a validation oracle, not as the loop.**

The reason is not performance or cost — it is that **the optimiser must be able to
change the geometry**, and only the Python stack gives you a scriptable
parametric CAD kernel in the same process as the solver and the ML layer.

| Requirement | Python (CadQuery + Gmsh + CalculiX + scikit-learn) | MATLAB (PDE + Stats/ML + Global Opt) |
|---|---|---|
| Parametric solid modelling in-process | **Yes** — CadQuery drives OpenCASCADE directly | **No** — `importGeometry` reads STEP, cannot author it |
| STEP export per design point | Native, ~50 ms | Requires external CAD via COM/API |
| Mesh control | Full Gmsh API, per-face sizing, 2nd order | `generateMesh(Hmax, GeometricOrder)` only |
| M, K matrix access | CalculiX `*MATRIX OUTPUT`; MAPDL `HBMAT` | `assembleFEMatrices` (toolbox DOF ordering) |
| GP surrogate | scikit-learn, GPyTorch, BoTorch | `fitrgp` (good, ARD kernels supported) |
| Deep-learning surrogate (GNN) | PyTorch / PyG | Deep Learning Toolbox (weaker for graphs) |
| Cost to ship commercially | **$0** — all GPL/BSD | Per-seat runtime licence, or MATLAB Compiler |
| Parallel DOE | `multiprocessing`, Dask, any cloud runner | Parallel Computing Toolbox (extra licence) |
| Simulink co-simulation | No | **Yes** — genuine advantage |
| Measured-data correlation | `scipy.signal`, `pyFRF` | **Best in class** (System ID Toolbox) |

MATLAB's two genuine wins — Simulink co-simulation and measured-data
correlation — are both *outside* the optimisation loop. Use it there if you
want, which is exactly what `matlab/dynopt_pipeline.m` is for.

## 1.2 Three fidelity tiers

The single most important architectural idea in this project: **do not pick one
solver.** Run a ladder, and make them interchangeable behind one interface.

```
Tier 0  reference beam FE  (numpy)      ~11 ms    DOE, surrogate training, CI
Tier 1  Gmsh + CalculiX    (GPL)       ~10-60 s   shape verification, shippable
Tier 2  PyAnsys / MAPDL    (licensed)  ~30-300 s  prestress, contact, sign-off
```

Every tier returns the same `ModalResult`, so switching is one line:

```python
solver = get_solver("reference", build_frame)          # tier 0
solver = get_solver("calculix",  MotorMountCAD())      # tier 1
solver = get_solver("ansys",     MotorMountCAD())      # tier 2
```

Tier 0 is what makes the project *demonstrable*. A reviewer can clone the repo
and run the whole pipeline in 90 seconds with no licences. That is worth more
on a portfolio than a screenshot of Workbench.

Tier 0 is also **verified against closed-form solutions** (`tests/test_beam_validation.py`,
agreement better than 0.1% against Euler–Bernoulli, fixed–fixed, axial and
torsional cases). "I validated my solver analytically" is a much stronger claim
than "I used Ansys".

## 1.3 The data pipeline, hook by hook

```
                 params: dict[str, float]
                          |
    ┌─────────────────────▼──────────────────────┐
    │ GEOMETRY   CadQuery -> OpenCASCADE         │
    │ hook: cq.exporters.export(solid, "p.step") │
    └─────────────────────┬──────────────────────┘
                          │  STEP (AP214)
    ┌─────────────────────▼──────────────────────┐
    │ MESH       gmsh.model.occ.importShapes()   │
    │            Mesh.ElementOrder = 2  (C3D10)  │
    │ hook: gmsh.write("mesh.inp")               │
    └─────────────────────┬──────────────────────┘
                          │  Abaqus .inp
    ┌─────────────────────▼──────────────────────┐
    │ SOLVE      *FREQUENCY, STORAGE=YES         │
    │ hook: subprocess.run(["ccx", "job"])       │
    │ out:  job.dat (eigenvalues), job.frd (phi) │
    └─────────────────────┬──────────────────────┘
                          │  ModalResult(f, phi, M, K)
    ┌─────────────────────▼──────────────────────┐
    │ CLASSIFY   rigid-body participation        │
    │ hook: family_frequencies(res, origin)      │
    │ out:  {bounce_z, sway_rock_1/2, torsion_z} │
    └─────────────────────┬──────────────────────┘
                          │  fixed-length continuous targets
    ┌─────────────────────▼──────────────────────┐
    │ SURROGATE  GP, ARD Matérn-5/2, log-target  │
    │ hook: FrequencySurrogate.fit(X, F, mass)   │
    └─────────────────────┬──────────────────────┘
                          │  f(x) in 0.02 ms
    ┌─────────────────────▼──────────────────────┐
    │ OPTIMISE   differential_evolution on GP    │
    │            + FEA verification each round   │
    └─────────────────────┬──────────────────────┘
                          │
                    optimised STEP
```

### The hooks that matter, with exact calls

**CAD → STEP (CadQuery).** Work in mm; OpenCASCADE and every downstream mesher
assume it.

```python
import cadquery as cq
solid = (cq.Workplane("XY").rect(w, h).extrude(t)
           .edges("|Z").fillet(r)
           .faces(">Z").workplane().hole(d))
cq.exporters.export(solid, "part.step")
```

**STEP → mesh (Gmsh).** `importShapes` goes straight through OCC — no
intermediate format, no healing step.

```python
gmsh.model.occ.importShapes("part.step")
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.ElementOrder", 2)      # C3D10 — see below
gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
gmsh.model.mesh.generate(3)
gmsh.write("mesh.inp")
```

> **The single most common automation error:** using linear tets (C3D4 /
> SOLID185). They are numerically over-stiff and report natural frequencies
> **10–25% high**. Always second order. This one setting invalidates more
> automated modal studies than anything else.

**Mesh → CalculiX.** Modal step:

```
*MATERIAL, NAME=PARTMAT
*ELASTIC
 68900.0, 0.33
*DENSITY
 2.7e-9
*SOLID SECTION, ELSET=EALL, MATERIAL=PARTMAT
*BOUNDARY
 FIXED, 1, 3
*STEP
*FREQUENCY, STORAGE=YES
 10
*NODE FILE
U
*END STEP
```

Watch the units: CalculiX is unit-agnostic, so mm/tonne/s means E in MPa and
ρ in tonne/mm³ (2.7e-9 for aluminium). Mixing mm geometry with SI density is
the second most common error, and it produces frequencies wrong by √1000 ≈ 31×.

**Ansys equivalents (PyAnsys).**

```python
mapdl.et(1, "SOLID187")                  # quadratic tet
mapdl.antype("MODAL")
mapdl.modopt("LANB", n_modes, 0, 10000)  # block Lanczos
mapdl.mxpand(n_modes, "", "", "YES")
mapdl.solve()
freqs = mapdl.post_processing.time_values
# full M and K for coupling analysis:
mapdl.run("/AUX2")
mapdl.run("HBMAT,'K.hb',,,ASCII,STIFF,YES,YES")
```

Prestressed modal — for a bolted bracket this shifts frequencies 5–15% and is
usually the reason FEA disagrees with a hammer test:

```python
mapdl.antype("STATIC"); mapdl.pstres("ON"); mapdl.solve()
mapdl.antype("MODAL");  mapdl.pstres("ON"); mapdl.solve()
```

## 1.4 Adding a part: the `PartSpec` boundary

The pipeline holds no knowledge of any specific geometry. A part is one class:

```python
from dynopt.geometry.base import PartSpec, DesignSpace, DesignVariable
from dynopt.dynamics.modal import ModeFamilySpec

class DroneArm(PartSpec):
    name = "drone_arm"

    @property
    def design_space(self):
        return DesignSpace([
            DesignVariable("length", 0.10, 0.40, 0.25, "m", "arm length"),
            DesignVariable("dia",    0.008, 0.030, 0.016, "m", "tube OD"),
            DesignVariable("thk",    0.0005, 0.004, 0.0015, "m", "wall"),
        ])

    @property
    def mode_families(self):                 # optional but recommended
        return ModeFamilySpec(
            dofs={"bending": (1, 2, 4, 5), "axial": (0,), "torsion": (3,)},
            branches={"bending": 2},
        )

    def build_frame(self, params):           # tier 0 (optional)
        ...  # -> FrameModel

    def build_cad(self, params):             # tier 1 (optional)
        ...  # -> CadQuery Workplane

    def solver_hints(self, params):          # optional
        return {"fix_selector": "z_plane", "fix_plane": 0.0}
```

Then register it in `geometry/registry.py` and it is available everywhere —
solvers, DOE, surrogate, optimiser, CLI (`--part drone_arm`) — with no core
change. `tests/test_partspec.py` proves this by driving a cantilever-with-
tip-mass through the whole pipeline and checking its first bending frequency
against the closed-form Rayleigh estimate.

### What belongs to the part, and why

| Concern | Why it is the part's, not the core's |
|---|---|
| `mode_families` | Which rigid-body motions form a family is a property of the structure. "Bounce / sway-rock / torsion" means nothing without a payload on supports. |
| `classification_origin` | Rigid-body motion is measured about *something* — a payload, a sensor seat, a bearing. |
| `solver_hints` (BCs) | A selector like "clamp the lowest face" silently picks the wrong face once a design variable moves geometry. The part knows which face is the mounting plane; the solver does not. |
| Tier availability | A beam idealisation is right for frame-like parts and simply wrong for bulky solids. The part declares which tiers it supports. |

### Which tiers to implement

Implement `build_frame` only when the part is genuinely frame-like — struts,
arms, brackets built from slender members. For a bulky solid a beam model is
not approximate, it is wrong, and no validity envelope rescues it. Such a part
supplies `build_cad` only and runs coarse-mesh 3D as its cheap tier.

## 1.5 Tier 1 hardening: what actually breaks

The Tier 1 path was written before it was ever executed. Running it against
Tier 0 found six defects, and **every one produced a confident wrong answer
rather than an error**. That is the defining hazard of automated FEA: the
toolchain is happy to mesh and solve nonsense.

| Defect | Symptom | Why it is silent |
|---|---|---|
| Legs extruded then rotated dipped below z=0 | Face selector clamped 4 leg tips | Solve succeeds, BCs are just wrong |
| Legs met the plate face without penetrating | STEP held 5 disjoint volumes | OCC finds no intersection for touching solids; `union` returns a compound |
| Corner fillet scaled with the plate | Disjoint volumes again, but only on *large* designs | Default design passed; 3 of 6 sampled designs failed |
| Fixed 28 mm bore vs parametric span | Self-intersecting solid, mesh aborts | Only below ~55 mm span |
| `ELSET=EALL` in the deck | ccx aborts | Gmsh never emits `EALL`; empty `.dat` still passed an existence check |
| Gmsh 1D/2D elements passed through | Spurious modes or abort | A 3D mesh file legitimately contains `T3D3` and `CPS6` |

Three rules follow, and they generalise well beyond this part:

1. **A disconnected model announces itself as rigid-body modes.** Any
   zero-frequency mode in a constrained model means part of the structure is
   attached to nothing. Assert on it — never eyeball the frequency list.
2. **Validate geometry across the design space, not at the default point.** An
   optimiser spends all its time away from your default. Checking "exactly one
   solid volume, flat face at the mounting plane" over a Sobol sample costs
   seconds and catches the entire class.
3. **Touching is not joining.** CAD booleans on tangent solids silently fail.
   Make mating features *overlap*, then trim — and verify the volume count.

`tests/test_tier1_calculix_integration.py` encodes all of it. It runs the real
toolchain (CadQuery -> Gmsh -> CalculiX); unit tests on the beam tier cannot
catch any of these.

## 1.6 Licensing — decide this before you write a business plan

| Component | Licence | Commercial use |
|---|---|---|
| CadQuery / OCP | Apache 2.0 / LGPL | Yes |
| Gmsh | GPL-2.0+ | Yes (as a separate process) |
| CalculiX | GPL-2.0+ | Yes (as a separate process) |
| NumPy/SciPy/scikit-learn | BSD | Yes |
| **Ansys academic** | Academic | **No — explicitly forbidden** |
| MATLAB | Per-seat | Runtime licence or Compiler required |

Keep GPL tools behind a subprocess boundary rather than linking them, and the
copyleft does not reach your own code. If the project becomes a product, the
shippable path is Tier 0 + Tier 1. Tier 2 stays an internal validation step, or
becomes a customer-runs-it-on-their-own-licence integration.
