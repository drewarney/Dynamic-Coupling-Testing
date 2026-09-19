# MATLAB path

`dynopt_pipeline.m` is the MATLAB-centric implementation of the same loop.
Run it with:

```matlab
results = dynopt_pipeline(struct('rpmRange', [1000 6000], 'nTrain', 120));
```

With no `stepFile` it uses a closed-form lumped-parameter model of the mount so
you can exercise the loop immediately. Point `cfg.stepFile` at a STEP file to
switch to a full PDE Toolbox modal solve.

## Required toolboxes

| Toolbox | Used for |
|---|---|
| Partial Differential Equation Toolbox | `femodel`, `generateMesh`, `solve` — modal FEA |
| Statistics and Machine Learning | `fitrgp` with ARD Matérn 5/2 — the surrogate |
| Global Optimization | `ga`, `particleswarm` — global search on the surrogate |
| Optimization | `fmincon` — local polish |

## The honest comparison

MATLAB wins on Simulink co-simulation (motor controller plus structure in one
model) and on correlating against measured data with the System Identification
and Signal Processing toolboxes.

It loses decisively on the part this project is actually about: **MATLAB has no
parametric solid modeller.** `importGeometry` reads a STEP file but gives you no
way to change it. Closing the CAD loop means driving SolidWorks or Fusion
through their COM/API from outside MATLAB and re-importing each time — an extra
process boundary, a Windows dependency, and a per-design-point round trip.

See `docs/00_ARCHITECTURE.md` for the full comparison.
