# Theory: the formulas this pipeline implements

## 2.1 Modal analysis

The undamped free-vibration problem, after FE discretisation:

$$M\ddot{u} + Ku = 0 \quad\Longrightarrow\quad (K - \omega^2 M)\phi = 0$$

giving eigenpairs $(\omega_j^2, \phi_j)$ and $f_j = \omega_j / 2\pi$. Modes are
mass-normalised, $\phi_j^T M \phi_j = 1$, which makes $\tilde{K} = \Omega^2$ and
$\tilde{M} = I$ later on.

**Element matrices** (`fea/reference.py`) use the 12-DOF space frame element
with Timoshenko shear flexibility:

$$k_{xy} = \frac{EI_z}{(1+\Phi_y)L^3}\begin{bmatrix}
12 & 6L & -12 & 6L\\
6L & (4+\Phi_y)L^2 & -6L & (2-\Phi_y)L^2\\
-12 & -6L & 12 & -6L\\
6L & (2-\Phi_y)L^2 & -6L & (4+\Phi_y)L^2\end{bmatrix},
\qquad \Phi_y = \frac{12EI_z}{\kappa G A L^2}$$

with the consistent mass matrix

$$m_{xy} = \frac{\rho A L}{420}\begin{bmatrix}
156 & 22L & 54 & -13L\\
22L & 4L^2 & 13L & -3L^2\\
54 & 13L & 156 & -22L\\
-13L & -3L^2 & -22L & 4L^2\end{bmatrix}$$

$\Phi \to 0$ recovers Euler–Bernoulli, which is how the analytical verification
tests are run.

**Verification targets** (all implemented as tests):

| Case | Closed form |
|---|---|
| Cantilever bending | $f_n = \dfrac{\beta_n^2}{2\pi L^2}\sqrt{\dfrac{EI}{\rho A}}$, $\beta = 1.8751, 4.6941, 7.8548$ |
| Fixed–fixed bending | same with $\beta = 4.7300, 7.8532$ |
| Rod axial | $f_1 = \dfrac{1}{4L}\sqrt{E/\rho}$ |
| Shaft torsion | $f_1 = \dfrac{1}{4L}\sqrt{G/\rho}$ |

Measured agreement: **< 0.1%** on bending, **< 0.5%** on axial/torsional.

## 2.2 Participation and effective mass

A frequency you cannot excite does not matter. Project each mode onto the
rigid-body motion $r_d$ the excitation applies:

$$\Gamma_j = \frac{\phi_j^T M r_d}{\phi_j^T M \phi_j} = \phi_j^T M r_d,
\qquad M_{\text{eff},j} = \Gamma_j^2$$

$\sum_j M_{\text{eff},j} \to m_{\text{total}}$ as modes are retained, so the
captured fraction tells you whether your modal truncation is honest. Aim for
≥ 90% in the excitation direction before trusting a mode-superposition FRF.

## 2.3 Mode identity — the problem that decides whether the ML works

Frequencies come out of the eigensolver sorted by value. Sorted order is **not**
mode identity. Three separate failure modes bite, in increasing subtlety:

**(a) Mode crossing.** As geometry changes, "mode 3" switches from torsion to
bending. Regressing $f_3$ against parameters teaches the surrogate a
discontinuity that does not physically exist.

*Classical fix:* the Modal Assurance Criterion,

$$\mathrm{MAC}_{ij} = \frac{|\phi_{a,i}^T\phi_{b,j}|^2}
{(\phi_{a,i}^T\phi_{a,i})(\phi_{b,j}^T\phi_{b,j})}$$

Match by MAC, not by frequency order.

**(b) Degenerate eigenvalues.** Any structure with a rotational symmetry — a
4-leg mount, a 3-arm drone frame, a disc — has *repeated* eigenvalues. The
eigenvectors of a repeated eigenvalue are not unique: **any orthonormal rotation
within that eigenspace is equally valid**, and LAPACK returns a different
rotation at every design point. Vector MAC is meaningless here.

*Fix:* track the subspace. Orthonormalise both sets and compare principal
angles,

$$\mathrm{MAC}_{\text{sub}}(A,B) = \frac{\|Q_A^T Q_B\|_F^2}{\min(p,q)}$$

which is invariant to rotation within either subspace (`subspace_mac`).

**(c) Global shape drift.** Over a *wide* design space (span 40–120 mm, height
30–100 mm) the mode shapes themselves change so much that MAC against a fixed
reference design falls to 0.5–0.8 and the ordering of physical mode families
flips outright. Continuation tracking fixes this along a path; a scattered DOE
has no path.

*Fix — the one this project uses:* stop matching shapes and ask what each mode
physically **does**. Project every mode onto the six rigid-body motions of the
payload,

$$S_{jd} = \frac{|\phi_j^T M r_d|}{\sqrt{r_d^T M r_d}}, \qquad
\hat{S}_j = S_j / \|S_j\|$$

then score by **DOF group**, not by individual component:

$$E_{j,\mathcal{F}} = \sqrt{\sum_{d\in\mathcal{F}} \hat{S}_{jd}^2}$$

with groups

| Family | Rigid-body DOFs | Branches |
|---|---|---|
| `bounce_z` | $u_z$ | 1 |
| `sway_rock` | $u_x, u_y, r_x, r_y$ | 2 |
| `torsion_z` | $r_z$ | 1 |

**Why sway and rocking are merged.** On a payload carried by compliant legs the
lateral translation and the rocking rotation are inertially coupled through the
CG offset. They produce two *mixed* branches whose character swaps as the
height/span ratio changes. A measured signature of $[0.66\,u_x,\ 0.66\,r_y]$ is a
real mixed mode, not a classification failure — and deciding its label by argmax
over two tied components is a coin flip.

**This is not a cosmetic detail. Measured on the benchmark part:**

| Target definition | Held-out MAPE |
|---|---|
| Frequency order, no tracking | diverges |
| Vector MAC tracking | 33% |
| Subspace-MAC cluster tracking | 31–103% |
| Argmax rigid-body classification | 0.5–50% |
| **Group-energy classification (this project)** | **0.52–1.97%** |

Same solver, same GP, same DOE. The entire accuracy difference is the
definition of the regression target.

## 2.4 Cross-modal coupling

Modal decoupling holds **only** for proportional damping. Real hardware —
bolted joints, elastomer bushings, bearing films — has local dampers, which are
not proportional.

**Rayleigh (proportional) baseline.** $C = \alpha M + \beta K$ with

$$\zeta_i = \frac{\alpha}{2\omega_i} + \frac{\beta\omega_i}{2}$$

solved at two frequencies. Verified numerically: this leaves
$\max|\kappa_{ij}| \approx 4\times10^{-13}$ — decoupled to machine precision.

**Non-proportional case.** Project the real damping matrix:

$$\tilde{C} = \Phi^T C \Phi, \qquad
\kappa_{ij} = \frac{|\tilde{C}_{ij}|}{\sqrt{\tilde{C}_{ii}\tilde{C}_{jj}}}$$

A single grounded dashpot at the payload gives $\max\kappa = 1.0$.

**Frequency proximity.** Coupling strength alone is not enough — well-separated
modes barely interact even when $\kappa$ is large. Combine with the fractional
separation $\Delta_{ij} = |\omega_i - \omega_j| / \bar{\omega}$:

$$\eta_{ij} = \frac{\kappa_{ij}}{\sqrt{\kappa_{ij}^2 + (\Delta_{ij}/2)^2}}$$

$\eta > 0.3$ flags pairs that will exchange energy. **These must be separated
together** — push one mode up and it drags its partner along.

**Gyroscopic coupling** (rotating parts):

$$M\ddot{u} + (C + \Omega G)\dot{u} + Ku = f$$

with $G$ skew-symmetric, splitting each bending mode into forward- and
backward-whirl branches, $f_\pm(\Omega) \approx f_0 \pm g\,\Omega/60$.
Intersections with engine-order lines are critical speeds — the Campbell diagram.

## 2.5 Harmonic response

Steady state under $f(t) = Fe^{i\omega t}$:

$$U(\omega) = \underbrace{[K + i\omega C - \omega^2 M]^{-1}}_{H(\omega)} F$$

**Direct**: exact, $O(n^3)$ per frequency line.

**Mode superposition, coupling retained** — the fast path that stays correct:

$$\tilde{H}(\omega) = \Phi[\Omega^2 + i\omega\tilde{C} - \omega^2 I]^{-1}\Phi^T$$

The off-diagonal terms of $\tilde{C}$ are **kept** and the small (10–30 DOF)
system solved directly. Zeroing them is the textbook uncoupled-SDOF assumption;
on the benchmark part it understates band RMS response by **17.8%**.

Band-limited RMS, the scalar the optimiser can minimise:

$$u_{\text{rms}} = \sqrt{\int |H(f)|^2 S_{ff}(f)\,df}$$

## 2.6 The objective

$$J(x) = w_m\frac{m(x)}{m_0}
       + w_r\sum_j \alpha_j R_j(x)
       - w_s \tanh\!\left(\frac{\min_j d_j(x)}{\beta}\right)
       + w_f\,\Pi_{\text{floor}}$$

with the **smooth Gaussian resonance risk**

$$R_j = \sum_k w_k \exp\!\left(-\tfrac{1}{2}\left(\frac{f_j - f_{c,k}}{\sigma_k}\right)^2\right),
\qquad \sigma_k = \max(\beta f_{c,k},\ \tfrac{1}{2}(f_{hi,k}-f_{lo,k}))$$

and $\alpha_j$ the effective-mass fraction — a mode the excitation cannot reach
gets no weight.

**Why the risk kernel must be smooth.** The natural formulation
"maximise $\min_j |f_j - f_{exc}|$" is non-smooth and has a pathological failure
mode: pushing a mode *up through* a band is penalised at every intermediate
step, so a gradient optimiser never crosses even when the far side is much
better. The Gaussian is smooth, bounded, and its gradient vanishes once a mode
is clear — so the optimiser stops spending effort on safe modes and turns to
minimising mass. **This is the most consequential single line in the optimiser.**

Excitation bands from motor speed:

$$f_{lo,k} = \frac{\text{RPM}_{lo}\cdot o_k}{60}, \qquad
  f_{hi,k} = \frac{\text{RPM}_{hi}\cdot o_k}{60}$$

Default order weights $w_k = 1/o_k$: higher orders carry less energy, so a 4X
crossing matters less than a 1X crossing.

## 2.7 The surrogate

Independent GP per mode family, with three choices that matter:

1. **Regress $\log f$, not $f$.** Frequency $\propto \sqrt{K/M}$ is strictly
   positive and multiplicative in the parameters; in log space the response is
   far closer to linear. Predictions transform back with
   $\sigma_f \approx f\sigma_{\log f}$ (delta method).
2. **ARD Matérn-5/2.** A separate length scale per variable. Matérn rather than
   RBF because modal response is twice-differentiable, not analytic. The learned
   length scales are a **free global sensitivity analysis**: short scale = strong
   lever.
3. **Fit on unit-cube inputs.** Otherwise length scales are unit-dependent and
   meaningless.

Measured on the benchmark, 128 train / 200 test:

| Target | MAPE | p95 | 2σ coverage |
|---|---|---|---|
| `bounce_z` | 0.52% | 1.45% | 92% |
| `sway_rock_1` | 1.90% | 5.04% | 94% |
| `sway_rock_2` | 1.97% | 6.33% | 94% |
| `torsion_z` | 1.67% | 5.47% | 94% |
| mass | 0.153% | — | — |

Coverage near the nominal 95% means the GP's error bars are **calibrated** — it
knows what it does not know, which is what makes active learning and
trust-region optimisation valid. A gradient-boosted or neural surrogate gives
you the mean without an honest variance, so you never know when to re-run FEA.
