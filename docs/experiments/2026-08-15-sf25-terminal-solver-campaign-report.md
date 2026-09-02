# SF-25 terminal-solver campaign — scientific status report

- Date: 2026-08-15
- Author: orchestrator (Claude Fable 5), on owner directive
- Audience: a scientific team evaluating how to continue the Lester eq. (14)
  streamfunction program from the current state
- Authoritative raw log: the append-only bitácora in
  `docs/plans/active/lester-eq14/increments/SF-25-terminal-manifold-solver.md`
  (every experiment below is recorded there with prespecified protocols,
  raw numbers, and mechanical verdicts)
- Companion research records:
  `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` (method
  survey and shift-mechanism theory),
  `docs/decisions/2026-08-14-terminal-shelf-remediation-options.md`
  (remediation literature review S1-S6)

---

## 1. Executive summary

MacroFlow3D's Lester eq. (14) solver — Picard/adaptive-relaxation/Anderson/
Newton-Krylov with eta-, epsilon-, and amplitude-continuation — **fully
solves the sigma_Y^2 = 0.25 benchmark** (residual r_F <= 1e-6 at full
coupling) and is validated by a 19-entry test suite. At sigma_Y^2 = 1 the
solver hits a hard, reproducible wall at a **critical heterogeneity
amplitude a* in (0.500, 0.5125]** (in units of the normalized log-K field):
beyond it, no method in our arsenal reaches the algebraic gate.

The SF-25 campaign (26 branch commits, ~15 prespecified GPU experiments)
established that the wall is an **algebraic/spectral pathology of the
discrete nonlinear system, with no physical signature**:

1. At the wall, the Jacobian acquires a small negative generalized
   eigenvalue (measured: -2.06e-3) — the near-solution region becomes
   **repelling for every damped iteration map we possess AND for the
   explicit pseudo-time flow** (measured dynamically both ways).
2. The reachability boundary in the coupling parameter eta is sharply
   bracketed at 32^3 ((1-eta)* in [7.8e-4, 1.56e-3)) and moves ~30x AWAY
   from full coupling at 64^3 — **mesh refinement makes it worse**, the
   opposite of a resolution deficiency.
3. All states in the contested region are **physically indistinguishable**
   (velocity-reconstruction error ~2.5-2.6%, invariance defect ~2%, no
   gradient degeneracy anywhere near the regularization scale): the
   physical accuracy floor at 32^3 is set by spatial truncation, not by
   the solver. The algebraic fight below r_F ~ 1e-3 has no measurable
   physical content at this grid.
4. Two independent method families (implicit damped maps; explicit flow)
   independently bottom out at the same level, r_F ~= 4e-4.

The leading failure hypothesis is a **discretization-induced (spurious)
loss of the solution branch** — a documented phenomenon for finite-
difference approximations of nonlinear elliptic problems — possibly
combined with formulation differences from the reference paper (which
reports solving the same system at 256^3, sigma^2 up to 4, to r_F ~ 1e-16
with error-controlled explicit pseudo-time and no denominator
regularization). Section 8 grounds this and the alternatives in the
literature.

**Deliverables standing on the branch:** the audited shift-operator
enablers, a deterministic diagnostic instrument (7 heavy GPU cases), three
config-gated solver-robustness fixes with tests, and the complete
versioned evidence dossier. The originally-scoped terminal method
(shifted-Newton/LM schedule) was **falsified by its own prespecified
diagnostic gate before integration** — by design of the campaign protocol,
no falsified method was wired into the production solver.

---

## 2. The scientific problem (one page)

For steady Darcy flow `v = -K grad h`, `div v = 0` in a smooth, locally
isotropic 3D conductivity field, Lester et al. show the flow is
helicity-free and admits a global Clebsch pair `psi1, psi2` with

```
v = grad(psi1) x grad(psi2),   v.grad(psi_i) = 0.
```

Streamlines are intersections of two stream-surface families; purely
advective transverse macrodispersion is asymptotically zero. Computing
`(psi1, psi2)` enables structure-preserving particle transport (zero
spurious transverse leakage by construction), the project's route to
testing whether classical positive transverse-macrodispersion results are
numerical artifacts.

The pair solves the coupled nonlinear elliptic system (Lester eq. (14),
reformulated; `q = 1/k`, `A = -div(q grad)`):

```
A psi1 = -q S2(psi1, psi2) + affine RHS
A psi2 = -q S1(psi1, psi2) + affine RHS
```

with `S1, S2` quadratic couplings in second derivatives of the OTHER
field, triply periodic BCs, mean-zero gauge, and an affine/fluctuation
split carrying the mean flow (qbar = (1,0,0)). Denominator terms are
regularized by a configurable epsilon (studied; see F5/F10). The
nonlinearity strength scales with the log-conductivity amplitude; the
solver approaches the physical target by continuation in coupling (eta:
0 -> 1), regularization (epsilon), and field amplitude (lambda: Y =
lambda * Y_target).

Reference-paper benchmark (recorded from sec. 5.1 of the source paper):
homogenized initial guess, **explicit variable-step pseudo-time** to
r_F ~ 1e-16, 256^3, correlation length / grid spacing = 16, variances up
to sigma^2 = 4, no epsilon-regularization described. (Its spatial
discretization scheme is NOT recorded in our theory notes — verification
item V1, section 8.)

---

## 3. The solver as built (code map)

All paths relative to repo root. Everything below is merged and
suite-gated except where noted.

### Production stack (`src/physics/streamfunctions/`)

| Component | File(s) | Role |
|---|---|---|
| Elliptic operator, gradients | `DifferentialOperators.cu/.cuh` | `A = -div(q grad)` (face-coefficient FD), total gradients (SF-02/07) |
| Nonlinear sources | `NonlinearSources.cu/.cuh` | `S1, S2` + explicit epsilon regularization with logged scale (SF-09) |
| Coupled residual | `ResidualEvaluator.cu/.cuh` | `F(u)`, `r_F`, nonfinite accounting (SF-10) |
| Affine split, gauge | `affine_periodic_rhs.cuh`, `affine_gauge.cuh` | mean-flow/fluctuation decomposition, benchmark gauge (SF-06/19) |
| Outer solver | `StreamfunctionSolver.cu/.cuh` | adaptive-omega Picard loop, trial guard chain, Anderson and Newton integration, SF-25 hygiene interceptors (SF-14/15/24/25) |
| Anderson acceleration | `AndersonAccelerator.cu/.cuh` | depth-3..8 coupled AA, QR condition guard, restart-on-stagnation (SF-20/25) |
| Newton-Krylov phase | `NewtonKrylovSolver.cu/.cuh` | globalized Newton, Armijo, Eisenstat-Walker-style forcing, rescue window, rescue-omega reset (SF-24/25) |
| Matrix-free Jv | `JacobianVectorProduct.cu/.cuh` | directional-difference Jacobian action (SF-22) |
| Krylov solver | `CoupledGmres.cu/.cuh` | restarted GMRES, templated on the Jacobian operator (SF-23/25-T01) |
| Preconditioner | `BlockDiagonalMGPreconditioner.cu/.cuh` | per-component multigrid on `A` (SF-23) |
| Shifted operator | `ShiftedJacobianOperator.cu/.cuh` | `mu*A + J` action; `mu = 0` bitwise passthrough; unit-gated exact (SF-25-T01) |
| Continuation | `ContinuationController.cu/.hpp` | eta/epsilon legs (SF-17), outer lambda leg + eta-rescue ramp + per-stage attribution counters (SF-21/26) |
| Physical diagnostics | `Diagnostics.cu/.cuh` | reconstruction error `e_v`, invariance `v.grad(psi)`, cross-gradient degeneracy, divergence (SF-11) |
| Workspace/validation | `StreamfunctionWorkspace.cu/.cuh`, `StreamfunctionTypes.hpp` | exact memory accounting, config structs and validation |

### Campaign instrument (test-tier, never in production paths)

`tests/streamfunctions/terminal_solver_gpu_cases.cu` (~4.3k lines), heavy
cases invoked explicitly via `--case <name>` (not in default ctest):

| Case | Implements |
|---|---|
| `terminal_shifted_apply_unit` | exactness gate for `mu*A + J` (measured max_diff = 0) |
| `terminal_dgate_diagnostic` | E2 deterministic plateau freeze; E3 mu-sweep; E4 generalized inverse iteration; E5 LM mini-solve; E6 SER Psi-tc probe; E6b micro-step scan; E7 epsilon-fold probe; E8 harmonic-init probe |
| `terminal_resolution_probe` | R1a/R1b/R2a/R2b 64^3 resolution experiments |
| `terminal_floor_guard_continuation` | floor-guard ON-path evidence on the real R1a fixture |
| `terminal_eta_endgame` | S1 fixed-ladder eta walk, 2 hygiene arms, eta=1 codas + extrapolation |
| `terminal_shelf_probe_phase1` | S3 single-step mu-scan + 100-step sustained run; S6 physical-quality battery (percentiles, spatial concentration) vs converged baseline |
| `terminal_explicit_flow_probe` | P2-A explicit pseudo-time flow, 2 inits, stability-capped dtau servo (amendment P2-A'), argmin snapshot, control coda |
| `terminal_eta_endgame_64` | P2-C: the eta walk at 64^3 / l/h = 16 |

Hygiene ON/OFF unit tests live in `tests/streamfunctions/newton_gpu_cases.cu`
(`newton_rescue_omega_reset`), `streamfunction_anderson_gpu_cases.cu`
(`anderson_stagnation_restart` + validation), and
`streamfunction_picard_adaptive_gpu_cases.cu` (floor-guard validation).
The sigma^2 = 1 acceptance gate (deliberately red until solved) is
`heterogeneity_smoke_sigma1` in
`tests/streamfunctions/heterogeneity_continuation_gpu_cases.cu`.

---

## 4. Campaign chronology (what was run, where, result)

Every experiment was **prespecified in the bitácora before running**
(design, budgets, mechanical readout rules); all execution on the remote
V100; all numbers below are bitwise-reproducible (the plateau freeze was
reproduced exactly across 5 independent binary builds).

| # | Experiment (bitácora timestamp) | Instrument | Result |
|---|---|---|---|
| E2 | Plateau freeze (08-14T11:xx) | dgate | Deterministic frozen shelf: lambda=0.5125, eta=1, r_F = 1.1204722529922055e-3 |
| E3 | mu-sweep (same) | dgate | Shift mechanism CONFIRMED: mu=0 budget-exhausts; mu=0.01..0.1 converge (20x fewer inners); cliff at mu ~ \|lambda_min\| |
| E4 | Generalized inverse iteration | dgate | lambda_min^gen(J; A) = -2.0590892e-3 -> **J indefinite** at the shelf |
| E5 | LM mini-solve | dgate | FAILED at k=0: all damped steps ascend, slope -> 0+ -> spurious merit quasi-minimum |
| E6 | SER Psi-tc probe (08-14T12:10) | dgate | FAILED: monotone ascent 1.12e-3 -> 1.15e-2 (SER over-growth) |
| E6b | Micro-step scan (08-14T13:05) | dgate | mu=1: delta r_F = -3.58e-8/step; mu=10: -1.77e-8 -> shelf NOT a strict minimum, but traversal ~1e4-1e5 steps |
| E7 | epsilon = 1e-3 fold | dgate | REFUTING: identical failure -> epsilon-robust |
| E8 | Harmonic init (08-14T14:00) | dgate | REFUTING-INTRINSIC: independent init stalls at 8.3e-4 |
| R1/R2 | 64^3 resolution (08-14T15:20/55) | resolution_probe | R1 confounded (premature omega collapse); R2 full stack: WALL_AT_ELLH16 |
| S1 | Eta endgame 32^3 (08-14T17:40) | eta_endgame | Frontier eta = 0.9984375 (converged, 184 its); 0.99921875 dies AT its initial residual (9 its) -> **(1-eta)* in [7.8e-4, 1.56e-3)**; eta=1 codas die at 8.0e-4 / 1.2e-3 |
| S3micro | mu-scan + sustained (same) | shelf_probe | Single-step descent only for mu >= ~1; sustained fixed-mu iteration REVERSES at k=1 and ascends +31%/100 steps (accelerating) -> **FLOW_STALLS; the shifted-Newton family cannot converge here (unstable direction amplified for every mu)** |
| S6 | Physical battery (same) | shelf_probe | **DIFFUSE + F-SAT**: shelf vs converged-baseline physically indistinguishable (e_v 2.64% vs 2.52%; concentration ratio 1.017; \|grad psi1 x grad psi2\|_min = 33x epsilon-scale) |
| S2 | Hygiene evidence (same) | ctest + cases | Rescue-omega reset and Anderson restart validated ON/OFF; floor guard wiring proven (fired in S1b) but its trailing-window criterion does NOT match R1a-class deaths (their tails are genuinely flat) |
| P2-A | Explicit flow, cap 1.0 (08-15T00:20) | explicit_flow | CONFOUNDED (servo crossed explicit stability limit; forensics + amendment recorded) |
| P2-A' | Explicit flow, cap 0.005 (08-15T17:20) | explicit_flow | **Shelf is flow-REPELLING** (escapes after -5.3% descent); no attractor <= 1e-6 over tau ~ 5000 from either init; zero-init arm wanders [1e-4,1e-2] band 56% of horizon, min r_F = 4.05e-4; control codas flag residual step-size contamination in the O(1)+ band (state-dependent stiffness) |
| P2-C | Eta endgame 64^3 (same) | eta_endgame_64 | **BRACKET_MOVED_AWAY ~30x**: frontier eta = 0.95; pre-frontier stiffening (eta=0.95: 112 its vs 38 at 32^3) |

---

## 5. Established facts

- **F1.** The implicit stack (Picard + adaptive omega + Anderson + Newton-
  Krylov) solves sigma^2 = 0.25 to r_F <= 1e-6 at full coupling; Newton
  gives 1.96x wall-time speedup on the hard fixture (SF-24 G4).
- **F2.** A critical amplitude exists: 0.500*Y_unit converges, 0.5125*Y_unit
  does not (same field shape, 2.5% amplitude difference) -> a* in
  (0.500, 0.5125] at 32^3, l/h = 8.
- **F3.** At the wall, J is indefinite (lambda_min^gen = -2.06e-3); the
  shift (mu*A + J) repairs the LINEAR solves exactly as theory predicts
  (E3, 20x) but no outer damped iteration converges: the unstable
  direction is amplified by factor > 1 for every mu (measured, S3B).
- **F4.** The merit landscape at the shelf is a quasi-minimum, not a strict
  one (E6b descends infinitesimally); line-search methods are trapped by
  construction (E5), and flow methods are repelled dynamically (P2-A').
- **F5.** The wall is epsilon-robust (E7), init-robust (E8, P2-A' arm 2),
  and Krylov-budget-robust (10x-budget probe).
- **F6.** Eta-reachability boundary at 32^3: (1-eta)* in [7.8125e-4,
  1.5625e-3). The last reachable stage converges cleanly; the first
  unreachable one dies at its own warm-start residual with zero descent —
  a local-contractivity collapse, not a slow grind.
- **F7.** h-refinement moves the boundary the WRONG way: at 64^3 / l/h=16
  the frontier is eta = 0.95 ((1-eta)* ~ 30x larger). Caveat: the SF-18
  spectral generator yields a statistically-equivalent (not identical)
  realization at the finer grid.
- **F8.** The explicit pseudo-time flow (the reference paper's method
  class): repelled from the shelf; over tau ~ 5000 (1e6 steps, 431-437 s
  wall/arm at 0.45 ms/step) finds no attractor with r_F <= 1e-6 from
  either a shelf or a zero init; min reached 4.05e-4.
- **F9.** Cross-family floor agreement at r_F ~ 4e-4 (implicit stage death
  3.98e-4; flow minimum 4.05e-4).
- **F10 (F-SAT).** Physical saturation: shelf iterate (r_F = 1.1e-3),
  converged lambda=0.5 state (r_F <= 1e-6), and the flow argmin state all
  have e_v ~ 2.5-2.6%, invariance ~ 1.8-2.7%, identical defect percentile
  tables, spatial concentration ratios ~ 1.02, and cross-gradient minima
  33x above the regularization scale. **Below r_F ~ 1e-3 the algebraic
  residual buys nothing physically measurable at 32^3.**
- **F11.** No topological/degeneracy event underlies the wall at this
  amplitude: defects are spatially DIFFUSE and \|grad psi1 x grad psi2\|
  stays O(1) everywhere.
- **F12.** Instrument fidelity boundary: fixed-step explicit integration is
  clean near the shelf but contaminated by state-dependent stiffness in
  the O(1)+ residual band (self-reported by the prespecified control
  codas). The attractor-existence question is narrowed, not closed.

## 6. Methods tried — what worked, what didn't

| Method | Code | Verdict |
|---|---|---|
| Fixed/adaptive Picard + omega backtracking | `StreamfunctionSolver.cu` | Works below a*; omega-floor death at the wall |
| Anderson acceleration (R5) | `AndersonAccelerator.cu` | Strong accelerator below a*; stalls at the shelf (E8) |
| Newton-Krylov + Armijo (monotone) | `NewtonKrylovSolver.cu` | Solves sigma^2=0.25 1.96x faster; trapped at the merit quasi-minimum at the wall (E5-class) |
| Shifted Newton / LM `(mu*A + J)` | `ShiftedJacobianOperator.cu` + dgate | Linear mechanism verified (20x); outer iteration amplifies the unstable direction for every mu — falsified as terminal method (S3B) |
| Implicit Psi-tc (SER) | dgate E6 | Ascends (SER over-growth + same amplification) |
| Explicit pseudo-time (paper class), fixed-cap servo | `terminal_explicit_flow_probe` | Faithful near shelf; shelf repelling; band wandering; needs error control for the stiff band (F12) |
| Eta/epsilon/lambda continuation | `ContinuationController.cu` | Regular and efficient up to the boundary; boundary is a hard contractivity collapse, not a schedule problem (S1: finer steps bought exactly one halving) |
| Warm-start extrapolation to eta=1 | eta_endgame coda | No help (lands at the shelf level) |
| Mesh refinement 32^3 -> 64^3 | resolution/eta_endgame_64 | Makes reachability WORSE (F7) — falsifies the under-resolution explanation for the wall |
| State-machine hygiene (rescue-omega reset; floor guard; Anderson restart) | `StreamfunctionSolver.cu`, config-gated OFF | Implemented + validated; floor-guard criterion measured insufficient for trailing-flat deaths (kept OFF by default; parameters are a measured open item) |

## 7. What is delivered and merged on the branch

1. `ShiftedJacobianOperator` + templated `CoupledGmres` (unit-gated exact,
   zero impact on accepted call sites) — commits `4e22528`..`9985f90`.
2. The complete diagnostic instrument (8 heavy cases) — `65e7a6f`,
   `29dfdef`, `8bf1b51`, `8d1bc90`, `14b089b`, `70e0ad6`, `0029d7f`,
   `a80de8d`.
3. Config-gated robustness fixes + ON/OFF tests — `ffba20b`.
4. The evidence dossier (bitácora), two decision records, and this report.
5. NOT delivered (falsified before integration, per the campaign's
   diagnostic-first protocol): the LM/Psi-tc terminal schedule inside
   `NewtonKrylovSolver`; the sigma^2 = 1 demonstration.

Suite state: 17/19 ctest entries green; the 2 red entries are the
deliberate unsolved-science gates (`streamfunction_heterogeneity_smoke`
= sigma^2 = 1 acceptance; `streamfunction_terminal_dgate` = the honest
FAIL verdict of the terminal-method demonstration).

## 8. Failure-source analysis (research-grounded)

- **H1 — discretization-induced (spurious) branch pathology. LEADING.**
  The literature documents finite-difference discretizations of nonlinear
  elliptic problems generating spurious solution branches and spurious
  bifurcations absent from the continuum problem (mechanism papers:
  Beyn/Lorenz-era "A Mechanism for Spurious Solutions of Nonlinear
  Boundary Value Problems"; Stephens-Shubin JCP "Spurious Behavior for a
  Numerical Scheme of Nonlinear Elliptic Equations"; unified theory:
  Iserles et al., SIAM J. Numer. Anal. "A Unified Approach to Spurious
  Solutions"; dynamical-systems view: Yee-Sweby). Our evidence FOR:
  no physical signature at the wall (F10/F11); a sharp spectral event
  (F3); the reference paper solves the same physics regime at higher
  variance. Evidence AGAINST / open: the h-trend (F7) shows the boundary
  moving away under refinement, which for a *pure* coarse-grid artifact
  is unexpected — unless the artifact belongs to the discretization
  FAMILY (stencil/formulation), not to a single resolution.
- **H2 — formulation gap vs the reference implementation.** Differences we
  can enumerate: (a) our affine/fluctuation split with mean-zero
  projection vs (unknown) total-field treatment; (b) our epsilon
  regularization (paper: none described); (c) our face-coefficient FD
  operator vs the paper's (unrecorded) spatial scheme — plausibly
  pseudo-spectral on the periodic box; (d) our fixed-cap flow integrator
  vs their variable-step (= error-controlled) pseudo-time.
  **Verification item V1: recover the exact spatial discretization and
  integrator of the paper's sec. 5.1 from the source PDF.** (Our theory
  notes record the pseudo-time protocol and targets but not the spatial
  scheme.)
- **H3 — genuine continuum fold in eta/amplitude.** Deemed unlikely
  (Lester's existence theory for smooth fields; F10 shows nothing
  physical happening) but not formally excluded; only a branch-following
  computation (bordered continuation) or an existence argument settles it.
- **H4 — gauge-manifold tangent degeneracy (the original SF-25
  hypothesis).** PARTIALLY falsified as the sole mechanism: the shift
  repaired the Krylov solves exactly as the manifold picture predicts,
  yet the outer divergence persists; the measured negative eigenvalue is
  a genuine unstable direction, not a harmless manifold tangent.
- **H5 — epsilon-regularization reshaping the landscape.** Locally
  refuted (E7; F10's 33x margin) at 32^3; the epsilon x h interplay at
  finer grids is untested; the paper used no epsilon.

## 9. Viable alternatives (research-grounded, with cost)

- **A1 — Error-controlled explicit pseudo-time (embedded RK pair).**
  The literal completion of the paper-parity question and the direct
  answer to F12. Dedicated literature exists: "Optimal embedded pair
  Runge-Kutta schemes for pseudo-time stepping" (J. Comput. Phys., 2020);
  paired explicit RK (P-ERK) schemes for **locally stiff** systems
  (Vermeire) — our measured state-dependent stiffness is their exact use
  case; embedded SSP pairs (Conde et al.). Cost: ~40-80 instrument lines
  (Heun/Euler embedded pair + PI step controller) + one V100 session.
  Decisive: either finds the attractor (then the terminal method is
  settled and SF-25's goal is re-achievable) or establishes
  no-stable-solution with clean fidelity.
- **A2 — Pseudo-spectral spatial discretization of eq. (14).** Natural for
  the triply periodic box (cuFFT available), removes the FD truncation
  structure implicated by H1/H2, and likely matches the reference
  implementation (pending V1). Cost: medium-large (new operator + source
  evaluation path; the continuation/solver stack is discretization-
  agnostic and would be reused).
- **A3 — Bordered/pseudo-arclength continuation in eta.** Classifies the
  discrete branch at both grids (fold-with-turn-back vs repelling branch
  that continues to eta = 1), using the two measured brackets as starting
  points (Keller 1977; Ipsen et al. condition estimates). Cost: medium
  (bordering solves on the existing shifted GMRES).
- **A4 — Alternative formulation of the pair.** Double-potential /
  coupled-Laplace formulations with auxiliary boundary conditions (Zijl
  1986; Matanga 1993; "Computation of 3D Water Flows by the Double
  Potential Method", EPJ Web Conf. 2020). The dual-streamfunction
  literature independently documents that direct iterative solution of
  one streamfunction from the other is fragile — consistent with our
  findings — and offers better-posed couplings. Cost: large (new
  formulation), but the highest-ceiling structural fix.
- **A5 — Consume at the truncation floor (policy option, no new code).**
  F-SAT shows every state in the contested region is physically
  equivalent at 32^3. For the artifact-demonstration science (transport
  in the exactly-integrable reconstructed field v~ = grad psi1 x grad
  psi2, comparing conventional vs invariant-preserving trackers on the
  SAME field), the existing eta ~ 0.998 / r_F ~ 1e-3-class states are
  usable TODAY with measured quality. This decouples the macrodispersion
  program from the terminal-algebra question. Requires an owner decision
  on acceptance semantics; SF-26+ gates stay untouched until then.
- **A6 — (kept for completeness)** Levenberg-Marquardt under local error
  bounds for nonisolated solutions (Dan-Yamashita-Fukushima), geodesic
  acceleration (Transtrum-Sethna), deflation — surveyed in the two
  decision records; all merit-descent members of this family are now
  counter-indicated at the wall by F3/F4 unless combined with A2/A4.

Recommended sequencing for the next team iteration:
**V1 (read the paper's scheme) -> A1 (cheap, decisive instrument step) ->
A3 or A2 depending on V1's answer**, with A5 available immediately as a
parallel science track.

## 10. Reproducibility

- Branch: `science/lester-sf25-terminal-manifold-solver` (26 commits,
  `d11f63a..15654c9` + this closure).
- Build (local WSL debug): see `AGENTS.md`; disable compiler launchers
  (sccache is broken for CUDA locally).
- All campaign runs execute on the remote V100
  (`cmake --preset v100-release`), via `scripts/remote run <job> -- ...`:

```bash
./build/v100-release/streamfunction_operator_tests --case terminal_dgate_diagnostic
./build/v100-release/streamfunction_operator_tests --case terminal_shelf_probe_phase1
./build/v100-release/streamfunction_operator_tests --case terminal_eta_endgame
./build/v100-release/streamfunction_operator_tests --case terminal_eta_endgame_64
./build/v100-release/streamfunction_operator_tests --case terminal_explicit_flow_probe
./build/v100-release/streamfunction_operator_tests --case terminal_resolution_probe
ctest --test-dir build/v100-release --output-on-failure
```

- Determinism: the frozen plateau state reproduced bitwise
  (r_F = 1.1204722529922055e-3) across five independent builds/runs; the
  eta-walk arms are bitwise identical until a hygiene feature fires.
- Every probe's protocol, budgets, and mechanical readout rules were
  committed to the bitácora BEFORE the corresponding run; no gate or
  threshold was modified after seeing results (all instrument corrections
  are recorded as measurement-methodology amendments with raw first-run
  numbers retained).

## 11. References

Campaign-internal: the SF-25 bitácora; decision records of 2026-08-14
(both); `docs/theory/lester-2023-key-claims.md`;
`docs/validation/acceptance-gates.md`.

External (verified via literature search, 2026-08-14/15):

- Lester, Dentz, Bandopadhyay, Le Borgne, "The Lagrangian kinematics of
  three-dimensional Darcy flow", J. Fluid Mech. 918 (2021) A27; Lester et
  al., "Under what conditions does transverse macrodispersion exist in
  groundwater flow?", Water Resour. Res. 59 (2023) e2022WR033059.
- Grippo, Lampariello, Lucidi, SIAM J. Numer. Anal. 23 (1986); Li,
  Fukushima, Optim. Methods Softw. 13 (2000) — nonmonotone line searches.
- Kelley, Keyes, SIAM J. Numer. Anal. 35 (1998) — pseudo-transient
  continuation (attracting-steady-state assumption).
- Transtrum, Machta, Sethna, PRL 104 (2010), PRE 83 (2011) — sloppy
  canyons, geodesic acceleration.
- Keller (1977); Ipsen et al. (SIAM J. Numer. Anal.) — pseudo-arclength
  through folds.
- Spurious-solution mechanism literature: Beyn/Lorenz-era mechanism
  chapter; Stephens-Shubin (J. Comput. Phys.); Iserles et al. (SIAM J.
  Numer. Anal. 1991, unified theory); Yee-Sweby (dynamical approach).
- "Optimal embedded pair Runge-Kutta schemes for pseudo-time stepping",
  J. Comput. Phys. (2020); Vermeire, paired explicit RK for locally
  stiff systems; Conde et al., embedded SSP pairs.
- Zijl (1986); Matanga (1993); EPJ Web Conf. (2020) double-potential
  method — dual-streamfunction computation and its documented
  difficulties.
