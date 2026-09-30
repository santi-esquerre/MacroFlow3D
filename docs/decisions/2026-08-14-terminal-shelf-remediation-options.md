# Terminal-shelf remediation: research record and option plan

- Date: 2026-08-14
- Author: orchestrator (Claude Fable 5), on owner directive
- Scope: SF-25 terminal-solver campaign, sigma_Y^2 = 1 critical-amplitude wall
- Status: OPTION MENU AWAITING OWNER SELECTION — this record authorizes no
  implementation and no runs by itself

## 1. Mandate

Owner directive (2026-08-14): perform an exhaustive numerical-analysis
investigation of the measured terminal failure and produce a remediation plan.
Hard constraint: **sanear el solver que tenemos** — remedies must be
modifications, schedules, or policies expressed on the components the project
already owns. Wholesale method replacement is out of mandate.

## 2. Measured failure modes (evidence: SF-25 D-gate dossier)

All numbers below are from the versioned SF-25 bitácora (runs `sf25-dgate1..4`,
`sf25-resprobe`, `sf25-resprobe2`), deterministic on V100.

- **FM1 — spurious merit quasi-minimum.** E5: every damped shifted-Newton step
  ascends `r_F`; the directional slope tends to 0 from above. Consistent with
  `J^T F ~ 0` while `F != 0` (inferred: no `J^T` action exists matrix-free).
- **FM2 — indefinite, near-singular Jacobian.** E4: generalized eigenvalue
  `lambda_min^gen(J; A) = -2.059e-3` (J indefinite in the A-metric). E3: at
  `mu = 0` GMRES exhausts its budget; `mu >= |lambda_min|`-scale restores fast
  inner convergence (>= 20x fewer inner iterations, cliff at
  `mu ~ |lambda_min|`). The shift mechanism `(mu*A + J)` works exactly as
  theorized.
- **FM3 — quasi-flat descent shelf.** E6: SER-scheduled Psi-tc at
  `mu in [0.01, 0.33]` ascends monotonically (divergence stop). E6b: single
  implicit steps at `mu in {1, 10}` DESCEND by `-3.58e-8` / `-1.77e-8`
  (relative descent ~3e-5 per step, inner = 1). The transition window
  `mu in (0.33, 1)` is unexplored. The shelf is therefore NOT a strict merit
  local minimum — a descent direction exists but traversal at the measured
  rate needs ~1e4-1e5 steps.
- **FM4 — state-machine fragility.** R1: both 64^3 stages died
  `omega_floor_rejected` MID-DESCENT (residual still falling). R2a: after a
  Newton step failure the rescue window inherits the collapsed persistent
  `omega` (~floor), so every rescue trial rejects by construction (SF-24
  design gap).
- **FM5 — amplitude criticality.** Amplitude identity:
  `Y(lambda=0.5125, sigma^2=1) = 0.5125*Y_unit` fails while
  `Y(lambda=1, sigma^2=0.25) = 0.500*Y_unit` (same unit field) converges to
  1e-6: critical amplitude `a* in (0.500, 0.5125]*Y_unit` at `l/h = 8`.
  Eta-cliff: achievable `r_F` jumps `1.2e-7 -> ~1.1e-3` across
  `eta in (0.996875, 1]` (four decades per 0.3% coupling).
- **FM6 — accelerator stall.** E8: Anderson depth 5 stalls on the same shelf
  (`stagnated`, 48 accepted / 12 rejected). No restart-on-stagnation logic
  exists in SF-20.

Robustness facts: the shelf is epsilon-robust (E7: identical failure at
`epsilon = 1e-3`) and init-robust (E8: harmonic/paper-style init stalls at the
same ~1e-3 class). R2 verdict `WALL_AT_ELLH16`: doubling `l/h` to the paper's
ratio did not move the reachability frontier for the as-designed stack
(caveat: the 64^3 deaths were FM4 machinery deaths, so the 64^3 wall is
suggested, not shelf-proven).

## 3. Resource inventory (what "the solver we have" means)

Audited and available on the SF-25 branch:

1. Residual evaluator `F` (GPU, cheap) — SF-10.
2. `A = -div(q grad)` + mean-zero projection + MG-preconditioned PCG —
   SF-02..05.
3. Matrix-free `Jv` — SF-22.
4. `CoupledGmres` templated on the Jacobian operator — SF-25 T01.
5. `ShiftedJacobianOperator` (`mu*A + J`, exact unit test, `mu = 0` bitwise
   passthrough) — SF-25 T01.
6. `BlockDiagonalMGPreconditioner` — SF-23.
7. Newton-Krylov phase: Armijo line search, Eisenstat-Walker-style forcing,
   rescue window — SF-24.
8. Anderson accelerator (depth 3-8) — SF-20.
9. Adaptive-omega Picard with floor semantics — SF-15.
10. `ContinuationController`: generic axis stepper
    (`start/target/initial_step/min_step/max_step`, halving-to-floor), outer
    lambda leg, eta-rescue ramp, epsilon leg — SF-17/21.
11. MG grid-transfer operators (prolongation/restriction) — SF-05; grid
    continuation is already planned scope (SF-27).
12. Physical diagnostics: `v.grad(psi)` residuals, reconstruction mismatch,
    gradient collinearity — SF-11.
13. The D-gate diagnostic instrument (deterministic E2 freeze + probe
    harness) — SF-25 T02..C06.

## 4. Literature findings mapped to failure modes

### 4a. Nonmonotone line searches (targets FM1, FM3)

- Grippo, Lampariello, Lucidi (SIAM J. Numer. Anal. 23, 1986, 707-716):
  accept if `phi(x+alpha*d) <= max_{0<=j<=M} phi(x_{k-j}) + gamma*alpha*
  grad(phi).d`. Explicit motivation: monotone Armijo "creeps along the bottom
  of a narrow curved valley" — verbatim our FM3 phenomenology.
- Zhang, Hager (SIAM J. Optim. 14, 2004): average-based nonmonotone variant
  (smoother envelope, same guarantees).
- Li, Fukushima (Optim. Methods Softw. 13, 2000, 181-201): derivative-free
  norm condition `||F(x+alpha*d)|| <= (1+eta_k)||F(x)|| - sigma||alpha*d||^2`
  with `sum eta_k < inf`. Global convergence without any merit gradient —
  decisive fit for us because we have no `J^T` action; it permits transient
  residual GROWTH under a summable envelope, exactly what escaping a merit
  quasi-minimum requires.

Fit to inventory: a pure acceptance-policy change inside the existing SF-24
line-search loop (item 7). The E6 probe already accepted bounded rises (2x)
but combined them with an aggressive SER schedule; the literature separates
the two: conservative step policy + nonmonotone acceptance envelope.

### 4b. Pseudo-transient continuation done right (targets FM2, FM3)

- Kelley, Keyes (SIAM J. Numer. Anal. 35, 1998, 508-523): Psi-tc
  `(delta_tau_k^{-1} V + J) s = -F` converges to steady states through regions
  where line-search and trust-region globalizations stagnate at local minima,
  under (i) the flow `u' = -V^{-1}F` having an attracting steady state and
  (ii) an accuracy-bounded timestep policy (SER is one option, not the only
  one). `V` need not be the identity — `V = A` gives the preconditioned flow
  in the A-metric, which is EXACTLY our `ShiftedJacobianOperator` with
  `mu = 1/delta_tau`.
- PETSc `TSPSEUDO` documents the same SER-based practice and its failure mode
  (too-aggressive timestep growth near stiff features).

Diagnosis of our E6 failure: SER grew `delta_tau` off small accepted rises,
compounding ascent (`r_F` 1.12e-3 -> 1.15e-2 over 8 steps); the E6b
micro-step measurement proves the flow itself descends at
`delta_tau <= 1` (`mu >= 1`). The remediation is a **descent-servoed,
conservative `mu` policy** on the machinery we already built, not a new
method. Lester et al. 2023 (sec. 5.1) use the explicit variant of this same
flow to 1e-16 — the implicit A-metric version is our stack's native
expression of the paper's method class.

### 4c. Geodesic acceleration for canyon traversal (targets FM3)

- Transtrum, Machta, Sethna (PRL 104, 060201, 2010; PRE 83, 036701, 2011):
  least-squares landscapes of "sloppy" models form plateaus and narrow
  canyons; LM steps stall there. Adding the geodesic (second-order)
  correction — the directional second derivative of `F` along the step,
  obtainable with ONE extra residual evaluation by finite differences —
  dramatically improves traversal speed and success rate.

Fit to inventory: an additive correction inside the existing step
computation; needs only `F` evaluations (item 1) and the existing shifted
solve (items 4-6).

### 4d. Fold theory and pseudo-arclength (targets FM5)

- Keller (1977); condition estimates: Ipsen et al. (SIAM J. Numer. Anal.,
  2007): natural parameter continuation loses solvability at a simple fold
  (`J_u` singular at the fold); the pseudo-arclength bordered system stays
  nonsingular through simple folds.
- Interpretation of our data: as `lambda -> a*`, the fixed-point map
  derivative crossing the unit circle is equivalent to `J` acquiring the
  measured small negative generalized eigenvalue; lambda-floor exhaustion of
  natural continuation at 0.5125 is the textbook fold signature. Whether the
  branch TURNS BACK (no 32^3 solution past `a*` on this branch) or merely
  passes near-singular is the single most important unresolved diagnosis.

Fit to inventory: a bordered extension of the existing lambda leg (item 10)
solved with the existing shifted GMRES (bordering algorithm). Medium cost;
diagnosis-first value.

### 4e. Homotopy endgames (targets FM5, the eta-cliff)

- Numerical-algebraic-geometry practice (power-series/Cauchy endgames;
  HomotopyContinuation.jl endgame documentation): near a singular endpoint
  the tracker switches to GEOMETRIC sample points `t_k = 1 - h^k R_0` plus
  extrapolation to the endpoint, instead of stepping onto the singularity.
- Our eta axis already implements geometric halving-to-floor toward
  `eta = 1`; the measured frontier `eta = 0.996875 -> r_F = 1.2e-7` is one
  endgame sample. The endgame remediation: lower the eta `min_step` floor,
  walk `1 - eta` down geometrically with warm starts, record the achievable
  `r_F(eta)` scaling, and optionally Richardson-extrapolate the eta-family
  states to an `eta = 1` initial guess (axpy on existing fields). If the
  eta-walk gets `(1-eta)*||q o S||` below ~1e-6, the final switch lands with
  an initial residual at gate level — potentially inside the true problem's
  Newton basin, bypassing the shelf entirely.

Fit to inventory: config change (axis floor) + trivial host-side
extrapolation.

### 4f. Anderson hygiene (targets FM6)

- Toth, Kelley (2015): first local convergence proof (contractive maps).
- Pollock, Rebholz: one-step residual bounds for general depths in
  contractive AND noncontractive settings; depth/damping balance.
- Evans et al. (2020): adaptive damping from iteration gain; restart
  variants prevent stagnation; nonmonotone globalization via adaptive
  regularization (J. Sci. Comput., 2023).

Fit to inventory: restart-on-stagnation (clear history), adaptive damping
inside SF-20 (item 8). Honest expectation: LOW probability of traversing the
shelf alone (E8 measured the stall); cheap hygiene that compounds with
4a/4b.

### 4g. Nested iteration / mesh sequencing (targets FM5 basin geometry)

- Standard multigrid results (Brandt FAS; nested-iteration literature):
  solving on a coarse grid and prolonging the solution enlarges the
  effective basin of attraction and avoids stagnation at spurious
  attractors; FMG is the canonical form for nonlinear elliptic problems.
- Our fit WITHOUT new architecture: the SF-18 spectral generator hashes
  amplitudes per integer mode, so the same-seed 16^3 field is numerically
  the low-pass of the 32^3 field (Gaussian spectral decay makes the missing
  modes O(e^-10) at `l = 8`; must be verified by a diagnostic print). Solve
  16^3 at the critical amplitude (the discrete `a*` shifts with `h` —
  unknown direction), prolong with the existing MG transfer, warm-start the
  32^3 stage. This is SF-27's planned scope pulled forward as a probe.

### 4h. Honest theoretical boundary

If the shelf were a STRICT local minimum of `||F||^2`, no acceptance-policy
or step-policy change could escape it with descent-flavored methods; only
branch methods (4d), problem-path methods (4e, 4g), or acceptance
redefinition could. E6b's measured descent (`-3.6e-8`/step) proves it is NOT
strict — which is precisely the regime where the nonmonotone (4a) and
flow-servoed (4b) literature applies. This is the scientific justification
for the plan below.

## 5. Option menu (all are saneamiento of existing components)

### S1 — Eta-endgame (config + trivial host code)

Lower the eta-axis `min_step` floor; walk `1 - eta` geometrically with warm
starts (each refined stage must still converge to 1e-6); optionally
extrapolate the eta-family to an `eta = 1` initial guess. Readout: the
achievable-eta frontier either extends (cliff climbable -> a path to a
gate-level `eta = 1` warm start) or a fold `eta* < 1` appears (decisive
diagnosis). Cost: config + one V100 session. Risk: none (pure schedule).

### S2 — State-machine hygiene (small code in SF-15/SF-24 logic)

(i) Rescue re-entry resets the persistent `omega` to its initial value (the
R2a design gap). (ii) Omega-collapse guard: an omega-floor rejection while
the trailing-window residual slope is still descending triggers a bounded,
structured omega reset instead of stage death (new exit accounting, no
silent behavior change). (iii) Anderson restart-on-stagnation, bounded and
counted. Rationale: R1/R2 deaths were machinery, not mathematics; these
fragilities confound every other probe. Cost: one worker node + unit tests.

### S3 — Terminal descent-flow saneamiento of the Newton phase (core option)

Behind config, terminal phase only: (i) monotone Armijo -> Li-Fukushima/GLL
nonmonotone acceptance with a bounded summable envelope; (ii) Newton
direction -> `(mu_k A + J)` direction with a conservative, descent-servoed
`mu_k` policy seeded by a PRESPECIFIED single-step mu-scan on the frozen E2
state over `mu in {0.3, 0.4, 0.5, 0.7, 1, 2}` (maps descent-rate vs `mu`
across the unexplored transition window and yields a step-count budget
estimate BEFORE any loop runs); (iii) a large-but-cheap step budget (E6b:
inner = 1 at `mu = 1`, so one step ~ one preconditioned GMRES inner + one
residual evaluation; 1e5 steps is minutes at 32^3 on V100); (iv) optional
geodesic-acceleration correction (one extra `F` evaluation per step) if the
scan shows curvature-limited steps; (v) an abort rule: sustained descent
rate below a floor -> honest BLOCKED. This is the paper's pseudo-time method
class expressed implicitly through the audited shift+GMRES+MG stack (the
A-metric preconditioned flow). Cost: the mu-scan probe is minutes on the
existing instrument; the policy change is one worker node on
`NewtonKrylovSolver` + the D-gate instrument as its proving ground before
promotion.

### S4 — Coarse-grid warm start (SF-27 essence pulled forward)

Verify the same-seed low-pass property 16^3 vs 32^3 (diagnostic print);
solve 16^3 at `lambda = 0.5125`; prolong with existing MG transfers;
warm-start the 32^3 stage. Readout: either a basin entry or a measured
`a*(h)` shift — both are scientific gains. Cost: small worker node (wiring a
prolongation call + probe case).

### S5 — Pseudo-arclength lambda leg (medium; diagnosis-first)

Bordered extension of the existing lambda continuation to answer THE
question: does the branch fold back at `a*` (no 32^3 solution past `a*` on
this branch -> only grid/discretization work can help) or continue
(basin/stiffness problem -> S3 justified)? Implement only if S1-S3 leave the
question open. Cost: medium (bordering algorithm on existing GMRES).

### S6 — Physical-quality measurement of the shelf / eta<1 pair (instrument-only)

SF-11 diagnostics + spatial localization of the defect on the frozen state:
reconstruction mismatch `||v - grad(psi1) x grad(psi2)|| / ||v||`, invariance
defect statistics and spatial histogram, gradient collinearity maps. Decides
consumability of the eta<1 pair (transport-in-reconstructed-field mode) AND
tests the localized-obstruction hypothesis for `a*`. Cost: small instrument
extension, minutes to run.

## 6. Recommended sequencing (for owner selection)

- **Phase 1 (one worker + one V100 session, all cheap, all prespecified):**
  S2 (hygiene) + S1 (eta-endgame run) + the S3 single-step mu-scan
  micro-probe + S6 (measurement). Four independent readouts, no policy
  commitment.
- **Phase 2 (gated on the mu-scan):** S3 full descent-flow policy, proven
  first inside the D-gate instrument (as the E5/E6 successor probe), then
  promoted into `NewtonKrylovSolver` behind config with compile-gate worker
  discipline and full audit.
- **Phase 3 (contingent):** S4 if S1-S3 do not traverse; S5 if the fold
  question remains open after that.

Rejected as out-of-mandate (new methods, not saneamiento): full FAS
nonlinear-multigrid rewrite; deflated continuation; swapping to external
nonlinear frameworks (PETSc SNES / SUNDIALS KINSOL); stochastic or
derivative-free global optimization.

## 7. References

- L. Grippo, F. Lampariello, S. Lucidi, "A nonmonotone line search technique
  for Newton's method", SIAM J. Numer. Anal. 23 (1986) 707-716.
- H. Zhang, W. W. Hager, "A nonmonotone line search technique and its
  application to unconstrained optimization", SIAM J. Optim. 14 (2004).
- D.-H. Li, M. Fukushima, "A derivative-free line search and global
  convergence of Broyden-like method for nonlinear equations", Optim.
  Methods Softw. 13 (2000) 181-201.
- C. T. Kelley, D. E. Keyes, "Convergence analysis of pseudo-transient
  continuation", SIAM J. Numer. Anal. 35 (1998) 508-523.
- M. K. Transtrum, B. B. Machta, J. P. Sethna, "Why are nonlinear fits to
  data so challenging?", Phys. Rev. Lett. 104 (2010) 060201; "Geometry of
  nonlinear least squares with applications to sloppy models and
  optimization", Phys. Rev. E 83 (2011) 036701.
- H. B. Keller, "Numerical solution of bifurcation and nonlinear eigenvalue
  problems" (1977); I. C. F. Ipsen et al., "Condition estimates for
  pseudo-arclength continuation", SIAM J. Numer. Anal. (2007).
- A. Toth, C. T. Kelley, "Convergence analysis for Anderson acceleration"
  (2015); S. Pollock, L. Rebholz, "Anderson acceleration for contractive and
  noncontractive operators" (2019/2021); J. A. Evans et al., damping
  heuristics (2020); "Nonmonotone globalization for Anderson acceleration
  via adaptive regularization", J. Sci. Comput. (2023).
- Homotopy endgames: HomotopyContinuation.jl endgame documentation
  (power-series and Cauchy endgames; geometric approach to singular
  endpoints).
- Nested iteration / FMG: A. Brandt, FAS multigrid; standard multigrid
  texts; nested-iteration basin-of-attraction practice.
- D. R. Lester et al. (2023), sec. 5.1: explicit variable-step pseudo-time
  integration to 1e-16 at 256^3, `l/h = 16` — the method-class benchmark for
  S3.

## 8. Relationship to prior escalation options

The earlier escalation offered (A2) explicit pseudo-time probe, (B2)
omega-rescue fix, (C2) BLOCKED closure. Under this record: (B2) == S2(i);
(A2) is subsumed by S3, whose implicit A-metric form uses MORE of our
existing stack than a literal explicit integrator, with the explicit form
retained inside S3 as a paper-parity control variant; (C2) remains available
if the owner prefers closure over remediation.
