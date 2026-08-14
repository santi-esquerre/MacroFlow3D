# SF-25 — Manifold-robust terminal solver

- State: `active`
- Goal: `Implementar el solver terminal robusto a la variedad de gauge en eta=1: Newton desplazado (mu*A + J) con schedule Levenberg-Marquardt y contingencia pseudo-transitoria, activado por el diagnostico D-gate.`
- Depends on: `SF-24`
- Unlocks: `SF-26`
- Branch: `science/lester-sf25-terminal-manifold-solver`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Code orchestrator (Fable) + delegated workers`
- Started: `2026-08-14`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Cross the `eta = 1` plateau that defeats both the damped Picard/Anderson
map (SF-21: non-contractive for `sigma_Y^2 >= 1` beyond `lambda ~ 0.5`)
and pure Newton-Krylov (SF-26 Gate A forensics: every inner GMRES solve
budget-exhausts; a 10x-budget probe did not complete). Working hypothesis
(falsifiable, tested FIRST by this increment's D-gate): at `eta = 1` the
Clebsch pair's recombination gauge freedom produces a solution MANIFOLD
whose tangent space renders the Jacobian near-singular in a large mode
cluster; the shift `(mu*A_blk + J)` moves that cluster to `~mu` in the
preconditioned spectrum, restoring Krylov convergence with the accepted
SF-23 preconditioner untouched. Full analysis, method survey, and
references: `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md`.

## Preconditions

- SF-24 `done` (Newton phase, guard chain, rescue protocol — the terminal
  schedule extends it and recovers it bitwise at `theta = 0`).
- The SF-26-carried wiring (per-stage attribution counters, smoke Newton
  wiring, five-config enablement) merged with the restructuring PR — the
  D-gate's frozen-plateau protocol and forensics depend on it.

## In scope

- D-gate diagnostic (frozen plateau state; mu-sweep; spectral probe; LM
  mini-solve) BEFORE any solver integration.
- `ShiftedJacobianOperator` (apply = Jv + mu*A per component) and the
  duck-typed operator generalization of `CoupledGmres::solve` (defaulted,
  zero impact on accepted call sites).
- Terminal schedule in the Newton phase: LM `mu_k = theta * r_F_k`
  (primary; theta calibrated by the D-gate), config surface, reports,
  memory accounting; `theta = 0` bitwise-recovers SF-24.
- SER/Ψtc schedule as CONTINGENCY ONLY (triggered iff the LM mini-solve
  fails while the mu-sweep confirms the mechanism), with its own recorded
  decision before implementation.
- Acceptance demonstration: a NEW test fixture (not the SF-26-owned smoke
  case) running the sigma^2=1 32^3 continuation with the terminal
  schedule enabled to `lambda = 1`.

## Out of scope

- The SF-26 moved-verbatim gates themselves (re-imposed by SF-26 with the
  sanctioned wiring change there).
- Gauge-fixed/bordered reformulations (the recorded next research
  direction if the D-gate falsifies the shift mechanism).
- Explicit pseudo-time; mixed precision; grids beyond 32^3 fixtures;
  preconditioner changes.

## Files and symbols

- Add `src/physics/streamfunctions/ShiftedJacobianOperator.cuh/.cu` (or
  colocated in `NewtonKrylovSolver.*` if cleaner).
- Extend `CoupledGmres::solve` with a defaulted duck-typed operator
  template parameter (accepted call sites unchanged).
- Extend `NewtonKrylovConfig` with the `terminal` sub-config
  (`theta`, `mu_min`, `mu_max`, mode; defaults preserving SF-24 bitwise).
- New `tests/streamfunctions/terminal_solver_gpu_cases.cu` + registry +
  ctest entries (cheap diagnostic-unit cases + heavy demonstration).

## Implementation specification

1. D-GATE FIRST (heavy diagnostic case, orchestrator-run on the V100):
   freeze the plateau state deterministically (warm-start `lambda=0.5125`
   from the accepted `(0.5, 1)` sigma^2=1 smoke state; run the accepted
   Picard/Anderson stage to its stagnation exit); then (a) mu-sweep
   `mu in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0}` with the
   accepted GMRES (restart 10, budget 100), recording per-mu
   iterations/achieved reduction; (b) shifted-power spectral probe of
   `M^-1 J`'s smallest modes; (c) LM mini-solve with theta calibrated
   from (a).
2. Only after the D-gate passes: the operator wrapper, the GMRES
   generalization, the LM schedule inside the accepted Newton phase
   (rescue/retry protocol and trial guard chain UNCHANGED; Armijo stays).
3. Shifted-apply exactness is unit-gated against an independent
   recompute; `theta = 0` bitwise equivalence with SF-24 is gated.
4. The demonstration fixture runs the full sigma^2=1 32^3 continuation
   with terminal enabled.

## Expected numerical effect

At the plateau, GMRES on the shifted system converges in O(10) iterations
instead of budget-exhausting; LM steps descend superlinearly onto the
solution manifold (any manifold point is physically valid — every
acceptance metric is gauge-invariant); the sigma^2=1 continuation reaches
`lambda = 1` with per-stage `r_F <= 1e-6`.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/v100-release --output-on-failure -R streamfunction_terminal
./build/v100-release/streamfunction_operator_tests --case terminal_dgate_diagnostic
./build/v100-release/streamfunction_operator_tests --case terminal_sigma1_demonstration
```

(Validation executes on the remote V100 per the standing venue policy;
workers are compile-gate only.)

## Acceptance thresholds

- D-gate: some `mu > 0` restores GMRES convergence with `>= 10x` fewer
  iterations than `mu = 0` on the frozen plateau system; the LM
  mini-solve reaches `r_F <= 1e-4` (10x below the observed plateau floor)
  within its recorded budget. FAILURE of the mu-sweep = the hypothesis is
  falsified: report BLOCKED with the spectral evidence (no downstream
  gate may be weakened).
- Shifted-apply exactness to roundoff; `theta = 0` bitwise == SF-24 path;
  determinism; exact memory accounting (zero when disabled).
- Demonstration: sigma^2=1 32^3 continuation with terminal enabled
  reaches `lambda = 1`, every accepted stage `r_F <= 1e-6`.
- No regression: full suite green except the known SF-26-owned smoke
  entry; default-pipeline byte-compares clean.

## Regression surface

- Accepted SF-24 Newton behavior (theta = 0 bitwise), CoupledGmres
  call-site compatibility, Newton-phase memory accounting, continuation
  interaction, suite runtimes.

## Failure and rollback policy

- The D-gate decides before solver work is written; a falsified mechanism
  is a structured BLOCKED outcome, not a tuning license.
- The SER contingency requires its own recorded decision (non-monotone
  acceptance interacts with the guard chain) BEFORE implementation.
- No SF-26 gate, seed, tolerance, budget, or fixture value may change.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] D-gate diagnostic executed and passed (or BLOCKED honestly reported).
- [ ] Shifted operator + GMRES generalization implemented and unit-gated.
- [ ] LM terminal schedule integrated; theta=0 bitwise equivalence gated.
- [ ] sigma^2=1 32^3 demonstration reaches lambda=1 (r_F<=1e-6 per stage).
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-25 complete and selects SF-26.
<!-- completion-checklist:end -->

## Advancement rule

SF-26 re-imposes the moved-verbatim heterogeneity gates with the terminal
schedule as its sanctioned wiring change once SF-25 is accepted and merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-14T13:30Z | created by the owner-directed option (B) restructuring | This increment exists because the accepted SF-22..24 Newton-Krylov stack, correctly wired (SF-26 Gate A evidence), cannot cross the `eta=1` plateau for `sigma_Y^2>=1`: per-stage attribution showed every inner GMRES budget-exhausting (~112 Jv/accepted step) and a prespecified 10x-budget probe timed out without completing. The research record (`docs/decisions/2026-08-14-manifold-robust-terminal-solver.md`) derives the gauge-manifold hypothesis (Clebsch recombination freedom => near-singular Jacobian cluster exactly at eta=1), surveys Ψtc (Kelley-Keyes) and inexact LM under local error bounds (Dan-Yamashita-Fukushima) — both reducing to the SAME shifted system `(mu*A+J)delta=-F` whose A-metric shift restores the preconditioned spectrum with the SF-23 preconditioner untouched — and fixes the diagnostic-first plan: the falsifiable D-gate decides before any solver code. | Gates prespecified in this file; the mu-sweep grid, frozen-state protocol, and thresholds are bound BEFORE any run. | Activate only when named by `NEXT` after the restructuring PR is merged. |
| 2026-08-14T15:20Z | activation on `master=fc7a48b` (restructure merged; checker OK next=SF-25) | PRESPECIFIED protocol refinements, bound BEFORE any implementation or run (decisions E1-E5 of this increment): **E1 (D-gate enabler reading):** the mu-sweep requires the shifted-apply MECHANISM, so the duck-typed operator parameter on `CoupledGmres::solve` (defaulted to `JvpWorkspace`, all accepted call sites unchanged) and the standalone `ShiftedJacobianOperator` module are D-gate ENABLERS shipped before the diagnostic; what remains gated behind the D-gate is the TERMINAL SCHEDULE integration into the Newton phase (the behavioral solver change). **E2 (freeze protocol):** the plateau state is frozen with newton DISABLED (the SF-21-characterized regime, and ~4x cheaper): run the sigma^2=1 32^3 smoke continuation (seed 12345, ell=8, epsilon 1e-2, Anderson R5 config) to its lambda floor (~1580 s, deterministic), take the accepted `(lambda=0.5, eta=1)` state, warm-start the `lambda=0.5125` problem, run the accepted Picard/Anderson stage to its stagnation exit; that state is THE frozen plateau instance. **E3 (mu-sweep):** solve `(mu*A_blk + J)delta = -F` at `mu in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0}` with the accepted GMRES (restart 10, budget 100, rel_tol = sqrt(r_F_frozen) clamped to [1e-8,1e-1] — the SF-24 forcing value at the frozen state, so the sweep answers exactly the question the Newton phase asks); record per-mu status/inner iterations/true-residual reduction. MECHANISM CONFIRMED iff some mu>0 converges with >=10x fewer iterations than mu=0's budget exhaustion. **E4 (spectral probe):** generalized inverse iteration toward the smallest (J, A_blk) eigenpair using the shifted solve itself (v_{k+1} = normalize(solve(mu*A+J, A_blk v_k)) at mu=1e-3, 20 iterations), reporting the generalized Rayleigh quotient trajectory <v,Jv>/<v,A v>; hypothesis-supporting if |lambda_min^gen| <= 1e-2 (vs O(1) regular modes). **E5 (LM mini-solve + theta rule):** theta = mu*/r_F_frozen with mu* = the SMALLEST swept mu that converged within budget; bounded shifted-Newton loop (max 30 steps, Armijo + the identical guard chain, test-local driver) must reach r_F <= 1e-4. D-GATE VERDICT: E3 fails => hypothesis falsified => BLOCKED with the E4 evidence; E3 passes but E5 fails => SER contingency decision point (own recorded decision); both pass => proceed to the terminal-schedule integration. Venue: workers compile-gate only; every run orchestrator-driven on the V100. | DAG: T01 enablers (CoupledGmres operator param + ShiftedJacobianOperator) -> T02 diagnostic test case -> D-GATE (orchestrator, V100) -> [T03 terminal schedule + T04 tests + T05 io surface] -> I01 -> F01 (incl. sigma^2=1 demonstration). | Delegate T01; regenerate base refs at fc7a48b. |
| 2026-08-14T12:10Z | D-GATE FIRST RUN on chain `d11f63a->4e22528->65e7a6f->9985f90` (V100 job `sf25-dgate`; raw log `/tmp/dg_main.log`; unit case PASS with max_diff=0 and bitwise mu=0 passthrough) | **E2 freeze:** reproduced exactly (lambda_floor_exhausted, 0.5, eta=1, 83 stages); the lambda=0.5125 stage froze at r_F=1.1205e-3 (in the prespecified band) but exited `omega_floor_rejected` (k=20) rather than `stagnated` — BOTH are documented SF-21 plateau signatures (the ramp attempts stagnated; the direct lambda attempt hits the omega floor first: every Picard trial rejected). AMENDMENT E6a (methodology, recorded pre-corrective): the E2 exit-reason assert accepts {stagnated, omega_floor_rejected}. **E3 — MECHANISM CONFIRMED:** mu=0.1 -> 5 inner iterations (20x under the mu=0 budget exhaustion), mu=0.03 -> 12, mu=0.01 -> 38, mu<=0.003 -> budget-exhausted; gate PASS. **E4 — the near-null cluster is real and SIGNED:** generalized Rayleigh converges to lambda_min^gen = -2.059e-3 — ~500x below the O(1) regular modes AND NEGATIVE (J indefinite at the plateau); quantitative cross-consistency: the E3 convergence cliff sits exactly where mu crosses |lambda_min| (shifted mode mu+lambda crosses zero at mu~2e-3). **E5 — FAILED at k=0, informatively:** theta=8.92, GMRES converged (38 its), but EVERY alpha in the ladder was Armijo-rejected with r_F_trial approaching r_F_k from above (directional slope ~0+ at alpha->0). Combined with the freeze's omega-floor (Picard trials all rejected too), the evidence identifies the plateau as a SPURIOUS LOCAL MINIMUM of the merit 0.5*||F||^2 (J^T F ~ 0 with F != 0, J indefinite) — precisely the case the Kelley-Keyes Psi-tc theory addresses ("line-search globalization stagnates at local minima; pseudo-transient continuation succeeds") and the case NO monotone-Armijo method can escape. Per the PRESPECIFIED decision tree (activation E5 rule): E3-pass + E5-fail => the SER contingency decision point. **DECISION E6 (recorded HERE, before the corrective run):** extend the diagnostic with a Psi-tc/backward-Euler probe at the SAME frozen state: loop k<=60: evaluate r_F (PASS at <=1e-4); mu_k = 1/dtau_k; solve (mu_k*A+J)delta=-F (same GMRES config, rel_tol clamp(sqrt(r_F_k),1e-8,1e-1)); candidate = P(state+delta) accepted WITHOUT Armijo but under STRICT bounded-non-monotone safeguards: nonfinite OR r_F_new > 2*r_F_k => reject, dtau/2, retry (<=3 per step, else FAIL); global divergence stop r_F > 10*r_F_frozen => FAIL; SER update dtau_{k+1} = clamp(dtau_k * r_F_{k-1}/r_F_k, 1, 1e6); dtau_0 = 1/mu_star = 100 (E3-calibrated). The D-gate verdict rule per the prespecified tree becomes: PASS iff E2 (amended) AND E3 AND (E5 OR E6). The probe is falsifiable: the flow's LINEARIZATION at the frozen state is weakly unstable along the -2e-3 mode, so E6 tests precisely whether the nonlinear flow traverses the ridge — if it also fails, the increment reports the structured evidence and the gauge-fixed/bordered formulation becomes the recorded direction. Scope note: E6 is a DIAGNOSTIC (test-local); any production non-monotone acceptance remains a separate design surface for the PR review. | Raw numbers retained above and in the remote log; the E4 trajectory (20 values) converges to 5 significant figures. | Corrective C02: E6a assert amendment + the E6 probe section; rerun the diagnostic. |
