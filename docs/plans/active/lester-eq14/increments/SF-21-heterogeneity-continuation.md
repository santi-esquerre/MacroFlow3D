# SF-21 — Heterogeneity continuation

- State: `active`
- Goal: `Implementar continuación adaptativa en heterogeneidad para campos gaussianos físicos.`
- Depends on: `SF-20`
- Unlocks: `SF-22`
- Branch: `science/lester-sf21-heterogeneity-continuation`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-11T21:05Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the target lognormal conductivity without assuming strongly heterogeneous
Picard convergence from a homogeneous initial state.

## Preconditions

- SF-19 provides periodic `Y`, Darcy flow, and target mean-flux semantics.
- SF-20 provides Anderson acceleration validated on the recorded eta=1 stall
  fixtures (re-sequencing decision, 2026-08-11).

## In scope

- `K_lambda=exp(lambda*Y)`, q/hierarchy rebuilding per accepted stage, adaptive
  lambda steps, eta rescue, and small physical Gaussian runs.

## Out of scope

- Grid transfer, Anderson, `256^3`, and exponential covariance.

## Files and symbols

- Extend `ContinuationController`, solver config/history, and benchmark configs.
- Reuse the affine-periodic Darcy solve at each lambda stage where the reference
  velocity is required.

## Implementation specification

1. Begin at lambda zero with exact zero fluctuations and permit eta one.
2. Use lambda step `0.1`, minimum `0.0125`, maximum `0.2`; halve on failure and
   grow by `1.5` after two easy stages.
3. On failed lambda, restore the accepted state, solve the current lambda at
   eta zero, then ramp eta to one before retrying epsilon reduction.
4. Rebuild `q` and one MG hierarchy per lambda and reuse it across all nonlinear
   iterations/stages at that lambda.

## Expected numerical effect

Small smooth Gaussian cases reach lambda one with controlled residual and
physical metrics instead of catastrophic first-step divergence.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_continuation
./build/wsl-debug/macroflow3d_pipeline <fixed-32-or-64-gaussian-config>
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Fixed-seed `32^3` smoke reaches lambda one for `sigma_Y^2=0.25` and one.
- The `64^3` suite reaches lambda one for `0.25,1,2.25,4` with `ell/h=8`.
- Every accepted stage reports `r_F`, velocity, invariance, divergence,
  degeneracy, eta, epsilon, and MG rebuild count.

## Regression surface

- Continuation rollback, conductivity overflow/underflow, Darcy recomputation,
  hierarchy lifetime, and total runtime.

## Failure and rollback policy

- Minimum lambda-step failure is a recorded physical/numerical failure; do not
  skip to a later lambda.
- `sigma_Y^2=6.25` is prohibited in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Lambda stepping and hierarchy lifecycle are implemented.
- [ ] Eta rescue follows the documented ordering.
- [ ] Fixed-seed 32/64 Gaussian suites meet the acceptance set.
- [ ] Full Gate 3A metrics and experiment notes are recorded.
- [ ] Gate 4 interpretation and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-21 complete and selects SF-22.
<!-- completion-checklist:end -->

## Advancement rule

SF-22 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-10T18:27Z | activation on `master=a515188` (SF-19 closure merged via PR #31) | SF-20 activated after verifying `NEXT: SF-20`, SF-19 `done`, and checker `OK (29 increments, next=SF-20)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Nested continuation architecture:** an OUTER lambda leg on the reusable SF-17 stage machine with the spec-locked axis (initial step 0.1, min 0.0125, max 0.2, halve on failure, grow 1.5 after two easy stages, exact clamp at 1); each lambda attempt is a warm-started solve at `(eta=1, epsilon=epsilon_start)` from the last ACCEPTED state; the epsilon leg runs ONLY after `lambda=1` is accepted at `eta=1` (the overview's staged-regularization rule: epsilon never changes during the lambda leg). Baseline stage: `lambda=0` exact-zero fluctuations (zero-source solve, homogeneous K=1). (2) **Eta rescue (spec item 3, exact ordering):** on a failed lambda attempt — restore the accepted state bitwise, solve the ATTEMPTED lambda at `eta=0` (warm-started), then ramp eta 0->1 with the SF-17 eta axis (warm-started stages, same locked eta numbers); rescue success ACCEPTS the lambda attempt; ANY rescue failure (eta-zero solve failure or eta-leg floor/failure) fails the lambda attempt -> restore + lambda-step halving; a minimum-lambda-step failure is a RECORDED structured physical/numerical failure and the failed interval is never skipped. (3) **Per-lambda problem construction:** the caller provides the SF-18 periodic `Y` once; the solver receives `Y_lambda = lambda*Y` with `ConductivityRepresentation::log_conductivity_y` (existing `q=exp(-Y)` kernel, no new q path); `K_lambda = exp(lambda*Y)` is built by a small kernel for the SF-19 affine-periodic Darcy solve, which is re-run PER ATTEMPTED LAMBDA VALUE to supply the reference velocity and measured `v_rms` (prescribed mean flux `(1,0,0)` => the benchmark gauge `vbar=1` is exact by construction); `q` and ONE MG hierarchy are rebuilt ONCE per lambda value and REUSED across every solve call at that lambda (eta-rescue ramps included) via a minimal default-preserving solver extension `CoefficientState{rebuild(default), reuse}` that skips the q-fill, hierarchy population, and affine-RHS assembly when the caller guarantees unchanged conductivity and gauge — a documented caller contract exactly parallel to SF-17's `warm_start` extension; the MG rebuild count is recorded per stage (spec item 4). Overflow note: for `sigma_Y^2 <= 4`, `lambda*Y` stays within ~±8, `K_lambda` within ~[3e-4, 3e3] — no overflow regime; `sigma_Y^2=6.25` is PROHIBITED per the spec. (4) **Stage history (Gate-3A metric set, spec threshold 3):** every attempt (lambda, rescue-eta, epsilon) appends one record with: axis + lambda/eta/epsilon values, attempted step, accepted flag + failure reason, exit reason, Picard iterations + final omega, r_F/r1/r2, `e_v` (velocity reconstruction), invariance errors, reconstructed-flow divergence, |c| percentiles + degeneracy explained/unexplained split, and the cumulative MG rebuild count. (5) **Scope split and fixtures (PRESPECIFIED before implementation):** (a) library driver `run_streamfunction_heterogeneity_continuation` extending `ContinuationController`; (b) ctest smoke: fixed seed 12345, 32^3, dx=1, ell=8 (ell/h=8, L/ell=4), sigma_Y^2 = 0.25 AND 1.0, GATE: reaches lambda=1 (every accepted stage r_F <= 1e-6 = picard tolerance, epsilon fixed 1e-2); (c) the 64^3 suite (dx=1, ell=8, ell/h=8, L/ell=8, seed 12345, sigma_Y^2 in {0.25, 1, 2.25, 4}) is run as a RECORDED EXPERIMENT (docs/experiments/) through the pipeline binary with exact configs/commands/artifacts, GATE: reaches lambda=1 for all four variances; epsilon legs to 1e-6 are additionally run and recorded where the local budget allows but are NOT a gate beyond the spec text; wall time and MG rebuild counts recorded; an honest failure at the minimum lambda step for any variance BLOCKS the increment with evidence rather than being relabeled. (6) **Pipeline surface (what makes the spec's pipeline validation command real):** extend the strict `streamfunction_solver` YAML with `field_source: stochastic (default, byte-identical current behavior) | periodic_gaussian {sigma2, corr_length, seed, normalize_variance}` and `darcy_source: pipeline (default) | affine_periodic`, plus a `continuation.lambda{enabled(false), start, initial_step, min_step, max_step, backtrack_factor, growth_factor, easy_streak}` subsection; all defaults preserve SF-19-era behavior byte-identically; new benchmark configs under `apps/` for the 32^3 smoke and the four 64^3 variances; stage_history export extended with the new record fields. (7) **Gate 4 interpretation (recorded):** SF-20 validates invariant CONSTRUCTION under physical heterogeneity in the smooth, locally isotropic, triply periodic regime; degeneracy populations must be interpreted against the Darcy-slow-zone split (Gate 3A); NO transport or transverse-dispersion claims are made or citable from this increment. | Base commit is this activation commit on `master=a515188`. Gate 1 + Gate 2 + Gate 3A + Gate 4 (interpretation) apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Known risk accepted: the sigma_Y^2=4 64^3 leg is the genuinely hard regime; its failure mode is a documented BLOCKED outcome, not a silent scope cut. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-11T13:30Z | PRESPECIFIED 32^3 smoke gates FAILED honestly — increment BLOCKED pending owner decision | Both decision-5(b) smokes were run VERBATIM (no tuning) on the remote V100 (v100-release, sm_70) after the local Debug run proved impractically slow; tree = base+T01+T02 (`dcabc25`). **Results: BOTH FAIL with `lambda_floor_exhausted`.** sigma_Y^2=0.25: final_lambda=0.3734, 161 stages (100 accepted, 134 rescue), 27 MG rebuilds, 7.75 h. sigma_Y^2=1: final_lambda=0.10, 100 stages (60 accepted, 89 rescue), 11 rebuilds, 4.55 h. Integrity intact: EVERY accepted stage honored r_F <= 1e-6 (that sub-check PASSED); no degeneracy anywhere (unexplained=0, |c| p0.1% ~ 0.85-0.96 of v_rms); physical metrics clean and smoothly varying (e_v ~ 6e-4..2e-3, invariance ~ 2e-4..1.5e-3, e_div ~ 4e-5..2e-4). **Diagnosis (orchestrator):** a pure asymptotic-rate stall of adaptive Picard at FULL nonlinear coupling: within each eta rescue the ramp sails to eta=0.95 (14-80 iterations/stage) and then iteration counts DIVERGE as eta->1 (0.98125: ~150; 0.996875: ~450; 1.0: budget-exhausted at 500 with r_F stuck at 1-5x tolerance) — the textbook signature of the fixed-point map's spectral radius approaching 1 at eta=1, worsening with lambda; there is NO catastrophic divergence, NO guard trips, NO degeneracy — the spec's anticipated failure mode ("catastrophic first-step divergence") did not occur; what fails is the CONVERGENCE RATE of the dashboard-locked Picard machinery (max_iter=500, tolerance 1e-6, SF-15 adaptive omega) at eta=1. The mechanism SF-20 built (lambda stepping, exact rescue ordering, one-hierarchy-per-lambda with 27 rebuilds over 161 stages, bitwise rollback, complete Gate-3A stage records) operated exactly as specified throughout. | Per the spec's failure policy ("minimum lambda-step failure is a recorded physical/numerical failure; do not skip") and activation decision 5, this is recorded as a BLOCKED outcome with evidence, NOT relabeled and NOT tuned post hoc (budgets/tolerances/omega dynamics are dashboard-locked; amending them after seeing results would violate the prespecification discipline). Scientific reading: the plan's own architecture anticipated Picard's asymptotic insufficiency — Anderson acceleration (SF-22) exists to accelerate exactly this coupled fixed-point map and Newton-Krylov (SF-25..27) for terminal convergence — but the empirical finding is that plain adaptive Picard already stalls at sigma_Y^2=0.25, 32^3, i.e. EARLIER in the plan than the sequencing assumed. T04 (64^3 suite incl. sigma^2=4) was NOT run: it would fail harder at large cost. Evidence: remote job `sf20-smokes` log (12.3 h total), stage tables with full Gate-3A metrics per stage. | Owner decision required: (a) re-sequence the plan (pull Anderson/Newton before heterogeneity continuation) and re-scope SF-20's gates accordingly via a reviewed spec change; (b) amend the SF-20 gate parameters (e.g. Picard budget) via a justified, versioned spec amendment; or (c) other direction. No further SF-20 execution until the owner decides. |
| 2026-08-11T14:10Z | re-sequenced into slot SF-21 (was SF-20); state blocked -> pending | Owner decision (option (a)): Anderson (now SF-20) runs first; this increment re-activates afterwards with UNCHANGED gates (the 32^3 smokes and 64^3 suite stand as specified). The already-built and orchestrator-audited machinery is PARKED for reuse at re-activation: T01 library (CoefficientState + lambda/rescue driver) `88076a0` [accepted], T02 tests+smokes `dcabc25` [unit cases accepted; smoke gates honestly failed], T03 pipeline surface `8305c10` [recovered, pending full validation] — all in their Claude worker worktrees on top of activation `0177ead`; expect a rebase over the merged Anderson work. The blocked-evidence row above remains the scientific record of WHY. | Re-activation should wire Anderson into the heterogeneity stage solver per the SF-20 (Anderson) deliverables. | Wait for SF-20 (Anderson) to complete; then re-activate under the standard protocol. |
| 2026-08-11T21:05Z | RE-ACTIVATION on `master=0c87e3b` (SF-20 Anderson closure merged via PR #33; checker `OK (29 increments, next=SF-21)`) | Re-activated after deep UNDERSTAND (Lester 2023 pp. 1-10 read from the primary PDF: eq. (10) helicity identity, (11)-(13) integrability and dual-streamfunction representation, eq. (14) with S_i/B, the §4 zero-D22/D33 surface-integral proof, and §5.1's own numerical method — homogenized Krylov initial estimate then explicit variable-step pseudo-time to 1e-16; our recorded eta=1 stall is the damped-fixed-point analogue of that stiffness, and Anderson is the prespecified remedy validated on the exact recorded stall fixtures in 64/88 iterations). Re-activation decisions, PRESPECIFIED before any run: (R1) reuse the parked audited machinery by porting T01 `88076a0` -> T02 `dcabc25` -> T03 `8305c10` onto this base; the seven 2026-08-10 activation decisions remain binding except where superseded here; rebase conflict surface is exactly {StreamfunctionSolver.cu/.cuh, StreamfunctionTypes.hpp, StreamfunctionWorkspace.cu/.cuh} (CoefficientState gates the setup phase; Anderson lives in the loop). (R2) **Anderson wiring:** (a) `solve_streamfunctions` clears the Anderson history at solve entry when enabled — the accelerator's premise is the history of ONE fixed-point map instance, and continuation stages are different maps (different lambda/eta/Darcy reference/v_rms); today's code has no entry-clear, so stage k+1 would inherit stage k's history and staging, its first delta pair straddling the parameter change; entry-clear is a no-op for fresh accelerators, preserving every SF-20 single-call behavior bitwise, and gets its own unit contract; (b) library default stays `anderson.enabled=false`; the SF-21 smoke fixtures and all five benchmark configs enable it explicitly with the SF-20-validated defaults depth=5, start_iteration=5, condition_limit=1e12 — no per-case tuning after seeing results; (c) `anderson.enabled` stays constant across all stages of one continuation run (workspace `prepared_for` identity includes it; per-stage flipping would force re-prepare = allocation in the loop); eta=0 rescue solves converge in < start_iteration iterations so Anderson is naturally inert there, no special-casing; (d) T03 gains a strict-YAML `streamfunction_solver.anderson{enabled(false), depth, start_iteration, condition_limit}` subsection, defaults byte-preserving. (R3) Gates UNCHANGED per the owner re-sequencing decision: identical fixtures, seeds, axes, budgets, tolerances, and thresholds as the 2026-08-10 activation; sigma_Y^2=6.25 prohibited; a lambda-floor failure is again a structured BLOCKED outcome, never relabeled — the honest prior is that lambda->1 at sigma^2>=1 is beyond the tested Anderson territory (the stall fixtures sat at the old floor points lambda*~0.11/0.39). (R4) Validation split (adopted practice): 32^3 smoke gates + full suite + 64^3 experiment on the V100 (checksum-verified sync, md5 spot checks); local Debug build + cheap ctest tier + byte-compare pipeline runs; the heterogeneity smoke ctest entry joins the SF-20 heavy registry (excluded from the default local tier). (R5) T02 port adaptations: adopt the SF-20 heavy/cheap registry mechanism (parked T02 predates it), enable Anderson per R2b, add the entry-clear cross-stage unit contract; the four unit/injection cases stand unchanged in substance. (R6) Spec-discrepancy record: the spec line "Out of scope: ... Anderson" predates the re-sequencing; interpretation recorded rather than silently chosen — Anderson *algorithm development* is out of scope (delivered by SF-20); *enabling the validated accelerator* in this increment's stage solves is in scope per the 2026-08-11T14:10Z owner decision and `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`; the spec text itself is not modified. (R7) The 64^3 suite runs as the recorded experiment on the final audited head (docs/experiments note with exact configs/commands/artifacts); epsilon legs to 1e-6 recorded where budget allows, not a gate beyond the spec text. | Base commit is this activation commit on `master=0c87e3b`. Gate 1 + Gate 2 + Gate 3A + Gate 4 apply; human review required, so the PR stops at `awaiting_review` with `NEXT` unchanged. | Build the intra-increment DAG; delegate the T01 port (+ entry-clear) to an isolated worker, then T02/T03 ports in parallel. |
