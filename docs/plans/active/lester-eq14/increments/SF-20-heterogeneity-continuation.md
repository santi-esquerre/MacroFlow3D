# SF-20 — Heterogeneity continuation

- State: `active`
- Goal: `Implementar continuación adaptativa en heterogeneidad para campos gaussianos físicos.`
- Depends on: `SF-19`
- Unlocks: `SF-21`
- Branch: `science/lester-sf20-heterogeneity-continuation`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-10T18:27Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reach the target lognormal conductivity without assuming strongly heterogeneous
Picard convergence from a homogeneous initial state.

## Preconditions

- SF-19 provides periodic `Y`, Darcy flow, and target mean-flux semantics.

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
- [ ] Dashboard marks SF-20 complete and selects SF-21.
<!-- completion-checklist:end -->

## Advancement rule

SF-21 may prolong accepted lambda-one solutions to finer versions of the same
periodic realization.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-10T18:27Z | activation on `master=a515188` (SF-19 closure merged via PR #31) | SF-20 activated after verifying `NEXT: SF-20`, SF-19 `done`, and checker `OK (29 increments, next=SF-20)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Nested continuation architecture:** an OUTER lambda leg on the reusable SF-17 stage machine with the spec-locked axis (initial step 0.1, min 0.0125, max 0.2, halve on failure, grow 1.5 after two easy stages, exact clamp at 1); each lambda attempt is a warm-started solve at `(eta=1, epsilon=epsilon_start)` from the last ACCEPTED state; the epsilon leg runs ONLY after `lambda=1` is accepted at `eta=1` (the overview's staged-regularization rule: epsilon never changes during the lambda leg). Baseline stage: `lambda=0` exact-zero fluctuations (zero-source solve, homogeneous K=1). (2) **Eta rescue (spec item 3, exact ordering):** on a failed lambda attempt — restore the accepted state bitwise, solve the ATTEMPTED lambda at `eta=0` (warm-started), then ramp eta 0->1 with the SF-17 eta axis (warm-started stages, same locked eta numbers); rescue success ACCEPTS the lambda attempt; ANY rescue failure (eta-zero solve failure or eta-leg floor/failure) fails the lambda attempt -> restore + lambda-step halving; a minimum-lambda-step failure is a RECORDED structured physical/numerical failure and the failed interval is never skipped. (3) **Per-lambda problem construction:** the caller provides the SF-18 periodic `Y` once; the solver receives `Y_lambda = lambda*Y` with `ConductivityRepresentation::log_conductivity_y` (existing `q=exp(-Y)` kernel, no new q path); `K_lambda = exp(lambda*Y)` is built by a small kernel for the SF-19 affine-periodic Darcy solve, which is re-run PER ATTEMPTED LAMBDA VALUE to supply the reference velocity and measured `v_rms` (prescribed mean flux `(1,0,0)` => the benchmark gauge `vbar=1` is exact by construction); `q` and ONE MG hierarchy are rebuilt ONCE per lambda value and REUSED across every solve call at that lambda (eta-rescue ramps included) via a minimal default-preserving solver extension `CoefficientState{rebuild(default), reuse}` that skips the q-fill, hierarchy population, and affine-RHS assembly when the caller guarantees unchanged conductivity and gauge — a documented caller contract exactly parallel to SF-17's `warm_start` extension; the MG rebuild count is recorded per stage (spec item 4). Overflow note: for `sigma_Y^2 <= 4`, `lambda*Y` stays within ~±8, `K_lambda` within ~[3e-4, 3e3] — no overflow regime; `sigma_Y^2=6.25` is PROHIBITED per the spec. (4) **Stage history (Gate-3A metric set, spec threshold 3):** every attempt (lambda, rescue-eta, epsilon) appends one record with: axis + lambda/eta/epsilon values, attempted step, accepted flag + failure reason, exit reason, Picard iterations + final omega, r_F/r1/r2, `e_v` (velocity reconstruction), invariance errors, reconstructed-flow divergence, |c| percentiles + degeneracy explained/unexplained split, and the cumulative MG rebuild count. (5) **Scope split and fixtures (PRESPECIFIED before implementation):** (a) library driver `run_streamfunction_heterogeneity_continuation` extending `ContinuationController`; (b) ctest smoke: fixed seed 12345, 32^3, dx=1, ell=8 (ell/h=8, L/ell=4), sigma_Y^2 = 0.25 AND 1.0, GATE: reaches lambda=1 (every accepted stage r_F <= 1e-6 = picard tolerance, epsilon fixed 1e-2); (c) the 64^3 suite (dx=1, ell=8, ell/h=8, L/ell=8, seed 12345, sigma_Y^2 in {0.25, 1, 2.25, 4}) is run as a RECORDED EXPERIMENT (docs/experiments/) through the pipeline binary with exact configs/commands/artifacts, GATE: reaches lambda=1 for all four variances; epsilon legs to 1e-6 are additionally run and recorded where the local budget allows but are NOT a gate beyond the spec text; wall time and MG rebuild counts recorded; an honest failure at the minimum lambda step for any variance BLOCKS the increment with evidence rather than being relabeled. (6) **Pipeline surface (what makes the spec's pipeline validation command real):** extend the strict `streamfunction_solver` YAML with `field_source: stochastic (default, byte-identical current behavior) | periodic_gaussian {sigma2, corr_length, seed, normalize_variance}` and `darcy_source: pipeline (default) | affine_periodic`, plus a `continuation.lambda{enabled(false), start, initial_step, min_step, max_step, backtrack_factor, growth_factor, easy_streak}` subsection; all defaults preserve SF-19-era behavior byte-identically; new benchmark configs under `apps/` for the 32^3 smoke and the four 64^3 variances; stage_history export extended with the new record fields. (7) **Gate 4 interpretation (recorded):** SF-20 validates invariant CONSTRUCTION under physical heterogeneity in the smooth, locally isotropic, triply periodic regime; degeneracy populations must be interpreted against the Darcy-slow-zone split (Gate 3A); NO transport or transverse-dispersion claims are made or citable from this increment. | Base commit is this activation commit on `master=a515188`. Gate 1 + Gate 2 + Gate 3A + Gate 4 (interpretation) apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Known risk accepted: the sigma_Y^2=4 64^3 leg is the genuinely hard regime; its failure mode is a documented BLOCKED outcome, not a silent scope cut. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
