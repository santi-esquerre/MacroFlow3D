# SF-20 — Anderson acceleration

- State: `active`
- Goal: `Incorporar Anderson acceleration con profundidad configurable y salvaguardas.`
- Depends on: `SF-19`
- Unlocks: `SF-21`
- Branch: `science/lester-sf20-anderson`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-11T15:40Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Reduce Picard iterations without weakening the validated residual, gauge,
degeneracy, or rollback safeguards.

## Preconditions

- SF-19 provides periodic `Y` fields, affine-periodic Darcy flow, and the
  SF-17 warm-started continuation machinery (including the SF-20-era
  heterogeneity driver and `CoefficientState` extension, parked pending
  re-activation of the heterogeneity increment).
- Re-sequencing motivation (2026-08-11 owner decision, see
  `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`): plain
  adaptive Picard stalls asymptotically at full coupling `eta=1` on
  physical Gaussian fields (32^3, sigma_Y^2=0.25 stalls at lambda~0.37;
  sigma_Y^2=1 at lambda~0.10) with iteration counts diverging as
  `eta->1` — the fixed-point map's spectral radius approaches 1.

## In scope

- Anderson depth 3–8, default 5, start iteration 5, coupled history, small dense
  least-squares solve, conditioning reset, and Picard fallback.

## Out of scope

- Newton, concurrent block solves, and mixed precision.

## Files and symbols

- Add `AndersonAccelerator.cuh/.cu` and optional config/history fields.
- Store coupled `Delta X` and `Delta F`, four scalar fields per history level.

## Implementation specification

1. Form dot products on GPU and transfer only the small dense matrix/vector.
2. Solve the least-squares problem by pivoted QR; reject/reset if condition
   estimate exceeds `1e12`.
3. Project accelerated candidates and pass them through the same residual and
   degeneracy safeguard as Picard.
4. On any failed acceleration, clear history and accept/retry the normal Picard
   candidate according to SF-15.

## Expected numerical effect

The fixed benchmark suite converges in no more nonlinear iterations than Picard
and normally fewer, with the same final solution and diagnostics.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_anderson
scripts/remote exec -- "<fixed-picard-vs-anderson-benchmark>"
```

## Acceptance thresholds

- Depth 3/5/8 memory equals `4*m` scalar fields plus small dense storage.
- Ill-conditioned and rejected controls fall back safely.
- Default depth five does not increase iteration count on the acceptance suite
  and final fields agree with Picard within nonlinear tolerance.
- On the SF-20-era stall fixtures (32^3 physical Gaussian, ell=8, seed 12345,
  sigma_Y^2 = 0.25 and 1.0, eta=1, epsilon=1e-2), Anderson-accelerated
  Picard converges `r_F <= 1e-6` within the standard 500-iteration budget
  where plain Picard exhausted it.

## Regression surface

- GPU memory, history ordering, dense solve robustness, projection, and
  continuation rollback.

## Failure and rollback policy

- Anderson remains disabled by default if it does not improve the fixed suite.
- Never accept an accelerated candidate that fails the Picard safeguard.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Coupled history and pivoted least-squares solve are implemented.
- [ ] Projection, conditioning reset, rejection, and fallback tests pass.
- [ ] Memory for depths 3, 5, and 8 is measured.
- [ ] Fixed-suite comparison against Picard is recorded.
- [ ] Gate 3A review and full regressions pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-20 complete and selects SF-21.
<!-- completion-checklist:end -->

## Advancement rule

SF-21 may resume the heterogeneity continuation using Anderson-accelerated
Picard as the stage solver.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-11T14:10Z | re-sequenced into slot SF-20 (was SF-22) | Owner decision (option (a), 2026-08-11): pull Anderson acceleration BEFORE the heterogeneity continuation, motivated by the SF-20-era honest BLOCKED evidence (asymptotic Picard stall at eta=1 on physical Gaussian fields; see the SF-21 heterogeneity bitácora and `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`). Spec adjustments in this re-sequencing: Depends on SF-19; a new PRESPECIFIED acceptance threshold targeting the recorded stall fixtures; preconditions updated. All other gates unchanged. | The stall fixtures give Anderson a sharply defined, already-measured target: plain Picard needed >500 iterations at eta=1 (r_F stuck at 1-5x tolerance) where eta=0.95 needed only ~80. | Activate only when named by `NEXT` after this re-sequencing PR is merged. |
| 2026-08-11T15:40Z | activation on `master=577e21c` (re-sequencing merged via PR #32) | SF-20 (Anderson) activated after verifying `NEXT: SF-20`, SF-19 `done`, and checker `OK (29 increments, next=SF-20)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Insertion point and semantics:** Anderson type-II mixing over the COUPLED state `X=[u1;u2]` wraps the accepted SF-15 adaptive loop: each outer iteration's MAP step still produces the Picard candidate `u_hat` (f = u_hat - u); with Anderson enabled, history depth m of (DeltaX, DeltaF) pairs, and k >= start_iteration, an accelerated candidate `x_acc = x_k + f_k - (X_k + F_k) gamma` (gamma from the small least-squares min ||f_k - F_k gamma||) is formed FIRST, mean-zero projected, and evaluated through EXACTLY the SF-15 trial guard chain (nonfinite -> degeneracy -> percentile -> Armijo with the omega=1 arm); acceptance updates the state and appends history; ANY rejection clears the history and the iteration falls back to the normal SF-15 backtracking of the Picard candidate — Anderson never bypasses a safeguard and never introduces a new fixed point. (2) **Config:** `AndersonConfig{enabled=false, depth=5 (validated 3..8), start_iteration=5 (>=1), condition_limit=1e12}` composed into `StreamfunctionSolverConfig`; `enabled=false` is the bitwise-preserving default (every existing suite is the regression net); distinct validation messages. (3) **Memory and workspace:** history + scratch live in an optional workspace component allocated ONLY when enabled (disabled path allocates nothing, so the SF-12 closed-form workspace memory test stays unchanged on its fixtures; the closed form gains a documented conditional term for enabled configs); exact-byte accounting: history == 4*m*n*sizeof(real) (DeltaX,DeltaF for both components per level) + the small dense/gram scratch, reported per depth. (4) **Least squares:** GPU fixed-shape deterministic dot products form the m x m gram/rhs; the tiny dense solve is host-side pivoted QR (own routine, m <= 8) with condition estimate from the R diagonal; estimate > condition_limit => clear history and skip acceleration that iteration (recorded event counter). (5) **PRESPECIFIED fixtures/gates (fixed NOW, before implementation; never adjusted after a run):** (a) EQUIVALENCE: with `enabled=false` the entire existing suite must stay green/bitwise-unchanged. (b) NON-REGRESSION (convergent cases): trig a=0.5 32^3 and homogeneous 16^3 with Anderson depth 5 ON: picard_iterations(anderson) <= picard_iterations(plain), final r_F <= 1e-6, and fields agree with the plain-Picard solution: RMS(u_i^A - u_i^P) <= 1e-4 (100x nonlinear tolerance; fields are O(1e-2..1e-1) here) — the operationalization of the spec's "agree within nonlinear tolerance". (c) **STALL FIXTURES (the re-sequencing threshold):** Y from `generate_periodic_gaussian_field` (32^3, dx=1, ell=8, seed 12345, normalize_variance=true), scaled to the RECORDED failed lambda intervals and solved DIRECTLY (zero-source init, eta=1, epsilon=1e-2, solver defaults, budget 500, conductivity = lambda*Y via log representation, Darcy reference = SF-19 affine-periodic solve on exp(lambda*Y)): fixture A sigma_Y^2=1, lambda*=0.1125; fixture B sigma_Y^2=0.25, lambda*=0.3859375 (= last accepted + min step from the recorded runs). CONTROL: plain Picard (enabled=false) must exhaust the budget (`budget_exhausted`, certifying "where plain Picard exhausted it"; if a control unexpectedly converges, STOP and report — the fixture design is then invalid and must be revisited honestly, not tuned). GATE: Anderson depth 5 converges `r_F <= 1e-6` within the same 500 budget on BOTH fixtures. (d) MEMORY: measured history bytes == 4*m*n*sizeof(real) exactly for m = 3, 5, 8. (e) SAFEGUARDS: an injected ill-conditioned/rejected-candidate control falls back safely (history cleared, run still converges via the Picard path on a convergent fixture). (6) **Scope:** library (`AndersonAccelerator.cuh/.cu` + solver integration) + tests only; NO pipeline/YAML surface in SF-20 (that rides with the SF-21 heterogeneity re-activation); flow/stochastic modules read-only. (7) **Remote V100 use is authorized** for the expensive plain-Picard control runs (established precedent; local Debug is impractically slow for 500-iteration 32^3 budgets); runtimes recorded. | Base commit is this activation commit on `master=577e21c`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
