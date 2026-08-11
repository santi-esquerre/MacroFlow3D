# SF-20 — Anderson acceleration

- State: `done`
- Goal: `Incorporar Anderson acceleration con profundidad configurable y salvaguardas.`
- Depends on: `SF-19`
- Unlocks: `SF-21`
- Branch: `science/lester-sf20-anderson`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-11T15:40Z`
- Completed: `2026-08-11 (explicit owner approval of PR #33; closure metadata commit on the same PR)`
- PR: [#33 — SF-20: Anderson acceleration — type-II mixing over the coupled Picard map with full safeguard reuse; converges the recorded Picard-stall fixtures](https://github.com/santi-esquerre/MacroFlow3D/pull/33)
- Commit: `147b325` `(frozen audited source head; later branch commits are increment-state documentation only)`

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
- [x] Coupled history and pivoted least-squares solve are implemented.
- [x] Projection, conditioning reset, rejection, and fallback tests pass.
- [x] Memory for depths 3, 5, and 8 is measured.
- [x] Fixed-suite comparison against Picard is recorded.
- [x] Gate 3A review and full regressions pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-20 complete and selects SF-21.
<!-- completion-checklist:end -->

## Advancement rule

SF-21 may resume the heterogeneity continuation using Anderson-accelerated
Picard as the stage solver.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-11T14:10Z | re-sequenced into slot SF-20 (was SF-22) | Owner decision (option (a), 2026-08-11): pull Anderson acceleration BEFORE the heterogeneity continuation, motivated by the SF-20-era honest BLOCKED evidence (asymptotic Picard stall at eta=1 on physical Gaussian fields; see the SF-21 heterogeneity bitácora and `docs/decisions/2026-08-11-anderson-before-heterogeneity.md`). Spec adjustments in this re-sequencing: Depends on SF-19; a new PRESPECIFIED acceptance threshold targeting the recorded stall fixtures; preconditions updated. All other gates unchanged. | The stall fixtures give Anderson a sharply defined, already-measured target: plain Picard needed >500 iterations at eta=1 (r_F stuck at 1-5x tolerance) where eta=0.95 needed only ~80. | Activate only when named by `NEXT` after this re-sequencing PR is merged. |
| 2026-08-11T15:40Z | activation on `master=577e21c` (re-sequencing merged via PR #32) | SF-20 (Anderson) activated after verifying `NEXT: SF-20`, SF-19 `done`, and checker `OK (29 increments, next=SF-20)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Insertion point and semantics:** Anderson type-II mixing over the COUPLED state `X=[u1;u2]` wraps the accepted SF-15 adaptive loop: each outer iteration's MAP step still produces the Picard candidate `u_hat` (f = u_hat - u); with Anderson enabled, history depth m of (DeltaX, DeltaF) pairs, and k >= start_iteration, an accelerated candidate `x_acc = x_k + f_k - (X_k + F_k) gamma` (gamma from the small least-squares min ||f_k - F_k gamma||) is formed FIRST, mean-zero projected, and evaluated through EXACTLY the SF-15 trial guard chain (nonfinite -> degeneracy -> percentile -> Armijo with the omega=1 arm); acceptance updates the state and appends history; ANY rejection clears the history and the iteration falls back to the normal SF-15 backtracking of the Picard candidate — Anderson never bypasses a safeguard and never introduces a new fixed point. (2) **Config:** `AndersonConfig{enabled=false, depth=5 (validated 3..8), start_iteration=5 (>=1), condition_limit=1e12}` composed into `StreamfunctionSolverConfig`; `enabled=false` is the bitwise-preserving default (every existing suite is the regression net); distinct validation messages. (3) **Memory and workspace:** history + scratch live in an optional workspace component allocated ONLY when enabled (disabled path allocates nothing, so the SF-12 closed-form workspace memory test stays unchanged on its fixtures; the closed form gains a documented conditional term for enabled configs); exact-byte accounting: history == 4*m*n*sizeof(real) (DeltaX,DeltaF for both components per level) + the small dense/gram scratch, reported per depth. (4) **Least squares:** GPU fixed-shape deterministic dot products form the m x m gram/rhs; the tiny dense solve is host-side pivoted QR (own routine, m <= 8) with condition estimate from the R diagonal; estimate > condition_limit => clear history and skip acceleration that iteration (recorded event counter). (5) **PRESPECIFIED fixtures/gates (fixed NOW, before implementation; never adjusted after a run):** (a) EQUIVALENCE: with `enabled=false` the entire existing suite must stay green/bitwise-unchanged. (b) NON-REGRESSION (convergent cases): trig a=0.5 32^3 and homogeneous 16^3 with Anderson depth 5 ON: picard_iterations(anderson) <= picard_iterations(plain), final r_F <= 1e-6, and fields agree with the plain-Picard solution: RMS(u_i^A - u_i^P) <= 1e-4 (100x nonlinear tolerance; fields are O(1e-2..1e-1) here) — the operationalization of the spec's "agree within nonlinear tolerance". (c) **STALL FIXTURES (the re-sequencing threshold):** Y from `generate_periodic_gaussian_field` (32^3, dx=1, ell=8, seed 12345, normalize_variance=true), scaled to the RECORDED failed lambda intervals and solved DIRECTLY (zero-source init, eta=1, epsilon=1e-2, solver defaults, budget 500, conductivity = lambda*Y via log representation, Darcy reference = SF-19 affine-periodic solve on exp(lambda*Y)): fixture A sigma_Y^2=1, lambda*=0.1125; fixture B sigma_Y^2=0.25, lambda*=0.3859375 (= last accepted + min step from the recorded runs). CONTROL: plain Picard (enabled=false) must exhaust the budget (`budget_exhausted`, certifying "where plain Picard exhausted it"; if a control unexpectedly converges, STOP and report — the fixture design is then invalid and must be revisited honestly, not tuned). GATE: Anderson depth 5 converges `r_F <= 1e-6` within the same 500 budget on BOTH fixtures. (d) MEMORY: measured history bytes == 4*m*n*sizeof(real) exactly for m = 3, 5, 8. (e) SAFEGUARDS: an injected ill-conditioned/rejected-candidate control falls back safely (history cleared, run still converges via the Picard path on a convergent fixture). (6) **Scope:** library (`AndersonAccelerator.cuh/.cu` + solver integration) + tests only; NO pipeline/YAML surface in SF-20 (that rides with the SF-21 heterogeneity re-activation); flow/stochastic modules read-only. (7) **Remote V100 use is authorized** for the expensive plain-Picard control runs (established precedent; local Debug is impractically slow for 500-iteration 32^3 budgets); runtimes recorded. | Base commit is this activation commit on `master=577e21c`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-11T22:05Z | T02 audit finding T02-F1: amendment to decision 5(e)(ii) | T02 implemented every prespecified gate verbatim and honestly reported one failure: the hostile-`condition_limit` safeguard control expected `anderson_accepted == 0`, but observed accepted=2 alongside condition_resets=2. Orchestrator triage: the defect is in the ORCHESTRATOR'S OWN control expectation, not in the T01 implementation — with a single history column (m=1) the R-diagonal condition estimate is MATHEMATICALLY exactly 1 (max/min over one entry), so any validated limit (>1) passes and a one-column (secant-step) acceleration may legitimately proceed; the candidate still traverses the FULL SF-15 guard chain, so the safety property that matters ("never accept a candidate that fails the Picard safeguard") is intact. A one-column history has no conditioning question to guard. | **Amended criterion (decision 5(e)(ii) revision, recorded BEFORE the corrective run):** the hostile-limit control gates on (i) convergence, (ii) `anderson_condition_resets >= 1` (every m>=2 attempt is condition-rejected), and (iii) a NEW unit assertion making the m=1 property an explicit tested contract: `form_candidate` with exactly one column must report `condition_estimate == 1.0` exactly. The `anderson_accepted == 0` arm is DROPPED (it rested on a false assumption about the estimator at m=1). Additionally the corrective must document the m=1 secant-acceleration property explicitly in `AndersonAccelerator.cuh` (small doc-only src change). All other gates unchanged; nothing tuned to make a failing physical result pass — this is a safeguard-semantics clarification, and the observed control run (converged, 11 iterations, r_F=7.2e-7, resets=2) already satisfies the amended gates. | Corrective C01 amends the control gate + adds the m=1 unit assertion + the header doc paragraph; the expensive `streamfunction_anderson_stall` entry runs on the remote V100 in parallel (orchestrator-executed audit). |
| 2026-08-11T19:15Z | `147b325`, integration validation | Three-node DAG (T01 library, T02 tests, corrective C01) completed and orchestrator-audited node by node. T01 `f45834d`: `AndersonAccelerator.cuh/.cu` + SF-15 loop integration exactly per decisions 1-4 — type-II mixing over the coupled state (algebra verified: `x_acc = u_hat - sum gamma_j (DeltaX_j+DeltaF_j)` via `x+f=u_hat`), full SF-15 guard-chain reuse with the omega=1 Armijo arm, rejection clears history and falls back to unchanged Picard backtracking, `clear()` resets staging (no stale deltas), deterministic fixed-size cublasDdot gram + host pivoted QR with R-diagonal condition estimate, optional workspace component with exact-byte accounting (history 4*m*n*8 exact), disabled default bitwise-preserving. T02 `e6456d2`: decision-5 suite verbatim; cheap tier green; **audit finding T02-F1 honestly reported (hostile-limit control's `accepted==0` arm failed)** — root cause was the ORCHESTRATOR'S OWN control expectation (the m=1 condition estimate is mathematically exactly 1; a one-column secant acceleration is legitimate and fully guarded); amendment versioned (`ad086c3`) BEFORE corrective C01 `3c27dcc` (gate re-specified, m=1 `condition_estimate==1.0` made an explicit tested contract, header doc added). T02/C01 each interrupted mid-validation and recovered per runbook s13 where needed (T02: worker-killed validation reruns executed by the orchestrator). Single integrator: linear chain on `ad086c3`, zero conflicts, patches byte-identical (verified 58954/70839/10794 B); integrator interrupted mid-validation with the integration itself complete — validation executed by the orchestrator. | **Acceptance evidence.** Validation split adopted (owner-endorsed): REMOTE V100 release full suite 10/10 (1274 s) including `streamfunction_anderson_stall`; LOCAL Debug control build + pipeline byte-invariance (three configs: stdout IDENTICAL vs the orchestrator's own base refs; artifacts byte-identical except manifest run identity) + checker OK. **Re-sequencing threshold PASSED with recorded numbers (V100):** fixture A (sigma2=1, lambda*=0.1125): control exhausted 500 iterations (r_F=5.07e-6) certifying the stall; Anderson depth 5 converged in **64 iterations** (r_F=9.03e-7, 39.6 s; 45 accepted / 7 rejected / 0 resets). Fixture B (sigma2=0.25, lambda*=0.3859375): control exhausted 500 (r_F=1.17e-5); Anderson converged in **88 iterations** (r_F=9.38e-7; 53/15/0). Rejection counters demonstrate the guard chain actively filtering accelerated candidates with convergence via Picard fallback. Non-regression: trig 23->11 iterations, field RMS agreement 3.4e-9 (bound 1e-4); homogeneous exact; memory exact for depths 3/5/8; disabled equivalence bitwise. | Orchestrator FINAL_AUDIT complete (PASS). Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-11T19:25Z | `fbb1d19` published, PR #33 open | Delivery branch pushed and [PR #33](https://github.com/santi-esquerre/MacroFlow3D/pull/33) opened as `awaiting_review` with the frozen audited source head `147b325` (later commits are increment-state documentation only). | PR description carries the DAG, the T02-F1 corrective cycle with the versioned 5(e)(ii) amendment, the full stall-fixture numbers (64/88 iterations vs exhausted 500-budget controls), the validation split evidence, and the reviewer flags. No agent merges; `NEXT` remains `SF-20`. | Await explicit human review/approval of PR #33; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-21`) on this same PR. |
| 2026-08-11T19:30Z | PR #33 head `d3afa98`, human approval | The repository owner explicitly approved PR #33 with the instruction "Apruebo la PR #33, hacé el cierre". No GitHub review object exists (`reviews=0`); the approval fact is this recorded instruction. Verified before closure: PR #33 `OPEN` at head `d3afa98` — exactly the published state; frozen audited source head `147b325` unchanged (later commits are increment-state documentation only), so the approval applies to the audited content. | The approval covers the items flagged at publication: the seven activation decisions plus the versioned 5(e)(ii) amendment (m=1 secant-acceleration property as an explicit tested contract; normal-equations conditioning caveat), the disabled-by-default choice (enabling deferred to SF-21 wiring), the heavy/cheap ctest registry split with the V100 stall pass on record, the adopted validation split (remote suite + local byte-compares), and the mandatory-review paths. | Closure metadata commit on this PR: set `done`, complete checklist, advance `NEXT` to `SF-21`. |
| 2026-08-11T19:30Z | closure metadata commit | SF-20 set `done`; checklist completed 7/7; dashboard updated (`SF-20` checked, `Last completed increment: SF-20`, `NEXT: SF-21`, active goal `none`); checker rerun. The new `NEXT: SF-21` exists only on this PR branch until a human merges it and does not authorize work ahead of the default branch. | Metadata/documentation-only diff (increment spec + dashboard); frozen audited source head remains `147b325`. | Human merges PR #33; SF-21 (heterogeneity re-activation with Anderson wired into the stage solver, reusing the parked audited machinery against its UNCHANGED gates) may activate only after this closure state is visible on `master`. |
