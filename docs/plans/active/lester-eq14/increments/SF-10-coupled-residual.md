# SF-10 — Coupled residual

- State: `done`
- Goal: `Implementar el residuo acoplado y las reducciones adimensionales del solver.`
- Depends on: `SF-09`
- Unlocks: `SF-11`
- Branch: `science/lester-sf10-coupled-residual`
- Worktree: `Claude-managed per-node isolated worktrees (native isolation: worktree)`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-07T17:39Z`
- Completed: `2026-08-07 (PR #21 merged to master)`
- PR: [#21 — SF-10: coupled Lester residual evaluator with dimensionless reductions and |c| histogram](https://github.com/santi-esquerre/MacroFlow3D/pull/21)
- Commit: `dd83caa28bb4b3e0655ed7d407ca9219ea803fdb` (canonical default-branch squash; tree identical to audited branch tip `54a2720`, source head `8b0b825`)

## Scientific or engineering intent

Create the authoritative nonlinear convergence measure using exactly the same
discrete equations that Picard and future Newton iterations solve.

## Preconditions

- SF-09 provides accepted source terms; SF-06 provides affine RHS terms.

## In scope

- `F1=A*u1-div(q*gbar1)+eta*q*S2`, its paired `F2`, reusable reductions,
  dimensionless normalization, and histogram percentiles.

## Out of scope

- Velocity reconstruction, Picard updates, and convergence control.

## Files and symbols

- Add `src/physics/streamfunctions/ResidualEvaluator.cuh/.cu` and diagnostic
  reduction workspace.
- Reuse the exact `A`, affine RHS, and nonlinear source modules.

## Implementation specification

1. Compute `F` from operator and RHS arrays rather than re-discretizing the PDE.
2. Normalize component one by `q_rms*v_rms/L_ref` and component two by
   `q_rms/L_ref`; combine as `sqrt((r1^2+r2^2)/2)`.
3. Compute RMS/Linf and a fixed 512-bin logarithmic histogram for `|c|` without
   sorting or allocating inside an iteration.
4. Expose raw as well as normalized values in a POD report.

## Expected numerical effect

Convergence decisions become grid-size independent and consistent with the
linear equations.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Direct `A*u-b` and residual evaluator agree to reduction roundoff.
- CPU/GPU RMS and Linf agree within `1e-12` relative on deterministic fixtures.
- Histogram percentile bin error is bounded and documented.

## Regression surface

- Operator/source sign consistency, units, reduction synchronization, and
  percentile behavior for zeros.

## Failure and rollback policy

- Never accept an independently re-discretized residual as a substitute.
- Retain exact host reductions in tests if the production histogram needs later
  tuning.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Both coupled residual components use shared discrete primitives.
- [x] Dimensionless normalization is implemented and tested.
- [x] RMS, Linf, and histogram reductions use persistent workspace.
- [x] CPU/GPU accuracy thresholds pass.
- [x] Full regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-10 complete and selects SF-11.
<!-- completion-checklist:end -->

## Advancement rule

SF-11 may add physical diagnostics without changing this convergence residual.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-07T17:39Z | active; base `master=origin/master=e81cd47378849370272f8ed727659b025de16f44` | Activated SF-10 documentation state under the Claude Code orchestration harness. | Preflight verified on the default branch: SF-09 `done` via PR #19 (`75eafef`) and closure PR #20 (`e81cd47`); checker PASS (`29 increments, next=SF-10`); clean tree. Reuse surface inspected in code: `LesterPositiveDiffusionOperator` (SF-02), `MeanZeroProjector`/`MeanZeroWorkspace` (SF-03), `assemble_affine_periodic_rhs` + workspace/diagnostics (SF-06), `enqueue_total_streamfunction_gradients`/`enqueue_streamfunction_hessian_vector_b` (SF-07/08), `enqueue_streamfunction_nonlinear_sources` (SF-09), `blas` reductions with `ReductionWorkspace`. Interpretive decision recorded for human review: the authoritative convergence residual uses the projected combined right-hand side, `F_i = A u_i - P(div_h(q gbar_i) - eta q S_pair)`, because the locked decisions require projecting right-hand sides and `A u` is discretely mean-zero on the periodic domain, so the literal unprojected spec formula would stagnate at the raw compatibility defect; raw RHS means remain reported as diagnostics and tests verify the projected/literal relationship explicitly. sccache remains disabled locally (documented in SF-09 activation). Persistent Goal `Implementar el residuo acoplado y las reducciones adimensionales del solver.`; delivery branch `science/lester-sf10-coupled-residual`. | Build the SF-10 intra-increment DAG and delegate implementation to isolated workers. |
| 2026-08-07T18:40Z | integration validation; `8b0b8254435a388ae2e467fe00da2d78ab6700c0` | Four-node DAG (T01 CPU oracles `7edd5f6`; T02 production evaluator `3afaf50`; corrective C01 `2fe7eb0` from the T02 audit — percentile overflow convention aligned to +infinity and fail-fast isotropy/extent validation; T03 GPU acceptance cases `8b0b825`) executed by isolated Sonnet workers, each independently audited, then verified by a single isolated integrator: linear approved chain from base `c11bb9a`, per-file content equality (only the C01.1 hunks differ from raw T01), exact 8-file/+2287 diffstat, clean `git diff --check`, no integration-only changes. | Integrator validation (fresh worktree, sccache launcher disabled): configure/build 107 targets, checker, all 9 `coupled_residual_*` cases, targeted CTest 1/1, full CTest 2/2, `run_operator_tests` 8/8, PSPTA-small smoke — all exit 0. Key metrics: GPU-vs-CPU-oracle F1/F2 normalized RMS `5.3e-16`, boundary-free Linf `2.53e-14` (threshold `5e-11`); direct module-composition agreement `0` (eta=0) and `3.37e-15` (eta=1); CPU/GPU RMS/Linf/q_rms reductions worst `1.34e-14` (spec threshold `1e-12`), `r1/r2/r_F` vs reference `<=1e-14`, `L_ref=1` exact; histogram bin-for-bin exact on wide and split ranges (615/615 under/overflow) with edge separations `1.28e-4/1.25e-4` (guard `1e-9`); percentile-vs-exact-sorted worst deviation `1.065` within documented `bin_factor^2=1.134`; homogeneous control (q=1, zero fluctuations) exactly zero at eta=0 and eta=1; `mean(F_i)` normalized `<=1e-17` and projected RHS means `<=2.8e-17`; error contract 36 checks (`invalid_argument` incl. C01 anisotropic fail-fast; `std::logic_error` for unprepared/never-enqueued workspace; accepted read-only u1==u2 overlap finite); mutants pairing_swap `0.909`, rhs_sign_flip `1.449`, projection_omitted `0.147` over documented thresholds `0.09/0.14/0.014`. Hardware: local Debug `sm_86` RTX 3050. | Orchestrator final audit. |
| 2026-08-07T18:40Z | root final audit PASS; head frozen at `8b0b825` | Orchestrator personally audited the full diff `c11bb9a..8b0b825` (8 files, +2287) against the SF-10 spec: `F1=A u1 - P(rhs_affine1 - eta*q.*S2)`, `F2=A u2 - P(rhs_affine2 - eta*q.*S1)` composed exclusively from the accepted SF-02/03/06/07/08/09 modules (no re-discretization; equality with manual module composition proven by test); locked normalization `r1=RMS(F1)*L_ref/(q_rms*v_rms)`, `r2=RMS(F2)*L_ref/q_rms`, `r_F=sqrt((r1^2+r2^2)/2)`; prepare-once workspace with no allocation or host sync in the enqueue path (CUDA runtime calls enumerated and checked); fixed 512-bin log10 `\|c\|` histogram without sorting, binning arithmetic identical between GPU kernel and CPU reference; percentile helper with documented one-bin-width error bound and +infinity overflow convention. Projected-residual interpretive decision (recorded at activation) empirically confirmed and covered by the projection-omitted mutant; discrete mean-zero of `A u` verified. Inherited isotropic-grid restriction documented with C01 fail-fast. Orchestrator independently reran the entire suite on the control checkout at `8b0b825`: build exit 0, 76/76 case verdicts PASS, CTest 2/2, `run_operator_tests` 8/8, smoke OK, `git diff --check` clean, checker PASS. | Gate 1 PASS; Gate 2 PASS; Gate 3A operator subset PASS for the SF-10-applicable metrics (authoritative normalized `r_F` defined and tested; `\|c\|` histogram/percentile machinery; gauge means and raw compatibility defects reported). Physical `e_v`, Darcy-invariance `e_i`, `e_div`, and full physical Gate 3A remain N/A for this pre-diagnostics increment (SF-11+) and are not inferred; Gate 4/Gate 5/V100 N/A, no claim made. Implementation frozen; mandatory human review pending (`src/physics/streamfunctions/`). | Publish PR as awaiting_review; do not advance NEXT. |
| 2026-08-07T18:47Z | awaiting_review; PR [#21](https://github.com/santi-esquerre/MacroFlow3D/pull/21); frozen audited source head `8b0b8254435a388ae2e467fe00da2d78ab6700c0` | Published the frozen SF-10 implementation for mandatory human review on branch `science/lester-sf10-coupled-residual`. The PR records scope, DAG/worker/corrective/integrator provenance, exact commands, all per-criterion metrics, gate determinations, the projected-residual interpretive decision flagged for the reviewer, and intentionally untouched areas. | Metadata commits after `8b0b825` are documentation-only (`47566de` evidence/state, this row); no source, test, or CMake change after the audited head. Residual risks: local Debug `sm_86` evidence only; isotropic-grid restriction inherited from SF-02/SF-06 (documented fail-fast); six discarded HVP scratch fields per the SF-08 API contract (SF-23 optimization candidate). | Await explicit human review of PR #21; on approval add only the closure metadata commit (done/checklist/dashboard NEXT→SF-11) on the same PR; do not merge. |
| 2026-08-07T18:44Z | human approval / merge; PR [#21](https://github.com/santi-esquerre/MacroFlow3D/pull/21) -> `master` commit `dd83caa28bb4b3e0655ed7d407ca9219ea803fdb` | Repository owner manually merged the audited SF-10 PR (mergedAt 2026-08-07T18:44:56Z) and subsequently gave the explicit closure instruction ("Listo"). GitHub records the PR as merged with no separate review object, so closure records the manual owner merge plus the explicit instruction as the human approval event rather than inventing a review. | Verified `git rev-parse dd83caa^{tree} == 54a2720^{tree}` (`148edbbfa55b8f0dfd120b9388ce17b29cc91cb0`): the merged content is byte-identical to the audited published branch state, and `git diff dd83caa 8b0b825 -- src tests CMakeLists.txt` is empty, so the merged source equals frozen audited head `8b0b825` exactly. Gate 1, Gate 2, and Gate 3A operator subset determinations remain PASS; no post-audit scientific/source change entered `master`. | Repair the stale versioned harness state: mark SF-10 done and advance dashboard NEXT to SF-11. |
| 2026-08-07T18:51Z | formal closure repair; branch `chore/close-lester-sf10` | Metadata-only closure repair prepared because PR #21 entered `master` while the increment document still said `awaiting_review` and the dashboard still selected SF-10 (same exceptional pattern as SF-08/PR #18 and SF-09/PR #19, runbook §12). | Set `State: done`; recorded canonical default-branch commit `dd83caa28bb4b3e0655ed7d407ca9219ea803fdb`; completed the remaining checklist items; dashboard repair checks SF-10, sets `Last completed increment: SF-10`, keeps the active runtime goal cleared, and selects `NEXT: SF-11`. Scientific implementation and evidence are unchanged. | Publish the metadata-only closure-repair PR; human merges it; start SF-11 only after the repaired state is visible on the default branch. |
