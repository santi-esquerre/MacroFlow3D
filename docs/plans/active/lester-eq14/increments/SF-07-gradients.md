# SF-07 — Streamfunction gradients

- State: `done`
- Goal: `Implementar gradientes periódicos cell-centered con contribuciones afines.`
- Depends on: `SF-06`
- Unlocks: `SF-08`
- Branch: `science/lester-sf07-gradients`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf07-gradients`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A operator subset`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-05T18:35Z`
- Completed: `2026-08-06T01:57:28Z`
- PR: [#16](https://github.com/santi-esquerre/MacroFlow3D/pull/16)
- Commit: `e7f90db39bca1906b8bbc5dd94195edf7084cfac`

## Scientific or engineering intent

Provide one validated definition of total streamfunction gradient for nonlinear
sources, invariance metrics, and velocity reconstruction.

## Preconditions

- SF-06 defines periodic fluctuations and affine gradients.

## In scope

- Second-order centered, triply periodic cell-centered gradients for both
  streamfunctions, including affine constants.

## Out of scope

- Hessians, fused source kernels, face reconstruction, and higher order.

## Files and symbols

- Add `src/physics/streamfunctions/DifferentialOperators.cuh/.cu`.
- Add analytic and CPU/GPU gradient tests.

## Implementation specification

1. Use `dx`, `dy`, `dz` explicitly even though current production grids are
   isotropically spaced.
2. Wrap cell indices independently in all three directions.
3. Add the affine vector after differentiating the periodic fluctuation.
4. Provide an output-buffer API for tests; production fusion is deferred.

## Expected numerical effect

Affine fields are exact, periodic modes converge at second order, and all later
operators share the same total gradient convention.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Pure affine gradients agree to roundoff.
- Periodic trigonometric fields show L2 order at least 1.8.
- Linf errors decrease monotonically from `16^3` through `64^3`.

## Regression surface

- Grid spacing assumptions, periodic indexing, and future shared-memory stencil
  layout.

## Failure and rollback policy

- Retain the explicit-buffer reference kernel until convergence is demonstrated.
- Do not fuse or introduce fourth-order differences in this increment.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Total-gradient API and kernel are implemented.
- [x] Affine exactness and periodic convergence tests pass.
- [x] Spacing and indexing conventions are documented.
- [x] Full regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-07 complete and selects SF-08.
<!-- completion-checklist:end -->

## Advancement rule

SF-08 may use the accepted total gradients in Hessian-vector products and `B`.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-05T18:35Z | active; master=`origin/master=031e1af` | Activated SF-07 documentation state. | Checker PASS (`next=SF-07`); dependency SF-06 done; persistent Goal `Implementar gradientes periódicos cell-centered con contribuciones afines.`; branch `science/lester-sf07-gradients`; worktree `~/src/MacroFlow3D/.agents/worktrees/lester-sf07-gradients`. | Build DAG. |
| 2026-08-06T00:45Z | integration validation; `d1abed9` | Independent T06 integration: fast-forwarded only the verified linear chain `94351f6 -> 8298832 -> 5e442df -> a227392 -> d1abed9`; T02 original `60a8e95` equals `8298832`, and T03 original `bd5f559` equals `5e442df` by stable patch-id and per-file blobs. Scope audit: eight SF-07 files only (gradient API/kernel, independent CPU oracle, gradient tests, CMake registrations); centered x-fastest independent xyz wraps; affine additions post-difference; explicit anisotropic spacings; six exact/non-null/anti-alias outputs; one async checked launch without production allocation/copy/sync; no Hessians/fusion/faces/higher order/PSPTA/SF-08. Invalid history retained, not evidence: T02 PGID 855840/856672; T03 PGID 857392/858270 plus later sessionless attempt; each replaced by source-only audit; T05 first detached build lacked new-file compilation/binario, only its authorized serial retry was valid. T04 had positive binary/tests but this run is the independent evidence. Initial T06 build PGID 47554 was invalid: it detached/no recoverable exit status; its pre-recovery case outputs are diagnostic only. Authorized recovery `cmake --build build/wsl-debug --clean-first -j1`, session `90485`, completed with explicit `exit_code=0`; one serial clean build only. Environment: `asus-santi`, Linux `7.1.5-1-cachyos`, i7-12650H (16 logical CPUs), RTX 3050 Laptop 4096 MiB, driver 610.43.03, CUDA 13.3.73, GCC 16.1.1 (nvcc host GCC 15.3.0), CMake 4.4.2, preset `wsl-debug` (CUDA arch 86). | Initial/final checker PASS (`29 increments, next=SF-07`); configure PASS. Recovery build PASS (warnings pre-existing/out of SF-07: unused `NZ`; device `long double` treated as double). SF-07 cases PASS: affine six RMS/Linf=0; independent oracle global normalized RMS `6.04e-17..9.29e-17`, boundary Linf `1.83e-16..4.30e-16`; unequal-spacing 16/32/64 L2/Linf table: p1x `.0252006/.0397526,.00633669/.0101426,.00158646/.00254852` orders `1.99166,1.99791`; p1y `.0866514/.119625,.0221691/.0323837,.00557436/.00825593` `1.96667,1.99167`; p1z `.201330/.287379,.0530250/.0767023,.0134298/.0194912` `1.92482,1.98123`; p2x `.0990123/.132321,.0253327/.0358802,.00636989/.00915097` `1.96661,1.99166`; p2y `.0552487/.0977637,.0141133/.0249967,.00354741/.00628437` `1.96889,1.99222`; p2z `.0667875/.0912068,.0170874/.0246920,.00429661/.00629513` `1.96664,1.99167`; every Linf strictly decreases. Error contract: 24 exact `std::invalid_argument` rejections; allowed input-input overlap and anisotropic positive spacing at `9.29e-17`; mutants omit-affine `.277108 > .1`, dx-for-yz `.289053 > .1`, clamp-boundary `.160674 > .01`. Full streamfunction harness PASS; `run_operator_tests` 8/8 PASS; focused CTest 1/1 (6.93s), full CTest 2/2 (7.06s). Smoke `apps/config_pspta_small.yaml`: seed 42, grid 64x32x32, head 10 iterations `1.02e+01 -> 1.77e-13`, divergence min/max `-8.1175e-14/8.2979e-14`, particles active/exited `387/113`, stalls/nonzero/max fails `0/0/0`; legacy PSPTA output observed only. `git diff --check master...HEAD` PASS. Gate 1 PASS; Gate 2 PASS; Gate 3A operator subset PASS insofar as SF-07 has no coupled solver/source/reconstruction: `r_F`, `e_v`, Darcy invariance, reconstructed divergence, denominator/gauge-restoration metrics are N/A. Gate 3A physical completion, Gate 4, Gate 5, and V100 are N/A: no Lester solver/sources/reconstruction or production physics path exists; no remote run authorized. No acceptance/done claim. | root audit |
| 2026-08-06T00:49Z | root audit PASS; validating | Personal inspection of `git diff 96786f1..HEAD`, commits, and eight-file SF-07 scope: centered fórmulas y signos use post-derivative affine addition; x-fastest indexing with independent wrap xyz and explicit `dx/dy/dz`; six outputs have aliasing/overflow checks; productive path has no allocation/copy/sync; independent CPU reference uses analytic derivatives. Positive cases, exact `std::invalid_argument` rejections, and mutants were inspected. Root rerun commands were incremental no-work build, 5 cases, suite, `run_operator_tests` 8/8, CTest 2/2, and smoke. | PASS metrics: affine 0; oracle max `9.29e-17`, boundary `4.30e-16`; minimum order `1.924818` with decreasing Linf; mutants `0.277/0.289/0.161`; smoke head `1.02e1 -> 1.77e-13`. Gate 1/2/3A operator subset PASS; complete physics/Gate 4/V100 N/A with documented scope rationale. `bash scripts/hooks/check-lester-increments.sh` and `git diff --check` PASS. Implementation is frozen; next step is publish PR/revisión humana, with no review, PR, completion, dashboard/NEXT, or runtime Goal change asserted here. | Publish PR and await human review. |
| 2026-08-06T00:52Z | `97d7fe7` metadata correction | The prior literal statement that personal inspection used `git diff 96786f1..HEAD` was inexact. The authoritative scientific audit inspected `git diff master...96786f1` for the integrated implementation and evidence, together with the full commits; the validating transition was inspected separately with `git diff 96786f1..97d7fe7`. This correction changes neither the PASS result, metrics, nor acceptance-gate determinations. | Documentation-only, append-only correction; no numerical effect and no state, checklist, dashboard/NEXT, Goal, completion, PR, or commit metadata change. | Independent validation of this correction. |
| 2026-08-06T00:55Z | C02 independent validation of `ca27bd2` | Verified `git show ca27bd2`, `git diff --check 97d7fe7..ca27bd2`, prior-bitácora hash equality, and complete `master...HEAD` SF-07-only history/scope; confirmed the scientific audit range is `master...96786f1` and the validating transition is `96786f1..97d7fe7`. | Checker PASS (`29 increments, next=SF-07`); C01/full-range diffs PASS; C01 adds one append-only row and has zero functional effect, so the clean T06R/root binary evidence at `96786f1`/`97d7fe7` remains applicable without rebuild or tests. No state, checklist, dashboard/NEXT, Goal, completion, PR, or commit metadata change. | root final re-audit and publish PR |
| 2026-08-06T01:00Z | awaiting_review; [PR #16](https://github.com/santi-esquerre/MacroFlow3D/pull/16); initial published head `c12f2236e2ccdd54303490a939ac8ebb8065dd2e` | Draft PR opened from `science/lester-sf07-gradients` to `master` with the reproducible scope/body: centered second-order triply periodic cell-centered total gradients for `psi1`/`psi2`, explicit `dx/dy/dz`, post-derivative affine terms, and six caller-owned buffers; production kernel/API, independent long-double oracle/fixtures, positive/error/mutant tests, and evidence/state documentation. | Prior clean T06R/root evidence remains the validation record: checker; preset; serial clean build; five named gradient cases; streamfunction suite; `run_operator_tests`; focused/full CTest; pipeline smoke; diff check; root incremental no-work re-run plus same acceptance suite. Metrics: all affine six RMS/Linf 0; GPU/CPU normalized RMS max `9.29285e-17`, boundary `4.29506e-16`; minimum L2 order `1.924818` with all Linf decreasing; 24 exact invalid_argument and two allowed `9.29285e-17`; mutants `.277108/.289053/.160674`; CTest 2/2, legacy 8/8; smoke head `1.02e1 -> 1.77e-13`, divergence `-8.1175e-14/+8.2979e-14`, particles `387/113`, zero stalls/fails. Gate 1/2/3A operator subset PASS; physical Gate 3A/Gate 4/Gate 5/V100 N/A because no coupled solver, sources, reconstruction, or performance claim. Risks/intentionally untouched: explicit-buffer kernel intentionally unfused; no production consumer; local RTX 3050 only/V100 N/A; superseded invalid attempts retained; SF-06 RHS/operator/PCG/MG, Hessians/nonlinear sources/residual/Picard, face velocity/reconstruction, runtime configs/transport/legacy PSPTA, and SF-08+ docs/state untouched. Implementation frozen; mandatory human review pending; no acceptance or merge assertion. | human review |
| 2026-08-06T01:57:28Z | done; [PR #16](https://github.com/santi-esquerre/MacroFlow3D/pull/16) merged by `santi-esquerre`; commit `e7f90db39bca1906b8bbc5dd94195edf7084cfac`, tree `bb08fc4da9289fbe840040114353dec5ae63ac40` | Explicit user approval (“Listo, mergeado, cerrá el SF-07…”) and confirmed PR merge close SF-07; the merged tree is identical to audited head tree `f0d680f25003e4e7b724d9855f780065e9a66d6b`. GitGuardian SUCCESS; GitHub formal review list empty, so human approval is recorded only from the explicit user confirmation and merge. | Validation evidence: affine 0; GPU/CPU max normalized RMS `9.29285e-17`, boundary `4.29506e-16`; minimum order `1.924818` with Linf decreasing; 24 exact `invalid_argument` rejections plus 2 allowed inputs; mutants `.277108/.289053/.160674`; T06R clean build exit 0; CTest 2/2, legacy 8/8; smoke head `1.02e1 -> 1.77e-13`, divergence `-8.1175e-14/+8.2979e-14`, particles `387/113`, zero fails; Gate 1/2/3A operator subset PASS; full physical Gate 3A/Gate 4/5/V100 N/A. Residual risks: explicit-buffer kernel remains unfused with no production consumer; validation was local RTX only; invalid attempts are retained as superseded history. | Independent closeout validation and closeout PR; do not claim the closeout PR merged. |
