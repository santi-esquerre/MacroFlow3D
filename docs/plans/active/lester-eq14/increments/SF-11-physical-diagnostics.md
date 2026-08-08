# SF-11 — Physical diagnostics

- State: `done`
- Goal: `Reconstruir v_psi sobre CompactMAC y calcular los diagnósticos físicos obligatorios.`
- Depends on: `SF-10`
- Unlocks: `SF-12`
- Branch: `science/lester-sf11-physical-diagnostics`
- Worktree: `Claude-managed per-node isolated worktrees (native isolation: worktree)`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-07T19:02Z`
- Completed: `2026-08-08 (explicit owner approval of PR #23; closure metadata commit on the same PR)`
- PR: [#23 — SF-11: CompactMAC velocity reconstruction and mandatory physical diagnostics](https://github.com/santi-esquerre/MacroFlow3D/pull/23)
- Commit: `d32268a1cad89ec778bb0606a98bd6c6d6adfa07` (frozen audited source head; later commits on the PR are documentation-only)

## Scientific or engineering intent

Prevent acceptance based solely on algebraic residual by measuring velocity
reconstruction, invariance, divergence, angle, magnitude, and nondegeneracy.

## Preconditions

- SF-10 provides total cell gradients, `c`, and reduction infrastructure.

## In scope

- CompactMAC face reconstruction and all Gate 3A physical metrics.

## Out of scope

- Exact discrete curl formulations, output writers, and trajectory invariance.

## Files and symbols

- Add `src/physics/streamfunctions/Diagnostics.cuh/.cu`.
- Reuse `VelocityField` CompactMAC layout and natural MAC divergence.

## Implementation specification

1. At each normal face, use the normal centered derivative and consistently
   interpolated tangential derivatives before forming the cross product.
2. Compare directly with Darcy CompactMAC components.
3. Report L2/Linf by component, magnitude error, correlation, and robust angular
   error; exclude only explicitly counted near-zero pairs from angles.
4. Compute Darcy invariance at one documented common location and split
   cross-gradient degeneracy by Darcy-speed threshold.

## Expected numerical effect

The homogeneous flow reconstructs exactly; manufactured fields exhibit
second-order physical-metric convergence.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Uniform velocity reconstruction relative error `<=1e-13`.
- Manufactured velocity and divergence errors have L2 order at least 1.8.
- Invariance metrics agree with independent CPU calculations within `1e-12`
  relative plus spatial truncation error.

## Regression surface

- CompactMAC indexing, interpolation placement, periodic boundary faces, and
  treatment of low-speed cells.

## Failure and rollback policy

- Do not claim algebraic divergence freedom for this initial reconstruction.
- If face interpolation fails second-order convergence, stop before adding
  output or solver acceptance logic.

## Completion checklist

<!-- completion-checklist:start -->
- [x] CompactMAC reconstruction is implemented and documented.
- [x] Required velocity, invariance, divergence, and degeneracy metrics exist.
- [x] Uniform and manufactured thresholds pass.
- [x] Low-speed exclusions are explicit and counted.
- [x] Full regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-11 complete and selects SF-12.
<!-- completion-checklist:end -->

## Advancement rule

SF-12 may define the stable public ownership/API around the accepted numerical
primitives and reports.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-07T19:02Z | active; base `master=origin/master=a3a5718a18c2546f06a4a75545b31eb4e64cc612` | Activated SF-11 documentation state under the Claude Code orchestration harness. | Preflight verified on the default branch: SF-10 `done` via PR #21 (`dd83caa`, tree identical to audited `54a2720`) and closure PR #22 (`a3a5718`); checker PASS (`29 increments, next=SF-11`); clean tree. Reuse surface inspected in code: `VelocityField` CompactMAC layout (U-face `i` between cells `i-1`,`i`; periodic duplicate boundary planes), `enqueue_total_streamfunction_gradients` (SF-07, independent spacings), SF-09 `c=g1×g2` convention and `kMaxDegeneracyThresholds`, SF-10 workspace/enqueue/synchronize pattern with `blas` reductions, and the CPU oracle library in `tests/streamfunctions/reference_operators.*`. Interpretive decisions recorded for human review: (1) reconstruction is interpolate-then-cross — face gradients use the natural compact normal derivative plus arithmetic two-cell averages of the SF-07 cell-centered tangential derivatives, then the cross product's face-normal component is stored (the normal compact derivative cancels algebraically from that component; documented); (2) the documented common location for invariance/magnitude/angle/degeneracy metrics is cell centers with per-component two-face averaging of the Darcy MAC field; (3) all normalizations use the measured cell-centered `v_D,rms` with no hidden floors (degenerate normalizations surface as NaN/Inf); (4) face reductions run over unique faces only; (5) the module supports independent spacings (its dependencies do), diverging deliberately from the SF-10 chain's isotropic fail-fast — documented; (6) `\|c\|` percentiles stay in SF-10's evaluator, SF-11 adds exact min/max/mean and the Darcy-speed-split degeneracy counts. No algebraic divergence-freedom claim will be made (spec rollback rule). sccache remains disabled locally (documented in SF-09 activation). Persistent Goal `Reconstruir v_psi sobre CompactMAC y calcular los diagnósticos físicos obligatorios.`; delivery branch `science/lester-sf11-physical-diagnostics`. | Build the SF-11 intra-increment DAG and delegate implementation to isolated workers. |
| 2026-08-07T20:21Z | integration validation; `d32268a1cad89ec778bb0606a98bd6c6d6adfa07` | Five-node DAG (T02 production `Diagnostics.cuh/.cu` `137e973`; T01 CPU mirrors `907dc27` (= `a0b97bf`); corrective C01 `4bdb5e6` from the T01/T02 audits — literal-division gradient mirror and empty-angle-set NaN convention pinned on both sides; T03 GPU acceptance cases `63f2099`; corrective C02 `d32268a` — comment-only accurate statement of the raw-moment Pearson degenerate-input instability found by T03) executed by isolated Sonnet workers, each independently audited, then verified by a single isolated integrator: exact 5-commit linear chain from base `36d72f8`, merge-base equality, exact 8-file/+3274 diffstat, clean `git diff --check`, no integration-only changes. | Integrator validation (fresh worktree, sccache launcher disabled): configure/build 109 targets, checker OK, all 7 `physical_diagnostics_*` cases PASS, targeted CTest 1/1, full CTest 2/2, `run_operator_tests` 8/8, PSPTA-small smoke — all exit 0. Key metrics: uniform reconstruction worst deviation `6.83e-16` (threshold `1e-13`) on isotropic 16³ AND anisotropic 8x10x12 grids; GPU-vs-CPU-oracle worst continuous field `6.55e-14` (threshold `1e-12`), worst face `1.20e-13`, duplicate periodic planes exactly equal; convergence orders `e_v=1.900`, `rms_div=1.846` (threshold `1.8`); exact GPU/CPU count agreement under runtime-asserted separation margins ≥`1.11e-4` (guard `1e-9`): angle included/excluded `3034/1062`, degeneracy total/low-speed/unexplained `820/374/446`; empty-angle-set NaN convention matched both sides; error contract 43/43; mutants (swapped cross order `0.970`, one-sided interpolation `3.73e-3`, wrong divergence spacing `8.21`) all above documented thresholds with ≥10x margins. Hardware: local Debug `sm_86` RTX 3050. | Orchestrator final audit. |
| 2026-08-07T20:21Z | root final audit PASS; head frozen at `d32268a` | Orchestrator personally audited the full diff `36d72f8..d32268a` (8 files, +3274) against the SF-11 spec: interpolate-then-cross CompactMAC reconstruction reusing SF-07 total gradients exactly (no re-discretization), documented normal-derivative cancellation, all-planes writes with exact duplicate-plane equality; per-component face errors, Pearson correlation, magnitude error, robust angle with counted exclusions, Darcy invariance at the documented cell-center common location, natural MAC divergence with `e_div=L_ref*RMS/v_rms`, and `\|c\|` min/max/mean plus Darcy-speed-split degeneracy counts; prepare-once workspace with no allocation or host sync in the enqueue path (CUDA calls enumerated) and one synchronize in the report step; measured `v_D,rms` normalization with no hidden floors (degenerate normalizations surface as NaN/Inf); deliberate documented anisotropic-spacing support (dependencies do not require isotropy); explicit "approximately, not algebraically, divergence-free" statement per the spec rollback rule. Orchestrator independently reran the entire suite on the control checkout at `d32268a`: build exit 0, 83/83 case verdicts PASS, CTest 2/2, `run_operator_tests` 8/8, smoke OK, `git diff --check` clean, checker PASS. | Gate 1 PASS; Gate 2 PASS; Gate 3A physical subset PASS for SF-11 scope (`e_v`, `e_i`, `e_div` defined/normalized/tested; `\|c\|` extremes and split degeneracy counts; explicit counted exclusions; `\|c\|` percentiles remain in the SF-10 evaluator by design). Scientific findings for the human reviewer: (1) interpretive decisions from activation (interpolate-then-cross; cell-center common location; measured `v_D,rms` normalization; unique-face reductions; NaN surfacing); (2) T03 empirically showed the raw-moment Pearson correlation is numerically unstable for exactly-degenerate inputs (NaN or spurious ±1 by sign-collapse across two reduction kernels) — documented as an accepted limitation (C02, comments only), meaningful only for non-degenerate inputs. Gate 4/Gate 5/V100 N/A, no claim. Implementation frozen; mandatory human review pending (`src/physics/streamfunctions/`). | Publish PR as awaiting_review; do not advance NEXT. |
| 2026-08-07T20:26Z | awaiting_review; PR [#23](https://github.com/santi-esquerre/MacroFlow3D/pull/23); frozen audited source head `d32268a1cad89ec778bb0606a98bd6c6d6adfa07` | Published the frozen SF-11 implementation for mandatory human review on branch `science/lester-sf11-physical-diagnostics`. The PR records scope, DAG/worker/corrective/integrator provenance, exact commands, all per-criterion metrics, gate determinations, the interpretive design decisions, and the Pearson degenerate-input finding flagged for the reviewer. | Metadata commits after `d32268a` are documentation-only (`ce0782a` evidence/state, this row); no source, test, or CMake change after the audited head. Residual risks: local Debug `sm_86` evidence only; divergence order `1.846` nearer the `1.8` floor than velocity (`1.900`), recheck at 64³ when solver output exists; raw-moment Pearson instability for exactly-degenerate inputs documented as accepted limitation. | Await explicit human review of PR #23; on approval add only the closure metadata commit (done/checklist/dashboard NEXT→SF-12) on the same PR; do not merge. |
| 2026-08-08T19:18Z | human approval received; PR [#23](https://github.com/santi-esquerre/MacroFlow3D/pull/23) OPEN at head `a9e1f76` | Repository owner explicitly approved PR #23 by direct instruction ("Apruebo la PR #23, hacé el cierre"). No GitHub review object exists; the approval fact is this recorded instruction. Verified before closure: PR head `a9e1f76` matches the published state exactly and the frozen audited source head `d32268a` is unchanged (all commits after it are documentation-only), so the approval applies to the audited content. | Approval is valid for source head `d32268a1cad89ec778bb0606a98bd6c6d6adfa07`; no source, test, or scientific-configuration change occurred after the final audit. | Add the closure metadata commit on this same PR: set `done`, complete the checklist, advance dashboard `NEXT` to SF-12; human merges. |
| 2026-08-08T19:18Z | closure metadata commit (this commit); State `done` | Completed the SF-11 checklist (all seven items evidenced by the final audit and the recorded human approval), set `Completed`, checked the SF-11 dashboard entry, set `Last completed increment: SF-11`, advanced `NEXT` to `SF-12`, and cleared the active runtime goal. Checker run before committing. | Metadata-only change on the delivery branch; the audited source head `d32268a` is untouched. The new `NEXT: SF-12` exists only on this PR branch until the human merges PR #23, so the default branch still prevents premature advancement. | Human merges PR #23; SF-12 may activate only after this closure state is visible on `master`. |
