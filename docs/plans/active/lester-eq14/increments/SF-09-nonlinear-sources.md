# SF-09 — Nonlinear sources

- State: `awaiting_review`
- Goal: `Implementar c, S1, S2 y la regularización explícita del denominador.`
- Depends on: `SF-08`
- Unlocks: `SF-10`
- Branch: `science/lester-sf09-nonlinear-sources`
- Worktree: `Claude-managed per-node isolated worktrees (native isolation: worktree)`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-07T14:58Z`
- Completed: `not completed`
- PR: [#19 — SF-09: nonlinear Lester sources S1, S2 with explicit denominator regularization](https://github.com/santi-esquerre/MacroFlow3D/pull/19)
- Commit: `ad7a04acd6bf4013c42f245cf46e31e0ab4ebf3e` (frozen audited source head; final merge commit recorded at closure)

## Scientific or engineering intent

Construct the two nonlinear Lester sources with transparent, dimensionless
regularization and enough diagnostics to distinguish degeneracy from low Darcy
speed.

## Preconditions

- SF-08 validates gradients, Hessian-vector products, and `B`.

## In scope

- Cross-gradient `c`, regularized denominator, `S1`, `S2`, source RHS terms,
  finite-value flags, basic degeneracy counts, and CPU/GPU tests.

## Out of scope

- Picard, adaptive epsilon continuation, full physical metrics, and kernel
  optimization.

## Files and symbols

- Add `src/physics/streamfunctions/NonlinearSources.cuh/.cu` and source workspace
  types.
- Extend test references with cross product and Lester source formulas.

## Implementation specification

1. Compute `c=g1 cross g2` and
   `d=|c|^2+(epsilon*v_rms)^2` in double.
2. Compute each `S_i=((B cross g_i) dot c)/d` and write only the two source
   arrays plus requested diagnostic accumulators.
3. Default `epsilon=1e-2`; never insert an unreported hard-coded floor.
4. Count nonfinite cells and `|c|/v_rms` below configured thresholds.

## Expected numerical effect

Smooth nondegenerate controls yield finite second-order sources; regularization
dependence is explicit rather than hidden.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Homogeneous affine control has `S1,S2` at roundoff.
- Manufactured source errors converge with L2 order at least 1.8 away from
  degeneracy.
- CPU/GPU nonfinite flags and threshold counts agree exactly.

## Regression surface

- Source index pairing (`A psi1` uses `S2`), units of epsilon, and denominator
  behavior near stagnation.

## Failure and rollback policy

- A nonfinite output or unexplained denominator collapse fails the increment.
- Do not increase epsilon merely to make a failing manufactured case pass.

## Completion checklist

<!-- completion-checklist:start -->
- [x] c, both sources, and explicit regularization are implemented.
- [x] Source pairing and sign are covered by tests.
- [x] Homogeneous and manufactured thresholds pass.
- [x] Nonfinite and degeneracy diagnostics agree with CPU reference.
- [ ] Full regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-09 complete and selects SF-10.
<!-- completion-checklist:end -->

## Advancement rule

SF-10 may assemble the coupled residual from these validated sources.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-07T14:58Z | active; base `master=origin/master=3b7a0a05e53732f415f84e2794f4772f340f9bf6` | Activated SF-09 documentation state under the Claude Code orchestration harness. | Preflight verified on the default branch: clean tree; `master` and `origin/master` at `3b7a0a0`; SF-08 `done` with PR #18 merged as `855dcf14458d3ac92ef31a7a30e373d5d4b16a1b`; checker PASS (`29 increments, next=SF-09`); no open PRs. Baseline validation on the base commit: `cmake --preset wsl-debug` (sccache launcher disabled locally with `-DCMAKE_CUDA_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER=` because sccache 0.17.0 currently fails CUDA compiles with `fatbinary fatal: Could not open input file '*.ptx'`; sccache is documented optional), `cmake --build build/wsl-debug -j` exit 0, `ctest` 2/2 PASS, PSPTA-small smoke completed. Persistent Goal `Implementar c, S1, S2 y la regularización explícita del denominador.`; delivery branch `science/lester-sf09-nonlinear-sources`; execution uses Claude-managed per-node isolated worktrees. | Build the SF-09 intra-increment DAG and delegate implementation to isolated workers. |
| 2026-08-07T15:45Z | integration validation; `ad7a04acd6bf4013c42f245cf46e31e0ab4ebf3e` | Three-node DAG executed by isolated Sonnet workers, each independently audited, then combined by a single isolated integrator. Approved nodes: T01 CPU oracles `1449ede09d0ec174487761ac2f7b476569426e0d`; T02 production kernel `d41493fecfc1d1be614552efb5bd0bf52a8a4127` (integrated as patch-id-identical `d4d54ea`, parent reordered for dependency-compatible order, content diff empty); T03 GPU acceptance cases `ad7a04a`. A first pair of T01/T02 workers was lost to a session interruption before committing; their replacements are the only evidence used. Integrator verified base `698ff8a`, exact 8-file/1627-insertion diffstat, empty content diffs vs every approved commit, no conflicts, no integration-only edits. | Integrator validation (fresh worktree, sccache launcher disabled): submodule init, configure, build (105 targets), checker, 8/8 `nonlinear_sources_*` cases, targeted CTest 1/1, full CTest 2/2, `run_operator_tests` 8/8, PSPTA-small smoke, `git diff --check` — all exit 0. Key metrics: GPU-vs-long-double oracle worst normalized RMS/boundary Linf `3.727e-14` (threshold `5e-11`); masked convergence (analytic `\|c\|^2>=10` mask, kept fraction 0.938–0.950, production `epsilon=1e-2`) L2 orders S1 `2.280/2.083`, S2 `2.912/2.094`, masked Linf strictly decreasing; unmasked regularization-dominated cross-check (`epsilon=5`) orders `1.933/1.978/1.982/1.980`; pure-affine control `max\|S\|=0` exactly with zero nonfinite counters; two-epsilon explicitness `3.73e-14` each with GPU fields genuinely differing (`max diff 2.48`); exact CPU/GPU count agreement — degeneracy `1653=1653`, `4096=4096` with separations `8.2e-10` and `1.0e-2` (guard `>1e-10`), nonfinite `3=3` S1 and `3=3` S2; error contract 67 checks (66 `invalid_argument` + 1 accepted read-only overlap, Linf `1.42e-27`); mutants pairing_swap `1.335`, cross_order_flip `2.0`, b_sign_flip `2.0`, unregularized_denominator `59.5` (threshold `1e-2` each). Hardware: local Debug `sm_86` RTX 3050 Laptop; kernel resources `REG:45 STACK:0 SHARED:0 LOCAL:0` (register-only, cuobjdump). | Orchestrator final audit. |
| 2026-08-07T15:45Z | root final audit PASS; head frozen at `ad7a04a` | Orchestrator personally audited the full diff `698ff8a..ad7a04a` (8 files, +1627) against the SF-09 spec: exact locked formulas `c=g1×g2`, `d=\|c\|^2+(epsilon·v_rms)^2` (only regularization; no hidden floor; `epsilon` default `1e-2`, `v_rms>0` validated), `S_i=((B×g_i)·c)/d` with correct index pairing; pointwise register-only kernel writing only the two source arrays plus caller-owned counters (`[0]/[1]` nonfinite S1/S2, `[2+t]` strict `\|c\|^2<(tau_t·v_rms)^2` in double); enqueue performs exactly one documented `cudaMemsetAsync` (counter zeroing) plus the launch — no allocation, copy, or host sync; complete `std::invalid_argument` contract incl. cross-type byte-range aliasing checks; scope additive-only, no SF-10 work, PSPTA/configs untouched. Orchestrator independently reran on the control checkout at `ad7a04a`: build exit 0, all 8 cases `verdict=PASS`, full CTest 2/2, `run_operator_tests` 8/8, smoke OK, `git diff --check` clean, checker PASS. | Gate 1 PASS; Gate 2 PASS; Gate 3A operator subset PASS for the SF-09-applicable metrics (explicit configurable denominator regularization, degeneracy/nonfinite diagnostics, convergence order with documented away-from-degeneracy mask). Coupled residual `r_F`, reconstruction `e_v`, invariance `e_i`, divergence `e_div`, and gauge-restoration metrics are N/A for this pre-residual operator increment and are not inferred; Gate 4/Gate 5/V100 N/A, no claim made. Known scientific observation recorded: the smooth fixture contains naturally near-degenerate cells (min `\|c\|^2`≈`9.5e-4` at 32^3), so unmasked convergence at production epsilon degrades (~0.65) — expected source sensitivity as `\|c\|→0`, motivating the documented mask and the SF-17 epsilon-continuation design; epsilon was not raised to pass any case. Implementation frozen; mandatory human review pending. | Publish PR as awaiting_review; do not advance NEXT. |
| 2026-08-07T15:52Z | awaiting_review; PR [#19](https://github.com/santi-esquerre/MacroFlow3D/pull/19); frozen audited source head `ad7a04acd6bf4013c42f245cf46e31e0ab4ebf3e` | Published the frozen SF-09 implementation for mandatory human review on branch `science/lester-sf09-nonlinear-sources`. The PR records scope, DAG/worker/integrator provenance, exact commands, oracle/order/zero/epsilon/count/contract/mutant metrics, gate determinations, the near-degeneracy convergence observation, and intentionally untouched areas. | Metadata commits after `ad7a04a` are documentation-only (`a3af6b4` evidence/state, this row); no source, test, or CMake change after the audited head. Residual risks: local Debug `sm_86` evidence only; mask/epsilon test constants are fixture-specific and documented; `analytic_hessian_vector_b` validity domain documented by T01. | Await explicit human review of PR #19; on approval add only the closure metadata commit (done/checklist/dashboard NEXT→SF-10) on the same PR; do not merge. |
