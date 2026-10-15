# SF-33 — GPU inlet-label streamfunctions (equation (14) on the `x1`-non-periodic slab)

- State: `done`
- Goal: `Implementar en GPU, dentro de src/physics/streamfunctions/, la formulación de etiquetas de entrada decidida por SF-29 (slab no periódico en x1, condición de salida, esténciles de cuarto orden y Newton-Krylov con continuación) y verificar que reproduce el prototipo CPU y converge bajo refinamiento hasta 128^3.`
- Depends on: `SF-29`
- Unlocks: `SF-34`
- Branch: `science/lester-sf33-gpu-inlet-label-streamfunctions`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator (sessions 2026-10-06/07)`
- Started: `2026-10-06T18:45Z on base 37bfb25`
- Completed: `2026-10-07 (owner approval of the audited head ddf8bec in the orchestrator session; claim (a) and (d) established, claim (b) not established and superseded by docs/decisions/2026-10-07-label-transport-constructor.md; closure metadata commit on PR #51)`
- PR: `https://github.com/santi-esquerre/MacroFlow3D/pull/51` (open; human-review increment; audited source-bearing head `ddf8bec845eb0463e456707f1d0b161ef72f56a7`)
- Commit: `ddf8bec845eb0463e456707f1d0b161ef72f56a7` (source-bearing audited head); metadata commits above it on the PR branch

## Scientific or engineering intent

Implement on the GPU the formulation locked by `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`
(items 1-7) and establish two claims: (a) the GPU code solves the same discrete problem as the SF-29 CPU prototype
(identical inputs give identical metrics); (b) on the production stack (SF-18 field, SF-19 Darcy) the labels
converge to the Darcy labels under refinement beyond the prototype's 32^3, and the open item `sigma_Y = 1`
(decision item 6) is measured at `ell/h = 16, 32`. It does not claim anything about transport or `alpha_T`.

Context: `docs/experiments/2026-10-02-sf29-inlet-labels.md` (R1-R8),
`docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`,
`docs/experiments/2026-10-05-sf30-streamline-closure-gate.md` (production-stack oracle instrument).

## Preconditions

- `SF-29` is `done` on the default branch (decision record accepted, prototype artifact and saved 16^3 solutions
  committed).
- `SF-28` (periodic tricubic spline) and `SF-30` (closure-gate integrator, SF-19 potential accessor) are `done`
  on the default branch.

## In scope

- New files under `src/physics/streamfunctions/inlet_slab/` (new subdirectory): slab grid and storage (vertex
  planes `x1 = j/N`, `j = 0..N`, plane 0 = inlet Dirichlet data, unknown planes 1..N, periodic in `x2`, `x3`),
  4th-order stencils (decision item 4), the residual (equation rows on planes 1..N-1, outlet oblique rows on
  plane N, decision item 3), a matrix-free Jacobian-vector product, Newton-Krylov with a preconditioner validated
  against direct solves, amplitude continuation with bisection (decision item 5).
- Inlet labels (D-1, decision item 2) from the SF-19 face velocity on `x1 = 0` (normalized triangular
  construction, computed spectrally in `x2`, `x3` on the face), and `v_perp,in` for the outlet rows.
- Gate 3A metrics through `Diagnostics.cuh` where applicable, plus per-label `e_psi` (SF-29 normalization) and
  `e_v` with a 4th-order reconstruction.
- A production-stack oracle: streamline tracing of the SF-19 potential on the SF-28 spline (the SF-30 integrator,
  `apps/closure_gate/streamline_integrator.hpp`) from every vertex back to the inlet plane, inlet labels evaluated
  at the feet, with its round-trip check (the prototype's oracle method).
- A small app/driver for the experiment runs and fast contract tests (stencil exactness, Jacobian vs FD, `k = 1`
  control) registered in `ctest`; the convergence ladders are documented experiments, not `ctest` entries.
- An export script under a new artifact directory `docs/experiments/artifacts/<date>-sf33-gpu-inlet-labels/`
  that reads the SF-29 prototype inputs through `cases.load_case` of the SF-29 artifact (read-only).

## Out of scope

- Any change to the frozen periodic stack (every existing file under `src/physics/streamfunctions/`,
  `src/numerics/`, `src/multigrid/`), to SF-18, SF-19 (beyond read-only use of existing accessors) or SF-28.
- Runner/pipeline wiring, YAML configuration of the pipeline, the long domain, the tracker, transport.
- Backflow (`v1 <= 0` on the inlet face): detected and reported, not handled.
- The periodic-medium acceptance against SF-30's return map (SF-34).

## Files and symbols

- `src/physics/streamfunctions/inlet_slab/*` (new): e.g. `InletSlabGrid.cuh`, `InletLabels.{cu,cuh}`,
  `SlabStencils4.cuh`, `SlabResidual.{cu,cuh}`, `SlabJacobianVectorProduct.{cu,cuh}`,
  `SlabNewtonKrylov.{cu,cuh}`, `SlabOracle.{cu,cuh}` (names indicative; one responsibility per file).
- Read-only reuse: `src/physics/stochastic/` (SF-18 generator), `src/physics/flow/AffinePeriodicFlowSolver.cuh`
  (SF-19 solve and `potential_fluctuation()` accessor), `src/numerics/interpolation/` (SF-28 spline),
  `apps/closure_gate/streamline_integrator.hpp` (SF-30 integrator), `src/physics/streamfunctions/Diagnostics.cuh`,
  `src/physics/streamfunctions/CoupledGmres.cuh` if its interface fits (no modification).
- `apps/inlet_slab/` (new driver), `tests/` (new fast tests), `CMakeLists.txt` (new targets only).
- `docs/experiments/<date>-sf33-gpu-inlet-labels.md` (new experiment note) and its artifact directory.

## Implementation specification

1. Grid and data layout: vertex grid `(N+1) x N x N` on `[0, 1]^3`, `h = 1/N`, unknowns `u_i` on planes 1..N
   (`psi1 = x2 + u1`, `psi2 = x3 + u2`); double precision; one preallocated workspace, no allocation in the
   Newton/Krylov loops.
2. Stencils (decision item 4, the prototype's `Stencils4`/`derivs4`): centered 4th order in `x2`, `x3` and on
   planes 2..N-2; skewed 4th order on planes 1 and N-1 (`d1` on 5 planes, `d11` on 6 planes, Fornberg weights);
   one-sided 5-point `d1` on plane N for the outlet rows; mixed `d1j` = the `x1` stencil applied to the 4th-order
   `d_j`. Weights are generated once on the host and checked on polynomials in a fast test (exact to degree 4/5).
3. Residual: `F_i = -q (lap_h psi_i - grad(ln k) . grad_h psi_i - S_i)` on planes 1..N-1 with
   `S_i = ((B x grad psi_i) . c) / |c|^2` (same index), `c = grad_h psi1 x grad_h psi2`, no regularization; outlet
   rows `(2q/h) (v_perp,in - c_perp)` (Neumann `d1 psi_i = 0` when `v_perp,in = 0`). `grad ln k` is exact where
   the field is analytic (prototype inputs) and the SF-18 spectral gradient otherwise; the choice is logged.
   Norms `r_F` (equation rows) and `r_out` (outlet rows) as in the prototype.
4. Jacobian-vector product: analytic directional derivative of the residual, verified against central FD
   (step ladder) in a fast test at 12^3.
5. Newton-Krylov: right-preconditioned restarted GMRES/FGMRES. The preconditioner is to be validated: the
   per-mode `k = 1` operator of the prototype is a baseline only (it fails beyond `eps = 0.25` at `N >= 32`,
   SF-29 R7); candidates such as a block multigrid on the frozen-coefficient linearization must be shown, at
   16^3-32^3, to reproduce the direct-solve Newton iterates (step residual <= `lin_tol`) before they are used at
   64^3-128^3. Every Newton step is solved to a logged linear tolerance; linear failures are recorded separately
   from nonlinear failures.
6. Continuation: amplitude `eps` in `k = exp(eps Y)`, stages `0.25 -> 0.5 -> 1`, bisection on failure (minimum
   stage logged), each stage warm-started from the previous accepted state; the full path is logged (PATH line
   format of the prototype).
7. Inlet labels and oracle: (a) inlet face `v1` from SF-19 (or the prototype input in step 9a), positivity check
   `min v1 > 0` with abort and a recorded `inlet_backflow` status otherwise; (b) production oracle = backward
   tracing with the SF-30 integrator on the SF-28 spline of the SF-19 potential, round-trip check per plane,
   tolerance ladder 1e-8 / 1e-10.
8. SF-19 cross-check: on the closure-probe `gauss` field (the SF-29 field, analytic `ln k`) solve SF-19 at
   16/24/32 and compare its inlet-face `v1` and `v_perp,in` with the prototype's spectral reference
   (`cases.load_case`): record the differences and their observed order.
9. Runs (detached V100 jobs):
   (a) prototype reproduction: inputs exported from the SF-29 artifact (analytic `ln k`, `grad ln k` at vertices,
   inlet labels `psi0`, `v_perp,in`, oracle labels) for `gauss`, `gauss_ch`, `control2d` at `eps` 0.25 and 0.5,
   `N` in {16, 24, 32} where the prototype has an `i1o4` CASE line in
   `docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/raw/sweep2/` (16/24 everywhere listed, 32 only for
   `gauss:0.25`, `gauss_ch:0.25`), plus a field-level comparison with the saved 16^3 solutions
   (`raw/sweep2/solutions/*_16_*.npz`);
   (b) production refinement: one SF-18 continuum field with Gaussian covariance, `ell = 1/4`,
   `sigma_Y = eps` in {0.5, 1.0}, grids 32/64/128 (`ell/h` = 8, 16, 32), labels vs the production oracle of
   step 7b;
   (c) Gate 3A metrics, V100 wall time and peak device memory at 128^3.
10. Failure behavior: NaN/Inf, `inlet_backflow`, linear-solver failure, continuation floor reached, and
    stagnation are distinct logged exits; nothing is clamped.

## Expected numerical effect

New capability only; no existing output changes. Expected: (a) GPU metrics equal to the prototype's to roundoff
amplification of the same discrete problem; (b) at `eps = 0.5` orders near the prototype's (1.8-2.1 on 20-28)
or higher on 32/64/128, limited by the 2nd-order SF-19 inlet velocity and the oracle's spline accuracy; at
`eps = 1` the orders are not predicted (open item of the decision record). The frozen periodic stack and every
existing configuration must produce byte-identical outputs.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R inlet_slab      # fast contract tests only
scripts/remote --increment SF-33 sync
scripts/remote --increment SF-33 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
scripts/remote --increment SF-33 run sf33-proto -- "<step 9a driver command>"
scripts/remote --increment SF-33 wait sf33-proto
scripts/remote --increment SF-33 run sf33-ladder -- "<step 9b driver command, 32/64/128, eps 0.5 and 1.0>"
scripts/remote --increment SF-33 wait sf33-ladder
scripts/remote --increment SF-33 run sf33-ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote --increment SF-33 wait sf33-ctest-full
scripts/remote --increment SF-33 run sf33-bytecmp -- "<byte-compare of the three default configs before/after, as in SF-27>"
scripts/remote --increment SF-33 wait sf33-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- (a) Prototype reproduction (same discrete problem): for every case of step 9a, `e_v` and `e_psi` equal to the
  prototype's `i1o4` CASE values within 1e-6 relative, `r_F <= 1e-10`, and the saved 16^3 solutions reproduced
  field-wise (max relative difference recorded; expected at the level of the solver tolerance).
- (b) Production refinement on the same continuum field: at `eps = 0.5`, `e_v` and `e_psi` observed orders
  >= 1.8 on both pairs of 32/64/128 with `r_F <= 1e-10` and no floor. At `eps = 1` the measured orders are
  recorded: PASS if >= 1.8 on both pairs; otherwise the open item of the decision record stays open with the
  measured orders. No criterion presupposes the answer.
- (c) Gate 3A metrics, V100 wall time and peak device memory reported at 128^3.
- (d) Production oracle: round trip <= 1e-8 relative on every plane; the oracle labels at 32^3 agree with the
  SF-29 spectral oracle on the step-8 field to the order of the SF-19/spline discretization (recorded).
- SF-19 cross-check (step 8) recorded with its observed order.
- Fast contract tests pass; full `ctest` on V100 green; byte-compare of the default configs clean.

## Regression surface

- The frozen periodic stack (SF-02..SF-26): no file changes; `git diff <base> <head> -- src/physics/streamfunctions/*.cu* src/physics/streamfunctions/*.hpp src/numerics src/multigrid` must be empty.
- Default pipeline configs: byte-compare as in SF-27 (only manifest timestamps may differ).
- `ctest` wall time (new entries must be fast contract tests).

## Failure and rollback policy

- (a) not met: implementation defect; the increment stays active until the discrete problems agree.
- (b) not met at `eps = 0.5`: stop and return to the owner with the ladders (no tolerance or threshold tuned);
  the decision record's validity claim is then re-examined.
- Preconditioner not validated against direct solves: the 64^3-128^3 runs do not start.
- `inlet_backflow` on a planned field: recorded, the case is excluded only with an owner decision.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded (fast tier, campaign A and the gate-reduced campaign B; claim (b) recorded as not established at the spec amplitudes — superseded by the owner decision of 2026-10-07, constructor = label transport, `docs/decisions/2026-10-07-label-transport-constructor.md`).
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [x] Required human review is recorded (owner approval of the audited source-bearing head `ddf8bec` on 2026-10-07; closure metadata only afterwards).
- [x] PR and commit identifiers are recorded.
- [x] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-34` becomes eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-06T00:00Z | not started | Specification created by the SF-29 closure PR (decision record `docs/decisions/2026-10-06-eq14-inlet-label-formulation.md`). | Formulation locked with bounded validity (owner, Option A). | Activate only when the checker reports it READY. |
| 2026-10-06T18:45Z | activation on `master=37bfb25` (checker OK ready=SF-32 SF-33, nonterminal=none; SF-32 active in a parallel session; no dependency; per-increment mirror `--increment SF-33`) | UNDERSTAND: numerical contract in the orchestration record (discrete problem locked to the SF-29 `i1o4`; exact-pair control `k = k(x1)`, `u1 = Phi(x3)` derived; two-stage preconditioner plan with a validation gate; inlet `v1` = SF-19 U-face flux, `vperp`/`vD` = spline flow; oracle by `trace_to_plane`; recorded deviations: `Diagnostics.cuh` not applicable (periodic cell-centred), `CoupledGmres.cuh` not reusable, byte-compare = the SF-26/27 trio, reachable-amplitude enumeration for the stage inputs). Scientific-rigor skill invoked (categories B + C). DAG N0 -> {N1, N3, N4} -> N2 -> N5 -> N6 (gate) -> N7 -> N8 -> N9; one integrator. | Human-review increment. | Launch N0; base refs job on V100. |
| 2026-10-06T19:19Z .. 2026-10-07T01:00Z | chain accepted (details in `docs/experiments/2026-10-06-sf33-gpu-inlet-labels.md`): N0 `2766beb` (foundation; criterion restated to the prototype's 32->64 stencil-order gate), N1 `0d6755c` (analytic JVP; k = 1 outlet rows `+(2q/h) d1`), N2 `4366127` (GMRES, P-A = full plane-averaged linearization exact at k = 1 and x1-only states, Newton, continuation), N3 `2be35a8` (D-1 labels, SF-18/19/28 inputs, production oracle; FINDING: round trip ~h_max^3 on the spline flow -> `h_max = h/8`), N4 `f198975` (exporter, `.npy`, loader; N0 metrics = prototype on real data to 4e-15), C1 `0681c19` (robust test gates; FINDING: legacy-stream pageable copies race the non-blocking ctx stream, 11/15 failures under load), N5 `8dee959` (driver, scripts; local 16^3 reproduction to 1e-14), C2/C2b `82a4af9`/`681b039` (stream-safe copies), N6 `f5b050f` (V100 campaign A: 12/14 of 9a reproduce the prototype to ~1e-14; gate FAIL at 24^3 eps 0.5: GMRES+P-A stall), N7a `a198369` (Eisenstat-Walker forcing), N7b `30347aa` (pseudo-transient continuation, SER), N7c `d093c7a`..`ea53d09` (Galerkin coarse correction on the weak `xi1 = 0` subspace: 24^3 eps 0.5 converges to 1e-13; colored assembly + banded LU), C3 `bfe4efb` (driver defaults), N6b `3676bb0` (V100: the full 9a matrix PASSES at <= 3.8e-11 vs full-precision references; 32^3 eps 0.5 FAILS -> gate FAIL for 64-128 at eps 0.5/1.0, spec rule applied). | Audits in `.claude/orchestration/SF-33-gpu-inlet-label-streamfunctions/audits/` (runtime record); evidence under the artifact `analysis/`. | Campaign B reduced to the gate-validated amplitude (eps 0.25). |
| 2026-10-07T01:08Z .. 13:37Z | N8' `13c6bf8`, C4 `964c43a` (grid-scaled shift `mu_0 (h_ref/h)^2`), N8'' `e113ff9`: eps 0.25 production ladder (SF-18 `sigma^2 = 1`, `ell = 1/4`, seed 3001) — 32^3 and 64^3 converged (orders 32->64: `e_v` 2.13, `e_psi` 3.01, `e_psi1` 3.08, `e_psi2` 2.87, `e_i` 2.54/2.35, `e_div` 3.75), 128^3 `continuation_floor` (genuine GMRES plateau for eps >= 0.14, flat restart curves; coarse LU 79 % of 2 h 59 min; peak device 6.5 GB; host RSS 4.8 GiB); production 32^3 eps 0.5 converged; oracle round trips <= 1.2e-9 at `h/8` on every grid (`applied_scale` identical across grids); 32^3 oracle vs the SF-29 spectral oracle 0.6 % RMS, order 2.00. Orchestrator job with a relaxed GMRES stagnation rule cancelled (cannot reach eta within the cap at the measured rate). N9 `ddf8bec`: experiment note. | Acceptance: (a) PASS on the full 9a matrix; (b) NOT MET at eps 0.5/1.0 (not run: spec rule) and at eps 0.25 met on 32->64 only; (c) recorded (converged at 64^3; 128^3 for a non-converged run); (d) PASS; SF-19 cross-check orders 1.92-2.04 recorded. | Integrate; full ctest + byte-compare on V100; publish as `blocked`. |
| 2026-10-07T14:50Z | FINAL_AUDIT positive on the integrated head `ddf8bec` (fast-forward of the 25 audited commits; no integration changes); State -> `blocked` | V100 (`scripts/remote --increment SF-33`, mirror `~/MacroFlow3D-SF-33`, preset `v100-release`, CUDA 11.4; source of the mirror tree = head source, `git diff 964c43a..ddf8bec -- src apps tests CMakeLists.txt` empty): job `sf33-ctest-full-bytecmp` — **26/26 tests passed, Total Test time (real) 2787.25 s** (20 pre-existing + 6 `inlet_slab_*`); byte-compare of the SF-26/27 precedent trio (`config_pspta_small`, `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation`) vs `~/sf33_base_refs` (base `37bfb25` build): IDENTICAL excluding manifests, manifests identical modulo timestamps. Local: wsl-debug build, 6/6 fast tests, `ctest -N` = 26, checker OK. Frozen stack: `git diff 37bfb25..ddf8bec` on `src/**` (except `inlet_slab/`), `apps/**` (except `inlet_slab/`), `CMakePresets.json`, `scripts/` empty. | Increment outcome: Goal part 1 met (claim (a)); Goal part 2 ("converge bajo refinamiento hasta 128^3") NOT met at the spec's amplitudes — spec failure policy "(b) not met at eps = 0.5: stop and return to the owner"; `blocked` until the owner decides (options in the experiment note's Result section: scalable-preconditioner increment / accept the partial and re-scope SF-34 / re-examine the decision record's resolution claim). | PR opened for the owner (human-review increment; no agent merges); SF-34 stays blocked. |
| 2026-10-07T16:30Z | closure metadata (State -> `done`); source-bearing head unchanged (`ddf8bec`) | Owner decision (orchestrator session 2026-10-07, after the options analysis `.claude/orchestration/SF-33-gpu-inlet-label-streamfunctions/analysis/options-analysis.md`): the audited head is approved; the deliverables are accepted as the slab discrete problem (claim (a)), the production oracle and inlet labels (claim (d)), the production setup, the metrics and the fast tests; claim (b) remains NOT established at the spec amplitudes (linear-solver plateau; no threshold tuned) and is superseded, not resolved: the Newton-Krylov solve of equation (14) is retired as production constructor and label transport by backward streamline tracing is adopted (decision record `docs/decisions/2026-10-07-label-transport-constructor.md`; increment SF-35). The eq. (14) residual and the FD metrics remain as diagnostics; the slab solver stays an instrument for cross-construction at <= 64^3, eps <= 0.5. | Human review recorded in the checklist; master-checklist entry checked; checker OK. No source, test or scientific-configuration change after approval. | SF-35 specified in the same PR (docs only); it becomes READY once this closure is merged on the default branch. |
