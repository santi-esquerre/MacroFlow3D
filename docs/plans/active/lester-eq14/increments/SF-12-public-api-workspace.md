# SF-12 — Public API and workspace

- State: `awaiting_review`
- Goal: `Definir la API pública, ownership y workspace reutilizable del solver.`
- Depends on: `SF-11`
- Unlocks: `SF-13`
- Branch: `science/lester-sf12-public-api-workspace`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-08T19:55Z`
- Completed: `not completed`
- PR: [#24 — SF-12: public streamfunction API, owned workspace, and exact memory estimator](https://github.com/santi-esquerre/MacroFlow3D/pull/24) (`awaiting_review`)
- Commit: `not recorded`

## Scientific or engineering intent

Establish explicit data ownership and reusable GPU storage before nonlinear
loops make allocation and coupling behavior difficult to change.

## Preconditions

- SF-11 completes the numerical primitive and report inventory.

## In scope

- Public problem/config/result types, streamfunction fields, persistent
  workspace, memory estimator, and module boundaries.

## Out of scope

- Pipeline YAML, Picard behavior, field export, and transport consumption.

## Files and symbols

- Add `StreamfunctionTypes.hpp`, `StreamfunctionWorkspace.cuh/.cu`, and
  `StreamfunctionSolver.cuh` under `src/physics/streamfunctions/`.
- Define `AffineGauge`, `StreamfunctionProblemView`, `StreamfunctionFields`,
  `StreamfunctionSolverConfig`, and `StreamfunctionSolveReport`.

## Implementation specification

1. Accept nonowning grid, `K` or `Y`, reference Darcy velocity, BCs, and gauge;
   reject inconsistent dimensions and nonperiodic v1 benchmark BCs.
2. Own `u1/u2` in `StreamfunctionFields`; own all scratch and solver vectors in
   `StreamfunctionWorkspace`.
3. Allocate for sequential block solves by default and expose no concurrent
   solve mode yet.
4. Report fine-grid-equivalent field count and exact allocated bytes.

## Expected numerical effect

None beyond existing primitives; memory allocation becomes predictable and
hot-loop-safe.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- API type/size validation tests pass.
- Repeated workspace use performs no allocation after construction/resizing.
- Estimated bytes equal actual owned `DeviceBuffer` capacities.

## Regression surface

- Include dependencies, CUDA ownership, move semantics, and peak memory.

## Failure and rollback policy

- Do not expose unstable internal kernels in the public interface.
- If the 24.6-field budget cannot be met, record each extra field and redesign
  ownership before SF-13.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Public types and ownership boundaries are implemented.
- [ ] Workspace covers all accepted primitives without hot-loop allocation.
- [ ] Memory estimator is tested at all target grid sizes.
- [ ] API validation and full regression tests pass.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-12 complete and selects SF-13.
<!-- completion-checklist:end -->

## Advancement rule

SF-13 may implement the first complete homogeneous solve through this API.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-08T19:55Z | activation on `master=6934291` (SF-11 closure merged via PR #23) | SF-12 activated after verifying `NEXT: SF-12`, SF-11 `done`, and checker `OK (29 increments, next=SF-12)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) `AffineGauge` already exists as the accepted SF-06 type in `affine_gauge.cuh`; SF-12 re-exports it through `StreamfunctionTypes.hpp` instead of redefining it. (2) `StreamfunctionSolver.cuh` is declaration-only; the `solve_streamfunctions` body is SF-13's explicit deliverable (`StreamfunctionSolver.cu`), so no stub implementation is added and nothing may call the declaration yet. (3) The "estimated bytes equal actual owned DeviceBuffer capacities" threshold requires byte introspection of sub-workspaces whose buffers are private; SF-12 adds additive, behavior-neutral `allocated_device_bytes()`/estimate accessors to the accepted SF-03/SF-05/SF-06/SF-10/SF-11 workspace types (no numerical or control-flow change). (4) "Tested at all target grid sizes" is interpreted as allocation-backed estimator==actual equality at `16^3/32^3/64^3` on the local 4 GiB GPU plus pure-host estimator evaluation with closed-form cross-checks at `128^3/256^3` (the full `256^3` workspace exceeds local VRAM). (5) Solver-level validation enforces the inherited SF-02/SF-05/SF-06/SF-10 isotropic-spacing and MG-coarsenable restrictions and triply periodic BCs, while SF-11 diagnostics alone remain anisotropy-capable. (6) Estimator output is categorized (fields / solve path / diagnostics path) and compared against the plan's 24.6-field budget; the accepted-primitives inventory is expected to exceed it, triggering the spec's rollback-policy path: record each extra field and produce an ownership-redesign decision record before SF-13. (7) Workspace move-safety: the MG preconditioner holds `MGHierarchy*`, so the hierarchy is held behind a stable address (`unique_ptr`) or moves are explicitly deleted with documented rationale. | Base commit is this activation commit on `master=6934291`. Gate 1 + Gate 2 apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-08T22:05Z | `0610cd0`, integration validation | Four-node DAG completed and orchestrator-audited node by node: T01 `71d8828` (public types `StreamfunctionTypes.hpp`, `StreamfunctionFields` owning `u1/u2`, unified `StreamfunctionWorkspace` owning all scratch/solver storage incl. one MG hierarchy behind `unique_ptr` + one projected-PCG workspace for sequential-only block solves, declaration-only `StreamfunctionSolver.cuh`, exact memory estimator, additive behavior-neutral byte-introspection accessors on the accepted SF-03/05/06/10/11 workspaces); T02 `417599a` (six GPU acceptance cases); T03 `6af9185` (decision record on the memory inventory vs the 24.6-field budget, status `proposed`); corrective C01 `0610cd0` (SF-04 provenance fixes, scoped capacity-never-shrinks doc, `problem.grid` consistency rejection + 2 test checks, decision-record provenance sentence — resolving all four MINOR audit findings). Single integrator verified the linear chain (merge-base == base, 20 files +2767/−0, `diff --check` clean, `solve_streamfunctions` declaration-only, docs/plans/apps/particles/multigrid untouched) and reran the full suite green. | Acceptance evidence (worker + integrator + orchestrator reruns all agreeing): error contract 76/76 incl. per-face BC, per-direction spacing/extent, MG-coarsenability, `problem.grid` consistency, and 12 use-before-prepare checks; estimator == actual owned `DeviceBuffer` capacities EXACTLY (every report field, three-way vs an independent closed-form reconstruction) at `16^3`=2,241,941 B, `32^3`=17,762,195 B, `64^3`=141,830,801 B, host-only at `128^3`=1,134,065,039 B and `256^3`=9,070,736,783 B; allocation freedom: warmup + 3 full-chain repeats (affine RHS, coupled residual, physical diagnostics, coefficient hierarchy, projected PCG+MG solve) with byte-identical owned bytes/pointers and exact `cudaMemGetInfo` stability, PCG converged each pass; re-prepare semantics: non-MG capacities/pointers never shrink/move (16,558,451 B constant), aggregate restored exactly at the original grid/config; budget report: 67.58 fine-grid-equivalent fields (~8.45 GiB at `256^3`) > 24.6 recorded explicitly per the rollback policy, with every extra field attributed (SF-10 residual 21.0, SF-11 diagnostics 27.0, PCG 5.0, MG 4.57, scratch 8.01, fields 2.0) and four adjudicable redesign options in `docs/decisions/2026-08-08-sf12-streamfunction-memory-inventory-vs-budget.md`. Full suite: ctest 2/2, streamfunction runner 89/89 PASS, `run_operator_tests` 8/8, PSPTA smoke OK, checker OK. Hardware: RTX 3050 Laptop 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout, then publish PR as `awaiting_review`. |
| 2026-08-08T22:20Z | `0610cd0`, final audit PASS | Orchestrator personally re-audited the integrated head against the original spec on the control checkout: fresh reconfigure/build, ctest 2/2, 89/89 case verdicts, 8/8 operator tests, smoke, checker all green; every spec acceptance threshold and checklist item (except the two human-review/closure items) has explicit evidence; the only behavior change beyond new files is the C01 `problem.grid` consistency rejection; accessor additions to accepted modules verified hunk-by-hunk as behavior-neutral. Gate 1 + Gate 2 PASS; Gate 3A/4/V100 N/A (no solver loop or numerics introduced). | Flagged for the human reviewer: (1) the 24.6-field budget overshoot (67.58) and the `proposed` ownership-redesign decision record — option (a) bring-up acceptance recommended, (b) gradient-borrowing as first optimization; (2) the seven interpretive decisions recorded at activation; (3) mandatory-review path `src/physics/streamfunctions/`. Frozen audited source head: `0610cd0`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-08T22:30Z | `57488f4` published, PR #24 open | Delivery branch pushed and [PR #24](https://github.com/santi-esquerre/MacroFlow3D/pull/24) opened as `awaiting_review` with the frozen audited source head `0610cd0` (later commits on the branch are increment-state documentation only). | PR description carries the DAG, audit summaries, full acceptance evidence, the budget-overshoot decision record (`proposed`), interpretive decisions, and remaining risks. No agent merges; `NEXT` remains `SF-12`. | Await explicit human review/approval of PR #24; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-13`) on this same PR. |
