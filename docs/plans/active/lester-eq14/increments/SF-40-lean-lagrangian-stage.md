# SF-40 — Lean parametric Lagrangian stage (Grid3D, memory level S+), bitwise tracker speedups and Par2 dead-buffer cleanup

- State: `pending`
- Goal: `Bajar el pico de memoria de dispositivo del camino lagrangiano (campo SF-18, flujo SF-19, splines SF-28, tracker) con un stage liviano paramétrico en (nx, ny, nz) para que la caja de 3328x384x384 celdas (4.9e8; hoy caben 1.5e8) corra en una V100-32GB con pico medido <= 30000 MiB, con salidas del tracker idénticas byte a byte al camino actual, speedups bit a bit del tracker y sin buffers muertos en el baseline Par2.`
- Depends on: `SF-37`
- Unlocks: `SF-38`
- Branch: `science/lester-sf40-lean-lagrangian-stage`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 (change type B: no intended numerical change; invariance target = byte identity)`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Authority: `docs/decisions/2026-10-22-lean-lagrangian-stage.md` (owner decisions of 2026-10-22) and the audit
`docs/experiments/2026-10-22-lagrangian-memory-audit.md` (§4 memory map, §8 bitwise levels, §9 feasibility). The current
cube stage peaks at ~213 B/cell by exact bytes (measured nvidia-smi peaks 25921-27981 MiB at 1.34e8 cells), so it
fits ~1.5e8 cells in 30000 MiB. SF-38/SF-39 need boxes of ~4.9e8 cells (3328x384x384) at the `lambda/h` fixed by SF-37
(8 / 16 / 32 at `sigma^2` 1 / 2.25 / 4). SF-40 lowers the device peak to memory level **S+** (60.57 B/cell at the SF-19
PCG peak, context excluded) by lifetime and placement changes only. Memory and time only; every numerical result stays
bit for bit.

Allowed claims:

- (a) the lean path reproduces the current path's outputs byte for byte on the cube;
- (b) the exact per-phase device bytes equal the closed form of level S+;
- (c) whether 3328x384x384 fits in 30000 MiB (measured);
- (d) the non-cubic stage is correct on exact controls;
- (e) bitwise tracker speedups (measured, no threshold);
- (f) the Par2 `FlowWorkspace` is 24 B/cell smaller with byte-identical outputs.

Not allowed: any `alpha` estimate (`alpha_T` is not presupposed, Rule 1), any change of numerical semantics, any claim
about `lambda/h` (SF-37 owns it).

## Preconditions

- `SF-36` and `SF-37` are `done` on the default branch.
- SF-37 byte-compare tooling and base references exist:
  `docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification/scripts/{run_sf37.sh,json_strip_cmp.py,lines_cmp.py,sf37_base_refs.sh}`.
- The SF-36 memory sampler exists:
  `docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core/scripts/run_sf36.sh:89-97`.
- The decision record `docs/decisions/2026-10-22-lean-lagrangian-stage.md` and the audit
  `docs/experiments/2026-10-22-lagrangian-memory-audit.md` are on the default branch.
- The base commit is recorded in the activation bitácora row; base binaries for the byte comparisons are built from it.

## In scope

Principle: everything additive and opt-in. `build_production_stage`, `solve_affine_periodic_flow`, the multigrid
(`src/multigrid`), the preconditioner, operators, blas, SF-18 and SF-28 stay intact (existing bodies unchanged). The
only shared numerical code touched is a **new variant** of `projected_pcg_solve` in `src/numerics/solvers/pcg.cuh`
(existing body untouched). The new path is selected by `inlet_slab --track forward --stage lean [--grid nx,ny,nz]`.

Bitwise rules (normative):

1. The lean SF-19 lives in the same translation unit `src/physics/flow/AffinePeriodicFlowSolver.cu` (its kernels and
   reductions are in an anonymous namespace, `:19`, `:262`), so the same compiled kernels are used (same FMA
   contraction, same reduction order).
2. `exp` and `eps*Y` stay on the host, as today.
3. Every reused buffer starts at an address returned by `cudaMalloc` or `cudaHostAlloc` (never a sub-offset), because
   CUB/cuBLAS reduction paths may depend on alignment.
4. Copies on the legacy stream: `cudaDeviceSynchronize` after an upload; `ctx.synchronize()` before overwriting a host
   staging buffer.
5. `h_tilde` is recombined on device with the same `blas::axpy` sequence (memset 0, then `axpy(G_0, w_0)`,
   `axpy(G_1, w_1)`, `axpy(G_2, w_2)`), uploading one offloaded corrector at a time.
6. No unified (managed) memory.
7. Reused n-sized buffers that receive CompactMAC face arrays (`(nx+1) ny nz`, `nx (ny+1) nz`, `nx ny (nz+1)` elements,
   `AffinePeriodicFlowSolver.cu:25-36`) are allocated with capacity >= the CompactMAC size (grow-only `DeviceBuffer`;
   +1/nx).
8. `--save-vd`, `v_rms` and the v1 face-vs-spline diagnostic are rejected or printed `n/a` explicitly; no fallback to
   the full stage.

Phases (each frees what it no longer uses; the host has 125 GiB):

- **F:** SF-18 produces `Y`, downloaded to host; no spectral vertex evaluation.
- **S1:** `lnK = eps*Y` and `K = exp(lnK)` on host (as today); prefilter `lnK` into `coef_logk`, downloaded to host;
  the SF-18 spectrum is released.
- **S2:** lean SF-19 at level S+:
  - one corrector buffer on device; each `w_d` downloaded after its `K_eff` column;
  - `rhs_b` (discarded scratch) and `rhs_a` on dead buffers allowed by the lifetimes (`rhs_a` is read only before the
    first `M.apply`; the guard `pcg.cuh:291-296` forbids it inside the PCG workspace, so the variant defines its own
    guard);
  - K loaded directly into the level-0 MG coefficient, coarse levels built by the public
    `coarsen_coefficient_geometric` loop (`src/multigrid/coefficient_hierarchy.cu:86-107`; no change in
    `src/multigrid`);
  - external-vector PCG with `z = Ap = b0` (MG level-0 `b`; `z` and `Ap` are never live together and `b0` is dead
    outside `M.apply`);
  - `b_hat` in pinned, mapped host memory;
  - final velocity, divergence and inlet U plane on dead buffers; optional faces downloaded to host for the Pollock
    comparator.
- **S3:** K/PCG/MG released; prefilter `h_tilde` into `coef_pot`; `InletLabels::prepare_device` deferred to here;
  `coef_logk` uploaded; both coefficient arrays handed to the tracker on device.
- **T:** tracker (16 B/cell + ~650 B/particle).

DAG suggestion for the future run (the SF-40 orchestrator may refine it; V100 nodes are serialized; L1, L5, L6 and L7
can start in parallel):

- **L1** external-vector variant of `projected_pcg_solve` (`z == Ap` allowed, external/host-mapped `b_hat`) and a
  byte-for-byte ctest against the workspace version (pinned mapped memory covered in the same test).
- **L2** `solve_affine_periodic_flow_lean` (additive in `AffinePeriodicFlowSolver.{cuh,cu}`) with an exact per-phase
  byte report; ctest: `h_tilde`, `G`, `K_eff`, corrector results and `u/v/w` byte-identical to the current solver at
  32^3, 64^3 and 64x16x16. Depends on L1.
- **L3** lean `Grid3D` stage `src/physics/particles/lagrangian_labels/LongBoxStage.{cuh,cu}` (`LongBoxSpec`,
  `LongBoxStage`, `build_long_box_stage`; names of SF-38) with field sources (SF-18 by seed, the analytic controls, and
  a caller-provided host `Y` array for the tiling control), automatic MG levels by the minimum dimension (an additive
  function equal to `slab_auto_mg_levels(N)` on cubes), and a tracker API accepting device coefficients and a non-cubic
  grid (existing API intact). Depends on L2.
- **L4** driver `--stage lean` and `--grid nx,ny,nz` (`apps/inlet_slab/inlet_slab_driver.cuh`, only with
  `--track forward`), explicit rejections. Depends on L3.
- **L5** campaign script `docs/experiments/artifacts/<date>-sf40-lean-lagrangian-stage/scripts/run_sf40.sh` with parts
  `bytecmp`, `mem`, `controls`, `par2`, `l6`; reuses the SF-37 byte-compare scripts and base references; samples memory
  by PID.
- **L6** tracker speedups: fused evaluation of the two splines (interleaved `double2` coefficients, same accumulation
  order per spline, `--fmad=false`, `eval`/`eval_ext` contract intact, no change in `src/numerics/interpolation`);
  plane and time passes in a single launch as an opt-in (existing `timings_s` fields keep their meaning).
- **L7** Par2: remove `FlowWorkspace::{residual, aux1, aux2}` (`src/physics/common/workspaces.cuh`).
- Integration, then V100 jobs, then the experiment note, then the final audit.

## Out of scope

- Non-bitwise techniques (mixed precision, single corrector with imposed gradient, host flow solve, multiple GPUs):
  they need a new owner decision.
- Medium (b) (Dirichlet in `x`) and the non-periodic spline (SF-41).
- The `read_timeseries_csv` header bug (audit §7): SF-38 or a separate bugfix PR.
- Wiring into `EnsembleRunner` (SF-38).
- The SF-38 long-box controls beyond criterion F, the medium A/B, the runner and the Par2 baseline configs (SF-38).
- `int32` limits of frozen kernels (recorded; 4.9e8 < 2^31).
- Any change of default outputs.

## Files and symbols

- Extended (additions only): `src/numerics/solvers/pcg.cuh` (external-vector `projected_pcg_solve` variant),
  `src/physics/flow/AffinePeriodicFlowSolver.{cuh,cu}` (`solve_affine_periodic_flow_lean`),
  `src/physics/particles/lagrangian_labels/` tracker API (device coefficients, non-cubic grid, fused spline evaluation,
  single-launch plane/time passes), `apps/inlet_slab/inlet_slab_driver.cuh` (`--stage lean`, `--grid`),
  `src/physics/common/workspaces.cuh` (remove `FlowWorkspace::{residual, aux1, aux2}`; L7).
- New: `src/physics/particles/lagrangian_labels/LongBoxStage.{cuh,cu}` (`LongBoxSpec`, `LongBoxStage`,
  `build_long_box_stage`); an additive automatic-MG-levels function by minimum dimension (equal to
  `slab_auto_mg_levels(N)`, `src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cu:269-277`, on cubes).
- Tests (directory names verified against `ls tests/`; the SF-40 orchestrator may rename the tests in the activation
  row): `tests/streamfunctions/` next to `projected_pcg_gpu_cases.cu` (PCG variant, `projected_pcg_external`,
  `<= 60 s`); `tests/flow/` next to `affine_periodic_flow_tests.cu` (lean SF-19, `affine_periodic_flow_lean`,
  `<= 60 s`); `tests/lagrangian_labels/long_box_tests.cu` (`lagrangian_long_box`, `<= 60 s`; extended, not created, by
  SF-38).
- Campaign script and artifact: `docs/experiments/artifacts/<date>-sf40-lean-lagrangian-stage/scripts/run_sf40.sh`;
  experiment note `docs/experiments/<date>-sf40-lean-lagrangian-stage.md`.
- Read-only: `build_production_stage`, `solve_affine_periodic_flow`, `src/multigrid/**`,
  `src/numerics/{blas,operators,constraints,interpolation}`, `src/physics/stochastic`, SF-18, SF-28.

## Implementation specification

1. Bitwise rules 1-8 and phases F, S1, S2, S3, T as in "In scope" are normative.
2. Numerical conventions: unchanged. Same kernels, same reduction order, same host `exp`; the lean path is a
   re-arrangement of lifetimes and placement, never of arithmetic.
3. Ownership: `LongBoxStage` owns the device coefficient arrays and passes them to the tracker; the host owns `Y`,
   `coef_logk`, offloaded correctors and (S+) the pinned `b_hat`.
4. Logging: one line per phase
   `MEM_PHASE phase=<F|S1|S2|S3|T> exact_bytes=<n> cufft_work=<n>` (cuFFT work areas from `cufftGetSize`).
5. Failure behavior: rejected options (`--save-vd`, the v1 diagnostic, `--stage lean` without `--track forward`) exit
   non-zero with a message naming the option; an allocation failure exits non-zero naming the phase; no silent
   fallback to the full stage.
6. A technique that breaks bytes is removed, not tuned.

## Expected numerical effect

None: byte-identical outputs on the default and lean paths. Expected memory: S+ 60.57 B/cell at the SF-19 PCG peak plus
context, i.e. `8 x [7 n + 4 sum_{l>=1} n_l]` bytes plus the enumerated fixed-size scratch (level-0 coefficient = K,
level-0 `x`, `b` (= `z` = `Ap`), `r`, PCG `r`, `p`, active corrector, coarse levels `x,b,r,coefficient`); FFT phases
<= ~40 B/cell; tracking 16 B/cell + ~650 B/particle. At 3328x384x384 (4.9e8 cells) the estimate is 28847 MiB including
500 MiB of context (audit §8-§9). Expected time: S+ adds ~8 s per realization at 4.9e8 cells (estimate); PCG iterations
may grow on the 52x6x6 coarsest level (`slab_auto_mg_levels` on the minimum dimension 384 gives 7 levels; 50 GSRB
sweeps on a 52-cell-long coarse grid); the tracker speedup is not predicted (recorded).

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j 6
ctest --test-dir build/wsl-debug --output-on-failure -R 'projected_pcg_external|affine_periodic_flow_lean|lagrangian_long_box'
scripts/remote --increment SF-40 sync
scripts/remote --increment SF-40 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j 2>&1 | tail -n 20; echo BUILD_EXIT=${PIPESTATUS[0]}"
A=docs/experiments/artifacts/<date>-sf40-lean-lagrangian-stage/scripts
for part in bytecmp l6 controls mem par2; do scripts/remote --increment SF-40 run sf40-$part -- "bash $A/run_sf40.sh build/v100-release $part"; scripts/remote --increment SF-40 wait sf40-$part; done
scripts/remote --increment SF-40 run sf40-ctest-full-bytecmp -- "ctest --test-dir build/v100-release --output-on-failure; <SF-27 trio byte-compare vs base refs>; <SF-35/36/37 default-path byte-compare vs base refs>"
scripts/remote --increment SF-40 wait sf40-ctest-full-bytecmp
bash scripts/hooks/check-lester-increments.sh
```

All remote jobs are detached and serialized (`docs/runbooks/remote-v100.md` §0); the local cases are fast `-R`-filtered
tests only. The test names above may be renamed by the SF-40 orchestrator in the activation row.

## Acceptance thresholds

Pre-registered verbatim in the activation bitácora row before any run.

- **A. Bytes.** `--stage lean` (with L6) against the base binary (build of the activation base commit), every one of:
  every `.npy`; `moments_planes.csv`; `moments_time.csv`; `track.json` without `timings_s`; the
  `FIELD`/`TRACK`/`BACKFLOW`/`TPLANE`/`TTIME`/`TSTATUS`/`STATUS` lines; the `SETUP` lines of SF-18, SF-19 and the three
  correctors (`SlabProductionSetup.cu:396-437`; the `STAGE`, `inlet` and `v1(U-face)` lines are excluded because
  `v_rms` and the face diagnostic are `n/a` in the lean path); and the exit code. For the Pollock case also
  `pollock.json` without timings and the Pollock moment CSVs. Cases: `N` 64/128/256 x `eps` 1/1.5/2 x seeds 3101-3103;
  `N` 512 x `eps` 1/2, seed 3102; one case with `--covectors` and one with `--comparator pollock`; the same set for L6
  alone on the current stage. Every comparison exactly equal.
- **B. Tests.** The L1 and L2 ctests are byte-for-byte green (local fast and V100 full).
- **C. Exact bytes.** The per-phase exact-byte report equals the S+ closed form (formula and enumerated scratch in the
  experiment note) with difference 0 bytes, at 512^3 and at 3328x384x384.
- **D. Measured memory.** nvidia-smi by PID at 200 ms; the CUDA context is measured separately (process creating the
  context only). At 512^3 (`eps` 2, seed 3102): measured peak <= exact bytes of the largest phase + context + 3 %. At
  3328x384x384 (field + flow + splines + 1e4 particles injected on `x1 = 0` and traced over the full length, planes at
  `2^m ell`, `m = 0, 1, ...`, and at `L1 = 3328 h`): measured peak <= 30000 MiB and exit 0 in two runs: `sigma^2` 2.25
  with `ell = 16 h`, seed 4001; `sigma^2` 4 with `ell = 32 h`, seed 4001.
- **E. Convergence.** The lean SF-19 converges to `rtol = 1e-10` in both D runs with the automatic hierarchy (coarsest
  52x6x6); iterations and times recorded.
- **F. Non-cubic (exact controls, no bitwise reference).**
  - F1: `k = 1` on 64x16x16 and 128x16x16 reproduces SF-36 A1 (same thresholds), and the lean SF-19 gives `K_eff = I`
    and mean flux `e1` to `<= 1e-12`.
  - F2: layered `k = exp(0.5 cos 2 pi x1)` on 64x32x32 and 128x32x32 (`h = 1/32`, the SF-36 owner-amended A2 rule
    `N >= 32`) reproduces the amended SF-36 A2: transverse positions and covectors constant to `<= 1e-12`; `tau`
    `<= 1e-9` relative against the 1-D quadrature; `I` recorded, no threshold.
  - F3: a particle shifted by `+1` in `x2` or `x3` sees the same `v_s` to `<= 1e-14`.
  - F4 (tiling): the 64^3 cube field of SF-18 (seed 3101, `eps` 1) repeated twice in `x1` as a 128x64x64 host `Y`
    array, both runs at `--pcg-rtol 1e-12`: `G` and `K_eff` equal to the cube's to `<= 1e-8` relative, and plane
    statistics (`sigma_T^2`, mean and var `tau`) at planes {0.25, 0.5, 1, 2, 4} equal to `<= 1e-8` relative.
- **G. Par2.** `apps/config_pipeline_par2.yaml` outputs and the SF-27 trio byte-identical to the base; `FlowWorkspace`
  allocation lower by exactly 24 B/cell (byte report).
- **R. Regression.** Full V100 `ctest` green; SF-27 trio clean; SF-35/36/37 default-path byte-compare clean;
  `git diff <base>..<head>` empty in `src/multigrid`, `src/numerics/{blas,operators,constraints,interpolation}` and
  `src/physics/stochastic`; the bodies of `build_production_stage`, `solve_affine_periodic_flow` and
  `projected_pcg_solve` unchanged (additions only in their files).
- **Recorded (no threshold).** Time of S vs S+, L6 speedup per pass, host RSS, maximum cells measured, the feasibility
  table per `sigma^2` and `lambda/h`.

## Regression surface

- SF-35/36/37 default cube path (shared module, driver).
- SF-19/PCG callers (the frozen stack uses the existing `projected_pcg_solve`).
- Par2 pipeline and its configs.
- `ctest` wall time.

## Failure and rollback policy

- A, B, C, F, G or R not met -> BLOCKING defect (corrective DAG).
- D or E not met at 3328x384x384 -> recorded with the measured peak/iterations; the domain of SF-38 is decided by the
  owner (stop rule R3 of `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`); SF-40 may close with the measured
  capacity if the owner accepts.
- A technique that breaks bytes is removed (not tuned); "almost identical" is not accepted; if no L6 technique is
  bitwise, the speedup is recorded as not achieved (owner).

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Implementation matches the scope and contains no unrelated changes.
- [ ] Targeted validation passes and its evidence is recorded.
- [ ] Required regression tests pass.
- [ ] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-38` becomes eligible after this increment is merged and marked `done` on the default branch (SF-38 also depends on
`SF-36`, done).

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-22T15:19Z | not started | Specification created on the governance PR (branch docs/lester-sf40-lean-stage-spec) after the owner decisions of 2026-10-22 (`docs/decisions/2026-10-22-lean-lagrangian-stage.md`; audit `docs/experiments/2026-10-22-lagrangian-memory-audit.md`). Goal memory bound stated as measured peak <= 30000 MiB (the planning text said "32GiB"; 30000 MiB is the owner's decision value and criterion D). | Depends on SF-37 done on master; SF-38 now depends on SF-40. | Activate only when the checker reports it READY; the activation row contains thresholds A-G and R verbatim and the base commit before any run. |
