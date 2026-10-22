# SF-40 — Lean parametric Lagrangian stage (Grid3D, memory level S+), bitwise tracker speedups and Par2 dead-buffer cleanup

- State: `done`
- Goal: `Bajar el pico de memoria de dispositivo del camino lagrangiano (campo SF-18, flujo SF-19, splines SF-28, tracker) con un stage liviano paramétrico en (nx, ny, nz) para que la caja de 3328x384x384 celdas (4.9e8; hoy caben 1.5e8) corra en una V100-32GB con pico medido <= 30000 MiB, con salidas del tracker idénticas byte a byte al camino actual, speedups bit a bit del tracker y sin buffers muertos en el baseline Par2.`
- Depends on: `SF-37`
- Unlocks: `SF-38`
- Branch: `science/lester-sf40-lean-lagrangian-stage`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 (change type B: no intended numerical change; invariance target = byte identity)`
- Human review: `required`
- Owner: `Claude Opus 5.5 orchestrator (session 2026-10-22)`
- Started: `2026-10-22T17:17Z on base 9485beb`
- Completed: `2026-10-22` (owner approval of PR #58 in the orchestrator session, 2026-10-22: "Apruebo el pr, hacé el cierre"; closure by the metadata-only commit on the same PR)
- PR: `https://github.com/santi-esquerre/MacroFlow3D/pull/58` (opened 2026-10-22 from `science/lester-sf40-lean-lagrangian-stage`, FINAL_AUDIT head `1eca1d1`; approved head `061901c`; closure metadata commit on the same PR; merge by the owner)
- Commit: `bc089455efc5620bba63412dbe2c5400d4ac18a6` (source-bearing audited head H1 = integration I1); docs/evidence/metadata only above it: N8 `1110b3e`, FINAL_AUDIT `1eca1d1`, PR record `061901c` (approved PR head `061901c613db815a72c2f7a7e9b45e42747c6273`), then the closure metadata commit

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

Phases (each frees what it no longer uses; the V100 host has 125.35 GiB of RAM, audit `raw/host_memory_pcie.txt`):

- **F:** SF-18 produces `Y`, downloaded to host; no spectral vertex evaluation.
- **S1:** `lnK = eps*Y` and `K = exp(lnK)` on host (as today); prefilter `lnK` into `coef_logk`, downloaded to host;
  the SF-18 spectrum is released.
- **S2:** lean SF-19 at level S+:
  - one corrector buffer on device; each `w_d` downloaded after its `K_eff` column;
  - `rhs_b` (discarded scratch) and `rhs_a` on dead buffers allowed by the lifetimes (`rhs_a` is read only before the
    first `M.apply`; the guard `pcg.cuh:291-296` forbids it inside the PCG workspace, so the variant defines its own
    guard); `rhs_a`, `rhs_b` and `K` must also be pairwise non-overlapping (assembler guard
    `src/physics/streamfunctions/affine_periodic_rhs.cu:79-86`);
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
  plane and time passes in a single launch as an opt-in: each thread runs the plane pass and then the
  time pass with their own unchanged step sequences (the two integrations are not merged: their landing rules
  differ, audit §6); existing `timings_s` fields keep their meaning.
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
   `MEM_PHASE phase=<F|S1|S2|S3|T> exact_bytes=<n> cufft_work=<n>` (cuFFT work areas from `cufftGetSize`); the `MEM_PHASE` lines also print the categories of
   `AffinePeriodicFlowWorkspace::memory_report()` and the cuFFT work areas of the generator and of both prefilters (audit
   §5: these exact reports exist and are not printed today).
5. Failure behavior: rejected options (`--save-vd`, the v1 diagnostic, `--stage lean` without `--track forward`) exit
   non-zero with a message naming the option; an allocation failure exits non-zero naming the phase; no silent
   fallback to the full stage.
6. A technique that breaks bytes is removed, not tuned.

## Expected numerical effect

None: byte-identical outputs on the default and lean paths. Expected memory: S+ 60.57 B/cell at the SF-19 PCG peak plus
context, i.e. `8 x [7 n + 4 sum_{l>=1} n_l]` bytes — the seven n-sized arrays are the level-0 coefficient (= K), level-0 `x`, `b` (= `z` = `Ap`), `r`, PCG
`r`, `p` and the active corrector; the coarse levels hold `x, b, r, coefficient` — plus enumerated fixed-size
scratch (reduction blocks, mean-zero workspaces, PCG reduction scalar); FFT phases
<= ~48 B/cell (lean second prefilter 48.09 B/cell, audit §9); tracking 16 B/cell + ~650 B/particle. At 3328x384x384 (4.9e8 cells) the estimate is 28847 MiB including
500 MiB of context (audit §8-§9); the audit measured an unmodeled overhead of 485 / 622 / 748 MiB growing with size
(~1129 MiB extrapolated at 4.9e8 cells), so the margin to 30000 MiB is 0.5-1.1 GiB (audit §4, §9). Host: the current
path's host RSS (197.0 B/cell) would be ~90 GiB of the 125.35 GiB host at 4.9e8 cells; the lean path's host arrays
(`Y`, `K`, `coef_logk`, offloaded correctors, `b_hat`, optional faces) are enumerated in the experiment note and the
host RSS is recorded. Expected time: S+ adds ~8 s per realization at 4.9e8 cells (estimate); PCG iterations
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
- **D. Measured memory.** nvidia-smi by PID at 200 ms; the CUDA context is measured separately by a probe process that creates the
  `CudaContext` as the stage does and allocates nothing else. At 512^3 (`eps` 2, seed 3102): measured peak <= exact bytes of the largest phase + context + 3 %. At
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
- D not met at 512^3 -> the excess over exact bytes + context is itemized in the experiment note; an allocation of the
  lean path missing from the exact-byte report is a C defect (BLOCKING); an excess not attributable to any allocation
  of the code (driver, library or allocator overhead; the audit measured 485-748 MiB of unmodeled overhead, §4) is
  recorded and the owner decides.
- A technique that breaks bytes is removed (not tuned); "almost identical" is not accepted; if no L6 technique is
  bitwise, the speedup is recorded as not achieved (owner).

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [x] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
- [x] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
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
| 2026-10-22T17:17Z | activation on `master=9485beb` (PR #57 merged; SF-37 `done`; checker OK ready=SF-34 SF-40, nonterminal=none); delivery branch `science/lester-sf40-lean-lagrangian-stage`; remote `scripts/remote --increment SF-40` (one mirror, one job at a time), outputs under `~/sf40_campaign`, base references under `~/sf40_base_refs` on the V100 host | UNDERSTAND (runtime record `.claude/orchestration/SF-40-lean-lagrangian-stage/understanding.md`; scientific-rigor category C: implementation, numerical contract = byte identity; contracts verified in code: `pcg.cuh:282-460` guard and z/Ap lifetimes, `projected_positive_mg_preconditioner.cu:19-34`, `coefficient_hierarchy.cu:86-108`, `AffinePeriodicFlowSolver.cu` anonymous-namespace kernels and CompactMAC sizes, `affine_periodic_rhs.cu:79-86`, `SlabProductionSetup.cu:269-277,451-690`, `inlet_slab_driver.cuh:2620-2823`, `LagrangianTracker*`; plan reviewed adversarially by three independent read-only reviewers - byte identity, feasibility, compliance - before this row, findings incorporated). **Pre-registration before any run.** Acceptance thresholds, verbatim from this file: - **A. Bytes.** `--stage lean` (with L6) against the base binary (build of the activation base commit), every one of: every `.npy`; `moments_planes.csv`; `moments_time.csv`; `track.json` without `timings_s`; the `FIELD`/`TRACK`/`BACKFLOW`/`TPLANE`/`TTIME`/`TSTATUS`/`STATUS` lines; the `SETUP` lines of SF-18, SF-19 and the three correctors (`SlabProductionSetup.cu:396-437`; the `STAGE`, `inlet` and `v1(U-face)` lines are excluded because `v_rms` and the face diagnostic are `n/a` in the lean path); and the exit code. For the Pollock case also `pollock.json` without timings and the Pollock moment CSVs. Cases: `N` 64/128/256 x `eps` 1/1.5/2 x seeds 3101-3103; `N` 512 x `eps` 1/2, seed 3102; one case with `--covectors` and one with `--comparator pollock`; the same set for L6 alone on the current stage. Every comparison exactly equal. - **B. Tests.** The L1 and L2 ctests are byte-for-byte green (local fast and V100 full). - **C. Exact bytes.** The per-phase exact-byte report equals the S+ closed form (formula and enumerated scratch in the experiment note) with difference 0 bytes, at 512^3 and at 3328x384x384. - **D. Measured memory.** nvidia-smi by PID at 200 ms; the CUDA context is measured separately by a probe process that creates the `CudaContext` as the stage does and allocates nothing else. At 512^3 (`eps` 2, seed 3102): measured peak <= exact bytes of the largest phase + context + 3 %. At 3328x384x384 (field + flow + splines + 1e4 particles injected on `x1 = 0` and traced over the full length, planes at `2^m ell`, `m = 0, 1, ...`, and at `L1 = 3328 h`): measured peak <= 30000 MiB and exit 0 in two runs: `sigma^2` 2.25 with `ell = 16 h`, seed 4001; `sigma^2` 4 with `ell = 32 h`, seed 4001. - **E. Convergence.** The lean SF-19 converges to `rtol = 1e-10` in both D runs with the automatic hierarchy (coarsest 52x6x6); iterations and times recorded. - **F. Non-cubic (exact controls, no bitwise reference).** - F1: `k = 1` on 64x16x16 and 128x16x16 reproduces SF-36 A1 (same thresholds), and the lean SF-19 gives `K_eff = I` and mean flux `e1` to `<= 1e-12`. - F2: layered `k = exp(0.5 cos 2 pi x1)` on 64x32x32 and 128x32x32 (`h = 1/32`, the SF-36 owner-amended A2 rule `N >= 32`) reproduces the amended SF-36 A2: transverse positions and covectors constant to `<= 1e-12`; `tau` `<= 1e-9` relative against the 1-D quadrature; `I` recorded, no threshold. - F3: a particle shifted by `+1` in `x2` or `x3` sees the same `v_s` to `<= 1e-14`. - F4 (tiling): the 64^3 cube field of SF-18 (seed 3101, `eps` 1) repeated twice in `x1` as a 128x64x64 host `Y` array, both runs at `--pcg-rtol 1e-12`: `G` and `K_eff` equal to the cube's to `<= 1e-8` relative, and plane statistics (`sigma_T^2`, mean and var `tau`) at planes {0.25, 0.5, 1, 2, 4} equal to `<= 1e-8` relative. - **G. Par2.** `apps/config_pipeline_par2.yaml` outputs and the SF-27 trio byte-identical to the base; `FlowWorkspace` allocation lower by exactly 24 B/cell (byte report). - **R. Regression.** Full V100 `ctest` green; SF-27 trio clean; SF-35/36/37 default-path byte-compare clean; `git diff <base>..<head>` empty in `src/multigrid`, `src/numerics/{blas,operators,constraints,interpolation}` and `src/physics/stochastic`; the bodies of `build_production_stage`, `solve_affine_periodic_flow` and `projected_pcg_solve` unchanged (additions only in their files). - **Recorded (no threshold).** Time of S vs S+, L6 speedup per pass, host RSS, maximum cells measured, the feasibility table per `sigma^2` and `lambda/h`. Reading rules RR0-RR12 and implementation contracts C-LEAN-PLACEMENT, C-SYNC, C-MEMPHASE, C-TRACKER-L6: `docs/experiments/artifacts/2026-10-22-sf40-lean-lagrangian-stage/preregistration.md` (committed in this activation commit, blob `b79276174dfc6afc26568257afb5f2c310258c4b`), unchanged afterwards. Restated key readings (no threshold changed): S+ closed form at the S2 (PCG) peak = `8 [7n + 4 sum_{l>=1} n_l] + 8 * 4 * (m - n) + S_scr` (rule-7 capacity m of PCG r, PCG p, corrector, MG level-0 x; pre-registered scratch list), C checked on every MEM_PHASE line with a cudaMemGetInfo completeness instrument; A cases = SF-37 V2 setting (`--np-side 256`, planes and times 0.25..16, `--tol 1e-8 --hmax-div 8 --max-arclength 1000`), N {64, 128, 256} x eps {1, 1.5, 2} x seeds 3101-3103, N 512 x eps {1, 2} seed 3102, `cov` and `pol` cases, sets LEAN (`--stage lean --fused-splines --single-launch`) and L6 (`--fused-splines --single-launch`) vs the base binary, every particles file, json without `**.timings_s` only, the listed lines; D: PID sampler 200 ms, context = minimum of three probes, D(512) iff peak `<= E_max * 1.03 + C`, D(3328) runs seed 4001, `--np-side 100 --max-arclength 1000`, (i) eps 1.5 ell 16 h, (ii) eps 2 ell 32 h, planes `2^m ell <= L1` and `L1`; G not met if the base itself is not deterministic (owner may waive); the lean path rejects `--analytic`, `--save-vd`, `--save-seed-velocity`; `ny == nz`, `h = 1/ny` (disclosed limitation); B includes `--full` V100 modes of the L1/L2 executables. | DAG (`.claude/orchestration/SF-40-lean-lagrangian-stage/dag.json`): L1 PCG external-vector variant, L5 campaign tooling, L6 tracker speedups, L7 Par2 dead buffers (parallel) -> L2 lean SF-19 (after L1) -> L3 `LongBoxStage` + tracker device API (after L2, L6) -> L4 driver -> I1 integration -> V100 jobs (orchestrator; J0 `sf40-base-refs` after L5; at H1 `sf40-build`, `sf40-dry`, `sf40-controls`, `sf40-bytecmp`, `sf40-l6`, `sf40-mem`, `sf40-par2`, `sf40-final`) -> N8 evidence + experiment note -> I2 -> FINAL_AUDIT -> `awaiting_review` + PR. Workers and the integrator run with the agent definitions present in the control checkout (sonnet / high, set locally by the owner). Persistent-goal feature not available in this runtime. | Launch L1, L5, L6, L7; J0 after L5 is accepted. |
| 2026-10-22T17:47Z | `active`; delivery branch at A0 `e256b06` + this row; no V100 run yet | Accepted after orchestrator audit (records `.claude/orchestration/SF-40-lean-lagrangian-stage/audits/`): L7 `82ac877` (FlowWorkspace residual/aux1/aux2 removed, ctest `flow_workspace_bytes` asserts 8n + sum_l 32 n_l), L1 `3313bdd` (`projected_pcg_solve_external`: pcg.cuh 244 added / 0 removed lines, body a statement-for-statement transcription of `projected_pcg_solve` except the guard and the vector sources; ctest `projected_pcg_external` bitwise vs the workspace version with z = Ap = MG level-0 b, b in MG level-0 x, b_hat on device and in pinned mapped host memory), L5 `0adc407` (campaign tooling `run_sf40.sh` parts baserefs/bytecmp/l6/controls/mem/par2/final implementing RR4-RR11, `closed_form.py`, `pid_peak.py`, `mem_verdict.py`, `setup_lines_cmp.py`; SF-37 compare tools copied byte-identically). L2 running (on L1), L6 running. **RR7 clarification (before any run; no threshold changed):** the manifest key list dropped in the Par2/trio comparisons is `timestamp` (wall-clock time of the run) and `build.git_hash` (configure-time commit of the build, build metadata that differs between a base and a head build by construction; on the V100 mirror `.git` is not synced, so both builds record the same value anyway); both values are recorded. Process (user constraint 2026-10-22): local builds only through a machine-wide lock (`mf3d-build`: one build at a time, 6 jobs on 6 cores, nice 15), enforced by a PreToolUse hook that denies direct builds and killing build processes; before the hook existed the L1 worker killed `ninja` processes machine-wide (possible interruption of other agents' local builds; no evidence affected). | J0 `sf40-base-refs`: mirror `~/MacroFlow3D-SF-40` synced from a clean detached worktree at `0adc407` (tree differs from `9485beb` only under `docs/`, checked by `git diff --quiet 9485beb 0adc407 -- . ':(exclude)docs'` and equal submodule gitlinks), so its build is the base build; first a dry run (`SF40_DRYRUN=1`, separate root, NOT evidence), then `run_sf40.sh ~/sf40_base_refs/bin baserefs`. | J0a build + dry baserefs, J0b real baserefs; audits of L2, L6 as they finish. |
| 2026-10-22T22:35Z | H1 `bc08945` (integration I1); jobs J0a, J0b, J1a, J1b, J1c finished exit 0; evidence imported by N8 `1110b3e` | Accepted after orchestrator audit (records `.claude/orchestration/SF-40-lean-lagrangian-stage/audits/`): L6 `3110367` (fused two-spline evaluation over one interleaved `double2` coefficient array and opt-in single-launch plane+time kernel, `--fmad=false`; ctest `lagrangian_tracker_l6`), L2 `663e111` (`solve_affine_periodic_flow_lean` at levels S/S+, additions only in `AffinePeriodicFlowSolver.{cuh,cu}`; ctest `affine_periodic_flow_lean` bitwise against the current solver), L3 `cea7c8c` (`LongBoxStage`, `box_auto_mg_levels`, tracker device-coefficient API, `MEM_GEOM`/`MEM_PHASE` lines; ctest `lagrangian_long_box` with the F1-F4 controls), L4 `a8bed18` (driver `--stage lean`, `--grid`, `--bhat`, `--context-probe [--warm]`, explicit rejections; the default `--track forward` code was refactored into a shared tail, checked by the R byte-compare); correctives C1 `c5348f2` (`run_sf40.sh` split the multi-word RR4 extra field into argv words; bug found by the J0a dry run, before any real run) and C2 `7833ecb` (`closed_form.py` accepts the empty F phase of a cells source). I1 (one integrator, chain over `ae769d1`: L1, L2, L6, L3, L4, L7 with a CMakeLists textual conflict resolved by keeping all test blocks, L5, C1, C2; patch-ids checked) -> H1 `bc089455efc5620bba63412dbe2c5400d4ac18a6`. V100 jobs (mirror `~/MacroFlow3D-SF-40`, GPU 0, one at a time): J0a `sf40-j0a-build-dry` 17:50-17:54Z (base build from a tree equal to `9485beb` outside `docs/`, `inlet_slab` md5 `ac806f1b1d8d5076eac4ad97e6f4ffc7`); J0b `sf40-j0b-base-refs` 17:58-18:55Z (base references: 31 A cases, 20 R runs, SF-27 trio, Par2 twice, deterministic); J1a `sf40-build` 19:27-19:30Z (H1 build, 43 ctests, head `inlet_slab` md5 `95febc0b40245fac1f4b19af12d569bd`, SASS check); J1b `sf40-dry` 19:31-19:38Z (not evidence); J1c `sf40-campaign` 19:39-22:11Z (parts controls, bytecmp, l6, mem, par2, final). Local incident during I1: `/home` reached 100 % and only the build directories of audited SF-40 node worktrees were removed. | Evidence: text subset of the V100 outputs (1469 files, tarball sha256 `b6b1f0a6b9b4b6df07332dec141daaa4ff9eb5c052f3bad5fd3a290f59c879f2`, imported byte-identically) in `docs/experiments/artifacts/2026-10-22-sf40-lean-lagrangian-stage/raw/`; experiment note `docs/experiments/2026-10-22-sf40-lean-lagrangian-stage.md`. **A met** (1317 IDENTICAL, 0 DIFF, 0 MISSING: LEAN 31 cases, L6 31 cases, R 20 runs). **B met** (L1/L2 ctests local, `--full` on V100, full V100 ctest 43/43 in 2968.36 s). **C met** (every `MEM_PHASE` difference 0 B; S2 equals the closed form, 8138281074 B at 512^3 and 29765452393 B = 60.655 B/cell at 3328x384x384; S2 is the largest phase at both sizes). **D met** (context 487 MiB; 512^3 PID peak 8249 MiB <= 8481.108; 3328x384x384 at sigma^2 2.25 and 4: PID peak 28885 MiB <= 30000, exit 0, margin 1115 MiB). **E met** (30 iterations per corrector, rel residual <= 8.9e-12, 7 levels, coarsest 52x6x6; 20 iterations at 512^3). **F met** (171 checks). **G met** (Par2 407 files + exit code and SF-27 trio identical; ctest `flow_workspace_bytes`; PID peak -3066 MiB, 3072 MiB expected). **R met** (full ctest, trio, R set, git-diff checks). Recorded: S vs S+ +2.31 s at 512^3; L6 speedups 1.515 (fused) / 1.046 (single launch) / 1.549 (both); host RSS 41.3 B/cell; max cells 490733568; feasibility table (the B&dD domain at lambda/h 16 needs ~32299 MiB and does not fit). Disclosed in the note: completeness-instrument sampling point for F/S1/S3 (after the cuFFT plans are destroyed), cuFFT peaks of S1/S3 not resolved by the 200 ms sampler, sigma^2 4 run with inlet backflow (labels not built, 3 seed-backflow particles), N8 trailer names the model actually used. | FINAL_AUDIT. |
| 2026-10-22T22:38Z | FINAL_AUDIT positive; State `awaiting_review`; source-bearing audited head H1 `bc08945`; integrated head H2 `1110b3e` (I2 = fast-forward `ae769d1` -> H1 -> N8, docs/evidence only above H1) | N8 accepted: the orchestrator checked all 1469 raw blobs `cmp`-identical to the tarball, ran `note_tables.py --selftest` (OK), regenerated the analysis byte-identically, read the note in full and checked the closed form, B/cell, D/E/F/G numbers, L6 ratios and feasibility arithmetic against raw. Final audit at H2: `git diff --quiet bc08945 1110b3e -- . ':(exclude)docs'`; `git diff 9485beb..bc08945` empty in `src/multigrid`, `src/numerics/{blas,operators,constraints,interpolation}`, `src/physics/stochastic`, `src/physics/streamfunctions`; `pcg.cuh` 244/0, `AffinePeriodicFlowSolver.cu` 358/0 and `.cuh` 85/0, with hunks only at include lines and after the existing bodies (`projected_pcg_solve` base lines 274-464, `solve_affine_periodic_flow` 742-881); checker OK. | Every criterion A-G and R is met by the pre-registered rules; the Recorded items are in the note. Human review is required: the human-review item and the master-checklist entry stay unchecked, and State is not `done`. | Push the delivery branch and open the PR; on explicit owner approval of the exact source-bearing head, add only the closure-metadata commit. |
| 2026-10-22T22:37Z | `awaiting_review`; PR #58 opened from `science/lester-sf40-lean-lagrangian-stage` (FINAL_AUDIT head `1eca1d1`; source-bearing audited head `bc08945`) | PR and commit identifiers recorded (metadata only; no source, test, configuration or evidence change; `git diff --quiet bc08945 1eca1d1 -- . ':(exclude)docs'` holds). | Human review of the source-bearing head `bc08945` with its docs/evidence head. The PR body lists the points for review: the completeness-instrument sampling point for F/S1/S3; the cuFFT peaks of S1/S3 not resolved at 200 ms; the sigma^2 4 run with inlet backflow (3 seed-backflow particles); 30 PCG iterations on the 52x6x6 coarsest grid; one realization per D size with a 1115 MiB margin; the interface limitations; the process disclosures. | On explicit owner approval: closure-metadata commit only (record the approval, check the human-review item, State `done`, master-checklist entry, checker). The owner merges; SF-38 becomes eligible only after the merge. |
| 2026-10-22T23:53Z | `done` (closure metadata, runbook §10); approved PR head `061901c` (source-bearing audited head `bc08945`; docs/evidence/metadata only above it) | **Human approval recorded truthfully:** the owner (repository owner, `santi-esquerre`) wrote to the orchestrator on 2026-10-22, after reading the PR and the orchestrator's explanation of it: "Apruebo el pr, hacé el cierre" (explicit instruction to finalize the reviewed PR, `docs/runbooks/autonomy-policy.md`, human approval semantics item 2); no GitHub review object exists. At that moment the PR head was `061901c613db815a72c2f7a7e9b45e42747c6273` (checked with `gh pr view 58`) and `git diff --quiet bc08945 061901c -- . ':(exclude)docs'` holds, so the approval covers the audited source-bearing head `bc08945` with its evidence, the experiment note and every disclosure listed in the PR (completeness-instrument sampling point for F/S1/S3; cuFFT peaks of S1/S3 not resolved at 200 ms; sigma^2 4 run with inlet backflow and 3 seed-backflow particles; 30 PCG iterations on the 52x6x6 coarsest grid; one realization per D size, 1115 MiB margin, ~140 MiB unattributed during tracking; interface limitations; process disclosures). **Timestamp disclosure:** the label 2026-10-22T22:38Z of the FINAL_AUDIT row is nominal, not a clock time (its commit `1eca1d1` is 22:35:50Z); the PR-record row 22:37Z matches its commit `061901c` (22:37:17Z) and the PR creation (22:37:02Z); content unchanged (append-only). Closure actions: State `done`, Completed, PR and Commit fields, checklist items 5 and 7, dashboard master-checklist entry and Later-phases item 4 status; no source, test, configuration, evidence or numerical change. | `bash scripts/hooks/check-lester-increments.sh` OK after the edit; `git diff --quiet bc08945 HEAD -- . ':(exclude)docs'` holds. | The owner merges PR #58. SF-38 becomes READY only when SF-40 is `done` on the default branch (SF-36 already is); its production geometry and `lambda/h` are fixed at its activation from SF-37 and the SF-40 feasibility table. |
