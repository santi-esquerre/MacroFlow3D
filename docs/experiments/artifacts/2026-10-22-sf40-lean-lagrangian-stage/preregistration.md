# SF-40 pre-registration: reading rules and implementation contracts (activation commit, before any run)

- Increment: `docs/plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md` (State `active` in the same
  commit). Base: `9485beb006a4b6bd82efc5c678dd10b9d1f4638e` (`master`). Written by the SF-40 orchestrator on
  2026-10-22 after UNDERSTAND and an adversarial review of the plan by three independent read-only reviewers
  (byte identity, feasibility, compliance); committed in the activation commit, before any build, test or run.
- What this file is: the durable text of the activation reading rules (RR0-RR12) and of the implementation contracts
  (C-*) that the activation bitácora row cites by blob. It changes no threshold, Goal, design value or failure-policy
  text of the specification; it resolves gaps inside the listed scope (SF-35/SF-36/SF-37 precedent). The acceptance
  thresholds A-G, R and "Recorded" are the specification text, copied verbatim into the activation row.
- No `alpha_L`/`alpha_T` and no `lambda/h` statement is made or allowed (Rule 1; SF-37 owns `lambda/h`).

## Reading rules

**RR0 (pre-registration, byte-breaking procedure).** No threshold is tuned after any run. A technique that breaks
bytes is removed (spec item 6); "almost identical" is not accepted. After a byte difference, corrective work may only
bring the code into conformance with bitwise rules 1-8 and the contracts below; changing operation order, kernels,
compile flags, precision or a contract to obtain equality is forbidden. If a contract-conformant implementation
still differs, the technique is removed (code path and its flag), the affected sets are re-run in full on the new
head, and the failed run stays in the bitácora (excluded from acceptance evidence, disclosed). Every V100 run is a
detached orchestrator job (`scripts/remote --increment SF-40 run`/`wait`, one job at a time) launched from a clean
worktree at an accepted commit and running committed scripts; workers never use the remote.

**RR1 (tests; B).** Fast ctests, each <= 60 s: `projected_pcg_external` (L1,
`tests/streamfunctions/projected_pcg_external_tests.cu`), `affine_periodic_flow_lean` (L2,
`tests/flow/affine_periodic_flow_lean_tests.cu`), `lagrangian_long_box` (L3, `tests/lagrangian_labels/long_box_tests.cu`),
`lagrangian_tracker_l6` (L6, new, disclosed) and `flow_workspace_bytes` (L7, new, disclosed). The L1, L2 and L3
executables also have a `--full` mode run on the V100 by the campaign part `controls` (L1/L2: 256^3 and 512x64x64,
where the CUB/cuBLAS paths of large n are exercised; L3: the F sizes). B is met iff the L1 and L2 ctests pass
byte-for-byte locally (fast) and in the full V100 ctest, and their `--full` modes pass on the V100.

**RR2 (CLI).** `inlet_slab --track forward --stage full|lean` (default `full`; `--stage lean` without
`--track forward` exits non-zero naming the option). `--grid nx,ny,nz` (lean only; `ny == nz`, all even, `>= 4`; box
`[0, nx h] x [0, 1]^2`, `h = 1.0/ny`, the SF-38 convention, because `InletLabels` takes one transverse N; the
restriction `ny == nz` is disclosed as a limitation). `--fused-splines`, `--single-launch` (L6, opt-in, both stages).
`--bhat mapped|device` (lean only, default `mapped` = S+; `device` = level S, used only for the recorded S vs S+ time
at 512^3; never at 3328x384x384, where S exceeds the card). `inlet_slab --context-probe SECONDS` (creates the
`CudaContext` exactly as the forward driver does, allocates nothing else, prints `CONTEXT_PROBE pid=<pid>
device=<id> driver=<v> runtime=<v> device_used=<bytes>`, sleeps, exits 0). Lean behaviour of every `--track forward`
option: `--sigma2/--ell/--seed` (SF-18), `--cells` (cube or box), `--eps`, `--pcg-rtol`, `--np-side`, `--x0`,
`--window`, `--planes`, `--times`, `--tol`, `--hmax-div`, `--min-step`, `--max-arclength`, `--covectors`,
`--save-particles`, `--summary`, `--comparator pollock` supported with the full stage's meaning; `--analytic`,
`--save-vd`, `--save-seed-velocity` rejected with a non-zero exit naming the option (rule 8; host stage fields not
built by the lean stage; no criterion uses them); no fallback to the full stage.

**RR3 (S+ placement, closed form, C).** The placement is contract C-LEAN-PLACEMENT (normative). Rule 7: the four
buffers that receive face arrays (PCG r, PCG p, corrector, MG level-0 x) have capacity
`m = max((nx+1) ny nz, nx (ny+1) nz, nx ny (nz+1))`. Closed form of S+ at the S2 (PCG) peak:
`8 [7 n + 4 sum_{l>=1} n_l] + 8 * 4 * (m - n) + S_scr`, with `S_scr` the pre-registered scratch list of C-MEMPHASE
(the spec's `8 x [7n + 4 sum n_l]` plus the rule-7 capacities and the enumerated scratch, both deferred by the spec
to the experiment note). C is evaluated on every `MEM_PHASE` line (F, S1, S2, S3, T) at 512^3 and 3328x384x384:
`exact_bytes` equals the sum of its listed items plus its listed cuFFT work areas, every item equals its geometric
size rule (C-MEMPHASE), S2 equals the S+ closed form, each with difference 0 bytes; S2 is the largest phase (derived
from "Expected numerical effect"). Completeness instrument: each `MEM_PHASE` line also prints `device_used`
(cudaMemGetInfo total - free at the phase peak); the note reports `device_used(phase) - device_used(stage start) -
exact_bytes` per phase against allocation granularity (2 MiB per allocation) and cuFFT plan internals; an excess
traced to an allocation of the lean path missing from the report is a C defect (BLOCKING). If a 3328 run stops by an
out-of-memory error in phase P, C at 3328 is evaluated on the phases completed before P and is "not evaluable
(consequence of D)" for P and later; this never counts as met.

**RR4 (A cases, fixed before any run).** Common arguments (the SF-37 V2 production setting):
`--track forward --sigma2 1 --ell 0.25 --seed S --n N --eps E --np-side 256 --planes 0.25,0.5,1,2,4,8,16
--times 0.25,0.5,1,2,4,8,16 --tol 1e-8 --hmax-div 8 --max-arclength 1000 --save-particles <run>/
--summary <run>.json` (summary outside the particles directory). Cases: N {64, 128, 256} x E {1, 1.5, 2} x S
{3101, 3102, 3103} (27); N 512 x E {1, 2}, S 3102 (2); `cov` = N 64, E 1, S 3101 + `--covectors`; `pol` = N 64,
E 2, S 3102 + `--comparator pollock` (31 cases). Reference: the base binary (RR11) at the default stage. Set LEAN =
the 31 cases + `--stage lean --fused-splines --single-launch`; set L6 = the 31 cases + `--fused-splines
--single-launch` (current stage). Per case, every comparison exactly equal: every file of the reference particles
directory exists in the head directory and is `cmp`-identical, except `*.json`, compared by `json_strip_cmp.py`
(SF-37 tool, exact float equality) dropping only `**.timings_s`; the head directory holds no extra file; lines
FIELD, TRACK, BACKFLOW, TPLANE, TTIME, TSTATUS, STATUS (`lines_cmp.py`, timing masks only); SETUP lines: for LEAN only
the lines beginning `SETUP   SF-18:`, `SETUP   SF-19:` and `SETUP   SF-19 corrector` (the STAGE, inlet and
v1(U-face) lines and `SETUP stage ... built in` are excluded by the spec), selected identically in both logs, file
order preserved; for L6 every SETUP line with the `built in <t>s` mask; case `pol` additionally the PTRACK, PPLANE,
PTIME, PSTATUS (and PFAIL, if present) lines; the exit code. Any DIFF or MISSING means A is not met.

**RR5 (D, E).** Sampler: `nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits
-lms 200` (all processes, all GPUs) plus device-wide `memory.used` of the assigned GPU, both started before and
stopped after the run; afterwards filtered to the PID of the `inlet_slab` process (printed by the lean path in
`MEM_GEOM pid=`, cross-checked with the child of `/usr/bin/time`); host RSS by `/usr/bin/time -v`; `free -b` before
each run; before each D run no other compute process may hold the assigned GPU. A PID series with fewer than 5
samples, or a PID peak below `max exact_bytes / 2^20` or above the device-wide peak, is NOT_EVALUABLE (sampler
failure = infrastructure: re-run with identical settings, disclosed). Context `C` = the minimum PID peak over three
`--context-probe 10` runs (all three recorded; a recorded-only "warm probe" adds one cuFFT D2Z/Z2D plan and one
cuBLAS call). D(512) uses the LEAN case N 512, E 2, S 3102 of RR4 (own run with the sampler); met iff measured PID
peak `<= E_max * 1.03 + C`, `E_max` = exact bytes of the largest MEM_PHASE (the reading `(E_max + C) * 1.03` is
reported alongside, not used as the verdict). D(3328): `--track forward --stage lean --fused-splines --single-launch
--grid 3328,384,384 --sigma2 1 --seed 4001 --np-side 100 --tol 1e-8 --hmax-div 8 --max-arclength 1000 --summary
<run>.json` (1e4 particles injected on x1 = 0 over the whole face; no `--save-particles`, no snapshot times), run (i)
sigma^2 2.25: `--eps 1.5 --ell 0.041666666666666664` (16 h) `--planes 0.041666666666666664,0.08333333333333333,
0.16666666666666666,0.3333333333333333,0.6666666666666666,1.3333333333333333,2.6666666666666665,5.333333333333333,
8.666666666666666`; run (ii) sigma^2 4: `--eps 2 --ell 0.08333333333333333` (32 h) `--planes 0.08333333333333333,
0.16666666666666666,0.3333333333333333,0.6666666666666666,1.3333333333333333,2.6666666666666665,5.333333333333333,
8.666666666666666` (planes `2^m ell <= L1` and `L1 = 3328 h`). D(3328) met iff both runs exit 0 with measured PID
peak `<= 30000` MiB; the per-PID peak is reported separately from the exit code (a non-memory failure is
classified as such). E met iff in both runs the three `SETUP   SF-19 corrector` lines read `status=converged` with
`rel_residual <= 1e-10`, `SF-19 ... converged=1`, `MEM_GEOM levels=7` and coarsest level 52x6x6; iterations and the
per-phase wall times (`MEM_PHASE seconds=`) recorded. Each D/E run is run once per audited head; runs on a
superseded head are excluded and disclosed; failures are recorded per the spec failure policy (owner, stop rule R3).

**RR6 (F).** `long_box_tests --full` on the V100 (part `controls`); the ctest runs a fast subset. All stages are
`LongBoxStage` instances. F1: Y = 0 (k = 1) on 64x16x16 and 128x16x16, M = 16, planes {0.25, 0.5, 1, 2, 3}, times
{0.5, 1, 2}, covectors on: SF-36 A1 thresholds; `max_ij abs(K_eff - I)` and `max_i abs(mean flux - e_i)` `<= 1e-12`.
F2: Y = cos(2 pi x1) at the cell centres, eps 0.5, 64x32x32 and 128x32x32, M = 16, same planes/times, covectors on:
transverse positions and covectors constant to `<= 1e-12`; tau `<= 1e-9` relative vs the 1-D quadrature of
`1/v_s1(x1)` on the same spline (SF-36 controls helper); I recorded. F3: SF-18 gaussian 128x32x32, sigma2 1, ell 0.25,
seed 3101, eps 1, 4096 fixed points (deterministic generator incl. points near the faces): for k in {2, 3},
`abs(v_s(x + e_k) - v_s(x))` `<= 1e-14` both absolute and relative to `abs(v_s(x))`, per point and component. F4:
cube = `LongBoxStage` on 64x64x64 with the SF-18 field (seed 3101, sigma2 1, ell 0.25, normalized) at eps 1; box =
`LongBoxStage` on 128x64x64 (h = 1/64) with the cube's unit-amplitude Y tiled twice in x1 as host Y; both at pcg
rtol 1e-12, M = 64, tol 1e-8, h/8, planes {0.25, 0.5, 1, 2, 4}: G and K_eff normwise relative error
`max abs(a - b) / max abs(b)` `<= 1e-8` (entrywise errors recorded); sigma_T^2 = (var x2 + var x3)/2, mean tau and
var tau per plane, `abs(a - b)/abs(b) <= 1e-8`.

**RR7 (G).** `macroflow3d_pipeline apps/config_pipeline_par2.yaml` from an empty working directory: base binary twice
(determinism control) and head once; every output file `cmp`-identical, except `manifest.json` compared by
`json_strip_cmp.py` dropping an explicit key list of time/date fields that L5 fixes from the code before J0 (no
line-regex masking). SF-27 trio (`config_pspta_small`, `config_streamfunctions_homogeneous`,
`config_streamfunctions_continuation`): the same rule. If base vs base differs on any file, G is **not met** on those
files (BLOCKING per the failure policy; the determinism evidence is attached; only the owner may waive). Byte report:
the ctest `flow_workspace_bytes` sums the capacities of the public members after `FlowWorkspace::allocate(grid, L)`
and asserts `8 n + sum_l 32 n_l` (the base layout, derived from the base code, is `32 n + sum_l 32 n_l`: 24 n more);
base and head Par2 PID peaks recorded as corroboration.

**RR8 (R).** Full V100 ctest (every registered test incl. the new ones; wall time compared with SF-37's 2901.55 s);
SF-27 trio vs base (RR7 rule); SF-35/36/37 default-path byte-compare vs base references = the SF-37 RR13 set
(`--construct trace` 32/64 x eps 0.5/1.0 x host/gpu and 64 eps 2 gpu; `--production` 32 eps 0.25
`--oracle-ladder`; the seven SF-36 forward cases; the three SF-37 opt-in cases with `--comparator pollock --save-vd
<dir>/vd --save-seed-velocity`) with every file and every line compared (incl. pollock_*, seed_v.npy, vd/*, PTRACK,
PPLANE, PTIME, PSTATUS, PFAIL, SEEDV; only `**.timings_s` dropped and timing fields masked); `git diff --numstat
9485beb..<head>` shows no change in `src/multigrid`, `src/numerics/{blas,operators,constraints,interpolation}`,
`src/physics/stochastic`, `src/physics/streamfunctions`, and zero deletions in `src/numerics/solvers/pcg.cuh` and
`src/physics/flow/AffinePeriodicFlowSolver.{cuh,cu}` with no hunk inside the base line ranges of the bodies of
`projected_pcg_solve` and `solve_affine_periodic_flow` (`build_production_stage` lives in an untouched file).
`validate_slepc_eigensolver` (Gate 2, needs the PETSc build absent from per-increment mirrors) is not run: no
eigensolver, operator or PETSc code is touched (disclosed, SF-36/37 precedent).

**RR9 (Recorded, no threshold).** S vs S+: stage and SF-19 times of the 512^3 LEAN case with `--bhat device` vs
default. L6 speedup per pass: N 256, E 2, S 3102 with the RR4 setting at the current stage, four variants (default,
`--fused-splines`, `--single-launch`, both): kernel seconds per pass and per launch; the three non-default variants
are also byte-compared with the default run by the RR4 rule (every file and line). Host RSS and `free -b` of every D
run; maximum cells measured; feasibility table per sigma^2 and lambda/h derived from the measured B/cell; CUDA
driver/runtime versions; MG iterations of an early-warning lean run on 1664x192x192 (same 52x6x6 coarsest level;
part `controls`, recorded).

**RR10 (timings, default outputs).** With `--single-launch`, `timings_s.plane_kernel` and `timings_s.time_kernel`
are null (not measurable separately) and `timings_s.single_kernel` carries the launch time; `single_kernel` (and
any new key) is emitted only with `--single-launch`/`--stage lean`; with default options the key set of
`track.json` and of the summary JSON is unchanged (structural check in L6/L4 acceptance).

**RR11 (campaign process).** Jobs in order, one at a time: J0 `sf40-base-refs` (mirror synced from a clean detached
worktree whose tree differs from `9485beb` only under `docs/` - checked by `git diff --quiet 9485beb <tree> -- .
':(exclude)docs'` and equal submodule gitlinks - so its build is the base build; tree hash and `inlet_slab` /
`macroflow3d_pipeline` md5 recorded; a dry run of the base-reference part into a separate root first, NOT evidence);
at the integrated head H1: `sf40-build`, `sf40-dry` (every part with tiny settings, separate root, NOT evidence),
`sf40-controls`, `sf40-bytecmp`, `sf40-l6`, `sf40-mem`, `sf40-par2`, `sf40-final` (full ctest + trio). This
replaces the spec's single `sf40-ctest-full-bytecmp` job (the default-path byte-compare runs in `sf40-bytecmp`, the
full ctest and the trio in `sf40-final`) and builds with a detached `run` job instead of `exec` (disclosed). Base
and head runs use the same V100 host and GPU model (V100-PCIE-32GB; glibc `exp` dispatch and CUB/cuBLAS
configuration are per CPU/GPU). An infrastructure interruption before a case's exit code exists is re-run with
identical settings and disclosed; nothing is re-run because of its result.

**RR12 (process notes).** The checker runs at activation and before every metadata commit. Workers and the
integrator run with the agent definitions present in the control checkout during this session (model `sonnet`,
effort `high`, set locally by the owner; the runbooks' "Opus / medium" is recorded as superseded for this run, not
changed in the repository). Files beyond the spec's "Files and symbols" list are disclosed in the PR (the two extra
tests, the new CLI options, `CMakeLists.txt`).

## Contracts

**C-LEAN-PLACEMENT (L2, normative).** Every reused buffer starts at a `cudaMalloc`/`cudaHostAlloc` base address;
every span is built from the buffer's `data()` with an explicit element count (n, or the exact CompactMAC count),
never from `capacity()`. The `MGHierarchy` is heap-allocated and never moved after the preconditioner is built.
Allocations at S2 start: `MGHierarchy(grid, L)` (L levels x {x, b, r, coefficient}); `levels[0].x` grown to
capacity m and resized back to n before the `ProjectedPositiveMGPreconditioner` is constructed; `pcg_r`, `pcg_p`,
`corrector` with capacity m and size n; `b_hat` = n doubles of pinned mapped host memory (`cudaHostAlloc(...,
cudaHostAllocMapped)` + `cudaHostGetDevicePointer`) for S+, or a device buffer of n for S; scratch per C-MEMPHASE.
Sequence: H2D of host K into `levels[0].coefficient`, `cudaDeviceSynchronize()`; the coarse levels by the
`coarsen_coefficient_geometric` loop exactly as `populate_coefficient_hierarchy` (`coefficient_hierarchy.cu:100-107`);
`A = LesterPositiveDiffusionOperator(grid, level-0 coefficient)`. For d = 0, 1, 2:
`assemble_affine_periodic_rhs(ctx, grid, q = level-0 coefficient, (e_d, e_d), rhs1 = levels[0].x, rhs2 = levels[0].b,
rhs workspace)`; `cudaMemsetAsync(corrector, 0)`; `projected_pcg_solve_external(A, M, b = levels[0].x,
x = corrector, r = pcg_r, p = pcg_p, z = Ap = levels[0].b, b_hat)`; `launch_affine_face_flux(K, corrector, e_d)` into
U = `pcg_r`, V = `pcg_p`, W = `levels[0].x` (exact face counts); `mean_unique_{u,v,w}` -> K_eff column d; for d < 2:
`ctx.synchronize()`, D2H of the corrector into host `w_d`. Host: symmetry defect, eigenvalues, Cramer G with the
existing helpers and their existing exception messages (not-SPD and singular-tensor throws kept verbatim).
`h_tilde = pcg_p`: `cudaMemsetAsync(0)`; for d = 0, 1: `ctx.synchronize()`, H2D `w_d` into `levels[0].r`,
`cudaDeviceSynchronize()`, `blas::axpy(G_d, levels[0].r, h_tilde)`; then `blas::axpy(G_2, corrector, h_tilde)`.
Final: `launch_affine_face_flux(K, h_tilde, G)` into u = `pcg_r`, v = `corrector`, w = `levels[0].x`;
`achieved_mean_flux` by `mean_unique_*`; `compute_divergence_kernel` (same launch configuration) into
`levels[0].b`; `div_max_abs` by `max_abs_reduction`, `div_rms` by `nrm2_host(divergence, PCG reduction
workspace) / sqrt(n)`; `ctx.synchronize()`; D2H of the inlet U plane `U[j (nx+1) + k (nx+1) ny]` into host index
`k + nz j` (the `inlet_v1_samples` layout of `build_production_stage`) and, if requested, of the full faces.
Release everything except `pcg_p`, moved out as `h_tilde` (capacity m). `x`, `r`, `p` of the PCG never live in MG
buffers; `b` may live in `levels[0].x` only because it is consumed before the first `M.apply`. Allocation failure ->
exception whose message names `phase=S2`.

**C-SYNC (all phases).** Every legacy-stream host-to-device copy (`cudaMemcpy`, `cudaMemcpy2D`, uploads of Y, lnK,
K, `w_d`, `coef_logk`, seeds, interleaved coefficients) is followed by `cudaDeviceSynchronize()` before any
ctx-stream use; every device-to-host copy and every overwrite of a buffer touched on the ctx stream is preceded by
`ctx.synchronize()`; any `cudaMemcpy*Async` to pageable memory is followed by `ctx.synchronize()`. No managed memory.

**C-MEMPHASE (L3/L4 producer, L5 consumer).** All byte counts are 64-bit (`%zu`/`%llu`). Lines:
`MEM_GEOM pid=<pid> nx= ny= nz= n= levels=<L> level_cells=<n_0>,...,<n_{L-1}> coarsest=<cx>x<cy>x<cz> face_max=<m>
bhat=<mapped|device> bhat_host_bytes=<b> cub_sum_bytes=<s(n_0)>,...,<s(n_{L-1})>` (`s(k)` =
`blas::sum_workspace_bytes(k)`, the CUB `DeviceReduce::Sum` temp query) and one line per phase
`MEM_PHASE phase=<F|S1|S2|S3|T> exact_bytes=<B> cufft_work=<W> device_used=<U> seconds=<t> items=<name>:<bytes>,...`
with `B = W + sum(items)` over every device allocation live at the phase peak (transient ones included). Item names
and size rules (`hs = (nx/2 + 1) ny nz`): F: `y` 8n, `sf18_spectrum` 16 hs, `sf18_reduce_blocks` (its capacity),
`sf18_mode_counter` 8, work `cufft_gen` (SF-18 report); the generator workspace and `y` are freed before S1. S1:
`lnk` 8n, `coef_logk` 8n, `prefilter_spectrum` 16 hs, work `cufft_prefilter_logk` (SF-28 report). S2:
`mg_l<l>_{x,b,r,coef}` 8 n_l (`mg_l0_x` 8m), `pcg_r` 8m, `pcg_p` 8m, `corrector` 8m, `bhat_device` 8n (S only), and
the scratch list S_scr: `scratch_pcg_d_scalar` 8, `scratch_pcg_mean_zero` s(n) + 8, `scratch_rhs_mean_zero`
s(n) + 8, `scratch_rhs_means` 32, `scratch_mg_mean_zero_l<l>` s(n_l) + 8 for every level, `scratch_reduce_blocks`
2048 (plus an explicitly named `scratch_<name>` item for any other fixed-size allocation, disclosed before the first
V100 run). The S2 line also prints the eight `AffinePeriodicFlowWorkspace::memory_report()` categories mapped from
the lean items (correctors, rhs_scratch, flux_scratch, reduction_scratch, affine_rhs_workspace, pcg_workspace,
mg_hierarchy, mg_preconditioner). S3: `h_tilde` 8m, `coef_pot` 8n, `prefilter_spectrum` 16 hs, `coef_logk` 8n,
`labels_<name>` (their capacities), work `cufft_prefilter_pot`. T: `coef_pot`/`coef_logk` 8n each or
`coef_interleaved` 16n (with the transient of building it), `labels_<name>`, `tracker_<name>` (every tracker device
buffer, incl. prepare transients). The lean path also prints the cuFFT work areas of the generator and of both
prefilters (spec item 4) in these lines.

**C-TRACKER-L6.** Opt-in (`TrackOptions::fused_splines`, `TrackOptions::single_launch`, default false), default path
byte-identical. Fused: one interleaved `double2 {pot, logk}` coefficient array; per axis one stencil call
(`detail::axis_stencil2` for eval_ext, `detail::axis_stencil` for eval; idx/w/dw shared by both splines); per spline
the same loop nesting, operand order and accumulation order as `evaluate_point`/`evaluate_point_hessian`,
accumulators initialised to 0.0, no peeling or splitting of the unrolled loops (only the loads are shared); the new
device functions and single-launch wrappers are `__forceinline__`; nothing in `src/numerics/interpolation` changes.
Single launch: one kernel whose thread runs, for its particle, `if (P > 0)` the unchanged plane pass with a fresh
`PassStats`, then `if (T > 0)` the unchanged time pass with a fresh `PassStats` (two step sequences, nothing
merged); `KernelArgs`/`PassStats` layouts unchanged. Compilation: every translation unit (library or test) that
includes `LagrangianTrackerDevice.cuh`, `LagrangianTrackerImpl.cuh` or the fused header is compiled with
`--fmad=false`; the new device-coefficient `prepare` and every tracker template instantiation live in
`LagrangianTracker.cu` (`LongBoxStage.cu` includes only `LagrangianTracker.cuh`); `cuobjdump -sass` of `inlet_slab`
shows no `CALL` to an out-of-line `evaluate_point*` / `axis_stencil*` / fused-evaluation function inside the tracker
kernels (checked locally and on the V100 build).
