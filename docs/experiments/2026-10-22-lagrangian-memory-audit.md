# Memory and time audit of the Lagrangian path after SF-37 (code analysis + existing measurements)

- Date: 2026-10-22
- Status: `complete — code analysis plus existing measurements; NOT pre-registered; no new runs except three read-only V100 queries (§5, §6); evidence for the owner decision of 2026-10-22, not acceptance evidence of any increment`
- Class: code reading at `da1c185`, a byte model (`scripts/memory_model.py`), re-tabulation of versioned SF-35/SF-36/SF-37 measurements.
- Increment context: SF-37 closed `done` (PR #56). SF-38 (long-domain infrastructure) was specified, and the non-periodic spline was named as a provisional increment, before this audit; this audit is the basis of the intermediate increment SF-40 and of the renumbering of the provisional spline to SF-41.
- Decision fed: [`2026-10-22-lean-lagrangian-stage.md`](../decisions/2026-10-22-lean-lagrangian-stage.md)
- Specification fed: [`SF-40-lean-lagrangian-stage.md`](../plans/active/lester-eq14/increments/SF-40-lean-lagrangian-stage.md)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) (Rule 1: `alpha_T` is not presupposed; this note makes **no** `alpha` claim), [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md) (domain)
- Artifacts: [`artifacts/2026-10-22-lagrangian-memory-audit/`](artifacts/2026-10-22-lagrangian-memory-audit/README.md) (`scripts/`, `analysis/`, `raw/`, `SHA256SUMS`)

Abbreviations: `A` = `artifacts/2026-10-22-lagrangian-memory-audit`; `S37` = `artifacts/2026-10-19-sf37-lagrangian-cube-verification`;
`S35F` = `artifacts/2026-10-17-sf35-sigma4-feasibility`; `MM` = `A/analysis/memory_model.txt` (stdout of `A/scripts/memory_model.py`);
`SM` = `A/analysis/sf37_measurements.txt`; `LH` = `A/analysis/lh_interpolation.txt`. All `path:line` references are at `da1c185`.

## 1. Question and scope

The owner requires the macrodispersion study to reach `sigma^2 = 4` on one V100-32GB with the Lagrangian consumer, at the `lambda/h` validated by SF-37
(8 / 16 / 32 at `sigma^2` = 1 / 2.25 / 4) and in a domain as long as possible (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`; SF-38). After SF-37 the
stage costs about 213 B/cell (exact bytes) and 512^3 = 2048x256x256 = 1.34e8 cells is the largest case that fits. Questions:

1. Where do the device bytes go, phase by phase, and does the byte model reproduce the measured peaks (including the 25921 ... 27981 MiB spread of SF-37)?
2. Which of those bytes can be removed **without changing a single result bit** (owner decision 1 of 2026-10-22), and what box does that allow within 30000 MiB?
3. What does the tracker itself cost (bytes, registers, time), and which of its costs are structural?
4. What else did the audit find in the pipeline, the Par2 baseline and the analysis path that matters for SF-38/SF-40?

Scope: reading and arithmetic only. Out of scope: any source change, any new run (three read-only queries excepted), any `alpha_L`/`alpha_T` statement. The
byte-saving levels of §8 are a **design analysis**, not an implementation or a measurement: none of them has been built or run.

## 2. Sources and method

- Code at `da1c185` (paths below). Every array of the stage was traced from its allocation to its release in `build_production_stage`
  (`src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cu:451-690`) and `run_track_forward` (`apps/inlet_slab/inlet_slab_driver.cuh:2620-2823`).
- Byte model: `A/scripts/memory_model.py` (stdlib, deterministic; output `MM`). "Exact" counts the CompactMAC faces `(nx+1)ny nz`, the half-spectrum
  `(nx/2+1) ny nz` and the `(N+1)` vertex planes; "nominal" counts every array as `n` cells (the convention of the planning session).
- Measurements already versioned: SF-35 feasibility (`S35F`), SF-36 note, SF-37 raw logs/JSON/`time -v` files (`S37`), SF-30 closure-gate summaries
  (cuFFT work areas). Re-tabulated by `A/scripts/tabulate_sf37_measurements.py` (`SM`) and `A/scripts/lh_interpolation.py` (`LH`).
- Three read-only queries on the V100 host (`scripts/remote --increment SF-37 exec`, mirror `~/MacroFlow3D-SF-37`): `cuobjdump -res-usage`, an `nvidia-smi` listing and a `free -b`/`nproc`/PCIe-link query (the last from the corrective node C1)
  (`A/raw/`). The binary is the **SF-37 campaign binary** `build/v100-release/inlet_slab` of 2026-10-20 16:38:52 -0300 (19:38:52Z; the `sf37-build` job started 19:37:32Z, `S37/logs/jobs/sf37-build.log`),
  built from the SF-37 measured source tree H1 `d866c40` (md5 `a2ecce53bdeda469b41593f706dd7028` in that log; SF-37 note line 11), not a build of `da1c185`. `git diff d866c40 da1c185 -- src apps CMakeLists.txt` is empty (checked), so the sources are identical;
  the md5 of the file on the remote was not recomputed.
- Evidence tags used in every table: `code path:line`, `run file`, `model` (printed by `memory_model.py`, section named), `estimate` (with basis). §11 classifies each claim.

## 3. SF-37 reading for memory and time

Sources: `docs/experiments/2026-10-19-sf37-lagrangian-cube-verification.md` (Result, Addendum), `docs/experiments/2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md`, `S37`.

- **Convergence in `h`.** The plume statistics converge with the grid at order about 2: V2 met 126/126 over the `lambda/h` ladder 8 ... 128, with observed orders 1.86-2.16 at `sigma^2 = 1`; V4 `e_v` order 1.99582-2.00679
  (SF-37 note, Result; post-hoc note I.1 and the V2 row of its Part I table). (The exponents in `S37/analysis/posthoc/exponents.txt` are growth exponents in `x1` at `N = 512`, not grid orders.)
- **Integration is limited by the cap.** `h_max = h/8` binds at every step: in the nine 512^3 V2 runs `rejected_steps = 0` and `nfev / (accepted + landing steps) = 6.0007-6.0008`, i.e. 6 field
  evaluations per step (`SM`; run `S37/raw/v2/v2_e*_N512.json`, `plane_pass`). V1 is therefore "met, vacuous" (RR15): the tolerance never selects the step (note, Result).
- **The grid deficit underestimates dispersion.** `S37/analysis/posthoc/bracket.txt` (O6 `sigma_T^2` relative to O6 at `N = 512`): at `x1 = 16`, `sigma^2 = 4`, seed 3102, -10.11 % at `N = 64` and -2.81 % at `N = 128`; in the 10 V2 failures on `sigma_T^2`
  the coarse grid gives less dispersion (post-hoc note, Part I table, V2 row). Sign verified for those rows only.
- **Production `lambda/h`** 8 / 16 / 32 at `sigma^2` 1 / 2.25 / 4, study value 32 = `N = 128` at `ell = 1/4` (note, Result). The smaller candidates fail in 11 of 756 cells.
- **Legacy resolution, discarded (owner decision 4).** The planning session estimated by interpolation the grid error at `lambda/h ~ 11` (MECOM 2025: `ell = 56.4`, `h = 5`, so `lambda/h = 11.28`; post-hoc note II.3,
  from the legacy input files, not re-read here). **Reproduced** by `A/scripts/lh_interpolation.py`: the worst-of-three-seeds errors against `N = 512` at `lambda/h` 10 (`N = 40`) and 20 (`N = 80`) come from the
  versioned `S37/analysis/posthoc/lh10.txt` (regenerated byte-identically here by running `S37/scripts/posthoc/lh10.py`), and a power law in `lambda/h` between them gives (`LH`):

  | `sigma^2` | `sigma_T^2(x1 = 16)` error | transverse-rate error | longitudinal-rate error |
  |---|---|---|---|
  | 1 | 1.51 % | 4.6e-4 `lambda` | 9.4 % |
  | 2.25 | 7.28 % | 3.5e-3 `lambda` | 35.1 % |
  | 4 | **15.73 %** | **8.2e-3 `lambda`** | **61.9 %** |

  The planning figures (about 15 %, 60 %, 8e-3 `lambda`) are the `sigma^2 = 4` row. These are **grid-discretization errors of the O6 ladder against its own `N = 512` run in the periodic cube** (superdiffusive,
  limited to `x1 <= 16`; post-hoc note), not values of any dispersivity, and not an `alpha` claim. They show why the SF-37 `lambda/h` is used and the legacy one is not.
- **Time at 512^3** (nine V2 runs, 65536 particles, `SM`): field 57.7-90.7 s (eight of nine within 57.7-63.6 s), stage 61.7-65.0 s, tracking 372.4-432.3 s (plane pass 191.5-223.4 s, time pass 180.8-220.2 s),
  total 524.1-561.4 s. Field + stage is about 2 min; the tracking of 65536 particles dominates (§6).
- **Corrector iterations.** All 27 SF-19 correctors of the nine 512^3 V2 logs took 20 PCG iterations (`SM`; `S37/raw/v2/*_N512.log`, `SETUP SF-19 corrector` lines). Used in §8 (`b_hat` touches).

## 4. Current memory map by phase and reconciliation with measured peaks

Phase map of `build_production_stage` followed by `run_track_forward` (exact bytes at 512^3, `MM` "section 4"; one cuFFT plan work area = one half-spectrum, see the note under the table).

| phase | B/cell | MiB | what is alive (code) |
|---|---|---|---|
| field: SF-18 generator | 24.06 (16.06 without `y_out`) | 3080 | `y_out` 8 (`SlabProductionSetup.cu:317`) + half-spectrum 8.03 (`PeriodicGaussianField.cu:283`) + one Z2D work area 8.03 (plan `:315-317`) |
| field: spectral vertex evaluation | 112.03 (+ Z2Z work area, not modeled) | 14340 | `y` 8 and the generator spectrum 8.03 (locals of `SlabFieldSource::gaussian`, alive until its return, `:317-323`) + 6 complex `N^3` arrays `in, F, ov, og[3]` = 96 (`:221-223`) for vertex values the tracker never reads |
| SF-19 solve | 196.67 | 25173 | workspace 156.62 + `K_dev` 8 + `lnK_dev` 8 + `u, v, w` 24.05 (`:479-499`); workspace detail below |
| first prefilter (`sh_ws`) | 196.71 | 25179 | workspace 156.62 + `lnK_dev` 8 + `sh_ws` coefficients 8 + spectrum 8.03 + 2 plan work areas 16.06 (`:562-563`; `K_dev, v, w` released `:514-516`, `u` `:541`) |
| **second prefilter (`sy_ws`)** | **212.74** | **27231** | workspace 156.62 + `lnK_dev` 8 + `sh_ws` 16.03 + `sy_ws` 16.03 + 2 work areas 16.06 (`:564`) |
| steps (v)-(vi) | 152.30 | 19494 | `sh_ws` 16.03 + `sy_ws` 16.03 + 7 vertex arrays `dpx..gz` 56.11 (`:592`) + `inputs` (q, lnk, grad_lnk x3 = 5 vertex arrays) 40.08 (`InletSlabGrid.cuh:142-165`) + `reference.vD` x3 24.05 (`:186-199`); SF-19 workspace and `lnK_dev` already released (`:570-571`) |
| tracking | 80.13 | 10256 | `inputs` + `reference` 64.13 kept by the driver + tracker `coef_pot`, `coef_logk` 16 (`LagrangianTracker.cu:237,256-257`) + 648 B/particle |

- **SF-19 workspace detail (156.62 B/cell, `AffinePeriodicFlowSolver.cuh:283-309`, `.cu:630-666`, `pcg.cuh:222-238`, `mg_types.hpp:27-50`):** correctors 24; `h_tilde` 8; `rhs_a, rhs_b` 16;
  `flux_u, flux_v, flux_w` 24.05 (CompactMAC: `(nx+1)ny nz` etc., `.cu:25-36`); `divergence` 8; PCG `r, z, p, Ap` 32 and `b_hat` 8; MG level 0 `x, b, r, coefficient` 32; coarse levels
  `32 sum_{l=1..7} 8^-l = 4.571` (8 levels at 512^3; `slab_auto_mg_levels(512) = 8`, `SlabProductionSetup.cu:269-277`). Nominal (all `n`): 156.57 and 196.57; the CompactMAC `+1/nx` adds 0.094. Per-level `MeanZeroWorkspace`
  CUB storage and `reduce_blocks` are a few KiB and not modeled.
- **Why the SF-19 workspace is alive at the second prefilter.** `htilde` is a span into `fws.h_tilde_`; it is read by the first prefilter (`:563`) and downloaded (`:568`), and `fws_p.reset()` only comes at `:570`, after both prefilters. Of its 156.62 B/cell only
  `h_tilde` (8) is needed there.
- **cuFFT work areas.** `cufftPlan3d` for the D2Z and Z2D plans (`PeriodicTricubicBSpline.cu:262-267`) allocates both work areas, which coexist until `:294-295`; `cufft_work_area_bytes` is their sum (`:302`). Measured at 256^3
  (SF-30 closure gate, `docs/experiments/artifacts/2026-10-05-sf30-closure-gate/raw/controls/a_generic3d_n256/summary.json:139-142`): coefficients 134217728 B, spectrum 135266304 B, work areas 270532608 B = exactly 2 x the spectrum.
  The model assumes each plan's work area equals one half-spectrum at every size (estimate; the 512^3 distribution below confirms it to 2-4 MiB). The SF-18 generator's work area is not printed anywhere (`closure_gate.cuh:815-824` prints only the statistics), so its 8.03 is the same assumption.
- **SF-18 field-phase bytes (discrepancy resolved).** The planning texts say "~24 B/cell" in one place and 16 in another. Both are the generator: 24.06 including the output field `y_out`, 16.06 for spectrum + work area alone. Neither is the field-phase peak:
  the peak is the spectral vertex evaluation, 112.03 B/cell (`SlabProductionSetup.cu:206-250`, `:317-323`), which produces vertex values and gradients for the FD diagnostics, not for the tracker. SF-35 measured the field phase at
  14825 MiB (`S35F/analysis/memory_phases.md`); model 14340 MiB, difference 485 MiB, consistent with a CUDA context of about the planning assumption (500 MiB) and no or a small Z2Z work area (interpretation; the Z2Z work area was not measured).

### Reconciliation with the measured peaks (512^3)

SF-37 logged `MEM_PEAK` (nvidia-smi `memory.used`, 200 ms, device-wide, includes the CUDA context) for every 512^3 run. Counts from both job logs (`SM`; `S37/logs/jobs/sf37-v2.log`, `sf37-science.log`, lines `MEM_PEAK ... N512`):

| MiB | runs | B/cell | reading |
|---|---|---|---|
| 25921 | 50 | 202.51 | plateau (SF-19 solve and first prefilter; 25173 MiB model + 748 MiB) |
| 25925 | 3 | 202.54 | plateau (4 MiB sampler quantization; the SF-35 feasibility value) |
| 25929 | 6 | 202.57 | plateau |
| 26951 | 1 | 210.55 | plateau + one plan work area (+1030) |
| 26953 | 3 | 210.57 | plateau + one plan work area (+1032) |
| 27981 | 9 | 218.60 | plateau + two plan work areas (+2060) |

72 runs in total. A single free parameter (the unmodeled overhead, 747.9 MiB = 25921 - 25173.1, `MM`) fits the plateau; the other two levels are then parameter-free predictions: plateau + 1028.0 = 26949.0 MiB (observed 26951-26953)
and plateau + 2056.0 = 27977.0 MiB (observed 27981); residuals 2-4 MiB. So the 200 ms sampler catches the cuFFT transient of the second prefilter fully in 9 of 72 runs, half of it in 4 and not at all in 59: the spread
is a sampling effect, and **the best estimate of the stage peak is the observed maximum 27981 MiB** (212.74 B/cell exact-byte peak = 27231.1 MiB, plus the same 748 MiB); other transients below the sampler's time resolution cannot be excluded. The planning text's "model 25716 MiB" is not reproduced: the model
plateau plus the 500 MiB assumption gives 25673 MiB (`MM`).

Other reconciliations (`MM`):

- **Tracking plateau.** The driver's own `peak_device_bytes` (`PhaseLog`, cudaMemGetInfo polled at phase edges only, `apps/inlet_slab/inlet_slab_driver.cuh:872-917`) reads 11652825088 B = 11113 MiB in all nine runs; the model gives 10256 MiB (80.13 B/cell);
  the difference 857 MiB is the 748 MiB overhead plus 109 MiB not itemized (candidates: D-1 label device buffers, the 42 MB of per-particle arrays, allocator granularity; not isolated). The tracker adds 16 B/cell = 2048 MiB to a plateau of 8208 MiB of stage arrays; those 8208 MiB are 64.13 B/cell
  that the tracker never reads (`inputs`: no reader in `src/physics/particles/lagrangian_labels/` or in `run_track_forward`; `reference.vD` only by the optional `--save-vd` export, `inlet_slab_driver.cuh:2538-2565`, used by the SF-37 runs at `eps` 1.5 and 2.0).
- **256^3.** Model plateau 3146.6 MiB; SF-35 J2a stage maximum 3769 MiB (SF-36 G: 3771) gives an overhead of 622 MiB. The overhead is therefore not a constant: 622 MiB at 256^3, 748 MiB at 512^3, ~485 MiB in the field phase
  (this section). A two-point linear fit (indicative only, two points) gives 604 MiB + 1.12 B/cell x cells, i.e. 1129 MiB at 3328x384x384. §9 carries the sensitivity.
- **Host RSS.** `/usr/bin/time -v` of the nine V2 512^3 runs: 25824852-25826816 kB = 197.0 B/cell (`SM`; the planning figure ~192 is not what the versioned files show); the Pollock runs (`keep_face_fluxes`, V3b) 28976760-28977636 kB = 221.1 B/cell (+24 = the three face
  arrays). Of the 197, `SlabFieldSource` holds `cell_` + `vtx_` + `vgrad_[3]` = 40.1 B/cell that the tracker never reads (`SlabProductionSetup.cuh:175`). Scaled linearly to 3328x384x384 the current host footprint would be 90.0 GiB;
  the V100 host has 134590115840 B = 125.35 GiB of RAM (`A/raw/host_memory_pcie.txt`), so the current host footprint would use 72 % of it.

## 5. Instrumentation findings

- **The 200 ms sampler misses transients.** Evidence: the distribution above (cuFFT transients of ~1-2 GiB seen in 13 of 72 runs). The sampler is `nvidia-smi --query-gpu=memory.used -i "$GPU_ID" -lms 200`
  (`S37/scripts/run_sf37.sh:452-459`; same pattern `docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core/scripts/run_sf36.sh:89-97`). `memory.used` is device-wide: it includes any other process on the same GPU.
- **`nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`.** Verified by the read-only query (`A/raw/nvidia_smi_visible_devices.txt`): with `CUDA_VISIBLE_DEVICES=1` it still lists GPU 0 and GPU 1 (Tesla V100-PCIE-32GB, 32510 MiB each). The campaign scripts therefore select the
  physical index with `-i`. A per-process figure needs a query by PID (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`); that command was **not exercised** in this audit.
- **`PhaseLog` measures only at phase edges.** `class PhaseLog` (`inlet_slab_driver.cuh:872-917`) calls `cudaMemGetInfo` in `begin()`/`end()` only; its `peak_used()` is `total - min(free)` over those polls, and the forward driver passes `workspaces_bytes = 0`
  (`inlet_slab_driver.cuh:2820`). In SF-37 it reads 11113 MiB for every run while nvidia-smi reads 25921-27981 MiB: it sees the tracking plateau, not the stage peak.
- **Exact-byte reports exist and are not printed on this path.** `AffinePeriodicFlowWorkspace::memory_report()` (`AffinePeriodicFlowSolver.cu:703-742`) fills `report.memory` (`:876`); `ProductionStageReport::summary()` does not print it. The SF-28
  `PeriodicTricubicBSplineReport` (`PeriodicTricubicBSpline.cu:298-304`, with `cufft_work_area_bytes`) is discarded at `SlabProductionSetup.cu:563-564`. `PeriodicGaussianFieldReport::cufft_work_area_bytes` (`PeriodicGaussianField.cu:389`) is stored in `R.sf18` and not printed.
  Only `apps/closure_gate/closure_gate.cuh:735-743` (`json_spline_report`) writes spline reports.
- **Proposed instrumentation (for SF-40, not done here):** print `memory_report()`, both spline reports and the generator work area in the SETUP block (they are exact and free), and log `cudaMemGetInfo` inside the stage at the allocation edges of §4.

## 6. Tracker

Sources: `src/physics/particles/lagrangian_labels/LagrangianTracker{.cu,.cuh,Impl.cuh,Device.cuh,Types.hpp}`, `src/numerics/interpolation/PeriodicTricubicBSpline.cuh`, `A/raw/cuobjdump_res_usage.txt`, `SM`.

- **Device memory: 16 B/cell + 648 B/particle.** Grid part: `coef_pot`, `coef_logk` (`LagrangianTracker.cu:237,256-257`). Per particle (`LagrangianTrackerImpl.cuh:63-79`, `LagrangianTrackerTypes.hpp:22-33`; derived, `MM` "section 6"): seeds 24 B; two `PassStats`
  of 88 B (7 `long long`, 2 `double`, 4 `int`) = 176 B; plane outputs `P x 32` and time outputs `T x 32` with `P = T = 7` = 224 + 224 B; total 648 B (covectors would add 672 B). 65536 particles: 42.5 MB.
- **Resource usage** (V100, `cuobjdump -res-usage` of the SF-37 campaign binary, `A/raw/cuobjdump_res_usage.txt`):

  | kernel | registers | stack/thread |
  |---|---|---|
  | `plane_kernel<5>` / `time_kernel<5>` (SF-37 runs, no covectors) | 255 / 255 | 1376 B / 1336 B |
  | `plane_kernel<11>` / `time_kernel<11>` (covectors) | 255 / 255 | 2024 B / 1888 B |
  | `flux_kernel` | 64 | 0 |

  Derived (`MM`; register file 65536 per SM and 80 SMs are V100 datasheet values, not queried): 255 registers allocate as 256, so 256 threads per SM and 20480 resident threads; 1e4 particles load 0.49 of them,
  65536 particles are 3.2 waves; local memory 28.2 MB. Register spilling is not reported (`LOCAL:0`; the 1376 B stack is the in-kernel array state). The default `block = 64` (`LagrangianTrackerTypes.hpp:61`).
- **Two splines, evaluated separately.** `SplineLagrangianField::eval_ext` (`LagrangianTrackerDevice.cuh:80-92`, called by `eval_checked`, `:109-114`) calls `evaluate_point_hessian` on the potential and `evaluate_point` on `ln k`. Each reads
  `4 x 4 x 4 = 64` coefficients (`PeriodicTricubicBSpline.cuh:223-371`), so 128 coefficient reads per evaluation. Each axis stencil includes a periodic reduction (`axis_stencil`, `PeriodicTricubicBSpline.cuh:169`); `axis_stencil2` calls `axis_stencil` and repeats the reduction
  (`:276-297`; `evaluate_point` `:223`, `evaluate_point_hessian` `:314`), so one evaluation does 3 x 2 + 3 = **9 coordinate reductions** (6 for the potential, 3 for `ln k`) although both splines share grid, coordinates and stencil. The arithmetic is compiled with `--fmad=false` (`CMakeLists.txt:243-254`).
- **Plane and time passes are two sequential launches** (`LagrangianTrackerImpl.cuh:192-212`) that integrate the same trajectory twice: in the nine V2 runs the plane pass makes 2.646e10-3.014e10 evaluations and the time pass 2.406e10-3.076e10 (`SM`, JSON `plane_pass`, `time_pass`); for `eps = 1`, seed 3101: 4.51e9 (plane) and 4.73e9 (time) accepted steps of 6 evaluations each.
  Merging them is not trivial: the landing rules differ (planes vs snapshot times), so the step sequences diverge after the first landing (inferred from the code, not tested).
- **Timings at 512^3, 65536 particles** (`SM`): plane pass 191.5-223.4 s and time pass 180.8-220.2 s (`tracker_timings_s`), tracking 372.4-432.3 s, stage 61.7-65.0 s, field 57.7-90.7 s. The cost per particle is set by `h_max = h/8` (§3).
- **Candidates for bitwise speedups (for the SF-40 specification; none is claimed to be faster, none has been tried):** (a) compute each axis stencil once and use it for both splines (same inputs, same pure function); (b) obtain `u` in `axis_stencil2` from the base stencil
  instead of repeating the reduction; (c) interleave the two coefficient arrays; (d) register-pressure measures that do not change the arithmetic. Each must reproduce the current outputs byte for byte, which is the owner's criterion.

## 7. Pipeline, Par2 and bugs

- **The tracker is not wired into the pipeline.** `git grep -n lagrangian -- src/runtime apps/macroflow3d_pipeline.cu` returns nothing; only `apps/inlet_slab/inlet_slab_driver.cuh` includes `lagrangian_labels` (`git grep -ln lagrangian_labels -- apps src`). `EnsembleRunner` (`src/runtime/ensemble/EnsembleRunner.cu`) runs
  the Par2/PSPTA path; the ensemble runner for the Lagrangian study is SF-38 scope.
- **Par2 baseline device bytes, ~149 B/cell** (derived from the allocations at `EnsembleRunner.cu:156-170`, `workspaces.cuh`, `fields.cuh:248-300`; `MM` "section 7"; config `apps/config_pipeline_par2.yaml`: 2048x256x256, `solver: mg_cg` = `PCG_MG`
  (`solve_head.cuh:106`), `mg_levels: 6`): `K_field` 8 + `head_field` 8 + `FlowWorkspace` {`residual, aux1, aux2` 24, `rhs` 8, MG 36.57, PCG `r, z, p, Ap` 32} + `StochasticWorkspace::logK` 8 + `PaddedVelocityField` 24.2 = **148.77 B/cell** = 19043 MiB at 1.34e8 cells.
  `VelocityDiagnostics` adds 24 B/cell when `diagnostics.velocity_field` is true (`velocity_diagnostics.cuh:37-53`: three `ScalarField`). Par2_Core particle arrays and the mode arrays are not modeled.
- **`FlowWorkspace::{residual, aux1, aux2}` are never read** (24 B/cell = 3.0 GiB at 1.34e8 cells, 16 % of the Par2 total). Proof by exhaustive search at `da1c185`:
  - `git grep -nE '\baux[12]\b' -- . ':!docs'`: in `src/` and `apps/` the only hits are `src/physics/common/workspaces.cuh:114-115` (declaration), `:138-139` (`resize` in `allocate`), `:159-160` (reset in `clear`); all other hits are in `legacy/` (unrelated `double *aux, *aux2` arguments of the legacy solver).
  - `git grep -nE '(\.|->)residual\b'` restricted to the access forms on a workspace (`flow_ws|flow|workspace|ws|fw` followed by `.residual`): no match; the remaining `.residual` hits belong to streamfunction reports (`report.residual`).
  - `src/physics/flow/solve_head.cu:90-219` reads only `mg_hierarchy`, `rhs`, `cg_workspace`, `pcg_workspace` of the workspace (`rhs` at `:154-155,213-217`).
  - Instances: `EnsembleRunner.cu:168`, `apps/analyze_invariant_quality.cu:449`, `apps/compare_eigensolver_backends.cu:268`.
- **Bug (recorded; out of SF-40 scope by owner decision; affects SF-38 criterion D).** `read_timeseries_csv` (`src/analysis/macrodispersion/MacrodispersionAnalysis.hpp:144-156`) skips one header line, but `CsvTimeSeriesWriter::write`
  (`src/io/writers/CsvTimeSeriesWriter.hpp:52-53`) writes two (`# format_version=1` and the column header). The column header is parsed by `strtod` as an all-zero row 0 (`:160-182`). This is known and handled in a test
  (`tests/lagrangian_labels/lagrangian_gpu_host_tests.cu:342-349`, "drop exactly that artifact row"), but `AnalysisRunner.cpp:51` reads the file as is and feeds it to `compute_macrodispersion`; the effect of the zero row on `compute_macrodispersion` was **not traced** in this audit.
- **`int32` indices in frozen kernels** limit grids to about 2.1e9 cells: e.g. `const int idx = i + Nx * (j + Ny * k)` in `src/multigrid/smoothers/gsrb_3d.cu:140` and `residual_3d.cu:143`, `bc_stencil_helpers.cuh:38`, and `for (int idx = tid; idx < total_cells; ...)`
  in `src/numerics/operators/varcoeff_laplacian.cu:88`. 3328x384x384 = 4.91e8 < 2^31 = 2.15e9, the B&dD domain at `lambda/h = 16` (5.50e8) also fits that limit; at `lambda/h = 32` (4.40e9) it does not. Not audited exhaustively.

## 8. Bitwise memory levels (O1, O2, S, S+)

Definition: a level changes only **where** values live and **when** buffers are released; every floating-point operation runs in the same order on the same operands, so outputs are byte-identical to the current ones (owner decision 1). Device B/cell at the SF-19 PCG peak, context excluded, nominal
arrays, 7 MG levels (`MM` "section 8"):

| level | what changes | B/cell | max cells (30000 MiB, 500 MiB context) |
|---|---|---|---|
| current | none (exact peak at the second prefilter 212.74; SF-19 phase 196.67) | 196.57 nominal | 1.45e8 (212.74) to 1.57e8 (196.67) |
| O1 | orchestration only (below) | **92.571** = K 8 + L0 coefficient copy 8 + L0 `x,b,r` 24 + coarse 4.571 + PCG `r,z,p,Ap` 32 + `b_hat` 8 + active corrector 8 | 3.34e8 |
| O2 | O1 + `K` loaded directly into the MG level-0 coefficient | **84.571** (no copy of K) | 3.66e8 |
| S | O2 + a PCG variant with external vectors where `z`, `Ap` and MG level-0 `b` share one buffer | **68.571** | 4.51e8 |
| S+ | S + `b_hat` in pinned, mapped host memory | **60.571** | 5.11e8 |

**O1 mechanisms** (all placement or lifetime changes):
- Two of the three correctors are copied to the host after their solve and uploaded again, one at a time, for the recombination `h_tilde = sum G_d w_d`. The recombination is the loop `axpy(G_d, w_d, h_tilde)` after a `memset` (`AffinePeriodicFlowSolver.cu:839-845`); `axpy` is elementwise
  `y[i] = a x[i] + y[i]` (`src/numerics/blas/axpy.cu:7-13`), so the same sequence of calls on re-uploaded operands is bitwise identical. The flux of each corrector is formed right after its solve (`:790-796`), so only the active corrector is needed on the device.
- `flux_u/v/w` (`:790-796`) and `divergence` (`:864`) live on buffers that are dead at that moment (PCG `r,z,p,Ap`, `b_hat`, MG buffers). **CompactMAC caveat:** the face arrays have `(nx+1)ny nz`, `nx(ny+1)nz`, `nx ny(nz+1)` elements (`:25-36`), more than `n`; placing them on "dead n-sized" buffers
  requires those buffers to be allocated with capacity >= the CompactMAC size (+`1/nx`: 0.03 % at 3328). `DeviceBuffer::resize` never shrinks capacity (`src/core/DeviceBuffer.cuh:75-87`) and `prepared_for` tests `size()`, so a larger-capacity buffer resized to `n` is accepted.
- The RHS assembler writes `rhs_a, rhs_b` and discards `rhs_b` (`:776-780`). `rhs_b` can live in PCG `z` (not yet live: first written by `M.apply`, `pcg.cuh:368`). `rhs_a` is the PCG input `b`, which the guard `pcg.cuh:291-296` forbids inside the PCG workspace (`b` and `x` may not overlap `r, z, p, Ap, projected_rhs, d_scalar`);
  it may live in an MG level-0 buffer, because `b` is dead after `blas::copy(b, b_hat)` (`pcg.cuh:321`) and the MG buffers are first used by `M.apply` (`:368`). The RHS assembler **does** have an aliasing guard, which the planning text assumed absent: `require_exact_nonoverlapping_spans`
  (`src/physics/streamfunctions/affine_periodic_rhs.cu:79-86`, called at `:126`) rejects overlap among `q`, `rhs1`, `rhs2` only. It does not prevent either placement above (neither overlaps `K`).
- `lnK` stays on the host (the host vector exists, `SlabProductionSetup.cu:479`) and is uploaded when the prefilter needs it; `u, v, w` are allocated after the PCG phase (they are written only at `AffinePeriodicFlowSolver.cu:848`). The post-PCG phase must release or reuse the PCG/MG buffers: with all of them alive it would be 124.57 B/cell (`MM`), with
  release or reuse 56 B/cell (K 8 + `h_tilde` 8 + upload scratch 8 + `u,v,w` 24 + `divergence` 8) < S+. This is a design condition of SF-40, not an observation.

**O2.** `populate_coefficient_hierarchy` copies `K` into level 0 (`src/multigrid/coefficient_hierarchy.cu:95-98`, a device-to-device `cudaMemcpyAsync`) and then builds the coarse levels with the public `coarsen_coefficient_geometric` (`:100-107`, definition `:67-84`). Loading `K` straight into
the level-0 coefficient and running the same coarsening loop gives identical coefficients; the operators `A` and `launch_affine_face_flux` then read that buffer. Nothing writes the level-0 coefficient during the solve.

**S** (new variant of `projected_pcg_solve` taking external vectors; the existing function and its guard stay):
- `Ap` is written by `A.apply(p, Ap)` (`pcg.cuh:387`), read by `dot(p, Ap)` and `r -= alpha Ap` (`:406`), and then dead; `recompute_true_residual` uses `Ap` as scratch (`:325-331`). `z` is written by `M.apply(r, z)` (`:368`, `:429`) and read by `project(z)`, `dot(r, z)` and `p = z + beta p` (`:446`), then dead.
  `Ap` is live from `:387` to `:406` (and is scratch inside `recompute_true_residual`, `:411`), `z` from `:368`/`:429` to `:446`; the intervals do not intersect: **`z` and `Ap` are never live together**.
- MG `apply` (`src/numerics/solvers/projected_positive_mg_preconditioner.cu:19-34`): `b0 <- r`, project, scale (`:28-30`); `x0 <- 0` (`:31`); V-cycle; project `x0`; `z <- x0` (`:33-34`). `b0` is written first and last read inside the V-cycle, `z` is written only at the end, so `b0` and `z` may share a buffer.
  `r` (the PCG vector) is read once at `:28` and is a different buffer from the MG level-0 `r`.
- Hence `z`, `Ap` and MG level-0 `b` share one buffer: 3 buffers become 1 (-16 B/cell). Every operation sees the same values in the same order.

**S+** and the `b_hat` traffic (`pcg.cuh:321-331,411,455`): per corrector `b_hat` receives one write (`copy`, `:321`), one `project` (sum read, then subtract read + write = 3 passes of traffic, `MeanZeroProjector.cu:77-86`), and one read per `recompute_true_residual` (the initial one plus one every `check_every = 10` iterations).
With the observed 20 iterations: 3 recomputes, so 1 + 1 + 2 + 3 = **7 array-traffic passes per corrector** (the planning figure "7-8"; 8 at 30 iterations). At 3328x384x384 (3.93 GB per pass) three correctors move 82.4 GB; at an **assumed** effective PCIe bandwidth of 10-12 GB/s (not measured on this host) (the link is PCIe gen3 x16, 15.75 GB/s theoretical, `A/raw/host_memory_pcie.txt`) that is
**6.9-8.2 s per realization** (`MM`), the planning estimate of ~+8 s. Pinned/mapped memory changes where `b_hat` lives, not its values. Open: the real mapped-memory bandwidth, and whether kernels reading `b_hat` over PCIe overlap with other work.

## 9. Feasibility at the SF-37 `lambda/h`

`MM` "section 9". Box 3328x384x384 = 4.907e8 cells (`468.0` MiB per B/cell); MG: `slab_auto_mg_levels(384) = 7` levels, coarsest 52x6x6.

| level | B/cell | MiB (cells x B/cell) | + 500 MiB context | within 30000 MiB |
|---|---|---|---|---|
| current, exact stage peak | 212.74 | 99564 | 100064 | no |
| current, SF-19 phase | 196.67 | 92039 | 92539 | no |
| O1 | 92.571 | 43323 | 43823 | no |
| O2 | 84.571 | 39579 | 40079 | no |
| S | 68.571 | 32091 | 32591 | no |
| **S+** | **60.571** | **28347** | **28847** | **yes (margin 1153 MiB)** |

- **`lambda` extents** of the box (cells / `lambda/h`): 416 x 48 x 48 at 8 (`sigma^2` 1), **208 x 24 x 24 at 16** (2.25), **104 x 12 x 12 at 32** (4).
- **The 500 MiB context is an assumption** (planning). The measured unmodeled overhead is 485 MiB (field phase), 622 MiB (256^3 stage), 748 MiB (512^3 stage); S+ plus 750 MiB is 29097 MiB (margin 903), plus 1000 MiB 29347 (margin 653), plus the two-point extrapolation 1129 MiB 29476 (margin 524). **The 30000 MiB target is met at S+ with
  a margin of 0.5-1.1 GiB depending on a quantity that was measured at two grid sizes only.**
- **B&dD domain** (204.8 x 25.6 x 25.6 `lambda`; SF-38 spec, lines 20 and 47): 5.50e8 cells at `lambda/h` 16 = 1.08x the S+ maximum cell count (needs 31757 MiB without context, 32257 with) - **does not fit even at S+**; 4.40e9 cells at 32 = 8.6x - infeasible on one GPU.
  Going below ~60 B/cell needs non-bitwise changes or medium (b) (§10).
- **FFT phases stay below S+.** Derived from §4: the SF-28 prefilter of one field is coefficients 8 + half-spectrum 8.03 + two work areas 16.06 = 32.09 B/cell by itself (measured 540016640 B / 16777216 cells = 32.19 at 256^3, SF-30 summary), the generator 24.06. In a lean stage with the SF-19 workspace freed and the first spline's spectrum released, the second prefilter is
  `lnK` 8 + `sh` coefficients 8 + `sy_ws` 16.03 + work areas 16.06 = **48.09 B/cell** (56.13 if the first spectrum is kept) < 60.57. The 96 B/cell spectral vertex evaluation (§4) must not exist in the lean stage; if kept, the field phase (112 B/cell) would exceed S+.
- **MG coarse solve.** The same rule applied to the minimum dimension 384 gives a 52x6x6 coarsest grid; the coarse solve is 50 GSRB sweeps (`mg_types.hpp:20`). The coarsest grid is 52 cells long in `x`, so the coarse correction may under-solve and PCG iterations may grow beyond 20 (not tested; the byte count above assumes none). The 8-level cube has a 4^3 coarsest grid.
- **Host side.** At 197.0 B/cell the current host footprint extrapolates to 90.0 GiB at 3328x384x384 (§4); host RAM is 125.35 GiB. Two correctors offloaded in O1 add 2 x 3.93 GB on the host.
- **Time** (estimate): the stage is about 62-65 s at 1.34e8 cells (§3) with no per-phase breakdown in the logs; a box 3.66x larger is of the order of 7 min per realization for field + stage (2 min x 3.66) if the cost scales with cells (estimate, not measured), plus ~7-8 s for the S+ traffic.

## 10. Medium (b), Dirichlet in `x` (recorded, no scope change)

Recorded for SF-38 (medium A/B evaluation, `docs/plans/active/lester-eq14/increments/SF-38-long-domain-infrastructure.md`); **no scope change** of SF-40 results from it.

- **Validity.** Medium (b) = `solve_head` with Dirichlet in `x` and periodic `y/z` (the Par2 and Beaudoin-de Dreuzy configuration), velocities by `compute_velocity_from_head`. The helicity-free property is that of the form `v_s = k_s grad Phi_s` (`v_s . curl v_s = 0` pointwise, decision 2026-10-17, item 2) and does not use periodicity in `x`; medium (b) is item 7 of that decision.
- **Needs:** the non-periodic spline (SF-41, formerly the "provisional SF-40"), buffers of about 5 `lambda` at each `x` face (SF-38 stops particles at `L1 - 5 lambda`; B&dD inject at >= 5 `lambda`; post-hoc note II.4) i.e. 10 `lambda` of 208 `lambda` = **4.8 % more cells** at `lambda/h` 16 (derived), and re-verification of the `lambda/h` of SF-37 (which was established on the periodic cube).
- **Memory saving: not established by this audit.** The planning session claimed 10-20 % (one non-projected PCG, ~53-60 B/cell). A buffer count of the non-projected solve (`src/physics/flow/solve_head.cu:195-219`: `rhs`, head `h`, MG, PCG `r,z,p,Ap`) with the §8 tricks gives 68.57 B/cell (K as L0 coefficient 8 + L0 `x,r` 16 + shared 8 + coarse 4.57 + PCG `r,p` 16 + `h` 8 + `rhs` 8),
  the same as level S; the planning range is **estimate (planning session 2026-10-22, not re-derived)**.
- **Time saving.** One solve instead of three correctors is a factor 3 on the PCG part (all three correctors take 20 iterations, §3). The share of the PCG in the 62-65 s stage is not measured (estimate for the total effect: open).
- **Does not relax** the required `lambda/h` (the grid error of §3 is a property of the tracker's spline representation).

## 11. Classification and limits

| claim | class |
|---|---|
| Array inventory, lifetimes and release points of the stage (§4); CompactMAC sizes; spectrum and work-area sizes per plan (shape) | confirmed in code |
| Model reproduces the SF-37 plateau (fit, 1 parameter) and the +1028 / +2056 MiB levels (parameter-free, residual 2-4 MiB); the 9 / 4 / 59 counts | confirmed in runs (72 runs) + derived |
| The sampler misses the cuFFT transient in most runs; stage peak best estimate = 27981 MiB at 512^3 | derived from the distribution (consistent with exact-byte peak + 748 MiB) |
| Work area of one plan = one half-spectrum at 512^3 | estimate (measured as the sum of two plans at 256^3; consistent with the 512^3 levels) |
| Unmodeled overhead 485 / 622 / 748 MiB; linear extrapolation 1129 MiB at 4.9e8 cells | derived by difference (two points; the extrapolation is indicative only) |
| Host RSS 197.0 B/cell (221.1 with Pollock faces); 90 GiB extrapolation | confirmed in runs (RSS); extrapolation derived; host RAM 125.35 GiB: confirmed by query |
| Tracker 16 B/cell + 648 B/particle; 255 registers, 1376 B stack; 9 coordinate reductions and 128 coefficient reads per evaluation | confirmed in code / confirmed in runs (cuobjdump of the SF-37 binary, built from sources identical to `da1c185`) |
| 20480 resident threads; 1e4 particles = 0.49 load | derived (V100 datasheet constants) |
| Tracking 372-432 s at 512^3; 6.0008 evaluations per step; `rejected_steps = 0` | confirmed in runs |
| Grid error at `lambda/h` 11.28: 15.7 % / 8.2e-3 `lambda` / 61.9 % at `sigma^2` 4 | derived (power-law interpolation of versioned post-hoc numbers); not an `alpha` value |
| Par2 148.77 B/cell; `residual, aux1, aux2` never read | derived from allocations; confirmed in code (exhaustive `git grep`) |
| CSV header bug (two header lines vs one skipped) | confirmed in code and by a test comment; effect on `compute_macrodispersion` not traced |
| `int32` index limit ~2.1e9 cells | confirmed in code (representative kernels; not exhaustive) |
| Levels O1, O2, S, S+ B/cell and the lifetime arguments (`z`/`Ap`/`b0`; `rhs_a`, `rhs_b`; `axpy` elementwise) | derived from code reading; **not built, not run**; bitwise equality is a design claim to be verified by byte-compare in SF-40 |
| S+ at 3328x384x384 = 28847 MiB with 500 MiB context; margin 0.5-1.1 GiB | derived; margin depends on the unmodeled overhead (open) |
| `b_hat` over PCIe: 7 traffic passes per corrector; +6.9-8.2 s per realization | derived passes; bandwidth is an assumption (not measured) |
| Coarse solve on 52x6x6 may increase PCG iterations | open question (risk, not tested) |
| Medium (b): 4.8 % more cells; memory saving 10-20 % | derived (cells); saving: planning estimate, not reproduced here |
| Tracker speedup candidates (§6) | proposals; no speedup measured |

**Limits.** (1) The byte model rests on the code at `da1c185`. The SF-37 measurements come from sources identical to `da1c185` in `src/`, `apps/` and `CMakeLists.txt` (§2); the SF-35 feasibility binary is the P2 build of a different tree (`S35F`), and the match of its 25925 MiB with the SF-37 plateau supports the same allocation pattern.
(2) The cuobjdump data come from the SF-37 binary (same sources; CUDA/GCC versions were not recorded in the SF-37 logs). (3) Planning numbers not re-derived are marked as such; four planning statements were not reproduced: the model value 25716 MiB at 512^3 (here 25673 MiB with the 500 MiB assumption), steps (v)-(vi) ~154 B/cell (here 152.30), host RSS ~192 B/cell (versioned files: 197.0), and "no aliasing guard" in the RHS assembler (there is a pairwise non-overlap guard, §8); the medium (b) memory saving of 10-20 % is carried as a planning estimate. (4) No new run was made: nothing here is acceptance evidence of any increment. (5) Rule 1: no statement about `alpha_L` or `alpha_T`.
