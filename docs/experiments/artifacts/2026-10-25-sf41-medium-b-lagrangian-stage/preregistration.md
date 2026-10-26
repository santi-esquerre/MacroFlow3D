# SF-41 pre-registration: reading rules and implementation contracts (activation commit, before any run)

- Increment: `docs/plans/active/lester-eq14/increments/SF-41-medium-b-lagrangian-stage.md` (State `active` in the same commit).
  Base: `2cf48c19dac641e1ce8c823c5cf2d6165ac66757` (`master`, PR #64 merged 2026-10-24T23:37:33Z; owner decisions E1-E12 of
  `docs/decisions/2026-10-24-sf41-sf39-planning-decisions.md`). Written by the SF-41 orchestrator after UNDERSTAND (code
  reconnaissance, two adversarial reviews and the pre-activation review `docs/experiments/2026-10-24-sf41-pre-activation-review.md`);
  committed in the activation commit, before any build, test or run of SF-41.
- What this file is: the durable text of the activation reading rules (RR0-RR12) and implementation contracts (C-*) that the
  activation bitácora row cites. It changes no threshold, Goal, design value or failure-policy text; it fixes the items the
  specification leaves to the activation row (Implementation specification 1) and resolves gaps inside the listed scope. The
  acceptance thresholds A-H are the specification text, copied verbatim into the activation row.
- No `alpha_L` / `alpha_T` statement is made or allowed (Rule 1). Plume statistics produced by SF-41 runs (E, F's `D_P`, H's recomputed
  `C_STAT`) are recorded, never interpreted as macrodispersion.

## Reading rules

**RR0 (process).** No threshold is tuned after any run. Every V100 run is a detached orchestrator job
(`scripts/remote --increment SF-41 run`/`wait`), one job at a time on one GPU, launched from a clean worktree at a recorded commit
and running committed scripts; workers never use the remote. Local work is light: builds only through
`/home/sesquerre/.local/bin/mf3d-build`, new test cases `<= 32^3` by explicit `ctest -R`, nothing larger (owner rule). Every
pre-registered part runs to completion whatever the intermediate verdicts (no outcome-dependent stopping); a scientific failure is
never re-run with changed settings; an infrastructure interruption (ssh drop, host reboot, script crash before a case's exit code
exists) is resumed with identical settings, disclosed, and the superseded attempt excluded. A defect found by a gate is corrected by a
recorded corrective node and the affected part is re-run on the corrected head (thresholds unchanged), disclosed. Dry runs are not
evidence. GPU-hours per part are reported.

**RR1 (base and head).** Base binaries are built on the SF-41 mirror from a clean tree equal to `2cf48c1` (job `sf41-basebuild`,
before any SF-41 source reaches the mirror) and copied with their md5 to `~/sf41_base_refs/bin`: `inlet_slab`,
`macroflow3d_pipeline`, `lagrangian_ensemble`, `lagrangian_medium_ab`. They produce the G references (job `sf41-baserefs`). Head =
the integrated source-bearing commit H1; every head job runs on a tree equal to H1 (or a later commit that differs from H1 only under
`docs/`), echoed with its tree hash in the job log.

**RR2 (memory sampler; D, E).** As SF-38 RR2: `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms 200` filtered by the run's PID (PID peak), the device-wide `memory.used` of the physical GPU,
`/usr/bin/time -v`, `free -b` before the run, and a precheck that no compute process is on the GPU. Verdict on the PID peak
(`<= 30000 MiB`) and exit 0; gates `>= 5` samples and PID peak `<=` device-wide peak.

**RR3 (A).** Program P-SPLINE (C-SPLINE) in its full tier on the V100. Boxes `nx != ny == nz`, `ny` a power of two, `h = 1/ny`,
`L1 = nx h`; `S = sin(2 pi x2) cos(2 pi x3)`. Point sets: `PS(box)` = 1e4 points uniform in `[0, L1] x [0, 1)^2` from
`std::mt19937_64(20261025)` plus 2000 points per face uniform in `[0, 1/16] x [0, 1)^2` and `[L1 - 1/16, L1] x [0, 1)^2` (same
generator, continued); `MARGIN(box)` = per grid, `x1 = -h r` and `x1 = L1 + h r` with `r = (j + 1/2)/64`, `j = 0..31`, transverse
coordinates from the same generator. Checks:
- node interpolation on 32x16x16 and 64x16x16 with random cell samples (`std::mt19937_64(4101)`, uniform `[-1, 1]`): every cell centre;
- partition of unity: constant 3.0 on 32x16x16, 64x32x32, 128x64x64, `PS`: `max |s - 3|`; gradient and `MARGIN` recorded;
- reproduction: `p(x1) = 1 + 0.5 (x1 - L1/2) - 0.7 (x1 - L1/2)^2 + 0.3 (x1 - L1/2)^3` and `p(x1) = (x1 - L1/2)^2` on 64x16x16 and
  128x16x16 (`L1 = 4, 8`), `PS` plus `MARGIN`: value and gradient relative (Euclidean per point); Hessian recorded;
- order ladder: `f = exp(x1) sin(3 x1) (1 + 0.05 S)` on 32x16x16, 64x32x32, 128x64x64 (`L1 = 2`), `PS` of `L1 = 2`, errors in max and
  RMS norms (value: absolute; gradient: Euclidean per point; Hessian: max over points and six components); order = `log2(e_h/e_{h/2})`;
  gate on the minimum over the pairs 16->32 and 32->64 (value max and RMS `>= 3.8`, gradient max and RMS `>= 2.8`, Hessian max `>= 1.8`).
  Predicted (pre-activation emulation `a_ladder.txt`): 3.9024 / 4.3659 / 2.9042 / 3.3858 / 1.9154. Recorded: the `x`-only sub-ladder of
  `g(x1) = exp(x1) sin(3 x1)` on `L1 = 2` at 16/32/64/128 cells per unit length (boxes `nx x 4 x 4`) on 4000 near-face points, and the
  measured-minus-emulated error per level;
- CPU vs GPU on 32x16x16 and 64x32x32 with the ladder field: full pipeline value; evaluation only (GPU coefficients evaluated by the host
  and by the device) value, gradient, Hessian; `PS` plus `MARGIN`; relative to the max of the CPU reference; recorded: coefficient
  difference relative to `max |c|`, full-pipeline gradient and Hessian per region (interior `0.5 <= x1 <= L1 - 0.5`, near-face, margin);
- Hessian path: `memcmp` of value, gx, gy, gz from the Hessian path and the value path, host and device, on `PS` of 64x32x32;
- no allocation: `cudaMemGetInfo` free bytes before/after four batched evaluations of 1e4 points (value and Hessian kernels), 64x32x32;
- shift: dyadic coordinates `k / 2^16` (17 values per axis, other two coordinates from two fixed settings), `q` in {-2, -1, 1, 2, 3}
  periods in `x2` and in `x3`, value, gradient, Hessian, `memcmp` and max abs (gate `<= 1e-14`), on 32x16x16 and 64x32x32, host and device;
- evaluation domain: `MARGIN` points evaluate finite and equal (`<= 1e-12` relative) to the end cubic piece evaluated independently from
  the coefficients; points at `x1 = -h/2 - h/8` and `x1 = L1 + h/2 + h/8` are reported outside without any coefficient read (the
  functor's report flag; the test runs under `compute-sanitizer --tool memcheck` in the full tier).
A met iff every gated check passes on every listed box, host and device.

**RR4 (B).** Program P-CROSS on the V100: SF-18 Gaussian, `sigma2 = 1` unit variance with `normalize_variance`, `eps = 2`,
`ell = 0.25` (`lambda/h = 32`), seed 6001, box 1664x128x128 (`L1 = 13`, 52 `lambda`); the SF-19 lean affine-periodic solve
(`solve_affine_periodic_flow_lean`, `rtol = 1e-10`, automatic hierarchy) gives `h_tilde` (host copy); the periodic SF-28 spline and
the non-periodic spline (C-SPLINE, closure on the raw samples) represent the same samples; evaluation set = 1e5 points uniform in
`[0.5, L1 - 0.5] x [0, 1)^2` (`std::mt19937_64(6001)`; distance `>= 2 lambda = 0.5` from both faces) plus the cell centres with
`x1 >= 0.5` and `x1 <= L1 - 0.5` of the three `x1` lines `(j, k) = (0, 0), (64, 32), (37, 101)`. Quantities: value
`max |s_np - s_p| / max |s_p|`; gradient `max |g_np - g_p|_2 / max |g_p|_2`; Hessian `max_{points, comps} |H_np - H_p| / max |H_p|`;
gate `<= 1e-11` each. Recorded: along the three lines over their full length, `|s_np - s_p| / max |s_p|` (and the gradient and
Hessian analogues) against the knot distance from each face; the per-knot factor fitted (least squares of the log) on the knots whose
difference exceeds `1e3` times the interior floor (median over knots 64..nx-65); the same comparison for the `ln k` spline (`eps Y`).

**RR5 (C).** Program P-CONTROLS on the V100 (fast sizes also in `ctest`). Common: `G1 = 1`; `h = 1/16` (64x16x16, 128x16x16) or
`h = 1/32` (64x32x32, 128x32x32); `k = 1` from `eps Y = 0`; `tol = 1e-8`, `h_max = h/8`; heads written into the sample array:
C1 `H = G1 (L1 - x1)` at cell centres; C2 the exact discrete layered head (constant face flux `q = G1 L1 / R`, `R` the series
resistance with harmonic face `K` and the half-cell Dirichlet faces, `H` accumulated cell by cell from the west face); C3/C4 the
written `f` of the specification. Spline field: C-SPLINE for `f` and for `ln k`; fused path when built, and the unfused path, both.
- C1: seeds `X0 = 0`, window `[0,1)^2`, `M = 16`; planes {0.25, 0.5, 1, 2, 3} (`L1 = 4, 8`), times {0.5, 1, 2}; checks as the spec
  (transverse displacement abs; `tau` relative to `(X - X0)/G1` on planes and to `t` on times as positions; `|I|` abs; covectors abs).
- C2: the specification's profile, planes, `X0 = 0`, `M = 16`, window `[0,1)^2`; `tau` against the SF-36/SF-38 controls quadrature
  (G7K15 adaptive, atol 1e-15 per piece, split at half cells) of `1/v_s1` along the straight line on the host functor of the same
  spline; plane records only for `tau`; `I` against the quadrature of `(grad ln k . g + tr H)/g1` recorded.
- C3: the specification's field; 4096 points from `std::mt19937_64(20261022)` uniform in `[0, L1] x [0,1)^2`, every fourth with its
  `x2` (cycling: `x2`, `x3`, both) set to a multiple of `h`; dyadic set as RR3; gate per component as the specification.
- C4a: 64x16x16 and 128x16x16, `k = 1`, `X0 = 0.5`, `M = 16`, window `[0.1, 0.9]^2`; plane pass with planes {1.0, `L1 + 0.5`},
  time pass with times {1.0, `2 L1`}; tracker (fused and unfused) and Pollock non-periodic (plane and time modes) on the `k = 1`
  faces (`U = G1`, `V = W = 0`): every particle `exit_outlet`, the first plane / time reached by all, no non-finite reached record or
  final position; counts exact.
- C4b: as the specification (seeds `x2, x3 = (m + 1/2)/64`, inclusion by analytic `v1 >= 0.05` and `|psi0 - round(psi0)| >= 0.02`);
  reference classification computed inside the program from the exact stream function and an independent host adaptive RK
  integration of the analytic field (Dormand-Prince 8(5,3) or 5(4) with `rtol = 1e-12`); Pollock on face fluxes integrated
  analytically (`U = h^2`-averages of `v1` over each x-face, idem `V`, `W`; per-cell divergence recorded); semantic map below;
  counts exact.
- C5: the lean stage (C-STAGE) on the C1 and C2 media (`Y` cells), `mg_levels` automatic, `rtol = 1e-10`, `cg_max_iter = 2000`.
- Semantic map (tracker ↔ Pollock): `exit_inlet` (7 ↔ 8), `exit_outlet` (8 ↔ 9), "reached the plane / time" (`ok` with the
  record finite ↔ `kStatusActive` with the record finite).

**RR6 (D).** Program P-CONTROLS `--only D` on the V100. Field: SF-18 Gaussian, unit variance, `normalize_variance`, `eps = 2`,
`ell = 0.125`, seed 4001, generated once per size and fed (host `Y`) to both `DirichletMedium` and the lean stage, run one after the
other in one process. Sizes 128x32x32, 512x64x64 (`G1 = 1`) and 3328x256x256 (`G1 = G[0]` of `build_long_box_stage` on the same `Y`,
`eps`, automatic hierarchy). `DirichletMedium` with `mg_levels = box_auto_mg_levels`, `cg_rtol = 1e-10`, `cg_max_iter = 2000`,
`keep_host_head = true`. Compared: the head (`memcmp` of the host arrays), the `B_SOLVER` fields (converged, iterations, initial,
final, rel, qbar, Q_in, Q_out, balance, max_div, fallback) as printed by `lagrangian_medium_ab`'s format, string-equal. The mapped-host
head variant of the lean stage is compared the same way at 128x32x32 and 512x64x64. Recorded: at 3328x256x256 the comparison of both
lines with SF-38's committed C0 line (`docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/raw/sf38_campaign/bpilot/bpilot.log`).

**RR7 (E).** P-RUNNER (C-RUNNER-B) with `--medium b`, seed 4001, `--nr 1`, `normalize_variance` on, `G1 = 1`, `X0 = 5 lambda`, window
`[0.1, 0.9]^2`, `M = 100`, `tol = 1e-8`, `--hmax-div 8`, `--planes auto --times auto`, fused splines and single launch on (when built),
twin series and Pollock bridge on, `flow_stats.json`. E1 `--grid 2048,256,256 --eps 1 --ell 0.0390625`; E2 `--grid 3328,384,384
--eps 1.5 --ell 0.041666666666666664`; E3 `--grid 3328,384,384 --eps 2 --ell 0.08333333333333333`; E4 `--grid 6656,256,256 --eps 2
--ell 0.25`. Solve verdict: the stage's solver line `converged=1` with `rel <= 1e-10`, or the C0 fallback (`rel <= 1e-8` with balance
met, recorded); balance from the stage's `Q_in`, `Q_out`. Memory per RR2 with the device head; if the PID peak of a grid exceeds
30000 MiB, that grid is re-run once with the mapped-host head (the pre-authorized fallback E1; disclosed, both runs reported). The
per-phase `MEM_PHASE` exact bytes and the non-solve-phase rule of the specification are read from the log.

**RR8 (F).** P-RUNNER `--medium b` with the specification's parameters; seeds 6101, 6102, 6103 (one realization each, `--seed0 <s>
--nr 1`); `eps` 1 and 2; grids 384x32x32, 768x64x64, 1536x128x128, 3072x256x256 with `--ell 0.5`; `--planes` explicit
`X0 + m lambda`, `m = 1..14` (`X0 = 2.5`); Pollock bridge on (for `D_P`), twin off allowed; 24 runs. `e_v` per plane = the tracker's
`rms_ev` (flux-weighted over the particles that reached the plane). Verdict per (eps, seed, plane): strictly decreasing over the four
rungs and `log2(e_64/e_128) >= 1.8`. `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` (`sigma_T^2 = (var x2 + var x3)/2`, each with its
own flux weights over its own contributors); `SE_D` by a paired block bootstrap over the 4 blocks of 50 x 50 seeds (block of particle
`p = m2 M + m3`: `(m2 // 50) * 2 + (m3 // 50)`), `B = 2000`, `numpy.random.default_rng(20261025).integers(0, 4, size=(2000, 4))`;
V3b classification: `concordant` iff `|D_P|` decreases on every refinement and `|D_P(128)| <= max(0.05 sigma_T^2(consumer, 128),
2 SE_D)`, else `discordant` (recorded, not blocking).

**RR9 (G).** Base references (job `sf41-baserefs`, base binaries) and head runs (part `final`) with identical arguments:
(i) the SF-27 trio and the SF-38 20-run R set exactly as SF-38 RR9 / the `baserefs` part of `run_sf38.sh` (compare tools
`json_strip_cmp.py`, `lines_cmp.py`; manifest keys `timestamp`, `build.git_hash` dropped); (ii) a medium (a) runner run
`lagrangian_ensemble --grid 512,64,64 --eps 2 --ell 0.125 --seed0 4001 --nr 2 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto
--times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on`: every file under the output
directory byte-identical except `timings.json`, `DONE` files included; (iii) an A/B run `lagrangian_medium_ab --grid 512,64,64
--ell 0.125 --eps 2 --seed0 4001 --nr 2`: every file byte-identical except `timings.json`, with the new exit files of the head listed
as allowed extras; (iv) full V100 `ctest` green (the new SF-41 entries included); `git diff 2cf48c1..H1` empty for
`tests/interpolation/periodic_tricubic_bspline_tests.cu`, `tests/interpolation/spline_hessian_tests.cu` and
`src/numerics/interpolation/PeriodicTricubicBSpline.{cuh,cu}`.

**RR10 (H).** Head A/B runs with the SF-38 RR5/RR6 commands (`--grid 2048,256,256 --ell 0.0390625 --eps 1 --seed0 4001 --nr 10` and
`--grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 10`). Per realization: the new exits file (per plane: particles that
crossed `x = 0`; per run: totals); `e2_b.csv` and `e2_a.csv` compared byte for byte with the committed SF-38 files of the same
realization. Where exits are non-zero: `ab_decision.py` (SF-38, unchanged) recomputes the `C_STAT` lines on the head outputs,
recorded beside the originals. Environment control per the failure policy.

**RR11 (exit codes; In scope 3).** Tracker: `exit_inlet = 7`, `exit_outlet = 8` (`PassStats::status`, `status_plane`/`status_time`
arrays); they are not counted in the 7-slot `status_hist`, but in new totals `exit_inlet`, `exit_outlet` per pass (and per plane /
per time in the non-periodic outputs). Pollock: `exit_inlet = 8`, `exit_outlet = 9` (16-slot histograms unchanged).

**RR12 (stop handling).** After FINAL_AUDIT: A, B, C, D or G not met -> corrective (BLOCKING) per the failure policy; E per its
policy; F not met at an amplitude -> State `blocked`, bitácora row naming the rule (as R1), PR for the owner with the stop in title and
body; otherwise `awaiting_review` with every recorded outcome (H exits, B decay factors, V3b classifications, measured-minus-emulated A
errors) reported explicitly.

## Implementation contracts

**C-SPLINE (non-periodic spline; In scope 2).** New files next to `PeriodicTricubicBSpline` (that file unchanged): a view
`{coeff, nx, ny, nz, hx, hy, hz, Lx, Ly, Lz}` with x-extended coefficients (`nx + 4` per `x` line: `c_{-2}..c_{nx+1}`), host/device
`evaluate_point` and `evaluate_point_hessian` returning a domain flag (no read when outside `[-h/2, L1 + h/2)`), Hessian path value and
gradient bitwise equal to the value path; device prefilter = not-a-knot `x` line solve (closure of the specification) and the SF-28
symbol in `y`/`z` by batched two-dimensional cuFFT; CPU mirror = the same `x` solve and cyclic tridiagonal `y`/`z` line solves; a
memory report (coefficients, spectrum, cuFFT work) as SF-28; no allocation during evaluation. `--fmad=false` for every translation unit
that instantiates it for the tracker.

**C-STAGE (lean medium (b) stage; In scope 1, 2).** `DirichletLongBoxStage` built phase by phase with `MEM_PHASE` lines (exact bytes =
sum of items + cuFFT work) and the SF-40 conventions (C-SYNC); solve = `solve_head`'s operator and right-hand side with `K` in the MG
level-0 coefficient, the right-hand side in a dead level-0 buffer, `z = Ap = level-0 b` through a new external-vector non-projected PCG
with `pcg_solve`'s arithmetic and order; `DirichletMedium`'s diagnostics and `rel` convention; head kept (device, or pinned mapped host
memory by option) and turned in place into `f`; `ln k` and `f` represented by C-SPLINE (fused interleaved array by option); host faces
by option; a solver record with the `B_SOLVER` fields.

**C-EXIT (In scope 3).** Non-periodic field functors (unfused and fused) report the domain; the tracker's passes turn a crossing of
`x1 < 0` (`exit_inlet`) or `x1 > L1` (`exit_outlet`) into the RR11 status without evaluating outside the domain; the periodic
instantiations keep their code paths (compile-time selection). Pollock: a non-periodic face view with `U` on `nx + 1` planes; a crossing
of the face `x = 0` moving upstream or `x = L1` ends the particle (RR11 codes); periodic mode unchanged.

**C-RUNNER-B (In scope 4).** `lagrangian_ensemble --medium b` with the defaults and outputs of the specification; medium (a) with one
`--hmax-div` value unchanged in bytes, file set and `DONE` signature; checkpoint/restart as SF-38 C-RUNNER.

**C-AB.** `lagrangian_medium_ab` medium (b) Pollock through the non-periodic view (from `DirichletMedium`'s host faces), exits written to a
new per-realization file; every pre-existing output format unchanged.
