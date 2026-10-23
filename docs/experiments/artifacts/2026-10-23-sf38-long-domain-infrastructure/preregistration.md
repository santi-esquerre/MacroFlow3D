# SF-38 pre-registration: reading rules and implementation contracts (activation commit, before any run)

- Increment: `docs/plans/active/lester-eq14/increments/SF-38-long-domain-infrastructure.md` (State `active` in the same
  commit). Base: `d8ffa7cad739cc8c61d870abbfb11cd748b314af` (`master`, PR #61 merged; owner decisions D1-D21 of
  `docs/decisions/2026-10-23-sf38-activation-decisions.md`). Written by the SF-38 orchestrator on 2026-10-23 after
  UNDERSTAND (code reconnaissance of every interface below) and an adversarial review of the owner options; committed in
  the activation commit, before any build, test or run.
- What this file is: the durable text of the activation reading rules (RR0-RR13) and implementation contracts (C-*) that
  the activation bitácora row cites. It changes no threshold, Goal, design value or failure-policy text of the
  specification; it resolves gaps inside the listed scope. The acceptance thresholds A, B, C0, C, D, E, F, G are the
  specification text, copied verbatim into the activation row.
- No `alpha_L` / `alpha_T` statement is made or allowed (Rule 1). Dispersion statistics produced by SF-38 runs (B, D, the
  Par2 pilot) are recorded, never interpreted; the only statistics read for a verdict are the C primary statistics.

## Reading rules

**RR0 (process).** No threshold is tuned after any run. Every V100 run is a detached orchestrator job
(`scripts/remote --increment SF-38 run`/`wait`), one job at a time on one GPU (decision D16), launched from a clean
worktree at an accepted commit and running committed scripts; workers never use the remote. Local validation is light
(builds only through `mf3d-build`, a few `ctest -R` cases under 60 s, grids `<= 32^3`-sized smokes). Dry runs are not
evidence. GPU-hours per part are reported in the experiment note (D16).

**RR1 (base and head).** Base binaries are built on the SF-38 mirror from a clean tree equal to `d8ffa7c` (job J0),
before any SF-38 source reaches the mirror; they produce the F regression references. Head = the integrated commit
H1; every head job runs on a tree equal to H1 (or to a later commit that differs from H1 only under `docs/`).

**RR2 (memory sampler; B, C0).** As SF-40: `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms 200` (all processes) filtered by the PID of the run (PID peak), plus the device-wide
`memory.used` of the physical GPU (`-i <n>`), `/usr/bin/time -v`, `free -b` before the run, and a precheck that no
compute process is on the GPU. The verdict uses the PID peak (`<= 30000 MiB`) and exit 0. Gates: `>= 5` samples, PID
peak `<=` device-wide peak.

**RR3 (B).** Runs with the SF-38 ensemble runner (C-RUNNER), seed 4001, one realization each, `sigma2 = 1` for SF-18
(unit-variance `Y`, `normalize_variance` on) and `k = exp(eps Y)`:
B1 `--grid 2048,256,256 --eps 1 --ell 10/256`; B2 `--grid 3328,384,384 --eps 1.5 --ell 16/384`;
B3 `--grid 3328,384,384 --eps 2 --ell 32/384`. Production track inputs (C-PROD): `X0 = 0`, window `[0.1, 0.9]^2`,
`M = 100`, `tol = 1e-8`, `h_max = h/8`, fused splines and single launch on; planes `x1 = m lambda` for
`m = 1, 2, 4, ...` while `m lambda <= L1`, plus `x1 = L1 - 10 lambda` (twin plane) and `x1 = L1`; times
`t_n = n lambda / 2`, `n = 1 ... ceil(2 L1 / lambda)` (box units; `qbar = 1`); twin series and Pollock bridge (plane and
time passes) on. "Converge to `rtol = 1e-10`": each of the three corrector lines `converged` with relative residual
`<= 1e-10`; "automatic hierarchy": `mg_levels` printed equal to `box_auto_mg_levels(nx, ny, nz)`.

**RR4 (C0).** The medium (b) solver pilot is run by the A/B application (C-AB) with seed 4001, `sigma^2 = 4`
(`eps = 2`, `ell = 32/256`), grid 3328x256x256, medium (a) then medium (b) on the same `K` in one process. Read for C0
only: the (b) solver line (`converged`, iterations, `||r||/||r_0||`), `Q(x=0) = h^2 sum_{j,k} U[0,j,k]`,
`Q(x=L1) = h^2 sum_{j,k} U[nx,j,k]`, balance `|Q(x=0) - Q(x=L1)| / |Q(x=0)|`, the maximum cell divergence
`max_c |sum of outward face velocities of c| / qbar` (dimensionless; recorded) and the PID peak of the process. Its E1/E2
outputs are not inspected; the A/B part recomputes all ten realizations, 4001 included, from scratch (byte equality of
the two 4001 outputs is recorded as a determinism check).

**RR5 (E1/E2).** As Implementation 3, with: `qbar` = mean of `U` over the `x`-face plane `i = 0` of the medium (for (a)
1 to PCG level); interior `x`-face planes `i` with `i h` in `[5 lambda, L1 - 5 lambda]`, interior cell layers `i` with
`(i + 1/2) h` in that interval; plane variances are population variances (divide by the face count); E2 from the Pollock
plane moments (flux-weighted, biased, over particles that reached the plane: `sigma_T^2 = (var_x2 + var_x3)/2`,
`var tau = var_t`); planes `X_m = X0 + m lambda`, `X0 = 5 lambda`, `m = 1, 2, 4, ...` while `X_m <= L1 - 5 lambda`;
`m*` = largest such `m`. Pollock plane pass only (empty time list), injection window `[0.1, 0.9]^2`, `M = 100`,
Pollock's own weights. (b): head drop `G1^(a) L1` with `G1^(a)` the `G[0]` of the (a) SF-19 solve of the same
realization (sign: `h(0) = G1 L1`, `h(L1) = 0`, so the mean head gradient is `-G1`, matching the mean potential
gradient of (a)), `mg_levels = 7`, `cg_rtol = 1e-10`.

**RR6 (C).** The decision is computed by the committed script `scripts/ab_decision.py` from the per-realization
primary statistics, exactly as the threshold text (delta-method CI for the ratio statistics, `t_{0.95,9} = 1.833`,
floor switch for `S_T`, paired-difference CI for the floor case and the tilts, both amplitudes), after all ten
realizations of both amplitudes exist. The script prints every statistic, `D_r`, CI and class, and the selection.

**RR7 (D).** Grid B3 (3328x384x384, `eps = 2`, `ell = 32/384`), production track inputs, seeds 4001-4003 (`N_R = 3`):
run D-A uninterrupted; run D-B the same command killed with SIGKILL after the `STAGE` line of the second realization
appears, then restarted with the same command; run D-C realization 4001 alone. Met iff every output file of D-B and
D-C equals D-A byte for byte, excluding exactly the files named `timings.json` and the job logs; and
`macroflow3d_pipeline` with `run_mode: analysis_only` on D-A's output directory (the analysis config written by the
runner) reproduces the runner's in-process `analysis/macrodispersion.csv` and its read series reproduce the in-process
moments, each value to `<= 1e-12` relative (absolute `<= 1e-300` where the reference is 0). Per-realization wall time
from `timings.json`.

**RR8 (E, D17).** The three configs run with the head `v100-release` `macroflow3d_pipeline` (`N_R = 10`). The paired
`dt/2` check: copies of each config made by the campaign script with `dt = 6.25`, `n_steps` doubled, snapshot interval
doubled, `NR = 3`, own output directory. `t_b` = `step dt` of the first snapshot in which some particle has
`x >= Lx - 5 lambda` (physical units). Realized mean and variance of `ln K` per realization from the D17 tool; its
cross-check with the pipeline's `K stats` line uses a diagnostics build of the same commit
(`-DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON`, `build/v100-diag`; the release preset does not print the line) running each
config with `n_steps = 1` and `NR = 10`; agreement at the printed precision (`%.4e`) is recorded.

**RR9 (F).** Regression set as SF-40 RR8 (the 20 SF-35/SF-36/SF-37 default-path runs: `--construct trace` x9,
`--production --n 32 --eps 0.25 --oracle-ladder` x1, the seven `--track forward` cases of seed 3001, the three
`--comparator pollock --save-vd --save-seed-velocity` cases) base vs head with the SF-37/SF-40 compare tools; SF-27
trio base vs head (manifest keys `timestamp` and `build.git_hash` dropped, as SF-40 RR7); full V100 `ctest` green with
the expected content change of `lagrangian_tracker_gpu_host` only (and the new SF-38 ctests); `git diff d8ffa7c..H1 --
'apps/*.yaml'` adds files only.

**RR10 (A).** Executed by `long_box_tests --only A` on the V100 (and in the full ctest at its fast sizes): F1 at
64x16x16 and 128x16x16 (`k = 1`, SF-36 A1 checks, `K_eff = I`, mean flux `e1`), F2 at 64x32x32 and 128x32x32 (layered,
amended A2), F3 at 64x16x16 and 128x16x16 (shift `+1`, `<= 1e-14`).

**RR11 (G).** SF-18 raw variance and applied scale are recorded from the `FIELD` lines for seeds 4001-4010 at the three
production geometries (field-only runs of the runner, `--field-only`) and at the two A/B grids (A/B runs).

**RR12 (seed-face assertion, D9).** For every seed, both transverse coordinates satisfy
`|n_T x - round(n_T x)| > 1e-6`; otherwise the runner and the A/B application exit non-zero before tracking.

**RR13 (disclosures).** Workers and the integrator run with the locally configured agent definitions (model `sonnet`,
effort `high`) instead of the checked-in `opus`/`medium` (as SF-40 RR12). Local builds only through `mf3d-build`.

## Implementation contracts

**C-RUNNER (ensemble runner, medium (a)).** Library `src/physics/particles/lagrangian_labels/EnsembleRunner.{cuh,cu}`,
executable `lagrangian_ensemble` (`apps/lagrangian_ensemble/`). One realization `r` (seed `seed0 + r`) is a pure
function of (command line, seed): `build_long_box_stage` (lean, `bhat` mapped, fused coefficients when
`--fused-splines`) -> `LagrangianTracker` prepare/track -> outputs. Output tree under `--out DIR` (io::OutputLayout):
`stats/realization_%04d_timeseries.csv` (flux-weighted `moments_time`, `CsvTimeSeriesWriter::write`),
`stats_twin/realization_%04d_timeseries.csv` (twin), `realizations/r_%04d/` with `write_track_outputs` files (flux
weighted), `twin/` (twin `moments_planes.csv`, `moments_time.csv`), `pollock/` (bridge: flux-weighted and twin plane and
time moments, per-particle arrays), `field.json` (SF-18 statistics), `timings.json` (all wall-clock values; nothing
time-dependent anywhere else), and `DONE` written last. Atomicity: a realization is built in `r_%04d.tmp/` and in
`*.tmp` stats files, then renamed; `DONE` after the renames; a restart skips realizations with `DONE` and deletes stale
`*.tmp`. After the last realization: in-process `compute_macrodispersion` over the flux-weighted series ->
`analysis/macrodispersion.csv`, `ensemble/ensemble_timeseries.csv`, and `analysis_config.yaml` (a pipeline config with
`run_mode: analysis_only`, `output_dir` = `DIR`, `macrodispersion.enabled: true`, `NR`, `lambda`, `vmean_norm: 1`).
Twin: a copy of the result with `weights = 1/N_p` for every particle, `finalize_track_result`. Bridge:
`PeriodicFaceFluxes::upload_and_repack` of the stage faces (`keep_face_fluxes`), `track_pollock_lattice` with the same
injection, planes and times; twin moments recomputed on the host from the per-particle arrays with uniform weights, by
the same reduction rules as the comparator. Seed-face assertion RR12. Options at least: `--grid`, `--sigma2`, `--ell`,
`--eps`, `--seed0`, `--nr`, `--x0`, `--window`, `--np-side`, `--planes`, `--times`, `--tol`, `--hmax-div`,
`--fused-splines`, `--single-launch`, `--twin on|off`, `--pollock-bridge on|off`, `--field-only`, `--out`. Prints the
SF-40 `FIELD`, `SETUP`, `MEM_GEOM`/`MEM_PHASE`, `TRACK`, `TPLANE`, `TTIME`, `TSTATUS`, `STATUS` lines per realization
(prefixed by `R=<r>`), and a `STAGE` line when the stage is built.

**C-DIRICHLET (medium (b) library).** `src/physics/particles/lagrangian_labels/DirichletMedium.{cuh,cu}`: from the host
unit-amplitude `Y` and `eps`, `K` is formed on the host exactly as `LongBoxStage` S1 (`lnK = eps * Y`, `K = exp(lnK)`),
uploaded; `solve_head` (PCG_MG, `mg_levels` explicit, `cg_rtol`, pin off) with Dirichlet `west = G1 L1`, `east = 0`,
periodic `y`/`z`, from `init_head_guess`; the `FlowWorkspace` is released before any face array is allocated;
`compute_velocity_from_head` into CompactMAC arrays; `Q(x=0)`, `Q(x=L1)`, balance, maximum cell divergence and `qbar`;
optional download of the faces to the host (E1) and device repack to the periodic Pollock layout
(`repack_compact_mac_to_periodic_faces`, outlet plane dropped). Every allocation listed in a per-phase memory line;
nothing in `src/physics/flow` changes. E1 plane statistics: a host function on CompactMAC arrays (RR5), shared by both
media.

**C-AB (A/B application).** Executable `lagrangian_medium_ab` (`apps/lagrangian_medium_ab/`), reusing the runner's
output/atomicity/restart helpers: per realization (seed) medium (a) (lean stage with `keep_face_fluxes` and
`keep_host_field`, E1, Pollock plane pass) then medium (b) (C-DIRICHLET on the same `Y`, E1, Pollock plane pass);
outputs per realization: E1 profiles and interior statistics per medium, E2 plane moments per medium, the (b) solver
record, primary statistics `S_T`, `S_tau`, `V1`, `theta_2`, `theta_3` per medium, `field.json`, `timings.json`, `DONE`.

**C-PAR2.** `apps/config_pipeline_par2_sf38_s{1,225,4}.yaml` exactly as the specification ("In scope", Par2 bullet;
`n_steps` 95749 / 116786 / 110887, snapshots every `floor(n_steps/100)`, `stochastic.seed: 7001`,
`analysis.macrodispersion.NR: 10`, `lambda` = `corr_length`, `vmean_norm` = `J`); D17 tool `par2_field_stats`
(`apps/par2_field_stats/`): reads a pipeline config with the existing loader, regenerates each realization's field with
`generate_K_field` exactly as `EnsembleRunner` (seed `seed + r`), prints per realization mean and variance of `ln K`
and `K` min/max/mean (`compute_field_stats`).
