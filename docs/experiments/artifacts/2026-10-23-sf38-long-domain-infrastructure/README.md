# SF-38 campaign tooling (V100 jobs)

Artifact of increment SF-38 (`docs/plans/active/lester-eq14/increments/SF-38-long-domain-infrastructure.md`). The normative
reading rules and contracts are `preregistration.md` (RR0-RR13, C-RUNNER, C-DIRICHLET, C-AB, C-PROD, C-PAR2); this directory
holds only the scripts that execute them. The scripts never run on a GPU locally: the orchestrator launches them as detached
`scripts/remote --increment SF-38 run <job> -- "..."` jobs, one at a time on the shared mirror (RR0, decision D16). **Output of
any `SF38_DRYRUN=1` run is NOT evidence** (tiny settings, separate roots, labelled in `run.env` and `results.txt`). The scripts
are modelled on the SF-40 campaign tooling (`../2026-10-22-sf40-lean-lagrangian-stage/`).

## Layout

| File | Role |
|---|---|
| `scripts/run_sf38.sh` | the campaign runner: `bash scripts/run_sf38.sh <build_dir> <part>` |
| `scripts/ab_decision.py` | threshold C (RR6): the A/B equivalence test from the two `ab_primary.csv`; `--selftest` |
| `scripts/b_verdict.py` | threshold B (RR2, RR3): `B_VERDICT` per production grid; `--selftest` |
| `scripts/c0_verdict.py` | threshold C0 (RR4): `C0_VERDICT` from the `B_SOLVER` line and the memory only; `--selftest` |
| `scripts/mem_verdict.py` | shared readers (RR2 gates, wall, RSS, `free -b`, `MEM_*` lines, `R=<r> ` prefix tolerant); `--selftest` |
| `scripts/d_checks.py` | threshold D (RR7): tree byte-compare, CSV compare, `TTIME` vs stats CSV, per-realization walls; `--selftest` |
| `scripts/g_table.py` | threshold G (RR11): raw variance / applied scale per seed and geometry; `--selftest` |
| `scripts/par2_pilot.py` | threshold E (RR8): dt/2 and K-stats config copies, `t_b`, field statistics, K-stats agreement, paired differences; `--selftest` |
| `scripts/pid_peak.py`, `scripts/json_strip_cmp.py`, `scripts/lines_cmp.py` | SF-40 tools, copied **byte-identical** (below) |
| `scripts/posthoc_ab.py` | N9, **post-hoc, not pre-registered** (added 2026-10-24 after FINAL_AUDIT at the owner's request): `python3 scripts/posthoc_ab.py raw analysis` writes `analysis/posthoc_ab.md` (P1 pairing efficiency, P2 point estimates vs margins, P3 paired boundary-layer profile, P4 mean-flow forcing and tilt, P5 memory of medium (b) at the production grid, P6 recorded Pollock non_ok / x-exit evidence) from `raw/sf38_campaign/ab/` and `prodgrid/B2.log` only; stdlib only, deterministic (byte-identical rerun); `--selftest`. It changes no verdict, threshold or other `analysis/` file; `note_tables.py` is unmodified and still regenerates its ten files byte-identically |

Provenance of the copied scripts (source `docs/experiments/artifacts/2026-10-22-sf40-lean-lagrangian-stage/scripts/`):

```
e88794ee3fd8615cafa42c5058e20d4430d40281de0066ebd39ac2ef75f87756  pid_peak.py
3f55dc1f92753170356992006ece7fe7acbaf007ac73bd06b2d131ea806fe654  json_strip_cmp.py
4f82a56e47d208fb1345952a9516acdf1779cfae1c2327fb417d1b6ad812a708  lines_cmp.py
```

`mem_verdict.py` is an adaptation of the SF-40 module of the same name (the `closed_form.py`-dependent S+ checks are dropped; the
log readers accept the runner's `R=<r> ` line prefix, which `pid_peak.py` does not; the PID falls back to the process tree
`RUN.pid_tree` when a log prints no `MEM_GEOM pid=`). Everything is stdlib python3 (no numpy) and bash 4.2-safe (the V100 host).
Selftests: `for f in scripts/*.py; do python3 $f --selftest; done` (all pass locally).

## Runner conventions

`<build_dir>` holds the executables of the part (HEAD parts: `build/v100-release` of the audited head; `baserefs`: the directory of the
BASE binaries, `$HOME/sf38_base_refs/bin`, built from the tree equal to `d8ffa7c`, job J0). `ctest` of `final` runs on `<build_dir>`.
The mirror (repository) root is derived from the script path.

Environment: `SF38_COMMIT` (recorded in `run.env`), `SF38_OUT_ROOT` (default `~/sf38_campaign`; dry `~/sf38_dryrun`),
`SF38_BASE_REFS` (default `~/sf38_base_refs`; dry `$SF38_OUT_ROOT/base_refs`), `SF38_DRYRUN=1`, `SF38_DIAG_PIPE`, `SF38_CTEST_ARGS`,
`SF38_AB_EXTRA_ARGS`, `SF38_AB_SKIP_C0_GUARD`, `PYTHON` (documented at the top of `run_sf38.sh`).

Per part, in `$SF38_OUT_ROOT/<part>/` (for `baserefs`, in `$SF38_BASE_REFS`): `run.env` (commit, binary md5s, host,
`CUDA_VISIBLE_DEVICES`, GPU name, driver, `nvidia-smi` banner, nvcc, gcc), `exit_codes.tsv` (`run<TAB>rc`, exactly two columns),
`walls.tsv` (`run<TAB>wall_s<TAB>end_utc`), `results.txt` (verdict, comparison and `RECORDED` lines), `sha256.txt` (sha256 of every
output file of each finished run and of the derived files), `resumed.txt`, and **`PART_EXIT`** (`0` iff no blocking item; absent when
the invocation was interrupted by a signal). The last line of `results.txt` is `RECORDED part_seconds=<this invocation>
part_runs_seconds=<sum of the run walls> part=<part>` (decision D16: GPU time of the part for the experiment note).
A run `R` writes `R/` (the runner's `--out` or the working directory), `R.log` (stdout+stderr), `R.time` (`/usr/bin/time -v`), `R.free`
(`free -b` before), and when sampled (`S`) `R.apps.csv`, `R.dev.csv`, `R.pid_tree`; with the exclusive-GPU precheck (`X`)
`R.precheck`.

RR2 sampler (parts `prodgrid`, `bpilot`, `ab`, `controls` smoke, `par2pilot`): `nvidia-smi --query-compute-apps=timestamp,pid,used_memory
--format=csv,noheader,nounits -lms 200` (all processes) filtered by the PID of the run, `nvidia-smi --query-gpu=timestamp,memory.used -i
<physical gpu> -lms 200` (device-wide), `/usr/bin/time -v`, `free -b` before, and a precheck that no compute process holds the GPU
(`GPU_BUSY`: the run is not started and no rc is recorded; the part exits non-zero). Gates: `>= 5` samples, PID peak `<=` device-wide
peak, the PID of the log (`MEM_GEOM pid=`) equal to the process-tree PID; a gate failing makes the memory item `NOT_EVALUABLE`
(re-run with identical settings, disclosed).

Resume (SF-37 RR12, as SF-40): a recorded rc = finished, never re-run; partial outputs without rc are removed and the run is re-run
with identical settings (listed in `resumed.txt`); nothing is re-run because of its result. Comparisons and verdicts are recomputed on
every invocation from the finished runs. The part's exit status is non-zero iff a blocking item failed. A `scripts/remote run` job
exports `CUDA_VISIBLE_DEVICES`; `GPU_ID` is its first element.

Dry run (`SF38_DRYRUN=1`, NOT evidence): every part has a tiny-settings mode (production grids -> 64x16x16 boxes, Par2 -> a 64x16x16 /
2400-step / 1000-particle copy of each config, R set reduced to 7 runs at 4 threads, full ctest -> one fast case, the exclusive-GPU precheck
skipped on a shared workstation); it exercises the plumbing (kill/restart, resume, verdict parsing, comparisons) before the V100 jobs.
Parts `bpilot` and `ab` need the A/B application (`lagrangian_medium_ab`, a separate node); their plumbing was exercised locally with a stub
that prints the documented `B_SOLVER` line and writes a synthetic `ab_primary.csv` (the stub is not part of the artifact).

## Parts

All commands below are `bash <A>/scripts/run_sf38.sh <build_dir> <part>`; `<A>` is this directory.

### `baserefs` (BASE binaries; job J0) - RR9 references
The 20-run R set exactly as SF-40 `r_items` (SF-37 RR13 commands: `--construct trace --threads 32` N 32/64 x eps 0.5/1.0 x host/gpu and
N 64 eps 2 gpu = 9 runs; `--production --n 32 --eps 0.25 --oracle-ladder` = 1; the seven `--track forward` cases of seed 3001 = 7; the
three `--comparator pollock --save-vd <dir>/vd --save-seed-velocity` cases = 3) with the base `inlet_slab` into `$SF38_BASE_REFS/r/`
(`r/{trace,fwd,neutral}/<name>`, `r/prod32`), and the SF-27 trio (`apps/config_pspta_small.yaml`,
`apps/config_streamfunctions_homogeneous.yaml`, `apps/config_streamfunctions_continuation.yaml`, each from an empty working directory)
with the base `macroflow3d_pipeline` into `$SF38_BASE_REFS/trio/<cfg>/`. Line: `BASEREFS_RECORDED r_runs= trio_runs=3
recorded_in_exit_codes= unexpected_exit_codes=` (`unexpected_exit_codes` must be 0; the expected exit codes are those of SF-40:
trace `0 17 19 21`, forward `0 18 22`, pollock `0 18 22 23`).

### `controls` (HEAD) - A and the fast tests
`long_box_tests --only A` (RR10) -> `A_VERDICT run=long_box_only_A exit= all_pass= checks_pass= checks_fail= reported=[<test line>] PASS|FAIL`
(from the test's own `A_VERDICT all_pass=<0|1> checks_pass=<n> checks_fail=<n>` line, exit 0, 0 failing checks);
`long_box_tests --full`, `dirichlet_medium_tests --full`, `dirichlet_medium_tests` (the `lagrangian_dirichlet_medium` ctest), and
`lagrangian_ensemble_tests <lagrangian_ensemble> <macroflow3d_pipeline>` (the `lagrangian_ensemble` ctest) ->
`CONTROL_VERDICT run= exit= checks_pass= checks_fail= all_pass= PASS|FAIL`; a runner smoke (`--grid 64,16,16 --ell 0.25 --eps 1 --seed0 4001
--nr 2 --np-side 10 --twin on --pollock-bridge on`) -> `SMOKE_VERDICT run=runner_smoke exit= status_ok=2/2 done_files=2/2 PASS|FAIL`; and a
sampler smoke (`--grid 256,32,32`, one realization, RR2 sampler) -> `SMOKE_VERDICT run=sampler_smoke ...` and `SAMPLER_SMOKE run=sampler_smoke
PID_PEAK ... OK|NOT_EVALUABLE`.

### `prodgrid` (HEAD) - B, RR2/RR3
Three runs, each `lagrangian_ensemble --grid <G> --eps <E> --ell <L> --seed0 4001 --nr 1 --x0 0 --window 0.1,0.9,0.1,0.9 --np-side 100 --planes auto
--times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on --out <dir>` under the RR2 sampler:
B1 `2048,256,256` eps 1 ell 0.0390625; B2 `3328,384,384` eps 1.5 ell 0.041666666666666664; B3 `3328,384,384` eps 2 ell 0.08333333333333333.
`b_verdict.py` writes, per run, `B_VERDICT run= grid= exit= correctors=<x:status,its,rel;y:...;z:...> sf19_converged= mg_levels=
stage_mg_levels= auto_levels= pid_peak_mib= limit_mib=30000 mem=OK|NOT_EVALUABLE(reasons) PASS|FAIL|NOT_EVALUABLE` and `B_ALL
PASS|FAIL|NOT_EVALUABLE`. PASS iff exit 0, the three corrector lines `converged` with `rel_residual <= 1e-10` and `SF-19 ... converged=1`,
`mg_levels` (SETUP and STAGE lines) equal to `box_auto_mg_levels(nx, ny, nz)` (Python copy of the C++ rule; 7 at both production
shapes), PID peak `<= 30000 MiB`. RECORDED: wall, host max RSS, `free -b`, every `MEM_GEOM`/`MEM_PHASE` line, `STATUS`, `TSTATUS`, `POLLOCK`,
`TWIN`, `BACKFLOW` lines, device-wide and PID peaks. A failing item is R3 (domain reduced by owner decision).

### `bpilot` (HEAD) - C0, RR4
`lagrangian_medium_ab --grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 1 --out <dir>` under the RR2 sampler. `c0_verdict.py`
reads only the solver record (the lines whose first token is `B_SOLVER`) and the memory; it never opens `ab_primary.csv` or any E1/E2
file and prints no E1/E2/primary value. `C0_VERDICT run= exit= solver_converged= iters= rel= balance= max_div= pid_peak_mib= limit_mib=30000
mem= mode=primary|fallback|failed PASS|PASS_FALLBACK|FAIL|NOT_EVALUABLE [reasons]`. PASS: converged, `rel <= 1e-10`, balance `<= 1e-8`,
PID peak `<= 30000`, exit 0. PASS_FALLBACK: `rel` in (1e-10, 1e-8] (or stalled with `rel <= 1e-8`), balance `<= 1e-8` (the A/B then uses
`1e-8`, recorded; `SF38_AB_EXTRA_ARGS` carries the application's tolerance option). `max_div` is recorded. The assumed grammar of the
`B_SOLVER` line (aliases accepted) is documented at the top of `c0_verdict.py`.

### `ab` (HEAD) - C, RR5/RR6
Refuses to start (`AB_REFUSED`, exit non-zero) unless `$SF38_OUT_ROOT/bpilot/results.txt` has a `C0_VERDICT ... PASS|PASS_FALLBACK` line
(failure policy: C0 not met -> stop before the A/B; `SF38_AB_SKIP_C0_GUARD=1` overrides). Then `lagrangian_medium_ab --grid 2048,256,256 --ell
0.0390625 --eps 1 --seed0 4001 --nr 10 --out <dir>/ab_s1` and `--grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 10 --out <dir>/ab_s4`
(RR2 sampler, recorded), then `ab_decision.py ab_s1/ab_primary.csv ab_s4/ab_primary.csv`, whose lines (`C_INPUT`, one `C_D` per statistic and
realization, one `C_STAT` per statistic with the estimate, SE, CI, margin and class, `C_AMP`) end with `C_VERDICT selected=a|b equivalent=k/10
non_equivalent= inconclusive=`. Also `RECORDED AB_DETERMINISM_4001 D_BYTES ...`: byte equality (excluding `timings.json`) of the pilot's
`realizations/r_0000` with the A/B's. The `ab_primary.csv` grammar (long form `seed,medium,S_T,S_tau,V1,theta_2,theta_3`, or wide
`<stat>_a/<stat>_b`) is documented at the top of `ab_decision.py`; exactly ten paired realizations per amplitude are required.

### `ensemble` (HEAD) - D, RR7
B3 grid (3328x384x384, eps 2, ell 32/384), production inputs as `prodgrid`, twin and bridge on, seeds 4001-4003. **D-A**: one uninterrupted run
(`--nr 3`). **D-B**: the same command; the runner process is sent SIGKILL as soon as the line `R=1 STAGE` is in its stdout (realization 0
done, realization 1 in progress), then restarted with the identical command in the same directory (`DB_run1`, `DB_run2`; both command lines are
`RECORDED D_INVOCATION`). **D-C**: `--seed0 4001 --nr 1`. `DA_analysis`: a copy of D-A reduced to `stats/` and `analysis_config.yaml` is
processed with `cd <copy> && macroflow3d_pipeline analysis_config.yaml`. Lines (`d_checks.py`): `D_KILL ... CONFIRMED|NOT_CONFIRMED`;
`D_BYTES name=DB_vs_DA ... IDENTICAL|DIFF` (every file except those named `timings.json`); `D_BYTES name=DC_vs_DA_r0000 ...` (only
`realizations/r_0000/**`, `stats/realization_0000_timeseries.csv`, `stats_twin/realization_0000_timeseries.csv`, `timings.json` excluded - the
run-level aggregates of a one-realization run differ by construction);
`D_CSV name=analysis_only_vs_in_process:<file> ... byte_equal= within= PASS|FAIL` for `analysis/macrodispersion.csv` and
`ensemble/ensemble_timeseries.csv` (each value `<= 1e-12` relative, absolute `<= 1e-300` where the reference is 0; byte equality reported);
`D_TTIME name=DA_stats_vs_TTIME ...` (the `stats/` CSVs parsed as `read_timeseries_csv` does against the `R=<r> TTIME` lines printed with
`%.12e`; the time column is compared at `1e-9` because the log prints it with `%.10g`); `RECORDED D_WALL` per realization from `timings.json`;
`D_VERDICT PASS|FAIL`.

### `fieldonly` (HEAD) - G, RR11
`lagrangian_ensemble --field-only --grid <G> --ell <L> --seed0 4001 --nr 10 --out <dir>/F_<B1|B2|B3>` at the three production geometries (eps
irrelevant), then `g_table.py`: `G_FIELD set=geom|ab name= grid= ell= seed= raw_mean= raw_variance= applied_scale= final_variance=` per seed,
`G_SUMMARY`, `G_COMPLETE name= expected=4001-4010 found= missing= OK|INCOMPLETE`, `G_VERDICT RECORDED_COMPLETE|INCOMPLETE`. The two A/B grids'
values are read from every `field.json` below `$SF38_OUT_ROOT/ab/ab_s1` and `ab_s4` when part `ab` has run (re-run this part after `ab` to
include them). Recorded only, no threshold.

### `par2pilot` (HEAD) - E, RR8 (`par2-pilot` is accepted)
Needs `macroflow3d_pipeline` and `par2_field_stats`. For `s1`, `s225`, `s4`: (1) `par2_field_stats <cfg>` (10 realizations) -> `E_LNK` (mean and
variance of `ln K` per realization) and `E_LNK_SUMMARY`; (2) `macroflow3d_pipeline apps/config_pipeline_par2_sf38_<c>.yaml` from its own working
directory (`run_<c>/`, N_R = 10, timeout 4 h, RR2 sampler recorded) -> `E_OUTPUTS` (10 `stats/` CSVs, `analysis/macrodispersion.csv`,
`ensemble/ensemble_timeseries.csv`, snapshots of every realization), `E_TB` per realization (first snapshot `snapshots/r_%04d/step_%08d.csv`
with some particle at `x >= Lx - 5 lambda`; `t_b = step dt`) and `E_TB_SUMMARY`, `RECORDED E_RTIME` (per-realization time from the mtimes of
the `stats/` CSVs; first interval includes setup) and the total wall / 10; the alpha_L(t) series files are copied unchanged to
`series/<c>_macrodispersion.csv` and `series/<c>_ensemble_timeseries.csv` (not interpreted); (3) the paired dt/2 check: a copy of the config
with `dt: 6.25`, `n_steps`, `snapshots.every`, `sample_every` and `output_every` doubled, `NR: 3` and its own output directory is run
(`dt2_<c>/`); the dt run's first three realizations are re-analysed with `run_mode: analysis_only` (`pair_<c>.in/`) and the difference of the two
`analysis/macrodispersion.csv` (joined on `t`) is `RECORDED E_PAIRED` and written to `series/<c>_paired_dt_minus_dt2.csv`; (4) the diagnostics build
`build/v100-diag` (`cmake -S . -B build/v100-diag -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
-DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON -DMACROFLOW3D_ENABLE_PETSC=OFF`, target `macroflow3d_pipeline`, bare `-j`) is built by the part when absent,
and runs each config with `n_steps: 1`, `NR: 10` (copies `ks_<c>/`) -> `E_KSTATS` per realization (string equality of the `%.4e` `K stats`
prints with the D17 tool's `Kmin/Kmax/Kmean`) and `E_KSTATS_ALL`. `E_VERDICT PASS|FAIL`: the three N_R = 10 runs exit 0 with outputs present.
Everything else is `RECORDED` (no threshold).

### `final` (HEAD build dir) - F
`ctest --test-dir <build_dir> --output-on-failure` -> `F_CTEST exit= tests= failed=0 wall_s= PASS|FAIL` and `ctest_times.tsv` (per-test
`name<TAB>status<TAB>seconds`; the single expected content change is `lagrangian_tracker_gpu_host`, D13); the SF-27 trio from empty working
directories vs `$SF38_BASE_REFS/trio/<cfg>` (SF-40 RR7 rule, manifest keys `timestamp` and `build.git_hash` dropped) -> `F_TRIO IDENTICAL|DIFF`;
the 20-run R set re-run with the head `inlet_slab` and compared with `$SF38_BASE_REFS/r` using the SF-40 `bytecmp` rules for the R items only
(every reference file `cmp`-identical, `*.json` by `json_strip_cmp.py` dropping `**.timings_s` (trace: also `transport.seconds_*`), the
`lines_cmp.py` token sets, exit code equal to the base `Exit status`) -> `F_RSET IDENTICAL|DIFF`. The orchestrator-local item
`git diff d8ffa7c..H1 -- 'apps/*.yaml'` (adds files only) is not scripted (the mirror has no git history).

## Job plan (RR0, RR1; one job at a time on the shared mirror)

1. J0 `sf38-base-refs` (before any SF-38 source reaches the mirror): build the base tree `d8ffa7c`, `bash <A>/scripts/run_sf38.sh
   $HOME/sf38_base_refs/bin baserefs` (the base binaries are `inlet_slab` and `macroflow3d_pipeline` in `$HOME/sf38_base_refs/bin`).
2. At H1 (integrated head): build `build/v100-release`; optionally `SF38_DRYRUN=1` of each part (NOT evidence); then, each
   `bash <A>/scripts/run_sf38.sh build/v100-release <part>` with `SF38_COMMIT=<head>`, in the order `controls`, `prodgrid`, `bpilot`, `ab`,
   `ensemble`, `fieldonly`, `par2pilot`, `final`.
   `par2pilot` needs the diagnostics build, which the part builds on first use (a long compile inside a detached job).

## Evidence import (N8, by the orchestrator)

Same scheme as SF-40: after the campaign, a short `scripts/remote --increment SF-38 exec` packs the text subset of
`~/sf38_campaign` and `~/sf38_base_refs` (excluding `bin/*`) into a tarball (`sf38_evidence.tar.gz`, sha256 recorded), which is extracted unchanged
into `raw/` (V100 relative layout: `raw/sf38_base_refs/`, `raw/sf38_campaign/<part>/`, `raw/job_logs/`). Included per part: `run.env`,
`exit_codes.tsv`, `walls.tsv`, `results.txt`, `sha256.txt`, `resumed.txt`, `PART_EXIT`, every `*.log`, `*.time`, `*.free`, `*.precheck`,
`*.pid_tree`, the sampler files `*.apps.csv` / `*.dev.csv`, the `*.json` summaries, `ab_primary.csv`, `ab_decision.out`, the `series/` files,
the Par2 `stats/`, `analysis/` and `ensemble/` CSVs and `cfg/*.yaml`. **Excluded from `raw/`** (listed with their sha256 in the per-part
`sha256.txt` manifests, which are the integrity record): every `*.npy` (Pollock and track per-particle arrays), the Par2 `snapshots/` trees
(~1000 files per config) and any other file larger than 2 MB (the list `sf38_evidence.excluded_large` is reported), the base binaries
`sf38_base_refs/bin/*`, and the R-set and trio bulk output directories of the base references (their sha256 are in the manifests). The
D byte-comparison verdicts (`D_BYTES`, `D_CSV`, `D_TTIME`) are in `results.txt`; the compared trees themselves stay on the host. Re-verification:
`for d in <part dirs>; do (cd raw/$d && sha256sum -c --ignore-missing sha256.txt | grep -vc ': OK'); done` gives 0 for each part.

Imported (orchestrator, 2026-10-23): `sf38_evidence.tar.gz` sha256 `3cc527af3297b57dfd7044cd58895af13c36559e54c10b00df2e3e9f7278957e`
(1129 files, none excluded as large; the pack excluded `*.npy`, `sf38_base_refs/bin/*`, `*/snapshots/*` and `*/output/*`), and the
addendum `sf38_par2_agg.tar.gz` sha256 `4519456bc938fbd642f1d0d54773d610a82ca10148afd000ae77a6f84075356f` (36 files): the Par2 runs'
`output/*/analysis/macrodispersion.csv`, `output/*/ensemble/ensemble_timeseries.csv`, `manifest.json`, `effective_config.yaml`, plus
`par2pilot/realization_series.sha256` (sha256 of the 69 per-realization `stats/` CSVs kept on the host). Transfer: base64 through
`scripts/remote --increment SF-38 exec`, sha256 equal on both hosts.

## Notes and readings (reported to the orchestrator)

* **dt/2 copies**: RR8 lists `dt = 6.25`, `n_steps` doubled, snapshot interval doubled, `NR = 3`, own output directory. The script also
  doubles `analysis.macrodispersion.sample_every` (and the cosmetic `transport.output_every`), because the time series is sampled every
  `sample_every` steps: without it the dt/2 series would be sampled at twice the time density and its `macrodispersion.csv` would not be
  comparable row by row with the dt run. This is a reading of RR8 (E has no threshold; the paired differences are RECORDED only).
* **`ab_primary.csv` and `B_SOLVER` grammars** are assumptions on the N4 application (documented in `ab_decision.py` / `c0_verdict.py`).
* `D-C` excludes `timings.json` as D-B does (RR7 names the exclusion for D-B and D-C jointly: "every output file of D-B and D-C ... excluding
  exactly the files named `timings.json`"); the TTIME time column is compared at `1e-9` (printed `%.10g`).
