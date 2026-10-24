# SF-38: long-domain infrastructure, medium A/B equivalence test, ensemble runner and Par2 baseline pilot

- Date: 2026-10-23 (first evidence job `sf38-j0a-base-build` attempt 2 started 16:22:53Z; last job `sf38-j1c-final` finished 22:08:19Z)
- Status: `measurement complete; every pre-registered part ran to completion; the verdict per criterion is in "Result"; the increment State is set by the orchestrator after FINAL_AUDIT and by human review, not by this note`
- Increment: [`SF-38-long-domain-infrastructure.md`](../plans/active/lester-eq14/increments/SF-38-long-domain-infrastructure.md)
- Decision record: [`2026-10-23-sf38-activation-decisions.md`](../decisions/2026-10-23-sf38-activation-decisions.md) (owner decisions D1-D21); parent records [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md) (items 7-9, stop rules R3, R4), [`2026-10-22-lean-lagrangian-stage.md`](../decisions/2026-10-22-lean-lagrangian-stage.md)
- Predecessor note (structure; the stage under test): [`2026-10-22-sf40-lean-lagrangian-stage.md`](2026-10-22-sf40-lean-lagrangian-stage.md)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) (Rule 1: `alpha_T` is neither accepted as physical by default nor presupposed zero; this note makes no `alpha_L` / `alpha_T` statement), [`beaudoin-de-dreuzy-2013-key-claims.md`](../theory/beaudoin-de-dreuzy-2013-key-claims.md) (the long, transversally periodic domain the infrastructure is built for)
- Artifacts: [`artifacts/2026-10-23-sf38-long-domain-infrastructure/`](artifacts/2026-10-23-sf38-long-domain-infrastructure/README.md)
  (`preregistration.md`, `scripts/`, `raw/`, `analysis/`; the post-hoc readings of the section "Post-hoc readings" are `scripts/posthoc_ab.py` and `analysis/posthoc_ab.md`)
- Commits: base `master d8ffa7cad739cc8c61d870abbfb11cd748b314af` (PR #61 merged); activation A0 `e4da02b63404613f362c05146f8dbb9c113fd704`
  (pre-registration `preregistration.md` committed there, sha256 `aca09f067e14d9461bae396bb25d40c8e2afceba90170785d0e88bd06337f0db`, re-computed at
  the base of this note and equal to the value in the activation row); **source head H1 `ddbfc6e5846ee4626cc3f7c966db6fbdc4583f8c`** (integrated nodes
  N1-N7; `SF38_COMMIT=ddbfc6e...` in every head `run.env`); every head-binary job ran on the mirror tree `14af3f9` (`TREE=14af3f9 H1=ddbfc6e` echoed by
  the job commands, `R:job_logs/sf38-j1a-head-build.log`, `R:job_logs/sf38-j1b-campaign.log`, `R:job_logs/sf38-j1c-final.log`); `git diff --stat ddbfc6e 14af3f9`
  changes one file, the SF-38 specification (checked by the author of this note). The base binaries were built by job J0a from tree `e4da02b`
  (`TREE=e4da02b BASE_EQUIV=d8ffa7c`, `R:job_logs/sf38-j0a-base-build.log`; the orchestrator's record is that `e4da02b` equals `d8ffa7c` in sources).
  Imported evidence commit `f68d2df`.
- Host/GPU (`R:sf38_campaign/controls/run.env`, identical in the eight head parts): `localhost.localdomain`, `Tesla V100-PCIE-32GB` (banner `32510MiB`),
  `CUDA_VISIBLE_DEVICES=0`, NVIDIA driver `470.239.06`, `nvidia-smi` banner CUDA 11.4, `nvcc` `11.4` (patch level not recorded in `run.env`), `gcc (GCC) 9.3.1 20200408`;
  host RAM `134590115840` B (`free -b`, e.g. `R:sf38_campaign/prodgrid/B2.free`). The `nvidia-smi` banner and sampler clocks are host-local time, three hours behind the
  UTC stamps (`nvidia-smi` banner `15:34:43` vs `DATE_UTC=2026-10-23T18:34:32Z`; derived).
- Binaries (`build/v100-release`, preset `v100-release`): base `inlet_slab` md5 `88526e87410416e2b5e91038e8406f76`, `macroflow3d_pipeline` md5 `f5f5204efff7c923f3e85e07e93cfda4`
  (`R:job_logs/sf38-j0a-base-build.log`, `R:sf38_base_refs/run.env`); head `inlet_slab` `51ba3cee3e439f81fee83dde302e44b0`, `macroflow3d_pipeline`
  `e18fc3034ee3034cc9ab7ba146c6da15`, `lagrangian_ensemble` `d413f038089d50fc26c14efe56573f13`, `lagrangian_medium_ab` `40e588df2b4fba9fc3b5a139cbdacbd7`,
  `par2_field_stats` `498ca561cdba7358ac018a285a9b9682`, `long_box_tests` `4e5df73b6ddedc0aaca274feb30a1a8b`, `dirichlet_medium_tests`
  `22eeb1ba1c84adc69937416731c92920`, `lagrangian_ensemble_tests` `c3462defebcfb045345033ebd698b0de` (`R:job_logs/sf38-j1a-head-build.log`, every head `run.env`);
  diagnostics-build `macroflow3d_pipeline` (`build/v100-diag`, only for the `K stats` cross-check) `f7c8c0cd382052d5b8a056dc28af56af`
  (`R:sf38_campaign/par2pilot/results.txt`). The final R-set item `IDENTICAL binary:md5_differs head=51ba3cee... base=88526e87...` confirms head != base build.

Abbreviations: `A` = `artifacts/2026-10-23-sf38-long-domain-infrastructure/`; `R:` = `A/raw/`; `ANA` = `A/analysis/` (`parts.md`, `B.md`, `B_medium.md`, `C.md`,
`C_D.md`, `E1_recorded.md`, `D.md`, `E.md`, `F.md`, `G.md`, and the post-hoc `posthoc_ab.md`, written by `A/scripts/posthoc_ab.py`, not by `note_tables.py`); `PRE` = `A/preregistration.md` (RR0-RR13, C-RUNNER, C-DIRICHLET, C-AB, C-PAR2); `SPEC` = the increment
specification; `BIT` = its bitacora rows. Every number below is copied from a cited committed file under `R:` or computed by `A/scripts/note_tables.py` from `raw/`
(outputs in `ANA`, regenerated byte-identically by `python3 A/scripts/note_tables.py A/raw A/analysis`; the tables below are verbatim copies of those outputs).
Labels: *measured* = read from a raw file; *derived* = arithmetic on measured values (formula given); *not in the imported evidence* = absent from `raw/`.
1 MiB = 1048576 B. Box units: the transverse period is 1, `lambda = ell`.

## Question

The increment's exact Goal: `Construir la infraestructura del dominio largo (stage no cúbico SF-18/SF-19/SF-28 en las geometrías de producción por amplitud, 2048x256x256 a lambda/h 10 para sigma^2 = 1 y 3328x384x384 a lambda/h 16 y 32 para sigma^2 = 2.25 y 4, con pico de memoria de dispositivo medido <= 30000 MiB; runner de ensemble con checkpoint por realización, serie gemela y puente Pollock para el Gate 5; baseline Par2 RWPT en el mismo dominio físico) y elegir entre el medio affine-periódico y el de Dirichlet en x con un test de equivalencia pre-registrado.`

Claims allowed (SPEC): (a) the non-cubic stage is correct on controls and fits the production grids; (b) the choice between medium (a) (SF-19 affine-periodic on the
long box) and medium (b) (`solve_head` with Dirichlet in `x`, periodic `y`/`z`) by the pre-registered equivalence test; (c) a reproducible, restartable ensemble
runner with the Gate 5 twin series and Pollock bridge; (d) the Par2 RWPT baseline configuration runs in the same physical domain (pilot ensemble). Not allowed: any
`alpha` estimate (SF-39), any change of SF-18/SF-19/SF-28/SF-32 numerics, tracker, Pollock or Par2 path.

## Hypothesis

Nature of the change (infrastructure plus one pre-registered decision test): the new code builds the long-box stage from existing
numerics and writes per-realization outputs; SPEC "Expected numerical effect": no existing output changes except `run_mode: analysis_only`, which no longer reads the
`CsvTimeSeriesWriter` header as a zero row. "Whether medium (b) has interior statistics equivalent to (a) within the margins is not predicted." Pre-registered
thresholds (SPEC "Acceptance thresholds", copied verbatim into the activation row of the bitacora, 2026-10-23T16:17Z; reading rules in `PRE`):

| criterion | pre-registered rule (abridged; the verbatim text is the activation row) |
|---|---|
| A controls | 64x16x16 and 128x16x16: `k = 1` reproduces SF-36 A1 (`<= 1e-12`; `tau`, `I` `<= 1e-9`), SF-19 gives `K_eff = I` and mean flux `e1`, shift `+1` in `x2`/`x3` `<= 1e-14`; 64x32x32 and 128x32x32 layered `k(x1)`: transverse positions and covectors `<= 1e-12`, `tau` `<= 1e-9` relative, `I` recorded (D21) |
| B production grids | seed 4001, B1 2048x256x256 `eps` 1 `ell` 10/256; B2 3328x384x384 `eps` 1.5 `ell` 16/384; B3 3328x384x384 `eps` 2 `ell` 32/384: three SF-19 correctors `converged` to `rtol = 1e-10`, automatic hierarchy; PID peak `<= 30000` MiB over field + stage + `1e4` particles over the full length + every production output (plane and time passes, twin, Pollock bridge); wall and RSS recorded |
| C0 (b) solver pilot | seed 4001, `sigma^2 = 4`, 3328x256x256: `solve_head` `mg_levels = 7` reaches `||r||/||r_0|| <= 1e-10`; flux balance `<= 1e-8`; max cell divergence recorded; PID peak of (b) `<= 30000` MiB; fallback `<= 1e-8` with balance met |
| C A/B equivalence | seeds 4001-4010, `sigma^2` 1 (2048x256x256) and 4 (3328x256x256); five primary statistics per amplitude (`S_T`, `S_tau`, `V1`, `theta_2`, `theta_3`), 90 % CI (`t_{0.95,9} = 1.833`), margins `ln 1.05`, `ln 1.025`, `ln 1.05`, floor `2.5e-3 lambda` for `S_T`, tilts `0.005`; equivalent / non-equivalent / inconclusive; (a) iff all ten equivalent, otherwise (b) (R4, D7) |
| D runner | rerun bitwise identical; killed-and-restarted job byte-identical (twin, bridge included); CSVs through the fixed `read_timeseries_csv` reproduce the in-process moments to `<= 1e-12` relative; per-realization wall recorded |
| E Par2 pilot | three configs, `N_R = 10` each, to completion with outputs preserved; time per realization, `alpha(t)` series, `t_b`, realized `ln K` variance, paired `dt/2` check recorded (no threshold) |
| F regression | SF-35/36/37 cube outputs byte-identical; full V100 `ctest` green (single expected change: `lagrangian_tracker_gpu_host`, D13); SF-27 trio byte-compare clean; no existing config changed |
| G recorded | SF-18 raw variance and applied scale, seeds 4001-4010, long box |

Failure policy (SPEC): A or D not met = defect (BLOCKING); B not met = R3 (domain reduced by owner decision); C0 not met = stop before the A/B (owner); C selects (b) = R4 (SF-41
non-periodic-in-`x1` spline increment specified and `done` before SF-39 activates); E fails = recorded, owner.

## Build / environment

- Base build: job `sf38-j0a-base-build` (attempt 2), `cmake --preset v100-release` then `cmake --build build/v100-release -j`, `[262/262]`, `BUILD_EXIT=0`, `ctest -N` `Total Tests: 43`
  (`R:job_logs/sf38-j0a-base-build.log`). Head build: job `sf38-j1a-head-build`, `[102/102]`, `BUILD_EXIT=0`, `ctest -N` `Total Tests: 46`
  (`R:job_logs/sf38-j1a-head-build.log`); the three extra entries are `lagrangian_dirichlet_medium`, `lagrangian_ensemble`, `lagrangian_medium_ab` (the diff of `CMakeLists.txt` between `d8ffa7c` and `ddbfc6e` adds exactly these three `add_test` lines, checked by the author of this note; the base `ctest -N` list itself is not in the imported evidence).
- Nothing else ran on the GPU during the sampled runs: each sampled run has a `.precheck` file `precheck: no compute process on GPU 0` (e.g.
  `R:sf38_campaign/prodgrid/B2.precheck`, `R:sf38_campaign/ensemble/DB_run1.precheck`); one job at a time on GPU 0 (decision D16). No `resumed.txt` exists in any part directory
  (`ls R:sf38_campaign/*/resumed.txt` finds none): nothing was re-run.
- Integrity: `sha256sum -c --ignore-missing sha256.txt` run inside each of the eight head part directories reports 0 failures (checked by the author of this note); the
  tarball `sf38_evidence.tar.gz` has sha256 `3cc527af3297b57dfd7044cd58895af13c36559e54c10b00df2e3e9f7278957e` and 1129 files (the file count of `raw/` re-counted: 1129);
  excluded by pattern: every `*.npy` (none in `raw/`), the base binaries, the Par2 snapshots and output bulk (their sha256 are in the per-part `sha256.txt`).
- Orchestrator checks (not in `raw/`, reported as received): `git diff --name-status d8ffa7c ddbfc6e -- 'apps/*.yaml'` lists three added files only; the frozen directories
  (`src/multigrid`, `src/numerics`, `src/physics/{stochastic,streamfunctions,flow}`, `src/physics/particles/{streamline_tracker,par2_adapter,pspta}`, `src/runtime`, `src/io`,
  `apps/inlet_slab`, `apps/closure_gate`) have an empty diff; no existing `lagrangian_labels` file is modified; the modified files are `CMakeLists.txt`, the SF-38 specification,
  `MacrodispersionAnalysis.hpp` (D13), `lagrangian_gpu_host_tests.cu` and `long_box_tests.cu`.
- Tooling (committed at H1): `A/scripts/run_sf38.sh` (parts `baserefs`, `controls`, `prodgrid`, `bpilot`, `ab`, `ensemble`, `fieldonly`, `par2pilot`, `final`), `ab_decision.py`, `b_verdict.py`,
  `c0_verdict.py`, `mem_verdict.py`, `d_checks.py`, `g_table.py`, `par2_pilot.py`, `pid_peak.py`, `json_strip_cmp.py`, `lines_cmp.py`, described in `A/README.md`.

## Config(s)

- **A (RR10)**: `long_box_tests --only A` (F1 `k = 1` at 64x16x16 and 128x16x16, `M = 16`; F2 layered at 64x32x32 and 128x32x32; F3 shift at 64x16x16 and 128x16x16, 4096 points).
- **B (RR3), verbatim** (`R:job_logs/sf38-j1b-campaign.log`, `+ [B1]` ... lines): `lagrangian_ensemble --grid <G> --eps <E> --ell <L> --seed0 4001 --nr 1 --x0 0 --window 0.1,0.9,0.1,0.9
  --np-side 100 --planes auto --times auto --tol 1e-8 --hmax-div 8 --fused-splines --single-launch --twin on --pollock-bridge on --out <dir>` with `(G, E, L)` = B1 `(2048,256,256; 1; 0.0390625)`,
  B2 `(3328,384,384; 1.5; 0.041666666666666664)`, B3 `(3328,384,384; 2; 0.08333333333333333)`. Resolved planes/times (`RESOLVED` line, e.g. B1: planes `0.0390625 ... 5, 7.609375, 8`, 410 times
  `n lambda/2`); `sigma2 = 1` unit-variance `Y` with `normalize_variance` (`FIELD` line).
- **C0 (RR4)**: `lagrangian_medium_ab --grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 1 --out <dir>`; **C (RR5-RR6)**: `lagrangian_medium_ab --grid 2048,256,256 --ell 0.0390625 --eps 1 --seed0 4001
  --nr 10` and `--grid 3328,256,256 --ell 0.125 --eps 2 --seed0 4001 --nr 10`, then `ab_decision.py` (`R:sf38_campaign/ab/ab_decision.out`). (b): `mg_levels_b = 7`, `cg_rtol_b = 1e-10`,
  `cg_max_iter_b = 2000`, `m_star = 64`, planes `X0 + m lambda` (`R:sf38_campaign/bpilot/bpilot.log` `RESOLVED` line; `R:sf38_campaign/ab/ab_s4/ab_manifest.json`).
- **D (RR7), verbatim** (`R:sf38_campaign/ensemble/results.txt` `RECORDED D_INVOCATION` lines): B3 inputs with `--nr 3` (D-A, D-B twice) and `--nr 1` (D-C), seeds 4001-4003; D-B run 1 is sent SIGKILL
  as soon as `^R=1 STAGE ` appears in its stdout; `DA_analysis`: `cd <copy of D-A reduced to stats/ and analysis_config.yaml> && macroflow3d_pipeline analysis_config.yaml`.
- **E (RR8, C-PAR2)**: `A/raw/sf38_campaign/par2pilot/cfg/{s1,s225,s4}.yaml` (copies of `apps/config_pipeline_par2_sf38_s{1,225,4}.yaml` as run): `s1` 2048x256x256, `dx` 5, `sigma2` 1,
  `corr_length` 50, `mg_levels` 6, `n_steps` 95749, `inject_x` 250; `s225` 2496x288x288, `sigma2` 2.25, `corr_length` 60, `mg_levels` 5, `n_steps` 116786, `inject_x` 300; `s4` 2496x288x288,
  `sigma2` 4, `corr_length` 120, `mg_levels` 5, `n_steps` 110887, `inject_x` 600; all `dt` 12.5, `n_particles` 10000, `stochastic.seed` 7001, `NR` 10, `snapshots.every` 957 / 1167 / 1108. The `dt/2`
  copies (`cfg/dt2_*.yaml`): `dt` 6.25, `n_steps` doubled (191498 for `s1`), `snapshots.every` and `sample_every` doubled, `NR` 3. The `K stats` copies (`cfg/ks_*.yaml`): `n_steps` 1, `NR` 10.
- **F (RR9)**: `ctest --test-dir build/v100-release --output-on-failure`; the SF-27 trio (`apps/config_pspta_small.yaml`, `apps/config_streamfunctions_homogeneous.yaml`,
  `apps/config_streamfunctions_continuation.yaml`) and the 20-run R set (commands: the `+ [<run>]` lines of `R:job_logs/sf38-j0b-baserefs.log` for the base, `R:job_logs/sf38-j1c-final.log` for the head) with base and head binaries.
- **G (RR11)**: `lagrangian_ensemble --field-only --grid <G> --ell <L> --seed0 4001 --nr 10 --out <dir>` at the three production geometries; the two A/B grids come from the A/B runs' `field.json`.

## Commands

All head jobs are detached `scripts/remote --increment SF-38 run <job> -- "..."` + `wait` calls, one at a time on the shared mirror `~/MacroFlow3D-SF-38` (the exact command strings are line 4 of each log in
`R:job_logs/`).

| id | job | window (UTC, 2026-10-23) | exit | what |
|---|---|---|---|---|
| J0a-1 | `sf38-j0a-base-build` attempt 1 | ~16:20 | `BUILD_EXIT=1` | ran after a failed sync (`Par2_Core` submodule not populated) on an empty mirror; cancelled; **not evidence** (bitacora row 2026-10-23T17:26Z; the attempt-1 log is not in the imported evidence) |
| J0a | `sf38-j0a-base-build` attempt 2 | 16:22:53-16:25:40 | 0 | base build (tree `e4da02b`), md5 of base `inlet_slab` / `macroflow3d_pipeline` copied to `~/sf38_base_refs/bin`, `ctest -N` 43 |
| J0b | `sf38-j0b-baserefs` | 18:21:01-18:23:59 | 0 | tree `76c3473`; part `baserefs`: 20-run R set + SF-27 trio with the base binaries; `BASEREFS_RECORDED r_runs=20 trio_runs=3 recorded_in_exit_codes=23 unexpected_exit_codes=0` |
| J1a | `sf38-j1a-head-build` | 18:33:17-18:33:38 | 0 | head build (tree `14af3f9`), md5 lines, `ctest -N` 46 |
| J1b | `sf38-j1b-campaign` | 18:34:31-21:13:12 | 0 | `for part in controls prodgrid bpilot ab ensemble fieldonly par2pilot`, each `SF38_COMMIT=ddbfc6e... bash $A/scripts/run_sf38.sh build/v100-release <part>`; every `PART_DONE ... rc=0` |
| J1c | `sf38-j1c-final` | 21:14:12-22:08:19 | 0 | part `final`: full `ctest`, trio, R set vs the base references |

Per part: start (`run.env` DATE_UTC), `PART_EXIT`, number of runs in `exit_codes.tsv`, non-zero rcs, `part_seconds` and `part_runs_seconds` (RECORDED lines of `results.txt`).

| part | start (UTC) | PART_EXIT | runs | non-zero rc | part_seconds | part_runs_seconds |
|---|---|---|---|---|---|---|
| baserefs | 2026-10-23T18:21:02Z | 0 | 23 | 21 | 177 | 164 |
| controls | 2026-10-23T18:34:32Z | 0 | 7 | - | 144 | 126 |
| prodgrid | 2026-10-23T18:36:56Z | 0 | 3 | - | 494 | 469 |
| bpilot | 2026-10-23T18:45:10Z | 0 | 1 | - | 170 | 155 |
| ab | 2026-10-23T18:48:00Z | 0 | 2 | - | 2627 | 2607 |
| ensemble | 2026-10-23T19:31:47Z | 0 | 5 | 9 | 1573 | 1546 |
| fieldonly | 2026-10-23T19:58:00Z | 0 | 3 | - | 38 | 19 |
| par2pilot | 2026-10-23T19:58:38Z | 0 | 16 | - | 4474 | 4215 |
| final | 2026-10-23T21:14:13Z | 0 | 24 | 21 | 3246 | 3231 |

derived: sum of `part_seconds` = 12943 s = 3.595 h; sum of `part_runs_seconds` = 12532 s = 3.481 h (GPU 0, one job at a time).

## Outputs inspected

### 0. Completeness and job outcomes

Job exits: J0a 0, J0b 0, J1a 0, J1b 0, J1c 0 (`finished exit=0` lines of each log in `R:job_logs/`). Every part's `PART_EXIT` is `0` (table above). The non-zero return codes in
`exit_codes.tsv` are deliberate: `baserefs` and `final` list the three trace cases `r_trace_N32_e1.0_host`, `r_trace_N32_e1.0_gpu`, `r_trace_N64_e2.0_gpu` with code `21`
(`STATUS roundtrip_exceeded`; the same codes in base and head, section F), and `ensemble` lists `DB_run1` with `9` (SIGKILL by the script, section D). No other non-zero code occurs.

### A. Controls on the long box (met)

`R:sf38_campaign/controls/long_box_only_A.log`; the part line is `A_VERDICT run=long_box_only_A exit=0 all_pass=1 checks_pass=30 checks_fail=0 ... PASS`
(`R:sf38_campaign/controls/results.txt`). Decisive values (measured):

| control | check | observed |
|---|---|---|
| F1 `k = 1`, 64x16x16 and 128x16x16 | `K_eff = I`; mean flux `e1`; transverse displacement `<= 1e-12`; `tau = X` relative; covectors; `|I| <= 1e-14` | all `max 0.000e+00` (both sizes, 2048 records each) |
| F2 layered, 64x32x32 | transverse positions constant `<= 1e-12`; covectors constant `<= 1e-12`; `tau` vs quadrature rel `<= 1e-9` | `4.614e-16`; `2.985e-14`; `4.771e-10` |
| F2 layered, 128x32x32 | same | `4.441e-16`; `2.962e-14`; `4.771e-10` |
| F2 recorded (no threshold) | `I` vs independent quadrature, both sizes | max abs err `2.046e-06`, max `|I_ref|` `3.208e-03` over 2048 records |
| F3 shift `+1` in `x2`, `x3` (4096 points), 64x16x16 / 128x16x16 | `<= 1e-14` | max abs `4.774e-15` (rel `1.770e-15`) / `6.384e-15` (rel `3.691e-15`), violating 0 |

The other control programs (`R:sf38_campaign/controls/results.txt`): `long_box_tests --full` 175 checks pass, 0 fail; `dirichlet_medium_tests --full` 26 / 0; `dirichlet_medium_tests` (fast) 24 / 0;
`lagrangian_ensemble_tests` (fast) 71 / 0; runner smoke (`--grid 64,16,16`, 2 realizations) `status_ok=2/2 done_files=2/2 PASS`; sampler smoke (`--grid 256,32,32`)
`status_ok=1/1 done_files=1/1 PASS`, `SAMPLER_SMOKE ... samples=17 pid_peak_mib=565 dev_peak_mib=569 status=OK`.

### B. Production grids (met at the three geometries)

B verdict lines (copied from `prodgrid/results.txt`): correctors are `its,rel_residual` for x;y;z.

| run | grid | exit | correctors (its,rel) x; y; z | mg_levels/auto | PID peak MiB | device-wide peak MiB | samples | wall s | host max RSS kB | TIMING r=0 s | B_VERDICT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 2048x256x256 | 0 | converged,20,3.294e-14; converged,20,2.592e-14; converged,20,2.804e-14 | 7/7 | 8257 | 8263 | 262 | 53.93 | 9021988 | 52.088 | PASS |
| B2 | 3328x384x384 | 0 | converged,30,8.367e-14; converged,30,6.495e-14; converged,30,7.528e-14 | 7/7 | 28885 | 28895 | 1022 | 209.11 | 31325308 | 207.139 | PASS |
| B3 | 3328x384x384 | 0 | converged,30,8.941e-12; converged,30,8.486e-12; converged,30,4.614e-12 | 7/7 | 28885 | 28895 | 976 | 200.71 | 31325124 | 198.898 | PASS |

Stage, backflow and status lines (copied from `B*.log`).

| run | STAGE status | stage s | inlet_vmin | labels_available | seed_backflow | backflow_enc plane | backflow_enc time | TSTATUS plane hist | TSTATUS time hist | TWIN planes/times | STATUS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | ok | 38.786 | 4.054640e-02 | 1 | 0 | 0 | 0 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 10/410 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |
| B2 | ok | 175.256 | 5.043988e-03 | 1 | 0 | 72 | 75 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 10/416 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |
| B3 | inlet_backflow | 166.465 | -4.297129e-02 | 0 | 0 | 1117 | 1011 | 10000,0,0,0,0,0,0 | 10000,0,0,0,0,0,0 | 9/208 | ok tracker_non_ok_plane=0 tracker_non_ok_time=0 seed_backflow=0 pollock_weighted_not_active=0 |

MEM_PHASE lines (copied from `B*.log`; MiB = bytes / 1048576, derived).

| run | phase | exact_bytes | exact MiB | device_used B | device_used MiB | seconds |
|---|---|---|---|---|---|---|
| B1 | F | 3223324680 | 3074.002 | 2666528768 | 2543.000 | 0.873793 |
| B1 | S1 | 5371854848 | 5123.000 | 3740270592 | 3567.000 | 4.241701 |
| B1 | S2 | 8146667625 | 7769.268 | 8664383488 | 8263.000 | 28.361181 |
| B1 | S3 | 6584535016 | 6279.502 | 4952424448 | 4723.000 | 4.307281 |
| B1 | T | 2418951048 | 2306.891 | 2937061376 | 2801.000 | 0.073570 |
| B2 | F | 11782326280 | 11236.502 | 8372879360 | 7985.000 | 3.595685 |
| B2 | S1 | 19636420608 | 18726.750 | 12298747904 | 11729.000 | 17.724351 |
| B2 | S2 | 29765452393 | 28386.547 | 30298603520 | 28895.000 | 128.284927 |
| B2 | S3 | 23775677416 | 22674.253 | 16440623104 | 15679.000 | 21.863574 |
| B2 | T | 8193545144 | 7813.974 | 8712617984 | 8309.000 | 0.083368 |
| B3 | F | 11782326280 | 11236.502 | 8372879360 | 7985.000 | 3.797751 |
| B3 | S1 | 19636420608 | 18726.750 | 12298747904 | 11729.000 | 17.907992 |
| B3 | S2 | 29765452393 | 28386.547 | 30298603520 | 28895.000 | 126.386810 |
| B3 | S3 | 23572512768 | 22480.500 | 16235102208 | 15483.000 | 15.751644 |
| B3 | T | 7923258824 | 7556.208 | 8439988224 | 8049.000 | 0.037765 |

derived: PID memory in the tracking and bridge phase. The sampler clock origin is the first sample of the PID (an approximation of the process start); `late` = samples more than 5 s after the stage build time `STAGE seconds`.

| run | stage s | PID peak MiB | samples within 1 MiB of the peak | late samples | late max MiB |
|---|---|---|---|---|---|
| B1 | 38.786 | 8257 | 140 | 44 | 4725 |
| B2 | 175.256 | 28885 | 632 | 134 | 15611 |
| B3 | 166.465 | 28885 | 622 | 135 | 15611 |

Decisive readings (measured unless marked): all three runs exit 0; the three SF-19 corrector lines are `converged` with relative residual `<= 1e-10` (largest `8.941e-12`, B3 corrector x), `mg_levels` printed equals
`auto_levels` (7); the PID peaks 8257 / 28885 / 28885 MiB are below 30000 MiB (margin at the two 3328 grids 30000 - 28885 = 1115 MiB, derived), each with `samples >= 5` and PID peak `<=` device-wide peak
(8257 <= 8263, 28885 <= 28895); the 28885 MiB equals the value SF-40 measured for the lean stage alone at this grid (`2026-10-22-sf40-lean-lagrangian-stage.md`, result (c)). The PID peak is reached in the stage
build (phase S2: exact 28386.547 MiB, device-wide `device_used` 28895 MiB at B2 and B3); the table "tracking and bridge phase" (derived from the sampler files) shows the later phases, including the Pollock bridge repack,
at most 15611 MiB (3328 grids) and 4725 MiB (B1). The twin series and the bridge ran in all three (`TWIN`, `POLLOCK` lines present; all `1e4` particles `ok` in the plane and time passes, `STATUS ok`).
**B3 disclosure**: the SF-19 flow of seed 4001 at `eps = 2` has a negative minimum inlet velocity (`inlet_vmin = -4.297129e-02`), so `STAGE status=inlet_backflow` and the D-1 inlet labels are unavailable
(`labels_available=0`; `Q0-1=nan` in the `SETUP inlet:` line); no seed started in backflow (`seed_backflow=0`), `backflow_enc` counts of 1117 (plane pass) and 1011 (time pass) particle encounters are recorded, and the run
completed with exit 0 and all particles `ok` (inlet backflow is "recorded, not an exit condition", `BACKFLOW` line). Realization 0 of D (same seed) shows the identical line (`R:sf38_campaign/ensemble/DA.log`).

### C0. Medium (b) solver pilot (met, primary mode)

`C0_VERDICT run=bpilot exit=0 solver_converged=1 iters=50 rel=9.705149e-11 balance=2.404573e-10 max_div=1.034116e-07 pid_peak_mib=19745 limit_mib=30000 mem=OK mode=primary PASS`
(`R:sf38_campaign/bpilot/results.txt`). `B_SOLVER`: `converged=1 iterations=50 initial=6.724494e+06 final=6.526221e-04 rel=9.705149e-11 qbar=1.002469787644e+00 Q_in=1.002469787644e+00
Q_out=1.002469787885e+00 balance=2.404573e-10 max_div=1.034116e-07 fallback=0` (`R:sf38_campaign/bpilot/bpilot.log`). The relative residual reaches `1e-10` with a margin of 3 % (derived: `9.705e-11 / 1e-10`);
the flux balance is `2.4e-10` against the threshold `1e-8`; the maximum cell divergence (recorded) is `1.034e-07`. Per RR4 this note reports only the solver record and the memory of the pilot, not its E1/E2 or primary values.
Wall `153.4 s`, host max RSS `15974088` kB, 752 sampler samples, PID peak 19745 MiB (device-wide 19749). The medium (b) memory phases are in the tables of section C (A/B runs, realization 0 of each).

### C. Medium A/B equivalence test (rule applied; medium (b) selected)

Inputs: ten paired realizations (seeds 4001-4010) per amplitude, `C_INPUT amp=1 ... n_pairs=10 lambda=0.0390625` and `amp=4 ... n_pairs=10 lambda=0.125` (`R:sf38_campaign/ab/results.txt`); both A/B runs exit 0
(`ab_s1`: wall 1084.17 s, host RSS 10073368 kB, PID peak 12337 MiB; `ab_s4`: wall 1520.1 s, host RSS 15977092 kB, PID peak 19745 MiB; 5367 and 7500 samples, `status=OK`). The C0 guard passed
(`RECORDED c0_guard=ok`). The decision was computed by the committed `ab_decision.py` exactly as the threshold text (RR6); the statistics are printed here as inputs of the rule only:

`C_STAT` lines of `ab/results.txt` (= `ab/ab_decision.out`). `est` is `R = ln(Pbar_a/Pbar_b)` for `test=ratio` and the paired difference `Dbar` for `test=diff|floor`; `margin` is the half-width on the same scale (`ln 1.05` = 0.04879, `ln 1.025` = 0.02469; floor margin = `2.5e-3 lambda`, box units). Statistics in box units (S_T) and normalized (S_tau, V1, theta). Values printed with 4 significant digits; full precision in the raw lines.

| amp | stat | test | Pbar_a | Pbar_b | est | se | ci_lo | ci_hi | margin | CI width / (2 margin) (derived) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S_T | floor | 3.198e-05 | 3.305e-05 | -1.08e-06 | 7.142e-06 | -1.417e-05 | 1.201e-05 | 9.766e-05 | 0.134 | equivalent |
| 1 | S_tau | ratio | 0.0819 | 0.08089 | 0.01246 | 0.01304 | -0.01144 | 0.03636 | 0.02469 | 0.968 | inconclusive |
| 1 | V1 | ratio | 0.6151 | 0.6151 | -0.0001057 | 2.322e-05 | -0.0001482 | -6.313e-05 | 0.04879 | 0.001 | equivalent |
| 1 | theta_2 | diff | 1.144e-05 | 9.932e-05 | -8.787e-05 | 0.0003878 | -0.0007988 | 0.000623 | 0.005 | 0.142 | equivalent |
| 1 | theta_3 | diff | 4.632e-05 | -2.102e-05 | 6.734e-05 | 0.0005173 | -0.0008808 | 0.001015 | 0.005 | 0.190 | equivalent |
| 4 | S_T | ratio | 0.01631 | 0.01739 | -0.06383 | 0.07906 | -0.2087 | 0.08109 | 0.04879 | 2.970 | inconclusive |
| 4 | S_tau | ratio | 2.38 | 2.21 | 0.07412 | 0.0829 | -0.07783 | 0.2261 | 0.02469 | 6.154 | inconclusive |
| 4 | V1 | ratio | 3.169 | 3.171 | -0.0004837 | 0.000316 | -0.001063 | 9.555e-05 | 0.04879 | 0.012 | equivalent |
| 4 | theta_2 | diff | 0.000318 | 0.006189 | -0.005872 | 0.003882 | -0.01299 | 0.001244 | 0.005 | 1.423 | inconclusive |
| 4 | theta_3 | diff | -0.0004618 | -0.003709 | 0.003248 | 0.005752 | -0.007295 | 0.01379 | 0.005 | 2.109 | inconclusive |

`C_AMP amp=1 equivalent=4/5 non_equivalent=0 inconclusive=1`
`C_AMP amp=4 equivalent=1/5 non_equivalent=0 inconclusive=4`

`C_VERDICT selected=b equivalent=5/10 non_equivalent=0 inconclusive=5`. How the verdict follows the rule: medium (a) is selected iff all ten primary statistics are equivalent; here five are equivalent (`S_T`, `V1`,
`theta_2`, `theta_3` at `sigma^2 = 1`; `V1` at `sigma^2 = 4`), **none is non-equivalent**, and five are inconclusive (`S_tau` at `sigma^2 = 1`; `S_T`, `S_tau`, `theta_2`, `theta_3` at `sigma^2 = 4`); the rule sends any inconclusive
statistic to medium (b) (owner decision D7: "a medium whose equivalence was not shown is not used"). The selection of (b) is therefore the pre-registered outcome of an equivalence test that did not show equivalence; it is **not**
a demonstrated difference between the media: in no statistic does the 90 % interval lie entirely outside its margin. Derived from the table: for `S_T`, `S_tau`, `theta_2`, `theta_3` at `sigma^2 = 4` the CI width exceeds the full
equivalence window `2 margin` (ratios 2.970, 6.154, 1.423, 2.109), so no estimate at `N_R = 10` could have been classified equivalent with those standard errors; for `S_tau` at `sigma^2 = 1` the ratio is 0.968 and the interval
(`[-0.01144, 0.03636]`) crosses the upper margin `0.02469`. Determinism of the pilot: `RECORDED AB_DETERMINISM_4001 D_BYTES name=AB_4001_vs_BPILOT files_ref=10 files_cand=10 identical=10 diff=0 missing=0 extra=0 excluded=1/1 IDENTICAL`
(the A/B recomputed realization 4001 from scratch and its outputs equal the pilot's byte for byte, `timings.json` excluded). Per-realization values `P_a`, `P_b`, `D_r` of all ten statistics and seeds (the rule's inputs): `ANA/C_D.md`
(100 rows, copied from the `C_D` lines of `R:sf38_campaign/ab/results.txt`).

Post-hoc remark (derived, not part of the rule; section "Post-hoc readings", P2): none is non-equivalent, but for three of the four inconclusive statistics at `sigma^2 = 4` the point estimate lies outside its margin
(`S_T` est -0.06383 against margin 0.04879; `S_tau` 0.07412 against 0.02469; `theta_2` -0.005872 against 0.005; the fourth, `theta_3`, has 0.003248 against 0.005), so the data lean neither way statistically and the selection of (b) is the rule's.

Medium (b) solver record (`B_SOLVER` lines of `ab/ab_s*.log`, one per realization; `rel` = ||r||/||r_0|| of `solve_head`, `balance` = |Q(x=0) - Q(x=L1)|/|Q(x=0)|, `max_div` = max cell divergence / qbar; recorded, thresholds C0 apply to the pilot only).

| amp | r | converged | iterations | rel | balance | max_div | qbar_b | fallback |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 20 | 3.409160e-11 | 2.346444e-11 | 3.599659e-09 | 1.000922970400e+00 | 0 |
| 1 | 1 | 1 | 20 | 4.852166e-11 | 8.885592e-11 | 3.262726e-09 | 1.000933410380e+00 | 0 |
| 1 | 2 | 1 | 30 | 3.862414e-15 | 1.481638e-13 | 1.057051e-09 | 1.001093660346e+00 | 0 |
| 1 | 3 | 1 | 30 | 3.327655e-15 | 1.002563e-13 | 4.708909e-10 | 1.001075650437e+00 | 0 |
| 1 | 4 | 1 | 30 | 2.370506e-14 | 1.871934e-13 | 6.667377e-10 | 1.001133855148e+00 | 0 |
| 1 | 5 | 1 | 30 | 2.318838e-14 | 1.124364e-13 | 7.321170e-10 | 1.001246906662e+00 | 0 |
| 1 | 6 | 1 | 30 | 6.976863e-15 | 8.871349e-15 | 4.911715e-10 | 1.001176292637e+00 | 0 |
| 1 | 7 | 1 | 30 | 3.027506e-14 | 1.246619e-13 | 7.046025e-10 | 1.001019706159e+00 | 0 |
| 1 | 8 | 1 | 30 | 1.039845e-14 | 4.701768e-14 | 5.940944e-10 | 1.001186202665e+00 | 0 |
| 1 | 9 | 1 | 30 | 5.737415e-15 | 8.740143e-14 | 4.633477e-10 | 1.000962729886e+00 | 0 |

derived: n=10, converged=10, fallback=0, max rel=4.852e-11, max balance=8.886e-11, max max_div=3.600e-09, iterations min..max=20..30.

| amp | r | converged | iterations | rel | balance | max_div | qbar_b | fallback |
|---|---|---|---|---|---|---|---|---|
| 4 | 0 | 1 | 50 | 9.705149e-11 | 2.404573e-10 | 1.034116e-07 | 1.002469787644e+00 | 0 |
| 4 | 1 | 1 | 60 | 3.758303e-11 | 2.864093e-11 | 8.504916e-08 | 1.004021516873e+00 | 0 |
| 4 | 2 | 1 | 50 | 8.934188e-11 | 6.510609e-12 | 7.133667e-08 | 1.006849077393e+00 | 0 |
| 4 | 3 | 1 | 50 | 1.228667e-11 | 1.022004e-11 | 7.533400e-08 | 1.005063049777e+00 | 0 |
| 4 | 4 | 1 | 60 | 6.333934e-12 | 2.411122e-12 | 1.085584e-07 | 1.007852806599e+00 | 0 |
| 4 | 5 | 1 | 50 | 1.639606e-11 | 1.032260e-11 | 2.657913e-08 | 1.013060417316e+00 | 0 |
| 4 | 6 | 1 | 50 | 4.659241e-11 | 6.759726e-11 | 5.687026e-08 | 1.011887705831e+00 | 0 |
| 4 | 7 | 1 | 50 | 1.151881e-12 | 1.376986e-12 | 2.859194e-08 | 1.003161837727e+00 | 0 |
| 4 | 8 | 1 | 50 | 3.489768e-11 | 5.803466e-12 | 1.226371e-07 | 1.005835258879e+00 | 0 |
| 4 | 9 | 1 | 60 | 2.875652e-12 | 4.286735e-13 | 4.003353e-08 | 1.007472469972e+00 | 0 |

derived: n=10, converged=10, fallback=0, max rel=9.705e-11, max balance=2.405e-10, max max_div=1.226e-07, iterations min..max=50..60.

Medium (b) memory phases, realization 0 (`MEM_PHASE_B` lines of `ab/ab_s*.log`; MiB derived).

| run | phase | exact_bytes | exact MiB | device_used B | device_used MiB | seconds |
|---|---|---|---|---|---|---|
| ab_s1 | K | 2147483648 | 2048.000 | 2664431616 | 2541.000 | 3.090000 |
| ab_s1 | solve | 12424724480 | 11849.141 | 12940476416 | 12341.000 | 5.395557 |
| ab_s1 | release | 2147483648 | 2048.000 | 2664431616 | 2541.000 | 0.011771 |
| ab_s1 | velocity | 5378678784 | 5129.508 | 5896142848 | 5623.000 | 0.029746 |
| ab_s1 | faces | 6451363840 | 6152.500 | 6969884672 | 6647.000 | 2.380657 |
| ab_s4 | K | 3489660928 | 3328.000 | 4006608896 | 3821.000 | 5.263301 |
| ab_s4 | solve | 20190177280 | 19254.854 | 20708327424 | 19749.000 | 17.462132 |
| ab_s4 | release | 3489660928 | 3328.000 | 4006608896 | 3821.000 | 0.017678 |
| ab_s4 | velocity | 8739364864 | 8334.508 | 9259974656 | 8831.000 | 0.044908 |
| ab_s4 | faces | 10483138560 | 9997.500 | 11004805120 | 10495.000 | 4.065902 |

Non-decisive E1 quantities (SPEC: "every E1/E2 statistic ... is recorded"; the full profiles are in `R:sf38_campaign/ab/ab_s*/realizations/r_*/e1_{a,b}.csv`, the E2 plane moments in `e2_{a,b}.csv`, the per-realization JSON in `e1.json`, `primary.json`, `status.json`):

Recorded E1 quantities from `ab/ab_s*/realizations/r_*/e1.json` and `e1_{a,b}.csv` (means over the ten realizations; `qbar` of (a) is the plane mean of `U` at `i=0`, 1 to solver level).

| amp | medium | qbar min | qbar max | qbar mean | <var_i(v2)>/qbar^2 mean | <var_i(v3)>/qbar^2 mean |
|---|---|---|---|---|---|---|
| 1 | a | 1.000000 | 1.000000 | 1.000000 | 0.1044 | 0.1046 |
| 1 | b | 1.000923 | 1.001247 | 1.001075 | 0.1044 | 0.1046 |
| 4 | a | 1.000000 | 1.000000 | 1.000000 | 0.8447 | 0.8052 |
| 4 | b | 1.002470 | 1.013060 | 1.006767 | 0.8458 | 0.8077 |

Ensemble-mean plane profile of `var_i(v1)/qbar^2` (west face, nearest plane to the listed distance; SE = sd over the ten realizations / sqrt(10)).

| amp | medium | x1/lambda | mean | SE |
|---|---|---|---|---|
| 1 | a | 0.00 | 0.6052 | 0.0175 |
| 1 | a | 0.20 | 0.6046 | 0.0200 |
| 1 | a | 0.50 | 0.6058 | 0.0229 |
| 1 | a | 1.00 | 0.6129 | 0.0236 |
| 1 | a | 2.00 | 0.6064 | 0.0178 |
| 1 | a | 4.00 | 0.6067 | 0.0180 |
| 1 | a | 8.00 | 0.6196 | 0.0200 |
| 1 | b | 0.00 | 0.7963 | 0.0358 |
| 1 | b | 0.20 | 0.7937 | 0.0357 |
| 1 | b | 0.50 | 0.7709 | 0.0345 |
| 1 | b | 1.00 | 0.7163 | 0.0302 |
| 1 | b | 2.00 | 0.6402 | 0.0205 |
| 1 | b | 4.00 | 0.6117 | 0.0178 |
| 1 | b | 8.00 | 0.6209 | 0.0200 |
| 4 | a | 0.00 | 2.5639 | 0.2553 |
| 4 | a | 0.25 | 2.5461 | 0.2132 |
| 4 | a | 0.50 | 2.5573 | 0.1812 |
| 4 | a | 1.00 | 2.5027 | 0.1710 |
| 4 | a | 2.00 | 2.7118 | 0.2405 |
| 4 | a | 4.00 | 3.5182 | 0.6585 |
| 4 | a | 8.00 | 4.1019 | 0.6570 |
| 4 | b | 0.00 | 4.5315 | 0.6989 |
| 4 | b | 0.25 | 4.4495 | 0.6651 |
| 4 | b | 0.50 | 4.1180 | 0.5817 |
| 4 | b | 1.00 | 3.3669 | 0.4009 |
| 4 | b | 2.00 | 2.9654 | 0.2898 |
| 4 | b | 4.00 | 3.5808 | 0.6731 |
| 4 | b | 8.00 | 4.1139 | 0.6615 |

derived (literal reading of the specification's recorded boundary-layer quantity; not scripted by the campaign tooling): `M` = mean over the interior planes `x1 in [5 lambda, L1 - 5 lambda]` of the ensemble-mean profile; a plane is `outside` if `|mean_i - M| > 2 SE_i`; `west d` (`east d`) = the largest distance from the west (east) face of an `outside` plane in the west (east) half, in `lambda`. The same procedure is applied to medium (a), which has no `x`-face condition, as a control of the estimator.

| amp | medium | M | interior planes | interior planes outside | west d/lambda | east d/lambda |
|---|---|---|---|---|---|---|
| 1 | a | 0.6151 | 1949 | 115 | 99.60 | 79.30 |
| 1 | b | 0.6151 | 1949 | 116 | 99.60 | 79.30 |
| 4 | a | 3.1693 | 3009 | 365 | 42.38 | 47.22 |
| 4 | b | 3.1708 | 3009 | 365 | 42.34 | 47.22 |

The boundary-layer thickness of (b) is a recorded quantity of SPEC threshold C that no campaign script computes; the last table is a literal derived reading. Applied literally, the criterion "stays within 2 SE of the interior mean" is
dominated by plane-to-plane sampling noise: medium (a), which has no condition at the `x` faces, gets the same order of "thickness" as (b) (and 5.9 % of its interior planes lie outside 2 SE at `sigma^2 = 1`, derived `115/1949`), so
**no boundary-layer thickness is claimed**. The profile table shows what was recorded near the west face (ensemble mean of `var_i(v1)/qbar^2` against distance), with its SE; it is not interpreted here. The informative reading of the boundary layer is the paired one (post-hoc, P3 of the section "Post-hoc readings"): the difference (b) minus (a) of `var_i(v1)/qbar^2` on the same field, which removes the field's own plane-to-plane fluctuation that dominates the literal reading above.

### D. Ensemble runner (met)

Lines of `ensemble/results.txt` (verbatim, paths shortened where marked `...`).

```
D_KILL run=DB_run1 rc=9 killed_by_script=1 r0_status_ok=1 r1_stage_line=1 r1_status_line=0 CONFIRMED
D_BYTES name=DB_vs_DA files_ref=99 files_cand=99 identical=99 diff=0 missing=0 extra=0 excluded=3/3 IDENTICAL
D_BYTES name=DC_vs_DA_r0000 files_ref=32 files_cand=32 identical=32 diff=0 missing=0 extra=0 excluded=3/1 IDENTICAL
D_CSV name=analysis_only_vs_in_process:analysis/macrodispersion.csv ref=... cand=... rows=208/208 cols=7 max_rel=0.000e+00 max_abs_at_zero=0.000e+00 byte_equal=1 within=1 PASS
D_CSV name=analysis_only_vs_in_process:ensemble/ensemble_timeseries.csv ref=... cand=... rows=208/208 cols=8 max_rel=0.000e+00 max_abs_at_zero=0.000e+00 byte_equal=1 within=1 PASS
D_TTIME name=DA_stats_vs_TTIME realizations=3 rows=624 max_rel_moments=4.859e-13 max_rel_t=3.200e-10 active_equal=1 PASS
D_VERDICT PASS (D-A, D-B restart, D-C, analysis_only reproduction all met)
RECORDED run=DA exit=0 wall_s=589 host_max_rss_kb=31355600
RECORDED run=DB_run2 exit=0 wall_s=394 host_max_rss_kb=31355300
RECORDED run=DC exit=0 wall_s=201 host_max_rss_kb=31325176
```

Per-realization wall (`RECORDED D_WALL`, seconds, from `timings.json`).

| run | r | total | stage | prepare | track | bridge_repack | bridge_track | write |
|---|---|---|---|---|---|---|---|---|
| DA | 0 | 196.468 | 164.822 | 0.038 | 20.989 | 8.816 | 0.283 | 0.165 |
| DA | 1 | 194.111 | 161.502 | 0.035 | 20.551 | 10.117 | 0.284 | 0.195 |
| DA | 2 | 194.881 | 166.631 | 0.035 | 20.539 | 6.158 | 0.285 | 0.164 |
| DB | 0 | 194.839 | 164.518 | 0.037 | 21.012 | 7.547 | 0.273 | 0.177 |
| DB | 1 | 195.749 | 167.530 | 0.036 | 20.616 | 6.039 | 0.279 | 0.165 |
| DB | 2 | 194.713 | 161.602 | 0.035 | 20.628 | 10.672 | 0.293 | 0.165 |
| DC | 0 | 197.617 | 165.687 | 0.049 | 21.051 | 9.105 | 0.280 | 0.179 |

keys of `timings.json` of D-A r=0 (imported file): commit, mem_lines, pollock_timings_s, r, seconds, seed, stage_status, tracker_timings_s

Reading (measured): D-B (killed after `R=1 STAGE`, `D_KILL ... r0_status_ok=1 r1_stage_line=1 r1_status_line=0 CONFIRMED`, rc 9, wall 362 s; restarted with the identical command, which printed `R=0 SKIP done (DONE present)`)
equals D-A on 99 of 99 files; D-C (realization 4001 alone, per the reading clarification of the bitacora row 2026-10-23T17:26Z) equals D-A's realization 0 on 32 of 32 files; the `analysis_only` reproduction of the in-process
`analysis/macrodispersion.csv` (208 rows, 7 columns) and `ensemble/ensemble_timeseries.csv` (208 rows, 8 columns) is byte-equal (`max_rel=0.000e+00`); the printed `TTIME` moments agree with the stats CSVs to
`max_rel_moments=4.859e-13` (`<= 1e-12`) and the time column to `max_rel_t=3.200e-10` (the log prints it with `%.10g`, compared at `1e-9`). Per-realization wall time (D-A) 194.1-196.5 s, of which stage 161.5-166.6 s,
tracking 20.5-21.0 s, bridge repack 6.2-10.1 s (derived ranges from the table); host RSS `31355600` kB (D-A). B-run walls for the same grid: 198.9 s (B3, `TIMING r=0`).

### E. Par2 baseline pilot (completed; recorded)

`E_VERDICT PASS (the three N_R = 10 runs exit 0 with outputs present)` (`R:sf38_campaign/par2pilot/results.txt`); `E_OUTPUTS cfg=<c> stats=10/10 macrodispersion=1 ensemble=1 snapshot_files=1010
realizations_with_snapshots=10/10 ok=1` for the three configs; `E_KSTATS_ALL cfg=<c> agree=10/10` for the three (the D17 tool's `ln K` / `K` statistics agree with the pipeline's `K stats` line at the printed `%.4e` precision in all 30
realizations).

Par2 pilot runs (`par2pilot/results.txt`; wall = `/usr/bin/time` elapsed of the whole 10-realization run; time per realization from the mtimes of `stats/` CSVs, first interval includes setup; `t_b` in physical time units (`step * dt`, `dt = 12.5`); `dt/2` wall for 3 realizations).

| cfg | exit | wall s | interval s r=1..9 (min..max) | interval s r=0 | t_b min | t_b max | PID peak MiB | host max RSS kB | dt/2 run wall s | K-stats agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 0 | 334 | 32.700..32.779 | 37.812 | 526350 | 598125 | 16495 | 630604 | 110 | cfg=s1 agree=10/10 pipeline_lines=10 |
| s225 | 0 | 571 | 56.332..56.378 | 61.651 | 379275 | 495975 | 25169 | 632152 | 182 | cfg=s225 agree=10/10 pipeline_lines=10 |
| s4 | 0 | 622 | 61.463..61.615 | 66.698 | 180050 | 235450 | 25169 | 631688 | 196 | cfg=s4 agree=10/10 pipeline_lines=10 |

`E_LNK` per realization (realized mean and variance of `ln K` from `par2_field_stats`, K seeds 7001-7010) and `E_TB`.

| cfg | r | K seed | mean lnK | var lnK | first snapshot step with x >= Lx - 5 lambda | t_b |
|---|---|---|---|---|---|---|
| s1 | 0 | 7001 | -8.066403e-04 | 9.925601e-01 | 47850 | 598125 |
| s1 | 1 | 7002 | 2.522975e-03 | 1.010451e+00 | 47850 | 598125 |
| s1 | 2 | 7003 | -2.753560e-03 | 1.020288e+00 | 47850 | 598125 |
| s1 | 3 | 7004 | 9.215458e-03 | 1.015348e+00 | 42108 | 526350 |
| s1 | 4 | 7005 | 1.642264e-03 | 1.001317e+00 | 45936 | 574200 |
| s1 | 5 | 7006 | -1.017104e-02 | 1.004237e+00 | 46893 | 586162.5 |
| s1 | 6 | 7007 | 4.786690e-03 | 9.668611e-01 | 45936 | 574200 |
| s1 | 7 | 7008 | 1.649756e-02 | 9.852075e-01 | 43065 | 538312.5 |
| s1 | 8 | 7009 | 2.905975e-03 | 9.942607e-01 | 46893 | 586162.5 |
| s1 | 9 | 7010 | 2.408322e-02 | 9.862952e-01 | 44979 | 562237.5 |
| s225 | 0 | 7001 | -2.110130e-03 | 2.229931e+00 | 37344 | 466800 |
| s225 | 1 | 7002 | 6.009870e-03 | 2.272477e+00 | 38511 | 481387.5 |
| s225 | 2 | 7003 | -1.519657e-03 | 2.300654e+00 | 39678 | 495975 |
| s225 | 3 | 7004 | 1.366578e-02 | 2.283341e+00 | 30342 | 379275 |
| s225 | 4 | 7005 | 6.288372e-03 | 2.256281e+00 | 37344 | 466800 |
| s225 | 5 | 7006 | -1.774260e-02 | 2.257440e+00 | 39678 | 495975 |
| s225 | 6 | 7007 | 4.172413e-03 | 2.176069e+00 | 36177 | 452212.5 |
| s225 | 7 | 7008 | 2.453278e-02 | 2.214025e+00 | 33843 | 423037.5 |
| s225 | 8 | 7009 | 3.316899e-03 | 2.235446e+00 | 38511 | 481387.5 |
| s225 | 9 | 7010 | 4.335106e-02 | 2.216597e+00 | 36177 | 452212.5 |
| s4 | 0 | 7001 | 7.016872e-03 | 3.993288e+00 | 14404 | 180050 |
| s4 | 1 | 7002 | 1.885182e-02 | 4.101289e+00 | 15512 | 193900 |
| s4 | 2 | 7003 | 2.163832e-02 | 3.928941e+00 | 18836 | 235450 |
| s4 | 3 | 7004 | 9.339072e-02 | 4.185435e+00 | 15512 | 193900 |
| s4 | 4 | 7005 | -2.868004e-02 | 3.989543e+00 | 15512 | 193900 |
| s4 | 5 | 7006 | 2.633312e-02 | 3.955639e+00 | 17728 | 221600 |
| s4 | 6 | 7007 | 9.969182e-03 | 3.881279e+00 | 14404 | 180050 |
| s4 | 7 | 7008 | 7.698206e-02 | 3.884000e+00 | 15512 | 193900 |
| s4 | 8 | 7009 | 7.543238e-02 | 3.948124e+00 | 15512 | 193900 |
| s4 | 9 | 7010 | 7.668080e-02 | 3.935226e+00 | 16620 | 207750 |

`E_LNK_SUMMARY cfg=s1 n=10 mean_lnK_mean=4.792289815381e-03 var_lnK_mean=9.976824684395e-01 var_lnK_sd=1.603975706379e-02`
`E_LNK_SUMMARY cfg=s225 n=10 mean_lnK_mean=7.996478183577e-03 var_lnK_mean=2.244226014771e+00 var_lnK_sd=3.717008140922e-02`
`E_LNK_SUMMARY cfg=s4 n=10 mean_lnK_mean=3.776152426749e-02 var_lnK_mean=3.980276505486e+00 var_lnK_sd=9.566401528973e-02`

Paired `dt` vs `dt/2` differences (`RECORDED E_PAIRED`, first 3 realizations; difference of the two `analysis/macrodispersion.csv` joined on `t`; column names are the pipeline's; these are numerical differences between two time steps, recorded without threshold, not coefficient estimates).

| cfg | column | rows | unmatched | max_abs_diff | max_rel_diff | rms_diff |
|---|---|---|---|---|---|---|
| s1 | alpha_x | 9574 | 0 | 2.697648e-01 | 2.754778e+03 | 5.853068e-02 |
| s1 | alpha_y | 9574 | 0 | 6.086619e-02 | 4.876252e+03 | 1.269577e-02 |
| s1 | alpha_z | 9574 | 0 | 6.000865e-02 | 3.355047e+03 | 1.335137e-02 |
| s1 | dvar_dt_x | 9574 | 0 | 2.634421e-01 | 2.754778e+03 | 5.715886e-02 |
| s1 | dvar_dt_y | 9574 | 0 | 5.943964e-02 | 4.876252e+03 | 1.239822e-02 |
| s1 | dvar_dt_z | 9574 | 0 | 5.860220e-02 | 3.355047e+03 | 1.303845e-02 |
| s225 | alpha_x | 11678 | 0 | 1.227682e+00 | 8.102349e+00 | 2.324526e-01 |
| s225 | alpha_y | 11678 | 0 | 1.557756e-01 | 5.111451e+03 | 3.510236e-02 |
| s225 | alpha_z | 11678 | 0 | 1.996946e-01 | 3.676275e+03 | 3.356462e-02 |
| s225 | dvar_dt_x | 11678 | 0 | 1.438690e+00 | 8.102349e+00 | 2.724054e-01 |
| s225 | dvar_dt_y | 11678 | 0 | 1.825495e-01 | 5.111451e+03 | 4.113558e-02 |
| s225 | dvar_dt_z | 11678 | 0 | 2.340171e-01 | 3.676275e+03 | 3.933354e-02 |
| s4 | alpha_x | 11088 | 0 | 1.898548e+00 | 4.060116e+01 | 3.282713e-01 |
| s4 | alpha_y | 11088 | 0 | 2.019993e-01 | 1.768714e+03 | 3.597006e-02 |
| s4 | alpha_z | 11088 | 0 | 1.883933e-01 | 8.957962e+04 | 3.319669e-02 |
| s4 | dvar_dt_x | 11088 | 0 | 4.449721e+00 | 4.060116e+01 | 7.693859e-01 |
| s4 | dvar_dt_y | 11088 | 0 | 4.734358e-01 | 1.768714e+03 | 8.430483e-02 |
| s4 | dvar_dt_z | 11088 | 0 | 4.415468e-01 | 8.957962e+04 | 7.780474e-02 |

Notes (measured): the realized variance of `ln K` averaged over the ten realizations is 0.9977 / 2.2442 / 3.9803 (sd 0.0160 / 0.0372 / 0.0957) for `s1` / `s225` / `s4`, against the nominal `sigma2` 1 / 2.25 / 4 (Par2's generator is not
variance-normalized, recorded). The `alpha(t)` series (`analysis/macrodispersion.csv`, `ensemble/ensemble_timeseries.csv` and the ten `stats/` CSVs of each config) and the Par2 snapshots are preserved on the V100 host under
`~/sf38_campaign/par2pilot/run_<c>/output/sf38_par2_<c>/`. The aggregate series (`analysis/macrodispersion.csv`, `ensemble/ensemble_timeseries.csv`), `manifest.json` and `effective_config.yaml` of every run
(`run_*`, `dt2_*`, `ks_*`) were imported afterwards by the orchestrator (evidence addendum `sf38_par2_agg.tar.gz`, sha256 `4519456bc938fbd642f1d0d54773d610a82ca10148afd000ae77a6f84075356f`, 36 files; e.g.
`R:sf38_campaign/par2pilot/run_s1/output/sf38_par2_s1/analysis/macrodispersion.csv`, sha256 `ea4f86e1add7a1da2d3774fb81132212ffa5278c525081626bafb49fc60df7bd`, equal to its line in `R:sf38_campaign/par2pilot/sha256.txt`);
the per-realization `stats/` CSVs (69 files, 67 MB) and the snapshots stay on the host, with the sha256 of the former in `R:sf38_campaign/par2pilot/realization_series.sha256`; no `alpha` series is reported or interpreted in this note. The `E_PAIRED` columns are differences of the pipeline's analysis columns between the two time
steps (the file `series/<c>_paired_dt_minus_dt2.csv` of the campaign tooling, `dt` minus `dt/2`; also not imported); they are recorded numbers without threshold and without interpretation. `t_b` has the resolution of the snapshot
interval (957 / 1167 / 1108 steps, i.e. 11962.5 / 14587.5 / 13850 time units, derived `every * dt`).

### F. Regression (met)

ctest (`final/ctest_times.tsv`): 46 tests, all Passed; sum of per-test seconds = 3064.98 s (derived); ctest 'Total Test time (real)' 3065.06 s (`final/ctest.log`).

| test | status | seconds |
|---|---|---|
| lagrangian_tracker_gpu_host | Passed | 13.79 |
| lagrangian_long_box | Passed | 33.50 |
| lagrangian_dirichlet_medium | Passed | 4.30 |
| lagrangian_ensemble | Passed | 47.68 |
| lagrangian_medium_ab | Passed | 56.66 |

Longest tests: streamfunction_operator_tests 1335.05 s; streamfunction_picard_fixed 419.31 s; streamfunction_gmres 261.46 s.

`IDENTICAL` lines in `final/results.txt` by class: binary 1, r 269, trio 172; `DIFF`/`MISSING` lines: 0.

Exit codes head vs base for the 20-run R set and the trio (`final/exit_codes.tsv`, `sf38_base_refs/exit_codes.tsv`):

| run | head rc | base rc |
|---|---|---|
| trio_config_pspta_small | 0 | 0 |
| trio_config_streamfunctions_homogeneous | 0 | 0 |
| trio_config_streamfunctions_continuation | 0 | 0 |
| r_trace_N32_e0.5_host | 0 | 0 |
| r_trace_N32_e0.5_gpu | 0 | 0 |
| r_trace_N32_e1.0_host | 21 | 21 |
| r_trace_N32_e1.0_gpu | 21 | 21 |
| r_trace_N64_e0.5_host | 0 | 0 |
| r_trace_N64_e0.5_gpu | 0 | 0 |
| r_trace_N64_e1.0_host | 0 | 0 |
| r_trace_N64_e1.0_gpu | 0 | 0 |
| r_trace_N64_e2.0_gpu | 21 | 21 |
| r_prod | 0 | 0 |
| r_fwd_smoke_g3d_n16 | 0 | 0 |
| r_fwd_g3d_n32_cov | 0 | 0 |
| r_fwd_gauss_n64_e1_cov | 0 | 0 |
| r_fwd_gauss_n64_e2 | 0 | 0 |
| r_fwd_gauss_n64_e2_tol | 0 | 0 |
| r_fwd_gauss_n40_e1 | 0 | 0 |
| r_fwd_gauss_n80_e2 | 0 | 0 |
| r_neutral_gauss_n64_e1_cov | 0 | 0 |
| r_neutral_gauss_n64_e2 | 0 | 0 |
| r_neutral_gauss_n64_e2_tol | 0 | 0 |

Reading (measured): `F_CTEST exit=0 tests=46 failed=0 wall_s=3065 PASS` (`100% tests passed, 0 tests failed out of 46`, `Total Test time (real) = 3065.06 sec`, `R:sf38_campaign/final/ctest.log`); the new/changed SF-38 tests pass (table above);
`F_TRIO IDENTICAL` (172 `IDENTICAL` lines, manifest keys `timestamp` and `build.git_hash` dropped); `F_RSET IDENTICAL` (269 `IDENTICAL` lines for the 20 runs, exit codes equal to the base, including the three `21`);
0 `DIFF` and 0 `MISSING` lines in `R:sf38_campaign/final/results.txt`. The expected content change of D13 (`lagrangian_tracker_gpu_host`) is in the test source, not an output byte (the test passes at H1).
The git-diff items of F (no existing config changed; frozen directories untouched) are the orchestrator checks listed under "Build / environment".

### G. Field statistics (recorded)

`G_FIELD` lines of `fieldonly/results.txt`: SF-18 raw variance (before `normalize_variance`) per seed and geometry; `final_variance` is 1 in every line.

Raw variance:

| seed | B1 | B2 | B3 | AB_s1 | AB_s4 |
|---|---|---|---|---|---|
| 4001 | 1.000850 | 1.000472 | 1.013711 | 1.000850 | 1.029846 |
| 4002 | 1.001878 | 1.002178 | 1.013527 | 1.001878 | 1.008659 |
| 4003 | 1.008005 | 1.008284 | 0.996734 | 1.008005 | 0.996349 |
| 4004 | 1.000672 | 1.000289 | 1.015162 | 1.000672 | 1.027265 |
| 4005 | 1.004121 | 1.003938 | 1.002417 | 1.004121 | 1.026544 |
| 4006 | 0.997832 | 0.997917 | 0.990213 | 0.997832 | 0.983442 |
| 4007 | 0.998166 | 0.997612 | 0.992540 | 0.998166 | 0.986255 |
| 4008 | 1.008260 | 1.007851 | 1.010783 | 1.008260 | 1.005404 |
| 4009 | 1.002038 | 1.002377 | 1.019774 | 1.002038 | 1.034870 |
| 4010 | 0.997000 | 0.996608 | 0.991039 | 0.997000 | 0.975685 |

Applied scale (as printed):

| seed | B1 | B2 | B3 | AB_s1 | AB_s4 |
|---|---|---|---|---|---|
| 4001 | 0.999575 | 0.999764 | 0.993214 | 0.999575 | 0.985403 |
| 4002 | 0.999062 | 0.998913 | 0.993305 | 0.999062 | 0.995699 |
| 4003 | 0.996021 | 0.995884 | 1.001637 | 0.996021 | 1.001830 |
| 4004 | 0.999664 | 0.999856 | 0.992504 | 0.999664 | 0.986640 |
| 4005 | 0.997946 | 0.998037 | 0.998794 | 0.997946 | 0.986986 |
| 4006 | 1.001086 | 1.001043 | 1.004930 | 1.001086 | 1.008383 |
| 4007 | 1.000918 | 1.001196 | 1.003751 | 1.000918 | 1.006944 |
| 4008 | 0.995896 | 0.996098 | 0.994652 | 0.995896 | 0.997309 |
| 4009 | 0.998982 | 0.998813 | 0.990257 | 0.998982 | 0.983008 |
| 4010 | 1.001503 | 1.001700 | 1.004511 | 1.001503 | 1.012384 |

`G_SUMMARY` lines (copied):

| name | n | raw_variance_mean | raw_variance_sd | applied_scale_mean | applied_scale_sd |
|---|---|---|---|---|---|
| B1 | 10 | 1.001882 | 0.003941 | 0.999065 | 0.001962 |
| B2 | 10 | 1.001753 | 0.004045 | 0.999130 | 0.002015 |
| B3 | 10 | 1.004590 | 0.011283 | 0.997755 | 0.005608 |
| AB_s1 | 10 | 1.001882 | 0.003941 | 0.999065 | 0.001962 |
| AB_s4 | 10 | 1.007432 | 0.021536 | 0.996459 | 0.010673 |

`G_COMPLETE name=B1 expected=4001-4010 found=10 missing=- OK`
`G_COMPLETE name=B2 expected=4001-4010 found=10 missing=- OK`
`G_COMPLETE name=B3 expected=4001-4010 found=10 missing=- OK`
`G_COMPLETE name=AB_s1 expected=4001-4010 found=10 missing=- OK`
`G_COMPLETE name=AB_s4 expected=4001-4010 found=10 missing=- OK`
`G_VERDICT RECORDED_COMPLETE (10 seeds at each of the three production geometries; G has no threshold)`

### Recorded (no threshold)

- **GPU time per part (D16)**: the `part_seconds` / `part_runs_seconds` columns of the first table (`RECORDED part_seconds=... part_runs_seconds=...`, last line of each `results.txt`); head parts
  `controls ... par2pilot` 9520 s (derived sum of `part_seconds`, equals the J1b window 18:34:31-21:13:12 = 9521 s within 1 s), `final` 3246 s, `baserefs` 177 s; all parts 12943 s = 3.595 h (derived). Adding the build jobs
  (J0a attempt 2 167 s, J1a 21 s; windows above) gives 13131 s = 3.648 h (derived: 12943 + 167 + 21); the failed J0a attempt 1 is not counted (log not imported). The Par2 pilot dominates (4474 s of the 12943 s, 34.6 %; derived).
- **Medium (b) memory** (memory-phase table of section C, measured): at 3328x256x256 the `solve` phase has exact 20190177280 B (19254.854 MiB) and `device_used` 19749 MiB, which is the PID peak of the A/B run (19745 MiB); the face
  repack phase 10483138560 B (9997.500 MiB); the FlowWorkspace release phase returns to 3328.000 MiB (K and head). At 2048x256x256 the `solve` phase is 12424724480 B (11849.141 MiB) and the A/B run's PID peak 12337 MiB.
- **Time per realization**: runner, B grids: `TIMING r=0` 52.088 / 207.139 / 198.898 s; D-A 194.1-196.5 s per realization; A/B pairs 1084.17 s / 10 realizations and 1520.1 s / 10 (derived 108.4 s and 152.0 s per pair);
  Par2 pilot per realization (table E): `s1` 32.70-32.78 s for r=1..9, 37.81 s for r=0 (setup included), 33.4 s mean wall/10; `s225` 56.33-56.38, 61.65, 57.1; `s4` 61.46-61.62, 66.70, 62.2.

## Disclosures / deviations

- **Agent model**: workers and the integrator ran with the locally configured agent definitions (model `sonnet`, effort `high`) instead of the checked-in `opus` / `medium` (RR13); local builds went through `mf3d-build`. The author of
  this note is such a worker. No threshold or reading rule depends on it.
- **J0a attempt 1** failed (`BUILD_EXIT=1`, empty mirror after a failed sync of the `Par2_Core` submodule) and was cancelled; attempt 2 is the base build. Its log is not in the imported evidence (bitacora row 2026-10-23T17:26Z).
- **Correctives before any evidence run** (bitacora row 2026-10-23T17:26Z; found by the orchestrator's audit): N6 round 1 ran the A block in the default ctest (MAJOR; corrected by `8c29cd5`, the A block runs only on explicit `--only A`); N3 round 1
  aborted the run on any non-ok particle and wrote an absolute `output_dir` (MAJOR; corrected by `be499e3`: non-ok particles recorded in a deterministic `status.json` and the `STATUS` line, `analysis_config.yaml` with `output_dir: "."`).
- **Reading clarifications ratified before any evidence run** (bitacora rows 2026-10-23T17:26Z and 18:31Z; no threshold changed): D-C is compared with D-A on the realization outputs (`realizations/r_0000/**`, the realization-0 CSVs of `stats/` and
  `stats_twin/`), because run-level aggregates of a one-realization run differ by construction; exit code 0 means the run completed, non-ok particles are recorded with counts and never an exit condition; the `dt/2` copies also double
  `sample_every` and `output_every`; D-C excludes `timings.json` like D-B; the `TTIME` time column is compared at `1e-9`, moments at `1e-12`; "inside the margin" is the closed interval; `S_T`, its floor trigger `0.05 lambda` and its margin `2.5e-3 lambda`
  are in box units; the `analysis_only` reproduction runs on a copy of D-A reduced to `stats/` and `analysis_config.yaml`; the `git diff` item of F is checked by the orchestrator; the `ab` part refuses to start without a C0 PASS.
- **Dry runs** (`SF38_DRYRUN=1`, tiny settings, local, listed in the bitacora) are not evidence and are not imported.
- **RR4 and the pilot log**: the pilot's `bpilot.log` also contains the E1/E2 and `PRIMARY` lines of realization 4001; the author of this note saw them while reading the log for the solver record and quotes none of them as C0 evidence. Seed 4001 enters
  the A/B identically (byte-equal outputs, `AB_DETERMINISM_4001`), so no information beyond the A/B's own `C_D` lines exists in the pilot.
- **Differences from the plan found in the raw evidence**: (i) B3 and D realization 0 have `status=inlet_backflow` (section B), not a failure of any criterion; (ii) the boundary-layer thickness of threshold C is not computed by the campaign tooling (section C);
  (iii) the Par2 `alpha(t)` outputs are preserved on the host but only their sha256 are imported (section E); (iv) host sampler timestamps are local time, 3 h behind the UTC stamps.
- **No GPU or `ctest` computation was run for this note.** All numbers come from the imported evidence; the script `A/scripts/note_tables.py` only parses text.

## Result

Per criterion (SPEC thresholds A-G), with the decisive numbers:

| criterion | verdict | decisive numbers |
|---|---|---|
| A controls | **met** | `A_VERDICT all_pass=1 checks_pass=30 checks_fail=0`; F1 `0.000e+00`; F2 `4.614e-16` / `2.985e-14` / `4.771e-10` (64x32x32), `4.441e-16` / `2.962e-14` / `4.771e-10` (128x32x32); F3 `4.774e-15` / `6.384e-15`; `I` recorded `2.046e-06` abs |
| B production grids | **met** (B1, B2, B3) | exit 0; correctors `converged`, rel `<= 8.941e-12`; 7 levels = auto; PID peak 8257 / 28885 / 28885 MiB (`<= 30000`); wall 53.93 / 209.11 / 200.71 s; host max RSS 9021988 / 31325308 / 31325124 kB |
| C0 (b) solver pilot | **met** (primary mode) | `rel = 9.705149e-11`, 50 iterations, balance `2.404573e-10`, `max_div = 1.034116e-07`, PID peak 19745 MiB |
| C A/B equivalence | **medium (b) selected by the rule**: 5 equivalent, 0 non-equivalent, 5 inconclusive | `C_VERDICT selected=b equivalent=5/10 non_equivalent=0 inconclusive=5`; `sigma^2 = 1`: 4/5 equivalent, 1 inconclusive (`S_tau`); `sigma^2 = 4`: 1/5 equivalent (`V1`), 4 inconclusive; no statistic's interval lies outside its margin |
| D runner | **met** | D-B equals D-A 99/99 files; D-C equals D-A realization 0 32/32; `analysis_only` reproduction byte-equal (`max_rel 0`); `TTIME` vs stats `max_rel_moments 4.859e-13`; per-realization 194.1-197.6 s |
| E Par2 pilot | **completed; recorded** | `E_VERDICT PASS`; 3 x 10 realizations exit 0; 1010 snapshot files each; `K stats` agreement 30/30; realized `var ln K` means 0.9977 / 2.2442 / 3.9803; `t_b` ranges 526350-598125 / 379275-495975 / 180050-235450; `dt/2` paired differences recorded |
| F regression | **met** (campaign items); git-diff items by the orchestrator | `ctest` 46/46, 3065 s; trio 172 `IDENTICAL`; R set 269 `IDENTICAL`, 0 `DIFF`, 0 `MISSING` |
| G field statistics | **recorded** (complete) | 10 seeds at B1, B2, B3, AB_s1, AB_s4; raw variance means 1.00188 / 1.00175 / 1.00459 / 1.00188 / 1.00743, applied-scale means 0.99907 / 0.99913 / 0.99776 / 0.99907 / 0.99646 |

Answer to the Goal's claims: the long-box stage, the runner with checkpoint, twin series and Pollock bridge, and the Par2 baseline configurations all ran on the production geometries and satisfied A, B, C0, D, E and the campaign part of F; the
pre-registered equivalence test selected medium (b) because five statistics did not show equivalence at `N_R = 10` (none showed a difference). No `alpha_L` / `alpha_T` statement follows from any number here.

## Caveats

- **`N_R = 10` precision at `sigma^2 = 4`**: four of the five `sigma^2 = 4` statistics (`S_T`, `S_tau`, `theta_2`, `theta_3`) have CI widths 1.4-6.2 times the equivalence window; a larger ensemble would narrow them but the rule is pre-registered and fixed at ten, one analysis, no interim look.
  The rule is asymmetric by construction (D7): lack of demonstrated equivalence selects (b).
- **A/B geometry**: `sigma^2 = 4` was run at 3328x256x256 (104 x 8 x 8 `lambda`, two thirds of the production width), `sigma^2 = 1` at its production box; the slowest transverse mode decays over `L_T/(2 pi)` (D4), so the A/B may understate (b)'s
  boundary layer; the production width (12 `lambda`) was not tested for (b) because (b) does not fit 3328x384x384.
- **One seed per B run and per D realization** (B: seed 4001; D: seeds 4001-4003); the PID peaks are the maximum of one run each (B2 and B3 both 28885 MiB, SF-40 also 28885 MiB); run-to-run variation is not characterized. The 200 ms sampler can miss
  transients shorter than 200 ms (as in SF-40); the S2 plateau has 632 / 622 samples within 1 MiB of the peak (derived, table in section B, B2 / B3).
- **B3 seed 4001 has inlet backflow**, so its D-1 labels are unavailable; of the three seeds run at this geometry (4001-4003) only seed 4001 has inlet backflow (D-A realizations 1 and 2 have positive `inlet_vmin` 7.067288e-04 and 3.040961e-03, `R:sf38_campaign/ensemble/DA.log`).
- **Medium (b) tilts and `qbar`**: `qbar_b` differs from 1 by up to 0.013 (A/B `sigma^2 = 4`, table in section C); every statistic is normalized by its own medium's `qbar` as pre-registered.
- **Eulerian `var_i(v1)` profiles** are recorded with ten realizations; their SE is large near the faces at `sigma^2 = 4`; the literal boundary-layer reading is not a measurement (section C).
- **Par2 pilot**: the generator differs from SF-18 (random spectral, not periodic, variance not normalized); the pilot's aggregate `alpha(t)` series are imported (addendum) and recorded only, the per-realization series and snapshots stay on the host; `t_b` resolution is the snapshot interval; the first per-realization time includes setup.
- The `dt/2` check covers three realizations per config and is recorded only. The `max_rel_diff` entries of `E_PAIRED` are dominated by near-zero denominators (values up to 8.958e+04) and are not a measure of agreement.
- The J0a attempt 1 log, the dry runs and the Par2 output files are not imported; `ctest -N` of the base build (43) is known from the J0a log only.
- No `alpha` claim: the B/D/E series and the C statistics are recorded as pre-registered inputs only, never interpreted (Rule 1).

## Post-hoc readings (not pre-registered; added 2026-10-24 after FINAL_AUDIT at the owner's request)

Five readings of the same ten paired realizations per amplitude, computed by `A/scripts/posthoc_ab.py` from `R:sf38_campaign/ab/` (and `R:sf38_campaign/prodgrid/B2.log`) into `ANA/posthoc_ab.md`
(`python3 A/scripts/posthoc_ab.py A/raw A/analysis`; a rerun is byte-identical; `note_tables.py` still regenerates every other `ANA` file byte-identically). Every number below is copied from `posthoc_ab.md`
and is *derived (post-hoc)*. **None of these readings changes any pre-registered verdict, threshold or analysis file; they are exploratory (forking-paths risk: several readings of one ten-realization sample, none specified before the data, no
multiplicity control); the medium decision stands as the rule's outcome (`C_VERDICT selected=b`).** No `alpha_L` / `alpha_T` statement is made; `S_T` and `S_tau` appear only as the rule's statistics.

**P1 Pairing efficiency** (*derived (post-hoc)*; `eff = sd(D) / sqrt(sd_a^2 + sd_b^2)` on the raw statistic, 1 = no gain from pairing, 0 = perfect pairing; `corr` = Pearson correlation of the ten (a), (b) values):

| amp | stat | corr(a,b) | eff |
|---|---|---|---|
| 1 | `V1` | 1.000 | 0.007 |
| 1 | `S_T` | 1.000 | 0.022 |
| 1 | `S_tau` | 0.932 | 0.271 |
| 1 | `theta_2` | 0.324 | 0.935 |
| 1 | `theta_3` | 0.646 | 0.934 |
| 4 | `V1` | 1.000 | 0.010 |
| 4 | `S_T` | 0.691 | 0.565 |
| 4 | `S_tau` | 0.815 | 0.545 |
| 4 | `theta_2` | 0.752 | 0.686 |
| 4 | `theta_3` | -0.309 | 1.097 |

Reading: pairing is almost perfect for `V1` at both amplitudes and for `S_T` at `sigma^2 = 1`; at `sigma^2 = 4` it reduces the sd of the difference only to about 0.55-0.69 of the unpaired value for `S_T`, `S_tau`, `theta_2`, and not at all for
`theta_3` (1.097); the tilts are weakly paired also at `sigma^2 = 1` (0.93). The standard errors of the inconclusive `sigma^2 = 4` statistics are therefore mostly medium-specific variation, not variation of the shared field.

**P2 Point estimates against the margins** (*derived (post-hoc)*, from the `C_STAT` lines; table in section C above): 3 of the 5 inconclusive statistics have their point estimate outside the margin, all three at `sigma^2 = 4` (`S_T`, `S_tau`,
`theta_2`; values in section C); the other two inconclusive statistics have `|est| < margin`: `S_tau` at `sigma^2 = 1` (est 0.01246, margin 0.02469, observed half-width 0.0239) and `theta_3` at `sigma^2 = 4` (0.003248 against 0.005, half-width 0.01054).
Rough sample sizes at which the CI half-width would equal `margin - |est|`, `N ~ 10 (h10 / (margin - |est|))^2` (rough: constant sd and point estimate, t-quantile change ignored; not a power calculation): about 38 for `S_tau` at `sigma^2 = 1`
and about 362 for `theta_3` at `sigma^2 = 4`.

**P3 Paired boundary-layer profile** (*derived (post-hoc)*): `D_i = var_i(v1)/qbar_b^2 - var_i(v1)/qbar_a^2` per realization and plane, ensemble mean over the ten realizations, `M_a` the interior mean of the (a) profile (`M_a` = 0.6151 at
`sigma^2 = 1`, 3.1693 at `sigma^2 = 4`); `(a)` is the control (no `x`-face condition). `mean D / M_a`, nearest plane to the distance from the face in `lambda`:

| amp | face | 0 | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|---|---|
| 1 | west | 0.3108 | 0.1682 | 0.0549 | 0.0196 | 0.0082 | 0.0060 | 0.0004 |
| 1 | east | 0.3827 | 0.2046 | 0.0726 | 0.0260 | 0.0077 | 0.0024 | -0.0000 |
| 4 | west | 0.6208 | 0.2727 | 0.0800 | 0.0376 | 0.0197 | 0.0022 | -0.0013 |
| 4 | east | 0.2889 | 0.1139 | 0.0789 | 0.0220 | 0.0093 | 0.0075 | -0.0018 |

Thickness readings, defined in `posthoc_ab.md` (post-hoc): `d_1%` (smallest distance beyond which every plane up to `20 lambda` has `|mean D| <= 0.01 M_a`) is 3.800 / 3.800 `lambda` (west / east) at `sigma^2 = 1` and 4.281 / 3.969 `lambda`
at `sigma^2 = 4`. `d_2SE` (every plane up to `20 lambda` has `|mean D| <= 2 SE`) is 15.100 / 12.900 `lambda` at `sigma^2 = 1` and 3.906 / 10.000 `lambda` at `sigma^2 = 4`; at `sigma^2 = 1` the paired SE is small enough
(9.262e-04 at `5 lambda`, west) to resolve a residual of at most 0.6 % of `M_a` beyond `5 lambda` (64 of 151 planes between `5 lambda` and `20 lambda` violate `2 SE` on the west face), so the two thickness readings differ by a small tail, not by
a different layer. Per-`lambda` decay ratio of `mean D` between 1 and 4 `lambda`: 0.365 / 0.335 (`sigma^2 = 1`, west / east), 0.417 / 0.435 (`sigma^2 = 4`); the latter correspond to e-folding lengths 1.14 / 1.20 `lambda` (derived
`-1/ln(ratio)`), the same order as the 1.3 `lambda` recorded as the slowest-mode assumption of D4 for the 8 `lambda` width. The paired layer is a few `lambda` (about 4 `lambda` at 1 % of `M_a`), against the 99.6 / 79.3 and 42.3 / 47.2 `lambda`
of the noise-dominated literal reading above; the width caveat (A/B at 8 `lambda` against the production 12 `lambda`) stands.

**P4 Mean-flow forcing and tilt.** *Confirmed in code*: medium (a) imposes the mean flux, `qbar = (1, 0, 0)` (`AffinePeriodicFlowSolver.cuh:156-167`, `LongBoxStage.cu:251-253`), and the mean gradient `G` is the free solution of
`K_eff G = qbar` (`AffinePeriodicFlowSolver.cuh:212-213`, `AffinePeriodicFlowSolver.cu:965` for the lean path used by the stage, `:837` for the full path), with transverse components; medium (b) imposes the head drop `G1 L1` along `x`
(`DirichletMedium.cu:153-160`: `xmin` Dirichlet `G1 L1`, `xmax` Dirichlet 0, `y`/`z` periodic; `G1 = G[0]` of the (a) solve, `lagrangian_medium_ab_main.cu:599`), so its mean gradient is along `x` and its mean transverse flux is free.
*Hypothesis consistent with the data* (not tested): the media differ in the macroscopic forcing everywhere in the domain, not only near the `x` faces. Data (*derived (post-hoc)*): at `sigma^2 = 4` the sd over the ten realizations
of `theta_2` / `theta_3` is 0.0166 / 0.0155 in (b) against 0.0068 / 0.0058 in (a) (the (a) values are interior-window fluctuations: the whole-domain mean flux of (a) is `e1` by construction); `qbar_b` has mean 1.0068 and sd 3.5e-3 (`qbar_a` = 1);
the tilt of (b) is correlated with the transverse-to-longitudinal ratio of the mean gradient of the (a) solve of the same field (Pearson, n = 10: `theta_2` against `G2/G1` -0.919 and `theta_3` against `G3/G1` -0.950 at `sigma^2 = 4`;
-0.980 and -0.997 at `sigma^2 = 1`); and the tilt is a near-constant offset along `x1`: over four quarters of the interior layers, the spread of the quarter-mean `<v2>/qbar` of (b) minus (a) on the same field is 0.29 of its overall mean
absolute value at `sigma^2 = 4` (0.28 for `<v3>/qbar`; 0.052 and 0.057 at `sigma^2 = 1`), while the quarter means of (b) alone, and of (a) alone, fluctuate by 7.2 and 15.7 times their overall mean (`<v2>`, `sigma^2 = 4`), i.e. the quarter-to-quarter
variation is the field's own and is common to both media. Geometric consequence (ratio only, no alpha statement): in fixed axes, the transverse variance of a plume with mean flow tilted by `theta` picks up approximately `theta^2 sigma_L^2`;
the sd of each tilt component of (b) at `sigma^2 = 4` is 3.310 and 3.103 times the 0.005 per-component margin and the rms `|theta|` of (b) is 0.0227 = 3.197 times the 0.0071 vector bound assumed in D19 (at `sigma^2 = 1`: 0.257, 0.348 and 0.290).

**P5 Memory of medium (b) at the production grid** (*derived (post-hoc)*; includes the CUDA context in the PID figures; the non-periodic spline stage is in none of the figures): PID peak of `ab_s4` (3328x256x256, 218103808 cells) 19745 MiB = 94.93 B/cell,
of `ab_s1` (2048x256x256, 134217728 cells) 12337 MiB = 96.38 B/cell (exact allocations of the `solve` phase 92.57 B/cell in both). Linear extrapolation to 3328x384x384 (490733568 cells): 44426 MiB (from `ab_s4`) / 45107 MiB (from `ab_s1`),
against the 30000 MiB ceiling (64.10 B/cell with context). The lean non-projected estimate of the memory audit section 10, 68.57 B/cell (*estimate from code count, not measured*), gives 32091 MiB for the flow solve alone. The measured SF-40 level at the same
box is 60.655 B/cell exact (`B2` phase `S2`), PID peak 28885 MiB = 61.72 B/cell. Conclusion: medium (b) as measured does not fit 3328x384x384; a lean (b) flow stage plus the spline would have to come in at or below the SF-40 level (~61 B/cell) to fit;
otherwise stop rule R3 (domain reduction by owner decision) applies at `sigma^2 >= 2.25`.

These readings do not alter the C verdict (5 equivalent, 0 non-equivalent, 5 inconclusive), the selection of (b) by the rule, or any other criterion of "Result".

## Next step

SF-38 closes with medium (b) selected: by stop rule R4 (SPEC failure policy, decision 2026-10-17 item 7) a non-periodic-in-`x1` spline increment (SF-41) must be specified and `done` before SF-39 activates (owner decision; this note
does not set the State). The Gate 5 baseline is prepared: the Par2 configs and their realized `ln K` variance and `t_b` ranges (E), the runner's twin series and Pollock bridge with checkpoint/restart (D), the SF-18 raw variance and applied scale per seed (G,
input to SF-39's `normalize_variance` choice), the production-grid memory and time (B). Human review of SF-38 is required before closure.

Risks exposed by the post-hoc readings, for the owner (not decisions):

- **Memory (P5)**: SF-41 must include a memory-lean medium (b) flow stage at 3328x384x384 together with the non-periodic spline, at or below the ~61 B/cell level of the SF-40 stage (30000 MiB is 64.10 B/cell with context); medium (b) as built is
  94.93 B/cell (44426 MiB extrapolated) and even the lean estimate (68.57 B/cell, 32091 MiB, estimate not measured) exceeds the ceiling before the spline. Otherwise stop rule R3 (domain reduction by owner decision) applies at `sigma^2 >= 2.25`.
- **SF-39 pairing and `N_R` (P1, P2)**: at `sigma^2 = 4` the pairing efficiency of `S_T`, `S_tau` and the tilts is 0.55-1.10, so paired ladders gain little variance reduction there (unlike `V1`, 0.010), and the rough `N_R` to resolve the
  `theta_3` margin from its current point estimate is of the order of hundreds (about 362), not ten.
- **Tilt of (b) in fixed axes (P4)**: the per-realization tilt of (b) at `sigma^2 = 4` has rms `|theta|` = 0.0227, 3.2 times the D19 vector bound; transverse moments measured in fixed axes pick up `theta^2 sigma_L^2` from it, which SF-39's design
  has to account for (how is not decided here).
