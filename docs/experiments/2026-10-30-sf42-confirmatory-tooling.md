# SF-42: tooling and verification for the confirmatory macrodispersion study

- What this is: the run record of increment SF-42
  (`docs/plans/active/lester-eq14/increments/SF-42-confirmatory-tooling-verification.md`): the tools and the verification that SF-39 did
  not have, built for the confirmatory study SF-43 (decision `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`, G1, G4-G8). This
  version records the first remote campaign (R1): acceptance A (byte identity), B (field identity) and C (injection window, seed-lattice
  rule, plane crossings), and the Gate 1 full `ctest` on the V100.
- What it is not: a macrodispersion estimate (no `alpha_L`, no `alpha_T`, no statement on the zero-transverse claim of Lester et al.
  2023; `docs/theory/lester-2023-key-claims.md`), and not evidence for SF-43. Its pilots (section G) are design inputs with seed ranges
  disjoint from SF-43's.
- Artifact root `R = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/` (see `R/README.md`); R1 evidence in
  `R/verification/`.
- Sections D and E (the MMS order study and the pseudo-spectral cross-check, second remote campaign R2) are recorded; sections F to H
  and the findings for SF-43 that depend on them are filled by the later nodes (library and coverage, pilots, reproducibility tooling).

## Question

1. Do the new opt-in options (Par2 on the SF-18 field, injection window, plane crossings, field dump; `lagrangian_ensemble
   --dump-field`) leave every existing path byte-identical to the increment base (acceptance A)?
2. Is the Par2 field with `stochastic.field_source: sf18_medium_b` bitwise equal to the field of `lagrangian_ensemble --medium b` for
   the same seed, with `var Y = 1` (acceptance B)?
3. Does the Par2 injection window put every particle inside the window, does the seed-lattice rule predict the runner's RR12 outcome,
   and do the Par2 plane crossings equal a brute-force reference at 1e-12 relative (acceptance C)?

## Hypothesis / expected numerical effect

From the specification ("Expected numerical effect"): none on existing paths. Par2 on the SF-18 field: `var Y = 1` and `var ln K =
eps^2` to rounding. Every new option is off by default and, when off, writes no new output, no new line in `effective_config.yaml` or
`manifest.json` and allocates nothing that could alter results. Expected verdicts: A, B, C met. Regression surface: Par2 config parsing,
the ensemble runner and the stats writers; the consumer (one flag in the option parser and one optional callback after phase S1);
`scripts/remote` (blinded-phase statement on exec/run); CMake (two new tests and executables).

## Build / environment

- Increment base (code): `3164e4fb5c2e3bebf4fdc9ae5647bc5b1d27384a` (master). Head: the tip of the delivery branch
  (`science/lester-sf42-confirmatory-tooling`); the R1 worktree started at `5f48caf9` (base plus the metadata-only activation commit)
  and received the approved node commits by `git cherry-pick`, in this order (new hash, node):
  `318dc83f` N1 (Par2 options), `7e90e005` N1b (`--dump-field`), `971c772e` N2 and `22b2837e` N2c (MMS driver), `6d2206db` and
  `d6144cfd` N3 (spectral), `d1569179` N4 (seed lattice), `e77abb8c` N5 and `34ef9bfe` N5c (reducer), `811db717` N6a (mcstat),
  `b21b5c7e` and `41d43ae4` N7 (reproducibility). The only conflict was in `CMakeLists.txt` at N4: N2 (`mms_dirichlet`) and N4
  (`seedlattice_probe`) both append a block at the end of the same `if(...)` section; both blocks were kept unchanged, separated by one
  blank line. No source was modified beyond that resolution. The R1 scripts (`R/verification/`) were committed on top
  (`96a335dc`, `daf0d928`).
- The head binaries of the remote jobs were built from the mirror tree `96a335dc9108` and run from `daf0d92856b6` (job `sf42-head`
  onwards; the second commit added only `R/verification/job_head.sh`). `git diff --stat 96a335dc HEAD -- src apps tests cmake
  CMakeLists.txt CMakePresets.json scripts/regress` is empty, so the sources of both trees are the same.
- Host: one Tesla V100-PCIE-32GB (GPU 0 chosen by `scripts/remote run` for every job, UUID `GPU-f4987463-...` in `R/verification/regress/*_env.txt`), driver 470.239.06, nvcc 11.4.r11.4, gcc 9.3.1, cmake 4.2.1, kernel
  3.10.0-1160.119.1.el7.x86_64; Python 3.11.7, numpy 1.26.4, scipy 1.11.4. Preset `v100-release` (clean build directory for both builds,
  `TMPDIR=$HOME/.cache/mf3d-tmp`). Mirror `~/MacroFlow3D-SF-42` (`--increment SF-42`), host outputs under `$HOME/sf42_campaign`.
- Binary md5 (`R/verification/regress/{base,head}_binaries.md5`, `R/verification/base_bin_md5.txt`):

  | binary | base `3164e4fb` | head (tree `96a335dc9108`) |
  |---|---|---|
  | `macroflow3d_pipeline` | `701fac2a1ff9c54cd7d3d53bbc1ddb24` | `efdff48e1d10707559ee77d1144f9eef` |
  | `lagrangian_ensemble` | `f3428662341e9ed7558de6b244d4e3c9` | `293a9d85d35bda2d69cffbad17e66bda` |
  | `inlet_slab` | `c6a9d847b76e205b55d53ac49a422100` | `54e89a6d0f4d2cc6116fcc944020044e` |
  | `lagrangian_medium_ab` | `2cd1d78b8a65d79f8b5adf68e588c45a` | `e2962c735c9b5cad8f2742c9f8607891` |

  The binaries differ (the head contains the new code), so `compare.sh` printed no "identical binaries" NOTE.
- Local (WSL, sm_86, `wsl-debug`): the same targets built; `ctest -R par2_sf42_options`, `seedlattice_probe`,
  `mms_dirichlet_self_check` pass (2.7 s, 1.9 s, 2.0 s); 243 Python tests of `R/scripts` pass with `python3 -W error` (1 skipped:
  `test_s025_gate5_from_per_realization_values`, host-only data); `scripts/tests/remote_blinded_test.sh` and
  `remote_gpu_select_test.sh` ALL PASS. These local runs are sanity only; the evidence below is from the V100.
- Base reference of `scripts/regress`: none existed for `3164e4f` (`~/regress/refs/` held `1e40cfe` and `3452a7d`), so it was built
  and run in this campaign (jobs `sf42-base-build`, `sf42-base`; reference at `~/regress/refs/3164e4f/out`, `REGRESS_DONE
  unexpected_rc=0 wall_s=206`). For the record, `src apps tests cmake CMakeLists.txt CMakePresets.json scripts/regress` are identical
  between `1e40cfe9` and `3164e4fb`.

## Config(s)

All in `R/verification/`: `bytes/par2_tiny.yaml` (the SF-39 `par2_s1.yaml` shortened: 256x32x32, `dx` 5, west head 12.5, 200
particles, 300 steps, NR 2, `mg_levels` 4; no new key); `identity/B_<case>.par2_config.yaml` (Par2 field-identity configs; `dx` is the
`%.17g` string of `1/ny`, `corr_length` is `ell`, `K_mean` 1, `sf18_medium_b: {sigma2: 1, eps, seed0}`, west head `L1`, `cg_rtol`
1e-10, `cg_max_iter` 2000, `output.dump_field` realization 0); `identity/C_window.par2_config.yaml` (128x32x32, `inject_window`
`[0.1, 0.9, 0.2, 0.8]`, `plane_crossings` with `dump_injection`, 2000 particles).

| case | grid | `ell` | `eps` | seed0 |
|---|---|---|---|---|
| `p2_e1` | 64x16x16 (`ny` = 16) | 0.25 | 1 | 420211 |
| `p2_e2` | 64x16x16 | 0.25 | 2 | 420212 |
| `f3_e1` | 96x24x24 (`ny` = 24 = 8 x 3) | 0.25 | 1 | 420213 |
| `f3_e2` | 96x24x24 | 0.25 | 2 | 420214 |
| window | 128x32x32 | 0.25 | 1 | 420221 |

Seeds are in the SF-42 reserved range 420201-420299 (the byte check uses 420201).

## Commands

From the mirror root, `V = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/verification`; every job with
`REMOTE_GPU_WAIT=7200` (CUDA jobs also `REMOTE_GPU_MIN_FREE_MIB=30000`), no `--gpu`, no `CUDA_VISIBLE_DEVICES`, one job at a time:

```bash
# from a separate checkout of the base 3164e4fb (Par2_Core populated): scripts/remote --increment SF-42 sync
scripts/remote --increment SF-42 run sf42-base-build -- "cd ~/MacroFlow3D-SF-42 && ... rm -rf build/v100-release && cmake --preset v100-release && cmake --build build/v100-release -j && md5sum ..."
scripts/remote --increment SF-42 run sf42-base -- "cd ~/MacroFlow3D-SF-42 && bash sf42_base_job.sh"
#   (copy of the base binaries to ~/sf42_campaign/base_bin, then REGRESS_SOURCE_TREE=3164e4fb bash scripts/regress/run.sh build/v100-release ~/regress/refs/3164e4f/out)
# from the head worktree: scripts/remote --increment SF-42 sync
scripts/remote --increment SF-42 run sf42-head-build -- "cd ~/MacroFlow3D-SF-42 && ... rm -rf build/v100-release && cmake --preset v100-release && cmake --build build/v100-release -j && md5sum ..."
scripts/remote --increment SF-42 run sf42-head -- "cd ~/MacroFlow3D-SF-42 && bash $V/job_head.sh"
scripts/remote --increment SF-42 run sf42-bytes -- "cd ~/MacroFlow3D-SF-42 && bash $V/bytes.sh ~/sf42_campaign/base_bin build/v100-release ~/sf42_campaign/bytes"
scripts/remote --increment SF-42 run sf42-ctest -- "cd ~/MacroFlow3D-SF-42 && ctest --test-dir build/v100-release --output-on-failure"
scripts/remote --increment SF-42 run sf42-identity -- "cd ~/MacroFlow3D-SF-42 && bash $V/identity.sh build/v100-release ~/sf42_campaign/identity"
```

`sf42_base_job.sh` was a file of the base checkout only (copy: `R/verification/job_base.sh`): it copies the base binaries aside and runs
`run.sh` for the reference. `job_head.sh` runs
`run.sh` on the head build and `compare.sh <base ref> <head out>` and copies the comparison and md5 lists to `~/sf42_campaign/regress/`.
Jobs (times from the job logs, `R/verification/logs/`; GPU-h = wall hours of the job, one GPU; `R/verification/jobs.tsv`):

| job | start (UTC) | end (UTC) | wall (s) | GPU-h | exit | GPU |
|---|---|---|---|---|---|---|
| `sf42-base-build` | 2026-10-30 04:40:39 | 04:45:51 | 312 | 0.087 | 0 | 0 |
| `sf42-base` | 04:48:02 | 04:51:29 | 207 | 0.058 | 0 | 0 |
| `sf42-head-build` | 04:54:29 | 04:59:39 | 310 | 0.086 | 0 | 0 |
| `sf42-head` | 05:01:43 | 05:05:15 | 212 | 0.059 | 0 | 0 |
| `sf42-bytes` | 05:06:46 | 05:07:08 | 22 | 0.006 | 0 | 0 |
| `sf42-ctest` | 05:08:24 | 05:39:04 | 1840 | 0.511 | 0 | 0 |
| `sf42-identity` | 05:40:09 | 05:40:58 | 49 | 0.014 | 0 | 0 |
| `sf42-r2-build` (CPU: incremental build of the two R2 targets, 243 Python tests on the host stack) | 05:53:25 | 05:58:39 | 314 | 0.087 | 0 | 0 |
| `sf42-mms` (CUDA) | 05:59:32 | 06:24:59 | 1527 | 0.424 | 0 | 0 |
| `sf42-spectral-fv` (CUDA) | 06:26:03 | 06:30:58 | 295 | 0.082 | 0 | 0 |
| `sf42-spectral-ref` (CPU, 40 workers; holds the slot) | 06:31:50 | 09:01:37 | 8987 | 2.496 | 0 | 0 |
| total R1 | | | 2952 | 0.820 | | |
| total R2 | | | 11123 | 3.090 | | |
| running total R1 + R2 | | | 14075 | 3.910 | | |

Two builds are compile-only jobs that hold the GPU slot (0.173 GPU-h of the total). Base and head both ran on GPU 0 of the same host.
The first launch attempts of `sf42-base` and `sf42-bytes` were refused by the agent sandbox before reaching `scripts/remote` (no job was
created); they were resubmitted unchanged in meaning.

R2 jobs (second campaign, same mirror, started from the accepted R1 head `460c9178`; table rows above; `R/verification/jobs_r2.tsv`;
logs trimmed in `R/mms/logs/`, `R/spectral/logs/`). Every job with `REMOTE_GPU_WAIT=7200`, the CUDA ones also with
`REMOTE_GPU_MIN_FREE_MIB=30000`, no `--gpu`, one job at a time, GPU 0 chosen by `scripts/remote run` for all four (idle host:
0.0 % and 2.0 % mean utilisation, 32510 MiB free). The sync of the R2 tree was done once before the first job. The build step
rebuilt the library (127 ninja steps, the incremental state of the mirror did not carry over); the binaries (md5
`27df9b39bd9464b8b5c9a80a969ee4cc` `mms_dirichlet`, `2850b02fbaa2783c38de75ba8f585514` `sf42_periodic_flow`) were built from sources
identical to those of the R1 head build (`git diff 96a335dc HEAD -- src apps tests cmake CMakeLists.txt CMakePresets.json
scripts/regress` empty). The 243 Python tests of `R/scripts` passed on the host stack (Python 3.11.7, numpy 1.26.4, scipy 1.11.4):
`Ran 243 tests in 18.273s`, `OK (skipped=1)`.

```bash
# R2, from the mirror root (R = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling)
scripts/remote --increment SF-42 sync
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-r2-build -- "cd ~/MacroFlow3D-SF-42 && export TMPDIR=\$HOME/.cache/mf3d-tmp CMAKE_BUILD_PARALLEL_LEVEL=16 && cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j --target mms_dirichlet sf42_periodic_flow && md5sum build/v100-release/mms_dirichlet build/v100-release/sf42_periodic_flow && cd $R/scripts && time python3 -W error -m unittest discover -s . -t ."
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-mms -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/mms/run_mms.sh build/v100-release ~/sf42_campaign/mms"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-spectral-fv -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/spectral/run_spectral.sh build/v100-release ~/sf42_campaign/spectral fv"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-spectral-ref -- "cd ~/MacroFlow3D-SF-42 && SF42_WORKERS=40 bash $R/scripts/spectral/run_spectral.sh build/v100-release ~/sf42_campaign/spectral spectral"
# results copied from the host (small files) into R/mms, R/spectral; the FV dumps (4.7 GB, 192 npy) stay on the host,
# ~/sf42_campaign/spectral/fv, with the sha256 list R/spectral/fv_npy.sha256 (sha256 of the list: 1cb50b7b8e17af59a2a88ad6021c6de16a21278a03a156a922f2a1c47293ffbb)
```

`run_mms.sh` runs `analyze.py` at the end (exit 0 = PASS); `run_spectral.sh ... spectral` runs `spectral.compare run` (reference pairs
`0.25:64,1:64,2.25:96,4:128` and the `summarize` step, defaults of the script) and writes `summary.{json,md}`. Host outputs:
`~/sf42_campaign/mms/` (JSON, 136 KB) and `~/sf42_campaign/spectral/` (`fv/` dumps and reports, `cases/`, `summary.*`). The sha256 of the
24 MMS JSON, taken as `sha256sum *.json | sha256sum`, is `15296bca49bf18ddd430dad257021decef80f1c0465131a62e2bcbe84b061162` on the host and in
`R/mms/`. `sf42-spectral-ref` took 2 h 30 min, dominated by the `sigma^2` = 4 reference at `M` = 256 (5634 s for seed 420001 and 1895 s for
seed 420002: 2171 and 743 matvecs for the three correctors).

R3 jobs (third campaign, same mirror, from the R3 tree = accepted head `7306eecf` + N6b `c1afdafc` + N6c `05d9d213` cherry-picked with
`-x` as `a9a7f23d`, `0a0ce386`, plus the R3 verification files; `R/verification/jobs_r3.tsv`, logs in `R/verification/logs/`). All four
are CPU-only Python jobs (no CUDA call), each with `REMOTE_GPU_WAIT=7200`, no `--gpu`, one at a time; `scripts/remote run` chose GPU 0
(idle host: 0.0 % and 2.0 % mean utilisation, 32510 MiB free) and held the slot for the whole job. `~/sf42_campaign/r3/.start` was created
before the first job. Start and end are the first and last timestamps of the job log (slot held).

| job | start (UTC) | end (UTC) | wall (s) | GPU-h | exit | GPU |
|---|---|---|---|---|---|---|
| `sf42-hosttests` (F, 309 tests on the host stack) | 2026-10-30 09:13:39 | 09:14:19 | 40 | 0.011 | 0 | 0 |
| `sf42-recompute` (H) | 09:15:26 | 09:15:39 | 13 | 0.004 | 0 | 0 |
| `sf42-reconcile` (H) | 09:17:37 | 09:17:40 | 3 | 0.001 | 0 | 0 |
| `sf42-noise` (G, estimator noise) | 09:20:13 | 09:34:24 | 851 | 0.236 | 0 | 0 |
| total R3 | | | 907 | 0.252 | | |
| running total R1 + R2 + R3 | | | 14982 | 4.162 | | |

```bash
# R3, R = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling, SF39 = docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study
scripts/remote --increment SF-42 exec -- "mkdir -p ~/sf42_campaign/r3 && touch ~/sf42_campaign/r3/.start"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-hosttests -- "cd ~/MacroFlow3D-SF-42/$R && mkdir -p ~/sf42_campaign/r3 && python3 -B verification/par2_alphax_s025.py ~/sf39_campaign ~/MacroFlow3D-SF-42/$SF39 ~/sf42_campaign/r3/par2_alphax_s025.txt && cd scripts && OMP_NUM_THREADS=1 MCSTAT_PAR2_ALPHAX_S025=~/sf42_campaign/r3/par2_alphax_s025.txt python3 -W error -B -m unittest discover -s . -t . -v"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-recompute -- "cd ~/MacroFlow3D-SF-42/$R/scripts && python3 -B -m recompute.sf39 --out-root ~/sf39_campaign --artifact ~/MacroFlow3D-SF-42/$SF39 --sets s025_prod,s1_prod,s225_prod,s4_prod --realizations 0,1,2 --report ~/sf42_campaign/r3/recompute_report.json"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-reconcile -- "cd ~/MacroFlow3D-SF-42/$R/scripts && python3 -B -m reconcile.solver --logs ~/.macroflow3d-remote/macroflow3d-SF-39/logs --runs-root ~/sf39_campaign/runs --check-sf39 --report ~/sf42_campaign/r3/reconcile_report.txt"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-noise -- "cd ~/MacroFlow3D-SF-42/$R/verification && bash noise_job.sh"
```

`~/sf39_campaign` was only read (the tools open it read-only; every output is under `~/sf42_campaign/r3/`); `sf42red prune` was not run.
The check appended to `noise_job.sh` (`find ~/sf39_campaign -newer ~/sf42_campaign/r3/.start -print | head -20`, run after the three
earlier jobs and the four reductions) printed nothing before `SF39_UNTOUCHED_CHECK_DONE`. The mirror's `src/external/Par2_Core` was
populated in the worktree by copying the files of the control checkout's submodule (same commit `a1eecaef`; the `git submodule update`
form of the common context was refused by the agent sandbox), only so that `scripts/remote sync` accepts the tree; nothing of it is
committed.

## A. Byte identity

Verdict: **met**.

1. `scripts/regress` set v2 (27 cases: three `macroflow3d_pipeline` configs including `config_par2_small`, the 20 `inlet_slab` R cases,
   `lagrangian_ensemble` medium (a) and (b), `lagrangian_medium_ab`, and the AnalysisRunner case) on the head against the reference of
   the base, job `sf42-head`: `cases compared: 27`, last line **`REGRESS IDENTICAL`** (`R/verification/regress/compare.txt`; every case
   `IDENTICAL`, e.g. `pipe_config_par2_small` 111 items, `runner_b` 81 items). Exit codes of the 27 cases equal the base's (three
   `inlet_slab` trace cases return their expected 21 in both). `run.sh` of both: `REGRESS_DONE unexpected_rc=0`.
2. One default `lagrangian_ensemble --medium b` realization, 512x64x64, `ell` 0.15625 (`lambda/h` 10), `sigma^2` 1 (`--sigma2 1 --eps 1`),
   seed 420201, `--np-side 100` (RR12 admissible: `python3 -m seedlattice.rule 64 0.1,0.9,0.1,0.9 100` is `OK`), window `[0.1, 0.9]^2`,
   `--times auto --tol 1e-8 --fused-splines --single-launch --twin on --pollock-bridge on`, no new flag, base binary against head binary
   in the same job (own empty cwd, identical relative arguments): `TREECMP IDENTICAL files=41`; only `realizations/r_0000/timings.json`
   was not compared (`R/verification/bytes/a_treecmp.txt`). The runner logs differ only in wall-clock fields.
3. Par2 config without new keys beyond the regress set: the SF-39 `par2_s1.yaml` shortened (above), base against head: `TREECMP
   IDENTICAL files=112`; `manifest.json` identical after dropping `timestamp` and `build.git_hash`; the raw manifests differ in exactly
   two lines, `"git_hash": "3164e4fb5c2e"` against `"96a335dc9108"` and the `timestamp` (`2026-10-30T02:07:03` against `...:07`)
   (`R/verification/bytes/b_manifest_diff.txt`). `effective_config.yaml`, the particle and macrodispersion outputs are byte-identical.

## B. Field identity

Verdict: **met**.

`lagrangian_ensemble --medium b --dump-field` against `macroflow3d_pipeline` with `field_source: sf18_medium_b` and `output.dump_field`
(job `sf42-identity`, `R/verification/identity/B_*.txt`). Each case: `Y.npy` and `K.npy` are **byte-identical files** (identical npy
headers; float64, shape `(nz, ny, nx)`), and `var Y` (mean-subtracted, biased) satisfies the 1e-12 limit:

| case | grid | `eps` | seed0 | `Y` | `K` | `var Y` | `abs(var Y - 1)` |
|---|---|---|---|---|---|---|---|
| `p2_e1` | 64x16x16 | 1 | 420211 | identical | identical | 1 | 0 |
| `p2_e2` | 64x16x16 | 2 | 420212 | identical | identical | 0.99999999999999978 | 2.2e-16 |
| `f3_e1` | 96x24x24 | 1 | 420213 | identical | identical | 0.99999999999999989 | 1.1e-16 |
| `f3_e2` | 96x24x24 | 2 | 420214 | identical | identical | 1.0000000000000002 | 2.2e-16 |

One power-of-two `ny` (16) and one with a factor 3 (24), two `eps` each. In addition the standalone run of the `par2_sf42_options` test
binary (86 checks, 0 failed; it is also ctest case 4) compares two realizations of a 32x16x16 case with a direct SF-18 generator call
bitwise (`B.Y_bitwise_equal_consumer_r0/r1`, `B.K_bitwise_equal_exp_eps_Y_r0/r1`, `var Y` 1.5 to 1e-12 for `sigma2` 1.5). `var ln K =
eps^2` follows from `ln K = eps Y` and is not tested separately.

## C. Injection window, seed-lattice rule, plane crossings

Verdict: **met** (with one finding on the common lattice size of an amplitude, below).

1. **Injection window.** Par2, 128x32x32, `inject_window: [0.1, 0.9, 0.2, 0.8]`, 2000 particles, `dump_injection` on
   (`R/verification/identity/C_window.txt`): 2000 injection rows, all with `x1 = 5 ell = 1.25`, all inside the window in `x2` and `x3`
   (x2 in `[0.100789, 0.899412]`, window `[0.1, 0.9]`; x3 in `[0.200203, 0.799294]`, window `[0.2, 0.8]`): **`INJECTION_WINDOW PASS`**. The
   `par2_sf42_options` test repeats it on a window near the periodic faces (`B.injection_inside_window`, `B.injection_fills_window`).
2. **Plane crossings against brute force.** The brute-force comparison is inside the `par2_sf42_options` test: 400 particles, 30 steps,
   wrapped periodic trajectories (5624 wrapped rows), eight planes of which every one is partially crossed (396 down to 79 of 400
   particles), several planes crossed in one step (224 particle-step pairs); the crossing-time and transverse moments (positions and
   seed-subtracted displacements, 12 quantities per plane) equal an independent reference built from full-precision per-step snapshots to
   1e-12 relative: **`B.crossings_equal_bruteforce_1e-12 PASS`**, run on the V100 as ctest case 4 (Passed, 9.46 s, in the full suite) and
   standalone (`R/verification/identity/C_par2_sf42_options.txt`).
3. **Seed-lattice rule.** The ctest `seedlattice_probe` (ctest case 43, V100: Passed, 1.50 s) and its standalone run
   (`R/verification/identity/C_seedlattice_probe.txt`): **1575 cases (996 `ON_FACE`, 579 `OK`), 0 mismatches** between the Python
   predictor `seedlattice.rule` and the production `find_seed_on_face`/`make_seed_lattice`, including the cases of the specification
   (`ny = 384`, window `[0.1, 0.9]^2`, `M = 128`: `ON_FACE m2=0 m3=1 axis=3`; `M = 100` on the SF-39 grids: `OK`; full plane `[0, 1]^2`).
   Every one of the 30 planning arms of `seedlattice_plan.md` (SF-43 grids and sources, fixed source of 0.8 x the production width)
   taken alone admits `M` values (at least one in `[47, 250]`; `R/verification/seedlattice_per_arm.txt`). The `M` that is admissible on
   all arms of one amplitude is the subject of Findings for SF-43 (a).
4. **Gate 1, full `ctest` on the V100** (job `sf42-ctest`, binaries of tree `96a335dc9108`, mirror at `daf0d92856b6`, build `v100-release`): **`100% tests passed, 0 tests
   failed out of 43`**, total test time 1839.22 s (`streamfunction_operator_tests` 1350.83 s; every other case under 62 s); the new
   cases `par2_sf42_options`, `mms_dirichlet_self_check`, `seedlattice_probe` and the extended `lagrangian_ensemble_medium_b` (with the
   `--dump-field` checks) pass. Log: `R/verification/logs/sf42-ctest.log`.

## D. MMS

Verdict: **met** (`analyze.py`: `Verdict D: PASS`; the minimum observed order of the four gated metrics over the two finest grid pairs
and the four `eps` is **1.997**, threshold 1.8). Results: `R/mms/` (24 JSON, `mms_summary.{md,json}`, `logs/`); jobs `sf42-r2-build`
and `sf42-mms` (section Commands, paragraph "R2 jobs").

Instrument (`apps/mms_dirichlet/`, `R/scripts/mms/`): the driver reuses the production medium (b) Dirichlet path unchanged (multigrid
hierarchy and geometric coarsening of `K`, `build_rhs_head`, `init_head_guess`, `VarCoeffLaplacian`, `MultigridPreconditioner` 2/2
smoothing with coarse 50, `pcg_solve_external`, `compute_velocity_from_head`) and adds only the manufactured source. Geometry of medium
(b): cell-centred, `h = 1/ny`, `nx = 4 ny` (`L1 = 4`), Dirichlet `x` faces (head `L1` at `x = 0`, 0 at `x = L1`), periodic `y`, `z`.
Manufactured problem: `K = exp(eps Y_a)` with a smooth `Y_a` (three modes, amplitude `0.5 + 0.4 + 0.3`; `d ln K/dx` nonzero at both
Dirichlet faces, `|d ln K / dx|` at least 0.13 at `eps = 0.5` in the self-check), head `h = (L1 - x) + p`, `p = sin(pi x/L1) q(y,z) +
sin(2 pi x/L1) r(y,z)` (zero at the faces), source `f = -div(K grad h)` analytic. Grids `ny` = 16, 32, 64, 128, 256 (the finest is
`1024 x 256 x 256` = 6.7e7 cells), `eps` = 0.5, 1, 1.5, 2, production `pcg rtol` 1e-10, then the algebraic control at `ny` = 256 with
`rtol` 1e-12. The driver's own `--self-check` (analytic gradient and source against finite differences, boundary values, `K = 1`
linear solution reproduced to 4.7e-14 in the head and 1.2e-13 in the fluxes including the Dirichlet faces) passed at the start of the
job. The JSON field `commit` reads `unknown` (the host has no git; the mirror tree is `460c9178` plus the untracked `src/external/Par2_Core`,
sources identical to `96a335dc`, see Build).

PCG: **20 iterations at every grid and every `eps`** (the convergence check is every 10 iterations and the check at iteration 10 did not
yet reach 1e-10; at 20 the relative residual is 1.3e-18 to 1.8e-15), converged at all 24 runs. The number
of iterations does not grow with `ny` (3 to 7 multigrid levels) nor with `eps`. Wall time of the whole job 1527 s for 24 runs.

Errors of the four gated metrics (absolute; `far` = at least two cells from the Dirichlet faces):

eps = 0.5

| ny | nx | MG levels | PCG it | rel. residual | head L2 | head max (far) | interior flux L2 | flux max (far) |
|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | 6.8e-18 | 2.341e-03 | 7.673e-03 | 6.744e-03 | 2.927e-02 |
| 32 | 128 | 4 | 20 | 6.4e-17 | 5.819e-04 | 1.909e-03 | 1.678e-03 | 7.412e-03 |
| 64 | 256 | 5 | 20 | 2.5e-16 | 1.453e-04 | 4.768e-04 | 4.190e-04 | 1.851e-03 |
| 128 | 512 | 6 | 20 | 7.4e-16 | 3.630e-05 | 1.192e-04 | 1.047e-04 | 4.636e-04 |
| 256 | 1024 | 7 | 20 | 1.8e-15 | 9.074e-06 | 2.979e-05 | 2.617e-05 | 1.159e-04 |

eps = 1

| ny | nx | MG levels | PCG it | rel. residual | head L2 | head max (far) | interior flux L2 | flux max (far) |
|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | 3.6e-18 | 2.599e-03 | 8.620e-03 | 8.894e-03 | 4.536e-02 |
| 32 | 128 | 4 | 20 | 3.5e-17 | 6.456e-04 | 2.147e-03 | 2.210e-03 | 1.138e-02 |
| 64 | 256 | 5 | 20 | 1.3e-16 | 1.611e-04 | 5.379e-04 | 5.515e-04 | 2.873e-03 |
| 128 | 512 | 6 | 20 | 3.5e-16 | 4.027e-05 | 1.345e-04 | 1.378e-04 | 7.178e-04 |
| 256 | 1024 | 7 | 20 | 8.3e-16 | 1.007e-05 | 3.362e-05 | 3.445e-05 | 1.795e-04 |

eps = 1.5

| ny | nx | MG levels | PCG it | rel. residual | head L2 | head max (far) | interior flux L2 | flux max (far) |
|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | 1.9e-18 | 3.029e-03 | 9.606e-03 | 1.266e-02 | 7.446e-02 |
| 32 | 128 | 4 | 20 | 2.0e-17 | 7.519e-04 | 2.405e-03 | 3.140e-03 | 1.839e-02 |
| 64 | 256 | 5 | 20 | 7.1e-17 | 1.877e-04 | 6.035e-04 | 7.836e-04 | 4.610e-03 |
| 128 | 512 | 6 | 20 | 1.9e-16 | 4.690e-05 | 1.508e-04 | 1.958e-04 | 1.153e-03 |
| 256 | 1024 | 7 | 20 | 4.7e-16 | 1.172e-05 | 3.770e-05 | 4.895e-05 | 2.883e-04 |

eps = 2

| ny | nx | MG levels | PCG it | rel. residual | head L2 | head max (far) | interior flux L2 | flux max (far) |
|---|---|---|---|---|---|---|---|---|
| 16 | 64 | 3 | 20 | 1.3e-18 | 3.639e-03 | 1.059e-02 | 1.874e-02 | 1.224e-01 |
| 32 | 128 | 4 | 20 | 1.2e-17 | 9.034e-04 | 2.658e-03 | 4.640e-03 | 3.066e-02 |
| 64 | 256 | 5 | 20 | 3.9e-17 | 2.255e-04 | 6.676e-04 | 1.158e-03 | 7.693e-03 |
| 128 | 512 | 6 | 20 | 1.1e-16 | 5.634e-05 | 1.669e-04 | 2.892e-04 | 1.926e-03 |
| 256 | 1024 | 7 | 20 | 2.8e-16 | 1.408e-05 | 4.173e-05 | 7.230e-05 | 4.818e-04 |

Observed orders over the two finest grid pairs (gate: >= 1.8 for the four gated metrics), then the reported metrics:

| eps | pair | head L2 | head max (far) | interior flux L2 | flux max (far) |
|---|---|---|---|---|---|
| 0.5 | 64-128 | 2.001 | 2.000 | 2.001 | 1.997 |
| 0.5 | 128-256 | 2.000 | 2.000 | 2.000 | 2.000 |
| 1 | 64-128 | 2.001 | 2.000 | 2.001 | 2.001 |
| 1 | 128-256 | 2.000 | 2.000 | 2.000 | 2.000 |
| 1.5 | 64-128 | 2.001 | 2.000 | 2.001 | 1.999 |
| 1.5 | 128-256 | 2.000 | 2.000 | 2.000 | 2.000 |
| 2 | 64-128 | 2.001 | 2.000 | 2.001 | 1.998 |
| 2 | 128-256 | 2.000 | 2.000 | 2.000 | 1.999 |

minimum gated order over every eps, metric and the two finest pairs: 1.997

Reported, not gated: errors at `ny` = 256 and observed orders (pairs 64-128, 128-256) per eps:

| metric | eps 0.5: err(256), p(64-128), p(128-256) | eps 1: err(256), p(64-128), p(128-256) | eps 1.5: err(256), p(64-128), p(128-256) | eps 2: err(256), p(64-128), p(128-256) |
|---|---|---|---|---|
| head max (all cells) | 2.98e-05, 2.00, 2.00 | 3.36e-05, 2.00, 2.00 | 3.77e-05, 2.00, 2.00 | 4.17e-05, 2.00, 2.00 |
| interior flux max (all) | 1.16e-04, 2.00, 2.00 | 1.79e-04, 2.00, 2.00 | 2.88e-04, 2.00, 2.00 | 4.82e-04, 2.00, 2.00 |
| Dirichlet faces L2 | 1.97e-05, 2.00, 2.00 | 3.19e-05, 2.00, 2.00 | 5.16e-05, 2.00, 2.00 | 8.18e-05, 2.00, 2.00 |
| Dirichlet faces max | 6.87e-05, 2.00, 2.00 | 1.30e-04, 2.00, 2.00 | 2.34e-04, 2.00, 2.00 | 4.02e-04, 2.00, 2.00 |
| first layer L2 | 1.19e-05, 1.98, 1.99 | 2.01e-05, 1.97, 1.98 | 3.29e-05, 1.96, 1.98 | 5.23e-05, 1.96, 1.98 |
| first layer max | 6.84e-05, 1.99, 1.99 | 1.30e-04, 1.98, 1.99 | 2.32e-04, 1.98, 1.99 | 3.99e-04, 1.98, 1.99 |

Dirichlet faces and first interior layer (reported, not gated): the specification and decision G6 anticipated first order ("first order
by construction"). **Measured: second order**, not first. The Dirichlet-face flux errors (L2 and max) have observed order 2.00 over
the two finest pairs of every `eps` (1.96 to 2.02 over all consecutive pairs), and the first interior layer approaches 2 from below (1.89 to 1.98 over the coarsest pairs, 1.98 to 1.99 over
the finest). The manufactured `K` has `d ln K/dx != 0` at both faces, so the first-order term that a face coefficient taken at the cell
centre would leave is present in the test; yet the face flux error is `O(h^2)`. This is what the cell balance gives for this scheme (the
face flux equals the first interior face flux plus `h` times the discretely-satisfied balance residual, so a second-order interior flux
yields a second-order Dirichlet-face flux); this is an argument consistent with the data, not a separate test. The Dirichlet-face
absolute errors are of the size of the interior ones (at `ny` = 256, `eps` = 2: 8.2e-5 in L2 against 7.2e-5 for the interior fluxes).

Algebraic-error control (`rtol` 1e-12 against 1e-10 at `ny` = 256, all four `eps`): the two runs stop at the same iteration (20) with
the same iterate, so every reported error is **bitwise equal** (`analyze.py`: maximum relative change 0.00 for all four `eps`). The
control is therefore not an independent tighter solve in the sense of a different iterate; the relevant statement is the final relative
residual of the production-tolerance run, 3e-16 to 1.8e-15 at `ny` = 256, far below the smallest relative discretization error of the data
(3.5e-6 relative head L2 error at `eps` = 0.5). The algebraic error does not contaminate the observed orders.

What transfers to medium (b) and what does not (decision G6): the instrument exercises the production operator, multigrid
preconditioner, PCG at the production tolerance and velocity routine in medium (b) geometry (aspect 4, Dirichlet in `x`, periodic in
`y`, `z`), with a coefficient that varies at the Dirichlet faces; the head and the interior fluxes of that discretization are second order
in this smooth regime, and the Dirichlet-face and first-layer fluxes are too. It does not transfer: the roughness of a Gaussian
random field. The manufactured `|ln K|` is at most `1.2 eps` (2.4 at `eps` = 2) with one wavelength of `Y_a` over a transverse period
(16 to 256 cells), whereas an SF-18 field at `sigma^2` = 4 has `|ln K|` of several units in 10^7 cells and `lambda/h` 4 to 64 for
`ell` = 1/4; the under-resolved range (`lambda/h` below 8) and the spline representation are covered by the spectral cross-check
(E) and the SF-39 ladders, not here. A clean MMS order says the scheme is consistent and that PCG does not pollute it at the
production tolerance; it does not bound the discretization error of a given random realization.

## E. Spectral cross-check

Verdict: **met** (`summary.md`: `summary: PASS`; all 8 ladders pass: 4 `sigma^2` x 2 seeds). Results: `R/spectral/` (`summary.{md,json}`,
`cases/*.json`, FV `report.json` and exit codes, sha256 list of the FV dumps, `logs/`); jobs `sf42-spectral-fv` (GPU) and
`sf42-spectral-ref` (CPU, 40 workers) (section Commands, paragraph "R2 jobs"). No `scan` was needed: the reference pairs chosen by N3 passed the
self-convergence criterion at every `sigma^2` on the first attempt, so there is one attempt per case and no rerun.

Design as run: `sigma^2` = 0.25, 1, 2.25, 4 (`eps` = 0.5, 1, 1.5, 2), seeds 420001 and 420002 (the first two of the SF-42 range 420001-420099), unit box, `ell` = 1/4 (box = 4 `lambda`), FV grids `N` = 16, 32, 40, 64, 128, 256 (`lambda/h` = 4, 8,
10, 16, 32, 64) of the SF-19 `AffinePeriodicFlowSolver` (driver `apps/sf42_periodic_flow`, SF-18 field normalised to `var Y` = 1, `K =
exp(eps Y)`, `rtol` 1e-12, `max_iter` 4000), compared on the exact face averages of the pseudo-spectral reference (finer of the pair `M`,
`2M`): `M` = 64 for `sigma^2` 0.25 and 1, 96 for 2.25, 128 for 4. The reference solves the same periodic cell problems
(`K^{-1/2} (-Delta)^{-1} K^{-1/2}` preconditioner, Galerkin band `|m| <= M/4`, three correctors combined to `K_eff G = qbar`, true
relative residual below 1e-12 enforced by a hard failure) on the continuous SF-18 field generated on a 64^3 mode set (the generator's
modes, not the FV sample). Error = relative L2 norm of the difference of all face fluxes (the `x`-face norm is recorded too).

Acceptance E, per ladder (all 8): FV PCG converged at all 48 levels (exit code 0 everywhere; 10 or 20 iterations per corrector at
`rtol` 1e-12 with the check every 10 iterations); the error is strictly decreasing along the whole ladder; observed order over the
finest pair (128 -> 256) is 2.00 in every ladder (all consecutive pairs 1.94 to 2.00, the lowest being the pair 16 -> 32 at `sigma^2` 4,
seed 420002); the reference self-convergence (`M` against `2M` on the same faces) is at least `3.15e3` (smallest ratio: `sigma^2` = 1, seed
420001, `N` = 256) times below the FV error against the threshold of 10 (every ratio, the smallest per ladder in the reference table).

Relative L2 error of all FV face fluxes along the ladder (`lambda/h` = 0.25 `N`) and observed orders of the two finest pairs:

| sigma^2 | seed | N=16 (lh 4) | N=32 (lh 8) | N=40 (lh 10) | N=64 (lh 16) | N=128 (lh 32) | N=256 (lh 64) | p(128-256) | p(64-128) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 420001 | 9.959e-03 | 2.522e-03 | 1.616e-03 | 6.325e-04 | 1.582e-04 | 3.957e-05 | 2.00 | 2.00 |
| 0.25 | 420002 | 6.882e-03 | 1.739e-03 | 1.114e-03 | 4.359e-04 | 1.091e-04 | 2.727e-05 | 2.00 | 2.00 |
| 1 | 420001 | 2.641e-02 | 6.731e-03 | 4.318e-03 | 1.691e-03 | 4.233e-04 | 1.059e-04 | 2.00 | 2.00 |
| 1 | 420002 | 1.708e-02 | 4.358e-03 | 2.796e-03 | 1.095e-03 | 2.741e-04 | 6.855e-05 | 2.00 | 2.00 |
| 2.25 | 420001 | 4.622e-02 | 1.183e-02 | 7.593e-03 | 2.976e-03 | 7.451e-04 | 1.863e-04 | 2.00 | 2.00 |
| 2.25 | 420002 | 2.863e-02 | 7.376e-03 | 4.738e-03 | 1.859e-03 | 4.656e-04 | 1.165e-04 | 2.00 | 2.00 |
| 4 | 420001 | 6.584e-02 | 1.695e-02 | 1.089e-02 | 4.269e-03 | 1.069e-03 | 2.675e-04 | 2.00 | 2.00 |
| 4 | 420002 | 4.034e-02 | 1.051e-02 | 6.759e-03 | 2.655e-03 | 6.657e-04 | 1.665e-04 | 2.00 | 2.00 |

Recorded errors at `lambda/h` = 10, 16 and 32 (relative L2, all faces and `x`-faces only), per `sigma^2` and seed:

| sigma^2 | seed | lh 10 all | lh 10 x | lh 16 all | lh 16 x | lh 32 all | lh 32 x |
|---|---|---|---|---|---|---|---|
| 0.25 | 420001 | 1.616e-03 | 1.479e-03 | 6.325e-04 | 5.788e-04 | 1.582e-04 | 1.448e-04 |
| 0.25 | 420002 | 1.114e-03 | 9.539e-04 | 4.359e-04 | 3.732e-04 | 1.091e-04 | 9.336e-05 |
| 1 | 420001 | 4.318e-03 | 4.045e-03 | 1.691e-03 | 1.584e-03 | 4.233e-04 | 3.964e-04 |
| 1 | 420002 | 2.796e-03 | 2.433e-03 | 1.095e-03 | 9.531e-04 | 2.741e-04 | 2.386e-04 |
| 2.25 | 420001 | 7.593e-03 | 7.222e-03 | 2.976e-03 | 2.829e-03 | 7.451e-04 | 7.083e-04 |
| 2.25 | 420002 | 4.738e-03 | 4.225e-03 | 1.859e-03 | 1.657e-03 | 4.656e-04 | 4.152e-04 |
| 4 | 420001 | 1.089e-02 | 1.040e-02 | 4.269e-03 | 4.075e-03 | 1.069e-03 | 1.021e-03 |
| 4 | 420002 | 6.759e-03 | 6.182e-03 | 2.655e-03 | 2.429e-03 | 6.657e-04 | 6.091e-04 |

Reference and comparison data per ladder (reference pair `M` / `2M`, retained band `mcut` = `M/4`, PCG iterations of the three correctors, wall time of the `M` and `2M` solves on 40 workers, smallest ratio FV error / self-convergence over the ladder with its `N`, self-convergence at `N` = 256, FV iterations per corrector, spread of the applied scale across the ladder, largest generator mismatch, field variance outside the modes of the `N` = 16 grid):

| sigma^2 | seed | M / 2M | mcut M / 2M | PCG its M (3 correctors) | PCG its 2M | true rel. residual 2M (max) | wall M / 2M [s] | min FV/selfconv (N) | selfconv at N=256 | FV its (min-max) | applied-scale spread | max dY/std (all N) | mode var. outside FV modes (N=16) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 420001 | 64 / 128 | 16 / 32 | 22/22/21 | 22/22/21 | 8.7e-13 | 2 / 11 | 8.03e+04 (256) | 4.93e-10 | 10-20 | 8.8e-16 | 5.3e-15 | 3.7e-17 |
| 0.25 | 420002 | 64 / 128 | 16 / 32 | 19/19/19 | 19/19/19 | 8.1e-13 | 1 / 10 | 8e+05 (256) | 3.41e-11 | 10-20 | 3.7e-16 | 2.7e-15 | 1.5e-17 |
| 1 | 420001 | 64 / 128 | 16 / 32 | 51/50/51 | 46/45/46 | 9.8e-13 | 3 / 22 | 3.15e+03 (256) | 3.36e-08 | 20-20 | 8.8e-16 | 5.3e-15 | 3.7e-17 |
| 1 | 420002 | 64 / 128 | 16 / 32 | 34/34/34 | 32/32/32 | 2.8e-13 | 2 / 15 | 1.09e+04 (256) | 6.28e-09 | 10-20 | 3.7e-16 | 2.7e-15 | 1.5e-17 |
| 2.25 | 420001 | 96 / 192 | 24 / 48 | 171/168/171 | 166/162/164 | 8.2e-13 | 30 / 500 | 3.14e+06 (256) | 5.93e-11 | 20-20 | 8.8e-16 | 5.3e-15 | 3.7e-17 |
| 2.25 | 420002 | 96 / 192 | 24 / 48 | 81/80/80 | 80/80/78 | 9.8e-13 | 14 / 237 | 1.78e+07 (256) | 6.54e-12 | 20-20 | 3.7e-16 | 2.7e-15 | 1.5e-17 |
| 4 | 420001 | 128 / 256 | 32 / 64 | 701/694/699 | 714/728/722 | 9.9e-13 | 313 / 5634 | 7.56e+08 (256) | 3.54e-13 | 20-20 | 8.8e-16 | 5.3e-15 | 3.7e-17 |
| 4 | 420002 | 128 / 256 | 32 / 64 | 230/229/227 | 248/246/244 | 1.0e-12 | 102 / 1895 | 1.48e+08 (256) | 1.13e-12 | 20-20 | 3.7e-16 | 2.7e-15 | 1.5e-17 |

Mean-gradient `G` and effective conductivity: FV (`N` = 256) against the reference (`2M`):

| sigma^2 | seed | G (reference) | max rel diff G_FV (N=256) | rel diff K_eff,xx |
|---|---|---|---|---|
| 0.25 | 420001 | (0.948244, 0.004477, 0.002157) | 1.56e-05 | 1.56e-05 |
| 0.25 | 420002 | (0.977860, -0.023348, 0.008525) | 1.18e-05 | 1.20e-05 |
| 1 | 420001 | (0.795830, 0.008912, 0.003211) | 5.25e-05 | 5.26e-05 |
| 1 | 420002 | (0.937603, -0.067935, 0.019528) | 4.05e-05 | 4.20e-05 |
| 2.25 | 420001 | (0.581022, 0.007611, 0.001317) | 9.63e-05 | 9.65e-05 |
| 2.25 | 420002 | (0.897104, -0.098995, 0.017900) | 7.55e-05 | 7.88e-05 |
| 4 | 420001 | (0.369174, 0.003821, -0.000814) | 1.38e-04 | 1.38e-04 |
| 4 | 420002 | (0.853363, -0.103655, 0.007424) | 1.12e-04 | 1.16e-04 |

Effective conductivity error (FV against reference) at coarser grids, and the FV cost:

| sigma^2 | seed | K_eff,xx rel. diff at lh 10 / 16 / 32 | FV iterations N=256 | FV flow seconds N=256 (V100) |
|---|---|---|---|---|
| 0.25 | 420001 | 6.35e-04 / 2.49e-04 / 6.23e-05 | [20, 20, 20] | 4.3 |
| 0.25 | 420002 | 4.90e-04 / 1.92e-04 / 4.80e-05 | [20, 20, 20] | 4.3 |
| 1 | 420001 | 2.14e-03 / 8.40e-04 / 2.10e-04 | [20, 20, 20] | 4.3 |
| 1 | 420002 | 1.71e-03 / 6.70e-04 / 1.68e-04 | [20, 20, 20] | 4.3 |
| 2.25 | 420001 | 3.91e-03 / 1.54e-03 / 3.86e-04 | [20, 20, 20] | 4.3 |
| 2.25 | 420002 | 3.20e-03 / 1.26e-03 / 3.15e-04 | [20, 20, 20] | 4.3 |
| 4 | 420001 | 5.58e-03 / 2.20e-03 / 5.52e-04 | [20, 20, 20] | 4.3 |
| 4 | 420002 | 4.69e-03 / 1.85e-03 / 4.63e-04 | [20, 20, 20] | 4.3 |

Checks of the comparison itself. Applied scale: the GPU generator's normalisation factor (`applied_scale`, `var Y` = 1) is the same at
every `N` of a seed to 8.8e-16 (seed 420001) and 3.7e-16 (seed 420002) relative, and the Python port of the SF-18 generator reproduces the
GPU `Y` dump at every level to 5.3e-15 of `std Y` at most, so the FV and the reference see the same field and the same normalisation; the
variance of the 64^3-mode field outside the modes of an FV grid (grids `N` < 64 differ by modes absent from the FV sample) is
at most 3.7e-17 (`N` = 16), negligible above. The effective conductivity scale that carries to the flux amplitude: relative difference of `K_eff,xx`
(FV against reference) is in the K_eff table. The self-convergence is also essentially a band statement here: the reference's self-convergence
is 3.4e-8 or better in all cases (3.5e-13 at `sigma^2` = 4, seed 420001), plausibly because the Gaussian-covariance field has a spectrum that decays faster than the retained bands; the
criterion shows the reference is orders of magnitude below the FV error for these smooth fields; it does not by itself show the FV is right (the FV error
decreases at order 2 against the same reference, which is the check).

What transfers to medium (b) and what does not (decision G6): what transfers is the interior scheme error of the FV discretization of
Darcy flow (the same stencil as `solve_head`, in the SF-19 operator class) as a function of `lambda/h` and `sigma^2` on SF-18
fields up to `sigma^2` = 4: relative L2 flux error from 1.1e-3 to 1.6e-3 at `lambda/h` = 10 for `sigma^2` = 0.25 up to 6.8e-3 to
1.1e-2 at `sigma^2` = 4, falling 4x per refinement, with the effective conductivity `K_eff,xx` error at `lambda/h` = 10 between 4.9e-4
and 5.6e-3. What does not transfer: the Dirichlet inlet/outlet boundary treatment (covered by the MMS, D), the spline representation of
the flow used by the consumer (SF-28; its divergence is measured elsewhere), the long-box end effects, and the particular periodic
affine problem (the SF-19 flow with mean gradient `G`; its streamlines do not close on the torus, but its velocity is a well-defined
function of `K` that both codes must reproduce). The FV error here is a flux-field error on a periodic cell problem, not an error of
`alpha_L`, and nothing here is evidence on `alpha_T`.

Independence: the reference does not share the FV solver nor its discretization; it shares the SF-18 generator (a Python port,
matched to the GPU to 5e-15 of `std Y`) and the definition of the problem, so a defect common to the generator definition would not
appear here. The reference band is `M/4` (half the Nyquist wavenumber of `M`), set by N3 against aliasing of the `K` spectrum tail;
the criterion E is evaluated with the unchanged threshold of 10.

## F. Statistics library and coverage

**Library on the host stack (R3).** The R3 tree contains the whole `mcstat` package (N6a, N6b coverage simulator, N6c studentized
bounds and bootstrap-t Gate 5 TOST). `cd R/scripts && OMP_NUM_THREADS=1 python3 -W error -m unittest discover -s . -t .` ends
`Ran 309 tests in 29.067s`, `OK (skipped=1)` locally (Python 3.13, numpy 2.5, scipy 1.18; the skipped test is the host-only one) and, on
the host stack (Python 3.11.7, numpy 1.26.4, scipy 1.11.4, job `sf42-hosttests`), `Ran 309 tests in 37.532s`, `OK` with **no skip**:
`mcstat.tests.test_sf39_reproduction.Par2PerRealization.test_s025_gate5_from_per_realization_values ... ok`.

The input of that test, `~/sf42_campaign/r3/par2_alphax_s025.txt`, was produced in the same job by `R/verification/par2_alphax_s025.py`,
which applies the SF-39 package's own functions (`sf39.par2.pooled`, `uc_mean`, `window`, `alpha_window`, read-only from the artifact) in
the way `sf39.analyze.gate5_block` produced `summary.json` `amplitudes.s025.C.par2.alpha_x`: pooled pilot + extension realizations
(`~/sf39_campaign/par2/raw/par2_s025` and `_ext`, first `n` = 25), `U_c` at the frozen `t_b,min` = 729712.5 with anchor `inject_x` = 250,
window `[0.75, 1] tau_c lam / U_c` with the `tau_c` and `lam` = 50 of `tau_b_star_s025.json`. It printed `n=25`,
`U_c=0.010191770851862745`, `window=[547236.3859, 729648.5145]`, `mean=0.27019457195374413`. The test then checks, at 1e-9 relative,
the count, mean and standard deviation of these 25 values against the committed record, and the Gate 5 primary Welch bounds computed from
the recorded consumer summary and these per-realization values against `gate5.primary.lo/hi` (at 5e-9, Deviation 3); all pass, so the
committed Par2 Gate 5 numbers are reproduced from the raw Par2 series on the host. The procedure is the reducer's own code path
(identified by reading `gate5_block`, from `ents = par2_entries(C, amp)[:n2]` to `P = np.array([a["x"] for a in al])`), not an
approximation.

Coverage simulation: pending (sf42-coverage).

## G. Design pilots

### Estimator noise by basis (R3, no new runs)

Question (a design input for SF-43, not evidence about `alpha_T`): in which basis, positions `pos` or displacements `disp`, do the
control-plane estimators `alpha_TP` and `alpha_TAP` have the lower noise at the SF-39 production sets? Data: the SF-39 consumer arrays,
read-only, reduced with `python3 -m sf42red reduce` (job `sf42-noise`, `R/verification/noise_job.sh`): production set of an amplitude =
every DONE realization of `<a>_A` (pass `hmax_8`) and `<a>_B` (single pass) = 25, 50, 62, 71 realizations at `sigma^2` 0.25, 1, 2.25, 4
(the sizes of the committed `*_prod` sets). Window: the G4 default `d in [25, 94]` lambda is covered by the SF-39 planes (first window
plane `d` = 25, or 24.999999999999996 at `sigma^2` 2.25 and 4, inside the 1e-9 tolerance; last 94; 71 window planes, 70 at `sigma^2` 4), so
no sub-window was needed. Weights `flux`, set `perplane`. Mean and SD (ddof = 1) across realizations; split-half particle share =
`sf42red` `splithalf_fraction` (mean over realizations of `(a_A - a_B)^2/4` over the sample variance; flag when above 0.25: never raised).
Table: `R/verification/noise_summary.csv` (made by `noise_summary.py` from the host tables, which stay on the host in
`~/sf42_campaign/r3/noise/`).

| `sigma^2` | n | estimator | mean pos | SD pos | share pos | mean disp | SD disp | share disp | SD ratio pos/disp |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 25 | `alpha_TP` | -1.877e-4 | 4.476e-3 | 4.0e-7 | 5.093e-4 | 1.206e-4 | 2.7e-6 | 37.13 |
| 0.25 | 25 | `alpha_TAP` | -1.877e-4 | 4.476e-3 | 4.0e-7 | 5.093e-4 | 1.206e-4 | 2.7e-6 | 37.13 |
| 1 | 50 | `alpha_TP` | 6.237e-3 | 1.103e-2 | 4.2e-4 | 7.217e-3 | 1.121e-3 | 2.0e-3 | 9.84 |
| 1 | 50 | `alpha_TAP` | 6.237e-3 | 1.103e-2 | 4.2e-4 | 7.217e-3 | 1.121e-3 | 2.0e-3 | 9.84 |
| 2.25 | 62 | `alpha_TP` | 2.833e-2 | 1.514e-2 | 7.1e-3 | 2.946e-2 | 4.073e-3 | 1.5e-2 | 3.72 |
| 2.25 | 62 | `alpha_TAP` | 2.833e-2 | 1.514e-2 | 7.1e-3 | 2.946e-2 | 4.073e-3 | 1.5e-2 | 3.72 |
| 4 | 71 | `alpha_TP` | 6.648e-2 | 1.918e-2 | 1.8e-2 | 6.538e-2 | 1.688e-2 | 1.5e-2 | 1.14 |
| 4 | 71 | `alpha_TAP` | 6.646e-2 | 1.917e-2 | 1.8e-2 | 6.537e-2 | 1.687e-2 | 1.5e-2 | 1.14 |

(Full precision in `noise_summary.csv`.) Reading as a design input: the displacement basis has the lower across-realization SD at every
amplitude, by a factor of 37, 9.8, 3.7 and 1.14 from `sigma^2` 0.25 to 4 (the gain shrinks as the amplitude grows). The split-half particle
share is below 2 % in both bases (far from the 0.25 flag), so the across-realization SD is realization-to-realization variation, not
particle noise. `alpha_TP` and `alpha_TAP` agree within 4e-4 relative in mean and SD for the same basis. No claim is made on the sign or
the size of `alpha_T` here (and no mechanism is claimed for the basis difference); the means are listed only to read the SDs against them.
The pilot has one realization set per amplitude and no uncertainty on the SDs themselves.

Other G parts: pending (later nodes).

## H. Reproducibility tooling

**Independent recomputation (R3, job `sf42-recompute`, `python3 -m recompute.sf39`).** From the runner's raw moment files of the host
tree (`~/sf39_campaign`, read-only), three realizations per amplitude (positions 0, 1, 2 of the seed-sorted committed `*_prod` set: seeds
5001, 5002, 5003 of `s025_A`, `s1_A`, `s225_A`, `s4_A`, pass `hmax_8`) were recomputed and compared with `analysis/reduced/*.csv` at
`rtol` = 1e-10 (`atol` = 1e-15): 46 values per realization, **596 values compared, 0 beyond tolerance, maximum relative difference
8.408e-12** (per realization 9.3e-14 to 8.4e-12; the three `s025` realizations 2.5e-12, 8.4e-12, 7.4e-12 are the largest), no column
needing per-particle arrays. The mean of the committed `qbar` of every set reproduces the committed `U` (`OK` at the four amplitudes).
The Par2 Gate 5 window values (`U_c`, window, mean and SD of `alpha_x`, `alpha_y`, `alpha_z`, `alpha_T_fixed` over all `N_R` Par2
realizations: 11 values per amplitude) agree with `summary.json` with maximum relative difference **0** at the four amplitudes.
`RECOMPUTE PASS` (acceptance: at least three realizations per amplitude to 1e-10 relative: met). Report:
`R/verification/recompute_r3_report.txt`.

**Solver-verdict reconciliation from the host logs (R3, job `sf42-reconcile`, `python3 -m reconcile.solver`).** Input: the 20 job logs of
SF-39's `scripts/remote` state on the host (`~/.macroflow3d-remote/macroflow3d-SF-39/logs`; md5-identical to the 20 `*.log` files committed
under `SF39/logs/job_logs`), cross-checked against every `solver_b.json` under `~/sf39_campaign/runs` (read-only). Result: **635 medium (b)
solves in 47 arms, 4 not converged, 0 conflicts, 0 unmapped, 0 unparsed**; the four are exactly the expected ones
(`SF39_EXPECTATION MATCH (4 expected non-converged solves)`): `s4_B` r=2 seed 5023 `fallback` (rel 3.04e-10, balance 8.10e-11), `s4_tw`
r=22 seed 5023 `fallback` (2.20e-10, 1.46e-11), `s4_w12` r=4 seed 7205 `fallback` (9.88e-10, 2.14e-9), `s4_w12` r=13 seed 7214
`not_converged_beyond_fallback` (3.39e-7, 2.64e-7). Totals by verdict: 631 `converged`, 3 `fallback`, 1 `not_converged_beyond_fallback`.
Host cross-check: `HOST_CHECK checked=614 mismatches=0 logged_solves_without_solver_b_json=19`; the census of the `solver_b.json` files is
`converged=611, fallback=3`. `max_div` is missing (line cut at about 200 characters) in 632 of the 635 logged lines, so the finiteness
test uses `rel`, `balance`, `qbar`, `Q_in`, `Q_out`. `RECONCILE OK`. Report: `R/verification/reconcile_r3_report.txt`.

## Findings for SF-43

(a) **Seed lattice.** With the source fixed at 0.8 x the production width (decision G4: 19.2 `lambda` at `sigma^2` 0.25, 1 and 2.25;
9.6 `lambda` at 4) and the SF-43 planning grids, the `M` admissible on every arm of an amplitude is only `M = 128` at `sigma^2` 0.25 and
1 (1.37 x SF-39's seed density), none at 2.25 (the `lambda/h` = 10 arms, whose `ny` is a multiple of 5, conflict with the production
`ny` = 384), and 51 values at 4 (58 / 100 / 202 nearest 0.6x / 1x / 2x). At 2.25 a source of 19.1 or 19.3 `lambda` admits 101 common
`M` in `[50, 250]` (orchestrator scan with `seedlattice.rule`); the alternatives are an `E_W` `lambda/h` without factor 5 or a per-arm
`M`. This is a choice for SF-43's pre-registration, and possibly the owner's, since G4 fixes 0.8 x the width. Data:
`R/scripts/seedlattice/data/seedlattice_plan.md`, `seedlattice_plan.json`.

(b) **MMS smoke.** In the MMS smoke (`ny` 16 to 64, `eps` 1) the Dirichlet faces and the first interior layer converge at order about
2, not about 1 as the specification anticipated (reported, not gated; the full ladder is in D).
Full ladder (R2, `ny` 16 to 256, four `eps`): the Dirichlet-face fluxes (L2 and max) converge at **order 2.00 over the two finest pairs
at every `eps`** (1.96 to 2.02 over all consecutive pairs) and the first interior layer at 1.98 to 1.99 over the finest pair, with
absolute errors of the size of the interior fluxes (at `ny` = 256, `eps` = 2: 8.2e-5 against 7.2e-5 in L2). The statement "first order
by construction" of decision G6 is not supported by the measurement for a smooth head and a coefficient with `d ln K/dx != 0` at the
faces; SF-43 should not budget a first-order boundary term for the production Dirichlet fluxes on that basis, with the caveat that
the manufactured head is smooth and that an MMS cannot bound the error of a random realization.
(c) **FV error scale (E).** Relative L2 error of the FV face fluxes against the pseudo-spectral reference, second order from `lambda/h` =
4: at `lambda/h` = 10 it is 1.1e-3 to 1.6e-3 (`sigma^2` 0.25) up to 6.8e-3 to 1.1e-2 (`sigma^2` 4), at 16 it is 4.4e-4 to 6.3e-4 up to
2.7e-3 to 4.3e-3, at 32 it is 1.1e-4 to 1.6e-4 up to 6.7e-4 to 1.1e-3 (two seeds each); the effective conductivity `K_eff,xx` is off
by 4.9e-4 to 5.6e-3 at `lambda/h` = 10. The constant grows with `sigma^2` by about 6 to 7 from 0.25 to 4 at fixed `lambda/h`; SF-43 can use
these as an input to the `E_h` budget (they are the interior scheme error on a periodic affine problem, not the error of an
estimand).

## Conclusion

Pending (later nodes).

## Caveats

Pending (later nodes).

## Deviations

1. `scripts/remote` sends the blinded-phase guard statement on every `exec`/`run` (a no-op without the host marker; command text, cwd,
   environment, output and exit status unchanged, tested): the specification asked for unchanged semantics, satisfied.
2. The Par2 option `analysis.plane_crossings.dump_injection` was added (opt-in, validated) to verify the window and the displacement
   moments.
3. The `mcstat` reproduction of SF-39's Gate 5 Welch bounds is tested at 5e-9 relative instead of 1e-9: the committed host numbers carry
   scipy 1.11.4's Student-quantile error (1.4e-9 relative at nu = 38.449); quantile-free values reproduce at <= 1e-12.
4. The specification's statement "powers of two fail whenever 3 divides `ny`" is not general (for example `ny` = 96 admits `M` = 128,
   `ny` = 384 admits `M` = 512); the cited case `ny` = 384, `M` = 128 fails as stated.
5. The Par2 secondary time window ends at the last snapshot with no particle beyond an explicit `X_exit` (Par2 maps Dirichlet faces to
   Closed, so `active` never decreases).
