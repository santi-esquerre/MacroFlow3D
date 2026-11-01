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
- Sections D and E (the MMS order study and the pseudo-spectral cross-check, second remote campaign R2) are recorded, and sections F to H:
  the statistics library and the coverage simulation (job `sf42-coverage`, R5 / R5b), the design pilots (R4) and the reproducibility tooling
  (R3), with the findings for SF-43 that depend on them.

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

R4 jobs (fourth campaign, same mirror, from the R4 trees `4b2593cd` (J0 to J3) and `55d43300` (J3B) = accepted head `4d54fa7c` + the
pilot tooling; `R/verification/jobs_r4.tsv`, logs in `R/verification/logs/`). One job at a time, each with `REMOTE_GPU_WAIT=7200`
(the CUDA ones also `REMOTE_GPU_MIN_FREE_MIB=30000`), no `--gpu`; `scripts/remote run` chose GPU 0 every time (idle host, 32510 MiB
free). Host outputs only under `~/sf42_campaign/pilots/`; `find ~/sf39_campaign -newer ~/sf42_campaign/r3/.start -print | head -20`
appended to `job.sh` printed an empty list (`SF39_UNTOUCHED_CHECK_DONE` straight after `DRIVER_EXIT=0`) in all four CUDA jobs. Binaries of
`sf42-r4-build` (md5): `lagrangian_ensemble` `ccef921eb383a5e144b7e12c256fdeda`, `macroflow3d_pipeline`
`ce441c3ad49575df419e2ce381dd85fe` (sources identical to the R1 to R3 builds; no production code changed in R4). The 330 Python tests of
`R/scripts` at tree `4b2593cd` passed on the host stack inside that job: `Ran 330 tests in 38.923s`, `OK (skipped=1)`.

```bash
# R4, R = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling
scripts/remote --increment SF-42 sync
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-r4-build -- "cd ~/MacroFlow3D-SF-42 && export TMPDIR=\$HOME/.cache/mf3d-tmp CMAKE_BUILD_PARALLEL_LEVEL=16 && cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j --target lagrangian_ensemble macroflow3d_pipeline && md5sum ... && (cd $R/scripts && OMP_NUM_THREADS=1 python3 -W error -B -m unittest discover -s . -t .)"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-pilot-grid -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/pilots/job.sh J1"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-pilot-particles -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/pilots/job.sh J2 --reduce"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-pilot-pairing -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/pilots/job.sh J3 --reduce"
scripts/remote --increment SF-42 sync          # tree 55d43300 (adds the J3B run table)
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-pilot-horizon -- "cd ~/MacroFlow3D-SF-42 && bash $R/scripts/pilots/job.sh J3B --reduce"
# local (R4b, no remote use): regenerate and tabulate from the committed R/pilots
cd $R/scripts && python3 -m pilots.report j1|j2|j3|j3b --data ../pilots --out <dir> && python3 -m pilots.report cost --data ../pilots --out <dir> --repo <repo root>
python3 -m pilots.tables ../pilots j1|j2|j2m|j2cross|j3|cost_fit|cost_new|resid|proj|proj_rows|stop
```

| job | start (UTC) | end (UTC) | wall (s) | GPU-h | exit | GPU |
|---|---|---|---|---|---|---|
| `sf42-r4-build` (CPU: incremental build of the two binaries, 330 Python tests on the host stack) | 2026-10-30 09:58:54 | 10:04:26 | 332 | 0.092 | 0 | 0 |
| `sf42-pilot-grid` (J1, CUDA: 9 consumer + 3 Par2 runs) | 10:05:42 | 10:14:57 | 555 | 0.154 | 0 | 0 |
| `sf42-pilot-particles` (J2, CUDA: 12 runs of 6 realizations, reduction) | 10:17:11 | 11:46:37 | 5366 | 1.491 | 0 | 0 |
| `sf42-pilot-pairing` (J3, CUDA: 4 + 4 runs of 8 realizations, reduction) | 11:48:01 | 12:13:55 | 1554 | 0.432 | 0 | 0 |
| `sf42-pilot-horizon` (J3B, CUDA: 3 Par2 runs of 8 realizations, reduction; Deviation 6) | 12:20:27 | 12:27:33 | 426 | 0.118 | 0 | 0 |
| total R4 | | | 8233 | 2.287 | | |

Start and end are the first and last timestamps of the job log (slot held). R4's local budget rule (J1 to J3 projected above 8 GPU-h
would cut the realization counts) did not bite: 1.69 GPU-h projected, 2.08 measured.

R5 job (fifth campaign, same mirror, tree `67548000` = the R3 head plus `R/verification/job_r5.sh`; the R4 chain head `8422b2e2` contains
the same `mcstat` package, unchanged since R3; `R/verification/jobs_r5.tsv`, trimmed log `R/verification/logs/sf42-coverage.log`).
CPU only; `scripts/remote run` nevertheless assigned GPU 0 and held the single project slot for the whole run (the GPU stayed idle:
0 MiB, 0 %), so its GPU-h are wall hours holding the slot, the convention of this note. The interrupted R5 worker (Deviation 10) wrote
`job_r5.sh` because the agent sandbox rejected `cd` inside the `run` text; the job itself was not touched afterwards.

```bash
# R5 (flags as for every job of this note; the launch environment is not part of the job record given to R5b)
scripts/remote --increment SF-42 sync          # tree 67548000
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-coverage -- "bash $R/verification/job_r5.sh 72"
scripts/remote --increment SF-42 wait sf42-coverage
# job_r5.sh: (cd $R/scripts && OMP_NUM_THREADS=1 MCSTAT_PAR2_ALPHAX_S025=~/sf42_campaign/r3/par2_alphax_s025.txt python3 -W error -m unittest discover -s . -t .),
#   then, if it passed, bash $R/scripts/mcstat/run_coverage.sh ~/sf42_campaign/coverage --profile full --workers 72
# local (R5b, no remote use): corrected summary and checks on a copy of the host outputs
cd $R/scripts && OMP_NUM_THREADS=1 python3 -W error -m mcstat.coverage summarize --out <copy of ~/sf42_campaign/coverage>
python3 -I $R/verification/indep_boott_unpaired.py 7 > $R/verification/indep_boott_unpaired.txt
```

| job | start (UTC) | end (UTC) | wall (s) | GPU-h | exit | GPU |
|---|---|---|---|---|---|---|
| `sf42-coverage` (CPU only: 309 host Python tests, smoke check, `full` coverage simulation on 72 workers) | 2026-10-30 16:45:15 | 2026-11-01 10:26:51 | 150096 | 41.693 | 0 | 0 |

Running total R1 to R5: 0.820 + 3.090 + 0.252 + 2.287 + 41.693 = **48.142 GPU-h** (173311 s). `~/sf39_campaign` was not touched
(`find -newer ~/sf42_campaign/r5_hosttests.log` empty at the end of the job).

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

### Coverage simulation (R5 job `sf42-coverage`, corrected report by R5b)

**What was run.** The `full` profile of `python3 -m mcstat.coverage` (decision G5.3 as a qualification report for SF-43, not a gate of
SF-42): 1134 cells, 35844 chunks, 165.1 million replicates, seed 20261030, plan hash
`0b08b6be86ce67841e71352cbed0191c4f2e7fc1e5422831f14042d705fa0bf0`; the 23 procedure labels are the sampling limit of a mean (Student,
bootstrap-t), `ContrastBound` paired / unpaired / interaction (percentile, Student, bootstrap-t), `LadderBound` (the same three), the
robust-sign pipeline K = 3 and K = 4 (all budget components with one method each), Gate 5 TOST (Student, bootstrap-t) and BCa
(reported, never judged), each on `normal`, `stress` (skewness 1.936, excess kurtosis 4.711) and its reflection `stress-`. Job
`sf42-coverage`: mirror tree `67548000` (R3 head plus `R/verification/job_r5.sh`), command `bash $R/verification/job_r5.sh 72`; host
unit tests first, `Ran 309 tests in 36.038s`, `OK` (no skip: the host-only test ran), then the smoke check (passed) and the study,
72 workers, `nice -n 10`; slot held 2026-10-30T16:45:15Z to 2026-11-01T10:26:51Z, `STATE=succeeded`, `EXIT_CODE=0`, `finished 35844
chunks in 150047 s (10782196 core-seconds)`, `summary: 1134 of 1134 cells complete; 428 verdict lines`. Host Python 3.11.7, numpy
1.26.4, scipy 1.11.4. Files: `R/coverage/` (`plan.json`, `summary.json`, `summary.md`, `env.txt`, `chunks.sha256` of the 35844 chunk files;
the chunks stay on the host in `~/sf42_campaign/coverage/chunks`), `R/verification/jobs_r5.tsv`, `R/verification/logs/sf42-coverage.log`
(trimmed). `summary.json` / `summary.md` were produced by the corrected `summarize` of this tree (R5b) from the final chunks. The host's own
summary was made by the older code (before the R5b report corrections); against it, all 1134 cells and 13734 tail rows have identical
misses, `R`, `pass`, `powered`, design effect and rate; the UCL differs by at most 8.7e-15 relative (scipy 1.11 against 1.18 beta quantile,
so the UCL is not bit-identical: 4940 of 13734 rows are, the rest differ at rounding level); the 428 verdicts have identical `qualified`,
passing and failing tails; `incomplete_dists` is empty everywhere and there are no incomplete groups. Chunk-list hash (`find chunks -type f |
LC_ALL=C sort | xargs sha256sum | sha256sum`, equal to `sha256sum R/coverage/chunks.sha256`):
`f7ae476d9858695bf2d7b83e2da3c9677792f04e89de7828c6317b7ddda35793`. `env.txt` is the host's, unchanged, and says `cpus 1`: that is the
`nproc` artefact corrected in R5b (`nproc` honours the `OMP_NUM_THREADS=1` that the script exported before the print); the host has 2 x
Xeon Gold 6138, 80 logical / 40 physical CPUs. The host `run.log` holds two `RuntimeWarning: invalid value encountered in multiply` from
`intervals.py:261` (workers of the bootstrap-t sampling cells; `0 * inf` in the unselected branch of `np.where`, values unaffected), removed
in R5b by moving the line inside `np.errstate`.

**Reproduction evidence (R5b, local stack Python 3.13 / numpy 2.5 / scipy 1.18, under `python3 -W error`).** The corrected code builds the
identical plan (`plan_hash(profile_full(), 20261030)` equals the host's; the 1134 cells equal those of `plan.json`). Recomputing host chunks
with `coverage.run_chunk(cell, j, seed)` gave `counts`, `extra`, `sumsq`, `nbatch` and `R` **exactly equal** to the host files (seconds
excluded) for 14 distinct chunks (chunk 0 of each; every procedure family except BCa): `sampling_student__base__normal__n20`, `tost_gate5__rho=0.5__stress__n20`,
`tost_gate5__rho=0.9__normal__n170`, `sampling_boott__base__stress__n20` (the family of the host warning), `contrast_paired__base__stress__n20`
(percentile, shared draws), `pipeline_K4_ip=2__bias_order=1__normal__n30`, `ladder_E_h__nonmono_ip=1_kappa=12__normal__n20` (seven, checked on a partial copy of
the host outputs while the job ran), and, on the final outputs, seven chunks of the N6c families `contrast_paired_student__base__stress__n20` (0.1 s locally, 0.4 s on the
host), `contrast_unpaired_boott__base__stress__n20` (32.8 s, 142.9 s), `ladder_E_h_student__order=1_ip=2_kappa=12__stress__n20` (17.6 s, 112.8 s),
`ladder_E_h_boott__order=1_ip=1_kappa=25__stress__n20` (39.2 s, 229.1 s), `pipeline_K4_ip=2_student__zero__stress__n30` (59.0 s, 407.9 s),
`pipeline_K4_ip=2_boott__zero__stress__n30` (42.7 s, 243.4 s), `tost_gate5_boott__rho=0__stress__n20` (35.0 s, 319.8 s). No mismatch was found; for the 10 of these chunks that took 17 s or more locally, the host took 4.4 to 9.1 times longer.

**False-fail probability of the criterion for an exactly nominal procedure (G5.3).** PASS is `UCL <= 1.25 x tail` at
`R = ceil(175 / smallest tail)`, so a procedure whose true miss rate equals the tail exactly still fails by noise alone. Computed exactly
(`coverage.nominal_fail_probability`: binomial threshold `k*` and a dynamic programme over the nested counts at the sorted tails; design effect 1):

| `R` | tails | cells | `k*` at the smallest tail (expected misses 175) | P(fail) at the smallest tail | P(some tail fails) per (cell, distribution, row) | P(some tail fails in any of the 3 distributions) |
|---|---|---|---|---|---|---|
| 24500 | 7 Holm | 234 | 195 | 0.0714 | 0.0815 | 0.2250 |
| 25000 | 7 Holm | 45 | 199 | 0.0691 | 0.0779 | 0.2159 |
| 49000 | 8 sampling | 48 | 195 | 0.0717 | 0.0819 | 0.2261 |
| 196000 | 13 component | 795 | 195 | 0.0720 | 0.0840 | 0.2314 |

(`summary.md` lists `k*` and the probability for every tail.) The earlier text that put this at "about 1e-3" was wrong by two orders of
magnitude. Consequences for reading the tables: a group `Q` (qualified) is stronger evidence than a marginal failure is evidence against.

**Results in numbers.** 13734 tail rows (cell x row x tail): 12401 PASS, 1285 FAIL, 48 reported only (BCa), none unpowered; unresolved
bootstrap tails 0 and non-finite limits 0 in all cells; largest design effect 1.079. Of the 428 verdict groups (row x scenario x `n`, the
verdict taking all tails in all three distributions), 8 are BCa (reported), 310 qualified and **110 not qualified: 101 clear** (some failing
tail with a point estimate of the miss rate above 1.25 x tail) **and 9 marginal** (every failing tail has a point estimate within 1.25 x
tail and only the 95 % limit exceeds it).

**Qualification matrix.** Entry `Q`: every tail qualifies in each of `normal`, `stress`, `stress-`; `k/t`: `k` of the `t` groups of that
procedure and `n` qualify (groups: the lower and upper limit; the two TOST boundaries x 3 `rho`; the ladder cases and production levels,
the pipeline biases and orders), suffix `m`: every non-qualified group is marginal only; blank: not simulated. The column is the sample size of the
cell (seeds per arm; `N` for the pipelines; pairs for Gate 5).

| procedure | method | 20 | 30 | 50 | 100 | 170 | 200 | 300 | 500 | 1000 | 1650 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sampling of the mean | Student | 0/2 | 0/2 | 0/2 | 0/2 |  | 0/2 |  | 0/2 | 0/2 | 0/2 |
| sampling of the mean | bootstrap-t | 0/2 | 0/2 | 0/2 | 0/2 |  | 1/2m |  | 1/2m | Q | Q |
| ContrastBound paired | Student | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1 |  |  |  |  |
| ContrastBound paired | percentile | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1 |  |  |  |  |
| ContrastBound paired | bootstrap-t | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1 |  |  |  |  |
| ContrastBound unpaired | Student | Q | Q | Q | Q |  | Q |  |  |  |  |
| ContrastBound unpaired | percentile | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1m |  |  |  |  |
| ContrastBound unpaired | bootstrap-t | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1 |  |  |  |  |
| ContrastBound interaction | Student | Q | Q | Q | Q |  | Q |  |  |  |  |
| ContrastBound interaction | percentile | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1m |  |  |  |  |
| ContrastBound interaction | bootstrap-t | 0/1 | 0/1 | 0/1 | 0/1 |  | 0/1 |  |  |  |  |
| LadderBound | Student | Q | Q | Q | Q |  | Q |  |  |  |  |
| LadderBound | percentile | Q | Q | Q | Q |  | Q |  |  |  |  |
| LadderBound | bootstrap-t | Q | Q | Q | Q |  | Q |  |  |  |  |
| robust-sign pipeline K = 3 | Student |  |  |  | Q |  |  | Q |  | Q |  |
| robust-sign pipeline K = 3 | percentile |  |  |  | Q |  |  | Q |  | Q |  |
| robust-sign pipeline K = 3 | bootstrap-t |  |  |  | Q |  |  | Q |  | Q |  |
| robust-sign pipeline K = 4 | Student |  | Q |  | Q |  |  | Q |  |  |  |
| robust-sign pipeline K = 4 | percentile |  | Q |  | Q |  |  | Q |  |  |  |
| robust-sign pipeline K = 4 | bootstrap-t |  | Q |  | Q |  |  | Q |  |  |  |
| Gate 5 TOST | Student | 2/6 | 2/6 | 0/6 | 0/6 | 0/6 |  |  |  |  |  |
| Gate 5 TOST | bootstrap-t | 2/6 | 2/6 | 2/6 | 0/6 | 1/6m |  |  |  |  |  |

**By distribution: the largest miss rate over the tails, divided by the tail.** `*` the cell fails with a point estimate above 1.25 (clear),
`~` it fails only at the 95 % limit (point estimate within 1.25, marginal), no mark: every tail passes. Limit of the mean (R = 49000,
tails 3.6e-3 to 0.05):

| method | limit | distribution | n=20 | n=30 | n=50 | n=100 | n=200 | n=500 | n=1000 | n=1650 |
|---|---|---|---|---|---|---|---|---|---|---|
| Student | lower | normal | 1.01 | 1.05 | 1.08 | 1.11 | 1.16~ | 1.05 | 1.02 | 0.98 |
| Student | lower | stress | 0.34 | 0.40 | 0.54 | 0.61 | 0.73 | 0.83 | 0.85 | 0.89 |
| Student | lower | stress- | 8.94* | 8.76* | 6.23* | 3.72* | 2.69* | 1.87* | 1.70* | 1.42* |
| Student | upper | normal | 1.00 | 1.06 | 0.96 | 1.04 | 1.00 | 0.99 | 1.02 | 1.08 |
| Student | upper | stress | 8.93* | 8.36* | 5.95* | 3.79* | 2.33* | 1.83* | 1.54* | 1.30* |
| Student | upper | stress- | 0.35 | 0.43 | 0.51 | 0.65 | 0.77 | 0.82 | 0.84 | 0.91 |
| bootstrap-t | lower | normal | 1.02 | 0.99 | 1.06 | 1.09 | 1.06 | 1.16~ | 1.02 | 1.01 |
| bootstrap-t | lower | stress | 0.83 | 0.89 | 0.93 | 0.97 | 1.00 | 1.00 | 0.97 | 1.12 |
| bootstrap-t | lower | stress- | 7.80* | 7.81* | 4.79* | 1.82* | 1.04 | 1.03 | 1.05 | 1.06 |
| bootstrap-t | upper | normal | 1.07 | 1.13~ | 1.02 | 1.02 | 1.01 | 0.99 | 1.02 | 1.02 |
| bootstrap-t | upper | stress | 8.27* | 7.86* | 4.91* | 1.69* | 1.14~ | 1.11 | 1.00 | 1.03 |
| bootstrap-t | upper | stress- | 0.85 | 0.90 | 0.95 | 0.95 | 0.96 | 1.03 | 1.03 | 1.00 |

Contrast bounds (R = 196000, the 13 component tails 8.9e-4 to 8.3e-3; the `percentile` rows are the N6b bound):

| bound | method | distribution | n=20 | n=30 | n=50 | n=100 | n=200 |
|---|---|---|---|---|---|---|---|
| paired | Student | normal | 1.08 | 1.00 | 0.96 | 0.97 | 1.05 |
| paired | Student | stress | 14.11* | 16.27* | 12.16* | 6.72* | 4.33* |
| paired | Student | stress- | 0.05 | 0.08 | 0.16 | 0.30 | 0.44 |
| paired | percentile | normal | 3.88* | 2.67* | 1.75* | 1.30* | 1.20~ |
| paired | percentile | stress | 34.42* | 26.93* | 14.00* | 5.62* | 2.99* |
| paired | percentile | stress- | 0.51 | 0.51 | 0.54 | 0.59 | 0.69 |
| paired | bootstrap-t | normal | 1.20~ | 1.11 | 1.11 | 1.11 | 1.05 |
| paired | bootstrap-t | stress | 13.97* | 14.95* | 10.35* | 3.51* | 1.43* |
| paired | bootstrap-t | stress- | 0.35 | 0.58 | 0.75 | 0.86 | 0.96 |
| unpaired | Student | normal | 1.01 | 1.09 | 1.00 | 1.04 | 1.02 |
| unpaired | Student | stress | 0.65 | 0.78 | 0.89 | 0.97 | 1.03 |
| unpaired | Student | stress- | 0.64 | 0.79 | 0.91 | 0.94 | 1.02 |
| unpaired | percentile | normal | 2.34* | 1.83* | 1.31* | 1.24~ | 1.15~ |
| unpaired | percentile | stress | 3.42* | 2.54* | 1.68* | 1.35* | 1.08 |
| unpaired | percentile | stress- | 3.46* | 2.65* | 1.93* | 1.31* | 1.16~ |
| unpaired | bootstrap-t | normal | 1.09 | 1.03 | 0.99 | 1.02 | 1.01 |
| unpaired | bootstrap-t | stress | 2.42* | 2.73* | 2.40* | 1.88* | 1.53* |
| unpaired | bootstrap-t | stress- | 2.34* | 2.69* | 2.62* | 1.82* | 1.39* |
| interaction | Student | normal | 1.08 | 1.11 | 1.10 | 1.04 | 1.11 |
| interaction | Student | stress | 0.63 | 0.76 | 0.88 | 0.94 | 0.99 |
| interaction | Student | stress- | 0.63 | 0.80 | 0.88 | 0.94 | 1.03 |
| interaction | percentile | normal | 2.17* | 1.72* | 1.47* | 1.21~ | 1.07 |
| interaction | percentile | stress | 3.42* | 2.68* | 1.97* | 1.27* | 1.14~ |
| interaction | percentile | stress- | 3.43* | 2.75* | 2.05* | 1.27* | 1.18~ |
| interaction | bootstrap-t | normal | 1.02 | 1.07 | 1.01 | 1.06 | 0.94 |
| interaction | bootstrap-t | stress | 2.45* | 2.87* | 2.56* | 1.99* | 1.33* |
| interaction | bootstrap-t | stress- | 2.40* | 2.69* | 2.32* | 1.99* | 1.42* |

Gate 5 TOST (R = 25000 for Student, 24500 for bootstrap-t; the 7 Holm levels 7.1e-3 to 0.05, true ratio at the margin; `hi` / `lo`: the two one-sided tests):

| method | rho | side | distribution | n=20 | n=30 | n=50 | n=100 | n=170 |
|---|---|---|---|---|---|---|---|---|
| Student | 0 | hi | normal | 0.00 | 0.01 | 0.13 | 0.92 | 1.01 |
| Student | 0 | hi | stress | 0.50 | 0.91 | 1.39* | 2.70* | 2.22* |
| Student | 0 | hi | stress- | 0.00 | 0.00 | 0.00 | 0.41 | 0.69 |
| Student | 0 | lo | normal | 0.00 | 0.00 | 0.15 | 0.97 | 0.99 |
| Student | 0 | lo | stress | 0.01 | 0.00 | 0.00 | 0.44 | 0.71 |
| Student | 0 | lo | stress- | 0.55 | 0.94 | 1.42* | 2.41* | 2.00* |
| Student | 0.5 | hi | normal | 0.11 | 0.41 | 0.94 | 1.03 | 1.01 |
| Student | 0.5 | hi | stress | 2.47* | 3.84* | 4.06* | 2.77* | 2.30* |
| Student | 0.5 | hi | stress- | 0.01 | 0.02 | 0.26 | 0.66 | 0.72 |
| Student | 0.5 | lo | normal | 0.11 | 0.42 | 0.95 | 0.99 | 1.02 |
| Student | 0.5 | lo | stress | 0.01 | 0.02 | 0.23 | 0.63 | 0.74 |
| Student | 0.5 | lo | stress- | 2.40* | 3.93* | 3.84* | 2.81* | 2.25* |
| Student | 0.9 | hi | normal | 0.97 | 0.97 | 1.01 | 1.08 | 1.02 |
| Student | 0.9 | hi | stress | 6.26* | 5.98* | 4.16* | 2.83* | 2.44* |
| Student | 0.9 | hi | stress- | 0.29 | 0.40 | 0.53 | 0.65 | 0.70 |
| Student | 0.9 | lo | normal | 1.02 | 1.03 | 0.96 | 1.07 | 1.03 |
| Student | 0.9 | lo | stress | 0.30 | 0.42 | 0.53 | 0.63 | 0.67 |
| Student | 0.9 | lo | stress- | 6.20* | 5.53* | 4.59* | 2.59* | 2.06* |
| bootstrap-t | 0 | hi | normal | 0.00 | 0.01 | 0.18 | 0.99 | 1.07 |
| bootstrap-t | 0 | hi | stress | 0.49 | 0.89 | 1.04 | 1.38* | 1.18~ |
| bootstrap-t | 0 | hi | stress- | 0.00 | 0.00 | 0.01 | 0.64 | 0.94 |
| bootstrap-t | 0 | lo | normal | 0.01 | 0.01 | 0.18 | 0.95 | 1.06 |
| bootstrap-t | 0 | lo | stress | 0.00 | 0.00 | 0.01 | 0.61 | 0.95 |
| bootstrap-t | 0 | lo | stress- | 0.50 | 0.90 | 0.96 | 1.31* | 1.15~ |
| bootstrap-t | 0.5 | hi | normal | 0.12 | 0.41 | 0.94 | 1.02 | 1.12 |
| bootstrap-t | 0.5 | hi | stress | 2.35* | 3.99* | 3.43* | 1.55* | 1.10 |
| bootstrap-t | 0.5 | hi | stress- | 0.01 | 0.02 | 0.37 | 0.97 | 1.03 |
| bootstrap-t | 0.5 | lo | normal | 0.11 | 0.41 | 1.02 | 1.12~ | 1.00 |
| bootstrap-t | 0.5 | lo | stress | 0.01 | 0.03 | 0.40 | 0.98 | 0.97 |
| bootstrap-t | 0.5 | lo | stress- | 2.24* | 3.59* | 3.37* | 1.57* | 1.14~ |
| bootstrap-t | 0.9 | hi | normal | 1.07 | 1.05 | 1.04 | 1.01 | 1.02 |
| bootstrap-t | 0.9 | hi | stress | 6.67* | 6.08* | 3.24* | 1.49* | 1.21~ |
| bootstrap-t | 0.9 | hi | stress- | 0.65 | 0.95 | 0.96 | 0.93 | 0.99 |
| bootstrap-t | 0.9 | lo | normal | 1.09 | 1.09 | 1.08 | 1.04 | 1.02 |
| bootstrap-t | 0.9 | lo | stress | 0.67 | 0.90 | 0.89 | 0.96 | 1.04 |
| bootstrap-t | 0.9 | lo | stress- | 7.33* | 5.97* | 3.51* | 1.63* | 1.17~ |

`LadderBound` (360 percentile cells = the 12 N6b cases x 2 production levels x 3 distributions x 5 `n`; 150 Student and 150 bootstrap-t cells =
the 5 lean cases of the `full` profile in the same layout) and the robust-sign pipeline (K = 3, 4;
all budget components with the method named) qualify at every `n`, distribution and tail, for all three methods. Largest UCL / tail
of the ladders: 0.66 (percentile), 0.30 (Student), 0.57 (bootstrap-t); largest rate / tail 0.56, 0.24, 0.47: the bounds are conservative
by a factor of at least 1.5. The pipeline had **0 misses out of 24500 replicates at every Holm level, in every cell** (largest UCL / tail
0.017 = the Clopper-Pearson limit of zero misses at `R` = 24500 divided by the smallest Holm level 7.14e-3): it is conservative by construction (union
bound, biases at their sign), which bears on the power of SF-43's decision, not on its validity.

**Per `n`: the (procedure, method) combinations that qualify at every tail and distribution.**

n=20: ContrastBound unpaired (Student); ContrastBound interaction (Student); LadderBound (Student); LadderBound (percentile); LadderBound (bootstrap-t)
n=30: ContrastBound unpaired (Student); ContrastBound interaction (Student); LadderBound (Student); LadderBound (percentile); LadderBound (bootstrap-t); robust-sign pipeline K = 4 (Student); robust-sign pipeline K = 4 (percentile); robust-sign pipeline K = 4 (bootstrap-t)
n=50: ContrastBound unpaired (Student); ContrastBound interaction (Student); LadderBound (Student); LadderBound (percentile); LadderBound (bootstrap-t)
n=100: ContrastBound unpaired (Student); ContrastBound interaction (Student); LadderBound (Student); LadderBound (percentile); LadderBound (bootstrap-t); robust-sign pipeline K = 3 (Student); robust-sign pipeline K = 3 (percentile); robust-sign pipeline K = 3 (bootstrap-t); robust-sign pipeline K = 4 (Student); robust-sign pipeline K = 4 (percentile); robust-sign pipeline K = 4 (bootstrap-t)
n=170: none
n=200: ContrastBound unpaired (Student); ContrastBound interaction (Student); LadderBound (Student); LadderBound (percentile); LadderBound (bootstrap-t)
n=300: robust-sign pipeline K = 3 (Student); robust-sign pipeline K = 3 (percentile); robust-sign pipeline K = 3 (bootstrap-t); robust-sign pipeline K = 4 (Student); robust-sign pipeline K = 4 (percentile); robust-sign pipeline K = 4 (bootstrap-t)
n=500: none
n=1000: sampling of the mean (bootstrap-t); robust-sign pipeline K = 3 (Student); robust-sign pipeline K = 3 (percentile); robust-sign pipeline K = 3 (bootstrap-t)
n=1650: sampling of the mean (bootstrap-t)

(`n` = 170, 500: nothing qualifies in full; at 500 the bootstrap-t limit of the mean qualifies for its upper limit and is marginal for the
lower; at 170 the bootstrap-t TOST has one boundary of one `rho` qualified, see below.) The Student limit of the mean never qualifies: it is
exact on normal data but fails on the skew side of `stress` / `stress-` at every `n` up to 1650.

**Marginal failures.** The nine marginal groups (each failing row individually compatible with a nominal procedure at the false-fail
probability above, about 0.08 per (cell, distribution, row) and 0.22 to 0.23 per three-distribution group) are: `ContrastBound` interaction
percentile, `n` = 200, `stress` and `stress-` (5 tails, 1.18); `ContrastBound` unpaired percentile, `n` = 200, `normal` and `stress-` (3
tails, 1.16); the bootstrap-t limit of the mean, lower, `n` = 500, `normal` (2 tails, 1.16); the same, upper, `n` = 200, `stress` (2 tails,
1.14); and the five bootstrap-t TOST groups at `n` = 170 (`rho` 0 hi 1.18 and lo 1.15, `rho` 0.5 lo 1.14, `rho` 0.9 hi 1.21 and lo 1.17;
2-3 tails each, the skew side only). Where several tails of one row fail on the same, skew, side (the TOST at 170), the pattern points to a
real residual excess of at most about 20 %, not to pure noise; isolated normal-data failures are noise. On normal data, 5 rows fail among
the 122 rows of Student and bootstrap-t sampling, Student and bootstrap-t TOST and the Student and bootstrap-t contrast bounds (Student
sampling lower `n` = 200: 1.16; bootstrap-t sampling upper `n` = 30: 1.13 and lower `n` = 500: 1.16; bootstrap-t TOST `rho` 0.5 lo `n` = 100: 1.12; bootstrap-t
paired contrast `n` = 20: 1.20; all marginal), against about 9.8 expected if all 122 rows were exactly nominal (122 rows at 0.08): no
clear failure of an exact or second-order procedure on normal data. The 101 clear groups are the cases below.

**Where no method qualifies.** (i) The paired contrast bound under `stress` at every `n` from 20 to 200: percentile 34.4 / 26.9 / 14.0 / 5.6 /
3.0 (largest rate / tail, `n` = 20, 30, 50, 100, 200), Student 14.1 / 16.3 / 12.2 / 6.7 / 4.3 (UCL / tail 12.60, 7.05, 4.60 at `n` = 50, 100, 200), bootstrap-t
14.0 / 15.0 / 10.4 / 3.5 / 1.43 (UCL / tail 10.76, 3.76, 1.56 at 50, 100, 200; at `n` = 200 10 of 13 tails still fail). It is the skewness
of the paired difference that none of the three removes. This confirms, and extends from `n` <= 100 to `n` = 200, what the small-scale N6c
runs predicted (UCL / tail 12.6, 7.1, 4.6 for Student and 10.8, 3.8, 1.48 for bootstrap-t at `n` = 50, 100, 200, as quoted in the orchestrator's audit). (ii)
Gate 5 TOST under `stress`: the Student version fails at every `n` from 20 to 170 and every `rho` except `rho` = 0 at `n` = 20 and 30 (rate /
tail 0.50 to 0.94, where the TOST is conservative: `rho` = 0 on normal data has a rate of about 0 at `n` = 20 and 30, because the standard error is large relative to the
margin; this "pass" is not evidence of robustness); at `n` = 170 its rate / tail is 2.0 to 2.4 under `stress`. The bootstrap-t TOST (UCL / tail
1.35 at `n` = 170, `rho` 0.9, as predicted) is within 1.25 x tail for `rho` = 0 at `n` = 20 to 50 (again the conservative regime), fails clearly at `n` <= 100
for `rho` 0.5 and 0.9 (up to 7.3) and at `n` = 100 for `rho` 0 (1.38), and at `n` = 170 qualifies only `rho` 0.5 hi; `rho` 0.5 lo, and both boundaries at `rho` 0
and 0.9 are marginal (1.14 to 1.21). Therefore **no TOST version qualifies both boundaries at any `n` up to 170 under `stress`**; the closest is the
bootstrap-t TOST at `n` = 170, `rho` 0.5 (one boundary qualified, the other marginal at 1.14). (iii) The limit of the mean (Student) at every `n` up to 1650 on the skew
side (8.9 at `n` = 20, 2.3 to 2.7 at 200, 1.3 to 1.4 at 1650; the opposite side over-covers); the bootstrap-t qualifies from `n` = 200 (lower) and 500 (upper), and at 1000 and 1650.
(iv) The percentile unpaired and interaction contrast bounds at every `n` <= 200 (they already fail on `normal` data up to `n` = 100: 2.3 at `n` = 20 for unpaired,
1.2 at 100; marginal at 200). The Student version of both qualifies at every `n` (conservative under `stress`: 0.63 to 0.65 at `n` = 20, rising to 0.99 to 1.03 at 200, exact on `normal`).

**A pattern the N6c small-scale runs did not predict.** The bootstrap-t bound of the **unpaired and interaction contrasts fails under both
stress signs** (rate / tail 2.3 to 2.5 at `n` = 20, 1.8 to 2.0 at 100, 1.3 to 1.5 at 200; clear at every `n`), while it is exact on normal data (0.94 to 1.09) and the Student bound of
the same contrasts qualifies at every `n`. An independent re-implementation that imports nothing from `mcstat`
(`R/verification/indep_boott_unpaired.py`, output `R/verification/indep_boott_unpaired.txt`, `python3 -I ... 7`; unpaired contrast, `stress`
mixture, `n` = 20, `R` = 20000, `B` = 2000, `Delta` = 5 standard errors) gives bootstrap-t rate / tail 1.87 / 1.86 / 1.88 at the tails 0.00833 / 0.00417
/ 0.0025 and Student 0.64 / 0.54 / 0.50; the study's cell (`B` = 12000, `R` = 196000) has bootstrap-t 2.02 / 2.24 and Student 0.65 / 0.54 at the first two of these tails (the third is not a tail of the plan). It
is therefore a property of the bootstrap-t under this mixture, not an implementation defect. Mechanism, as a **hypothesis** (not tested here): the skews of the two arms cancel in the
difference, so the Student bound is made conservative only by the kurtosis, while the bootstrap-t estimates the distribution of `T*` conditionally on the sample configuration
(an arm with or without minority-component points) and that conditional skew is correlated with the miss event, so its lower `T*` quantile is too shallow exactly when the
observed difference is low.

**Cost.** 10 782 196 core-seconds = 2995.1 host core-hours in 150 047 s of wall time on 72 workers (99.8 % busy), the slot held for 41.693 h
(`R/verification/jobs_r5.tsv`). The projection in the `coverage.py` docstring was 261 workstation core-hours and 5.4 to 7.3 h on 80 cores; the measured cost is 11.5 times the
core-hours and 5.7 to 7.7 times the projected wall time. The ladders are 2654 h (89 %). The 72 workers shared 40 physical cores (hyper-threading, 2 threads per core), the host
core is slower than the workstation's and numpy is older (1.26 against 2.5); 10 sampled chunks (17 s or more) took 4.4 to 9.1 times longer on the host than on the workstation, below
the overall factor of 11.5, which is not decomposed further. The cost table of the `coverage.py` docstring was corrected with these figures (workstation figures kept and labelled).

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

### Design pilots (R4, R4b)

Scope: the pilots are design inputs for SF-43. They are not evidence about macrodispersion: no sign or size of any `alpha` is read or
interpreted below (the `alpha_LP` means in the pairing tables are listed only because the coefficient of variation and the correlation
are computed from them). Committed outputs: `R/pilots/` (2.5 MB: per-run `RUN.json`, `cmd.txt`, trimmed `run.log.trim` and the Par2
`config.yaml`; the reduced per-realization tables with their sha256 input manifests; `hostref.json`; `j1_grid.*`, `j2_*`,
`j3_pairing.json`, `j3b_pairing.json`, `cost_projection*`, `presubmit_projection.json`; `J1/per_particle_sha256.txt`). No per-particle
array is committed (the arrays stay in `~/sf42_campaign/pilots/` on the host). Every number below is regenerated from the committed
inputs by `R/scripts/pilots` (`python3 -m pilots.report j1|j2|j3|j3b|cost --data R/pilots --out <dir> --repo <repo root>`, tables by
`python3 -m pilots.tables R/pilots <table>`); R4b regenerated all eight outputs into a scratch directory and they were
byte-identical to the files R4 had produced (before the R4b additions, which only add keys/columns). The only inputs that exist on the host
alone are the SF-39 PCG-iteration census and the planning SDs (`hostref.json`, made on the host by `pilots.hostref` from
`~/sf39_campaign` and `~/sf42_campaign/r3`; committed as read) and the per-particle arrays.

Jobs, one at a time (details under Commands): `sf42-r4-build` (CPU), `sf42-pilot-grid` (J1), `sf42-pilot-particles` (J2),
`sf42-pilot-pairing` (J3), `sf42-pilot-horizon` (J3B; see Deviation 6). Projected before submission with the SF-39-only cost model
(`presubmit_projection.json`): J1 0.105, J2 1.208, J3 0.376 GPU-h, sum 1.69 GPU-h, far below the 8 GPU-h that would have forced a cut
of the realization counts; measured 0.154, 1.491, 0.432 (J1 +46 %, J2 +23 %, J3 +15 %); J3B projected 0.135, measured 0.118. Total of R4: 2.287 GPU-h.

#### Grid feasibility (J1)

One realization per grid at the highest `sigma^2` that SF-43 plans on that grid (mapping of the R4 brief, identical to the committed
run table: 520x120x120 at 1, 832x96x96 and 832x192x192 at 4, 1040x240x240, 1040x480x480, 2080x480x480 and 1248x288x288 at 2.25,
2496x288x288 and 1664x192x192 at 4), consumer `--medium b`, flux weights, `h/8` single pass, planes `d` in [25, 94] `lambda`, the
amplitude's source (19.2 `lambda`; 9.6 at `sigma^2` 4), `M` the admissible lattice size nearest to SF-39's seeds per `lambda`
(`seedlattice.rule`); Par2 (`cg_rtol` 1e-10, `n_particles` 1e4, `sf18_medium_b` field with the consumer's seed, window injection, plane
crossings) on the three Gate 5 grids. All 12 runs exit 0 with a converged solve. Reference for the 3x rule: the median PCG iterations of
SF-39's production medium (b) solves at the same `sigma^2` (`hostref.json`, from the host `solver_b.json` files: A and B arms; 25, 50, 62
and 71 solves at `sigma^2` 0.25, 1, 2.25, 4, medians 20, 30, 40, 60).

| run | code | grid | sigma^2 | M | PCG iters | verdict | SF-39 median | ratio | flow stage (s) | wall (s) | peak GPU MiB (own PID) | runner exact MiB | rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c_s1_520x120` | consumer | 520x120x120 | 1 | 96 | 70 | converged | 30 | 2.33 | 4.7 | 13.4 | 1757 | 432 | ok |
| `c_s225_1040x240` | consumer | 1040x240x240 | 2.25 | 128 | 90 | converged | 40 | 2.25 | 17.2 | 35.8 | 3971 | 3460 | ok |
| `c_s225_1040x480` | consumer | 1040x480x480 | 2.25 | 128 | 90 | converged | 40 | 2.25 | 54.9 | 88.7 | 14365 | 13839 | ok |
| `c_s225_1248x288` | consumer | 1248x288x288 | 2.25 | 100 | 60 | converged | 40 | 1.50 | 22.0 | 39.0 | 6491 | 5979 | ok |
| `c_s225_2080x480` | consumer | 2080x480x480 | 2.25 | 128 | 90 | converged | 40 | 2.25 | 104.8 | 152.4 | 28203 | 27683 | ok |
| `c_s4_1664x192` | consumer | 1664x192x192 | 4 | 100 | 60 | converged | 60 | 1.00 | 12.3 | 30.9 | 4051 | 3543 | ok |
| `c_s4_2496x288` | consumer | 2496x288x288 | 4 | 100 | 120 | converged | 60 | 2.00 | 55.1 | 91.0 | 12471 | 11959 | ok |
| `c_s4_832x192` | consumer | 832x192x192 | 4 | 100 | 40 | converged | 60 | 0.67 | 7.1 | 22.4 | 2275 | 1772 | ok |
| `c_s4_832x96` | consumer | 832x96x96 | 4 | 100 | 50 | converged | 60 | 0.83 | 4.4 | 13.3 | 1701 | 443 | ok |
| `p_s1_1040x240` | Par2 | 1040x240x240 | 1 | - | 50 | converged | 30 | 1.67 | 10.8 | 18.2 | 7699 | - | ok |
| `p_s225_1248x288` | Par2 | 1248x288x288 | 2.25 | - | 40 | converged | 40 | 1.00 | 15.6 | 27.7 | 12881 | - | ok |
| `p_s4_1664x192` | Par2 | 1664x192x192 | 4 | - | 60 | converged | 60 | 1.00 | 12.2 | 20.7 | 7865 | - | ok |

Rule (spec G, pre-stated): more than 3x the SF-39 median, or no convergence, means "requires a different grid in SF-43". The largest
ratio is 2.33 (520x120x120); no grid trips the rule. PCG iterations are quantized by the production `check_every = 10`
(`src/numerics/solvers/pcg.cuh`: convergence is tested only every 10 iterations), so every count is a multiple of 10 and a ratio has a
resolution of one check interval. "Flow stage" is the runner's stage time (field, head solve, velocity, splines); the runner's exact
bytes are its own closed-form device accounting (`MEM_PHASE`, maximum over phases), the peak is the maximum of the job's own PID in
`nvidia-smi --query-compute-apps` sampled every second (never the GPU total). The peak own-PID memory exceeds the runner's exact bytes by 0.49 to 0.51 GiB on every grid from 832x192x192 up (the CUDA
context) and by 1.2 to 1.3 GiB on the two smallest. SF-43 use of 2496x288x288 (`lambda/h` 24, width 12): none is identifiable in the SF-43 specification. Its
`E_h` ladder at `sigma^2` 4 is 8 / 16 / 32, and 2496x288x288 was the Par2 grid of SF-39 at 2.25 and 4; it is run here at 4, the only
amplitude of the listed pair it could serve, and it would be an intermediate rung (`lambda/h` 24) that the specification does not
plan.

`sigma^2` 0.25 is not a J1 run (no grid is planned only there): J2 and J3 run 1040x240x240 at 0.25 (outside the J1 protocol, reported
for completeness; see Findings (f)). J2's recorded PCG counts are 60 in all 18 realizations (three `M`, six each), a ratio of 3.00 to
SF-39's median of 20, not more than 3; J3's are 60 (x7) and 70 (x1), mean 61.25, ratio 3.06 (the 70 alone is 3.5). With convergence tested
every 10 iterations a recorded 20 means 11 to 20 actual iterations and a recorded 60 means 51 to 60, so the actual ratio lies between
2.55 and 5.45 and the pre-stated 3x rule cannot resolve this case. It is not resolved here: it is flagged as an open decision on the
0.25 production grid for SF-43's pre-registration and the owner.

#### Particle count (J2)

SF-39's seeds per `lambda`: M = 100 seeds per side over a source of 0.8 of the production width, 0.8 x 25.6 = 20.48 `lambda` at
`sigma^2` 0.25 and 1, 0.8 x 24 = 19.2 at 2.25, 0.8 x 12 = 9.6 at 4: 100/20.48 = 4.883, 100/19.2 = 5.208, 100/9.6 = 10.42 seeds per
`lambda`. SF-43's source is 19.2 / 19.2 / 19.2 / 9.6 `lambda`, so the same density is `M` = 4.883 x 19.2 = 93.75 at 0.25 and 1, 100 at
2.25 and 4. Targets 0.6x / 1x / 2x: 56.25 / 93.75 / 187.5 at 0.25 and 1; 60 / 100 / 200 at 2.25 and 4. Admissible lattice sizes
(`seedlattice.rule`) on the SF-43 production grid and window, nearest to the target without repeats: **64, 128, 192** at 0.25 and 1
(1040x240x240; the admissible values there are the multiples of 64: 0.6x gives 64, 1x is also nearest to 64, which is taken, so 128;
that is 0.68x, 1.37x, 2.05x of SF-39's density) and **58, 100, 202** at 2.25 (1664x384x384) and 4 (3328x384x384). Six realizations per
`M`, the same six field seeds for the three `M` of an amplitude (common random numbers; seeds 420701, 420711, 420721, 420731 + 0..5), 12 runs.
Reduced with `sf42red reduce` (consumer, flux weights, set `perplane`, window [25, 94]); the particle component is
`mean_r[(a_A - a_B)^2]/4` of the two half-lattices (SF-39's split-half definition), its share is taken against the SF-39 planning
across-realization variance of the estimator on [25, 94] (R3 table, `hostref.json` `planning_sd`; `pos` and `disp` SD, the `disp` one is the
stricter for the transverse estimators), and projected to other `M` by the fixed `M^-2` planning law anchored on the three measured
components (`k = exp(mean(log c_i + 2 log M_i))`); the fitted exponent is reported beside it. `alpha_LP` is a position-basis
estimator only. `alpha_TAP` equals `alpha_TP` within 8e-4 relative in component and variance (maximum over all cells of `j2_components.csv`; R3: 4e-4 in mean and SD) and is left out of the table (full table in
`j2_components.csv`). Complete rows (all estimators, bases, `M`; own split-half share of the 6 realizations; planning share):

| sigma^2 | estimator (basis) | M | n | component `mean_r[(a_A-a_B)^2]/4` | own split-half share (6 real.) | planning SD (R3) | share of planning variance |
|---|---|---|---|---|---|---|---|
| 0.25 | alpha_TP (pos) | 64 | 6 | 7.307e-12 | 3.785e-06 | 4.476e-03 | 3.647e-07 |
| 0.25 | alpha_TP (pos) | 128 | 6 | 7.963e-13 | 4.125e-07 | 4.476e-03 | 3.974e-08 |
| 0.25 | alpha_TP (pos) | 192 | 6 | 1.441e-13 | 7.463e-08 | 4.476e-03 | 7.191e-09 |
| 0.25 | alpha_TP (disp) | 64 | 6 | 1.331e-12 | 1.064e-04 | 1.206e-04 | 9.158e-05 |
| 0.25 | alpha_TP (disp) | 128 | 6 | 1.024e-15 | 8.198e-08 | 1.206e-04 | 7.047e-08 |
| 0.25 | alpha_TP (disp) | 192 | 6 | 2.734e-17 | 2.188e-09 | 1.206e-04 | 1.881e-09 |
| 0.25 | alpha_LP (pos) | 64 | 6 | 8.732e-07 | 3.284e-03 | 2.888e-02 | 1.047e-03 |
| 0.25 | alpha_LP (pos) | 128 | 6 | 2.620e-09 | 9.780e-06 | 2.888e-02 | 3.141e-06 |
| 0.25 | alpha_LP (pos) | 192 | 6 | 7.223e-11 | 2.693e-07 | 2.888e-02 | 8.658e-08 |
| 1 | alpha_TP (pos) | 64 | 6 | 1.395e-07 | 2.962e-03 | 1.103e-02 | 1.147e-03 |
| 1 | alpha_TP (pos) | 128 | 6 | 1.309e-08 | 2.693e-04 | 1.103e-02 | 1.076e-04 |
| 1 | alpha_TP (pos) | 192 | 6 | 3.079e-09 | 6.360e-05 | 1.103e-02 | 2.530e-05 |
| 1 | alpha_TP (disp) | 64 | 6 | 1.480e-08 | 7.423e-03 | 1.121e-03 | 1.179e-02 |
| 1 | alpha_TP (disp) | 128 | 6 | 2.244e-10 | 1.124e-04 | 1.121e-03 | 1.787e-04 |
| 1 | alpha_TP (disp) | 192 | 6 | 8.814e-11 | 4.391e-05 | 1.121e-03 | 7.019e-05 |
| 1 | alpha_LP (pos) | 64 | 6 | 8.607e-04 | 4.994e-02 | 1.018e-01 | 8.307e-02 |
| 1 | alpha_LP (pos) | 128 | 6 | 1.274e-04 | 6.387e-03 | 1.018e-01 | 1.229e-02 |
| 1 | alpha_LP (pos) | 192 | 6 | 4.286e-05 | 2.085e-03 | 1.018e-01 | 4.136e-03 |
| 2.25 | alpha_TP (pos) | 58 | 6 | 1.327e-05 | 1.758e-01 | 1.514e-02 | 5.789e-02 |
| 2.25 | alpha_TP (pos) | 100 | 6 | 1.563e-06 | 1.557e-02 | 1.514e-02 | 6.819e-03 |
| 2.25 | alpha_TP (pos) | 202 | 6 | 1.997e-07 | 1.814e-03 | 1.514e-02 | 8.715e-04 |
| 2.25 | alpha_TP (disp) | 58 | 6 | 8.570e-07 | 8.464e-02 | 4.073e-03 | 5.166e-02 |
| 2.25 | alpha_TP (disp) | 100 | 6 | 1.796e-07 | 1.647e-02 | 4.073e-03 | 1.083e-02 |
| 2.25 | alpha_TP (disp) | 202 | 6 | 5.115e-08 | 4.460e-03 | 4.073e-03 | 3.084e-03 |
| 2.25 | alpha_LP (pos) | 58 | 6 | 2.748e-02 | 8.114e-01 | 2.291e-01 | 5.234e-01 |
| 2.25 | alpha_LP (pos) | 100 | 6 | 6.487e-03 | 1.331e-01 | 2.291e-01 | 1.236e-01 |
| 2.25 | alpha_LP (pos) | 202 | 6 | 3.609e-03 | 1.369e-01 | 2.291e-01 | 6.875e-02 |
| 4 | alpha_TP (pos) | 58 | 6 | 7.765e-06 | 4.478e-02 | 1.918e-02 | 2.111e-02 |
| 4 | alpha_TP (pos) | 100 | 6 | 5.334e-06 | 2.734e-02 | 1.918e-02 | 1.450e-02 |
| 4 | alpha_TP (pos) | 202 | 6 | 8.146e-08 | 3.863e-04 | 1.918e-02 | 2.215e-04 |
| 4 | alpha_TP (disp) | 58 | 6 | 4.440e-06 | 5.620e-02 | 1.688e-02 | 1.559e-02 |
| 4 | alpha_TP (disp) | 100 | 6 | 1.204e-06 | 1.145e-02 | 1.688e-02 | 4.226e-03 |
| 4 | alpha_TP (disp) | 202 | 6 | 1.300e-07 | 1.357e-03 | 1.688e-02 | 4.564e-04 |
| 4 | alpha_LP (pos) | 58 | 6 | 9.664e-01 | 2.554e-01 | 1.478e+00 | 4.421e-01 |
| 4 | alpha_LP (pos) | 100 | 6 | 1.001e-01 | 1.059e-01 | 1.478e+00 | 4.581e-02 |
| 4 | alpha_LP (pos) | 202 | 6 | 8.900e-02 | 4.513e-02 | 1.478e+00 | 4.071e-02 |

The component and share per estimator and basis, fitted exponent in `M` (standard error from three points), and the smallest admissible
`M` whose projected share is at most 0.25 and at most 0.10 (fixed `M^-2` law / fitted law):

| sigma^2 | grid | M tested | estimator (basis) | fitted exponent of M (se) | M at share <= 0.25 (fixed M^-2 law / fit) | M at share <= 0.10 (fixed / fit) |
|---|---|---|---|---|---|---|
| 0.25 | 1040x240x240 | 64,128,192 | alpha_TP (pos) | -3.53 (0.27) | 64 / 64 | 64 / 64 |
| 0.25 | 1040x240x240 | 64,128,192 | alpha_TP (disp) | -9.88 (0.37) | 64 / 64 | 64 / 64 |
| 0.25 | 1040x240x240 | 64,128,192 | alpha_LP (pos) | -8.54 (0.13) | 64 / 64 | 64 / 64 |
| 1 | 1040x240x240 | 64,128,192 | alpha_TP (pos) | -3.47 (0.04) | 64 / 64 | 64 / 64 |
| 1 | 1040x240x240 | 64,128,192 | alpha_TP (disp) | -4.81 (0.98) | 64 / 64 | 64 / 64 |
| 1 | 1040x240x240 | 64,128,192 | alpha_LP (pos) | -2.73 (0.02) | 64 / 64 | 64 / 64 |
| 2.25 | 1664x384x384 | 58,100,202 | alpha_TP (pos) | -3.34 (0.28) | 20 / 38 | 28 / 50 |
| 2.25 | 1664x384x384 | 58,100,202 | alpha_TP (disp) | -2.24 (0.31) | 26 / 28 | 38 / 44 |
| 2.25 | 1664x384x384 | 58,100,202 | alpha_LP (pos) | -1.59 (0.51) | 86 / 82 | 140 / 146 |
| 4 | 3328x384x384 | 58,100,202 | alpha_TP (pos) | -3.76 (1.49) | 14 / 38 | 22 / 46 |
| 4 | 3328x384x384 | 58,100,202 | alpha_TP (disp) | -2.85 (0.22) | 14 / 26 | 20 / 34 |
| 4 | 3328x384x384 | 58,100,202 | alpha_LP (pos) | -1.83 (1.13) | 68 / 62 | 106 / 106 |

| sigma^2 | M_plan (0.25, pos denominator; max over the three estimators) | same, disp denominator | M at 0.10 (pos) | M at 0.10 (disp) |
|---|---|---|---|---|
| 0.25 | 64 | 64 | 64 | 64 |
| 1 | 64 | 64 | 64 | 64 |
| 2.25 | 86 | 86 | 140 | 140 |
| 4 | 68 | 68 | 106 | 106 |

`M_plan` (column 2 of the second table) is the largest of the three estimators' conclusions at share 0.25 with the `pos` denominator;
the `disp` denominator gives the same values (the `alpha_LP` basis is `pos` only and dominates at 2.25 and 4). At `sigma^2` 0.25 and 1 the
conclusion 64 is the smallest admissible lattice size, not a share limit: every estimator's planning share at `M` = 64 is at most 8.3e-2
(`alpha_LP` at `sigma^2` 1; at most 1.0e-3 at 0.25), so the binding constraint there is the lattice rule. At 2.25 and 4 the share of
`alpha_LP` binds: 86 and 68 at 0.25, 140 and 106 at 0.10, against 20 to 38 (2.25) and 14 to 38 (4) for the transverse estimators
(`pos`, `disp`). Flagged by SF-39's rule (own split-half share above 0.25): `alpha_LP` at `sigma^2` 2.25 with `M` = 58 (0.81) and at 4
with `M` = 58 (0.26); the planning-share column shows the same two cells as the largest (0.52, 0.44). The fitted exponents are
-3.5 / -3.5 / -3.3 / -3.8 (`alpha_TP`, `pos`) and -1.6 / -1.8 (`alpha_LP`, 2.25 / 4); three points per fit and six realizations, standard
errors 0.02 to 1.5. The fixed `M^-2` law is the planning rule (it is not contradicted for `alpha_LP`: -1.59 +- 0.51 and
-1.83 +- 1.13; `alpha_LP` at 0.25 and 1: -8.5 and -2.7); the transverse components fall faster than `M^-2` over this range. At
0.25 and 1 the planning shares are between 1.9e-9 and 1.0e-3 (0.25) and 2.5e-5 and 8.3e-2 (1), where an exponent fitted to three
components has little meaning.

Crossing completeness of the consumer (fraction of seeded particles that reach the last plane and the first plane, minimum over the six
realizations): 1.0000 for all 12 runs at both planes (`pilots.tables j2cross`; `n_last = n_traced` in every reduced row).

#### Gate 5 pairing (J3 and J3B)

Matched configuration (SF-43 item 7): consumer with uniform weights (the twin-style series) and Par2 on the same `sf18_medium_b` field
per seed, window injection, plane crossings at the consumer's planes, `cg_rtol` 1e-10, eight seeds (420801 + 0..7 at 0.25, 420811 at 1,
420821 at 2.25, 420831 at 4), Par2 `transport.seed` 420800001 (the pipeline adds 1000 r). Grids: 1040x240x240 at 0.25 and 1,
1248x288x288 at 2.25, 1664x192x192 at 4; consumer `M` = 128, 128, 100, 100. Par2 time step: SF-39's rule scaled to the grid, `dt`
= 0.0244140625 dx (SF-39: `U_nom dt` = 0.0244140625 dx) and `n_steps = ceil(hz (L1 - 10 ell) / dt)`. **SF-39's Par2 horizon rule**
(SF-39 artifact `configs/par2_s{1,225,4}.yaml`, header comment, line 19): "`n_steps = ceil(1.2 (Lx - 10 lambda) / (J dt))` with `dt = 12.5`", i.e. 1.2 times
the nominal travel time `U_nom = K_g J` to `Lx - 10 lambda`; J3 uses `hz` = 1.2, J3B `hz` = 3.6 (Deviation 6). Statistics per amplitude:
mean and coefficient of variation of the paired `alpha_LP` per code, the Pearson correlation of the pairs (matched by seed), its 80 %
lower bound (`mcstat.sizing.corr_lower`), and the Gate 5 planning `n` (`gate5_conservative_inputs` + `gate5_sample_size`, Holm worst level
0.05/7, true ratio 1, power 0.8) at the lower-bound correlation and at 0 (the unpaired worst case):

**J3 (SF-39 horizon factor 1.2)**

| sigma^2 | grid | n pairs | mean L | mean P | CV L | CV P | corr | corr 80 % LB | Par2 steps | min Par2 crossed/injected at the last plane | min consumer reached/seeded | plan n (LB corr) | plan n (corr 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1040x240x240 | 8 | 0.2439 | 0.2436 | 0.105 | 0.095 | 0.979 | 0.955 | 46203 | 0.9977 | 1.0000 | 7 | 64 |
| 1 | 1040x240x240 | 8 | 1.065 | 0.9765 | 0.074 | 0.070 | 0.779 | 0.583 | 46203 | 0.9829 | 1.0000 | 16 | 34 |
| 2.25 | 1248x288x288 | 8 | 3.199 | 2.25 | 0.106 | 0.158 | 0.731 | 0.504 | 55444 | 0.9733 | 1.0000 | 63 | 115 |
| 4 | 1664x192x192 | 8 | 10.23 | 4.854 | 0.301 | 0.155 | 0.253 | -0.117 | 73925 | 0.9623 | 1.0000 | 426 | 390 |

**J3B (horizon factor 3.6)**

| sigma^2 | grid | n pairs | mean L | mean P | CV L | CV P | corr | corr 80 % LB | Par2 steps | min Par2 crossed/injected at the last plane | min consumer reached/seeded | plan n (LB corr) | plan n (corr 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1040x240x240 | 8 | 1.065 | 1.058 | 0.074 | 0.070 | 0.939 | 0.875 | 138609 | 1.0000 | 1.0000 | 7 | 34 |
| 2.25 | 1248x288x288 | 8 | 3.199 | 3.151 | 0.106 | 0.122 | 0.860 | 0.725 | 166331 | 0.9999 | 1.0000 | 26 | 83 |
| 4 | 1664x192x192 | 8 | 10.23 | 9 | 0.301 | 0.145 | 0.799 | 0.617 | 221774 | 0.9996 | 1.0000 | 205 | 380 |

Scope of the horizon finding. With `hz` = 1.2 a few Par2 particles have not reached the last control plane when the run ends (minimum
over the eight realizations of crossed/injected at the last plane: 0.9977, 0.9829, 0.9733, 0.9623 at `sigma^2` 0.25, 1, 2.25, 4); with
3.6 it is 1.0000, 0.9999, 0.9996 (1, 2.25, 4). The consumer reaches the last plane with every seeded particle in all 12 J2 runs and all
J3 runs (1.0000). This concerns **plane-crossing estimators** (`alpha_LP`, `alpha_TP` over planes) with a finite Par2 run: a plane-based
average over the particles that crossed is censored at the horizon (late particles are missing). SF-39's own Gate 5 used snapshot
time-window estimators: the window `W_c` ends before the first particle exits (`t_b,min` = the last snapshot with no particle at
`x >= Lx - 5 lambda`, pre-registration section 4, item C), the variance is that of all active particles at each snapshot, and Par2
realizations whose last clean snapshot precedes the end of `W_c` are counted and reported. By this definition it does not condition
on crossing and is not exposed to the plane-based censoring; I did not re-analyse SF-39 (and found no count of such realizations in the SF-39 note).

#### Cost model and SF-43 projection (J4)

Calibration points: every pilot run, with the code's own per-realization time (consumer `timings.json` `seconds.total`; Par2 field +
solve + velocity + transport from the stage lines), not the process wall: SF-39's job walls spread one start-up over at least 20
realizations, the pilot runs of 1 to 8 realizations would carry it per realization (wall-basis residuals are in `cost_projection.json`,
`new_points_under_sf39_model`, basis `wall`: for instance 13.4 s wall against 9.8 s per realization at 520x120x120). Consumer points keep
code `lagrangian` with their real `M^2` and one pass; Par2 at `cg_rtol` 1e-10 has its own code `par2_rtol1e-10` (J1 and J3, horizon
1.2) and, from J3B, `par2_rtol1e-10_h3.6` (horizon 3.6), each fitted from its own points only. Fit before and after:

| code | fit | points | rms rel | max abs rel | cells range | sigma^2 range | particles range | coefficients |
|---|---|---|---|---|---|---|---|---|
| lagrangian | SF-39 only | 29 | 0.119 | 0.447 | 4.19e+06 - 4.91e+08 | 0.25 - 4 | 10000 - 10000 | const=1.8, cells=1.53e-07, cells_sigma2=1.24e-08, trace=5.47e-07 |
| par2 | SF-39 only | 8 | 0.086 | 0.127 | 1.34e+08 - 2.07e+08 | 0.25 - 4 | 10000 - 10000 | cells=2.71e-07 |
| par2_fs | SF-39 only | 8 | 0.066 | 0.179 | 1.34e+08 - 2.07e+08 | 0.25 - 4 | 10000 - 10000 | cells=1.53e-07 |
| lagrangian | with pilots | 54 | 0.153 | 0.405 | 4.19e+06 - 4.91e+08 | 0.25 - 4 | 3364 - 40804 | const=3.68, cells=1.92e-07, cells_sigma2=8.87e-09, trace=4.47e-07 |
| par2 | with pilots | 8 | 0.086 | 0.127 | 1.34e+08 - 2.07e+08 | 0.25 - 4 | 10000 - 10000 | cells=2.71e-07 |
| par2_fs | with pilots | 8 | 0.066 | 0.179 | 1.34e+08 - 2.07e+08 | 0.25 - 4 | 10000 - 10000 | cells=1.53e-07 |
| par2_rtol1e-10 | with pilots | 7 | 0.138 | 0.208 | 5.99e+07 - 1.04e+08 | 0.25 - 4 | 10000 - 10000 | cells=2.07e-07 |
| par2_rtol1e-10_h3.6 | with pilots | 3 | 0.174 | 0.231 | 5.99e+07 - 1.04e+08 | 1 - 4 | 10000 - 10000 | cells=2.2e-07 |

The refit changes `lagrangian` only in the sense expected from added points (rms 0.119 to 0.153, largest 0.447 to 0.405, constant
1.8 s to 3.7 s). Residuals of the new points under the SF-39-only model (`(model - observed)/observed`; the Par2 pilots against SF-39's
own `par2` fit, which used `cg_rtol` 1e-6) and after the refit:

| point | code | observed s/real | SF-39-only model | relative error of the model | refit residual | in SF-39 range |
|---|---|---|---|---|---|---|
| `J1/c_s1_520x120` | lagrangian | 9.8 | 5.7 | -0.423 | -0.254 | no |
| `J1/c_s225_1040x240` | lagrangian | 31.7 | 22.0 | -0.308 | -0.243 | no |
| `J1/c_s225_1040x480` | lagrangian | 85.1 | 54.5 | -0.359 | -0.269 | no |
| `J1/c_s225_1248x288` | lagrangian | 35.1 | 27.4 | -0.220 | -0.110 | yes |
| `J1/c_s225_2080x480` | lagrangian | 148.5 | 107.3 | -0.278 | -0.187 | no |
| `J1/c_s4_1664x192` | lagrangian | 26.9 | 23.3 | -0.132 | -0.067 | yes |
| `J1/c_s4_2496x288` | lagrangian | 85.3 | 57.4 | -0.326 | -0.273 | yes |
| `J1/c_s4_832x192` | lagrangian | 15.1 | 12.6 | -0.168 | -0.048 | yes |
| `J1/c_s4_832x96` | lagrangian | 9.7 | 7.9 | -0.186 | -0.059 | yes |
| `J1/p_s1_1040x240` | par2_rtol1e-10 | 14.2 | 16.2 | +0.145 | -0.125 | no |
| `J1/p_s225_1248x288` | par2_rtol1e-10 | 20.1 | 28.0 | +0.394 | +0.065 | no |
| `J1/p_s4_1664x192` | par2_rtol1e-10 | 15.7 | 16.6 | +0.055 | -0.193 | no |
| `J2/c_s025_M128` | lagrangian | 26.2 | 20.5 | -0.218 | -0.123 | no |
| `J2/c_s025_M192` | lagrangian | 35.8 | 32.2 | -0.103 | -0.093 | no |
| `J2/c_s025_M64` | lagrangian | 20.3 | 13.5 | -0.334 | -0.149 | no |
| `J2/c_s1_M128` | lagrangian | 27.8 | 21.0 | -0.244 | -0.161 | no |
| `J2/c_s1_M192` | lagrangian | 38.9 | 32.7 | -0.159 | -0.155 | no |
| `J2/c_s1_M64` | lagrangian | 23.9 | 14.1 | -0.413 | -0.263 | no |
| `J2/c_s225_M100` | lagrangian | 54.0 | 55.3 | +0.026 | +0.172 | yes |
| `J2/c_s225_M202` | lagrangian | 81.4 | 83.4 | +0.025 | +0.059 | no |
| `J2/c_s225_M58` | lagrangian | 51.9 | 49.3 | -0.050 | +0.124 | no |
| `J2/c_s4_M100` | lagrangian | 131.5 | 119.5 | -0.091 | -0.008 | yes |
| `J2/c_s4_M202` | lagrangian | 193.0 | 175.7 | -0.090 | -0.087 | no |
| `J2/c_s4_M58` | lagrangian | 127.3 | 107.5 | -0.156 | -0.053 | no |
| `J3/c_s025_1040x240` | lagrangian | 26.6 | 20.5 | -0.229 | -0.136 | no |
| `J3/c_s1_1040x240` | lagrangian | 26.1 | 21.0 | -0.192 | -0.103 | no |
| `J3/c_s225_1248x288` | lagrangian | 37.5 | 27.4 | -0.270 | -0.166 | yes |
| `J3/c_s4_1664x192` | lagrangian | 27.8 | 23.3 | -0.159 | -0.095 | yes |
| `J3/p_s025_1040x240` | par2_rtol1e-10 | 11.4 | 16.2 | +0.426 | +0.090 | no |
| `J3/p_s1_1040x240` | par2_rtol1e-10 | 12.7 | 16.2 | +0.278 | -0.023 | no |
| `J3/p_s225_1248x288` | par2_rtol1e-10 | 17.7 | 28.0 | +0.581 | +0.208 | no |
| `J3/p_s4_1664x192` | par2_rtol1e-10 | 15.0 | 16.6 | +0.105 | -0.155 | no |
| `J3B/p_s1_1040x240_h3` | par2_rtol1e-10_h3.6 | 13.8 | 16.2 | +0.176 | -0.046 | no |
| `J3B/p_s225_1248x288_h3` | par2_rtol1e-10_h3.6 | 19.2 | 28.0 | +0.463 | +0.187 | no |
| `J3B/p_s4_1664x192_h3` | par2_rtol1e-10_h3.6 | 17.5 | 16.6 | -0.052 | -0.231 | no |

Systematic sign of the refit residuals. Residual = (model - observed)/observed, so negative means the model is below the observation. Of the 25 consumer pilot points, 23 are below the observation under the SF-39-only model and 22 under the refit (the three positive ones, +0.06 to +0.17, are the 1664x384x384 runs of J2 at `sigma^2` 2.25). The Par2 points under the refit of their own code have mixed signs (4 of 10 positive). The residual table with its covariates:

| point | job | sigma^2 | ny | MG levels | realizations | mean PCG iterations | SF-39 median | ratio | refit residual | first / rest time |
|---|---|---|---|---|---|---|---|---|---|---|
| `c_s1_520x120` | J1 | 1 | 120 | 3 | 1 | 70.0 | 30 | 2.33 | -0.254 | - |
| `c_s225_1040x240` | J1 | 2.25 | 240 | 4 | 1 | 90.0 | 40 | 2.25 | -0.243 | - |
| `c_s225_1040x480` | J1 | 2.25 | 480 | 4 | 1 | 90.0 | 40 | 2.25 | -0.269 | - |
| `c_s225_1248x288` | J1 | 2.25 | 288 | 5 | 1 | 60.0 | 40 | 1.50 | -0.110 | - |
| `c_s225_2080x480` | J1 | 2.25 | 480 | 5 | 1 | 90.0 | 40 | 2.25 | -0.187 | - |
| `c_s4_1664x192` | J1 | 4 | 192 | 6 | 1 | 60.0 | 60 | 1.00 | -0.067 | - |
| `c_s4_2496x288` | J1 | 4 | 288 | 5 | 1 | 120.0 | 60 | 2.00 | -0.273 | - |
| `c_s4_832x192` | J1 | 4 | 192 | 6 | 1 | 40.0 | 60 | 0.67 | -0.048 | - |
| `c_s4_832x96` | J1 | 4 | 96 | 5 | 1 | 50.0 | 60 | 0.83 | -0.059 | - |
| `c_s025_M128` | J2 | 0.25 | 240 | 4 | 6 | 60.0 | 20 | 3.00 | -0.123 | 0.98 |
| `c_s025_M192` | J2 | 0.25 | 240 | 4 | 6 | 60.0 | 20 | 3.00 | -0.093 | 0.94 |
| `c_s025_M64` | J2 | 0.25 | 240 | 4 | 6 | 60.0 | 20 | 3.00 | -0.149 | 1.01 |
| `c_s1_M128` | J2 | 1 | 240 | 4 | 6 | 71.7 | 30 | 2.39 | -0.161 | 0.98 |
| `c_s1_M192` | J2 | 1 | 240 | 4 | 6 | 71.7 | 30 | 2.39 | -0.155 | 0.98 |
| `c_s1_M64` | J2 | 1 | 240 | 4 | 6 | 71.7 | 30 | 2.39 | -0.263 | 1.17 |
| `c_s225_M100` | J2 | 2.25 | 384 | 7 | 6 | 30.0 | 40 | 0.75 | +0.172 | 1.00 |
| `c_s225_M202` | J2 | 2.25 | 384 | 7 | 6 | 30.0 | 40 | 0.75 | +0.059 | 0.99 |
| `c_s225_M58` | J2 | 2.25 | 384 | 7 | 6 | 30.0 | 40 | 0.75 | +0.124 | 0.98 |
| `c_s4_M100` | J2 | 4 | 384 | 7 | 6 | 60.0 | 60 | 1.00 | -0.008 | 1.01 |
| `c_s4_M202` | J2 | 4 | 384 | 7 | 6 | 60.0 | 60 | 1.00 | -0.087 | 0.95 |
| `c_s4_M58` | J2 | 4 | 384 | 7 | 6 | 60.0 | 60 | 1.00 | -0.053 | 0.86 |
| `c_s025_1040x240` | J3 | 0.25 | 240 | 4 | 8 | 61.2 | 20 | 3.06 | -0.136 | 0.93 |
| `c_s1_1040x240` | J3 | 1 | 240 | 4 | 8 | 70.0 | 30 | 2.33 | -0.103 | 0.97 |
| `c_s225_1248x288` | J3 | 2.25 | 288 | 5 | 8 | 57.5 | 40 | 1.44 | -0.166 | 1.02 |
| `c_s4_1664x192` | J3 | 4 | 192 | 6 | 8 | 55.0 | 60 | 0.92 | -0.095 | 1.10 |

negative residuals 22 of 25; Pearson / Spearman of residual with the iteration ratio -0.65 / -0.65; mean residual single-realization runs -0.168 (9), multi-realization runs -0.077 (16); median first / rest realization time 0.98

Hypothesis (not proven, evidence is local and observational): the SF-39-only coefficients, in particular `cells_sigma2`, encode the PCG
iteration counts of SF-39's grids; the pilot grids of SF-43 are different geometries (`ny` = 120, 240, 288, 480 and `nx` = 520, 1040,
1248, 2080) whose multigrid hierarchy has fewer levels (3 to 5 against 5 to 7 in SF-39: the depth stops when an extent becomes odd, so
`nx` = 1040 = 2^4 x 65 caps it at 4 whatever `ny`), and a shallower hierarchy needs more PCG iterations. Evidence in the table: the
pilot points with the SAME `ny` and MG depth as SF-39 (192 with 6 levels, 384 with 7) have iteration ratios 0.67 to 1.0 and the smallest
residuals (-0.10 to +0.17; the three runs at `sigma^2` 2.25 on 384, ratio 0.75, are the +0.06 to +0.17), while the 3 to 5 level grids,
except 832x96x96 (ratio 0.83, residual -0.06), have ratios 1.4 to 3.1 and residuals -0.09 to -0.27; the Spearman correlation between the iteration ratio and the residual is
-0.65 over 25 points. SF-39's levels are those logged in its runs (`STAGE medium=b ... mg_levels=` in the committed
logs: 5 at ny 64, 6 at 128 and 192, 7 at 256 and 384). The alternative that start-up per realization is the cause is not supported:
the median ratio of the first realization's time to the mean of the others in the 6 to 8 realization runs is 0.98. The single-realization
J1 runs have the lower mean residual (-0.17, 9 runs, against -0.08, 16 runs) but five of the nine J1 consumer runs are
also among those with iteration ratio at or above 2, so the two effects are confounded in this data. A cause check that was not made: a run of one grid with
a deeper hierarchy. The consumer effect is absorbed by the refit only partly (rms 0.153); the sign is a reason to read the anchored values
below as the more realistic planning numbers for geometries that were measured.

SF-43 projection (planning table of the R4 brief; not a commitment). Rows: production (h/8) 1040x240x240 @0.25 1100 and @1 1650,
1664x384x384 @2.25 110, 3328x384x384 @4 30; `E_h` rungs (0.25 and 1: 520x120x120 and 2080x480x480, 300 each; 2.25: 832x192x192 and
2080x480x480, 110 each; 4: 832x96x96 and 1664x192x192, 30 each); `E_W` (0.25: 1040x480x480 1100; 1: 1040x480x480 1650; 2.25 2x2
(832x192, 832x384, 1040x240, 1040x480) 110 each; 4 2x2 (832x96, 832x192, 1664x192, 1664x384) 30 each); `h_max` passes 8/16/32 (50 at
0.25, 1, 2.25, 30 at 4, counted as full three-pass realizations); Gate 5 in both codes (1040x240 @0.25 and @1, 1248x288 @2.25,
1664x192 @4) with the J3 planning `n` at the lower-bound correlation (7 / 16 / 63 / 426); G7 factorial (Par2, 1040x240 @0.25 and @1: 8 cells x 60
= 480 each); distance arms (2080x240x240 @1 and 3328x384x384 @2.25, 100 each); plus 5 % for SF-43 pilots and reserve. Consumer `M`
per row = `M_plan` (64, 64, 86, 68) unless stated; the planned Par2 rows use the cost of the horizon SF-43 plane estimators need,
`par2_rtol1e-10_h3.6` (calibrated on the three J3B runs, same seeds and grids as J3; Par2 per-realization time J3B/J3 = 1.09, 1.08, 1.17
while the steps are x3.00 and the transport stage x1.89, x1.91, x2.36; `cost_projection.json` `par2_horizon_cost_ratio`). GPU-h per
family and total (columns after "reserve" are sensitivities: each code's refit rms and largest residual, and "anchored": an arm on a
geometry measured in the pilots is multiplied by the mean observed / refit ratio of those measurements, others keep the model value;
"anchored +1 rms" applies each code's rms on top of the anchored value):

| variant | production | E_h | E_W | hmax | gate5_consumer | gate5_par2 | g7_par2 | distance | reserve 5 % | total GPU-h | -1 rms | +1 rms | +max residual | anchored | anchored +1 rms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M_plan | 16.3 | 22.1 | 45.3 | 3.5 | 3.1 | 2.1 | 3.5 | 4.2 | 5.0 | **105.0** | 88.8 | 121.2 | 146.6 | 130.7 | 150.9 |
| sf39_density | 18.1 | 23.2 | 47.2 | 4.6 | 3.6 | 2.1 | 3.5 | 4.4 | 5.3 | **112.0** | 94.7 | 129.3 | 156.4 | 139.1 | 160.5 |
| M_plan_share010 | 16.6 | 22.7 | 46.0 | 4.9 | 3.8 | 2.1 | 3.5 | 4.7 | 5.2 | **109.6** | 92.7 | 126.5 | 152.9 | 135.5 | 156.3 |
| variant_B_M_plan | 1.8 | 22.1 | 2.8 | 3.5 | 3.1 | 2.1 | 3.5 | 4.2 | 2.2 | **45.3** | 38.2 | 52.3 | 62.6 | 52.5 | 60.6 |
| M_plan_gate5_J3B_floor30 | 16.3 | 22.1 | 45.3 | 3.5 | 1.7 | 1.2 | 3.5 | 4.2 | 4.9 | **102.6** | 86.8 | 118.5 | 143.4 | 128.0 | 147.7 |
| M_common_128_128_86_68 | 20.6 | 24.5 | 49.6 | 4.6 | 3.2 | 2.1 | 3.5 | 4.5 | 5.6 | **118.2** | 100.0 | 136.5 | 165.1 | 147.4 | 170.1 |
| M_plan_par2_horizon1.2 | 16.3 | 22.1 | 45.3 | 3.5 | 3.1 | 2.0 | 3.3 | 4.2 | 5.0 | **104.7** | 88.7 | 120.6 | 146.0 | 130.1 | 150.0 |

Variants: `sf39_density` = `M` at SF-39's seeds per `lambda` (94, 94, 100, 100); `M_plan_share010` = `M` at share 0.10 (64, 64, 140, 106);
`variant_B_M_plan` = "variant B: displacement basis primary (SF-43 pre-registration choice, not assumed)": production and `E_W`
realization counts multiplied by `(SD_disp/SD_pos)^2` of the amplitude (0.027, 0.102, 0.269, 0.880 for `alpha_TP`), floor 30 per arm,
other rows unchanged (informational, not for the stop rule); `M_plan_gate5_J3B_floor30` = the Gate 5 `n` of the J3B pairing (7 / 26 / 205 at 1,
2.25, 4; 0.25 keeps the J3 value) with a floor of 30; `M_common_128_128_86_68` = `M` = 128 at 0.25 and 1, the only values admissible on
every constrained arm there (Finding (a); no `M` is common to all arms at 2.25, 86 being a production-grid value); `M_plan_par2_horizon1.2` = the
planned Par2 rows at the horizon-1.2 cost, kept only to show the horizon's cost (0.3 GPU-h).

GPU-h per arm and per 100 realizations at `M_plan` (so the arms can be resized; "in range" = inside the calibrated cells, `sigma^2` and
particle span of the arm's code; the two out-of-range arms are the Par2 rows at `sigma^2` 0.25, outside the `sigma^2` 1 to 4 span of the three
J3B points; the Par2 model has no `sigma^2` term):

| family | sigma^2 | arm | n | s/realization | GPU-h | GPU-h per 100 | in calibrated range |
|---|---|---|---|---|---|---|---|
| production | 0.25 | 1040x240 | 1100 | 17.2 | 5.27 | 0.479 | yes |
| production | 1 | 1040x240 | 1650 | 17.6 | 8.09 | 0.490 | yes |
| production | 2.25 | 1664x384 | 110 | 61.3 | 1.87 | 1.703 | yes |
| production | 4 | 3328x384 | 30 | 122.4 | 1.02 | 3.401 | yes |
| E_h | 0.25 | 520x120 | 300 | 6.1 | 0.51 | 0.169 | yes |
| E_h | 0.25 | 2080x480 | 300 | 100.8 | 8.40 | 2.800 | yes |
| E_h | 1 | 520x120 | 300 | 6.1 | 0.51 | 0.171 | yes |
| E_h | 1 | 2080x480 | 300 | 104.0 | 8.67 | 2.888 | yes |
| E_h | 2.25 | 832x192 | 110 | 12.9 | 0.40 | 0.360 | yes |
| E_h | 2.25 | 2080x480 | 110 | 112.4 | 3.43 | 3.121 | yes |
| E_h | 4 | 832x96 | 30 | 7.1 | 0.06 | 0.199 | yes |
| E_h | 4 | 1664x192 | 30 | 21.1 | 0.18 | 0.586 | yes |
| E_W | 0.25 | 1040x480 | 1100 | 52.2 | 15.96 | 1.451 | yes |
| E_W | 1 | 1040x480 | 1650 | 53.8 | 24.67 | 1.495 | yes |
| E_W | 2.25 | 832x192 | 110 | 12.9 | 0.40 | 0.360 | yes |
| E_W | 2.25 | 832x384 | 110 | 32.5 | 0.99 | 0.903 | yes |
| E_W | 2.25 | 1040x240 | 110 | 19.8 | 0.61 | 0.551 | yes |
| E_W | 2.25 | 1040x480 | 110 | 58.0 | 1.77 | 1.612 | yes |
| E_W | 4 | 832x96 | 30 | 7.1 | 0.06 | 0.199 | yes |
| E_W | 4 | 832x192 | 30 | 12.4 | 0.10 | 0.344 | yes |
| E_W | 4 | 1664x192 | 30 | 21.1 | 0.18 | 0.586 | yes |
| E_W | 4 | 1664x384 | 30 | 63.1 | 0.53 | 1.751 | yes |
| hmax | 0.25 | 1040x240 | 50 | 28.7 | 0.40 | 0.797 | yes |
| hmax | 1 | 1040x240 | 50 | 29.1 | 0.40 | 0.808 | yes |
| hmax | 2.25 | 1664x384 | 50 | 94.3 | 1.31 | 2.620 | yes |
| hmax | 4 | 3328x384 | 30 | 163.7 | 1.36 | 4.548 | yes |
| gate5_consumer | 0.25 | 1040x240 | 7 | 17.2 | 0.03 | 0.479 | yes |
| gate5_par2 | 0.25 | 1040x240 | 7 | 13.2 | 0.03 | 0.365 | **no** |
| gate5_consumer | 1 | 1040x240 | 16 | 17.6 | 0.08 | 0.490 | yes |
| gate5_par2 | 1 | 1040x240 | 16 | 13.2 | 0.06 | 0.365 | yes |
| gate5_consumer | 2.25 | 1248x288 | 63 | 29.8 | 0.52 | 0.828 | yes |
| gate5_par2 | 2.25 | 1248x288 | 63 | 22.7 | 0.40 | 0.631 | yes |
| gate5_consumer | 4 | 1664x192 | 426 | 21.1 | 2.50 | 0.586 | yes |
| gate5_par2 | 4 | 1664x192 | 426 | 13.5 | 1.59 | 0.374 | yes |
| g7_par2 | 0.25 | 1040x240 | 480 | 13.2 | 1.75 | 0.365 | **no** |
| g7_par2 | 1 | 1040x240 | 480 | 13.2 | 1.75 | 0.365 | yes |
| distance | 1 | 2080x240 | 100 | 31.6 | 0.88 | 0.878 | yes |
| distance | 2.25 | 3328x384 | 100 | 118.9 | 3.30 | 3.304 | yes |

STOP rule (owner, decision G10): the budget is 150 GPU-h; the rule applies to the projected total at `M_plan` and to its +1 rms value.

budget 150 GPU-h (G10); verdict **WITHIN_BUDGET**

| value checked by the rule | GPU-h | margin to the budget |
|---|---|---|
| total_M_plan | 105.0 | 45.0 |
| total_M_plan_plus_1rms | 121.2 | 28.8 |

The rule is not triggered (`stop_budget` = false). The margin of the +1 rms value is 28.8 GPU-h. Reading of the sensitivities, which are not the
rule's value: at `M_plan` the largest-residual total is 146.6 and the anchored value 130.7; the anchored +1 rms value, the most pessimistic
combination, is 150.9, 0.9 above the budget; the variants with denser lattices (`sf39_density`, `M_plan_share010`, `M_common`) are
112.0 / 109.6 / 118.2 at the rule's value and 160.5 / 156.3 / 170.1 anchored +1 rms. The two largest blocks are `E_W` (45.3 GPU-h, of which
the 1040x480x480 arms at 0.25 and 1 are 40.6) and `E_h` (22.1, of which the 2080x480x480 rungs are 20.5); production is 16.3. The
Par2 model is calibrated on 3 to 7 points per code (rms 0.14 to 0.17 on 1 coefficient); the Par2 rows total 5.6 GPU-h (gate5 2.1 + G7
3.5).

#### Reserved seeds

SF-42 reserves for its pilots, to be avoided by SF-43 (earlier campaigns used 4901, 5001+, 7101+, 7201+, 9001+): grid pilot `seed0`
420601-420699, particle pilot 420701-420799, pairing pilot 420801-420899, Par2 `transport.seed` 420800001 (+ 1000 r). Actually used: J1
420601-420609 (one realization each; the Par2 runs reuse the seed of their grid's consumer run: 420604, 420608, 420609); J2 420701-420706,
420711-420716, 420721-420726, 420731-420736 (the same six fields for the three `M` of an amplitude); J3 420801-420808, 420811-420818,
420821-420828, 420831-420838, Par2 transport seeds 420800001 + 1000 r (r = 0..7); J3B the J3 seeds of amplitudes 1, 2.25, 4 with the
same transport seed.

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

(d) **Par2 horizon and crossing completeness for plane-based estimators.** With SF-39's Par2 rule (`n_steps = ceil(1.2 (Lx - 10 lambda) / (J dt))`)
0.2 to 3.8 % of the injected Par2 particles have not crossed the last control plane when the run ends: minimum over eight realizations
of crossed/injected 0.9977 / 0.9829 / 0.9733 / 0.9623 at `sigma^2` 0.25 / 1 / 2.25 / 4. With horizon factor 3.6 (J3B, three times the steps) it is
0.9999 / 0.9996 at 2.25 / 4 and 1.0000 at 1, at +9 %, +8 % and +17 % Par2 time per realization (the transport stage x1.9 to x2.4). The
consumer reaches the last plane with every seeded particle in all J2 and J3 runs. The plane-based Par2 `alpha_LP` is censored by the
horizon, and so is its pairing: the correlation with the consumer's (80 % lower bound) is 0.78 (0.58), 0.73 (0.50), 0.25 (-0.12) at
horizon 1.2 and 0.94 (0.88), 0.86 (0.73), 0.80 (0.62) at 3.6 for `sigma^2` 1, 2.25, 4; the Gate 5 planning `n` at the lower-bound
correlation 16 / 63 / 426 against 7 / 26 / 205. SF-43 needs a stated Par2 horizon (a time step and step count in the pre-registration, and a
crossing-completeness criterion reported per realization) for any plane-crossing estimator; the 3.6 factor meets 0.9996 at the worst
amplitude. SF-39's time-window Gate 5 is not exposed to this censoring (section G, scope).

(e) **GPU memory of the largest grids.** Peak memory of the job's own PID: 28203 MiB at 2080x480x480 (4.8e8 cells; runner exact bytes
27683 MiB), 14365 at 1040x480x480, 12471 at 2496x288x288: 58.9 to 60.2 MiB per million cells. J2 ran the SF-43 production grids
(measured, not extrapolated): 3328x384x384 (4.9e8 cells) at `sigma^2` 4 peaks at 29881 MiB (runner exact bytes 28347.4 MiB, maximum over
`MEM_PHASE` lines; 60.9 MiB per million cells) and 1664x384x384 at 2.25 at 15703 MiB (exact 14173.7 MiB); the peak is the same for the
three `M` of each amplitude (58, 100, 202), so in these runs it does not depend on `M`. The planned 2080x480x480 rungs use 28203 MiB
(J1), of the V100's 32510 MiB. Only an otherwise empty GPU can run them (the R4 jobs waited for 30000 MiB free by `REMOTE_GPU_MIN_FREE_MIB`), which on the host shared with
colleagues is a scheduling constraint for SF-43, not a tuning one. Par2 on its three grids uses 7.7e3 to 12.9e3 MiB.

(f) **Multigrid depth and PCG iterations of the SF-43 grids.** The grids whose extents have few factors of two get a shallower hierarchy (MG
levels 3 to 5 at `ny` 120, 240, 288, 480 and `nx` 520, 1040, 1248, 2080 against 5 to 7 at SF-39's `ny` 64 to 384) and take more PCG
iterations than SF-39's median at the same `sigma^2`: ratios 2.0 to 2.3 at 520x120x120, 1040x240x240, 1040x480x480, 2080x480x480,
2496x288x288 (J1), 1.4 to 1.5 at 1248x288x288; the production grid at `sigma^2` 0.25 and 1 takes 60 to 61 and 70 to 72 mean iterations
(ratios 3.0 to 3.06 and 2.3 to 2.4, J2 and J3), while 192 and 384 (SF-39's own extents) give 0.67 to 1.0. No J1 grid trips the 3x rule
(maximum 2.33); the 0.25 production grid sits at 3.00 (J2: 60 in all 18 realizations) and 3.06 (J3: 60 x7, 70 x1, mean 61.25; the 70 alone 3.5) against the
median 20, outside the J1 protocol (J1 does not run 0.25). Since a recorded count is a multiple of 10 (11 to 20 actual for 20, 51 to 60 for
60), the actual ratio lies between 2.55 and 5.45 and the pre-stated rule cannot resolve it; this is an open decision on the 0.25
production grid for SF-43's pre-registration and the owner. Counts are
multiples of 10 (`check_every`). The link to the hierarchy depth is a hypothesis supported by the covariates (section G, cost model),
not a tested cause; it is also why the SF-39-only cost model under-predicts them (by up to 42 % on the J1 grids). The choice of grids is SF-43's.

(g) **`alpha_LP` drives `M` at 2.25 and 4.** The smallest admissible `M` at planning share 0.25 (0.10) is 86 (140) at 2.25 and 68 (106) at 4
for `alpha_LP`, against 20 (28) to 38 (50) and 14 (20) to 38 (46) for `alpha_TP` (`pos`, `disp`; fixed `M^-2` law / fitted law); at 0.25 and 1
every share at `M` = 64 is at most 8.3e-2, so the lattice rule (smallest admissible `M` 64 on the production grid, 128 common to all constrained arms,
Finding (a)) binds. The planning shares of `alpha_LP` at `M` of about 58 are 0.52 (2.25) and 0.44 (4).

(h) **Gate 5 planning `n` and why `sigma^2` 4 is large.** At the 80 % lower-bound correlation: 7 (0.25, horizon 1.2 only), 7 (1), 26 (2.25),
205 (4) with horizon 3.6; at correlation 0: 64, 34, 83, 380. The specification's unpaired planning figures are about 170 / 60 / 30 / 100;
the pilot's `n` at `sigma^2` 4 is 2 times (lower-bound correlation) to 3.8 times (unpaired) that, and at 2.25 the unpaired 83 is 2.8 times 30. It is driven by the consumer's coefficient of variation on 1664x192x192, 0.30 against
0.145 for Par2, on eight seeds, inflated by the conservative inputs (each CV = the 80 % upper bound of the SD over the 80 % lower bound of the mean at n = 8, and the
correlation at its lower bound 0.62). The `n` of 7 to 26 at 0.25 to 2.25 rests on eight pairs and is below any usable floor; SF-43 will have to set one (the J3B
scenario of the projection uses 30).

(i) **Cost and margin to the owner budget.** Refitted model, planning table (section G, J4): **105.0 GPU-h** at `M_plan` (+1 rms 121.2), budget
150: the STOP rule is not triggered; margins 45.0 and 28.8 GPU-h. The sensitivities that are not the rule's value go from 88.8 (-1 rms)
to 146.6 (+ largest residual) and 130.7 (anchored), and the most pessimistic combination, anchored +1 rms, is 150.9. `E_W` (45.3 GPU-h; the
1040x480x480 arms at 0.25 and 1 alone 40.6), `E_h` (22.1) and production (16.3) dominate; denser lattices cost 112.0 (SF-39 density), 109.6
(share 0.10), 118.2 (`M` common 128). The displacement-basis variant B is 45.3. Par2 at the 3.6 horizon adds 0.3 GPU-h.
The refit under-predicts the measured planned production configuration: J2 `c_s1_M64` took 23.9 s per realization against the projection's
17.6 s (`production 1 1040x240`, 17.64), and `c_s025_M64` 20.3 s against 17.2 (17.25). The anchored total, 130.7 GPU-h, is therefore the more
realistic planning number and the anchored +1 rms value (150.9) the pessimistic edge, at the G10 budget of 150; the margin of the realistic
number to 150 is about 19 GPU-h (19.3). The scenario with a large margin is the displacement-basis variant B (anchored 52.5).

(j) **Coverage of the intervals SF-43 may use (section F).** Qualification (G5.3: UCL at most 1.25 x tail, all tails, `normal`, `stress`,
`stress-`) by procedure and sample size: `LadderBound` (percentile, Student, bootstrap-t) qualifies at every simulated `n` (20 to 200) and the
robust-sign pipeline (K = 3 at `N` 100, 300, 1000; K = 4 at 30, 100, 300; any component method) at every `N`, with very large margins (ladders: UCL / tail at most
0.66; pipeline: 0 misses in 24500 at every Holm level, so it is very conservative; this bears on SF-43's power, not on validity). The Student contrast bound qualifies
for the **unpaired and interaction** contrasts at every `n` from 20 to 200 (conservative under `stress`, exact on `normal`); the percentile bound does not (it fails on
`normal` data at `n` <= 100) and neither does the bootstrap-t (fails under both stress signs, rate / tail up to 2.5; independent check in
`R/verification/indep_boott_unpaired.txt`). The limit of the mean: the Student limit never qualifies (skew side, 8.9 at `n` = 20, 1.3 to 1.4 at 1650);
the bootstrap-t limit qualifies at `n` = 1000 and 1650 and one side at 200 and 500. **No method qualifies** (a) the paired contrast bound under `stress` at any
`n` from 20 to 200 (Student up to 16, bootstrap-t 1.43 at 200 with 10 of 13 tails failing) and (b) Gate 5 TOST under `stress` at any `n` up to 170 (Student 2.0 to 2.4 at
170; bootstrap-t one boundary qualified and the other marginal at best, `rho` 0.5, 170), apart from `rho` = 0 at the smallest `n`, where the TOST is conservative and
"passes" without being robust. Both confirm N6c's small-scale predictions. If SF-43 needs the paired contrast or the TOST under skewed per-realization
distributions at these `n`, it needs a different procedure or a larger `n` (the paired bootstrap-t is at 1.43 at 200 and no larger `n` was simulated); the qualification of what is simulated does not extend to other `n`, tails or distributions (Caveats). **False-fail probability of the
criterion:** an exactly nominal procedure fails a (cell, distribution, row) with probability 0.069 to 0.072 at its smallest tail and 0.078 to 0.084 at some tail,
and a three-distribution group with probability 0.216 to 0.231 (section F table). So "pre-register only qualifying procedures" would reject, by noise alone, about
one nominal procedure in five on this evidence; of the 110 non-qualified groups, 9 are marginal (all failing rows within 1.25 x tail) and 101 clear, and the marginal
ones, in particular isolated normal-data rows, are not evidence of under-coverage, whereas several tails failing on the same skew side (the bootstrap-t TOST at 170)
indicate a residual excess of at most about 20 %. The same criterion at twice the replicates (350 expected misses at the smallest tail, `R` = 98000 for the
sampling tails, `coverage.nominal_fail_probability(R, tails, target=350)`) has a per-row false-fail probability of 0.003 (0.009 for three distributions); at 700 it is negligible.
**Host cost of a re-run:** the `full` plan took 2995 host core-hours, 41.7 h on 72 workers (11.5 times the workstation figure; ladders 89 %), and holds the single project
slot for that long under the current helper (CPU-only jobs hold a GPU slot); a re-run at SF-43's final `n` should be budgeted from this figure, restricted to the
candidate procedures, and doubling `R` doubles it.

## Conclusion

Acceptance A to E are met as recorded in their sections, the design pilots (G) report design inputs without a verdict; the reproducibility tooling (H) passed its recomputation and
reconciliation (`RECOMPUTE PASS`, `RECONCILE OK`). For the statistics library and coverage (F): the library passes its unit tests on both stacks (309 tests
on the host; 346 locally after the R5b report corrections, `OK (skipped=1)`), and the coverage simulation was run in full (1134 of 1134 cells, 428 verdict groups): 310
groups qualify, 110 do not (101 clear, 9 marginal), 8 BCa groups are reported. `LadderBound`, the pipeline and the Student unpaired and interaction contrast bounds
qualify at all simulated sizes; no method qualifies the paired contrast under `stress` (`n` <= 200) or Gate 5 TOST under `stress` (`n` <= 170); the bootstrap-t
bound of the unpaired and interaction contrasts fails under `stress`, which the small-scale runs had not shown. The coverage simulation is a qualification
report for SF-43 and not a gate of SF-42; nothing here estimates `alpha_L` or `alpha_T`.

## Caveats

- Coverage is qualified only for what was simulated: three distributions (a normal and the stress family at the upper bootstrap bounds of SF-39's skewness and
  kurtosis, one skew-and-kurtosis shape in both signs), the listed sample sizes, the tails of the plan, the ladder sequences and cases of the profile (the
  `coverage.py` docstring lists what is not covered), the pipeline at the production sizes, Gate 5 at the two boundary ratios and `rho` 0, 0.5, 0.9, cv 0.3, and contrast
  differences of 5 standard errors. A procedure that qualifies here can still under-cover on other data; one that does not qualify here may do so only
  at this stress level.
- The verdict is judged at the 95 % one-sided Clopper-Pearson limit with design-effect adjustment for the shared bootstrap draws (largest design effect 1.079). The
  false-fail probabilities assume a design effect of 1, so they are lower bounds for the shared-draw cells; the three-distribution value assumes independent distributions.
- The host UCL values differ from the local re-computation by at most 8.7e-15 relative (scipy 1.11 against 1.18); all counts and verdicts are identical. `env.txt`
  records `cpus 1` (artefact, Section F); the cost figures use the topology measured by `lscpu` (80 logical / 40 physical CPUs).
- The mechanism proposed for the bootstrap-t failure of the unpaired and interaction contrasts is a hypothesis; only its existence was reproduced by an
  independent implementation (one seed, `n` = 20, three tails).
- The cost ratio of 11.5 against the workstation is measured, not decomposed; the host was shared and the workers ran at `nice -n 10`.
- The older `RuntimeWarning` in the host `run.log` and the `env.txt` CPU count were corrected after the job started; the job ran the earlier code, and R5b
  verified (chunk recomputation, summary comparison) that the corrections change no chunk result or per-cell verdict row.

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
6. J3B (job `sf42-pilot-horizon`, 0.118 GPU-h) was not in the R4 brief. J3 showed that with SF-39's Par2 horizon rule (factor 1.2) 0.2 to
   3.8 % of the Par2 particles never reach the last control plane and that the plane-based Par2 `alpha_LP` pairs poorly; J3B repeated Par2
   only (J3 seeds, amplitudes 1, 2.25, 4; the consumer tables of J3 are reused, the consumer being independent of the Par2 horizon) with factor
   3.6. It is within the node's 8 GPU-h and is accepted as a justified deviation; it changes no earlier result.
7. The cost-model calibration points of the pilots use the code's own per-realization time, not the process wall (SF-39's fit uses job
   walls with a start-up spread over at least 20 realizations); the wall-basis residuals are in `cost_projection.json`.
8. Lattice sizes: J2 at `sigma^2` 0.25 and 1 ran `M` = 64, 128, 192 (0.68x, 1.37x, 2.05x of SF-39's density) because the nearest admissible
   value to 1x (93.75) is 64, already used for 0.6x, so the repeat-free choice is 128; J3's consumer ran `M` = 128 at 0.25 and 1 (the only
   `M` admissible on every constrained SF-43 arm, 1.37x) instead of the nearest-density 64, and 100 at 2.25 and 4. The J1 `sigma^2` per
   grid equals the brief's mapping; the Par2 time step is SF-39's rule scaled to the grid (CFL `U_nom dt / dx` = 0.0244140625, horizon
   factor 1.2; 3.6 in J3B); Par2 field seeds equal the consumer's of the same grid and amplitude.
9. The R4 worker was interrupted by the owner after all five remote jobs had finished and before its reporting stage (its commits were
   `4b2593cd`, `55d43300`; the experiment note, the cost-report completions and the artifact index were not done). R4b resumed locally
   from the uncommitted files of that worktree: copied unchanged (commit `0a8cd554`), then the report completions (anchored +1 rms, explicit
   STOP-rule record, Par2 3.6 cost code, residual diagnostics, consumer crossing completeness; commit `31af02b1`), unique cost-point labels
   (commit `ca467143`) and the note (commit `6b7f3efa`). R4c (note text only) then corrected the memory finding (the 3328x384x384 and
   1664x384x384 peaks were measured in J2, not extrapolated), stated the 0.25 PCG-count ratio and the 3x rule's resolution limit, and added
   the projection-versus-measured comparison to the cost finding. No job was re-run and `scripts/remote` was not used in R4b or R4c.
10. R5 (the coverage simulation) was interrupted by the owner while its worker polled the running remote job; the worker's only commit,
   `R/verification/job_r5.sh` (the job command in a file, because the sandbox rejected `cd` in `scripts/remote run` text), was kept unchanged as
   the record of what ran, and the running job was not touched. R5b continued from the R4 chain head plus that commit: phase 1, while the job ran
   (local only): the report corrections (the "about 1e-3" false-fail statement replaced by the exact probabilities, incomplete groups marked,
   `env.txt` CPU count by `nproc --all` and `lscpu`, the bootstrap-t `RuntimeWarning` removed by moving one line inside `np.errstate`); phase 2, after
   the job: the outputs, the corrected `summarize`, this section F, the cost table of the `coverage.py` docstring and `run_coverage.sh` header, the independent
   check `indep_boott_unpaired.py`. These corrections change no chunk result and no per-cell verdict row (evidence in section F: the plan hash, 14 host chunks
   recomputed exactly, all 13734 tail rows of the final summary equal to the host's old-code summary in counts, `R`, `pass`, `powered`, design effect and rate; the UCL at
   rounding level). The job's own `env.txt` keeps `cpus 1` and its `run.log` the two `RuntimeWarning` lines. The job held the single project slot for 41.693 h
   although it is CPU only.
