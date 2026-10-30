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
- Sections D to H and the findings for SF-43 that depend on them are filled by the later nodes (MMS, spectral cross-check, library
  and coverage, pilots, reproducibility tooling).

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
| total | | | 2952 | 0.820 | | |

Two builds are compile-only jobs that hold the GPU slot (0.173 GPU-h of the total). Base and head both ran on GPU 0 of the same host.
The first launch attempts of `sf42-base` and `sf42-bytes` were refused by the agent sandbox before reaching `scripts/remote` (no job was
created); they were resubmitted unchanged in meaning.

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

Pending (later nodes).

## E. Spectral cross-check

Pending (later nodes).

## F. Statistics library and coverage

Pending (later nodes).

## G. Design pilots

Pending (later nodes).

## H. Reproducibility tooling

Pending (later nodes).

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
