# SF-41 artifact: campaign tooling (DAG node N7)

Increment: `docs/plans/active/lester-eq14/increments/SF-41-medium-b-lagrangian-stage.md`. Reading rules: the pre-registration
(`git show fa3f253:docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/preregistration.md`, RR0-RR12). This node adds
the scripts only; the pre-registration, `raw/`, `analysis/` results and the experiment note are added by later nodes.

## Contents

```
scripts/run_sf41.sh                 campaign runner: run_sf41.sh <build_dir> <part>
scripts/a_verdict.py ... h_verdict.py   verdict scripts (python3 + numpy, deterministic, no network), each with --selftest
scripts/sf41lib.py                  shared log readers
scripts/selftest_all.sh             runs every --selftest (verdict scripts and the copied SF-38 tools)
scripts/make_bootstrap_blocks_f.py  writes analysis/bootstrap_blocks_f.npy (RR8)
analysis/bootstrap_blocks_f.npy     default_rng(20261025).integers(0, 4, size=(2000, 4)), int64; sha256 in analysis/bootstrap_blocks_f.sha256
scripts/{pid_peak,mem_verdict,json_strip_cmp,lines_cmp,d_checks,ab_decision}.py   verbatim copies of the frozen SF-38 tools
```

The SF-38 tools are byte copies (`cp`), used for: the RR2 memory sampler reader (`pid_peak.py`, `mem_verdict.py`), the SF-27 trio / R-set
byte compare (`json_strip_cmp.py`, `lines_cmp.py`), the tree compare helper kept for reference (`d_checks.py`) and the H recomputation
(`ab_decision.py`, unchanged: it recomputes the SF-38 `C_STAT` lines).

## Running

Everything below is run by the orchestrator as detached V100 jobs, one at a time (`scripts/remote --increment SF-41 run <job> -- "bash $A/scripts/run_sf41.sh <build_dir> <part>"`
then `wait`), `A=docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage`. Environment: `SF41_COMMIT` (recorded in `run.env`),
`SF41_OUT_ROOT` (default `~/sf41_campaign`), `SF41_BASE_REFS` (default `~/sf41_base_refs`), `SF41_DRYRUN=1` (local dry run, not evidence).

| part | binaries | runs | verdict lines in `results.txt` |
|---|---|---|---|
| `build` | `<build_dir>` (head) | CLEAN build, `ctest -N`, md5 of every executable | `BUILD_BINARIES`, `BUILD_VERDICT` |
| `baserefs` | `~/sf41_base_refs/bin` (base `2cf48c1`) | SF-27 trio, SF-38 20-run R set, G(ii) runner run, G(iii) A/B run | `BASEREFS_RECORDED` |
| `controls` | head | A (`--full`, memcheck of the domain group), C1-C4, C5, D (sampler), memory models | `A_VERDICT`, `C1..C5_VERDICT`, `C_VERDICT`, `D_VERDICT`, `RECORDED D_MEM` |
| `cross` | head | B (`--only B`) | `B_VERDICT` |
| `prodgrid` | head | E1-E4 (sampler), mapped-host re-run iff PID peak > 30000 MiB | `E_RUN`, `E_VERDICT id=`, `E_VERDICT ALL` |
| `evladder` | head | F: 24 runs | `F_CELL`, `F_VERDICT eps=`, `F_VERDICT ALL` (+ `RECORDED F_DP/F_V3B_SUMMARY`) |
| `abrerun` | head | H: the two SF-38 A/B commands | `H_REALIZATION`, `H_VERDICT` |
| `final` | head | full `ctest`, trio, R set, G(ii), G(iii) vs `~/sf41_base_refs` | `G_CTEST`, `G_TRIO`, `G_RSET`, `G_BYTES`, `G_VERDICT` |

Order: (base) `baserefs` on a tree equal to the increment base before any SF-41 source reaches the mirror (job `sf41-baserefs`), then, on the head
tree, `build`, then `controls cross prodgrid evladder abrerun final`. Every part writes under `$SF41_OUT_ROOT/<part>/` (baserefs: `$SF41_BASE_REFS`):
`run.env` (DATE_UTC, host, GPU, driver, nvcc, gcc, binary md5s, `SF41_COMMIT`, tree echo, binaries check, nvidia-smi banner), `exit_codes.tsv`
(`run<TAB>rc`), `walls.tsv`, `results.txt` (machine-readable verdict and `RECORDED` lines; ends with `RECORDED part_seconds=...` and
`PART_DONE part=<p> rc=<0|1>`), `sha256.txt`, `PART_EXIT`, and per run `<run>/`, `<run>.log`, `<run>.time` (`/usr/bin/time -v`), `<run>.free` (`free -b`),
`<run>.precheck` (no compute process on the GPU, before every sampled or exclusive run), and for sampled runs `<run>.apps.csv`, `<run>.dev.csv`,
`<run>.pid_tree` (RR2). A run whose rc is recorded is never re-run; a run with partial outputs and no rc is cleaned and re-run with identical settings
(`resumed.txt`); an unexpected exit code is recorded and makes the part's rc 1 (RR0: no outcome-dependent stopping, no retry with changed settings).

### Clean build and binary identity (orchestrator rule)

The shared mirror receives successive rsyncs of different trees and rsync keeps source mtimes, so an incremental build can keep stale objects. Therefore:

1. the head build is always a CLEAN build, made by `run_sf41.sh build/v100-release build`: `rm -rf build/v100-release && cmake --preset v100-release && cmake --build build/v100-release -j`
   (refused unless `<build_dir>` is `<repo>/build/<preset>`); the part echoes the tree id (git head/tree when `.git` exists, plus a sha256 digest of
   `src/ tests/ apps/ CMakeLists.txt CMakePresets.json`), records `ctest -N` (`ctest_N.log`, `ctest_names.txt`) and the md5 of every executable of the build
   directory in `$SF41_OUT_ROOT/build/binaries.md5` (`md5sum` format; the nine binaries the parts use are also echoed as `RECORDED binary_md5`);
   re-invoking the part moves the previous `build/` record to `build.superseded.<UTC>` and builds again;
2. every other head part verifies that file first (in `part_begin`): it REFUSES TO RUN (exit 2, `PART_REFUSED ...` and `PART_DONE rc=1` in `results.txt`)
   if the file is absent or any listed binary differs, and also if its own part directory holds results of another build (`.build_digest`); the
   check is echoed in `run.env` as `BINARIES_CHECK=ok(<n> binaries, md5file_sha256=...)`;
3. the base binaries in `~/sf41_base_refs/bin` come from a fresh mirror and are used as they are; `baserefs` only records their md5.

### Local dry run

`SF41_DRYRUN=1 SF41_OUT_ROOT=<scratch> bash run_sf41.sh <local build dir> <part>`: tiny settings (fast tiers of the test programs, E on 128x16x16 / 64x16x16,
F on 24x8x8 .. 48x16x16 with two rungs and two seeds, H on 64x16x16, G(ii)/(iii) on 64x16x16, the base references made from the same binaries, a one-run R set, no trio,
no `compute-sanitizer`, no exclusive-GPU precheck; `SF41_DRY_FORCE_MAPPED=1` exercises the mapped-host rule of E). The verdict scripts run with `--dry` where noted. Dry runs
are not evidence; H and the `interpolation_*` entries of G are expected NOT_OK / FAIL there (different grids, one ctest case).

## Verdict rules (reading rules RR; the scripts re-evaluate the logs, they do not trust the programs' own verdict fields)

- **A (RR3)** `a_verdict.py`: every gated `A_CHECK` is recomputed from `value`/`gate` (`*_order_ge_*`: value >= gate; others value <= gate), must agree with the
  program's field, and every required (name, box, side) of the RR3 full tier must be present; the program exits 0 with `A_VERDICT all_pass=1`; the
  `compute-sanitizer --tool memcheck ... --full --only domain` run exits 0 with `ERROR SUMMARY: 0 errors`. Recorded: ungated `A_CHECK`s, `A_LADDER_ORDER`, `A_EMUL`
  (measured-over-emulated), the x-only sub-ladder, `A_MEMORY`.
- **B (RR4)** `b_verdict.py`: configuration `tier=RR4 grid=1664x128x128 ell=0.25 lambda_over_h=32 eps=2 seed=6001`, three converged SF-19 correctors, `h_tilde` value/gradient/Hessian `rel <= 1e-11`
  (recomputed), `B_VERDICT all_pass=1`. Recorded: the `lnK` metrics, all `B_DECAY` lines and their mean per (field, quantity) (expected ~0.268), the profile row count.
- **C1-C5 (RR5)** `c_verdict.py`: per case the required `C_CHECK`s of the full tier (grid x path) re-evaluated (`<=x`, `==0`, `>0`); C4 also needs the Pollock path inside the controls
  program and every `C4_CHECK` of `pollock_nonperiodic_tests --only C4` PASS with counts equal to `expected=`; C5: `converged=1`, `rel <= 1e-10`, `balance <= 1e-8` recomputed, and on the C1
  medium `initial = rel = 0`, 0 iterations. Recorded: `C_RECORD` lines (C2 `I` vs quadrature, reversed head), the C5 head differences.
- **D (RR6)** `d_verdict.py`: `D_CHECK` identical head and B_SOLVER at 128x32x32, 512x64x64, 3328x256x256 (the `D_REF`/`D_LEAN` lines are compared again as strings), mapped-host identical at the two
  small sizes. D fails only on those comparisons or a non-zero exit. Recorded (not gated): the RR2 PID peak and device-wide peak of the whole process (`D_MEM`), the field-by-field comparison with SF-38's committed C0 line.
- **E (RR7, RR2)** `e_verdict.py`: solve `primary` (converged, rel <= 1e-10) or the C0 `fallback` (rel <= 1e-8), balance <= 1e-8 from `Q_in`/`Q_out`, exit 0, stage grid, PID peak <= 30000 MiB, and
  the non-solve-phase rule from the exact `MEM_PHASE` bytes (every phase F, S1, V, H, S3, T, B <= S2 and <= 30000 MiB; violation = placement defect, BLOCKING). A device run whose PID
  peak exceeds 30000 MiB needs the `E<k>_mapped` re-run (`PASS_MAPPED_HEAD` if it meets everything, FAIL `peak_above_30000_after_fallback` = R3 for the owner otherwise).
  If a device run dies by an out-of-memory error whose sampled peak stayed <= 30000 MiB, no fallback is run (RR7 triggers on the PID peak only) and the grid is FAIL (`exit=`): the
  orchestrator decides. Recorded: wall, RSS, per-phase bytes, `STATUS`/`TEXIT`/`TSTATUS`/`FLOWSTATS`/`RESOLVED_TIMES`, `status.json`, `flow_stats.json`, `solver_b.json`.
- **F (RR8)** `f_verdict.py`: per (eps, seed, plane): the four tracker `rms_ev` values (15 digits, `moments_planes.csv`) strictly decreasing and `log2(e_3/e_4) >= 1.8`; a missing run, a different plane list
  or a non-finite value is NOT MET. `D_P`, `SE_D` (committed bootstrap matrix, sha256 verified), V3b classification: recorded only; the recomputed `var_x2/var_x3` are checked against the
  programs' own moments CSVs (`csv_max_rel`).
- **G (RR9)** `g_compare.py` + the shell: `G_CTEST` (full ctest green), `G_TRIO`, `G_RSET` (SF-38 compare rules), `G_BYTES name=runner_medium_a` (every file except `timings.json`, `DONE` included) and
  `G_BYTES name=medium_ab` (same, `realizations/r_*/exits_b.csv` listed as the allowed extra), plus `interpolation_periodic_tricubic` and `interpolation_hessian` Passed in `ctest_times.tsv`.
  The emptiness of `git diff 2cf48c1..H1` for the periodic spline files (RR9 (iv)) is the orchestrator's check.
- **H (RR10)** `h_verdict.py`: per realization the exit counts (`exits_b.csv`), the byte comparison of `e2_b.csv` and of the control `e2_a.csv` with the committed SF-38 files, and the failure-policy
  class (`OK_ZERO_EXITS_IDENTICAL`, `EXITS_RECORDED`, `DEFECT_EXIT_AWARE_POLLOCK` = BLOCKING, `ENVIRONMENT_INVESTIGATE`, `MISSING`); with any exit the unchanged `ab_decision.py` recomputes the
  `C_STAT` lines on the head `ab_primary.csv` files (`RECORDED H_RECOMPUTED`) beside the originals (`RECORDED H_ORIGINAL`). The SF-38 decision is not re-decided.

## Evidence import (by the orchestrator)

Same scheme as SF-38. After the campaign (last job `sf41-final` finished 2026-10-25T04:22:44Z), one short `scripts/remote --increment SF-41 exec`
packed the text subset of `~/sf41_campaign`, `~/sf41_base_refs` and `~/sf41_campaign_superseded` into `~/sf41_evidence.tar.gz`, together with every
job log (`~/.macroflow3d-remote/macroflow3d-SF-41/logs/*.log` -> `raw/job_logs/`), every job status file (`.../status/*.env` -> `raw/job_status/`)
and the list of excluded large files (`raw/sf41_evidence.excluded_large`). Selection: every regular file except `*.npy` and `sf41_base_refs/bin/*`,
with size `<= 2048 KiB`; the list of larger non-`.npy` files is empty. **Excluded from `raw/`** (their sha256 are in the per-part `sha256.txt`
manifests, the integrity record): 862 `*.npy` files (tracker, Pollock and SF-27 / R-set per-particle arrays) and the base binaries.
The tarball was extracted unchanged into `raw/` (V100 relative layout: `raw/sf41_base_refs/`, `raw/sf41_campaign/<part>/`,
`raw/sf41_campaign_superseded/build_attempt1_tmpfull/`, `raw/job_logs/`, `raw/job_status/`).

Imported (orchestrator, 2026-10-25): `sf41_evidence.tar.gz` sha256 `2ef0ca6c2ab587639f507e1d36dfc2a15bb4ddfc1bc65e9b84a920091ae5a633`
(12303574 B, 1861 files, none excluded as large). Transfer: base64 through `scripts/remote --increment SF-41 exec`, sha256 equal on both hosts;
no absolute or `..` member, no link or device entry. Re-verification: `for d in raw/sf41_base_refs raw/sf41_campaign/*; do (cd $d && sha256sum -c
--ignore-missing sha256.txt | grep -vc ': OK'); done` gives 0 for each of the eight part directories (565, 258, 8, 33, 4, 984, 572, 196 listed
entries for baserefs, abrerun, build, controls, cross, evladder, final, prodgrid). The verdict commands of "Re-running a verdict from `raw/`"
reproduce the campaign's lines (`A_VERDICT ... PASS`, `C_VERDICT PASS`, `D_VERDICT ... PASS`, `B_VERDICT ... PASS`, `E_VERDICT ALL PASS`,
`F_VERDICT ALL MET`, `H_VERDICT ALL class=OK_ZERO_EXITS_IDENTICAL`, `G_VERDICT PASS`), all with exit 0.

Not evidence, imported for disclosure only: `raw/job_logs/sf41-pre-*.log` (pre-checks on intermediate node trees before integration, some on
incremental builds of the shared mirror that may have linked stale objects) and `raw/sf41_campaign_superseded/build_attempt1_tmpfull/` (head build
attempt 1, failed because the V100 `/tmp` (2.0 GB) filled with the temporaries of parallel `nvcc -dlink` processes (`clean_build.log`: `fatal error: error writing to /tmp/....s: No queda espacio en el
dispositivo`, `PART_EXIT` 1); every attempt-2 part ran
with `TMPDIR=$HOME/sf41_tmp`, echoed in the job command of each `raw/job_logs/sf41-<part>.log`). The remote job log is one file per job name,
so `raw/job_logs/sf41-build.log` is the attempt-2 log. Host-local clocks (`nvidia-smi`, sampler) are 3 h behind the UTC stamps.

## Re-running a verdict from `raw/`

Each verdict reads only the files of its part directory (`raw/sf41_campaign/<part>/`):

```
S=docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts; R=docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/raw/sf41_campaign
python3 $S/a_verdict.py $R/controls ; python3 $S/c_verdict.py $R/controls ; python3 $S/d_verdict.py $R/controls --run D
python3 $S/b_verdict.py $R/cross --run B
python3 $S/e_verdict.py $R/prodgrid
python3 $S/f_verdict.py $R/evladder        # defaults = the RR8 design
python3 $S/h_verdict.py $R/abrerun
python3 $S/g_compare.py verdict $R/final/results.txt --ctest-times $R/final/ctest_times.tsv
bash $S/selftest_all.sh
```
