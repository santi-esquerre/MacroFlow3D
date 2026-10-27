# SF-39 macrodispersion study: campaign mechanics

What this is for: how the SF-39 runs are launched, bookkept, restarted and imported. What it is not: the settings, estimators
and reading rules, which are fixed in [`preregistration.md`](preregistration.md) (normative; sections 1, 2, 6-9 are the ones
this file implements), nor any result. No `alpha_L` / `alpha_T` statement lives here.

Artifact root `A = docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/`. Everything below runs on the V100 host
from the mirror `~/MacroFlow3D-SF-39` (job working directory = mirror root), as detached `scripts/remote run` jobs (RR0); nothing
heavy runs locally and workers never use the remote. `$SF39_OUT_ROOT` = `$HOME/sf39_campaign` on the host (outside the mirror,
which `sync --delete` would erase).

## Files

| file | contract | role |
|---|---|---|
| `scripts/run_sf39.sh` | C-RUN | the six parts below; bookkeeping, md5 gate, restart rules |
| `scripts/sf39/geometry.py` | C-GEOM | arm table (section 1), plane algorithm (section 2), mirror of the runner's parser checks |
| `scripts/sf39/bytecheck.py` | RR3 (iii) | `--hmax-div 8,16,32` vs `8` byte comparison per F12 |
| `scripts/sf39/tests/test_geometry.py`, `test_bytecheck.py` | | synthetic known answers, seconds |
| `configs/par2_s{1,225,4}.yaml` | C-PAR2 inputs | Par2 pilot configs (copies of the SF-38 configs; only `stochastic.seed: 9001`, `NR: 20`, `output_dir: raw/par2_s{c}` and the header differ) |
| `configs/par2_s{c}_ext.yaml` | | Par2 extension, written by `python3 -m sf39.par2 mkext` after the budget |
| `configs/budget.json` | | final counts, created after the owner's budget (schema below) |

Geometry CLI (needs `PYTHONPATH=<A>/scripts`): `python3 -m sf39.geometry flags <arm>` (runner arguments of an arm except `--out`,
`--seed0`, `--nr`), `planes <P1..P7|arm>`, `arms`, `check` (the seven sha256 values of section 2 and every arm against the
runner's usage checks: grid even and >= 4, `ny == nz`, planes strictly increasing, `> x0 = 5 ell`, last `< L1`, window order,
RR12 seed-on-face for the `M = 100` lattice, `--hmax-div` list). Arm ids: `s{1,225,4}_A`, `_B`, narrow rungs `s1_n05 s1_n10
s1_n20 / s225_n08 s225_n16 s225_n32 / s4_n16 s4_n32 s4_n64`, twins `s{c}_tw`, width arms `s4_w24`, `s4_w12`, and check arms
`chk_<arm>` for the 14 narrow/twin/width geometries plus `chk_s4_A` (`--hmax-div 8,16,32`) and `chk_s4_A1` (`8`).

Local tests (seconds): `cd A/scripts && python3 -m unittest discover -s . -t .` (or `-p 'test_geometry*.py'`, `-p 'test_bytecheck*.py'`);
`bash -n scripts/run_sf39.sh`; `shellcheck -s bash scripts/run_sf39.sh`; `SF39_DRYRUN=1 bash scripts/run_sf39.sh build/v100-release <part>`
prints every command with resolved flags and touches nothing (dry runs are not evidence).

## Parts

`bash scripts/run_sf39.sh <build_dir> <part> [selector]`, `<build_dir>` = `build/v100-release`. Every part first verifies
`cd <build_dir> && md5sum -c $SF39_OUT_ROOT/build/binaries.md5` (the five binaries must be listed) and refuses with exit 2 and
`PART_REFUSED` otherwise (RR1: the build is never redone). No GPU is selected and no exclusivity is checked: the shared V100 is
governed by the `scripts/remote` slot and `REMOTE_GPU_MIN_FREE_MIB`.

| part | selector | what it runs | CUDA |
|---|---|---|---|
| `checks` | | (i) whole unit-test suite; (ii) `sf39.geometry check`; (iii) RR2 regression (`scripts/regress` reference into `$HOME/regress/refs/1e40cfe/out`, reused iff `REGRESS_DONE`, a directory without it stops this check; head into `checks/regress_head/out`; `compare.sh`, last line must be `REGRESS IDENTICAL`; `TMPDIR=$HOME/.cache/mf3d-tmp`); (iv) RR3(ii) solve checks, seed 4901, `--nr 1`, the 14 check arms, verdict from `solver_b.json` (`converged` or `fallback` accepted) or the `B_SOLVER` line on exit 4; (v) RR3(iii) byte check at the `s4_A` geometry (`chk_s4_A` vs `chk_s4_A1`); (vi) `sf39.reduce smoke $HOME/sf41_campaign` and `sf39.reduce signtable $HOME/sf41_campaign/evladder` | yes |
| `pilot` | `s1` `s225` `s4` `par2` (none = all four in this order) | per amplitude `s{c}_A`, the three narrow rungs and the twin, each `--seed0 5001 --nr 20` in `runs/<arm>`; `par2`: `macroflow3d_pipeline configs/par2_s{c}.yaml` (cwd `$SF39_OUT_ROOT/par2`) and `par2_field_stats <cfg> --nr 20` | yes |
| `checkpoint` | | `python3 -m sf39.checkpoint --out-root $SF39_OUT_ROOT --artifact A` (freeze, sealed pilot estimates, `nr_rule`, blinded report); a recorded rc 0 is never re-run | no |
| `extend` | `s1` `s225` `s4` `par2` | needs `configs/budget.json` (refuses with exit 2 and a message if absent or invalid, before anything is written); raises `--nr` in the same directories (A to `N_A`, narrow rungs to `N_narrow` with `s4_n16` at least 30, twin to `N_twin` with `s4_tw` at least 30), creates directory B (`s{c}_B`, `--seed0 5001+N_A --nr N_run-N_A`, only if `N_run > N_A`); `par2`: needs `configs/par2_s{c}_ext.yaml` equal to the output of `python3 -m sf39.par2 mkext configs/par2_s{c}.yaml <par2_NR>` (refuses otherwise) and runs it plus `par2_field_stats` on it; skipped for `par2_NR <= 20` | yes |
| `width` | | `s4_w24` (`--seed0 7101 --nr 30`), `s4_w12` (`--seed0 7201 --nr 30`) | yes |
| `analysis` | | `python3 -m sf39.analyze --out-root $SF39_OUT_ROOT --artifact A --dest $SF39_OUT_ROOT/analysis`; re-run on every invocation (deterministic, idempotent) | no |

Every runner invocation is `lagrangian_ensemble $(python3 -m sf39.geometry flags <arm>) --seed0 S --nr N --out DIR` under
`/usr/bin/time -v`, stdout and stderr to `<part>/logs/<id>.log`; id = `<arm>:s<seed0>:n<nr>` (check arms `chk_...`, byte-check
runs `byte_multi`, `byte_single`; Par2 `par2_s{c}[_ext]` and `fs_par2_s{c}[_ext]`). `par2_field_stats` stdout is also saved as
`par2/field_stats_par2_s{c}[_ext].txt` (the `FIELDSTAT` lines, sha256 in `results.txt`).

## Launching a part (the orchestrator; one job at a time)

```bash
# CUDA parts: wait for a slot with >= 30000 MiB free
env REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-39 run sf39-<part> -- \
  "PYTHONPATH=docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts bash docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh build/v100-release <part> [sel]"
scripts/remote --increment SF-39 wait sf39-<part>
# CPU-only parts (checkpoint, analysis): the same without REMOTE_GPU_MIN_FREE_MIB
env REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-39 run sf39-checkpoint -- "PYTHONPATH=... bash .../run_sf39.sh build/v100-release checkpoint"
```

Job names: `sf39-checks`, `sf39-pilot-s1` ... `sf39-pilot-par2`, `sf39-checkpoint`, `sf39-extend-<sel>`, `sf39-width`,
`sf39-analysis`. Never `--gpu`, never `CUDA_VISIBLE_DEVICES`, no GPU work under `exec`, no `sync` while an SF-39 job is `running`
or `waiting_gpu`; later syncs carry scripts only (RR1). The run script prepends `<A>/scripts` to `PYTHONPATH` itself.

## Outputs layout (`$SF39_OUT_ROOT`)

```
build/binaries.md5            md5sum format, bare file names relative to <build_dir> (written by the RR1 build job)
halted_arms.txt               arm<TAB>part<TAB>rc<TAB>utc, one line per runner exit 3/4 (RR6); the owner removes a line to release the arm
runs/<arm>/                   runner output (realizations/r_%04d/..., hmax_*/ for directory A, stats/, stats_twin/, ...)
par2/                         Par2 working directory: raw/par2_s{c}[_ext]/ (pipeline output), field_stats_*.txt
checkpoint/, analysis/        written by the checkpoint / analysis modules (sealed/ holds the blinded pilot estimates)
<part>/                       run.env  exit_codes.tsv  walls.tsv  results.txt  sha256.txt  resumed.txt  logs/  PART_EXIT
checks/runs/, checks/regress_head/   check runs (never data)
```

`exit_codes.tsv`: `id<TAB>rc`, no header, the last record of an id wins. `walls.tsv`: `id<TAB>wall_s<TAB>end_utc`. `results.txt`:
`PART_BEGIN`, `RUN` / `CHECK_<name> rc=<rc> verdict=<verdict>` / `ARM_HALT` / `UNEXPECTED_EXIT_CODE` lines, `RECORDED part_seconds=...`
(sum of run walls = GPU-busy upper bound on a shared GPU), and last `PART_DONE part=<p> rc=<0|1>` (rc 1 if any item needs the
owner). `sha256.txt`: sha256 of the part's small text files (logs, tsv, `run.env`) and of the per-realization `DONE`, `status.json`,
`solver_b.json`, `flow_stats.json`, `field.json`, `timings.json` of the runs of that invocation (hashes only: nothing is read).
`run.env` records the date, host, binaries md5 and gate result, `.mf3d_source_rev` of the mirror, the scripts commit
(`SF39_COMMIT`, else `.mf3d_source_rev`, else `git HEAD`), python, the `REMOTE_GPU_*` variables and the `nvidia-smi` banner.

## `configs/budget.json` (schema)

Created by the orchestrator after the owner's budget (checkpoint report, preregistration section 6, budget rule); every final `N` is
also recorded in the bitacora before any further run. One object per amplitude, positive integers (extra keys are ignored):

```json
{ "s1":   {"N_narrow": 24, "N_twin": 22, "N_hmax": 20, "N_R": 40, "N_A": 20, "N_run": 40, "par2_NR": 40},
  "s225": {"N_narrow": 20, "N_twin": 20, "N_hmax": 20, "N_R": 20, "N_A": 20, "N_run": 20, "par2_NR": 20},
  "s4":   {"N_narrow": 20, "N_twin": 20, "N_hmax": 20, "N_R": 30, "N_A": 20, "N_run": 30, "par2_NR": 30} }
```

Meaning (section 1): `N_A = max(20, N_hmax)` realizations in directory A, `N_run = max(N_R, N_A, N_twin)`, directory B holds
`N_A <= g < N_run`, `par2_NR = N_R` (Par2 extension runs `par2_NR - 20`). `extend` checks only: all keys present, `N_narrow`, `N_twin`,
`N_R`, `N_A`, `par2_NR >= 20`, `N_A >= N_hmax`, `N_run >= max(N_A, N_R, N_twin)`. The example above is illustrative, not a budget.

## Blinding (RR4)

Until the owner's budget is recorded nobody reads pilot plume statistics. Allowed to read: `DONE`, `status.json`, `solver_b.json`,
`flow_stats.json`, `timings.json`, `tau_b_star.json`, job logs, the part bookkeeping above and the checkpoint report. Not read nor
imported: runner `analysis/`, `ensemble/`, `stats*/`, per-realization moment CSVs and arrays of pilot arms, Par2 `stats/`, `analysis/`,
`ensemble/`, `checkpoint/sealed/`. The scripts echo only `STATUS`, `RUN_DONE`, `B_SOLVER`, `STAGE_FAILED`, `SEED_FACE_FAILED`
lines of a log. The `par2/field_stats_*.txt` files are field statistics (`var_lnK`), not plume statistics.

## Evidence import (text only)

Per-particle arrays (`*.npy`) never leave the host. Reduced CSVs, logs and sha256 lists are imported as base64 text through
`scripts/remote exec` (short steps), with sha256 checked on both hosts:

```bash
scripts/remote --increment SF-39 exec -- "cd \$HOME/sf39_campaign && mkdir -p export && tar -czf export/<name>.tgz --exclude='*.npy' <paths> && sha256sum export/<name>.tgz"
scripts/remote --increment SF-39 exec -- "base64 -w0 \$HOME/sf39_campaign/export/<name>.tgz" | base64 -d > <name>.tgz
sha256sum <name>.tgz     # must equal the remote value
```

Before the budget only the RR4 list above (and the checkpoint report) may be imported.

## Restart rules

* A runner id whose recorded rc is final (0 ok; 2 usage; 3 seed on a face; 4 unusable solve; 7 option mismatch: deterministic
  outcomes) is never re-run. Any other recorded rc, or partial outputs with no recorded rc (ssh drop, OOM caused by a colleague,
  reboot, exit 75/70 of the job wrapper), is re-run on the next invocation with identical settings (the runner skips
  realizations with `DONE`, removes stale `*.tmp` and unfinished `r_%04d`) and listed in `resumed.txt`; disclose it. Nothing is ever
  retried with changed settings, and a scientific outcome is never re-run (RR0).
* A runner exit 3 or 4 halts that arm (RR6): it is recorded in `halted_arms.txt`, the part continues with the other arms and ends
  with rc 1, and every later invocation refuses the arm (`ARM_HALTED`) until the owner decides and removes the line.
* Par2 ids are final only at rc 0; an interrupted or failed Par2 run is re-run from scratch after removing its
  `par2/raw/par2_s{c}[_ext]` directory (the pipeline has no restart).
* Raising `--nr` in the same output directory is how `extend` works (one `--seed0` per directory; the runner exits 7 on a
  different option signature). Do not reuse a directory with another seed.
* A directory in `$SF39_OUT_ROOT/<part>` produced by another build is refused (`STALE_PART_DIR`).

## Test-only knobs (never set in the campaign)

`SF39_BUDGET` (budget file path, to inspect `extend` commands with a test budget in a dry run), `SF39_TIME_BIN`, `SF39_REGRESS_HOME`
(base of `regress/refs/` and `.cache/mf3d-tmp`), all used only to test `run_sf39.sh` with fake binaries.
