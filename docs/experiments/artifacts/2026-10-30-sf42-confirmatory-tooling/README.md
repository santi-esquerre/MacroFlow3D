# SF-42 confirmatory tooling and verification: artifact root

What this is for: what the artifact root of increment SF-42
(`docs/plans/active/lester-eq14/increments/SF-42-confirmatory-tooling-verification.md`) contains, how to run its unit tests and where
the host outputs live. What it is not: the results or their interpretation (the experiment note
`docs/experiments/2026-10-30-sf42-confirmatory-tooling.md`) and not a macrodispersion estimate; nothing here estimates `alpha_L` or
`alpha_T`.

Artifact root `R = docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/`.

## Contents

| path | what | state |
|---|---|---|
| `scripts/mcstat/` | statistics and error-budget library (estimators, Student / bootstrap-t / BCa intervals, robust limit, `E_h` and `E_W` rules, TOST, split-plot analysis, Holm, coverage simulator, blinded variance tool, cost model) with known-answer and SF-39 reproduction tests | delivered |
| `scripts/mms/` | manufactured-solution order study of the Dirichlet head solve: `run_mms.sh`, `analyze.py` (C++ driver `apps/mms_dirichlet/`) | delivered; results in `mms/` |
| `scripts/spectral/` | pseudo-spectral cross-check of the FV velocity (C++ comparator `apps/sf42_periodic_flow/`): `sf18.py`, `darcy.py`, `compare.py`, `run_spectral.sh` | delivered; results in `spectral/` |
| `scripts/seedlattice/` | seed-lattice rule (RR12 predictor, `rule.py`), probe check against the production `find_seed_on_face` (`verify_probe.py`, ctest `seedlattice_probe`), SF-43 admissible-`M` plan (`plan.py`, `data/`) | delivered |
| `scripts/sf42red/` | reader and per-realization plane reduction for both codes, sha256 manifest and pruning tool | delivered |
| `scripts/recompute/`, `scripts/reconcile/` | independent recomputation of SF-39 reduced values from raw moment files; reconciliation of solver verdicts against job logs | delivered |
| `verification/` | R1 evidence of acceptance A, B, C: job scripts (`job_base.sh`, `job_head.sh`, `bytes.sh`, `identity.sh`, `identity.py`, `treecmp.py`), logs, comparison outputs, md5 lists, `jobs.tsv` (job table with GPU hours) | delivered |
| `mms/` | R2 results of the MMS order study (acceptance D): 24 JSON, `mms_summary.{md,json}`, job logs | delivered |
| `spectral/` | R2 results of the spectral cross-check (acceptance E): `summary.{md,json}`, `cases/`, FV `report.json`, `fv_npy.sha256` (the 4.7 GB FV dumps stay on the host), job logs | delivered |
| `scripts/pilots/` | design-pilot tooling (R4): run tables (`design.py`), host runner with own-PID GPU memory sampling (`run.py`, `job.sh`), reduction and analysis (`analyze.py`, `report.py`, `cost.py`, `tables.py`, `hostref.py`, `presubmit.py`, `jobs.py`); `heldout.py` (R6: cost model predicted against measured on the J1B points, no refit); `python3 -m pilots.report j1|j1b|j2|j3|j3b|cost`, `python3 -m pilots.tables`, `python3 -m pilots.heldout` | delivered |
| `pilots/` | R4 / R4b results of the design pilots (acceptance G): per-run `RUN.json`, `cmd.txt`, trimmed logs, Par2 configs, reduced per-realization tables with sha256 input manifests, `hostref.json`, `j1_grid.*`, `j2_*`, `j3_pairing.json`, `j3b_pairing.json`, `cost_projection*`, `presubmit_projection.json`; R6 additions: `J1B/` (the three grid pilots missing from J1: `RUN.json`, `cmd.txt`, trimmed `run.log.trim` per run, `per_particle_sha256.txt` of the 51 `.npy` files that stay on the host), `j1b_grid.{csv,json}` (from `pilots.report j1b`), `j1b_heldout.json` (from `pilots.heldout`); no per-particle arrays | delivered |
| `coverage/` | R5 / R5b result of the coverage simulation (acceptance F; job `sf42-coverage`, `full` profile, 1134 cells): `plan.json`, `summary.json` and `summary.md` (from the corrected `summarize`, qualification matrix, failures, false-fail probabilities), the host `env.txt` (unchanged, says `cpus 1`: Deviation 10), `chunks.sha256` (sha256 of the 35844 chunk files; the chunks stay on the host in `~/sf42_campaign/coverage/chunks`; the sha256 of the list is `f7ae476d9858695bf2d7b83e2da3c9677792f04e89de7828c6317b7ddda35793`) | delivered |
| `verification/jobs_r5.tsv`, `verification/logs/sf42-coverage.log`, `verification/job_r5.sh` | job row (41.693 GPU-h of slot time, CPU only), trimmed job record, and the wrapper the job ran; `verification/indep_boott_unpaired.{py,txt}`: independent check (no `mcstat` import) of the bootstrap-t over-miss of the unpaired contrast bound | delivered |
| `verification/job_r6_{build,head,hosttests}.sh`, `verification/jobs_r6.tsv`, `verification/logs/sf42-r6-*.log`, `verification/r6/{regress,bytes,identity}/` | R6 re-verification of the corrected tree `08079af16e2c` (acceptance A, B, C, Gate 1): wrappers of the clean build, of the regress run against the base reference (heads directory `SF-42-r6-<rev>`, copies under `~/sf42_campaign/r6/regress`) and of the host-stack test suite; job table with a `resource` column (`CUDA` / `CPU-only`; `slot_h` = wall hours holding the slot); trimmed job logs (the `sf42-r6-hosttests` log is the failed first run, with the excerpt of its `-v` log); the R6 copies of the regress, byte and identity outputs (the file sets of R1's `regress/`, `bytes/`, `identity/`, which stay as history) | delivered |
| `repro/` | result directory of the reproducibility tooling jobs (acceptance H): not created; the H evidence is `verification/recompute_r3_report.txt`, `verification/reconcile_r3_report.txt` and the host logs | not used |

## Unit tests

```bash
cd docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts && python3 -W error -m unittest discover -s . -t .
```

`-W error`: any DeprecationWarning fails. The code uses stdlib, numpy and scipy only and runs unchanged on the host stack (Python
3.11.7, numpy 1.26.4, scipy 1.11.4) and on the current local stack (Python 3.13, numpy 2.5, scipy 1.18). One test is skipped without
host data (`test_s025_gate5_from_per_realization_values`: set `MCSTAT_PAR2_ALPHAX_S025` to a text file with the 25 per-realization
Par2 `alpha_x` values of `s025`).

Contract tests of `scripts/remote` (local, seconds): `bash scripts/tests/remote_blinded_test.sh`,
`bash scripts/tests/remote_gpu_select_test.sh`.

On the host the same command is run by `verification/job_r6_hosttests.sh` with `MCSTAT_PAR2_ALPHAX_S025=$HOME/sf42_campaign/r3/par2_alphax_s025.txt`
set, so that the host-only test runs instead of skipping.

## Host layout (`$HOME/sf42_campaign`, outside the mirror `~/MacroFlow3D-SF-42`, which `sync --delete` would erase)

| path | content |
|---|---|
| `base_bin/` | `macroflow3d_pipeline`, `lagrangian_ensemble` built from the increment base `3164e4fb`, with `md5.txt` (extra byte checks) |
| `regress/` | `scripts/regress` comparison of head against base: `compare.txt`, both `binaries.md5`, exit codes, `env.txt` |
| `bytes/`, `identity/` | outputs of the jobs `sf42-bytes` and `sf42-identity` |
| `pilots/` | per-particle outputs of the design pilots (`J1`, `J2`, `J3`, `J3B`, and `J1B` of R6; reduced tables, `hostref.json`); sha256 in `R/pilots/J1/per_particle_sha256.txt`, `R/pilots/J1B/per_particle_sha256.txt` and the reduced-table manifests |
| `coverage/` | the full output of the coverage simulation (`plan.json`, `chunks/` 35844 files, `summary.*`, `env.txt`, `run.log`, `smoke/`); `R/coverage/` holds all but the chunks (`chunks.sha256`), `run.log` and `smoke/` |
| `r5_hosttests.log` | the host unit-test log of the coverage job (`Ran 309 tests in 36.038s`, `OK`) |
| `r3/` | R3 outputs (`par2_alphax_s025.txt`, recomputation and reconciliation reports, `noise/`); `r3/.start` is the marker of the `find ~/sf39_campaign -newer` checks |
| `r6/` | R6 outputs: `regress/`, `bytes/`, `identity/` (the R6 runs), `hosttests.log` (the `-v` log of the host suite); the J1B runs are under `pilots/J1B` |

`$HOME/regress/refs/3164e4f/out` (base reference) and `$HOME/regress/heads/SF-42-<tree>/out` follow the `scripts/regress/README.md`
protocol. `$HOME/sf39_campaign` is not touched by SF-42.
