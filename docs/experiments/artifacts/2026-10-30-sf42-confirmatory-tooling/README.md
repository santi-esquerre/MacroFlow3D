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
| `mms/`, `spectral/`, `coverage/`, `pilots/`, `repro/` | result directories of the later jobs (acceptance D, E, F, G, H) | pending (later nodes) |

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

## Host layout (`$HOME/sf42_campaign`, outside the mirror `~/MacroFlow3D-SF-42`, which `sync --delete` would erase)

| path | content |
|---|---|
| `base_bin/` | `macroflow3d_pipeline`, `lagrangian_ensemble` built from the increment base `3164e4fb`, with `md5.txt` (extra byte checks) |
| `regress/` | `scripts/regress` comparison of head against base: `compare.txt`, both `binaries.md5`, exit codes, `env.txt` |
| `bytes/`, `identity/` | outputs of the jobs `sf42-bytes` and `sf42-identity` |

`$HOME/regress/refs/3164e4f/out` (base reference) and `$HOME/regress/heads/SF-42-<tree>/out` follow the `scripts/regress/README.md`
protocol. `$HOME/sf39_campaign` is not touched by SF-42.
