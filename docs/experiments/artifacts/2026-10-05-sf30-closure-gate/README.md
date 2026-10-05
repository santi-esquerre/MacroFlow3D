# SF-30 streamline-closure gate — run matrix, analysis scripts and raw outputs

Artifacts of increment SF-30 (`docs/plans/active/lester-eq14/increments/SF-30-streamline-closure-gate.md`).

What this directory is for: launching the pre-registered run matrix of the executable
`closure_gate` (the return map of the face `x1 = 0` under the Darcy flow of a triply periodic
conductivity field), and turning its raw outputs into tables and the classification of the
decision rule. It is not production code, not a test, and nothing here is built or run by
CMake/ctest.

What it is not for: changing the matrix or the decision rule. Both were fixed and versioned in
the SF-30 bitácora (rows of 2026-10-05: D-1..D-12; the rule is D-5 as refined by D-11 and D-12)
BEFORE any matrix run. `run_matrix.sh` holds the exact run list; `analyze.py` holds the four
thresholds in one block marked "pre-registered ... do not edit".

## Layout

Experiment note: `docs/experiments/2026-10-05-sf30-streamline-closure-gate.md`.

| path | content |
|---|---|
| `scripts/run_matrix.sh` | launcher: the 103 pre-registered runs in five groups |
| `scripts/analyze.py` | analysis and classification under the pre-registered rule (Python 3 standard library only) |
| `scripts/followup_checks.py` | follow-up checks cited by the experiment note, tables F1-F4 (Python 3 standard library only; not part of the decision rule) |
| `scripts/probe_seeds16.csv` | the 16 seed points `(y0, z0)` of the 2026-10-02 probes (numpy `default_rng(3)`, `y0 = random(16)`, `z0 = random(16)`) |
| `scripts/fixtures/sample_summary_gaussian64.json` | a real `summary.json` written by `closure_gate` at commit `f9ca080` (local debug build) for `--field gaussian --sigma2 1 --ell 0.125 --seed 3001 --n 64` with the executable's default three-tolerance ladder (1e-6, 1e-8, 1e-10) |
| | used only as a schema fixture by `analyze.py --self-test`; not part of the experiment's raw data (under the pre-registered four-tolerance ladder that run is invalid for classification, which is what self-test check 12 reports) |
| `raw/<group>/<run-id>/` | the 103 pre-registered runs of job `sf30-matrix` (groups `controls`, `matched`, `matrix128`, `ladder`, `manyperiod`): `summary.json` (schema `sf30-closure-gate-1`), `streamlines.csv.gz` (per-seed return points at the working tolerance, gzip-compressed) and `timing.json` |
| `raw/<group>/status.tsv` | one line per run: `<run-id> <exit code> <wall seconds>` |
| `raw_followup/sensitivity/<run-id>/` | the 18 exploratory runs of job `sf30-post` (NOT pre-registered): `gaussian` seed 3001, `(sigma2, ell)` in {(0.25, 0.0625), (1, 0.0625), (4, 0.0625)} at 128^3 with 2, 4, 8, 16, 32 periods, and `(4, 0.0625)` at 256^3 with 2, 4, 8 periods; `summary.json` and `timing.json` only |
| `analysis/tables.md` | the twelve tables of `analyze.py` |
| `analysis/classification.json` | the per-run quantities and the per-case classification of `analyze.py` |
| `analysis/followup_tables.md` | tables F1-F4 of `followup_checks.py` |
| `logs/sf30-q1-controls.log` | instrument qualification job (30 analytic-control runs on the N2 candidate `f9ca080`) |
| `logs/sf30-matrix.log` | the matrix job (103 runs) |
| `logs/sf30-post.log` | the follow-up job (smoke + 18 exploratory runs) |
| `logs/sf30-smoke.log` | output of the `config_pspta_small` smoke run of job `sf30-post` |

## Run matrix

| group | runs | content |
|---|---|---|
| `controls` | 30 | analytic `lester2021`, `lester_brk`, `control2d` (eps 1), `two_mode`, `generic3d` (eps 0.5) at N = 64, 128, 256; each with the 16 probe seeds (`_p16`, working tol 1e-10) and with 1024 seeds (working tol 1e-8); `--pcg-rtol 1e-12` |
| `matched` | 22 | `gaussian2d` (matched 2-D control), six cases x seed 3001 x N = 64, 128, 256; plus `(4, 0.0625)` seeds 3002, 3003 at N = 128, 256 |
| `matrix128` | 30 | `gaussian`, six cases x seeds 3001..3005 at N = 128 |
| `ladder` | 14 | `gaussian`, six cases x seed 3001 at N = 64, 256; plus `(4, 0.0625)` seeds 3002, 3003 at N = 256 |
| `manyperiod` | 7 | `gaussian`, six cases x seed 3001 at N = 128 with 64 periods; plus `(4, 0.0625)` seed 3001 at N = 256 with 16 periods |

Cases are `(sigma2, ell)` with `sigma2` in {0.25, 1, 4} and `ell` in {0.125, 0.0625}.
All Gaussian runs use `--tols 1e-6,1e-8,1e-10,1e-12 --working-tol 1e-8`, 1024 seeds and the
default PCG tolerance. Run ids: `a_<field>_n<N>[_p16]`, `m_<s>_<l>_r<seed>_n<N>`,
`g_<s>_<l>_r<seed>_n<N>`, `p_<s>_<l>_r<seed>_n<N>_p<periods>`, with labels `s025`, `s1`, `s4`
for `sigma2` and `l8`, `l16` for `ell`. Print the full list without executing anything:

```bash
bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh all --list
```

## How to run (detached V100 jobs, from the repository root)

Every group is long-duration computation: run it only as a detached job on the SF-30 mirror
(`docs/runbooks/remote-v100.md`), never locally and never through `scripts/remote exec`.
Check `scripts/remote --increment SF-30 status <job>` for any job still running before a sync.

```bash
scripts/remote --increment SF-30 sync
scripts/remote --increment SF-30 exec -- "cmake --preset v100-release && cmake --build build/v100-release -j --target closure_gate"
scripts/remote --increment SF-30 run sf30-controls -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh controls"
scripts/remote --increment SF-30 wait sf30-controls
scripts/remote --increment SF-30 run sf30-matched -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh matched"
scripts/remote --increment SF-30 wait sf30-matched
scripts/remote --increment SF-30 run sf30-matrix128 -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh matrix128"
scripts/remote --increment SF-30 wait sf30-matrix128
scripts/remote --increment SF-30 run sf30-ladder -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh ladder"
scripts/remote --increment SF-30 wait sf30-ladder
scripts/remote --increment SF-30 run sf30-manyperiod -- "bash docs/experiments/artifacts/2026-10-05-sf30-closure-gate/scripts/run_matrix.sh manyperiod"
scripts/remote --increment SF-30 wait sf30-manyperiod
```

Launcher behaviour:

- options: `--bin <dir>` (default `build/v100-release`), `--out <dir>` (default `output_sf30`),
  `--force` (re-run runs whose `summary.json` exists; default is to skip them, i.e. resume),
  `--list` (print `<group> <run-id> <command>` and execute nothing);
- each run writes `<out>/<group>/<run-id>/{summary.json,streamlines.csv,timing.json}`
  (`streamlines.csv` is absent when the Darcy solve did not converge);
- a failing run never stops the group; every executed run appends
  a tab-separated line `<run-id> <exit code> <wall seconds>` to `<out>/<group>/status.tsv`;
- exit 0 iff every run of the group exited 0 (a skipped run counts with its last recorded exit
  code), 1 otherwise, 2 on a usage error;
- no GPU or thread option is passed: the job environment selects the device and the
  executable's default thread count applies.

## How to analyse

From the repository root (the committed `analysis/tables.md` was produced with exactly this
`raw` path; the path appears in the first line of the tables, so another invocation path
changes that line only):

```bash
A=docs/experiments/artifacts/2026-10-05-sf30-closure-gate
python3 $A/scripts/analyze.py --self-test
python3 $A/scripts/analyze.py $A/raw --tables $A/analysis/tables.md --json $A/analysis/classification.json
python3 $A/scripts/followup_checks.py $A --out $A/analysis/followup_tables.md
```

`followup_checks.py` output does not depend on the invocation path. Its tables: F1 recomputes
the period-1 `R` of every run from its `streamlines.csv.gz` and compares it with the summary;
F2 gives the pointwise grid convergence of the return map over the paired seeds; F3 the
distance between the 1e-8 and 1e-12 return points versus the number of periods (many-period
and follow-up runs); F4 the many-period `R`, `var(d2)`, `var(d3)` at the four tolerances.

`analyze.py` reads every `summary.json` under the given directory (recursively) and identifies
each run from the file's content (field, `sigma2`, `ell`, `seed`, `eps`, `n`, periods, seed set),
never from its directory name. It prints, in order: (1) analytic controls vs the spectral
probes, (2) analytic controls with 1024 seeds, (3) the matrix, (4) the matched 2-D controls,
(5) grid ladders, (6) the classification under the pre-registered rule, (7) the spec-literal
reading (information only, not the rule), (8) amplitude scaling, (9) the re-injection estimate,
(10) the many-period iteration, (11) the tolerance ladder, (12) the run inventory (missing runs,
non-converged Darcy solves, configuration deviations, runs invalid for classification).

Classes: `does_not_close`, `closes`, `ambiguous`, `incomplete`. A required run that is absent,
or present but lacking a quantity the rule needs (no 1e-8 entry, no `R`, tightest tolerance not
1e-12), gives `incomplete`; a failed validity criterion gives `ambiguous`.

`--self-test` builds synthetic `summary.json` trees and checks twelve scenarios of the rule, the
parse of a real `closure_gate` output (`R` at the working tolerance), and that the analysis'
expected run list equals `run_matrix.sh all --list` (103 runs, 30/22/30/14/7).

## Provenance of `raw/` and `raw_followup/`

Copied by the orchestrator from the SF-30 V100 mirror after the jobs `sf30-matrix` and
`sf30-post` (output directories `output_sf30/<group>/` and `output_sf30/sensitivity/`),
with each `streamlines.csv` gzip-compressed; nothing in `raw/` or `raw_followup/` is
written by the scripts of this directory. The analysis outputs live in `analysis/`.
