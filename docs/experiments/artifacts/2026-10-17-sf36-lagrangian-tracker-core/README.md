# SF-36 artifact: Lagrangian-label tracker core on the cube production stage

Evidence of increment SF-36 (forward DP5(4) tracker with per-particle labels, covector and divergence transport, and
moments per plane and per time; `inlet_slab --track forward`). This README says what is where and how to regenerate
it. It is not a result summary: results live in `analysis/` and, once written, in the experiment note
`docs/experiments/2026-10-17-sf36-lagrangian-tracker-core.md`.

- Increment spec: `docs/plans/active/lester-eq14/increments/SF-36-lagrangian-tracker-core.md` (thresholds A-J, owner
  amendment of E, A2 (`I`) and D, orchestrator reading rules).
- Code (not in this directory): `src/physics/particles/lagrangian_labels/`, `apps/inlet_slab/` (`--track forward`),
  tests `tests/lagrangian_labels/` (`lagrangian_controls_tests`, `lagrangian_gpu_host_tests`).
- Pattern: the SF-35 artifact `docs/experiments/artifacts/2026-10-07-sf35-label-transport/`.

## Provenance (filled by the campaign node, N8)

| item | value |
|---|---|
| head commit synced to the SF-36 mirror | `<TO FILL: git rev-parse HEAD of the clean worktree, also written to COMMIT>` |
| base commit of the references | `040e3ce` (SF-35 closure; job `sf36-base-refs`, `~/sf36_base_refs/` on the host) |
| V100 build | `build/v100-release` (`cmake --preset v100-release`) |
| jobs / dates | `<TO FILL: job names, start/end UTC, exit codes>` |

The synced mirror has no `.git`; the campaign node writes the head hash to `COMMIT` in this directory before
`scripts/remote sync` (absent -> `unknown` in `run.env`).

## Layout

| path | content | committed |
|---|---|---|
| `scripts/run_sf36.sh` | campaign runner, one part per detached job (`controls`, `bytecmp`, `mem`, `cost`, `stress`, `all`) | yes |
| `scripts/digest_sf36.py` | log parser -> markdown tables (python3 standard library) | yes |
| `scripts/lines_cmp.py` | selected-line comparison with timing fields masked (criterion F) | yes |
| `logs/<part>/` | copy of the host directory `~/sf36_campaign/<part>/`: `<name>.log`, `<name>.time` (`/usr/bin/time -v`), `<name>.mem` (nvidia-smi 200 ms sampler), `<name>.linecmp`, `results.txt` (BYTECMP / HEAD_PEAK_MIB / COST / STRESS_RESULT lines), `exit_codes.tsv`, `run.env`; remote job logs in `logs/jobs/<job>.log` | yes (text; not the saved `labels/` directories) |
| `raw/<part>/` | driver `--summary` JSON per run (copy of `~/sf36_campaign/<part>/raw` or `<part>/*.json`) | yes (text) |
| `analysis/` | digests: `controls.md`, `bytecmp.md`, `mem.md`, `cost.md`, `stress.md` | yes |
| `COMMIT` | head hash of the synced tree | N8 decides |

Host output root: `${SF36_OUT_ROOT:-$HOME/sf36_campaign}/<part>/`; base references: `${SF36_BASE_REFS:-$HOME/sf36_base_refs}`
(`trace/N<N>_e<eps>_<host|gpu>.{log,time,json}` + label directories, `trace/prod32.log`, `mem/N256_e2.mem`).

## Regeneration

From the repository root of the synced mirror; `A=docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core`.
Each line is one detached V100 job; serialize them (shared mirror and GPU).

```bash
scripts/remote --increment SF-36 run sf36-controls -- "bash $A/scripts/run_sf36.sh build/v100-release controls"
scripts/remote --increment SF-36 wait sf36-controls
scripts/remote --increment SF-36 run sf36-bytecmp -- "bash $A/scripts/run_sf36.sh build/v100-release bytecmp"
scripts/remote --increment SF-36 wait sf36-bytecmp
scripts/remote --increment SF-36 run sf36-mem -- "bash $A/scripts/run_sf36.sh build/v100-release mem"
scripts/remote --increment SF-36 wait sf36-mem
scripts/remote --increment SF-36 run sf36-cost -- "bash $A/scripts/run_sf36.sh build/v100-release cost"
scripts/remote --increment SF-36 wait sf36-cost
scripts/remote --increment SF-36 run sf36-stress -- "bash $A/scripts/run_sf36.sh build/v100-release stress"
scripts/remote --increment SF-36 wait sf36-stress
```

(`all` runs the five in that order as a single job.) Every part ends by running the digest non-fatally into
`$SF36_OUT_ROOT/analysis`. To regenerate the digests from copied logs:

```bash
python3 $A/scripts/digest_sf36.py $A/logs $A/analysis     # logs/<part>/ layout as above
```

## Parts and what they record

| part | criteria | what | blocking exit |
|---|---|---|---|
| `controls` | A, B, C, D, E1/E2, I (64^3) | `lagrangian_controls_tests --full`, `--diag`; `lagrangian_gpu_host_tests --full`; logs, exit codes, `SUMMARY:` lines | any test binary exit != 0 |
| `bytecmp` | F | head `inlet_slab` against the base: `--construct trace` 32^3/64^3 x eps {0.5, 1.0} x {host, gpu}, plus gpu eps 2 at 64^3/128^3: `cmp` of `psi1.npy`/`psi2.npy`, CASE/PLANE/BACKFLOW/GROWTH/RESIDUAL/TRANSPORT_STATUS/STATUS lines (timing fields masked), exit codes; `--production` 32^3 eps 0.25 CASE/ORACLE/ORACLE_SUMMARY/EXTRA lines against the base log and the SF-33 `logs/ladder_0.25_c4/N32.log` | any `DIFF` or exit-code mismatch, missing base references |
| `mem` | G | nvidia-smi 200 ms sampler around the head `--construct trace --n 256 --eps 2 --tracer gpu`; `HEAD_PEAK_MIB`, `BASE_PEAK_MIB` (max of the base `.mem`; 6239 recorded), probe 3769 +-2 % | head > base or outside the 2 % band |
| `cost` | J (no threshold) | `--track forward` at N 64/128/256 x eps 1, 2 x {plain, `--covectors`}, `M = 256`, planes and times 0.25,0.5,1,2,4: wall time, tracker timings, nfev per pass (TSTATUS), device peak (smi and driver), host max RSS | none (exit codes other than 0/18/22 are flagged) |
| `stress` | N5 finding (informative) | `--track forward --n 16 --ell 0.15 --seed 2 --eps 4 --sigma2 1 --np-side 16 --planes 0.5,1 --times 0.5,1`, default tol/`h_max`/cap, `timeout 1800`; exit code 124 = did not terminate | none |

The runner never retries and never changes settings after a failure; every exit code is in `exit_codes.tsv`.

## Local dry run (not evidence)

`SF36_DRYRUN=1` shrinks every grid to 16^3, `--np-side` to 8, the stress timeout to 60 s, runs the test binaries without
`--full`, and skips the comparisons against the base with `SKIP` when `SF36_BASE_REFS` is absent (always for the
32^3 `--production` comparison). Use `SF36_OUT_ROOT=<scratch>` to keep dry-run logs out of the home directory and out
of the repository. Unset, the switch changes nothing.

## Host notes

The V100 host runs bash 4.2: no empty-array expansion under `set -u` (the scripts use none, or the
`${a[@]+"${a[@]}"}` form) and no backslashes in `${var//pat/rep}` replacements (none are used).
`scripts/remote run` exports `CUDA_VISIBLE_DEVICES=<physical gpu>`; the sampler queries that index.
