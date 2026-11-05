# sf43add.camp: campaign side of SF-43 addendum 1

Plan generator, S0 report and seed registry of the exploratory arms J1, J3, J9 (and the conditional J8) of
`../../../preregistration-addendum-1.md`. Stdlib only (the frozen `sf43camp` / `sf43plan` modules are imported, never edited), Python 3.11
compatible, runs on the host. The analysis is the sibling package `sf43add.ana` (node A1-ANA); this package only calls it from the plan.

`A` = `docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion`. All commands run from the repository root with
`PYTHONPATH=A/scripts:docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts` (as `run_sf43.sh` sets it).

| command | reads | writes |
|---|---|---|
| `python3 -m sf43add.camp plan [--job production\|j8] [--costs C] [--budget-gpu-h X] [--disk-min-free-gb X] [--drop ARM\|J1\|J3\|J8\|J9 ...] [--allow-missing] [--out F]` | frozen `A/plan.json`, `A/prereg_addendum.json`, `A/binaries.json`, the files to hash, optionally `costs.json` | `A/plan_addendum.json` (production + `smoke` section) or `A/plan_addendum_j8.json` (`--out -` prints) |
| `python3 -m sf43add.camp finalize ...` | as `plan`, but `--costs`, `--budget-gpu-h`, `--disk-min-free-gb` are all required | the same files (deterministic) |
| `python3 -m sf43add.camp s0-report --campaign D [--out costs.json] [--no-du]` | ONLY `D/ledger.jsonl` (`unit_end`, `seed`, `hb`, and `reduced` records: that a segment was reduced, never the table), `D/REPORT.txt` (status, `gpu_h_used`), `D/segments/*/*/run.log` (last lines of a failed unit), the presence of `config.yaml` (Par2 vs consumer) and the size (allocated blocks, like `du`) of each realization's raw output | `D/costs.json`; prints the table and the cut rules. Never opens `reduced/`, `analysis/`, a table or an alpha |
| `python3 -m sf43add.camp check-seeds [--verify]` | frozen `A/seeds_registry.json`, `A/prereg_addendum.json`, `A/plan.json` | `A/seeds_registry_addendum.json` (`--verify`: compare with the committed one instead) |

## What `plan` builds (preregistration-addendum-1 sections 1, 4, 5, 6)

* Each arm = deep copy of its ORIGIN arm in the frozen plan (`par2` / `consumer` dict AND `reduce` block; `j9_w48`'s origin is the derived `j3_l5`,
  `j8_*` from `j1_*`) with ONLY the `factor` of `prereg_addendum.json` applied. `check_arm` recomputes the difference of the two runner dicts
  (canonical JSON, so `1` vs `1.0` counts) and fails unless it equals the factor exactly; a `reduce` block, `family` or `amplitude` that differs also
  fails. `n_conf = n_max = seeds.count`, `unit_size` 10, `primary false`, `no_reestimation true`, `heavy false`, `retained` as in the prereg, one CRN group
  per arm with the prereg's reserves, priorities 0..6 in the owner's order (`j1_base10 -> j1_H3 -> j1_D -> j1_DH3 -> j1_H2 -> j3_l5 -> j9_w48`).
* Top level: `cutoff_utc` 2026-11-07T02:30:00Z + `grace_minutes` 30 (asserted <= 03:00Z), `stage1_max_hours` 0 (one GPU: lane A only), `budget_gpu_h` 7.2 and
  `disk_min_free_gb` 100 until finalization, `min_retained` 2, the frozen `failure` policy, default executor, `binaries` = `A/binaries.json`, no
  `blinded_reestimation`, no `background`, no `lane_b_min_free_mib`. `recompute` / `reconcile` are the frozen tools' commands without
  `--primary-arm` (no primary arm) and without `--single-pass-hmax` (no `hmax_div_first` arm: single-pass roots carry the label -1).
* `analysis.command` is the A1-ANA contract verbatim (`run` for production, `run ... --j8-campaign {campaign_dir}` for J8, `check` for the smoke section).
* `analysis.sha256_list` = the frozen plan's own list (re-verified against its recorded hashes: a changed frozen file is an error) + `sf43add/**/*.py`
  without tests + `sf43camp/*.py` + `prereg_addendum.json`, hashed from the files ON DISK. The list is NOT derived from the imports of `sf43add.ana`: the
  integrator must regenerate the plan after both packages are integrated and check that the frozen packages the analysis imports are all in the list.
  Until then `plan` fails if `sf43add/ana/__init__.py` or `__main__.py` is absent; `--allow-missing` skips what is absent and records it in
  `_provenance.missing_allowed` (the committed plan of this node was generated that way: its hashes cover a tree without `ana/`).
* `smoke` (S0): campaign `/home/sesquerre/sf43_addendum_smoke`, `cutoff_minutes` 240, `budget_gpu_h` 1.0, arms `j1_base10` (1 realization), `j1_H3` (2), `j1_DH3` (2),
  `j3_l5` (1), `j9_w48` (1) with the same runner dicts, every seed retained so nothing is pruned, one unit per arm. It never takes `--costs` (S0 runs
  before any measurement; its record does not change when the plan is finalized).
* Provisional `est_seconds` / `raw_gb` / `min_free_mib` (from the frozen root arm; an upper bound): Par2 scales with the number of steps (horizon x3 / x2, `dt_half` x2;
  `s4_g5p` 17.5 s, 0.065 GB, 10000 MiB), consumer time with the cells (J3 = 1/8 and J9 = 1/2 of `s1_dist`: 54 s, 9000 MiB), consumer `raw_gb` unchanged (per-particle
  arrays do not depend on the grid), `min_free_mib` the root's. With `--costs`: `est_seconds` = 1.25 x measured, `raw_gb` measured, `min_free_mib` =
  max(1.25 x measured peak, the frozen origin's); an arm that was not in S0 (`j1_D`, `j1_H2`) takes the least-squares line of the measured Par2 arms against the step
  multiplier (proportional bound `max(y/m) m` if the line is not increasing from a non-negative intercept); `j8_*` use the measured `j1_*` of the same factor.
  `costs["drop"]` and `--drop` remove arms (and their groups).
* `--binaries PATH` (plan/finalize; default `A/binaries.json`): JSON read for ONLY the md5 of `lagrangian_ensemble` and `macroflow3d_pipeline` (32 lowercase hex digits or the command fails; keys starting with `_`, e.g. `_provenance`, and any other key are ignored); `_provenance.binaries_path` / `binaries_sha256` record the file used (e.g. `--binaries A/binaries_addendum.json` when the build embeds another git revision).
* The plan carries a `_provenance` block (sha256 of the frozen plan, prereg, binaries and costs files, dropped arms, missing files allowed); no timestamps, so two runs are byte-identical.

## S0 cut rules in `s0-report` (preregistration section 4)

`j9_w48` dropped if seconds per realization > 180, peak > 20480 MiB, CG iterations > 1500, or the runner rejects the geometry (`config_error`); anything else wrong with it
(OOM, other error) is a NOTE for the orchestrator, not an automatic drop. All `j1_*` dropped if any of `j1_base10`, `j1_H3`, `j1_DH3` has a structural failure (a unit class
other than `ok`, a seed record that is not `ok` / `converged|fallback`, fewer completed than planned, a completed segment that was never reduced, or the arm absent); `j3_l5` dropped likewise. The J1 reproduction pre-check
(`sf43add.ana repro`, seed 434001) is separate: if it fails run `plan --drop J1`. `costs.json` also carries `gpu_h_held` (REPORT.txt, else the last `hb`) and
`suggested_budget_gpu_h` = 8 - held - 0.5.

## Order of use (orchestrator)

```bash
python3 -m sf43add.camp check-seeds                      # registry + the frozen check_seeds; commit the file
python3 -m sf43add.camp plan                             # plan_addendum.json (after A1-ANA is integrated: no --allow-missing)
python3 -m sf43add.camp plan --job j8                    # plan_addendum_j8.json
REMOTE_GPU_COUNT_GRANTED=1 bash A/scripts/run_sf43.sh build/v100-release all --plan A/plan_addendum.json --dry-run [--smoke]
# ... S0 on the host (--smoke), then:
python3 -m sf43add.camp s0-report --campaign ~/sf43_addendum_smoke
python3 -m sf43add.camp finalize --costs ~/sf43_addendum_smoke/costs.json --budget-gpu-h <8 - S0 - 0.5> --disk-min-free-gb <host free GB - 60> [--drop J1]
```

`run_sf43.sh --dry-run` needs no build directory (the md5 and sha256 gates are skipped); it simulates with a virtual clock and `REMOTE_GPU_COUNT_GRANTED`
defaults to 2 (lane B only takes `heavy` units: none here, so the schedule is the same).

## Tests (no GPU, no data; from `A/scripts`)

```bash
python3 -W error -m unittest discover -s sf43add/camp/tests -t .     # ~12 s
```

`test_e2e.py` runs the real generated plan (production, smoke, J8; sizes scaled, factors not) through the real engine and `RunnerExecutor` with the fake runner binaries of
`sf43camp/tests_executor` (imported, not modified), then `recompute43` / `reconcile43`, `s0-report` and `finalize`. The fake Par2 runner writes no time series, so `U_time` /
`par2_exit_x` are dropped there; the fake consumer writes no moment CSVs, so the hook test keeps every realization (`min_retained` 4) and pruning is tested in a run without the
`recompute` hook.
