# SF-43 campaign scripts

The toolchain, in the order it runs (each README has the details):

| step | what | where |
|---|---|---|
| generate | design table -> `plan.json`, `prereg.json`, `prereg_smoke.json`, `seeds_registry.json`, `design_table.md` | `sf43plan/README.md` |
| run | one detached job: `run_sf43.sh <build_dir> all [--smoke]` -> the engine, the executor, reduce / prune, `recompute43`, `reconcile43` | `sf43camp/README.md` |
| analyse | `analyse_sf43.sh <campaign_dir> <build_dir> [--smoke]` (the engine's `analysis.command`) -> `<campaign_dir>/analysis/summary.json`, figures | `sf43ana/README.md` |
| coverage | CPU harness of the Gate 5 / sampling / pipeline coverage (the smoke job's background command) | `sf43cov/README.md` |
| addendum 1: campaign | plan generator, S0 report and seed registry of the exploratory arms J1, J3, J9 (+ conditional J8): `python3 -m sf43add.camp plan\|finalize\|s0-report\|check-seeds` -> `plan_addendum.json`, `plan_addendum_j8.json` | `sf43add/camp/README.md` |
| addendum 1: analysis | `python3 -m sf43add.ana run\|check\|repro\|j8-trigger` (the plan's `analysis.command`; blinded stdout, structure only) | `sf43add/ana/README.md` |
| addendum 1: post hoc | frozen-data post hoc analyses N1, N3, N5, N6, N8 and sigma^2 scaling (`posthoc/`, `../posthoc/` outputs; tests `python3 -W error posthoc/test_posthoc.py`) | `posthoc/` |
| addendum 1: acceptance F | independent reduction of the J1 reproduction seeds (`review_f/indep_reduce_f.py`) | `review_f/RUNBOOK.md` |

Qualification gate (H4, H13 against H15): the robust sign is testable only at sizes covered by a qualified coverage profile (`sf43cov` -> `qualification.json` -> `sf43ana/qualification.py`);
the pre-registration names the document (`qualification`, hashed in `analysis.sha256_list`); the smoke run passes its own (or none: every robust sign `p = 1`). In the smoke job the document is the real coverage run's, while the smoke arms have tiny realized sizes: its robust signs are therefore `p = 1`, status `not qualified at the realized sizes` (the smoke checks structure and counts only, so this is expected, and the Gate 5 / descriptive paths still run).

`analyse_sf43.sh`: production = `python3 -W default -m sf43ana run` (warnings are recorded in `summary.json`, never fatal; the unit tests keep `-W error`) with `<A>/prereg.json`, Gate 5 `auto` with the committed coverage summary, then `check-finite` (output printed, exit status ignored so the recompute / reconcile steps still run); `--smoke` =
`<A>/prereg_smoke.json`, the smoke job's own coverage output (or Gate 5 descriptive) and qualification document (or none), analysis stdout / stderr to files in `analysis/`, and ONLY the
output of `check-finite` (structure and counts, `STRUCTURE_OK`, `warnings_count`, `errors_count`) on stdout: no estimand, bound, p-value, decision or class reaches the job log.

Tests (from this directory, no GPU, no data):

```bash
python3 -W error -m unittest discover -s sf43ana/tests -t .            # ~4 min (null simulations dominate)
python3 -W error -m unittest discover -s sf43cov/tests -t .            # ~1 min
python3 -W error -m unittest discover -s sf43plan/tests -t .           # seconds
python3 -W error -m unittest discover -s sf43camp/tests -t .           # ~8 s
python3 -W error -m unittest discover -s sf43camp/tests_executor -t .  # ~20 s
python3 -W error -m unittest discover -s recompute43/tests -t .
python3 -W error -m unittest discover -s reconcile43/tests -t .
python3 -W error -m unittest discover -s tests_integration -t .        # the whole chain, ~2-3 min: generator -> engine -> analysis -> check-finite
```

(Tests added by C-GAPS: `sf42red` suite from `2026-10-30-sf42-confirmatory-tooling/scripts` -- `python3 -W error -m unittest discover -s sf42red/tests -t .` -- with `test_tracker_counts`.)

`tests_integration/test_chain.py`: the real generator (smoke universe, sizes scaled), the real engine and executor with the fake runner binaries, `recompute43`, `reconcile43`,
a scratch copy of `analyse_sf43.sh --smoke`, `check-finite` `STRUCTURE_OK`; with an injected unusable solve (group-wide replacement), an unreduced segment, and a blinded re-estimation.
Also `run_sf43.sh ... all --dry-run` (and `--smoke`) on the plans generated for the real design.
