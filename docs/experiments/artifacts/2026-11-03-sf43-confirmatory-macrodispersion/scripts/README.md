# SF-43 campaign scripts

The toolchain, in the order it runs (each README has the details):

| step | what | where |
|---|---|---|
| generate | design table -> `plan.json`, `prereg.json`, `prereg_smoke.json`, `seeds_registry.json`, `design_table.md` | `sf43plan/README.md` |
| run | one detached job: `run_sf43.sh <build_dir> all [--smoke]` -> the engine, the executor, reduce / prune, `recompute43`, `reconcile43` | `sf43camp/README.md` |
| analyse | `analyse_sf43.sh <campaign_dir> <build_dir> [--smoke]` (the engine's `analysis.command`) -> `<campaign_dir>/analysis/summary.json`, figures | `sf43ana/README.md` |
| coverage | CPU harness of the Gate 5 / sampling / pipeline coverage (the smoke job's background command) | `sf43cov/README.md` |

`analyse_sf43.sh`: production = `python3 -W error -m sf43ana run` with `<A>/prereg.json`, Gate 5 `auto` with the committed coverage summary; `--smoke` =
`<A>/prereg_smoke.json`, the smoke job's own coverage output (or Gate 5 descriptive), analysis stdout / stderr to files in `analysis/`, and ONLY the
output of `check-finite` (structure and counts, `STRUCTURE_OK`) on stdout: no estimand, bound, p-value, decision or class reaches the job log.

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

`tests_integration/test_chain.py`: the real generator (smoke universe, sizes scaled), the real engine and executor with the fake runner binaries, `recompute43`, `reconcile43`,
a scratch copy of `analyse_sf43.sh --smoke`, `check-finite` `STRUCTURE_OK`; with an injected unusable solve (group-wide replacement), an unreduced segment, and a blinded re-estimation.
Also `run_sf43.sh ... all --dry-run` (and `--smoke`) on the plans generated for the real design.
