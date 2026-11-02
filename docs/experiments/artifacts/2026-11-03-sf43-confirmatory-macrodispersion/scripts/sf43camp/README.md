# SF-43 campaign engine

One detached V100 job runs the whole confirmatory study unattended (up to about 5 days). `run_sf43.sh` is a thin wrapper; the logic is the
Python package `sf43camp/` (3.11-compatible, stdlib + scipy for one chi-square quantile, runs under `python3 -W error`; bash 4.2 safe).
Not a pre-registration: the plan (arms, N, seeds) is a separate file fixed by the pre-registration.

```bash
# the job (the helper exports CUDA_VISIBLE_DEVICES=<first>,<second> and REMOTE_GPU_COUNT_GRANTED)
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 \
  scripts/remote --increment SF-43 run sf43-campaign -- "bash <A>/scripts/run_sf43.sh build/v100-release all"
bash run_sf43.sh <build_dir> all [--resume] [--dry-run] [--smoke] [--plan <file>]      # --plan default: <A>/plan.json
cd <A>/scripts && python3 -W error -m unittest discover -s sf43camp/tests -t .          # ~8 s, no GPU
```

The script enters the repository root: `<build_dir>`, `campaign_dir` and every relative path of the plan are relative to it.
`PYTHONPATH` = this `scripts/` dir + the SF-42 scripts dir (`2026-10-30-sf42-confirmatory-tooling/scripts`, for the executor).

## Exit codes (never 70, 75, 97, 130: the helper's)

| code | meaning |
|---|---|
| 0 | DONE: plan complete, no lane crash, analysis ok |
| 10 | DONE_PARTIAL: cutoff reached, or an arm halted, or a lane crashed (analysis ok) |
| 11 | STOPPED: budget or disk guard (analysis ok) |
| 12 | final analysis failed (sha256 mismatch of an analysis file, or a command rc != 0); marker `ANALYSIS_FAILED` |
| 13 | refused to start: md5 mismatch of a binary, invalid plan, stale state (ledger exists without `--resume`, or a different plan), another launcher holds `campaign_dir/LOCK`, usage |
| 143 | SIGTERM/SIGINT: units terminated and recorded, no analysis (a cancel); `--resume` continues |

`campaign_dir` receives: `ledger.jsonl`, `LOCK`, `segments/<arm>/u<unit>_s<seed0>/` (one new directory per run of a unit), `reduced/<arm>/<segment>/`,
`analysis.log` `recompute.log` `reconcile.log`, `REPORT.txt` (per arm: planned / completed / replaced / failed / OOM retries / pending; stage times; GPU-h;
events) and one marker: `DONE`, `DONE_PARTIAL`, `STOPPED` or `ANALYSIS_FAILED`. The analysis (`analysis.command`, then `recompute`, then `reconcile`)
ALWAYS runs after the loop ends, except after a refused start or a signal.

## Queue, lanes, stages

* A **unit** = up to `unit_size` consecutive seeds of one arm inside one aligned seed block (`--seed0 s --nr k`; Par2 likewise with NR). Units are formed
  on demand from the arm's pending seeds, so a resumed or partly failed unit restarts at its first incomplete seed.
* Order = (priority class, progress fraction of the CRN group, group order, anchor seed, replacement flag, arm order); fraction = (first seed - group start)
  / (largest `n_target` in the group). All arms of a CRN group (same `seeds.start`, same `priority`; enforced) run the same seed block back to back, groups of a
  priority class advance together, and arms with fewer seeds simply have no later blocks.
* **Lane A** = first granted id: heavy units first (within the priority class the arm with the largest `min_free_mib`), then light units. **Lane B** = second
  id: heavy units only, in queue order. Before each claim a lane reads the free memory of its own GPU (`nvidia-smi --query-gpu=memory.total,memory.used -i <id>`,
  injectable; unreadable = enough) and claims only units whose arm has `min_free_mib` <= free; if none fits it retries after 120 s.
* **Stage 1** (two GPUs): lane B leaves (`LANE_B_RELEASED gpu=<id> utc=<..>`) when no heavy unit is left to claim, or when the next heavy unit would end after
  `stage1_start + stage1_max_hours` (never exceeded, nothing is killed). `stage1_start` is written once and reused on resume. After the release, or on a
  resume past the cap, only lane A runs (`RESUME_SINGLE_LANE`: relaunch with one GPU). One GPU granted: lane A only.
* Job log: one `SF43_UNIT lane= gpu= arm= seed0= nr= segment= utc=` and one `SF43_UNIT_END ... cls= completed= seconds=` line per unit, plus `STOP`, `OOM`,
  `REPLACEMENT`, `ARM_HALT`, `REESTIMATE`, `PROJECTION` (hourly, vs budget and cutoff), `DISK_GUARD`, `LANE_CRASH`, `FINAL` lines. Runner output goes to the unit's own
  `runner.log`.
* `--dry-run`: simulates the whole plan on a virtual clock with `est_seconds` (no process, no file): the ordered schedule (`DRYRUN SF43_UNIT ...`) and
  `DRYRUN_SUMMARY` (stage-1 / stage-2 hours, GPU-h, end vs cutoff). The md5 gate is skipped. `--smoke`: the plan's `smoke` section overlays the plan
  (its own `arms`, `groups`, `campaign_dir`, `budget_gpu_h`, ...; `cutoff_minutes` = cutoff relative to the start).

## Guards and rules

* **Budget**: GPU-h used = sum of (GPUs held x wall) over all sessions (2 x stage 1 + stage 2; persisted in `hb` records every 5 min, so a kill loses < 5 min).
  A unit is claimed only if `used + in-flight remaining + unit expected <= budget_gpu_h` (expected = measured per-arm rate, else `est_seconds`); else STOPPED.
* **Disk**: claim requires `free >= disk_min_free_gb + expected raw GB`; else all reduced, unpruned segments are pruned and the check repeats; else STOPPED.
  A `disk_full` unit end forces this check. Pruning also runs every 30 min. Reduction and pruning run in a background worker thread, never between units.
* **Cutoff**: at `cutoff_utc` no new claims; in-flight units may finish until `cutoff_utc + grace_minutes`, then their process trees are terminated and the unit is
  recorded `terminated` with the seeds the executor finds complete.
* **OOM** (`oom`): not a failure; the unit's remaining seeds wait 10 min, doubling to 60 min per consecutive OOM of that unit; the lane does other work meanwhile.
* **`unusable_solve`** at seed s of arm a in group g: s is replaced by the group's next reserve seed in EVERY arm of g that planned s (pending s dropped, completed s
  kept and listed in the ledger); the replacement unit sorts at the failed seed's position. Reserves exhausted = logged. Failure rate (failed / attempted) above
  `failure.halt_rate` with at least `min_for_rate` attempts halts the arm. `config_error` halts the arm; `other_error` retries once (same first seed), then halts it.
  Other arms continue. A lane that raises is logged (`LANE_CRASH`), the other lane continues, the exit is >= 10.
* **Blinded re-estimation** (arms in `blinded_reestimation.arms` only): once `fraction x n_conf` realizations are reduced, `executor.blinded_sd(arm, reduced_tables)` is
  called once (standard deviations only); `factor = max_e clip((sd_obs_ub / sd_plan_ub)^2, 1, max_factor)` with `sd_obs_ub` the 80 % upper bound
  `sd sqrt((n-1)/chi2_{0.2,n-1})`; `n_target = min(ceil(factor n_conf), n_max, seeds.count)`, never lower. Only SDs, n and N are logged.
* **Resume**: the ledger (append-only, flock, fsync; a torn last line is ignored) is replayed through the same `State.apply` as live operation; units claimed by a dead
  launcher are closed (`recovered`) with `executor.completed()`, and re-run from the first incomplete seed into a NEW segment directory; unreduced segments are re-reduced.

## Plan JSON (`plan_example.json`; validated by `plan.py`)

Top level: `campaign`, `schema_version`, `cutoff_utc` (`YYYY-MM-DDTHH:MM:SSZ`), `grace_minutes`, `stage1_max_hours` (30), `budget_gpu_h`, `disk_min_free_gb`, `campaign_dir`,
`binaries` (name -> md5, relative to `<build_dir>`), `analysis` {`command` (argv list; `{campaign_dir}`, `{build_dir}` substituted), `sha256_list` (path -> sha256, verified first)},
optional `recompute` / `reconcile` {`command`}, optional `blinded_reestimation` {`fraction`, `max_factor`, `arms`, `sd_plan_ub` {arm: {estimator: sd}}},
`failure` {`halt_rate`, `min_for_rate`}, `groups` {id: {`reserves`: {`start`, `count`}}} (outside the group's seed ranges), optional `executor` (`module:Class`, default
`sf43exec:Executor`, called as `Class(plan, build_dir)`), optional `poll_s` (2.0), optional `smoke`, and `arms`: `id`, `family`, `amplitude`, `runner` (`consumer`|`par2`),
`priority`, `heavy`, `min_free_mib`, `group`, `seeds` {`start`, `count`}, `n_conf`, `n_max` (1 <= n_conf <= n_max <= count), `unit_size`, `est_seconds` (per realization),
`raw_gb` (per realization), `consumer` / `par2` (the dict of the arm's runner, passed untouched), `reduce` (untouched), `retained` (seeds whose raw arrays survive pruning).

## Executor interface (duck typed; `fake.FakeExecutor` implements it for tests and the dry run)

`command(arm, seeds, segment_dir, build_dir)` -> `{argv, cwd, env, files {path: text}, log_path}` (the engine adds `CUDA_VISIBLE_DEVICES` = the lane's single id);
`classify(arm, seeds, segment_dir, exit_code, log_text)` -> `{cls in ok|oom|unusable_solve|config_error|disk_full|other_error, completed_seeds, failed_seed, verdicts, seconds, peak_mib?}`;
`completed(arm, segment_dir)`; `reduce(arm, segment_dir, out_dir)`; `prune(arm, segment_dir, reduced, retained_seeds)`; `blinded_sd(arm, reduced_tables)`; `expected_raw_gb(arm, n_seeds)`.
`arm` is the plan's arm dict. `reduce` must be idempotent (re-run after a kill).

Sizes to expect: consumer raw ~45-230 MB per realization (3-pass h_max ~2.1x), Par2 crossings ~20-30 KB per realization plus ~60 MB of snapshots per invocation; a runner
process costs ~4 s of start-up, so units should hold several seeds (5-10 consumer, 10-20 Par2).
