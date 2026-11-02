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
| 13 | refused to start: md5 mismatch of a binary, sha256 mismatch of an analysis file (`REFUSED analysis file sha256 mismatch: <paths>`; checked at start and again on `--resume`, before any unit is claimed), invalid plan, stale state (ledger exists without `--resume`, or a different plan), another launcher holds `campaign_dir/LOCK`, usage |
| 143 | SIGTERM/SIGINT: units terminated and recorded, no analysis (a cancel); `--resume` continues |

`campaign_dir` receives: `ledger.jsonl`, `LOCK`, `segments/<arm>/u<unit>_s<seed0>/` (one new directory per run of a unit; `run.log`, `segment.json`,
`seedmap.json`, `out/`, Par2 `config.yaml`), `reduced/<arm>/<segment>/<arm>__<segment>__d<lo>-<hi>.csv` (+ `.planes.csv`, `.manifest.json`; their `source`
paths are relative to the segment directory), `background/<name>.{log,done}`, `analysis.log` `recompute.log` `reconcile.log`, `DETERMINISM.txt`,
`REPORT.txt` (per arm: planned / completed / replaced / failed / OOM retries / pending / peak MiB; retention; UNREDUCED list; background jobs; determinism;
events), optionally `ANALYSIS_BLOCKED_UNREDUCED` and one marker: `DONE`, `DONE_PARTIAL`, `STOPPED` or `ANALYSIS_FAILED`. The analysis (`analysis.command`,
then `recompute`, then `reconcile`) ALWAYS runs after the loop ends, except after a refused start or a signal.

## Queue, lanes, stages

* A **unit** = up to `unit_size` consecutive seeds of one arm inside one aligned seed block (`--seed0 s --nr k`; Par2 likewise with NR). Units are formed
  on demand from the arm's pending seeds, so a resumed or partly failed unit restarts at its first incomplete seed.
* Order = (priority class, progress fraction of the CRN group, group order, anchor seed, replacement flag, arm order); fraction = (first seed - group start)
  / (largest **`n_conf`** in the group, not the re-sized `n_target`). All arms of a CRN group (same `seeds.start`, same `priority`; enforced) run the same seed
  block back to back, groups of a priority class advance together, and arms with fewer seeds simply have no later blocks. The extension seeds of a blinded
  re-estimation (positions >= `n_conf`) are separate units of priority 2 (or the arm's own if higher); a unit never mixes planned and extension seeds.
* **Priority first** (H13/F4): a lane considers only the claimable units of the highest pending priority. Within it **lane A** (first granted id) prefers the
  largest `min_free_mib` (near-empty GPU first), then queue order; **lane B** (second id; heavy units only) takes the first in queue order. A light priority-0
  unit therefore runs before a heavy priority-1 unit. Before each claim a lane reads the free memory of its own GPU (`nvidia-smi --query-gpu=memory.total,memory.used
  -i <id>`, injectable; unreadable = enough) and claims only units whose arm has `min_free_mib` <= free; if none fits it retries after 120 s.
* **Lane B's class** (H14/F7): heavy units whose `min_free_mib` <= its launch threshold, the plan's `lane_b_min_free_mib`, else the environment's
  `REMOTE_GPU_MIN_FREE_MIB_2`, else `REMOTE_GPU_MIN_FREE_MIB` (the helper does not export them to the job: put the number in the plan or in the job command),
  else every heavy unit. Units outside the class (near-empty) are taken only when the live free memory allows at once and are **never waited for**: lane B
  logs `LANE_B_RELEASED gpu=<id> utc=<..>` and leaves as soon as no pending unit of its class (and nothing it can claim now) remains, or at the 30-h cap.
* **Stage 1** (two GPUs): lane B also leaves when the next unit of its class would end after `stage1_start + stage1_max_hours` (never exceeded, nothing is
  killed), and at the cap even if it was waiting for memory. `stage1_start` is written once and reused on resume. After the release, or on a
  resume past the cap, only lane A runs (`RESUME_SINGLE_LANE`: relaunch with one GPU). One GPU granted: lane A only.
* **Lane pin** (optional arm field `lane: "A"|"B"`): only that lane claims the arm's units (a pinned light arm may run on lane B); when that lane does not exist
  (one GPU, or lane B already released) the pin is ignored and `LANE_PIN_IGNORED arm=<id>` is logged once.
* Job log: one `SF43_UNIT lane= gpu= arm= seed0= nr= segment= utc=` and one `SF43_UNIT_END ... cls= completed= seconds=` line per unit, plus `STOP`, `OOM`,
  `REPLACEMENT`, `ARM_HALT`, `REESTIMATE`, `PROJECTION` (hourly, vs budget and cutoff), `DISK_GUARD`, `LANE_CRASH`, `FINAL` lines. Runner output goes to the unit's own
  `runner.log`.
* `--dry-run`: simulates the whole plan on a virtual clock with `est_seconds` (no process, no file): the ordered schedule (`DRYRUN SF43_UNIT ...`) and
  `DRYRUN_SUMMARY` (stage-1 / stage-2 hours, GPU-h, end vs cutoff). The md5 and analysis sha256 gates are skipped. `--smoke`: the plan's `smoke` section overlays the plan
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
* **Blinded re-estimation** (H9; arms in `blinded_reestimation.arms` only, never an arm flagged `no_reestimation`: Gate 5 arms and the sigma^2 = 4 (12,16) rung): once
  `fraction x n_conf` realizations are reduced, `executor.blinded_sd(arm, [reduce-info dicts])` is called once and returns `({estimator: sd}, n)` (SDs and a count, no mean). Per estimator of the arm's
  `hw_target` (planning half-width): `N_req = ceil((t_{0.95,n-1} sd_UB80 / hw_target)^2)` with `sd_UB80 = sd sqrt((n-1)/chi2_{0.2,n-1})`; `n_target = min(floor(max_factor n_conf),
  max(n_conf, max_e N_req), n_max, seeds.count)`, never lower. The extension seeds are priority-2 units. Only SDs, n, N_req and N are logged (`reest` record).
* **Reductions** (X5): a failed reduction is retried 3 times within the session (back-off 1, 5, 15 min). Before the final analysis every completed but unreduced segment
  is re-submitted and waited for (3 rounds). What remains unreduced is written to the ledger (`unreduced`), `REPORT.txt` (UNREDUCED) and
  `ANALYSIS_BLOCKED_UNREDUCED`, and the analysis STILL runs: it must treat those seeds as MISSING (sf43ana's loader already reports a missing reduced table), not as the end of a prefix.
* **Retention** (G8, H21): at least `min_retained` (plan, default 5) raw-array realizations per arm. A seed is retained when it is in the arm's `retained` list (and its replacement
  if replaced), among the first `min_retained` positions of the arm's effective seed order (a replaced position holds its reserve), or among the first `min_retained` completed seeds in
  that order. Pruning decides per segment with the state of that moment (extra retention is possible while a position is open, never less); `RETENTION_SHORT` is logged if the
  report's count falls below the requirement. Arms with `primary: true` keep, for EVERY realization, `seeds`, `weights`, `planes_x23`, `planes_tau` (`sf42red prune --keep-arrays`; what the
  displacement-basis recomputation reads) and delete only the other arrays (time passes, Pollock, `planes_I`) of the non-retained ones. Par2 segments: the regular snapshots are
  deleted after the reduction (those of retained realizations stay), the crossing CSVs stay.
* **Pollock contrast** (spec items 5 and 8; disclosed SF-43 addition to the SF-42 reducer, like `--keep-arrays`): a consumer arm with `consumer.pollock_bridge` on (default) is
  reduced with `sf42red reduce --pollock`. Source: `lagrangian_ensemble --pollock-bridge on` writes ONE `<realization>/pollock/` per realization, whatever the number of `--hmax-div`
  passes (`EnsembleRunner.cu`; `pollock_weights` (Np), `pollock_status_plane` (u1, raw SF-32 codes, 8/9 = inlet/outlet exit in medium (b)), `pollock_planes_x23` (Np,P,2 unwrapped, NaN
  from the first plane not reached), `pollock_planes_t` (Np,P clock since injection, the consumer's `tau`), `pollock.json` (`seeds_equal_tracker`, `planes`), ...). The Pollock
  particles start at the consumer's `seeds` (checked through `seeds_equal_tracker`, planes compared with `track.json`); the displacement basis uses the pass's `seeds.npy`.
  Rows: `code = pollock`, family `plane`, the consumer's columns and estimator code (weights `flux` = `pollock_weights` and `unif`, bases pos/disp, sets perplane/common, split halves;
  no `expI`, no `time` family), `seed` = field seed, `hmax_div` = the label of the consumer rows of the same root/pass (identical Pollock rows under each pass label of a multi-pass
  realization), `source` = `<realization>/pollock`. Set rule = the consumer's: finite record and weight > 0 (a Pollock seed with non-positive RT0 weight is tracked but outside
  the sets, unlike the bridge's own `pollock_moments_planes.csv`, whose `contributing` counts it). The status code is not read: a stopped or exited particle has NaN records
  from there on. A realization without `pollock/` is a `MissingData` error of the reduction. `prune` deletes `pollock_*.npy` only when the table has the finite `code=pollock` rows
  of that directory (otherwise action `keep`, reported in the plan line "unreduced pollock array(s) kept"). `blinded_sd` reads `code = <the arm's runner>` rows only, so Pollock
  rows never enter the re-estimation. `recompute43` recomputes the Pollock rows of the realizations whose Pollock arrays are still on disk (the retained subset, same 1e-10
  criterion, independent of `sf42red`) and lists the others under NOT_COVERED.
* **Resume**: the ledger (append-only, flock, fsync; a torn last line is ignored) is replayed through the same `State.apply` as live operation; units claimed by a dead
  launcher are closed (`recovered`) with `executor.completed()`, and re-run from the first incomplete seed into a NEW segment directory; unreduced segments are re-reduced.
* **Ledger records the analysis tools read**: besides the engine's own kinds (`claim`, `unit_end`, `hb`, `reduced`, `pruned`, `reest`, ...), one `seed` record per solve
  `{kind, arm, group, segment (basename), segment_path, seed, seed0, unit, verdict, class (ok|unusable_solve), iterations, rel, balance, max_div, field_seed}` (sf43ana's loader and
  reconcile43 read exactly these) and `replacement` `{arm, group, seed, replaced_by}`. Solver verdicts: consumer `converged|fallback|...` as the runner prints; Par2 `converged` (Converged=YES),
  `fallback` (Converged=NO with final/initial residual <= 1e-8, usable; Par2 prints no balance) or `not_converged_beyond_fallback` (unusable: the seed is replaced group-wide).
  `max_div` and `iterations` come from `solver_b.json` (consumer) or the printed `iters=` (Par2). Per unit `peak_mib` (largest `MEM_PHASE device_used`/`exact_bytes`) is in `unit_end` and the REPORT (per arm max).
* **Seed map**: every segment has `seedmap.json` (`seed0`, `nr`, `seeds: {r: seed0 + r}`); the Par2 rows of the reduced tables carry `seed = seed0 + r` (the analysis pairs Gate 5 by seed) and
  reconcile43 cross-checks the printed field seed (`[4] SF-18 medium (b) field (seed=` or Par2's own generator `[4] K generation (seed=`, G7 factor G) against it.
* **Determinism pairs** (plan `determinism_pairs: [[x, y], ...]`, two arms with identical runner dicts and seeds, typically pinned to lanes A and B): once both arms are complete and reduced,
  every raw output file of every shared seed (consumer: the whole `out/realizations/r_XXXX/` tree; Par2: crossings, stats, snapshots of the realization) is compared by sha256, except logs and
  timing files (`*.log *timing* *texit* *mem_phase* prune_log.csv retained.txt`, extended by the plan's `determinism_exclude`), and the reduced rows value by value (all columns but `source`, `r`).
  `DETERMINISM.txt` has PASS/FAIL per file, the REPORT one line per pair; a FAIL never stops the job.
* **Background jobs** (plan `background: [{name, argv, nice (19), env}]`, `background_wait_minutes` default 0): started once at engine start with `CUDA_VISIBLE_DEVICES=""`, `nice`, own
  process group, `{campaign_dir}`/`{build_dir}` substituted in `argv`, log `background/<name>.log`; the exit code goes to `background/<name>.done` and the ledger (`background_start`,
  `background_end`); a resumed session skips a job whose done file exists (a job that was running when the launcher died has none and starts again). SIGTERM terminates them with the units.
  The final analysis waits up to `background_wait_minutes` for them, the rest is terminated; the REPORT lists their state.

## Plan JSON (`plan_example.json`; validated by `plan.py`)

Top level: `campaign`, `schema_version`, `cutoff_utc` (`YYYY-MM-DDTHH:MM:SSZ`), `grace_minutes`, `stage1_max_hours` (30), `budget_gpu_h`, `disk_min_free_gb`, `campaign_dir`,
`binaries` (name -> md5, relative to `<build_dir>`), `analysis` {`command` (argv list; `{campaign_dir}`, `{build_dir}` substituted), `sha256_list` (path -> sha256, verified at start, before any unit, and again before the analysis)},
optional `recompute` / `reconcile` {`command`}, optional `blinded_reestimation` {`fraction`, `max_factor`, `arms`, `hw_target` {arm: {estimator: half-width}}},
`failure` {`halt_rate`, `min_for_rate`}, `groups` {id: {`reserves`: {`start`, `count`}}} (outside the group's seed ranges), optional `executor` (`module:Class`, default
`sf43camp.runner_executor:RunnerExecutor`, built as `Class(plan, build_dir)`), optional `min_retained` (5), `lane_b_min_free_mib`, `determinism_pairs`, `determinism_exclude`,
`background`, `background_wait_minutes` (0), `poll_s` (2.0), `smoke`, and `arms`: `id`, `family`, `amplitude`, `runner` (`consumer`|`par2`),
`priority`, `heavy`, `min_free_mib`, `group`, `seeds` {`start`, `count`}, `n_conf`, `n_max` (1 <= n_conf <= n_max <= count), `unit_size`, `est_seconds` (per realization),
`raw_gb` (per realization), `consumer` / `par2` (the dict of the arm's runner; consumer optional `hmax_div_first: {"n": 30, "divs": [8,16,32]}`: the units inside the first n positions run with
`--hmax-div 8,16,32`, the others (later positions, reserve and extension seeds) with the arm's `hmax_div`; `unit_size` must divide n; single-pass segments of such an arm are
reduced with `--single-pass-hmax <hmax_div[0]>` so the production pass has one `hmax_div` label in both layouts, and `recompute43 --single-pass-hmax` accepts it), `reduce`,
`retained` (the human-review sample whose raw arrays survive pruning), optional flags `primary` (selective pruning above), `no_reestimation`, `lane` ("A"|"B").

## Executor interface (duck typed; `fake.FakeExecutor` implements it for tests and the dry run)

`command(arm, seeds, segment_dir, build_dir)` -> `{argv, cwd, env, files {path: text}, log_path}` (the engine adds `CUDA_VISIBLE_DEVICES` = the lane's single id);
`classify(arm, seeds, segment_dir, exit_code, log_text)` -> `{cls in ok|oom|unusable_solve|config_error|disk_full|other_error, completed_seeds, failed_seed, failed_seeds, verdicts, solver {seed: {verdict, iterations, rel, balance, max_div, field_seed}}, seconds, peak_mib}`;
`completed(arm, segment_dir)`; `reduce(arm, segment_dir, out_dir)` -> `{tables, ...}`; `prune(arm, segment_dir, reduced, retained_seeds)`; `blinded_sd(arm, [reduce results])` -> `({estimator: sd}, n)` (a plain
`{estimator: sd}` is accepted, n is then the number of reduced realizations); `expected_raw_gb(arm, n_seeds)`.
`arm` is the plan's arm dict. `reduce` must be idempotent (re-run after a kill).

Tests: `python3 -W error -m unittest discover -s sf43camp/tests -t .` (engine, fake executor, virtual clock) and `-s sf43camp/tests_executor -t .` (the real executor with fake runner binaries;
`test_e2e.py` runs the whole engine through `campaign.main` with the real `RunnerExecutor`, a CRN group, reduce, prune, a SIGTERM kill + `--resume`, the background job, the determinism pair
and the real `recompute43` / `reconcile43` hooks, ~6 s). No GPU, no data.

Sizes to expect: consumer raw ~45-230 MB per realization (3-pass h_max ~2.1x), Par2 crossings ~20-30 KB per realization plus ~60 MB of snapshots per invocation; a runner
process costs ~4 s of start-up, so units should hold several seeds (5-10 consumer, 10-20 Par2).
