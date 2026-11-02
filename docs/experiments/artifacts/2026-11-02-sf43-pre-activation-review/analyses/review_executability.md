I found no blocking item. Every arm of variant A runs with the existing runner options, common random numbers hold across `lambda/h`, the 19.08-λ window is admissible, and `bfddba78` does what H14 says. I did find 5 major and 16 minor problems. The most serious are integration defects in the prepared tooling that would fail quietly, not loudly.

Note: `design_cost.py` changed on disk while I was reviewing. Variant A in it is now production 200/200/150/150 and Gate 5 n = 200/200/200/250, giving 55.4 GPU-h and 38.8–51.2 h of wall. The review and decision texts still say 150/150/150/250, 54 GPU-h and 37–49 h. The numbers below use the current script.

Paths: `E` = `docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts`, `S42` = `docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts`. Line numbers prefixed with a commit id were read with `git show`.

## Major

**X1. The blinded re-estimation (H9, H13, review 4.d(iii)) never runs with the prepared engine and executor.**
- **Wrong argument type.** The engine passes the reduce-result dicts: `171add34:E/sf43camp/lanes.py:315` calls `self.ex.blinded_sd(a.spec, [s["info"] ...])`. The executor opens each item as a file path (`3b073d8a:E/sf43camp/runner_executor.py:398`, `with open(t, ...)`). That raises a TypeError, which the engine catches and logs as `REESTIMATE_ERROR ... (N unchanged)`.
- **Wrong return shape.** Even with paths, the executor returns `{"sd": ..., "n_obs": ...}` (:408) with keys like `d25-94|hmax-1|alpha_TP` (:405). `guards.resize` looks for keys such as `alpha_TP` at the top level (`171add34:guards.py:36`), as the engine's own fake and example plan use (`fake.py:65`, `plan_example.json:17`). Nothing matches, so the factor stays 1.
- **Wrong basis.** The executor defaults to `basis: pos` (:394). Under H2 the primary basis is `disp`; with a disp planning SD, the position-basis SD (37× and 9.8× larger at 0.25 and 1) would push every re-estimated arm to `max_factor`.
- **One arm, several roles.** At 0.25 and 1 the production arm is also the Gate 5 consumer arm. At 4 the (12,16) arm is an `E_h` rung, an `E_W` cell and the Gate 5 consumer. The engine has one `n_conf`, one priority and one re-estimation per arm.
- **Fix:** read `info["tables"]`; return a flat `{estimator: sd}` for window [25,94] and one pass; set `blind = {flux, disp, perplane}`; add an end-to-end test with the real executor. Pre-register that Gate 5 uses the first n pairs whatever the extension, and that the (12,16) arm at 4 is not re-estimated.

**X2. `reconcile43` will fail on the G7 cells that use Par2's own generator, and the job will then end `ANALYSIS_FAILED`.**
- Those cells print `[4] K generation (seed=N)` (`src/runtime/ensemble/EnsembleRunner.cu:394`).
- `reconcile43` maps a `Converged=` line only to a preceding `SF-18 ... (seed=` line (`f573edec:E/reconcile43/driver.py:43`); anything else is recorded as "unmapped" (:121). That is 4 cells × 60 seeds × 2 amplitudes = 480 unmapped lines.
- Any finding gives exit 1 (:275-277). Any nonzero exit of analysis, recompute or reconcile makes the job end `ANALYSIS_FAILED`, exit 12 (`171add34:campaign.py:56-66, 141-144`).
- The test fixtures only contain SF-18 lines.
- **Fix:** also accept `K generation \(seed=(\d+)\)`, cross-check against `segment.json` seed0 + r, and add a fixture.

**X3. Under H2, the independent recomputation covers the primary estimands only on the retained subset.**
- `recompute43` rebuilds `alpha_TP` and `alpha_LP` in the position basis from the moment CSVs for every realization. It cannot rebuild the cross covariance (`alpha_TAP`) or the displacement basis, which "use the arrays of the retained subset" (`f573edec:E/recompute43/sources.py:9-18`).
- `prune` deletes every `.npy` of non-retained realizations.
- So acceptance F's 1e-10 check reaches `alpha_TP` and `alpha_TAP` in the disp basis on only about 5 realizations per arm. The drafts do not disclose this.
- **Fix:** prune selectively, keeping `seeds`, `weights`, `planes_x23` and `planes_tau` for every realization of the Gate 4 arms (about 28 MB per realization at M = 128 and 17 MB at M = 100, roughly 80 GB on the host), or enlarge the retained subset of the primary arms and state the coverage in H2.

**X4. Two of the review's own scripts do not exist.**
- Review lines 402-405 and `scripts/README.md:11-12` list `gate4_power_sim.py` and `gate5_options.py`.
- The artifact contains only `design_cost.py`, `firstorder_window.py` and `lattice_check.py`, and `find` locates no copy anywhere; the agents ran the code inline (`gate4_power.md:1`, `gate5_coverage.md:1`).
- Sections 3.c-3.e, H5's "rate/tail ≤ 1.08" and the evidence behind H16 cannot be reproduced from the PR.
- **Fix:** commit the scripts or withdraw the claims.

**X5. One failed background reduction can silently shorten an arm in the in-job analysis.**
- A failed reduction is not retried within a session (`171add34:lanes.py:283`); unreduced segments are re-submitted only when a resumed session starts.
- `campaign.py:134-141` goes straight from `eng.run()` to the analysis.
- Under H13's "longest complete prefix", an unreduced early segment of 5-10 seeds cuts the arm there, possibly below its floor, so its hypothesis enters Holm with p = 1. This is unlikely (I/O error, full disk on the shared host) but costly.
- **Fix:** retry reductions with backoff, re-submit every unreduced segment before the final analysis, and have the analysis treat "done but unreduced" as an error rather than the end of the prefix.

## Minor

- **m1. G7 cannot share a cell with the Gate 5 Par2 arm.**
  - `design_cost.py:50` counts `8*g7 - g7` "(one cell is the Gate-5 Par2 arm)".
  - The mcstat fraction keeps cells with g·i·d·r = +1 and codes `dt` as −1 (`S42/mcstat/splitplot.py:45, 61-67`), consistent with decision G7's "dt against dt/2".
  - The matched configuration (SF-18, window, dt, 1e-10) is (+,+,−,+), product −1, so it is outside the fraction.
  - **Fix:** run all 8 cells (about +0.5 GPU-h).
- **m2. H15's coverage run cannot run "in parallel with the smoke test" as written.**
  - Both would need the one project-wide `run` slot.
  - Running it under `exec` breaks the runbook's "short, bounded" rule.
  - Locally there are 16 cores; the 45 core-hours are host core-hours.
  - **Fix:** run it as a background process inside the smoke-test job, or serially (adds 1.5-2 h to the critical path).
- **m3. H13's priority order is not what the engine does.**
  - Lane A takes any heavy unit before any light one, whatever its priority (`scheduler.py:192`). So the priority-3 distance arm at 2.25 (2.8 GPU-h) runs before every priority-1 light arm.
  - Re-estimation extensions keep the arm's priority (`scheduler.py:115-116`), not class 2.
  - Common-random-number blocks of heavy and light arms run in different stages.
  - The `lambda/h = 4` rung at 4 has priority 2 (`design_cost.py:63`) although H6 uses it for gating. The engine forbids mixed priorities inside one seed group (`plan.py`), so that rung would lose group-wide replacement.
  - **Fix:** state these rules in H13, and give the rung at 4 priority 1.
- **m4. The stage-1 estimate assumes the second GPU can run near-empty grids.**
  - With H14's 17000 MiB second threshold, lane B can only take half-class grids, and it waits while holding its GPU (`scheduler.py:188`, `lanes.py:108`).
  - Recomputed: stage 1 is 17.9-23.7 h, wall 40.0-52.9 h and 58.0-76.5 GPU-h, against the script's 16.7-22.0 h and 38.8-51.2 h.
- **m5. H18's premise is contradicted by the repository.**
  - SF-32's `apps/spurious_spreading/analyze.py:45-48` imports matplotlib unconditionally and ran inside host jobs that exited 0 and produced the committed PNG/PDF figures (`docs/experiments/2026-10-06-sf32-spurious-spreading.md:84-86, 144-157`).
  - The SF-42 README only says its tools use numpy and scipy, not that the host lacks matplotlib.
  - The decision itself (figures off-host) is fine; correct the stated reason.
- **m6. H14's wording conflicts with the runbook and `AGENTS.md`.**
  - H14 (decisions:162) says no command writes `CUDA_VISIBLE_DEVICES`.
  - The engine sets it to one id per lane (`procs.py:58`), which the runbook in `bfddba78` explicitly permits.
  - Root `AGENTS.md:198-203` still forbids it, and nothing updates it.
- **m7. H16 points to a value that is not there.**
  - It cites the "proposed value and reachability" in review section 3.d, but 3.d contains neither.
  - The f = 0.5 proposal exists only in `gate4_power.md:12, 186-188`.
- **m8. Disk and memory figures need correcting.**
  - "Consumer 45-230 MB per realization": the formula gives 95-231 MB with the Pollock bridge; 54 MB is the tracker-only part at 4.
  - "29.2 / 29.9 / 17 GB" are actually 29229 / 29881 / 17000 MiB.
  - 29881 MiB is a six-realization peak, leaving 119 MiB under the 30000 MiB threshold.
- **m9. The Par2 time-window secondary and the "of Par2" part of the B&dD eq. 6 fit (spec items 1 and 9) are not covered.**
  - They need regular Par2 snapshots: the reducer raises an error without them, and the executor defaults to none.
  - Snapshots are absent from the cost and disk figures, while the review says every arm is kept.
- **m10.** Nothing enforces G8's minimum of 5 retained realizations per arm (`min_retained=0`, `runner_executor.py:384`); a replaced retained seed leaves 4.
- **m11.** The 416x48x48 rung (`sigma^2 = 4`, `lambda/h = 4`) was never run in SF-42; add it to the smoke test.
- **m12.** H10 says M = 128 is "the only common admissible value"; the lattice check gives {128, 256}.
- **m13.** Par2 rows in the reduced table have `seed = NaN` (`S42/sf42red/reduce.py:526`). The Gate 5 pairing must derive seeds from `segment.json` seed0 + r.
- **m14.** Whether a replaced seed stays in an arm depends on run order (`scheduler.py:105-110`). Pre-register the rule for the analysis.
- **m15.** The per-realization `.planes.csv` tables would come to about 0.65 GB for variant A, against SF-39's 26 MB of committed reduced tables. Decide what is committed.
- **m16. Already in the orchestrator's audits; keep them as integration gates.**
  - Executor constructor: the engine calls `Class(plan, build_dir)`, which `RunnerExecutor` does not accept.
  - Directory layout: `reconcile43` globs `arms/*/*/run.log` (driver.py:191), but the engine writes `segments/...`.
  - Ledger key: `reconcile43` reads `replaced_by`; the engine writes `replacement`.
  - H5's Fieller and Hall procedures, and H6's order cap, two-triple minimum and Giles maximum, all need new code that mcstat does not provide.

## What survived

- **Variant A arms:**
  - **Runnable with existing options:** production, all three ladders plus the extra rung at 4, the width cells, h_max 8/16/32 on 30 production seeds, and the Gate 5 consumer at 2.25 on 1248x288x288 with production seeds.
  - **Doubling arms work:** the 1664x192 arm at 4 can serve as the Gate 5 consumer with 250 seeds.
  - **Par2 and G7:** factors are configuration-only.
  - **Distance arms:** planes up to 199 λ stay below the 208 λ box length.
  - **Twin reuse at 0.25 and 1:** the production runs supply the uniform-weight values (same grid, window and M).
- **Common random numbers hold.** The generator draws each Fourier mode from a stateless hash of the seed and integer mode index, so on the same box a finer grid reproduces the coarser field exactly, apart from modes that are negligible beyond the coarse cutoff. Par2 builds the bit-identical field when `dx = 1/ny` exactly, which the executor writes.
- **Displacement basis:** available for `alpha_TP` and `alpha_TAP` for both codes, and the Fieller inputs (`alpha_LP`, uniform weights) exist for both.
- **Lattice check:** I re-ran it. 0.25 and 1 give {128, 256}; 2.25 gives none at 19.2 λ and 121 values at 19.08 λ; 4 gives 65 values with or without the extra rung. M = 100 passes on all seven grids at 2.25. The window strings parse in both runners.
- **Helper (`bfddba78`):** I ran its `gpu_select` in memory. It returns `CHOSEN 1,0` with rc 0 for two GPUs, `CHOSEN 1` with rc 3 when only one qualifies, and none with rc 1. It degrades to one GPU at the 7200 s deadline, and count 1 is unchanged.
- **Runtime claims:**
  - OOM classification matches the actual error text.
  - A non-converged Par2 solve exits 0 (`EnsembleRunner.cu:416-420`) and the executor catches it.
  - Each segment has its own log, written by one polling thread.
  - `--min-retained` and the h_max pruning work.
  - The helper indeed truncates the log, kills a same-name session, and its `wait` ignores `tmux_active`.
- **Disk:** about 0.54 TB of raw output, consistent with the review.
- **Engine guards:** the cutoff (19:00Z) with grace, the budget and disk guards, and lane B's 30 h cap all work as described.

Not checked: the statistics behind H5 and H15, the remote host itself (free disk, matplotlib), and the commit test suites, which need temporary directories. The biggest risk is X1 and X2: the job would finish without the re-estimation and then end `ANALYSIS_FAILED` on reconciliation.

### Critical files for implementation
- `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43camp/lanes.py` and `scheduler.py` (commit `171add34`)
- `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43camp/runner_executor.py` (commit `3b073d8a`)
- `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/reconcile43/driver.py` and `recompute43/sources.py` (commit `f573edec`)
- `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts/design_cost.py`
- `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md`