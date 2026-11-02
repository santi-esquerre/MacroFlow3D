**Verdict.** The core claim holds: variant A finishes before the Sun 16:00 cutoff from the review's own launch window (Tue 22:00 to Wed 14:00), on two GPUs or one, with a day or more to spare. The stated margins and several supporting statements do not hold. No finding is BLOCKING for A. Six are MAJOR: phase 2, coverage run, GPU availability, cut order, re-estimation, variant C. Seven are MINOR.

How I checked: I ran `design_cost.py` and reproduced its output. I also re-simulated the schedule in memory with `python3 -B -c`, importing its arm list and nothing written to disk. The simulation follows the lane rules actually coded in the prepared engine: lane A takes heavy units first (priority, then near-empty); lane B takes heavy units only, limited by its GPU memory, and leaves only when no heavy unit remains. Units are 10 consumer or 20 Par2 realizations.

**Path aliases (absolute):**
- R = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-11-02-sf43-pre-activation-review.md`
- D = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md`
- DC = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts/design_cost.py`
- CS, EI, G4 = `/home/sesquerre/Projects/MacroFlow3D/.claude/orchestration/SF-43/phase1/agents/{cost_schedule,exec_inventory,gate4_power}.md`
- RB = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/runbooks/remote-v100.md`
- INC = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/plans/active/lester-eq14/increments/`
- E1 = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/agent-a87320eaf2de12002/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43camp/` (engine, commit 171add34)
- E2 = `runner_executor.py` in `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/agent-af1eb200aa845cf1c` (commit 3b073d8a)
- HR = `/home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/agent-a354032c9271d4a57/scripts/remote` (helper, commit bfddba78)

## Recomputed variant A wall (h)

| scenario | ×1.0 | ×1.101 | ×1.32 |
|---|---|---|---|
| DC as written | 36.9 | 40.6 | 48.7 |
| engine rules, both GPUs ≥ 30 GB free | 37.0 | 40.7 | 48.8 |
| engine rules, lane B GPU has 17–30 GB free (the H14 design point); lane B leaves at | 38.4 (17.9) | 42.3 (19.7) | 50.7 (23.6) |
| same, lane B time-shared with a colleague (×2) | 43.4 | 47.7 | 57.2 |
| lane A time-shared ×2 for the whole run | 76.3 | 84.0 | 100.7 |
| "fill": lane B also takes light units until 30 h | 26.8 | – | 40.7 |
| one GPU (plus the helper's 2 h wait) | 53.5 | 58.9 | 70.7 |
| one GPU, time-shared ×2 for the first 24 h | 65.7 | 71.1 | 82.7 |
| full 2× re-estimation, two GPUs / one GPU | 53.6 / 78.9 | – | 74.2 / 104.1 |

**Latest launch** that still completes by Sun 16:00, at ×1.32 with one 2 h resume (and the 2 h degraded-start wait on one GPU):

| variant | two GPUs | one GPU |
|---|---|---|
| A, lane B 17–30 GB free | Fri 11-06 11:18 | Thu 11-05 13:18 |
| A, both GPUs near-empty | Fri 13:12 | – |
| A, lane B time-shared | Fri 04:48 | – |
| A, lane A time-shared all run | Wed 09:18 | – |
| A, one GPU shared first 24 h | – | Thu 01:18 |
| A + full 2× re-estimation | Thu 11:48 | Wed 03:54 |
| B | Sat 01:00 | Fri 09:24 |
| C | Wed 23:48 (Wed 08:48 if lane B shared) | none at ×1.32; Wed 11:06 at ×1.101 |

## Findings

**F1 — MAJOR — phase 2 duration**
- **Attacked:** R:327, phase 2 in 10–16 h ("Tue 20:00–Wed 02:00 / Wed 04:00–10:00"). The "pessimistic" column only moves the merge time; it keeps the same 10–16 h.
- **Evidence:**
  - The analysis package (P2-A1/A2, 70–95 worker-hours, EI:209) moved from "planned" to "running" during this review, per `dag.json`.
  - It must be integrated, tested on the host and sha256-listed before launch. The mirror cannot be synced while the job runs (RB:97-100, EI:45), and the engine runs the final analysis inside the job behind a sha256 gate.
  - Integration is not free. Two engine–executor mismatches, both confirmed with Python:
    - E1 `campaign.py:210-215` builds the executor as `Class(plan, build_dir)` and catches only ImportError/AttributeError. E2:215 `RunnerExecutor` has no `__init__`, so the job raises `TypeError: RunnerExecutor() takes no arguments` at start.
    - E2:408 returns `{"sd":…, "n_obs":…}`. E1 `lanes.py:322` formats each value with `"%.6g"` outside its `try`, which raises `TypeError: must be real number, not dict`. The engine crashes at the first re-estimation, and again on every `--resume`, because no `reest` record was written.
  - Precedent: SF-39 took 21.7 h from activation to its first production job (INC/SF-39…md:318-320; 16:06Z → 13:46Z).
- **Correction:**
  - Restate phase 2 as about 12–25 h, giving a pessimistic launch around Wed 21:00.
  - Add the gate "analysis integrated, host-tested and hashed" to the table.
  - Publish the latest-launch table above as go/no-go clock limits.
  - Optionally decouple the final analysis into a post-campaign `run` job, with its hash committed before `blinded stop`. That removes the analysis package from the launch critical path.

**F2 — MAJOR — the coverage run cannot run in parallel with the smoke test**
- **Attacked:** D:170-171 (H15) "in parallel with the smoke test"; also R:287-288 and R:327.
- **Evidence:**
  - Every `run` job takes the single project-wide slot (RB:67-71, 83-84).
  - A 1.5–2 h CPU job must be a `run` job, not `exec` (RB:18-23, 34-40).
  - SF-42's CPU-only coverage job held the slot for 41.7 h (INC/SF-42…md:209).
- **Correction:** either one combined `run` job (coverage at `nice 19` in the background plus the smoke test), or add 1.4–2.1 h serially. Amend H15 and the critical-path row.

**F3 — MAJOR — GPU availability and sharing are not covered**
- **Attacked:** R:470-471, "the pessimistic margins (1.32) and the cutoff cover this"; R:307 one-GPU fallback; R:328 "up to 2 h per attempt".
- **Evidence:**
  - The job cannot start without one GPU with ≥ 30000 MiB free (H14, HR).
  - 17.9 planning GPU-h of A, 15.1 of them priority 1 (production at σ²=4, the 2080x480 E_h rungs at 0.25 and 1, h_max at 4), run only on a GPU with about 29.9 GB free. A colleague who later takes more than about 2.6 GB on lane A's GPU (32510 − 29881 MiB) blocks them for as long as he stays (`2026-10-27-remote-single-gpu.md`:43).
  - The observed colleague pattern is 84 % utilization at 1281 MiB (same file :15-16). Such a GPU is eligible for either lane and halves its throughput.
  - Walls under sharing are in the table. One GPU shared for the whole run takes 141 h at ×1.32 (118 h at ×1.101): infeasible from any launch.
  - So ×1.32 covers sharing of at most about 30 % of lane time. It does not cover a doubling.
- **Correction:**
  - (i) State that limit plainly.
  - (ii) Add an outcome-blind clock rule: with no GPU ≥ 30000 MiB by Thu 12:00, launch with a 17000 MiB first-GPU threshold. The engine already claims units against live free memory (E1 `scheduler.py:172-195`), so it would pick up near-empty units if memory frees.
  - (iii) Adopt "fill" (lane B also takes light units until the 30 h cap). Same GPU-h; wall 50.7 → 40.7 h, and 100.7 → 81.3 h when lane A is shared (CS:160 already listed this lever).
  - (iv) Shrink the near-empty block (see F8).

**F4 — MAJOR — what the engine cuts first under a binding cutoff is not what H13 says**
- **Attacked:** D:149-151 (H13 priorities, interleaving within a priority); R:373-374; the engine's own claim "a cutoff reduces N evenly".
- **Evidence:**
  - E1 `scheduler.py:191-193`: lane A takes a heavy unit of any priority before any light unit.
  - `scheduler.py:115-116` and `:48-49`: a re-estimation extends the same arm (priority 1, not H13's priority 2) and rescales the whole group's progress fractions.
  - One GPU, ×1.32, launch Fri 06:00: 5.0 planning GPU-h of priority 1 left undone, against 2.1 with priority ordered first; meanwhile the priority-3 distance arm at 2.25 (2.8 GPU-h) is completed.
  - Launch Fri 14:00: production at 0.25 keeps 50 seeds and its 520x120 rung 40, while the 2080x480 rung completes 100. That leaves 60 fine-rung realizations (about 2.3 near-empty GPU-h) with no ladder partner.
- **Correction:**
  - Order units by priority, then progress fraction. Apply near-empty-first only within one priority, or only while lane B is alive.
  - Queue extension seeds as separate priority-2 units, and compute group fractions on `n_conf`.
  - Fix the wording in R §4.d and in H13.

**F5 — MAJOR — the 2× upward re-estimation is not costed**
- **Attacked:** R:31-33 and D:121, "54 GPU-h (74), 37–49 h"; D:125-126.
- **Evidence:**
  - G5.2 allows "upward only, at most 2x" (`docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`:98). The engine's example plan uses `max_factor` 2.0.
  - Production plus E_W is 25.35 planning GPU-h, so the worst case is 78.9 planning GPU-h. Walls are in the table. The one-GPU fallback becomes infeasible for any launch after Wed 03:54.
  - The increase buys nothing: in the displacement basis, N = 150 already exceeds the precision N of about 7–30 (D:48-50).
- **Correction:** define the re-estimation on the precision/power-required N. Then 2 × 30 = 60 < 150, so it can never extend and costs nothing. Otherwise cap `n_max` and cost it. Fix the `blinded_sd` contract from F1 either way.

**F6 — MAJOR — variant C claims**
- **Attacked:** R:40-41 and R:386-388, "88 GPU-h, Saturday, least slack"; D "60–86 h".
- **Evidence:**
  - C has no one-GPU fallback: 116 h at ×1.32, and at ×1.101 the latest launch is Wed 11:06.
  - On two GPUs the latest launch is Wed 23:48, about 10 h after the review's own pessimistic launch time.
  - From a Wed 14:00 launch it completes around Sun 06:00, not Saturday.
  - Raw output is 943 GB, 1.7 times the 548 GB free.
- **Correction:** present C as "two GPUs from launch, launch by about Wed 22:00, pruning must work throughout, otherwise run A", with a pre-registered clock rule.

**F7 — MINOR — DC ignores lane memory classes**
- **Attacked:** R:305-306 and R:364.
- **Evidence:** DC:76-89 balances all heavy items across both lanes. At 17–30 GB free, lane B can run only half-class units, giving the table's 38.4 / 42.3 / 50.7 h. The engine also keeps lane B waiting (counted as GPU-h) while near-empty units it can never claim are still queued: +2.8 to +3.6 GPU-h.
- **Correction:** model lane classes; release lane B when nothing remaining fits its GPU; or adopt fill.

**F8 — MINOR — h_max is double counted**
- **Attacked:** DC:46, 55, 64 and R:349.
- **Evidence:**
  - H8 uses "the first 30 production seeds" (D:116). SF-39 showed the 3-pass and 1-pass runs are byte-identical (INC/SF-39…md:319).
  - So the separate h_max arm repeats work: 2.0 GPU-h, 1.1 of it near-empty.
  - E_int uses only the h/16 and h/8 passes (D:116). The h/32 pass is 4 of the 6 extra tracing units (CS:50-52): another 1.25 GPU-h.
- **Correction:** run the first 30 production seeds with `--hmax-div 8,16,32` (or `8,16`): −2.0 to −3.2 GPU-h.

**F9 — MINOR — ranking by information per GPU-h**
- **Attacked:** R:341-350.
- **Evidence:**
  - The ranking is asserted, not computed. It is by family and by total GPU-h, while the binding resource is near-empty GPU time (17.9 GPU-h, 34 % of A).
  - E_W dominates at 0.25 and ties with E_h at 1 (G4:86, 89). The expensive E_h rungs are at 0.25 and 1; E_h dominates at 2.25 and 4, where its rungs are cheap. "E_h rank 3 above E_W rank 4" hides this.
  - Kept but nearly worthless: h_max (E_int ≈ 1e-6).
  - Cut but valuable: the σ²=4 distance arm. It costs 1.44 GPU-h on a light grid, less than h_max, and is offered only in C.
  - Halving the ladder and width arms at 0.25 keeps P(RL > 0) = 1.000, with E[RL]/μ 0.66 against 0.76 (G4:86-87).
- **Correction:** rank per amplitude and arm on near-empty GPU-h, and offer "A + σ²=4 distance arm" to the owner.

**F10 — MINOR — disk figures**
- **Attacked:** R:311-313.
- **Evidence:**
  - At M = 100, σ² 2.25 and 4 write 112 and 95 MB per realization, not 45 (CS:204-205).
  - A writes 534 GB raw plus about 5 GB of Par2 snapshots, against 448 GB usable above the 100 GB guard. Without working pruning the job stops at about 84 % of the plan.
  - The coarse rungs write 66–86 GB/h, so hourly pruning leaves more than 60 GB of backlog (the engine's 30-minute pruning is fine).
  - B (397 GB) is the only variant that fits without pruning. `prune` has never run on a real campaign (EI:181).
- **Correction:** fix the figures, and make "reduce and prune every arm type in the smoke test" a launch gate.

**F11 — MINOR — critical-path details**
- R:332 "Sun 18:00 at the latest" ignores H13's 1 h of grace: the analysis cannot end before about 18:30–19:00.
- The spec sends a halted arm and a cost overrun to the owner (spec:124, 164). That needs a continuation job after the main one; not budgeted.
- The helper writes no heartbeat, so a death may not be seen for hours; "one 2 h resume" assumes immediate detection.
- H14's scope ("stage 1 of the single job") means the two-GPU path is first used at the production launch, not in the smoke test.
- "About three days early" (R:383) is about 2.2 days from the pessimistic launch.

**F12 — MINOR — the "practical ceiling of about 100 GPU-h"**
- **Attacked:** R:335-337.
- **Evidence:** it holds only with two GPUs for the full 30 h and a lane B that can be kept busy. With one GPU the ceiling is 83 / 77 / 71 / 64 planning GPU-h for launches at Tue 22:00 / Wed 06:00 / Wed 14:00 / Thu 00:00.
- **Correction:** state both ceilings.

**F13 — MINOR — small accounting points**
- B's Gate 5 n = 205 at σ²=4 is not on H15's coverage grid, so it would be analysed at 200.
- Two per-realization times in DC are not in the cost audit: 416x48 at 4 (4.0 s) and 3328x192_L208 (52 s).
- 1664x192 at 4 uses the single-realization J1 time (26.9 s) rather than the 8-realization J3 time (27.8 s).
- The helper drops to one GPU only after the full 2 h wait (HR:631-640). That 2 h is not in "59–70 h", and a second GPU freed later can never be added.

## Claims that survived

- DC's arm list matches H9 variant A exactly. Shared realizations are counted once:
  - production = Gate 5 consumer at 0.25 and 1;
  - the (24, 8) cell at 2.25;
  - the (12, 8) and (12, 16) cells at 4, the latter also the Gate 5 consumer at n = 250;
  - G7 reuses the Gate 5 Par2 arm as one cell (cell mean understated by 2 %, 0.04 GPU-h).
- Per-realization times match CS §1 or the M = 100 measurements. Totals 53.5 / 38.3 / 88.0 and the family split reproduce. With both GPUs near-empty, my engine-rule walls match DC to 0.1 h.
- The ×1.101 overhead is conservative once reduction runs in the background (CS:254). That roughly offsets the uncosted items: lane B waiting (about 3 GPU-h) and my estimate of solver iteration-cap hits at σ²=4 (about 3 GPU-h).
- The specified design does not fit and exceeds G10.
- A fits from the review's launch window: done Thu 14:24 to Fri 18:42 on two GPUs (with a resume), Sat 16:42 at worst on one GPU. It tolerates about one day of phase-2 slip.
- B: 38 GPU-h, Thursday.
- The capacity sums 144/136/128/118, the 19:00Z cutoff and the 03:00Z deadline are correct.
- The coverage run's 45 core-hours is calibrated on measured host seconds per cell (`gate5_coverage.md`:125-145).
- The memory thresholds (17000 and 30000 MiB) match the measured peaks.
- Reduction throughput keeps up with one lane.

Not checked: the statistical claims (power, SD ratios), and the engine and executor test suites were not run. The solver-cap cost is my estimate, and the simulator code was run inline and not saved (read-only).

Main risk: a colleague settling on lane A's GPU, or no 30 GB GPU when the job is due to start. The σ²=4 production arm can run only on a near-empty GPU, and under a binding cutoff the engine as written cuts priority-1 light arms first.

### Critical Files for Implementation
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/2026-11-02-sf43-pre-activation-review.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/decisions/2026-11-02-sf43-pre-activation-decisions.md
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/sf43-preact/docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts/design_cost.py
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/agent-a87320eaf2de12002/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43camp/scheduler.py
- /home/sesquerre/Projects/MacroFlow3D/.claude/worktrees/agent-a87320eaf2de12002/docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/sf43camp/lanes.py