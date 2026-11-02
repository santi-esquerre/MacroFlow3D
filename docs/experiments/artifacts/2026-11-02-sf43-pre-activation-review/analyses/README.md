# Analyses and adversarial reviews behind the SF-43 pre-activation review

Verbatim final reports of the read-only agents of 2026-11-02 (inputs to
[`../../../2026-11-02-sf43-pre-activation-review.md`](../../../2026-11-02-sf43-pre-activation-review.md); the literature reports are in
[`../bibliography.md`](../bibliography.md)). They are evidence of how each number was obtained, not authoritative text: where the review
and a report differ, the review (and the decision record) states the corrected value, and section 7 of the review lists what the
adversarial reviews changed. Paths in the reports refer to the session's read-only snapshot of `master` (`ef87a1bc`) or to the
orchestrator's runtime notes (`.claude/orchestration/SF-43/phase1/agents/`, not versioned); commit hashes of prepared tooling
(`bfddba78`, `171add34`, `3b073d8a`, `f573edec`, `f4ebc536`, `8905e0f0`) are worktree commits not yet on any branch.

| file | content |
|---|---|
| `cost_schedule.md` | per-realization costs, memory classes, the SF-42 projection reproduced, displacement-basis variants, two-stage schedules, critical path, disk |
| `gate4_power.md` | Gate 4 budget components from SF-39 data, robust-limit simulations, MDE, `E_h` with production finest, the `numerical` fraction |
| `gate5_coverage.md` | relevance of the `stress` family, Gate 5 options (Fieller form), restricted coverage cost, sizing, G7 effect sizes, cutoff validity |
| `exec_inventory.md` | the helper, runner options per arm, `run_sf39.sh` reuse, analysis tooling, disk, effort |
| `review_*.md` | the five adversarial reviews of the draft (statistics, numerics, physics and citations, executability, deadline and cost) |

The scripts that reproduce the quantitative tables are in [`../scripts/`](../scripts/README.md).
