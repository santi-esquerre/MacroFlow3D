# SF-00 — Increment harness

- State: `validating`
- Goal: `Establecer el harness incremental versionado del solver de Lester.`
- Depends on: `none`
- Unlocks: `SF-01`
- Branch: `chore/lester-increment-harness`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-increment-harness`
- Acceptance gate: `Gate 0`
- Human review: `not required`
- Owner: `Codex`
- Started: `2026-08-04T18:56Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `4243495 (base)`

## Scientific or engineering intent

Turn the solver design into durable execution state so future agents can work
one reviewable hypothesis at a time without inferring scope or completion.

## Preconditions

- The repository audit and detailed solver plan have been completed.
- The active Lester plan, theory note, and acceptance gates exist.

## In scope

- Add the sequential dashboard, 29 increment specifications, template, runbook,
  agent routing, and a structural consistency checker.

## Out of scope

- Solver, operator, CUDA, flow, stochastic, configuration, or runtime behavior.
- Starting SF-01 or creating scientific source files.

## Files and symbols

- Update `AGENTS.md`, `docs/AGENTS.md`, `docs/plans/README.md`, and the active
  Lester plan.
- Add `docs/plans/active/lester-eq14/increments/`,
  `docs/plans/increment-template.md`, and
  `docs/runbooks/lester-increment-workflow.md`.
- Add `scripts/hooks/check-lester-increments.sh` and its pre-commit entry.

## Implementation specification

1. Preserve locked mathematical, diagnostic, continuation, and memory choices
   in the dashboard.
2. Give every increment an exact Goal, linear dependency, branch, gate,
   implementation boundary, validation, completion checklist, and bitácora.
3. Make the checker reject inconsistent order, state, checklist, dependency,
   link, or `NEXT` information.
4. Finish this branch with SF-00 checked and `NEXT: SF-01`.

## Expected numerical effect

None.  This increment changes only agent workflow and documentation.

## Validation commands

```bash
bash scripts/hooks/check-lester-increments.sh
bash scripts/hooks/check-required-docs.sh
pre-commit run --all-files
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- All 29 increment files are structurally valid.
- Only SF-00 is active while work is in progress.
- No tracked scientific source or runtime config changes.
- Tier A completes without a regression.

## Regression surface

- Agent routing, pre-commit execution, and links to the authoritative plan.

## Failure and rollback policy

- Leave SF-00 active if the checker or Tier A fails.
- Correct the harness locally; do not weaken validation to make malformed state
  pass.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Dashboard contains locked decisions and a complete master checklist.
- [ ] SF-00 through SF-28 specifications exist and pass the checker.
- [ ] Template and execution runbook are complete.
- [ ] Root and docs agent routing point to the increment workflow.
- [ ] Pre-commit runs the increment checker.
- [ ] Tier A and documentation checks pass.
- [ ] Final evidence, commit, and PR state are recorded below.
- [ ] Dashboard marks SF-00 complete and selects SF-01.
<!-- completion-checklist:end -->

## Advancement rule

After this completion state is merged, SF-01 may create the independent
reference-test harness.  No scientific implementation may begin before then.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-04T18:56Z | `4243495`, active | Created the required worktree and persistent goal. | Goal text matches this specification; main checkout was clean. | Create and validate the harness documents. |
| 2026-08-04T19:02Z | working tree | Added the dashboard, 29 specifications, template, runbook, routing, and structural hook. | `check-lester-increments.sh`, required-docs check, `git diff --check`, ShellCheck, and YAML parsing passed. | Run Tier A. |
| 2026-08-04T19:05Z | working tree | Canonical local build failed in unchanged vendored code. | GNU 16/15 exposed missing `<cstdint>` in yaml-cpp; a compatibility build progressed but CUDA 13.3 then failed on removed `cub::Sum` usage in Par2. No external source was modified. | Validate with the canonical V100 toolchain. |
| 2026-08-04T19:12Z | validating | Synchronized through `scripts/remote`; V100 configure, build, CTest, and PSPTA-small smoke passed. | V100 build completed 85 targets; `operator_tests` passed; pipeline completed 500 steps with zero Newton failures. Sync also reported pre-existing undeletable PETSc/SLEPc incomplete directories. | Commit the implementation and close SF-00. |
