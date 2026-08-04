# Lester increment workflow

## Purpose

Use this runbook for every increment in
`docs/plans/active/lester-eq14-streamfunction-solver-plan.md`.  It makes the
Markdown state, persistent agent goal, worktree, evidence, and review lifecycle
agree.  It does not authorize working ahead of the dashboard's `NEXT` pointer.

## 1. Select the increment

- Read the dashboard and the complete specification linked by `NEXT`.
- Verify every `Depends on` increment is `done` on the default branch.
- Run `bash scripts/hooks/check-lester-increments.sh`.
- Stop if the checker or the dependency test fails.

## 2. Create one worktree and one goal

Create the exact branch named by the increment:

```bash
cd ~/src/MacroFlow3D
scripts/create-worktree.sh <branch-from-increment>
cd .agents/worktrees/<short-name>
```

Create one persistent runtime goal using the exact `Goal` sentence.  Codex
agents use `create_goal` and do not set a token budget unless the user explicitly
requested one.  A runtime without persistent goals records
`runtime goal unavailable` in the bitácora and still maintains the Markdown
goal and state.

In the branch:

- change `State` from `pending` to `active`;
- record owner, start time, branch/worktree, and the first bitácora entry;
- optionally open a draft PR so interrupted work remains visible.

There may be only one unfinished Lester runtime goal and one nonterminal Lester
increment at a time.

## 3. Work within the increment boundary

- Implement only the `In scope` section.
- Do not silently solve future increments.
- Append discoveries, failed hypotheses, commands, measurements, and decisions
  to the bitácora as they occur.
- Link large run evidence from `docs/experiments/`; do not paste full logs.
- When an observation invalidates a locked dashboard decision, stop and create
  a narrowly scoped decision/harness revision instead of changing the design
  silently.

Use the states as follows:

- `active`: implementation or investigation is underway;
- `validating`: implementation is frozen except for defects exposed by checks;
- `awaiting_review`: all checks pass and mandatory review is pending;
- `blocked`: progress requires a new decision or external change;
- `done`: every completion check is satisfied in the branch.

## 4. Validate and record evidence

Run the exact commands in the increment plus the fixed baseline:

```bash
bash scripts/hooks/check-lester-increments.sh
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
```

Scientific increments must also run the assigned acceptance gate and record:

- exact commit and build directory;
- exact config, seed, grid, covariance, and continuation values;
- residual and physical metrics required by Gate 3A;
- interpretation of failures and regularization dependence;
- local versus remote hardware.

Do not check a validation item for a command that was not run.  A failure keeps
the increment active or blocked and must be logged.

## 5. Close the branch

After validation and required review:

1. Complete every item between the completion-checklist markers.
2. Append a final bitácora row with commands, result, commit, PR, and residual
   risks.
3. Set the increment state to `done`.
4. Check its entry in the master checklist.
5. Set `NEXT` to the first remaining pending increment.
6. Run the harness checker again.
7. Merge only under `docs/runbooks/pr-workflow.md` and the autonomy policy.

The default branch remains authoritative: do not start the next increment until
the completion commit is visible there.  After merge, mark the runtime goal
complete.  If human review or merge is pending, leave the goal unfinished and
do not advance.

## 6. Resume interrupted work

- Read the runtime goal, increment metadata, and entire bitácora.
- Inspect the worktree and PR rather than recreating work.
- Re-run the last inconclusive command if its artifacts are unavailable.
- Continue from the recorded `Next action`.

Never infer completion from code presence alone; completion comes from the
versioned checklist and evidence.
