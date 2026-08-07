# PR workflow runbook

## Purpose

Formal branch / worktree / PR lifecycle for MacroFlow3D.

The repository supports two execution styles:

1. ordinary manual/single-agent work;
2. orchestrated increments executed by Fable with isolated Sonnet workers and a
   single isolated integrator.

Both end in a pull request. **Agents never merge pull requests.**

---

## 1. Workflow summary

### Generic work

```text
worktree -> edit -> validate -> audit/review as required -> push -> PR -> human merge
```

### Orchestrated increment

```text
Fable UNDERSTAND/PLAN
  -> Sonnet worker worktrees
  -> Fable AUDIT / corrective DAG
  -> Sonnet integration worktree
  -> Fable FINAL_AUDIT
  -> PR
  -> human review when required
  -> closure-only metadata finalization
  -> human merge
  -> default-branch advancement
```

---

## 2. Worktrees

### Generic/manual work

Use the existing helper when a task is not running through the Claude
orchestrator:

```bash
cd ~/src/MacroFlow3D
scripts/create-worktree.sh <type>/<short-name>
cd .agents/worktrees/<short-name>
```

Equivalent direct Git command:

```bash
git worktree add -b <type>/<short-name> .agents/worktrees/<short-name>
```

Branch naming:

- `chore/` — tooling, docs, agent harness, hooks, scripts;
- `fix/` — bug fixes;
- `feat/` — new capabilities;
- `science/` — scientific or numerical changes;
- `refactor/` — structural changes with no intended behavior change.

### Claude-orchestrated increments

Do **not** force every DAG node into the single manual worktree above.

- `increment-worker` uses native `isolation: worktree`;
- `increment-integrator` uses native `isolation: worktree`;
- each source-writing node therefore receives a separate Claude-managed
  worktree;
- the orchestrator remains the coordination/audit authority and must not
  implement substantial source changes in the control checkout;
- dependent workers receive exact approved predecessor commits and incorporate
  them explicitly.

The increment specification's `Branch` field names the final delivery branch,
not every temporary worker branch. Its `Worktree` field may retain a historical
manual path for old increments; new increments should document that execution
uses Claude-managed per-node worktrees.

---

## 3. Validate locally

Minimum local validation:

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
```

Run the relevant smoke command when the increment or acceptance gate requires
it.

Scientific changes add Tier B/C, Gate 3A/4, V100, convergence, or experiment
validation as specified by:

- the increment specification;
- `docs/validation/eval-tiers.md`;
- `docs/validation/acceptance-gates.md`.

A worker's successful test report is not final evidence. The orchestrator
independently audits and reruns the validation required for acceptance.

---

## 4. Publish the PR

Only the main orchestrator may publish an orchestrated increment.

Before publication verify:

- the exact final integrated commit is the one Fable audited;
- the working/resulting branch contains no unapproved source changes;
- all required checks and acceptance evidence are recorded;
- the increment state is appropriate for its review class.

Typical commands:

```bash
git push -u origin <branch-name>
gh pr create --fill
```

or the existing helper when compatible:

```bash
scripts/create-pr.sh
```

The PR description must include:

1. Goal / purpose;
2. scope and intentionally untouched areas;
3. DAG / delegated task summary for orchestrated increments;
4. implementation and integration summary;
5. exact commands run;
6. what passed and measured evidence;
7. acceptance criteria / gates;
8. corrective cycles or rejected evidence when material;
9. remaining risks.

Workers and the integrator must never push or create a PR.

---

## 5. Review classes

### High-autonomy

After Fable final audit and required automated checks, the delivery branch may
already contain final closure metadata (`State: done`, dashboard advancement).
The orchestrator publishes it as ready to merge. A human still performs merge.

### Mandatory human review

For scientific-core increments:

1. Fable final-audits the integrated source result.
2. The delivery branch records `State: awaiting_review`; `NEXT` remains on the
   current increment.
3. Fable pushes/updates the PR.
4. A human reviews the exact source-bearing PR head.
5. If changes are requested, return to corrective DAG -> integration -> final
   audit -> review.
6. When the human explicitly approves, resume the same PR.
7. Fable may add **only a closure metadata commit**:
   - complete the versioned checklist;
   - set the increment to `done`;
   - record PR/final audited commit and approval evidence;
   - append the bitacora closure entry;
   - check the master checklist entry;
   - advance `NEXT` to the first remaining pending increment;
   - clear/change the active runtime goal as required;
   - run `bash scripts/hooks/check-lester-increments.sh`.
8. Push that metadata-only commit to the same PR branch.
9. Human merges the PR.

A source/test/config change after approval invalidates this closure-only path and
requires renewed audit/review.

---

## 6. Merge — human only

Agents must never execute the merge.

After all checks and required review/closure steps pass, a human may use the UI
or, manually:

```bash
gh pr merge --squash --delete-branch
```

For Lester increments, the next increment remains blocked until the merged
closure state is visible on the default branch.

---

## 7. Post-merge verification

Before starting a dependent increment:

```bash
git fetch origin
git switch <default-branch>
git pull --ff-only
bash scripts/hooks/check-lester-increments.sh
```

Verify:

- predecessor increment is `done`;
- its dashboard entry is checked;
- `Last completed increment` is correct;
- `NEXT` selects the dependent increment;
- no other Lester increment is nonterminal.

Do not infer advancement from the existence of code alone.

---

## 8. Exceptional closure repair

If an implementation PR was merged while its increment remained
`awaiting_review`, follow the closure-repair procedure in
`docs/runbooks/autonomy-policy.md` and
`docs/runbooks/lester-increment-workflow.md`.

This repair is metadata-only and itself goes through a PR. Do not reopen or
rewrite the accepted scientific implementation.

---

## 9. Cleanup

Claude-managed worktrees are normally cleaned up by Claude Code when
appropriate. For a manually-created worktree:

```bash
cd ~/src/MacroFlow3D
git worktree remove .agents/worktrees/<short-name>
```

Do not delete a worktree that contains unrecorded evidence or uncommitted work.

---

## 10. Anti-patterns

- Do not push directly to `master` / `main`.
- Do not let workers or the integrator publish.
- Do not let any agent merge a PR.
- Do not merge without the required validation loop.
- Do not merge a human-review increment while its versioned state is still
  `awaiting_review`; finalize closure metadata first.
- Do not mix unrelated purposes in one branch.
- Do not accept "it compiles" as scientific validation.
- Do not use `--no-verify` to bypass scientific-core commit checks.
