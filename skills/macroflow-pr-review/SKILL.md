# macroflow-pr-review

Branch / worktree / PR workflow for MacroFlow3D.

## When to use

Use this skill when creating, reviewing, or merging pull requests.

## Workflow overview

1. Work in a **git worktree** (not the main checkout).
2. Push the worktree branch to origin.
3. Create a PR with `gh pr create`.
4. Merge only after checks pass with `gh pr merge`.

## Branch/worktree hygiene

### Create worktree
```bash
cd ~/src/MacroFlow3D
git worktree add -b <type>/<short-name> ~/worktrees/MacroFlow3D/<short-name>
cd ~/worktrees/MacroFlow3D/<short-name>
```

Branch name conventions:
- `chore/` — tooling, docs, AGENTS, hooks, scripts
- `fix/` — bug fixes
- `feat/` — new capabilities
- `science/` — scientific or numerical changes
- `refactor/` — structural changes with no intended behavior change

### Keep worktree clean
```bash
git status
git diff --stat
git log --oneline --decorate -n 5
```

One purpose per branch. Do not mix solver changes, transport changes, and tooling changes.

## Creating a PR

### Push branch
```bash
git push -u origin <branch-name>
```

### Create PR
```bash
gh pr create --fill
```

Or with explicit fields:
```bash
gh pr create \
  --title "chore: add pre-commit hooks" \
  --body-file /dev/stdin <<'EOF'
## What changed
...

## Commands run
...

## What passed
...

## Remaining risks
...
EOF
```

Helper script:
```bash
scripts/create-pr.sh
```

### PR description must include

1. **Scope** — what files/subsystems changed
2. **Commands run** — exact configure/build/test/smoke commands
3. **What passed** — test results, outputs checked
4. **Remaining risks** — open questions, known limitations
5. **Files intentionally left untouched** — if relevant

For scientific changes, also include the scientific change report template from `docs/validation/acceptance-gates.md`.

## What must block merge

- [ ] Build fails
- [ ] Tests fail
- [ ] Required gate evidence missing (see `docs/validation/acceptance-gates.md`)
- [ ] Scientific-core change without human review
- [ ] PSPTA/invariant/eigensolver/refinement change that does not align with the current execution phase (see `docs/plans/active/pspta-execution-plan.md`)
- [ ] Silent behavior change in configs
- [ ] Diagnostics weakened without explanation
- [ ] Transverse macrodispersion claimed as physical without control test

## Merge

Only after all checks pass:

```bash
gh pr merge --squash --delete-branch
```

Or if merge commits are preferred:
```bash
gh pr merge --merge --delete-branch
```

## Autonomy rules for PRs

**High autonomy** (agent can create and merge after automated checks):
- docs, AGENTS, skills, presets, hooks, automation scripts

**Mandatory human review** (agent must NOT merge):
- solver, operators, interpolation
- PSPTA tracking
- PETSc/SLEPc integration
- macrodispersion evals

See `docs/runbooks/autonomy-policy.md`.

## Checklist for PR author

- [ ] Single-purpose branch
- [ ] Local Tier A eval passed (configure → build → test → smoke)
- [ ] Correct gate identified and evidence provided
- [ ] PR description is complete
- [ ] Docs updated if behavior changed
- [ ] No files modified beyond stated scope

## Checklist for PR reviewer

- [ ] Change scope matches description
- [ ] Correct gate was used
- [ ] Evidence is present and credible
- [ ] No silent numerical changes
- [ ] No weakened diagnostics
- [ ] Remaining risks are stated

## Related

- `docs/runbooks/pr-workflow.md` — detailed PR workflow
- `docs/runbooks/worktree-pattern.md` — parallel worktree pattern
- `docs/runbooks/autonomy-policy.md` — what needs human review
- `.github/pull_request_template.md` — PR template
- `docs/validation/acceptance-gates.md` — gate definitions
