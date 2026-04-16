# PR workflow runbook

Formal branch / worktree / PR flow for MacroFlow3D.

---

## 1. Workflow summary

```
create worktree → edit locally → validate locally → push → create PR → review → merge
```

## 2. Create a worktree

```bash
cd ~/src/MacroFlow3D
git worktree add -b <type>/<short-name> ~/.codex/worktrees/<short-name>
cd ~/src/MacroFlow3D/.codex/worktrees/<short-name>
```

Or use the helper:

```bash
scripts/create-worktree.sh <type>/<short-name>
```

Branch naming:

- `chore/` — tooling, docs, AGENTS, hooks, scripts
- `fix/` — bug fixes
- `feat/` — new capabilities
- `science/` — scientific or numerical changes
- `refactor/` — structural changes, no intended behavior change

## 3. Validate locally

Run the fixed validation loop:

```
configure → build → test → smoke
```

Minimum:

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

For scientific changes, add Tier B/C evals as needed. See `docs/validation/eval-tiers.md`.

## 4. Push and create PR

```bash
git push -u origin <branch-name>
gh pr create --fill
```

Or use the helper:

```bash
scripts/create-pr.sh
```

The PR description must include:

1. Scope (files and subsystems)
2. Commands run
3. What passed
4. Remaining risks
5. Files intentionally left untouched

The `.github/pull_request_template.md` provides the structure.

## 5. Review

- **High-autonomy changes** (docs, AGENTS, skills, presets, hooks, scripts): automated checks sufficient.
- **Scientific-core changes** (solver, operators, PSPTA, PETSc/SLEPc, macrodispersion): mandatory human review.

See `docs/runbooks/autonomy-policy.md`.

## 6. Merge

Only after all checks pass:

```bash
gh pr merge --squash --delete-branch
```

Never merge scientific-core PRs without human approval.

## 7. Clean up worktree

```bash
cd ~/src/MacroFlow3D
git worktree remove ~/worktrees/MacroFlow3D/<short-name>
```

## 8. Anti-patterns

- Do not push directly to `master` / `main`.
- Do not merge without running the validation loop.
- Do not mix multiple purposes in one branch.
- Do not merge "it compiles" changes for scientific code without evidence.
- Do not bypass `--no-verify` on commits that touch scientific core.
