# Parallel worktree pattern

Support for multiple concurrent work streams via git worktrees.

---

## Concept

Use separate worktrees for independent work streams. Each worktree:

- has its own branch
- has its own build directory
- can be synced to `v100` independently
- avoids branch-switching conflicts

## Recommended layout

```
~/src/MacroFlow3D/                         # main checkout (reference)
~/src/MacroFlow3D/.agents/worktrees/
├── tooling/         # chore/tooling-*     — build, hooks, CI, scripts
├── docs/            # chore/docs-*        — documentation, AGENTS, runbooks
├── pspta/           # science/pspta-*     — PSPTA transport research
├── profiling/       # chore/profiling-*   — perf measurement, NVTX, benchmarks
└── petsc/           # feat/petsc-*        — PETSc/SLEPc integration work
```

## Creating worktrees

### Manual

```bash
cd ~/src/MacroFlow3D
git worktree add -b chore/tooling-sccache .agents/worktrees/tooling
```

### Helper script

```bash
scripts/create-worktree.sh chore/tooling-sccache
```

Override base directory:

```bash
MACROFLOW3D_WORKTREE_BASE=~/other/path scripts/create-worktree.sh chore/docs-update
```

## Working in a worktree

```bash
cd ~/src/MacroFlow3D/.agents/worktrees/tooling

# Normal git operations work
git status
git add -A
git commit -m "chore: add sccache support"

# Build works normally (separate build/ dir per worktree)
cmake --preset wsl-debug
cmake --build build/wsl-debug -j

# Sync to remote works normally
scripts/rsync_to_v100.sh
```

## Typical concurrent worktrees

| Worktree | Branch pattern | Focus | Review policy |
|----------|---------------|-------|---------------|
| tooling | `chore/tooling-*` | Build, CI, hooks, scripts | High autonomy |
| docs | `chore/docs-*` | Documentation, AGENTS, runbooks | High autonomy |
| pspta | `science/pspta-*` | PSPTA research, invariants | Human review required |
| profiling | `chore/profiling-*` | Perf, benchmarks, NVTX | High autonomy |
| petsc | `feat/petsc-*` | PETSc/SLEPc integration | Human review required |

## Rules

- One purpose per worktree/branch.
- Do not mix solver, transport, and tooling changes.
- Keep worktrees clean: `git status` should be understandable.
- Delete worktrees after merge: `git worktree remove <path>`.

## Listing worktrees

```bash
git worktree list
```

## Cleaning up

```bash
# Remove a specific worktree
cd ~/src/MacroFlow3D
git worktree remove .agents/worktrees/tooling

# Prune stale worktree refs
git worktree prune
```

## Related

- `docs/runbooks/local-wsl.md`
- `docs/runbooks/pr-workflow.md`
- `scripts/create-worktree.sh`
