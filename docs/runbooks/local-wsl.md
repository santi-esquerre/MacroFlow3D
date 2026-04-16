# Local WSL runbook

This runbook defines the canonical local workflow.

Local WSL is the source of truth for:
- editing,
- code reading,
- planning,
- light validation,
- preparing sync to the server.

The remote V100 machine is for heavy build/run/measure, not as the primary edit surface.

---

## 1. Preconditions

Assumed:
- repository lives inside WSL, not under `/mnt/c/...`
- Codex App is configured to use WSL
- work is done in Git worktrees
- the server alias `v100` already exists in `~/.ssh/config`

Suggested layout:
```bash
~/src/MacroFlow3D
~/src/MacroFlow3D/.agents/worktrees/<branch-name>
```

---

## 2. Worktree-first workflow

### Create a new worktree
```bash
cd ~/src/MacroFlow3D
git worktree add -b chore/example .agents/worktrees/example
```

### Enter it
```bash
cd ~/src/MacroFlow3D/.agents/worktrees/example
```

### Open that worktree in Codex App
Do not open the main checkout when the task involves edits.

---

## 3. Planning workflow

For any non-trivial task:

1. start a new thread in the worktree,
2. use Plan mode,
3. ask for:
   - relevant files,
   - change strategy,
   - validation commands,
   - risks,
4. only then allow edits.

Recommended starter prompt:
```text
Read the repository context first. Do not edit yet.

Before making any changes:
1. identify the relevant files and subsystems,
2. explain the implementation strategy,
3. list the exact commands you will run,
4. list the validation steps you will use,
5. list the main risks and regression points.

Do not start editing until the plan is complete.
```

---

## 4. Local configure/build/test

### 4.1 Debug configure (no PETSc)
```bash
cmake -S . -B build/wsl-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON \
  -DMACROFLOW3D_ENABLE_PROFILING=OFF \
  -DMACROFLOW3D_ENABLE_NVTX=OFF \
  -DMACROFLOW3D_ENABLE_PETSC=OFF
```

Adjust `CMAKE_CUDA_ARCHITECTURES` if your local GPU is not `86`.

### 4.2 Build
```bash
cmake --build build/wsl-debug -j
```

### 4.3 Run all tests available in that build
```bash
ctest --test-dir build/wsl-debug --output-on-failure
```

### 4.4 Run the small PSPTA smoke case
```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

---

## 5. High-signal targeted checks

### Operator tests
```bash
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
```

### Direct execution
```bash
./build/wsl-debug/run_operator_tests
```

Use local WSL primarily for:
- compile errors,
- obvious logic errors,
- quick smoke,
- documentation and script iteration.

Do not use it for final performance or scientific claims.

---

## 6. Local hygiene checklist before sync

Before syncing to `v100`, make sure:

- `git status` is clean enough to understand
- the worktree contains only intended changes
- the local build passed
- the runbook and AGENTS changes are updated if behavior changed

Useful commands:
```bash
git status
git diff --stat
git log --oneline --decorate -n 5
```

---

## 7. Sync to server

Preferred:
```bash
scripts/rsync_to_v100.sh
```

Then run:
```bash
scripts/remote_build_and_test.sh
```

If you only changed documentation or scripts, remote sync/build may be unnecessary.

---

## 8. Local tasks that should stay local

Keep these local:
- planning
- code review
- doc authoring
- AGENTS / runbook updates
- worktree management
- prompt/skill iteration
- small smoke validation

---

## 9. Anti-patterns

Avoid:
- editing the repo under `/mnt/c/...`
- doing all work in the main checkout
- pushing large unvalidated changes straight to the server
- treating local performance as representative of V100
- mixing multiple scientific hypotheses in one worktree
