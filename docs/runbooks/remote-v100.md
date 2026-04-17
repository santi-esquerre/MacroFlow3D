# Remote V100 runbook

This runbook defines the canonical remote workflow.

The server is used for:
- release builds,
- PETSc/SLEPc builds,
- heavy tests,
- profiling,
- production-like runs,
- larger scientific validation.

The server is **not** the primary edit surface.

---

## 1. Model

Source of truth:
- local WSL worktree

Execution surface:
- remote host `v100`

Transport mechanism:
- `scripts/remote`, which owns `rsync`, `ssh`, `tmux`, logs, status files, and polling.

Remote repo model:
- `~/MacroFlow3D` on `v100` is a synced execution mirror, not a Git checkout.
- Do not expect `.git/` to exist on the remote mirror.
- Do not run `git` commands there as part of the normal harness flow.
- `Par2_Core` must already be populated in the local worktree before sync; the remote mirror cannot materialize a missing submodule on its own.
- Remote-only PETSc/SLEPc trees under `src/external/petsc` and `src/external/slepc` are preserved across syncs and managed separately from the local worktree.

Canonical pattern:
1. edit locally,
2. validate lightly locally,
3. `scripts/remote sync`,
4. `scripts/remote exec -- ...` for one-shot remote work,
5. `scripts/remote run/status/tail/wait/cancel` for long jobs,
5. pull back logs/results if needed.

Do not handwrite ad hoc `ssh` / `tmux` / `rsync` command strings for normal work.
If you are bypassing `scripts/remote`, you are almost certainly taking the wrong path.

---

## 2. Remote repo layout

Recommended remote location:
```bash
~/MacroFlow3D
```

Recommended build directories:
```bash
~/MacroFlow3D/build/v100-release
~/MacroFlow3D/build/v100-petsc
~/MacroFlow3D/build/v100-prof
```

---

## 3. Canonical interface

Everything goes through one repo-local entry point:

```bash
scripts/remote sync
scripts/remote exec -- "<shell-command>"
scripts/remote run <job> -- "<shell-command>"
scripts/remote status <job>
scripts/remote tail <job>
scripts/remote wait <job>
scripts/remote cancel <job>
```

Remote defaults live in:

```bash
scripts/remote.env
```

That file defines:
- remote host alias
- remote repo path
- remote state root
- log / status / command / launcher directories
- tmux session prefix
- rsync exclusions
- polling interval
- retry behavior

## 4. Sync

### Canonical sync
```bash
scripts/remote sync
```

### Verify remote tree
```bash
scripts/remote exec -- "pwd && ls"
```

## 5. Remote configure/build

### 4.1 Release build without PETSc
```bash
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
```

### 4.2 Release build with PETSc/SLEPc
```bash
scripts/remote exec -- "cmake --preset v100-petsc && cmake --build build/v100-petsc -j"
```

## 6. Remote tests

### Release test pass
```bash
scripts/remote exec -- "ctest --test-dir build/v100-release --output-on-failure"
```

### PETSc/SLEPc targeted tests
```bash
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"
```

```bash
scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
```

## 7. Remote runs

### Small PSPTA smoke
```bash
scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"
scripts/remote wait pspta-small
```

### PSPTA production-like config
```bash
scripts/remote run pspta-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_pspta.yaml"
scripts/remote tail pspta-prod
scripts/remote wait pspta-prod
```

### Baseline Par2 config
```bash
scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"
scripts/remote wait par2-prod
```

### Cancel a long job

```bash
scripts/remote cancel pspta-prod
```

## 8. Profiling mode

When profiling:
- use a build with profiling/NVTX enabled,
- keep the config fixed,
- record exact commit and command line,
- do not change multiple variables at once.

Example profiling build:
```bash
scripts/remote exec -- "cmake --preset v100-prof && cmake --build build/v100-prof -j"
```

## 9. Result handling

For any meaningful remote run, record:
- commit hash
- build directory
- binary used
- config file used
- exact command
- relevant output path

If a run changes scientific conclusions, preserve the output directory and summarize it in:
- `docs/experiments/`
- or `docs/plans/`
- or the PR description

## 10. Failure triage

### Configure failure
Check:
- CUDA version / compiler
- `CMAKE_CUDA_ARCHITECTURES`
- PETSc/SLEPc paths
- missing `ninja`

### Build failure
Check:
- compiler output
- architecture mismatch
- stale build dir

### Test failure
Check:
- regression versus local
- environment mismatch
- accidental path/config drift

### Scientific output mismatch
Do not guess.
Compare:
- local config
- remote config
- build flags
- commit hash
- output manifests

## 11. Anti-patterns

Avoid:
- editing files directly on `v100`
- treating the remote tree as the canonical repo state
- running production-like experiments from unvalidated local changes
- overwriting remote outputs without preserving metadata
- bypassing `scripts/remote` with raw `ssh`, `tmux`, or `rsync`
