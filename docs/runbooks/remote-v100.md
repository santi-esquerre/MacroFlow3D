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

## 0. Policy: long-duration computation is always a detached V100 job

**Any computation whose duration is not short and bounded must run on V100 as
a `scripts/remote run <job>` detached background job, polled with
`scripts/remote wait <job>` / `scripts/remote tail <job>`. It must never run
inside a local WSL worktree (including a Claude Code subagent's isolated
worktree), and it must never run as a long blocking `scripts/remote exec`
call.**

This explicitly includes, but is not limited to:

- a full or near-full `ctest` run (as opposed to one fast `-R`-filtered case);
- any test binary that internally sweeps many cases (operator tests,
  streamfunction Picard/Anderson/continuation ladders, mesh refinement
  studies, etc.);
- production-like pipeline runs, ensemble runs, benchmarks, and profiling
  runs;
- PETSc/SLEPc builds and their test suites.

`scripts/remote exec` is reserved for short, bounded steps only: configure, a
single compile, a quick single-command smoke check — work that reliably
finishes in roughly under a minute. If a command's duration is unknown or
could plausibly run longer, treat it as long-duration and use `scripts/remote
run` + `scripts/remote wait`, not `exec`.

Rationale: local WSL has no GPU-equivalent execution path for CUDA test
binaries at V100 scale; heavy test/production runs there are both slower and
not representative. Claude Code subagents (`increment-worker`,
`increment-integrator`) must dispatch their heavy validation commands to V100
through this mechanism rather than executing them synchronously inside their
own isolated worktree.

### Concurrency: the remote mirror is shared, single-flight state

`REMOTE_REPO_DIR` (default `~/MacroFlow3D`, see `scripts/remote.env`) is one
shared execution surface — not one mirror per branch, worktree, or agent. A
`scripts/remote sync` overwrites the tree in place; running it while another
job is still executing on V100 corrupts that job's build/source state, and two
concurrent heavy jobs against the same build directory race each other.

Do not run `scripts/remote sync` or launch a new `scripts/remote run` job
while another job for the same increment/session is still `RUNNING`
(`scripts/remote status <job>`). If an orchestrated DAG has two nodes that
both require remote V100 execution, serialize them even if their local Git
write scopes would otherwise allow them to run in parallel — remote V100 is
shared external state under the DAG parallelism rule.

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

Full or multi-case test runs are long-duration computation (see Section 0):
launch them detached and wait, do not run them as a blocking `exec` call.

### Release test pass (detached)
```bash
scripts/remote run ctest-release -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-release
```

### PETSc/SLEPc targeted tests (detached)
```bash
scripts/remote run ctest-petsc-smoke -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"
scripts/remote wait ctest-petsc-smoke
```

```bash
scripts/remote run ctest-slepc-eigensolver -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
scripts/remote wait ctest-slepc-eigensolver
```

A single, genuinely fast, narrowly `-R`-filtered case may still use `exec` if
it reliably completes in well under a minute; when in doubt, use `run` +
`wait`.

## 7. Remote runs

### Legacy PSPTA smoke
```bash
scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"
scripts/remote wait pspta-small
```

### Legacy PSPTA production-like config
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
- running a full/heavy `ctest` suite, a production run, an ensemble, or a
  benchmark locally in WSL instead of as a detached V100 job
- running a long-duration remote command with blocking `scripts/remote exec`
  instead of `scripts/remote run` + `scripts/remote wait`
- running `scripts/remote sync` or a new `scripts/remote run` job while another
  job is still `RUNNING` against the same shared remote mirror
