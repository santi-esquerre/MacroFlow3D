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

### Concurrency: one increment, one mirror; one job per GPU

The V100 host has two GPUs and is shared by every agent. `scripts/remote`
isolates concurrent work in two ways:

1. **Per-increment mirrors.** `scripts/remote --increment SF-NN <subcommand>`
   (or `REMOTE_INCREMENT=SF-NN` in the environment) selects a private mirror
   `~/MacroFlow3D-SF-NN`, a private state root
   `~/.macroflow3d-remote/macroflow3d-SF-NN` (logs, status, commands,
   launchers) and the tmux prefix `macroflow3d-SF-NN`. A `sync` for one
   increment never touches another increment's tree or build directories.
   Without an id the legacy shared mirror `~/MacroFlow3D`, state root
   `~/.macroflow3d-remote/macroflow3d` and prefix `macroflow3d` are used,
   exactly as before. Explicit `REMOTE_REPO_DIR` / `REMOTE_STATE_ROOT` /
   `REMOTE_SESSION_PREFIX` overrides still win over the derived paths.
2. **Host-wide GPU locks.** Every `scripts/remote run` job takes an exclusive
   `flock` on `~/.macroflow3d-remote/gpu-<n>.lock` (n = 0 or 1) for its whole
   lifetime and exports `CUDA_VISIBLE_DEVICES=<n>` to the command. The locks are
   shared by all mirrors (including the legacy one), so two jobs can never
   share a GPU, and at most two `run` jobs execute on the host at once.

The rule is therefore:

- **one increment, one mirror** — always pass `--increment SF-NN` (or set
  `REMOTE_INCREMENT=SF-NN`) for increment work; never share a mirror between
  increments;
- **one job per GPU at a time** — enforced by the lock;
- **two jobs maximum across the host** — enforced by the two locks.

Within one increment the mirror is still single-flight: do not
`scripts/remote --increment SF-NN sync` while a job of that same increment is
`running` or `waiting_gpu`, because the sync overwrites the tree the job
executes against. Two DAG nodes of the same increment that both need remote
execution must still be serialized.

`scripts/remote exec` takes **no** GPU lock; it is only for short, bounded
configure/compile/smoke steps.

#### GPU selection and waiting

```bash
scripts/remote --increment SF-NN run <job> -- "<cmd>"            # --gpu auto (default): first free GPU
scripts/remote --increment SF-NN run <job> --gpu 1 -- "<cmd>"    # pin GPU 1
REMOTE_GPU_WAIT=1800 scripts/remote --increment SF-NN run <job> -- "<cmd>"
```

- `REMOTE_GPU_WAIT=<seconds>` (default `0`) is how long a job waits for a free
  GPU. While waiting, its state is `waiting_gpu` (`status` shows it;
  `wait` keeps polling through it).
- With `REMOTE_GPU_WAIT=0`, or when the wait expires, the job does not run:
  its state becomes `failed` with **exit code 75**, and its log lists the
  current holders of both GPUs (job, session, increment, start time, whether
  the session is alive). `scripts/remote wait <job>` returns 75.
- `status` reports the assigned GPU in the `gpu:` line and the mirror in the
  `working_dir:` line; the log records `GPU=<n> (CUDA_VISIBLE_DEVICES=<n>, ...)`.
- `cancel` terminates the job's whole process group (launcher, command and
  children), so its GPU lock is released immediately. Processes that detach
  themselves with `setsid`/`nohup &` escape the group and keep the lock until
  they exit.
- The holder files `~/.macroflow3d-remote/gpu-<n>.holder` are informational
  only; the kernel `flock` is the authority. A stale holder file never blocks
  a new job.

#### Mirror lifecycle

```bash
scripts/remote list-mirrors           # mirrors, state roots, current GPU holders
scripts/remote remove-mirror SF-NN    # delete ~/MacroFlow3D-SF-NN and its state root
```

- `remove-mirror` refuses (exit 1) while any tmux session with prefix
  `macroflow3d-SF-NN-` is alive. It never removes the shared `~/MacroFlow3D`.
- Remove an increment's mirror once its PR is merged (copy out any logs or
  outputs that the increment record cites first).
- A new per-increment mirror contains only what `sync` copies: it has **no**
  remote-only PETSc/SLEPc trees (`src/external/petsc`, `src/external/slepc`)
  and no build directories. `v100-release` builds work after a normal
  configure/build; `v100-petsc` builds need the shared mirror (no
  `--increment`) or a manual copy of the PETSc/SLEPc trees into the
  per-increment mirror.

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
scripts/remote [--increment <id>] sync
scripts/remote [--increment <id>] exec -- "<shell-command>"
scripts/remote [--increment <id>] run <job> [--gpu auto|0|1] -- "<shell-command>"
scripts/remote [--increment <id>] status <job>
scripts/remote [--increment <id>] tail <job> [--lines N] [--no-follow]
scripts/remote [--increment <id>] wait <job> [--interval SEC]
scripts/remote [--increment <id>] cancel <job>
scripts/remote list-mirrors
scripts/remote remove-mirror <id>
```

`--increment <id>` (also `--increment=<id>`, or env `REMOTE_INCREMENT=<id>`)
is a global option and must come **before** the subcommand. Use the same id
for every call of one increment (sync, exec, run, status, tail, wait, cancel);
job names are scoped to that id. Job states are `waiting_gpu`, `running`,
`succeeded`, `failed` (exit 75 = no GPU available), `cancelled` (130) and
`unknown`. See Section 0 for GPU locking and mirror lifecycle.

Remote defaults live in:

```bash
scripts/remote.env
```

That file defines:
- remote host alias
- remote repo path (derived per increment when an id is given)
- remote state root (derived per increment when an id is given)
- GPU lock root (`REMOTE_LOCK_ROOT`, host-wide) and `REMOTE_GPU_WAIT`
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
  job is still `running`/`waiting_gpu` against the same mirror
- sharing one remote mirror between two increments (always use
  `--increment SF-NN` / `REMOTE_INCREMENT=SF-NN` for increment work)
- running two heavy jobs on one GPU: do not bypass the GPU lock with
  `scripts/remote exec`, `CUDA_VISIBLE_DEVICES` overrides inside the command,
  or detached (`setsid`/`nohup &`) processes that outlive their job
- leaving per-increment mirrors on the host after the increment's PR is merged
  (`scripts/remote remove-mirror SF-NN`)
