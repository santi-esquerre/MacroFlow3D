# Remote V100 runbook

This runbook defines the canonical remote workflow.

The server is used for:
- release builds,
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
- any test binary that internally sweeps many cases (multi-case
  streamfunction Picard/Anderson/continuation ladders, mesh refinement
  studies, etc.);
- production-like pipeline runs, ensemble runs, benchmarks, and profiling
  runs.

`scripts/remote exec` is reserved for short, bounded steps only: configure, a
single compile, a quick single-command check that launches no CUDA work —
work that reliably finishes in roughly under a minute. Anything that runs on
a GPU, even for seconds, goes through `scripts/remote run`, so that it takes
the project's single GPU slot (Section 0, "Concurrency"). If a command's duration is unknown or
could plausibly run longer, treat it as long-duration and use `scripts/remote
run` + `scripts/remote wait`, not `exec`.

Rationale: local WSL has no GPU-equivalent execution path for CUDA test
binaries at V100 scale; heavy test/production runs there are both slower and
not representative. Claude Code subagents (`increment-worker`,
`increment-integrator`) must dispatch their heavy validation commands to V100
through this mechanism rather than executing them synchronously inside their
own isolated worktree.

### Concurrency: one increment, one mirror; one GPU for the whole project

The V100 host has two GPUs and is shared with colleagues outside this project,
whose jobs do not use our locks. Owner rule (decision
`docs/decisions/2026-10-27-remote-single-gpu.md`): **this project uses one GPU
at a time, always the least-loaded one.** `scripts/remote` isolates and
arbitrates in two ways:

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
2. **One host-wide GPU slot, least-loaded GPU.** Every `scripts/remote run`
   job takes an exclusive `flock` on `~/.macroflow3d-remote/gpu.lock` for its
   whole lifetime. The lock is shared by all mirrors (including the legacy
   one), so at most one `run` job of this project executes on the host at a
   time, across every increment and session. Once it holds the slot, the job
   samples `nvidia-smi` (5 samples about 1 s apart) and takes the GPU with the
   lowest mean utilization among those whose free memory (minimum over the
   samples) is at least `REMOTE_GPU_MIN_FREE_MIB`; ties go to more free
   memory, then to the lower index. It exports `CUDA_VISIBLE_DEVICES=<n>` to
   the command and writes the snapshot and the choice to its log.

The rule is therefore:

- **one increment, one mirror** — always pass `--increment SF-NN` (or set
  `REMOTE_INCREMENT=SF-NN`) for increment work; never share a mirror between
  increments;
- **one `run` job of this project on the host at a time** — enforced by the
  slot; when two increments are active their jobs queue on it;
- **the least-loaded GPU at start** — chosen by `scripts/remote`; do not pin
  `--gpu <n>` except to reproduce a GPU-specific result, and record why.

The choice is made once, when the job starts: a colleague may later start
work on the same GPU, and the job does not move. Jobs whose peak device memory
is known set `REMOTE_GPU_MIN_FREE_MIB` to that peak plus a margin, so that
they wait rather than start on a GPU a colleague has partly filled (memory is
a hard limit, utilization only slows both jobs). Long jobs set
`REMOTE_GPU_WAIT` so that a slot held by another session, or no eligible GPU,
makes them wait instead of failing. On exit 75 nothing ran: resubmit the same
command (restart-safe where the runner keeps `DONE` files).

Within one increment the mirror is still single-flight: do not
`scripts/remote --increment SF-NN sync` while a job of that same increment is
`running` or `waiting_gpu`, because the sync overwrites the tree the job
executes against. DAG nodes that need remote execution are serialized; the
slot enforces this for `run` jobs. There is no multi-GPU exception: the
two-GPU rule of decisions 2026-10-24 E6 and 2026-10-27 F13 is withdrawn.

`scripts/remote exec` takes **no** GPU lock and selects no GPU; it is only for
short, bounded configure/compile/check steps that launch no CUDA work.

#### GPU selection and waiting

```bash
scripts/remote --increment SF-NN run <job> -- "<cmd>"            # --gpu auto (default): least-loaded GPU
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=31000 \
  scripts/remote --increment SF-NN run <job> -- "<cmd>"          # wait up to 2 h for the slot and >= 31000 MiB free
scripts/remote --increment SF-NN run <job> --gpu 1 -- "<cmd>"    # pin (exceptional); still takes the slot
```

- `REMOTE_GPU_WAIT=<seconds>` (default `0`) is how long a job waits, under
  one deadline, for the slot and then for an eligible GPU. While waiting, its
  state is `waiting_gpu` (`status` shows it; `wait` keeps polling through it).
- `REMOTE_GPU_MIN_FREE_MIB=<MiB>` (default `0`) is the minimum free memory for
  a GPU to be eligible. A pinned GPU must meet it too.
- With `REMOTE_GPU_WAIT=0`, or when the wait expires, the job does not run:
  its state becomes `failed` with **exit code 75**, and its log lists the
  current slot holder (job, session, increment, GPU, start time, whether the
  session is alive) or the last GPU snapshot. `scripts/remote wait <job>`
  returns 75.
- If `nvidia-smi` fails or returns nothing usable, the job fails with
  **exit code 70**; there is no blind fallback to a fixed GPU.
- `status` reports the assigned GPU in the `gpu:` line and the mirror in the
  `working_dir:` line; the log records the per-GPU snapshot, the choice and
  its reason, and `GPU=<n> (CUDA_VISIBLE_DEVICES=<n>, ...)`.
- `cancel` terminates the job's whole process group (launcher, command and
  children), so the slot is released immediately. Processes that detach
  themselves with `setsid`/`nohup &` escape the group and keep the slot until
  they exit.
- The holder file `~/.macroflow3d-remote/gpu.holder` is informational only;
  the kernel `flock` is the authority. A stale holder file never blocks a new
  job.

#### Mirror lifecycle

```bash
scripts/remote list-mirrors           # mirrors, state roots, current GPU holders
scripts/remote remove-mirror SF-NN    # delete ~/MacroFlow3D-SF-NN and its state root
```

- `remove-mirror` refuses (exit 1) while any tmux session with prefix
  `macroflow3d-SF-NN-` is alive. It never removes the shared `~/MacroFlow3D`.
- Remove an increment's mirror once its PR is merged (copy out any logs or
  outputs that the increment record cites first).
- A new per-increment mirror contains only what `sync` copies: it has no
  build directories. `v100-release` builds work after a normal
  configure/build.

#### Regression byte-compare

Cleanup / refactor / performance PRs prove byte-identical outputs against a
build of their merge-base with `scripts/regress` (`run.sh` for the base and the
head build as detached `scripts/remote run` jobs, then `compare.sh`; references
cached under `~/regress/refs/<base-short-sha>/out`). Protocol:
`scripts/regress/README.md`.

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
~/MacroFlow3D/build/v100-prof
```

---

## 3. Canonical interface

Everything goes through one repo-local entry point:

```bash
scripts/remote [--increment <id>] sync
scripts/remote [--increment <id>] exec -- "<shell-command>"
scripts/remote [--increment <id>] run <job> [--gpu auto|<n>] -- "<shell-command>"
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
`succeeded`, `failed` (exit 75 = slot busy or no eligible GPU, resubmit; 70 = `nvidia-smi`
unusable), `cancelled` (130) and
`unknown`. See Section 0 for GPU locking and mirror lifecycle.

Remote defaults live in:

```bash
scripts/remote.env
```

That file defines:
- remote host alias
- remote repo path (derived per increment when an id is given)
- remote state root (derived per increment when an id is given)
- GPU slot lock root (`REMOTE_LOCK_ROOT`, host-wide), `REMOTE_GPU_WAIT` and
  `REMOTE_GPU_MIN_FREE_MIB`
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

### 4.1 Release build
```bash
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
```

## 6. Remote tests

Full or multi-case test runs are long-duration computation (see Section 0):
launch them detached and wait, do not run them as a blocking `exec` call.

### Release test pass (detached)
```bash
scripts/remote run ctest-release -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-release
```

Every `ctest` case that launches CUDA work uses `run` + `wait`, however fast,
so that it takes the project's single GPU slot.

## 7. Remote runs

### Par2 smoke
```bash
scripts/remote run par2-small -- "./build/v100-release/macroflow3d_pipeline apps/config_par2_small.yaml"
scripts/remote wait par2-small
```

### Baseline Par2 config
```bash
scripts/remote run par2-prod -- "./build/v100-release/macroflow3d_pipeline apps/config_pipeline_par2.yaml"
scripts/remote tail par2-prod
scripts/remote wait par2-prod
```

### Cancel a long job

```bash
scripts/remote cancel par2-prod
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
- running `scripts/remote sync` while a job is still `running`/`waiting_gpu`
  against the same mirror
- sharing one remote mirror between two increments (always use
  `--increment SF-NN` / `REMOTE_INCREMENT=SF-NN` for increment work)
- using more than one GPU, or a GPU not chosen by the slot: do not bypass it
  with CUDA work under `scripts/remote exec`, `CUDA_VISIBLE_DEVICES` overrides
  inside the command, multi-GPU programs, or detached (`setsid`/`nohup &`)
  processes that outlive their job
- leaving per-increment mirrors on the host after the increment's PR is merged
  (`scripts/remote remove-mirror SF-NN`)
