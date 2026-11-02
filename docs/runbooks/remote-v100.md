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
slot enforces this for `run` jobs. The single-GPU rule has one time-limited
exception, the SF-43 stage-1 two-GPU request (see "Two-GPU request (SF-43
stage-1 exception only)" below); the old two-GPU rule of decisions 2026-10-24
E6 and 2026-10-27 F13 remains withdrawn.

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
- `REMOTE_GPU_COUNT` (default `1`; only `1` or `2`) is the two-GPU request of
  the SF-43 stage-1 exception below; any other value, or `--gpu <n>` together
  with `REMOTE_GPU_COUNT=2`, is refused before any ssh.
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

#### Two-GPU request (SF-43 stage-1 exception only)

Scope and expiry: owner decision of 2026-11-02, recorded in
`docs/decisions/2026-11-02-sf43-pre-activation-decisions.md` (item "two-GPU
exception"), which supersedes items 1 and 3 of
`docs/decisions/2026-10-27-remote-single-gpu.md` for this scope only. It
applies to SF-43 stage 1 (at most about 30 h of wall time); afterwards the job
returns to one GPU. Every other increment, and SF-43's stage 2, keep the
single-GPU rule above. It is still ONE project job holding the ONE host-wide
slot; only the number of GPUs the job holds changes.

```bash
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 \
  scripts/remote --increment SF-43 run <job> -- "<cmd>"
# optional lower threshold for the second GPU (lane B runs only small grids):
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 \
  REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 scripts/remote --increment SF-43 run <job> -- "<cmd>"
```

- Selection: with the slot held, the first GPU is the best (existing ordering:
  lowest mean utilization, then more free memory, then lower index) among those
  with free memory >= `REMOTE_GPU_MIN_FREE_MIB`; the second is the best of the
  remaining GPUs with free memory >= `REMOTE_GPU_MIN_FREE_MIB_2` (default: equal
  to `REMOTE_GPU_MIN_FREE_MIB`; it applies only with `REMOTE_GPU_COUNT=2`).
  The helper chooses both; commands never write `CUDA_VISIBLE_DEVICES` and
  launch commands never use `--gpu`.
- Waiting: while fewer than two GPUs qualify, the launcher re-samples every
  30 s until the single `REMOTE_GPU_WAIT` deadline (always `7200` here). When two
  qualify it takes both: `CUDA_VISIBLE_DEVICES=<first>,<second>`, status
  `gpu: <first>,<second>`, holder `HOLDER_GPU=<first>,<second>`, and a log line
  naming both GPUs with the threshold each one met.
- Degradation and exit 75: at the deadline, if at least `REMOTE_GPU_COUNT_MIN`
  GPUs qualify (`REMOTE_GPU_COUNT_MIN`, default = `REMOTE_GPU_COUNT`; `1`
  allows degrading), the job starts with the first GPU only and logs
  `GPU_COUNT degraded: granted 1 of 2 (REMOTE_GPU_COUNT_MIN=1)`; otherwise it
  fails with **exit 75** exactly as for a single GPU (nothing ran). With
  `REMOTE_GPU_WAIT=0` the decision is made on the first sampling round. The job
  sees `REMOTE_GPU_COUNT_GRANTED` (`1` or `2`) in its environment in both
  cases (it is also `1` for ordinary single-GPU jobs).
- Lanes: both runners always use device 0 (`CudaContext ctx(0)`), so the job
  itself distributes the granted list: each lane exports `CUDA_VISIBLE_DEVICES`
  set to ONE entry of the list it received. This narrowing inside the job is
  the only permitted use of that variable; the job logs it.
- Releasing the second GPU mid-run needs no helper action: the job's second
  lane simply exits (its CUDA context frees the memory) and the job logs it.
  The flock remains the single project slot for the whole job; there is no
  release subcommand.
- GPU-h accounting: a job that holds two GPUs is charged 2 x wall time while
  both are held (1 x for a degraded start); the SF-43 ledger records the
  granted list and the time at which the second lane exited.
- With `REMOTE_GPU_COUNT=1` (the default) behavior and log lines are unchanged
  (apart from `REMOTE_GPU_COUNT_GRANTED=1` in the job environment); the blinded
  `exec-blinded.log` line of a `run` submission gains `gpu_count=N` only when
  `N` is not `1`.

##### Sharing with third-party processes (owner decision of 2026-11-02 (session), recorded in docs/decisions/2026-11-02-sf43-gpu-sharing-policy.md)

Supersedes the selection part of the two-GPU request above: before launching,
check whether anything runs on the GPUs; one GPU if a colleague is using one,
both only if both are free.

- A GPU is **free** iff `nvidia-smi --query-compute-apps` lists no compute
  process on it in any of the snapshot's samples (taken before and after the 5
  utilization samples; unreadable = none). At selection time the job has not
  started, so every process seen is third-party. Free GPUs rank before busy
  ones; the remaining order (lowest mean utilization, more free memory, lower
  index) and the memory thresholds are unchanged. This applies to
  `REMOTE_GPU_COUNT=1` too (the only change there is free-first).
- `REMOTE_GPU_COUNT=2` takes two GPUs only if BOTH chosen GPUs are free. If a
  chosen GPU has a third-party process and `REMOTE_GPU_COUNT_MIN=1`, the job
  starts AT ONCE (no waiting for the colleague) with the best single GPU (a free
  one if any; if none is free, the least loaded as before) and logs
  `two-GPU request degraded to 1: GPU <n> has <k> third-party compute
  process(es)` (one line per busy GPU of the pair). With `REMOTE_GPU_COUNT_MIN=2`
  the launcher keeps waiting as before.
- During the run the campaign engine applies the same rule to lane B (see
  `sf43camp/README.md`): at each claim boundary a third-party process on lane
  B's GPU releases that GPU (`LANE_B_RELEASED gpu=<id> reason=third_party
  pids=<n>`); lane A is not affected.
- Tests: `bash scripts/tests/remote_gpu_select_test.sh`,
  `bash scripts/tests/remote_gpu_count_test.sh` (canned CSV / fake host).

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

#### Blinded phase and exec logging

For a confirmatory study (decision `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md`
G8, protocol in `docs/runbooks/lester-increment-workflow.md` section 8A) the
orchestrator declares a blinded phase on the increment's mirror:

```bash
scripts/remote --increment SF-NN blinded start --label "SF-43 production"   # first production run
scripts/remote --increment SF-NN blinded status                              # active/inactive + log line count
scripts/remote --increment SF-NN blinded stop                                # at the analysis freeze
```

- The phase is the marker file `$REMOTE_STATE_ROOT/blinded.marker` on the host
  (start time, increment, local `user@host`, label). `start` refuses (exit 1) if
  one exists; `stop` refuses if none exists. `blinded` requires `--increment`.
- While the marker exists, every `exec` and every `run` submission first appends
  one tab-separated line to `$REMOTE_STATE_ROOT/logs/exec-blinded.log`:
  `UTC  increment=  who=user@host  kind=exec|run  attempt=N  command=<shell-quoted>`
  (`run` lines also carry `job=` and `gpu=`; `start`/`stop` are logged as
  `kind=blinded-start|blinded-stop`). The command field is `printf %q` of the
  exact string, so it is one line and decodes with `eval "printf %s $field"`.
  The log is append-only across phases and is kept after `stop`.
- Scope of the logging: the marker and the log live under the per-increment
  `$REMOTE_STATE_ROOT`, so only `exec` and `run` invoked with the SAME
  `--increment` (or `REMOTE_INCREMENT`) are logged. Not logged: `exec`/`run`
  with another or no `--increment`, `tail`/`status`/`wait` reads of job logs, and
  direct `ssh` to the host. The log is an integrity aid for the blinded protocol
  (declare what was read in the bitácora), not an access control.
- Protections of the log: the job name `exec-blinded` is reserved (the log is
  `logs/exec-blinded.log`, next to the job logs `logs/<job>.log`), so `run`,
  `status`, `tail`, `wait` and `cancel` refuse it and no job launcher can truncate
  the log; and `remove-mirror <id>` refuses (exit 1) while that increment's
  `blinded.marker` exists. After `blinded stop` the log is kept, but a later
  `remove-mirror` deletes the whole state root with it: copy the log into the
  evidence (at the freeze) first.
- Mechanism: `scripts/remote` always sends, in the same ssh invocation, a guard
  statement `if [ -e <marker> ]; then <append line>; fi` on its own line before
  `cd <mirror> && <command>`. No extra round trip. The command text itself is
  unchanged and parsed as before, so its environment, working directory,
  stdout/stderr and exit status are unchanged; the guard runs in a subshell and
  writes nothing to stdout/stderr unless it fails. The decision is taken on the
  host (the marker is the authority), hence the guard is sent even when no phase
  is active; with no marker it is a no-op test.
- `ssh_run` retries any non-zero exit (existing behaviour, unchanged): each retry
  re-sends the guard and logs a line with its `attempt=` number.
- A logging failure aborts the command: the guard exits 97 before the command
  runs, with `ERROR: blinded phase: log write failed; command NOT run` on stderr
  (an unlogged command would defeat the purpose). `scripts/remote` then exits 1
  after its retries, as for any failing command.
- Test: `bash scripts/tests/remote_blinded_test.sh` (fake host, no network).

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
- using more than one GPU (other than the SF-43 stage-1 two-GPU request), or
  a GPU not chosen by the slot: do not bypass it
  with CUDA work under `scripts/remote exec`, `CUDA_VISIBLE_DEVICES` overrides
  inside the command, multi-GPU programs, or detached (`setsid`/`nohup &`)
  processes that outlive their job
- leaving per-increment mirrors on the host after the increment's PR is merged
  (`scripts/remote remove-mirror SF-NN`)
