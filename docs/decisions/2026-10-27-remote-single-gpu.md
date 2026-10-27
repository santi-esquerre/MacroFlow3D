# Remote V100: one GPU for the whole project, the least-loaded one

- Status: accepted (owner instruction of 2026-10-27; ratified by the human merge of the PR that adds this record)
- Date: 2026-10-27
- Deciders: owner (session 2026-10-27); recorded and implemented by the orchestrator
- Supersedes (without editing those records):
  - 2026-10-24 E6, its two-GPU part ("independent jobs on GPU 0 and GPU 1"). The checkpoint after the pilot is kept;
  - 2026-10-27 F13 (two SF-39 jobs concurrently on distinct GPUs) and the runbook exception it added;
  - runbook rule "one job per GPU, two jobs maximum across the host" and `--gpu auto` = "first GPU whose lock is free".

## Context

The V100 host (2x Tesla V100-PCIE-32GB) is shared with colleagues outside the project. Their jobs do not take our flock files.
`scripts/remote run` held one lock per GPU, so two project jobs could run at once, one per GPU. `--gpu auto` took the first GPU
with a free lock and tried GPU 0 first, ignoring real load. At 2026-10-27 ~15:10Z a colleague's process held GPU 0 (84 % utilization,
1281 MiB), GPU 1 was idle, and `auto` would still have chosen GPU 0. Production SF-41/SF-39 jobs peak at 28855 MiB (3328x384^2) of
32510 MiB: on a GPU a colleague has partly filled they fail for lack of memory, or push the colleague's job out of memory.

## Decision

1. **One GPU at a time for the whole project.** `scripts/remote run` takes one host-wide lock (`~/.macroflow3d-remote/gpu.lock`)
   for the job's lifetime: at most one `run` job of the project on the host, across every increment, mirror and session. When two
   increments are active, their jobs queue.
2. **The least-loaded GPU, chosen at start.** Holding the lock, the job samples `nvidia-smi` 5 times about 1 s apart. Eligible GPUs
   have free memory (minimum over the samples) of at least `REMOTE_GPU_MIN_FREE_MIB` (default 0). Among them it takes the lowest
   mean utilization; ties go to more free memory, then the lower index. If none is eligible, the job waits under the same
   `REMOTE_GPU_WAIT` deadline, then exits 75. The snapshot and the choice go to the job log. Memory is a hard limit, utilization a
   soft one. The choice is made once; a job does not move when a colleague later starts on its GPU.
3. **No GPU work outside `run`.** `scripts/remote exec` takes no lock and selects no GPU. It is only for steps that launch no CUDA
   work, and every `ctest` case touching the GPU uses `run`. Pinning `--gpu <n>` is exceptional, for reproducing a GPU-specific
   result, and is recorded. Commands do not set `CUDA_VISIBLE_DEVICES` and run no multi-GPU programs.
4. **SF-39.** Its jobs run one at a time, `pilot` and `extend` lose their GPU argument, and parts that launch CUDA work set
   `REMOTE_GPU_MIN_FREE_MIB=30000` (measured peak plus a margin) and `REMOTE_GPU_WAIT=7200`. The checkpoint (F9) adds the projected
   wall time on one shared GPU.

## Consequences

- Wall time of every remote campaign roughly doubles relative to the two-GPU plans. SF-39: ~19 GPU-h typical, ~91 worst case (decision
  2026-10-24), now about that many wall hours, more when a colleague shares the chosen GPU. GPU-hours and the scientific design are
  unchanged.
- Timing measurements taken while a colleague shares the GPU are not comparable with dedicated-GPU timings. The job log's snapshot
  records the condition; performance claims cite it.
- A large job can wait indefinitely while colleagues hold memory on both GPUs. `REMOTE_GPU_WAIT` bounds the wait, and exit 75
  means nothing ran.
- Changed: `scripts/remote` (arbitration only), `docs/runbooks/remote-v100.md` (Section 0 and anti-patterns), `AGENTS.md` (remote
  section) and the SF-39 specification (compute, files, implementation, validation commands, bitácora). `.claude/agents/*.md` files are
  untouched.
