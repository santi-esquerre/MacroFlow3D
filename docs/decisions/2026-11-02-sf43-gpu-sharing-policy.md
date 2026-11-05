# SF-43: two GPUs only when both are free of third-party processes; otherwise one (the free one)

- Status: accepted (owner instruction in the orchestrator session of 2026-11-02, about 19:15Z; ratified by the human merge of the SF-43 PR
  that adds this record)
- Date: 2026-11-02
- Deciders: owner (session); recorded and implemented by the orchestrator
- Amends: H14 of `2026-11-02-sf43-pre-activation-decisions.md` (the two-GPU first stage) in its GPU selection only; scope, cap and
  expiry of H14 unchanged. `2026-10-27-remote-single-gpu.md` is not edited.

## Context

The first working smoke job of SF-43 (`sf43-smoke1b`, 2026-11-02T19:12:55Z) requested two GPUs. The helper's snapshot showed GPU 0 with
one compute process of another user (`compute_procs=1`, mean utilization 74.6 %, 645 MiB) and GPU 1 idle; both met the memory thresholds
(30000 / 17000 MiB), so the helper took both and lane B ran on the colleague's GPU. The helper selected by memory and utilization only.

## Decision

Owner instruction (verbatim, Spanish): "Antes de lanzar, fíjate si hay algo corriendo, tipo un proceso de terceros. Y si hay, usamos solo
una GPU, la que esté libre. Volvemos a la política anterior. Pero está, si están las dos libres, usamos las dos. En este caso hay un
proceso corriendo de otro compañero, así que habría que liberar esa GPU y usar solo una."

1. At launch a GPU is *free* when it has no compute process in the helper's snapshot. A two-GPU request (`REMOTE_GPU_COUNT=2`) takes two
   GPUs only if both are free and meet their memory thresholds; otherwise (one busy, or only one meeting its threshold) the job starts
   at once with one GPU, the free one if any (no waiting for the colleague). With no free GPU the previous single-GPU policy applies
   (least-loaded eligible GPU). Free GPUs also rank first in every single-GPU selection of the helper (every increment), before the
   previous order (utilization, free memory, index).
2. A third-party compute process that appears on lane B's GPU during stage 1 releases lane B at its next claim (its running unit
   finishes first); lane A is unaffected.
3. Applied at once to the smoke test: `sf43-smoke1b` was cancelled (19:20:32Z, which also served as the kill test) and resumed on one
   GPU (`sf43-smoke2`).

## Consequences

- Stage 1 may be shorter or absent; the one-GPU schedule of variant A (60-79 h of wall, launch by Thursday 2026-11-05 05:00 local for the
  cutoff) is the planning case whenever a colleague is on a GPU at launch.
- Implemented in `scripts/remote` (free-first ordering; degraded start) and the SF-43 engine (lane-B release), with tests; the runbook
  `docs/runbooks/remote-v100.md` documents it.

## Classification (docs/AGENTS.md)

| item | class |
|---|---|
| 1-3 | accepted scope (owner, 2026-11-02) |
| helper and engine behaviour | confirmed in code once the implementing commit is merged |
