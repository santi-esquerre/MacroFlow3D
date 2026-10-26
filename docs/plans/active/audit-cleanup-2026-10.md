# Codebase audit and cleanup 2026-10 (maintenance plan)

- Status: active
- Authority: [`docs/decisions/2026-10-26-audit-cleanup.md`](../../decisions/2026-10-26-audit-cleanup.md) (A1-A10)
- Base: `3452a7d` (SF-41 done)
- Not a Lester increment: it is not on the Lester dashboard and does not use its nonterminal slots;
  `scripts/hooks/check-lester-increments.sh` must keep passing.

## Objective

Leave the project as a lean pipeline suite (field generation, flow solve, transport) before the SF-39 campaign:
delete retired alternatives, fix the integrity defects found by the audit, cut validation cost, and record the
bitwise performance work that follows SF-39. Optimization order: correctness, reproducibility, maintainability,
memory, time.

## Rules

1. One purpose per PR (`AUD-NN`); branch prefixes per `docs/runbooks/pr-workflow.md` (`chore/`, `fix/`, `refactor/`).
2. Implementation in isolated worktrees (Claude Code `increment-worker`), audit by the orchestrator, one integrator;
   no agent pushes except the orchestrator, no agent merges.
3. Output bytes: unchanged in every PR (decision A7), except AUD-13 (A6) and AUD-20 (excluded provenance fields).
   Evidence: `scripts/regress` base vs head on the V100 (`scripts/remote --increment AUD-NN`, detached jobs) plus the
   full V100 ctest when compiled code changes; performance PRs also compare one production-grid realization and the
   per-process device peak.
4. PRs touching `src/physics`, `src/numerics`, `src/multigrid`, `src/runtime/ensemble`, `src/runtime/analysis` or
   physics configs need human review (`docs/runbooks/autonomy-policy.md`).
5. PRs do not edit this plan (parallel PRs would conflict on the log). The orchestrator ticks a line and appends
   its log row (append-only) in a bookkeeping commit after the PR is merged, batched when several merge together.

## Checklist

Before SF-39:

- [ ] AUD-00 — this plan and decision record
- [ ] AUD-01 — CI: GPU-free harness-hooks job on `master`
- [ ] AUD-02 — remove the 11 duplicate `streamfunction_operator_tests` subset ctest entries
- [ ] AUD-03 — regression harness `scripts/regress` + `apps/config_par2_small.yaml`
- [ ] AUD-04 — no silent failures: writer/reader error checks, cuBLAS size check
- [ ] AUD-05 — blocking context stream (legacy-stream ordering race)
- [ ] AUD-06 — threaded host `exp` in the medium (b) stage S1 (bitwise)
- [ ] AUD-10 — junk files, shim scripts, `skills/`
- [ ] AUD-11 — `legacy/` archived (tag `archive/pre-cleanup-2026-10`, branch `archive/legacy`)
- [ ] AUD-12 — PETSc/SLEPc + PSPTA invariants/eigensolver retired (no output change)
- [ ] AUD-13 — PSPTA engine and config surface retired (expected diff A6)
- [ ] AUD-14 — dead code and vendored yaml-cpp tests removed
- [ ] AUD-20 — build-time revision (`git describe --dirty`), V100 source revision

After SF-39:

- [ ] AUD-30 — Nsight profile of one production realization (experiment note; decides AUD-33/34)
- [ ] AUD-31 — medium (b) solve placement: PCG `r` = MG level-0 `b`, `z = Ap` = level-0 `x` (60.57 -> 52.57 B/cell; amends SF-41 E12)
- [ ] AUD-32 — plane and time passes in one launch (opt-in, bitwise)
- [ ] AUD-33 — host-side velocity/face phase, if AUD-30 supports it
- [ ] AUD-34 — GSRB colour-branch restructure, if AUD-30 supports it
- [ ] AUD-35 — move run orchestration (CLI, restart, `DONE`) out of `src/physics/particles/lagrangian_labels/EnsembleRunner.cu`, shared with `apps/lagrangian_medium_ab`
- [ ] AUD-36 — CMake target helper (accepted only if `ninja -t commands` is identical base vs head)

## Register: proposals (not implemented; non-bitwise or not binding)

- Lean Par2 baseline (~125 -> ~52.6 B/cell): Par2 memory is not binding at the SF-38/SF-39 grids.
- Fused residual + restriction in the MG (-8 B/cell): 27 boundary variants, FMA contraction may change bits.
- Shared cuFFT work area / in-place CompactMAC Pollock view: the spline phases are below the solve peak.
- Storing `1/K` per MG level: bitwise but +8 B/cell per level.
- `sincos` in the Par2 field generator: bit identity unverified under FMA contraction.
- One-time clang-format of the tree (pinned v18 + `.git-blame-ignore-revs`).

## Register: rejected (with reason)

- Merging the DP5(4) integrator copies and the spline direction functors: the GPU copies are verified bitwise against
  the host integrator; a shared template would make that control blind and mixes `--fmad=false` translation units.
- Wrapping the `pcg` variants into one: turns bitwise tests into self-comparisons.
- Consolidating test helpers: increments use "test file unchanged" as regression evidence.
- Replacing the hand-written JSON writers with nlohmann: changes instrument output bytes.
- Removing inert config keys (`output.save_*`, `flow.verify_velocity`, `transport.velocity_layout`): the strict parser
  would reject the pre-registered SF-38 Par2 configs used by SF-39 Gate 5.
- Deleting the SF-25 terminal-solver recorders: cited by experiment notes.
- Splitting `apps/inlet_slab/inlet_slab_driver.cuh`: single translation unit, no build gain.
- Tracker stack-limit memory carry-over (~136 MiB): refuted by SF-41 multi-realization logs.
- Device-side coefficient interleave: already done for medium (b); medium (a) only.

## Log

Append rows; do not rewrite prior rows.

| UTC | AUD id | Commit / PR | Action | Evidence | Next |
|---|---|---|---|---|---|
| 2026-10-26T16:30Z | AUD-00 | branch `chore/aud-00-audit-governance` | Plan and decision record written after the read-only audit at `3452a7d` (owner decisions of 2026-10-26). Archive refs created locally: tag `archive/pre-cleanup-2026-10` -> `3452a7d`, branch `archive/legacy` (subtree split, tree equal to `3452a7d:legacy`). | `check-lester-increments.sh`, `check-required-docs.sh` | Publish the archive refs with the owner's confirmation; open the AUD PRs. |
