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
5. The pre-SF-39 items are delivered in ONE pull request (owner request, 2026-10-26), one commit per AUD item so each
   item stays reviewable on its own; that PR ticks their lines and appends the log rows. Later items tick their line
   and append their row in their own PR. The log is append-only.

## Checklist

Before SF-39:

- [x] AUD-00 — this plan and decision record
- [x] AUD-01 — CI: GPU-free harness-hooks job on `master`
- [x] AUD-02 — remove the 11 duplicate `streamfunction_operator_tests` subset ctest entries
- [x] AUD-03 — regression harness `scripts/regress` + `apps/config_par2_small.yaml`
- [x] AUD-04 — no silent failures: writer/reader error checks, cuBLAS size check
- [x] AUD-05 — blocking context stream (legacy-stream ordering race)
- [x] AUD-06 — threaded host `exp` in the medium (b) stage S1 (bitwise)
- [x] AUD-10 — junk files, shim scripts, `skills/`
- [x] AUD-11 — `legacy/` archived (tag `archive/pre-cleanup-2026-10`, branch `archive/legacy`)
- [x] AUD-12 — PETSc/SLEPc + PSPTA invariants/eigensolver retired (no output change)
- [x] AUD-13 — PSPTA engine and config surface retired (expected diff A6)
- [x] AUD-14 — dead code and vendored yaml-cpp tests removed
- [x] AUD-20 — build-time revision (`git describe --dirty`), V100 source revision

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
| 2026-10-26T17:45Z | AUD-01..AUD-20 | branch `integration/aud-pre-sf39` (`a0a2dd8`) | Every pre-SF-39 item implemented in an isolated worktree, audited by the orchestrator (correctives: AUD-03 README `TMPDIR`; AUD-04 Par2 pipeline writer returns; AUD-10 theory README links; AUD-11 README wording; AUD-12 `.gitignore` CRLF; AUD-13 no edits to files AUD-10 deletes; AUD-20 `--exclude=*` and CRLF) and validated per item on the V100; one integrator composed the 13 commits (4 textual conflicts resolved keeping both intents; CI selftest step added). | Per item: `scripts/regress` 28 cases `REGRESS IDENTICAL` vs `3452a7d` (AUD-13: 27 cases with the A6 filter; without it exactly the 3 pipeline `effective_config.yaml` differ). AUD-02 ctest 41/41 in 1649 s (was 3182 s / 52 tests). AUD-05 ctest 52/52; production E2 byte-identical. AUD-06 production E2 byte-identical, S1 15.4/16.7 s -> 6.1/6.4 s. AUD-20 V100 builds record the commit (was `unknown`). Integrated `a0a2dd8`: composition check (single-owner files identical to their item), build 292/292, ctest 40/40 in 1646 s, regress 27 cases IDENTICAL (A6 filter), production E2 3328x384x384 byte-identical vs base (2 repetitions), S1 16.9/12.8 s -> 6.7/5.9 s, `commit` = `a0a2dd87881e`. | Single delivery PR. |
| 2026-10-26T18:00Z | AUD-00..AUD-20 | branch `refactor/audit-cleanup-pre-sf39` | Owner asked for one PR with everything: the per-item PRs #67-#79 are closed in favour of this branch (`integration/aud-pre-sf39` + this bookkeeping commit; docs only on top of the validated tree). Pre-SF-39 checklist ticked. | Evidence of the previous row applies unchanged (no compiled or compared file changes after `a0a2dd8`). | Owner review and merge; then SF-39 activates. Post-SF-39 items AUD-30..AUD-36 remain open. |
